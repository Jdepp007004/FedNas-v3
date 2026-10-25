"""Compare federated methods as the shared elastic MLP grows in width.

This is a resource/utility scaling study. Each scale reuses the same dataset,
client partitions, and capacity assignments across paired strategies. The
largest server-side supernet grows with the scale while each heterogeneous
client materializes only its assigned fixed subnet.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional hardware inventory
    psutil = None

from .config import DATASETS, get_dataset_spec
from .data import load_dataset
from .federated import (
    ClientArchitecture,
    _evaluate,
    _capacity_candidates,
    _current_cpu_affinity,
    _initialize_client_worker,
    _prime_client_worker,
    _set_worker_affinity,
    run_federated_experiment,
)
from .growth import widen_replicated_state
from .models import build_fixed_subnet, build_model, load_numpy_state, load_subnet_state
from .models import subnet_flops, subnet_parameters
from .provenance import build_array_run_manifest
from .stats import confidence_interval, paired_interval


DEFAULT_STRATEGIES = (
    "fedavg",
    "fedprox",
    "heterofl",
    "fjord",
    "fedrolex",
    "maxnet",
    "cc_efl",
)

DEVICE_NAMES = (
    "edge-1t",
    "low-2t",
    "mid-4t",
    "high-8t",
)
DEFAULT_DEVICE_CAPACITY_TIERS = (0, 0, 1, 2)


def _cpu_partition(thread_budgets: list[int], reserve_threads: int):
    available = _current_cpu_affinity()
    requested = sum(thread_budgets) + reserve_threads
    if requested > len(available):
        raise ValueError(
            f"CPU split needs {requested} logical processors but only "
            f"{len(available)} are available to this process"
        )
    profiles = []
    cursor = 0
    for client_id, thread_count in enumerate(thread_budgets):
        cpu_ids = available[cursor : cursor + thread_count]
        cursor += thread_count
        profiles.append(
            {
                "client_id": client_id,
                "device_id": f"device-{client_id + 1}",
                "profile_name": DEVICE_NAMES[client_id],
                "logical_cpu_ids": cpu_ids,
                "torch_threads": thread_count,
                "capacity_tier": DEFAULT_DEVICE_CAPACITY_TIERS[client_id],
            }
        )
    return profiles, available[cursor:], available


def _device_specs(result: dict, spec, input_dim: int) -> list[dict]:
    candidates = _capacity_candidates(spec)
    device_times: dict[int, list[float]] = defaultdict(list)
    for round_row in result["history"]:
        for timing in round_row.get("client_training_seconds", []):
            device_times[int(timing["client_id"])].append(float(timing["seconds"]))
    device_specs = []
    for client_id, profile in enumerate(result.get("client_cpu_profiles") or []):
        tier = int(result["client_capacity_tiers"][client_id])
        capacity = candidates[tier]
        if result["strategy"] == "fedavg":
            train_depth, train_width = spec.max_depth, spec.hidden_dim
        else:
            train_depth, train_width = capacity.depth, capacity.width
        parameters = subnet_parameters(
            input_dim, spec.num_classes, train_depth, train_width
        )
        times = device_times.get(client_id, [])
        device_specs.append(
            {
                "client_id": client_id,
                "device_id": profile["device_id"],
                "profile_name": profile["profile_name"],
                "logical_cpu_ids": profile["logical_cpu_ids"],
                "logical_processors": len(profile["logical_cpu_ids"]),
                "torch_threads": int(profile["torch_threads"]),
                "capacity_tier": tier,
                "capacity_depth": capacity.depth,
                "capacity_width": capacity.width,
                "training_depth": train_depth,
                "training_width": train_width,
                "training_parameters": parameters,
                "fp32_parameter_mib": parameters * 4 / (1024**2),
                "mean_local_training_seconds_per_round": (
                    float(np.mean(times)) if times else None
                ),
                "p95_local_training_seconds_per_round": (
                    float(np.percentile(times, 95)) if times else None
                ),
            }
        )
    return device_specs


def _scaled_spec(spec, factor: int, capacity_scaling: str):
    hidden_dim = spec.hidden_dim * factor
    if capacity_scaling == "proportional":
        widths = tuple(width * factor for width in spec.widths)
    else:
        # Keep the three heterogeneous client tiers fixed while adding the
        # expanded server width as a validation candidate. The last width
        # before the full supernet remains the largest client capability.
        client_widths = (
            spec.widths[0],
            spec.widths[min(1, len(spec.widths) - 1)],
            spec.widths[-2] if len(spec.widths) > 1 else spec.widths[-1],
        )
        widths = tuple(sorted(set((*client_widths, hidden_dim))))
    return replace(
        spec,
        hidden_dim=hidden_dim,
        widths=widths,
    )


def _mean_history(result: dict, field: str) -> float:
    values = [float(row[field]) for row in result["history"]]
    return float(np.mean(values)) if values else 0.0


def _seed_record(result: dict, scale: int, spec, input_dim: int) -> dict:
    final = result["final"]
    full_parameters = int(final["full_model_parameters"])
    full_flops = int(final["full_model_flops"])
    active_parameters = _mean_history(result, "mean_client_parameters")
    active_flops = _mean_history(result, "mean_active_flops")
    uploaded_per_round = _mean_history(result, "total_upload_bytes")
    downloaded_per_round = _mean_history(result, "total_download_bytes")
    tier_rows = []
    for capacity in _capacity_candidates(spec):
        parameters = subnet_parameters(
            input_dim, spec.num_classes, capacity.depth, capacity.width
        )
        tier_rows.append(
            {
                "depth": capacity.depth,
                "width": capacity.width,
                "parameters": parameters,
                "fp32_parameter_mib": parameters * 4 / (1024**2),
                "forward_flops_per_example": subnet_flops(
                    input_dim, spec.num_classes, capacity.depth, capacity.width
                ),
            }
        )
    device_specs = _device_specs(result, spec, input_dim)
    parallel_round_times = []
    client_cpu_seconds = []
    for round_row in result["history"]:
        times = [
            float(item["seconds"])
            for item in round_row.get("client_training_seconds", [])
        ]
        if times:
            parallel_round_times.append(max(times))
            client_cpu_seconds.append(sum(times))
    return {
        "dataset": result["dataset"],
        "scale_factor": scale,
        "seed": result["seed"],
        "strategy": result["strategy"],
        "variant": result["variant"],
        "rounds": result["rounds"],
        "clients": result["clients"],
        "client_capacity_tiers": result["client_capacity_tiers"],
        "score_macro_f1": float(final["selected_test_macro_f1"]),
        "score_accuracy": float(final["selected_test_accuracy"]),
        "full_supernet_macro_f1": float(final["full_model_test_macro_f1"]),
        "full_supernet_accuracy": float(final["full_model_test_accuracy"]),
        "full_supernet_parameters": full_parameters,
        "full_supernet_fp32_parameter_mib": full_parameters * 4 / (1024**2),
        "full_supernet_forward_flops_per_example": full_flops,
        "mean_active_client_parameters": active_parameters,
        "mean_active_client_fp32_parameter_mib": active_parameters * 4 / (1024**2),
        "mean_active_client_forward_flops_per_example": active_flops,
        "mean_upload_bytes_per_round": uploaded_per_round,
        "mean_download_bytes_per_round": downloaded_per_round,
        "mean_total_communication_bytes_per_round": uploaded_per_round + downloaded_per_round,
        "selected_deployed_parameters": float(final["selected_deployed_parameters"]),
        "selected_deployed_fp32_parameter_mib": (
            float(final["selected_deployed_parameters"]) * 4 / (1024**2)
        ),
        "selected_deployed_forward_flops_per_example": float(final["selected_deployed_flops"]),
        "coordinate_coverage": float(result["parameter_coordinate_coverage"]),
        "simulation_wall_clock_seconds": float(result["wall_clock_seconds"]),
        "mean_slowest_client_training_seconds_per_round": (
            float(np.mean(parallel_round_times)) if parallel_round_times else None
        ),
        "mean_total_client_cpu_seconds_per_round": (
            float(np.mean(client_cpu_seconds)) if client_cpu_seconds else None
        ),
        "client_device_specs": device_specs,
        "client_capacity_menu": tier_rows,
    }


def _run_progressive_growth(
    bundle,
    base_spec,
    target_spec,
    args,
    seed: int,
    client_cpu_profiles,
    coordinator_cpu_ids,
    client_executor,
    fixed_tier_assignments,
):
    """Train a small parent, widen it without changing its function, and continue."""
    if args.variant != "elastic_plain":
        raise ValueError("progressive_growth requires --variant elastic_plain")
    warmup_rounds = int(args.growth_warmup_rounds)
    if not 1 <= warmup_rounds < args.rounds:
        raise ValueError("growth warmup rounds must be between 1 and total rounds - 1")
    group_width = _capacity_candidates(target_spec)[-1].width
    if target_spec.hidden_dim % group_width:
        raise ValueError(
            "progressive_growth requires target width divisible by the largest client tier"
        )
    growth_factor = target_spec.hidden_dim // group_width
    capacity_widths = [item.width for item in _capacity_candidates(target_spec)]
    warmup_widths = tuple(sorted(set((*capacity_widths, group_width))))
    # _capacity_candidates treats the penultimate width as the largest client
    # tier, so append the parent width once more as the full-model candidate.
    if len(warmup_widths) < 2 or warmup_widths[-1] != group_width:
        raise ValueError("invalid parent-width capacity menu")
    warmup_widths = (*warmup_widths, group_width)
    warmup_spec = replace(base_spec, hidden_dim=group_width, widths=warmup_widths)

    warmup = run_federated_experiment(
        bundle,
        warmup_spec,
        variant=args.variant,
        rounds=warmup_rounds,
        seed=seed,
        num_clients=4 if client_cpu_profiles else args.clients,
        strategy="heterofl",
        optimizer_name="adam",
        client_cpu_profiles=client_cpu_profiles,
        fixed_tier_assignments=fixed_tier_assignments,
        coordinator_cpu_ids=coordinator_cpu_ids,
        client_executor=client_executor,
        include_final_test=False,
        return_global_state=True,
    )
    grown_state = widen_replicated_state(
        warmup["global_model_state"], group_width, target_spec.hidden_dim
    )

    # Validate the growth transform on validation inputs only. This checks
    # function preservation; validation labels do not control the schedule.
    parent = build_model(
        args.variant,
        bundle.input_dim,
        target_spec.num_classes,
        target_spec.max_depth,
        group_width,
    )
    child = build_model(
        args.variant,
        bundle.input_dim,
        target_spec.num_classes,
        target_spec.max_depth,
        target_spec.hidden_dim,
    )
    load_numpy_state(parent, warmup["global_model_state"])
    load_numpy_state(child, grown_state)
    parent.eval()
    child.eval()
    validation_inputs = torch.from_numpy(np.asarray(bundle.X_val[:512], dtype=np.float32))
    with torch.no_grad():
        growth_error = float(
            (parent.forward_full(validation_inputs) - child.forward_full(validation_inputs))
            .abs()
            .max()
            .item()
        )

    grown = run_federated_experiment(
        bundle,
        target_spec,
        variant=args.variant,
        rounds=args.rounds - warmup_rounds,
        seed=seed,
        num_clients=4 if client_cpu_profiles else args.clients,
        strategy="fedrolex_block",
        optimizer_name="adam",
        client_cpu_profiles=client_cpu_profiles,
        fixed_tier_assignments=fixed_tier_assignments,
        coordinator_cpu_ids=coordinator_cpu_ids,
        client_executor=client_executor,
        initial_global_state=grown_state,
        classifier_scale=float(growth_factor),
        replica_weight_scale=float(growth_factor),
        block_window_width=group_width,
        include_final_test=False,
        return_global_state=True,
    )
    selection_started = time.perf_counter()
    final_state = grown["global_model_state"]
    selected_rows = []
    capacity_specs = _capacity_candidates(target_spec)
    for tier_index, capacity in enumerate(capacity_specs):
        candidates = []
        candidate_offsets = range(0, target_spec.hidden_dim, group_width)
        for offset in candidate_offsets:
            local = build_fixed_subnet(
                args.variant,
                bundle.input_dim,
                target_spec.num_classes,
                capacity.depth,
                capacity.width,
                offset=offset,
            )
            load_subnet_state(local, final_state)
            with torch.no_grad():
                local.classifier.weight.mul_(growth_factor)
                for layer in local.layers[1:]:
                    layer.weight.mul_(growth_factor)
            score = _evaluate(
                local,
                bundle.X_val,
                bundle.y_val,
                ClientArchitecture(capacity.depth, capacity.width, offset),
                target_spec.batch_size,
            )
            candidates.append(
                {
                    "val_accuracy": score["accuracy"],
                    "val_macro_f1": score["macro_f1"],
                    "depth": capacity.depth,
                    "width": capacity.width,
                    "offset": offset,
                    "parameters": subnet_parameters(
                        bundle.input_dim,
                        target_spec.num_classes,
                        capacity.depth,
                        capacity.width,
                    ),
                    "flops": subnet_flops(
                        bundle.input_dim,
                        target_spec.num_classes,
                        capacity.depth,
                        capacity.width,
                    ),
                }
            )
        best_val = max(row[target_spec.selection_metric] for row in candidates)
        eligible = [
            row for row in candidates
            if row[target_spec.selection_metric] >= best_val - 0.002
        ]
        selected = min(eligible, key=lambda row: (row["flops"], row["parameters"]))
        selected_model = build_fixed_subnet(
            args.variant,
            bundle.input_dim,
            target_spec.num_classes,
            capacity.depth,
            capacity.width,
            offset=int(selected["offset"]),
        )
        load_subnet_state(selected_model, final_state)
        with torch.no_grad():
            selected_model.classifier.weight.mul_(growth_factor)
            for layer in selected_model.layers[1:]:
                layer.weight.mul_(growth_factor)
        test = _evaluate(
            selected_model,
            bundle.X_test,
            bundle.y_test,
            ClientArchitecture(
                capacity.depth, capacity.width, int(selected["offset"])
            ),
            target_spec.batch_size,
        )
        selected_rows.append(
            {
                "capacity_tier": tier_index,
                "capacity_depth": capacity.depth,
                "capacity_width": capacity.width,
                "client_count": fixed_tier_assignments.count(tier_index),
                **selected,
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
            }
        )

    # The full-model test score is a fixed endpoint. It is evaluated only
    # after the window offsets have been selected using validation data.
    full_model = build_model(
        args.variant,
        bundle.input_dim,
        target_spec.num_classes,
        target_spec.max_depth,
        target_spec.hidden_dim,
    )
    load_numpy_state(full_model, final_state)
    full_test = _evaluate(
        full_model,
        bundle.X_test,
        bundle.y_test,
        ClientArchitecture(target_spec.max_depth, target_spec.hidden_dim),
        target_spec.batch_size,
    )
    selection_seconds = time.perf_counter() - selection_started
    total_clients = max(len(fixed_tier_assignments), 1)

    def weighted(field: str) -> float:
        return float(
            sum(row[field] * row["client_count"] for row in selected_rows)
            / total_clients
        )

    final = grown["final"]
    final.update(
        {
            "selected_models_by_capacity": selected_rows,
            "selected_test_accuracy": weighted("test_accuracy"),
            "selected_test_macro_f1": weighted("test_macro_f1"),
            "selected_val_accuracy": weighted("val_accuracy"),
            "selected_val_macro_f1": weighted("val_macro_f1"),
            "selected_deployed_parameters": weighted("parameters"),
            "selected_deployed_flops": weighted("flops"),
            "selected_depth": max(row["depth"] for row in selected_rows),
            "selected_width": max(row["width"] for row in selected_rows),
            "full_model_test_accuracy": full_test["accuracy"],
            "full_model_test_macro_f1": full_test["macro_f1"],
        }
    )
    warm_history = [dict(row, phase="small_parent") for row in warmup["history"]]
    grown_history = [
        dict(row, round=int(row["round"]) + warmup_rounds, phase="expanded_model")
        for row in grown["history"]
    ]
    grown["history"] = warm_history + grown_history
    grown["history"][-1] = final
    grown["strategy"] = "progressive_growth"
    grown["rounds"] = int(args.rounds)
    grown["wall_clock_seconds"] = float(
        warmup["wall_clock_seconds"] + grown["wall_clock_seconds"] + selection_seconds
    )
    grown["channel_window_policy"] = "replica-block-cyclic"
    grown["growth"] = {
        "inspiration": "function-preserving Net2Wider-style model growth",
        "parent_width": int(group_width),
        "expanded_width": int(target_spec.hidden_dim),
        "replication_factor": int(growth_factor),
        "warmup_rounds": warmup_rounds,
        "expanded_rounds": int(args.rounds - warmup_rounds),
        "parent_validation_macro_f1": float(warmup["final"]["selected_val_macro_f1"]),
        "max_abs_logit_difference_before_finetuning": growth_error,
        "post_growth_coordinate_coverage": float(
            grown["parameter_coordinate_coverage"]
        ),
        "validation_window_selection_seconds": float(selection_seconds),
        "parent_model_wall_clock_seconds": float(warmup["wall_clock_seconds"]),
        "expanded_model_wall_clock_seconds": float(
            grown["wall_clock_seconds"] - warmup["wall_clock_seconds"]
        ),
    }
    grown["final"]["growth"] = grown["growth"]
    grown.pop("global_model_state", None)
    return grown


def _summarize(records: list[dict], scales: list[int], strategies: list[str]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["scale_factor"], record["strategy"])].append(record)

    summaries = []
    for scale in scales:
        fedavg = {
            row["seed"]: row for row in grouped.get((scale, "fedavg"), [])
        }
        cc_efl = {
            row["seed"]: row for row in grouped.get((scale, "cc_efl"), [])
        }
        for strategy in strategies:
            rows = grouped[(scale, strategy)]
            score = confidence_interval(
                (row["score_macro_f1"] for row in rows), bounds=(0, 1)
            )
            accuracy = confidence_interval(
                (row["score_accuracy"] for row in rows), bounds=(0, 1)
            )
            summary = {
                "scale_factor": scale,
                "strategy": strategy,
                "seeds": len(rows),
                "score_metric": "capacity_selected_test_macro_f1",
                "macro_f1_mean": score["mean"],
                "macro_f1_ci95_low": score["ci95_low"],
                "macro_f1_ci95_high": score["ci95_high"],
                "accuracy_mean": accuracy["mean"],
                "accuracy_ci95_low": accuracy["ci95_low"],
                "accuracy_ci95_high": accuracy["ci95_high"],
            }
            for field in (
                "full_supernet_parameters",
                "full_supernet_fp32_parameter_mib",
                "full_supernet_forward_flops_per_example",
                "mean_active_client_parameters",
                "mean_active_client_fp32_parameter_mib",
                "mean_active_client_forward_flops_per_example",
                "mean_upload_bytes_per_round",
                "mean_download_bytes_per_round",
                "mean_total_communication_bytes_per_round",
                "selected_deployed_parameters",
                "selected_deployed_fp32_parameter_mib",
                "selected_deployed_forward_flops_per_example",
                "coordinate_coverage",
                "simulation_wall_clock_seconds",
                "mean_slowest_client_training_seconds_per_round",
                "mean_total_client_cpu_seconds_per_round",
            ):
                summary[field] = float(np.mean([row[field] for row in rows]))
            summary["active_parameter_reduction_vs_full_fedavg"] = 1.0 - (
                summary["mean_active_client_parameters"]
                / max(summary["full_supernet_parameters"], 1.0)
            )
            summary["active_flops_reduction_vs_full_fedavg"] = 1.0 - (
                summary["mean_active_client_forward_flops_per_example"]
                / max(summary["full_supernet_forward_flops_per_example"], 1.0)
            )
            shared = sorted(set(fedavg) & {row["seed"] for row in rows})
            if strategy != "fedavg" and shared:
                method_by_seed = {row["seed"]: row for row in rows}
                delta = paired_interval(
                    [fedavg[seed]["score_macro_f1"] for seed in shared],
                    [method_by_seed[seed]["score_macro_f1"] for seed in shared],
                )
                summary["paired_macro_f1_delta_vs_fedavg"] = delta
            shared_cc = sorted(set(cc_efl) & {row["seed"] for row in rows})
            if strategy != "cc_efl" and shared_cc:
                method_by_seed = {row["seed"]: row for row in rows}
                delta_vs_cc_efl = paired_interval(
                    [cc_efl[seed]["score_macro_f1"] for seed in shared_cc],
                    [method_by_seed[seed]["score_macro_f1"] for seed in shared_cc],
                )
                summary["paired_macro_f1_delta_vs_cc_efl"] = delta_vs_cc_efl
            summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="har")
    parser.add_argument("--data-dir", default="data/non_medical")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--scales", nargs="+", type=int, default=(1, 4, 8))
    parser.add_argument(
        "--capacity-scaling",
        choices=("fixed", "proportional"),
        default="fixed",
        help=(
            "fixed keeps every client's width menu unchanged as the server supernet grows; "
            "proportional scales client widths with the supernet"
        ),
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument(
        "--growth-warmup-rounds",
        type=int,
        default=4,
        help="small-parent rounds before function-preserving expansion",
    )
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument(
        "--client-cpu-threads",
        nargs="+",
        type=int,
        default=None,
        help=(
            "run one affinity-limited worker per client; the four-device profile "
            "uses 1 2 4 8 logical processors"
        ),
    )
    parser.add_argument("--reserve-cpu-threads", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--variant",
        choices=("elastic", "elastic_plain"),
        default="elastic",
        help="elastic uses residual MLPs; elastic_plain disables residual skips for layerwise window routing",
    )
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--output", default=".benchmarks/model_scaling/har_widthscale_1_4_8.json")
    args = parser.parse_args()

    if any(scale < 1 for scale in args.scales):
        parser.error("scale factors must be positive integers")
    if any(strategy not in {
        "fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "maxnet", "cc_efl",
        "cc_efl_no_prox", "cc_efl_uniform", "cc_efl_random_window",
        "cc_efl_layercover", "progressive_growth",
    } for strategy in args.strategies):
        parser.error("unsupported strategy in model-scaling matrix")
    if "cc_efl_layercover" in args.strategies and args.variant != "elastic_plain":
        parser.error("cc_efl_layercover requires --variant elastic_plain")
    if "progressive_growth" in args.strategies:
        if args.variant != "elastic_plain":
            parser.error("progressive_growth requires --variant elastic_plain")
        if args.capacity_scaling != "fixed":
            parser.error("progressive_growth requires --capacity-scaling fixed")
        if not args.client_cpu_threads:
            parser.error("progressive_growth requires four CPU-limited client profiles")
        if args.clients not in (None, 4):
            parser.error("progressive_growth requires four clients")
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if args.reserve_cpu_threads < 1:
        parser.error("--reserve-cpu-threads must be positive")
    if args.client_cpu_threads and (
        len(args.client_cpu_threads) != 4
        or any(thread_count < 1 for thread_count in args.client_cpu_threads)
    ):
        parser.error("--client-cpu-threads requires four positive per-device budgets")

    torch.set_num_threads(args.torch_threads)
    base_spec = get_dataset_spec(args.dataset)
    bundle = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        max_samples=args.max_samples,
    )
    clients = args.clients or base_spec.clients
    client_cpu_profiles = None
    coordinator_cpu_ids = None
    available_cpu_ids = _current_cpu_affinity()
    fixed_tier_assignments = None
    if args.client_cpu_threads:
        if args.clients not in (None, 4):
            parser.error("the four-device CPU profile requires --clients 4")
        clients = 4
        try:
            (
                client_cpu_profiles,
                coordinator_cpu_ids,
                available_cpu_ids,
            ) = _cpu_partition(args.client_cpu_threads, args.reserve_cpu_threads)
        except ValueError as exc:
            parser.error(str(exc))
        fixed_tier_assignments = list(DEFAULT_DEVICE_CAPACITY_TIERS)
    client_executor = None
    client_executor_startup_seconds = 0.0
    original_cpu_affinity = None
    worker_affinity_verifications = []
    if client_cpu_profiles:
        startup_started = time.perf_counter()
        client_executor = ProcessPoolExecutor(
            max_workers=clients,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_client_worker,
            initargs=(bundle, base_spec),
        )
        worker_affinity_verifications = list(
            client_executor.map(_prime_client_worker, client_cpu_profiles)
        )
        if any(
            sorted(actual) != sorted(profile["logical_cpu_ids"])
            for actual, profile in zip(worker_affinity_verifications, client_cpu_profiles)
        ):
            client_executor.shutdown(wait=True)
            parser.error("a worker process did not accept its requested CPU affinity")
        client_executor_startup_seconds = time.perf_counter() - startup_started
        original_cpu_affinity = _current_cpu_affinity()
        _set_worker_affinity(coordinator_cpu_ids)
    records = []
    raw_runs = []
    for scale in args.scales:
        scaled_spec = _scaled_spec(base_spec, scale, args.capacity_scaling)
        client_tiers = _capacity_candidates(scaled_spec)
        print(
            f"Starting scale={scale}x: hidden_dim={scaled_spec.hidden_dim}, "
            f"client_tiers={[(tier.depth, tier.width) for tier in client_tiers]}, "
            f"evaluation_widths={list(scaled_spec.widths)}, "
            f"capacity_scaling={args.capacity_scaling}, "
            f"strategies={len(args.strategies)}, "
            f"seeds={len(args.seeds)}",
            flush=True,
        )
        for strategy in args.strategies:
            for seed in args.seeds:
                if strategy == "progressive_growth":
                    result = _run_progressive_growth(
                        bundle,
                        base_spec,
                        scaled_spec,
                        args,
                        seed,
                        client_cpu_profiles,
                        coordinator_cpu_ids,
                        client_executor,
                        fixed_tier_assignments,
                    )
                else:
                    result = run_federated_experiment(
                        bundle,
                        scaled_spec,
                        variant=args.variant,
                        rounds=args.rounds,
                        seed=seed,
                        num_clients=clients,
                        strategy=strategy,
                        optimizer_name="adam",
                        client_cpu_profiles=client_cpu_profiles,
                        fixed_tier_assignments=fixed_tier_assignments,
                        coordinator_cpu_ids=coordinator_cpu_ids,
                        client_executor=client_executor,
                    )
                record = _seed_record(result, scale, scaled_spec, bundle.input_dim)
                records.append(record)
                raw_runs.append(result)
                print(
                    f"scale={scale}x strategy={strategy} seed={seed} "
                    f"macro_f1={record['score_macro_f1']:.4f} "
                    f"active_params={record['mean_active_client_parameters']:.0f} "
                    f"sim_seconds={record['simulation_wall_clock_seconds']:.1f}",
                    flush=True,
                )

    if client_executor is not None:
        client_executor.shutdown(wait=True)
        _set_worker_affinity(original_cpu_affinity)

    source_archive = (
        Path(args.data_dir) / f"{args.dataset}.zip"
        if args.dataset != "digits" else None
    )
    summary = _summarize(records, args.scales, args.strategies)
    output = {
        "experiment": "elastic_model_width_scaling",
        "dataset": args.dataset,
        "train_samples": len(bundle.y_train),
        "validation_samples": len(bundle.y_val),
        "test_samples": len(bundle.y_test),
        "input_dim": bundle.input_dim,
        "num_classes": bundle.num_classes,
        "scales": list(args.scales),
        "capacity_scaling": args.capacity_scaling,
        "strategies": list(args.strategies),
        "seeds": list(args.seeds),
        "rounds": args.rounds,
        "clients": clients,
        "torch_intraop_threads": torch.get_num_threads(),
        "client_cpu_profiles": client_cpu_profiles,
        "coordinator_logical_cpu_ids": coordinator_cpu_ids,
        "worker_affinity_verifications": worker_affinity_verifications,
        "client_worker_pool_startup_seconds": client_executor_startup_seconds,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "available_logical_cpu_ids": available_cpu_ids,
            "physical_core_count": (
                psutil.cpu_count(logical=False) if psutil is not None else None
            ),
            "memory_bytes": (
                int(psutil.virtual_memory().total) if psutil is not None else None
            ),
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cpu_partition_mode": (
                "four affinity-limited virtual client processes"
                if client_cpu_profiles else "single-process simulator"
            ),
            "physical_device_measurements": False,
        },
        "metric_scope": {
            "weights": "parameter-count-derived float32 storage; excludes gradients, optimizer state, activations, runtime, and any OS overhead",
            "flops": "estimated forward operations per example from the benchmark model dimensions",
            "communication": "simulated tensor payload bytes; excludes protocol and network overhead",
            "wall_clock": (
                "local CPU simulation run time with client training executed in "
                "affinity-limited worker processes; excludes shared pool startup, "
                "and is not physical-device or network latency"
            ),
            "client_training_seconds": (
                "per-device local model construction, data-loader setup, and local "
                "training duration; excludes inter-process serialization and aggregation"
            ),
            "inference": "paired confidence intervals are descriptive; this scaling sweep is exploratory and does not establish non-inferiority",
        },
        "run_manifest": build_array_run_manifest(
            dataset=args.dataset,
            splits={
                "train": (bundle.X_train, bundle.y_train),
                "validation": (bundle.X_val, bundle.y_val),
                "test": (bundle.X_test, bundle.y_test),
            },
            source_path=source_archive,
            arguments={
                "base_dataset_spec": asdict(base_spec),
                "scales": list(args.scales),
                "capacity_scaling": args.capacity_scaling,
                "scaled_specs": {
                    str(scale): asdict(
                        _scaled_spec(base_spec, scale, args.capacity_scaling)
                    )
                    for scale in args.scales
                },
                "strategies": list(args.strategies),
                "seeds": list(args.seeds),
                "rounds": args.rounds,
                "clients": clients,
                "max_samples_override": args.max_samples,
                "torch_intraop_threads": torch.get_num_threads(),
                "client_cpu_profiles": client_cpu_profiles,
                "coordinator_logical_cpu_ids": coordinator_cpu_ids,
                "fixed_tier_assignments": fixed_tier_assignments,
            },
        ),
        "summary": summary,
        "seed_records": records,
        "raw_runs": raw_runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved scaling benchmark: {output_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
