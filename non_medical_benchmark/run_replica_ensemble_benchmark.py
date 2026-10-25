"""Benchmark a server-side ensemble trained by CPU-limited small clients.

The experiment first trains one 72-wide parent with four heterogeneous
clients. It then makes function-preserving replicas and assigns each client
to one replica per round. Members specialize on different client updates;
their scaled logits are summed into one server-side predictor.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import DATASETS, get_dataset_spec
from .data import load_dataset, make_client_loader, make_client_partitions
from .federated import (
    ClientArchitecture,
    _capacity_candidates,
    _current_cpu_affinity,
    _initialize_client_worker,
    _parallel_client_train,
    _prime_client_worker,
    _seeded_tier_assignments,
    _set_worker_affinity,
    _state_payload_bytes,
    aggregate_subnet_updates,
    evaluate_candidates,
    run_federated_experiment,
    state_to_numpy,
)
from .models import build_model, load_numpy_state, subnet_flops, subnet_parameters
from .provenance import build_array_run_manifest
from .run_model_scaling_benchmark import (
    DEFAULT_DEVICE_CAPACITY_TIERS,
    _cpu_partition,
)
from .stats import confidence_interval, paired_interval


def _parent_spec(base_spec):
    widths = (base_spec.widths[0], base_spec.widths[1], 72, 72)
    return replace(base_spec, hidden_dim=72, widths=widths)


def _ensemble_metrics(states, bundle, spec, members: int, split: str):
    X = bundle.X_val if split == "validation" else bundle.X_test
    y = bundle.y_val if split == "validation" else bundle.y_test
    model_logits = None
    inputs = torch.from_numpy(np.asarray(X, dtype=np.float32))
    for state in states:
        model = build_model(
            "elastic_plain", bundle.input_dim, spec.num_classes,
            spec.max_depth, spec.hidden_dim,
        ).eval()
        load_numpy_state(model, state)
        with torch.no_grad():
            logits = model.forward_full(inputs)
        model_logits = logits if model_logits is None else model_logits + logits
    predictions = model_logits.argmax(dim=1).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
    }


def _balanced_weights(bundle, spec):
    if spec.class_weighting != "balanced":
        return None
    counts = np.bincount(bundle.y_train, minlength=spec.num_classes).astype(np.float32)
    weights = len(bundle.y_train) / np.maximum(counts * spec.num_classes, 1.0)
    return torch.from_numpy(weights.astype(np.float32))


def _run_one_seed(
    bundle,
    spec,
    seed: int,
    members: int,
    warmup_rounds: int,
    ensemble_rounds: int,
    client_profiles,
    coordinator_cpu_ids,
    client_executor,
):
    clients = len(client_profiles)
    if members < clients or members % clients:
        raise ValueError("member count must be a positive multiple of client count")
    tiers = list(DEFAULT_DEVICE_CAPACITY_TIERS)
    parent = _parent_spec(spec)

    warmup = run_federated_experiment(
        bundle,
        parent,
        variant="elastic_plain",
        rounds=warmup_rounds,
        seed=seed,
        num_clients=clients,
        strategy="heterofl",
        optimizer_name="adam",
        client_cpu_profiles=client_profiles,
        fixed_tier_assignments=tiers,
        coordinator_cpu_ids=coordinator_cpu_ids,
        client_executor=client_executor,
        include_final_test=False,
        return_global_state=True,
    )
    parent_state = warmup["global_model_state"]
    member_states = []
    for _ in range(members):
        clone = {key: np.asarray(value).copy() for key, value in parent_state.items()}
        clone["classifier.weight"] /= float(members)
        clone["classifier.bias"] /= float(members)
        member_states.append(clone)
    parent_model = build_model(
        "elastic_plain", bundle.input_dim, parent.num_classes,
        parent.max_depth, parent.hidden_dim,
    ).eval()
    load_numpy_state(parent_model, parent_state)
    parity_inputs = torch.from_numpy(np.asarray(bundle.X_val[:128], dtype=np.float32))
    initial_ensemble_logits = None
    for member_state in member_states:
        member_model = build_model(
            "elastic_plain", bundle.input_dim, parent.num_classes,
            parent.max_depth, parent.hidden_dim,
        ).eval()
        load_numpy_state(member_model, member_state)
        with torch.no_grad():
            member_logits = member_model.forward_full(parity_inputs)
        initial_ensemble_logits = (
            member_logits
            if initial_ensemble_logits is None
            else initial_ensemble_logits + member_logits
        )
    with torch.no_grad():
        initial_parent_logits = parent_model.forward_full(parity_inputs)
    initial_ensemble_logit_error = float(
        (initial_ensemble_logits - initial_parent_logits).abs().max().item()
    )

    # Keep the four CPU profiles paired with the fixed 24/24/48/72 capacity
    # menu used in the main device-split study.
    assignments = tiers
    candidates = _capacity_candidates(parent)
    partitions = make_client_partitions(
        bundle.y_train, clients, alpha=parent.dirichlet_alpha, seed=seed
    )
    class_weights = _balanced_weights(bundle, parent)
    history = []
    started = time.perf_counter()

    for round_id in range(ensemble_rounds):
        tasks = []
        metadata = []
        for client_id, indices in enumerate(partitions):
            member_id = (round_id * clients + client_id) % members
            architecture = candidates[assignments[client_id]]
            tasks.append(
                {
                    "client_id": client_id,
                    "indices": indices,
                    "architecture": architecture,
                    "variant": "elastic_plain",
                    "global_state": member_states[member_id],
                    "server_control": None,
                    "client_control": None,
                    "drift_correction": False,
                    "class_weights": class_weights,
                    "poison_mode": "none",
                    "proximal_mu": 0.0,
                    "ordered_dropout": False,
                    "ordered_widths": tuple(parent.widths),
                    "optimizer_name": "adam",
                    "seed": seed + 2_000_003 * (round_id + 1) + client_id,
                    "device_profile": client_profiles[client_id],
                    "spec": parent,
                    "classifier_scale": float(members),
                    "classifier_bias_scale": float(members),
                }
            )
            metadata.append((client_id, member_id, indices, architecture))

        round_started = time.perf_counter()
        worker_results = list(client_executor.map(_parallel_client_train, tasks))
        round_upload_bytes = 0
        round_download_bytes = 0
        client_times = []
        client_params = []
        for worker_result, (client_id, member_id, indices, architecture) in zip(
            worker_results, metadata
        ):
            member_states[member_id] = aggregate_subnet_updates(
                member_states[member_id],
                [worker_result["update"]],
                [len(indices)],
                architectures=[architecture],
            )
            round_upload_bytes += sum(
                np.asarray(value).nbytes for value in worker_result["update"].values()
            )
            round_download_bytes += _state_payload_bytes(
                member_states[member_id], architecture
            )
            client_times.append(
                {
                    "client_id": client_id,
                    "seconds": float(worker_result["training_seconds"]),
                }
            )
            client_params.append(int(worker_result["parameter_count"]))

        validation = _ensemble_metrics(
            member_states, bundle, parent, members, split="validation"
        )
        history.append(
            {
                "round": round_id + 1,
                "mean_client_loss": float(np.mean([row["loss"] for row in worker_results])),
                "client_training_seconds": client_times,
                "participating_clients": clients,
                "mean_client_parameters": float(np.mean(client_params)),
                "mean_active_flops": float(
                    np.mean(
                        [
                            subnet_flops(
                                bundle.input_dim,
                                parent.num_classes,
                                candidates[assignments[index]].depth,
                                candidates[assignments[index]].width,
                            )
                            for index in range(clients)
                        ]
                    )
                ),
                "total_upload_bytes": int(round_upload_bytes),
                "total_download_bytes": int(round_download_bytes),
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "member_ids": [item[1] for item in metadata],
            }
        )

    validation_elapsed = time.perf_counter() - started
    test = _ensemble_metrics(member_states, bundle, parent, members, split="test")
    baseline = run_federated_experiment(
        bundle,
        parent,
        variant="elastic_plain",
        rounds=warmup_rounds + ensemble_rounds,
        seed=seed,
        num_clients=clients,
        strategy="heterofl",
        optimizer_name="adam",
        client_cpu_profiles=client_profiles,
        fixed_tier_assignments=tiers,
        coordinator_cpu_ids=coordinator_cpu_ids,
        client_executor=client_executor,
    )
    ensemble_params = members * subnet_parameters(
        bundle.input_dim, parent.num_classes, parent.max_depth, parent.hidden_dim
    )
    client_param_mean = float(np.mean([row["mean_client_parameters"] for row in history]))
    per_round_up = float(np.mean([row["total_upload_bytes"] for row in history]))
    per_round_down = float(np.mean([row["total_download_bytes"] for row in history]))
    return {
        "seed": int(seed),
        "method": "replica_ensemble",
        "clients": clients,
        "members": members,
        "warmup_rounds": warmup_rounds,
        "ensemble_rounds": ensemble_rounds,
        "total_rounds": warmup_rounds + ensemble_rounds,
        "client_capacity_tiers": assignments,
        "client_specs": client_profiles,
        "ensemble_members_parameters": int(ensemble_params),
        "ensemble_members_fp32_mib": float(ensemble_params * 4 / 1024**2),
        "mean_active_client_parameters": client_param_mean,
        "mean_active_client_fp32_mib": float(client_param_mean * 4 / 1024**2),
        "client_parameter_reduction_vs_ensemble": float(
            1.0 - client_param_mean / max(ensemble_params, 1)
        ),
        "mean_upload_bytes_per_round": per_round_up,
        "mean_download_bytes_per_round": per_round_down,
        "mean_total_communication_bytes_per_round": per_round_up + per_round_down,
        "final_round": ensemble_rounds,
        "validation_accuracy_at_last_round": history[-1]["validation_accuracy"],
        "validation_macro_f1_at_last_round": history[-1]["validation_macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "baseline_full_model_accuracy": baseline["final"]["full_model_test_accuracy"],
        "baseline_full_model_macro_f1": baseline["final"]["full_model_test_macro_f1"],
        "baseline_capacity_accuracy": baseline["final"]["selected_test_accuracy"],
        "baseline_capacity_macro_f1": baseline["final"]["selected_test_macro_f1"],
        "baseline_wall_clock_seconds": float(baseline["wall_clock_seconds"]),
        "ensemble_wall_clock_seconds": float(
            warmup["wall_clock_seconds"] + validation_elapsed
        ),
        "mean_slowest_client_training_seconds_per_round": float(
            np.mean([max(item["seconds"] for item in row["client_training_seconds"]) for row in history])
        ),
        "member_assignment_counts": {
            str(member_id): sum(member_id in row["member_ids"] for row in history)
            for member_id in range(members)
        },
        "history": history,
        "warmup_parent_parameters": int(
            subnet_parameters(bundle.input_dim, parent.num_classes, parent.max_depth, parent.hidden_dim)
        ),
        "warmup_parent_validation_macro_f1": float(
            warmup["final"]["selected_val_macro_f1"]
        ),
        "max_abs_initial_ensemble_logit_difference": initial_ensemble_logit_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="har")
    parser.add_argument("--data-dir", default="data/non_medical")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--members", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=8)
    parser.add_argument("--ensemble-rounds", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--client-cpu-threads", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--reserve-cpu-threads", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        default=".benchmarks/model_scaling/har_replica_ensemble_20members_28r_5seeds.json",
    )
    args = parser.parse_args()
    if len(args.client_cpu_threads) != 4 or any(value < 1 for value in args.client_cpu_threads):
        parser.error("the replica ensemble requires four positive per-device CPU budgets")
    if args.members < 4 or args.members % 4:
        parser.error("--members must be a positive multiple of four")
    if args.warmup_rounds < 1 or args.ensemble_rounds < 1:
        parser.error("warmup and ensemble rounds must be positive")
    if args.reserve_cpu_threads < 1 or args.torch_threads < 1:
        parser.error("thread budgets must be positive")

    torch.set_num_threads(args.torch_threads)
    base_spec = get_dataset_spec(args.dataset)
    bundle = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        max_samples=args.max_samples,
    )
    client_profiles, coordinator_cpu_ids, available_cpu_ids = _cpu_partition(
        args.client_cpu_threads, args.reserve_cpu_threads
    )
    startup_started = time.perf_counter()
    executor = ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_client_worker,
        initargs=(bundle, base_spec),
    )
    affinity_verifications = list(executor.map(_prime_client_worker, client_profiles))
    if any(
        sorted(actual) != sorted(profile["logical_cpu_ids"])
        for actual, profile in zip(affinity_verifications, client_profiles)
    ):
        executor.shutdown(wait=True)
        parser.error("a worker process did not accept its requested CPU affinity")
    pool_startup_seconds = time.perf_counter() - startup_started
    original_affinity = _current_cpu_affinity()
    _set_worker_affinity(coordinator_cpu_ids)
    rows = []
    try:
        for seed in args.seeds:
            row = _run_one_seed(
                bundle,
                base_spec,
                seed,
                args.members,
                args.warmup_rounds,
                args.ensemble_rounds,
                client_profiles,
                coordinator_cpu_ids,
                executor,
            )
            rows.append(row)
            print(
                f"seed={seed} ensemble_f1={row['test_macro_f1']:.4f} "
                f"capacity_f1={row['baseline_capacity_macro_f1']:.4f} "
                f"full_f1={row['baseline_full_model_macro_f1']:.4f} "
                f"ensemble_seconds={row['ensemble_wall_clock_seconds']:.1f}",
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
        _set_worker_affinity(original_affinity)

    ensemble_ci = confidence_interval(
        (row["test_macro_f1"] for row in rows), bounds=(0, 1)
    )
    baseline_ci = confidence_interval(
        (row["baseline_full_model_macro_f1"] for row in rows), bounds=(0, 1)
    )
    capacity_ci = confidence_interval(
        (row["baseline_capacity_macro_f1"] for row in rows), bounds=(0, 1)
    )
    paired_vs_full = paired_interval(
        [row["baseline_full_model_macro_f1"] for row in rows],
        [row["test_macro_f1"] for row in rows],
    )
    paired_vs_capacity = paired_interval(
        [row["baseline_capacity_macro_f1"] for row in rows],
        [row["test_macro_f1"] for row in rows],
    )
    source_archive = (
        Path(args.data_dir) / f"{args.dataset}.zip"
        if args.dataset != "digits" else None
    )
    output = {
        "experiment": "replicated-small-model-server-ensemble",
        "dataset": args.dataset,
        "model_family": "residual-free LayerNorm MLP",
        "members": args.members,
        "warmup_rounds": args.warmup_rounds,
        "ensemble_rounds": args.ensemble_rounds,
        "total_rounds": args.warmup_rounds + args.ensemble_rounds,
        "seeds": list(args.seeds),
        "four_device_cpu_profiles": client_profiles,
        "coordinator_logical_cpu_ids": coordinator_cpu_ids,
        "worker_affinity_verifications": affinity_verifications,
        "worker_pool_startup_seconds": pool_startup_seconds,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "available_logical_cpu_ids": available_cpu_ids,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "physical_device_measurements": False,
        },
        "aggregate": {
            "ensemble_test_macro_f1": ensemble_ci,
            "baseline_full_model_test_macro_f1": baseline_ci,
            "baseline_capacity_test_macro_f1": capacity_ci,
            "paired_ensemble_minus_full_fedavg_macro_f1": paired_vs_full,
            "paired_ensemble_minus_single_heterofl_capacity_macro_f1": paired_vs_capacity,
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
                "members": args.members,
                "warmup_rounds": args.warmup_rounds,
                "ensemble_rounds": args.ensemble_rounds,
                "seeds": list(args.seeds),
                "cpu_profiles": client_profiles,
                "coordinator_logical_cpu_ids": coordinator_cpu_ids,
                "base_dataset_spec": asdict(base_spec),
            },
        ),
        "runs": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved replica ensemble benchmark: {output_path}", flush=True)
    print(json.dumps(output["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
