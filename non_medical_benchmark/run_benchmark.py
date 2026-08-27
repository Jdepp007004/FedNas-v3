"""CLI entry point for the non-medical federated supernet benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

from .config import DATASETS, get_dataset_spec
from .data import load_dataset
from .federated import run_federated_experiment
from .stats import confidence_interval, paired_interval


def _summary(results: list[dict]) -> list[dict]:
    """Collapse seed-level output into the table used by the paper."""
    groups = defaultdict(list)
    for result in results:
        groups[(result["variant"], result["strategy"])].append(result)
    rows = []
    for (variant, strategy), seeded_results in sorted(groups.items()):
        finals = [item["final"] for item in seeded_results]
        accuracy_stats = confidence_interval(
            (item["selected_test_accuracy"] for item in finals), bounds=(0.0, 1.0)
        )
        f1_stats = confidence_interval(
            (item["selected_test_macro_f1"] for item in finals), bounds=(0.0, 1.0)
        )
        val_stats = confidence_interval(
            (item["selected_val_accuracy"] for item in finals), bounds=(0.0, 1.0)
        )

        def mean(key: str) -> float:
            return sum(float(item[key]) for item in finals) / len(finals)

        rows.append(
            {
                "variant": variant,
                "strategy": strategy,
                "seeds": len(finals),
                "test_accuracy_mean": accuracy_stats["mean"],
                "test_accuracy_std": accuracy_stats["std"],
                "test_accuracy_ci95_low": accuracy_stats["ci95_low"],
                "test_accuracy_ci95_high": accuracy_stats["ci95_high"],
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "macro_f1_ci95_low": f1_stats["ci95_low"],
                "macro_f1_ci95_high": f1_stats["ci95_high"],
                "val_accuracy_mean": val_stats["mean"],
                "val_accuracy_ci95_low": val_stats["ci95_low"],
                "val_accuracy_ci95_high": val_stats["ci95_high"],
                "upload_bytes_per_round_mean": mean("total_upload_bytes"),
                "download_bytes_per_round_mean": mean("total_download_bytes"),
                "active_flops_per_client_mean": mean("mean_active_flops"),
                "client_parameters_mean": mean("mean_client_parameters"),
                "participation_rate_mean": mean("participation_rate"),
                "wall_clock_seconds_mean": sum(
                    float(result.get("wall_clock_seconds", 0.0))
                    for result in seeded_results
                ) / len(seeded_results),
                "selected_depth_mode": max(
                    (item["selected_depth"] for item in finals),
                    key=lambda value: sum(final["selected_depth"] == value for final in finals),
                ),
                "selected_width_mode": max(
                    (item["selected_width"] for item in finals),
                    key=lambda value: sum(final["selected_width"] == value for final in finals),
                ),
            }
        )
    fedavg_rows = {
        row["variant"]: row
        for row in rows
        if row["strategy"] == "fedavg"
    }
    for row in rows:
        baseline = fedavg_rows.get(row["variant"])
        if baseline is None:
            continue
        row["upload_reduction_vs_fedavg"] = float(
            1.0 - row["upload_bytes_per_round_mean"]
            / max(baseline["upload_bytes_per_round_mean"], 1.0)
        )
        row["flops_reduction_vs_fedavg"] = float(
            1.0 - row["active_flops_per_client_mean"]
            / max(baseline["active_flops_per_client_mean"], 1.0)
        )
        row["parameter_reduction_vs_fedavg"] = float(
            1.0 - row["client_parameters_mean"]
            / max(baseline["client_parameters_mean"], 1.0)
        )
        if row["strategy"] != "fedavg":
            baseline_results = {
                result["seed"]: result["final"]
                for result in groups.get((row["variant"], "fedavg"), [])
            }
            method_results = {
                result["seed"]: result["final"]
                for result in groups.get((row["variant"], row["strategy"]), [])
            }
            shared_seeds = sorted(set(baseline_results) & set(method_results))
            if shared_seeds:
                delta = paired_interval(
                    [baseline_results[seed]["selected_test_accuracy"] for seed in shared_seeds],
                    [method_results[seed]["selected_test_accuracy"] for seed in shared_seeds],
                )
                row["paired_accuracy_delta_vs_fedavg_mean"] = delta["mean"]
                row["paired_accuracy_delta_vs_fedavg_ci95_low"] = delta["ci95_low"]
                row["paired_accuracy_delta_vs_fedavg_ci95_high"] = delta["ci95_high"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="digits")
    parser.add_argument("--data-dir", default="data/non_medical")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=("legacy", "elastic"), default=("elastic",))
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=(
            "fedavg", "fedprox", "static", "heterofl", "fjord", "fedrolex",
            "elastic", "maxnet", "scaffold", "elastic_scaffold", "cc_efl",
        ),
        default=("fedavg", "heterofl", "fjord", "fedrolex", "cc_efl"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(42,))
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--straggler-rate", type=float, default=0.0)
    parser.add_argument("--secure-aggregation", action="store_true")
    parser.add_argument(
        "--secure-aggregation-isolated",
        action="store_true",
        help="Run secure-sum clients in separate local processes (protocol harness).",
    )
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--attack-fraction", type=float, default=0.0)
    parser.add_argument(
        "--attack-type",
        choices=("none", "sign_flip", "label_flip", "backdoor"),
        default="none",
    )
    parser.add_argument("--attack-scale", type=float, default=5.0)
    parser.add_argument(
        "--defense",
        choices=("none", "trimmed_mean", "flame"),
        default="none",
    )
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    bundle = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        max_samples=args.max_samples,
    )
    results = []
    for variant in args.variants:
        for strategy in args.strategies:
            for seed in args.seeds:
                results.append(
                    run_federated_experiment(
                        bundle,
                        spec,
                        variant=variant,
                        rounds=args.rounds,
                        seed=seed,
                        num_clients=args.clients,
                        strategy=strategy,
                        dropout_rate=args.dropout_rate,
                        straggler_rate=args.straggler_rate,
                        secure_aggregation=args.secure_aggregation,
                        secure_aggregation_isolated=args.secure_aggregation_isolated,
                        dp_noise_multiplier=args.dp_noise_multiplier,
                        dp_clip_norm=args.dp_clip_norm,
                        dp_delta=args.dp_delta,
                        attack_fraction=args.attack_fraction,
                        attack_type=args.attack_type,
                        attack_scale=args.attack_scale,
                        defense=args.defense,
                        trim_ratio=args.trim_ratio,
                    )
                )

    payload = {
        "dataset": args.dataset,
        "input_dim": bundle.input_dim,
        "train_samples": len(bundle.X_train),
        "validation_samples": len(bundle.X_val),
        "test_samples": len(bundle.X_test),
        "seeds": list(args.seeds),
        "rounds": args.rounds,
        "dropout_rate": args.dropout_rate,
        "straggler_rate": args.straggler_rate,
        "secure_aggregation": args.secure_aggregation,
        "secure_aggregation_isolated": args.secure_aggregation_isolated,
        "dp_noise_multiplier": args.dp_noise_multiplier,
        "dp_clip_norm": args.dp_clip_norm,
        "dp_delta": args.dp_delta,
        "attack_fraction": args.attack_fraction,
        "attack_type": args.attack_type,
        "attack_scale": args.attack_scale,
        "defense": args.defense,
        "trim_ratio": args.trim_ratio,
        "summary": _summary(results),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
