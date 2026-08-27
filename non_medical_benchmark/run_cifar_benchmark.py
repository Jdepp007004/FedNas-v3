"""Run CIFAR-10/100 experiments for the elastic federated architecture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .cifar import load_cifar_bundle, run_cifar_experiment
from .stats import confidence_interval, paired_interval


def _summary(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[result["strategy"]].append(result)
    rows = []
    for strategy, seeded_results in sorted(groups.items()):
        finals = [result["final"] for result in seeded_results]
        accuracy = [float(item["selected_test_accuracy"]) for item in finals]
        accuracy_stats = confidence_interval(accuracy, bounds=(0.0, 1.0))
        rows.append(
            {
                "variant": "elastic_cnn",
                "strategy": strategy,
                "seeds": len(finals),
                "test_accuracy_mean": accuracy_stats["mean"],
                "test_accuracy_std": accuracy_stats["std"],
                "test_accuracy_ci95_low": accuracy_stats["ci95_low"],
                "test_accuracy_ci95_high": accuracy_stats["ci95_high"],
                "val_accuracy_mean": sum(float(item["selected_val_accuracy"]) for item in finals) / len(finals),
                "upload_bytes_per_round_mean": sum(float(item["total_upload_bytes"]) for item in finals) / len(finals),
                "download_bytes_per_round_mean": sum(float(item["total_download_bytes"]) for item in finals) / len(finals),
                "active_flops_per_client_mean": sum(float(item["mean_active_flops"]) for item in finals) / len(finals),
                "client_parameters_mean": sum(float(item["mean_client_parameters"]) for item in finals) / len(finals),
                "wall_clock_seconds_mean": sum(
                    float(result.get("wall_clock_seconds", 0.0))
                    for result in seeded_results
                ) / len(seeded_results),
            }
        )
    baseline = next((row for row in rows if row["strategy"] == "fedavg"), None)
    if baseline:
        for row in rows:
            row["upload_reduction_vs_fedavg"] = 1.0 - row["upload_bytes_per_round_mean"] / max(
                baseline["upload_bytes_per_round_mean"], 1.0
            )
            row["flops_reduction_vs_fedavg"] = 1.0 - row["active_flops_per_client_mean"] / max(
                baseline["active_flops_per_client_mean"], 1.0
            )
            row["parameter_reduction_vs_fedavg"] = 1.0 - row["client_parameters_mean"] / max(
                baseline["client_parameters_mean"], 1.0
            )
            if row["strategy"] != "fedavg":
                baseline_results = {
                    result["seed"]: result["final"]
                    for result in groups.get("fedavg", [])
                }
                method_results = {
                    result["seed"]: result["final"]
                    for result in groups.get(row["strategy"], [])
                }
                shared = sorted(set(baseline_results) & set(method_results))
                if shared:
                    delta = paired_interval(
                        [baseline_results[seed]["selected_test_accuracy"] for seed in shared],
                        [method_results[seed]["selected_test_accuracy"] for seed in shared],
                    )
                    row["paired_accuracy_delta_vs_fedavg_mean"] = delta["mean"]
                    row["paired_accuracy_delta_vs_fedavg_ci95_low"] = delta["ci95_low"]
                    row["paired_accuracy_delta_vs_fedavg_ci95_high"] = delta["ci95_high"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-dir", default="data/non_medical/cifar")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=5000)
    parser.add_argument("--max-test", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "elastic", "maxnet", "cc_efl"),
        default=("fedavg", "heterofl", "fjord", "fedrolex", "cc_efl"),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_cifar_bundle(
        args.dataset,
        data_dir=args.data_dir,
        download=args.download,
        max_train=args.max_train,
        max_test=args.max_test,
    )
    results = []
    for strategy in args.strategies:
        for seed in args.seeds:
            results.append(
                run_cifar_experiment(
                    bundle,
                    strategy=strategy,
                    rounds=args.rounds,
                    seed=seed,
                    clients=args.clients,
                    batch_size=args.batch_size,
                    local_epochs=args.local_epochs,
                    learning_rate=args.learning_rate,
                )
            )
    payload = {
        "dataset": args.dataset,
        "train_samples": len(bundle.train_dataset),
        "validation_samples": len(bundle.val_dataset),
        "test_samples": len(bundle.test_dataset),
        "rounds": args.rounds,
        "seeds": list(args.seeds),
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
