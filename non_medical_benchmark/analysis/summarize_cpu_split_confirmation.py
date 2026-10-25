"""Summarize paired CPU-split confirmation runs and per-device endpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ..stats import confidence_interval, paired_interval


def interval(values: list[float]) -> dict[str, float | int]:
    return confidence_interval(values, bounds=(0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw model-scaling JSON")
    parser.add_argument("--output", required=True, help="Analysis JSON output")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    seed_records = data["seed_records"]
    raw_runs = data["raw_runs"]
    records_by_method: dict[str, list[dict]] = defaultdict(list)
    raw_by_key = {
        (row["strategy"], int(row["seed"])): row for row in raw_runs
    }
    for row in seed_records:
        records_by_method[row["strategy"]].append(row)

    summary: dict[str, dict] = {}
    for method, rows in records_by_method.items():
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        summary[method] = {
            "seeds": len(rows),
            "capacity_selected_macro_f1": interval(
                [float(row["score_macro_f1"]) for row in rows]
            ),
            "capacity_selected_accuracy": interval(
                [float(row["score_accuracy"]) for row in rows]
            ),
            "full_supernet_macro_f1": interval(
                [float(row["full_supernet_macro_f1"]) for row in rows]
            ),
            "full_supernet_accuracy": interval(
                [float(row["full_supernet_accuracy"]) for row in rows]
            ),
            "median_simulation_seconds": statistics.median(
                float(row["simulation_wall_clock_seconds"]) for row in rows
            ),
            "mean_active_client_parameters": statistics.mean(
                float(row["mean_active_client_parameters"]) for row in rows
            ),
            "mean_selected_deployed_parameters": statistics.mean(
                float(row["selected_deployed_parameters"]) for row in rows
            ),
            "coordinate_coverage": statistics.mean(
                float(row["coordinate_coverage"]) for row in rows
            ),
        }

    baseline = records_by_method.get("fedavg", [])
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    paired: dict[str, dict] = {}
    for method, rows in records_by_method.items():
        if method == "fedavg":
            continue
        candidate_by_seed = {int(row["seed"]): row for row in rows}
        seeds = sorted(set(baseline_by_seed) & set(candidate_by_seed))
        fed = [baseline_by_seed[seed] for seed in seeds]
        candidate = [candidate_by_seed[seed] for seed in seeds]
        paired[method] = {
            "seeds": len(seeds),
            "capacity_selected_f1_delta_vs_fedavg_capacity": paired_interval(
                [float(row["score_macro_f1"]) for row in fed],
                [float(row["score_macro_f1"]) for row in candidate],
            ),
            "client_deployment_f1_delta_vs_full_fedavg_descriptive": paired_interval(
                [float(row["full_supernet_macro_f1"]) for row in fed],
                [float(row["score_macro_f1"]) for row in candidate],
            ),
            "full_supernet_f1_delta_vs_full_fedavg": paired_interval(
                [float(row["full_supernet_macro_f1"]) for row in fed],
                [float(row["full_supernet_macro_f1"]) for row in candidate],
            ),
            "median_paired_fedavg_to_candidate_time_ratio": statistics.median(
                float(base["simulation_wall_clock_seconds"])
                / max(float(other["simulation_wall_clock_seconds"]), 1e-12)
                for base, other in zip(fed, candidate)
            ),
        }

    first = seed_records[0]
    device_profiles = first.get("client_device_specs", [])
    per_device: dict[str, dict] = {}
    for profile in device_profiles:
        client_id = int(profile["client_id"])
        device_id = str(profile["device_id"])
        by_method: dict[str, dict] = {}
        for method, rows in records_by_method.items():
            f1_scores: list[float] = []
            accuracies: list[float] = []
            local_seconds: list[float] = []
            selected_parameters: list[float] = []
            active_parameters: list[float] = []
            for record in rows:
                seed = int(record["seed"])
                raw = raw_by_key[(method, seed)]
                raw_profile = raw["client_cpu_profiles"][client_id]
                tier = int(raw_profile["capacity_tier"])
                selected = next(
                    item
                    for item in raw["final"]["selected_models_by_capacity"]
                    if int(item["capacity_tier"]) == tier
                )
                spec = record["client_device_specs"][client_id]
                f1_scores.append(float(selected["test_macro_f1"]))
                accuracies.append(float(selected["test_accuracy"]))
                local_seconds.append(
                    float(spec["mean_local_training_seconds_per_round"])
                )
                selected_parameters.append(float(selected["parameters"]))
                active_parameters.append(float(spec["training_parameters"]))
            by_method[method] = {
                "selected_test_macro_f1": interval(f1_scores),
                "selected_test_accuracy": interval(accuracies),
                "mean_selected_deployment_parameters": statistics.mean(
                    selected_parameters
                ),
                "mean_active_training_parameters": statistics.mean(
                    active_parameters
                ),
                "mean_local_training_seconds_per_round": statistics.mean(
                    local_seconds
                ),
            }
        per_device[device_id] = {
            "profile_name": profile["profile_name"],
            "logical_processors": int(profile["logical_processors"]),
            "torch_threads": int(profile["torch_threads"]),
            "capacity_tier": int(profile["capacity_tier"]),
            "capacity_depth": int(profile["capacity_depth"]),
            "capacity_width": int(profile["capacity_width"]),
            "capacity_parameters": int(
                next(
                    item["parameters"]
                    for item in first["client_capacity_menu"]
                    if int(item["depth"]) == int(profile["capacity_depth"])
                    and int(item["width"]) == int(profile["capacity_width"])
                )
            ),
            "methods": by_method,
        }

    result = {
        "source": str(args.input),
        "dataset": data["dataset"],
        "scale_factor": data["scales"][0],
        "rounds": data["rounds"],
        "run_count": len(seed_records),
        "summary": summary,
        "paired_vs_fedavg": paired,
        "per_device": per_device,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved analysis: {output}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
