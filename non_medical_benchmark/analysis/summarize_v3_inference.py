"""Paired-seed inference and Holm correction for the corrected v3 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t, ttest_rel


TASK_FILES = {
    "digits": "phase2_digits_adam_20r_5seeds_v3.json",
    "adult": "phase2_adult_adam_10r_5seeds_v3.json",
    "bank_marketing": "phase2_bank_adam_10r_5seeds_v3.json",
    "har": "phase2_har_adam_8r_5seeds_v3.json",
}
ENDPOINTS = {
    "digits": ("selected_test_accuracy", "capacity-selected accuracy"),
    "adult": ("selected_test_macro_f1", "capacity-selected macro-F1"),
    "bank_marketing": ("selected_test_macro_f1", "capacity-selected macro-F1"),
    "har": ("selected_test_macro_f1", "capacity-selected macro-F1"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_result(reference: list[float], method: list[float]) -> dict:
    if len(reference) != len(method) or not reference:
        raise ValueError("paired observations must have matching nonzero lengths")
    differences = [candidate - baseline for baseline, candidate in zip(reference, method)]
    n = len(differences)
    delta = mean(differences)
    sample_sd = stdev(differences) if n > 1 else 0.0
    half_width = float(t.ppf(0.975, n - 1) * sample_sd / math.sqrt(n)) if n > 1 else 0.0
    p_value = float(ttest_rel(method, reference).pvalue) if n > 1 else 1.0
    if not math.isfinite(p_value):
        p_value = 1.0
    return {
        "n_seeds": n,
        "paired_delta_mean": delta,
        "paired_delta_ci95_low_unadjusted": delta - half_width,
        "paired_delta_ci95_high_unadjusted": delta + half_width,
        "paired_t_p_unadjusted": p_value,
        "per_seed_deltas": differences,
    }


def holm(rows: list[dict], p_key: str = "paired_t_p_unadjusted") -> list[dict]:
    order = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    running_max = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        raw = rows[index][p_key]
        running_max = max(running_max, min(1.0, (count - rank) * raw))
        rows[index]["paired_t_p_holm"] = running_max
    return rows


def seed_values(payload: dict, strategy: str, endpoint: str) -> dict[int, float]:
    return {
        int(row["seed"]): float(row["final"][endpoint])
        for row in payload["results"]
        if row["strategy"] == strategy
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default=".benchmarks")
    parser.add_argument("--output", default=".benchmarks/phase2_v3_inference.json")
    args = parser.parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    loaded = {}
    files = {}
    for task, filename in TASK_FILES.items():
        path = benchmark_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        endpoint, endpoint_name = ENDPOINTS[task]
        loaded[task] = (payload, endpoint, endpoint_name)
        files[task] = {"path": str(path), "sha256": file_sha256(path)}

    per_task_all_vs_fedavg = {}
    per_task_ablations = {}
    primary_across_tasks = []
    for task, (payload, endpoint, endpoint_name) in loaded.items():
        strategies = sorted({row["strategy"] for row in payload["results"]})
        baseline = seed_values(payload, "fedavg", endpoint)
        comparisons = []
        for strategy in strategies:
            if strategy == "fedavg":
                continue
            values = seed_values(payload, strategy, endpoint)
            shared = sorted(set(baseline) & set(values))
            result = paired_result(
                [baseline[seed] for seed in shared],
                [values[seed] for seed in shared],
            )
            comparisons.append(
                {
                    "task": task,
                    "endpoint": endpoint_name,
                    "reference": "fedavg",
                    "method": strategy,
                    **result,
                }
            )
        per_task_all_vs_fedavg[task] = holm(comparisons)

        cc = seed_values(payload, "cc_efl", endpoint)
        ablation_rows = []
        for strategy in ("cc_efl_no_prox", "cc_efl_uniform", "cc_efl_random_window"):
            if strategy not in strategies:
                continue
            values = seed_values(payload, strategy, endpoint)
            shared = sorted(set(cc) & set(values))
            result = paired_result(
                [cc[seed] for seed in shared],
                [values[seed] for seed in shared],
            )
            ablation_rows.append(
                {
                    "task": task,
                    "endpoint": endpoint_name,
                    "reference": "cc_efl",
                    "method": strategy,
                    **result,
                }
            )
        per_task_ablations[task] = holm(ablation_rows)

        values = seed_values(payload, "cc_efl", endpoint)
        shared = sorted(set(baseline) & set(values))
        result = paired_result(
            [baseline[seed] for seed in shared],
            [values[seed] for seed in shared],
        )
        primary_across_tasks.append(
            {
                "task": task,
                "endpoint": endpoint_name,
                "reference": "fedavg",
                "method": "cc_efl",
                **result,
            }
        )

    primary_across_tasks = holm(primary_across_tasks)
    rendered = {
        "protocol": "Paired two-sided Student-t tests on per-seed differences; 95% t intervals are unadjusted; Holm p-values control family-wise error.",
        "primary_cross_task_family": primary_across_tasks,
        "all_strategies_vs_fedavg_holm_by_task": per_task_all_vs_fedavg,
        "cc_efl_component_ablation_holm_by_task": per_task_ablations,
        "input_files": files,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
