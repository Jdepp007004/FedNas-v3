"""Paired sensitivity analysis for the two SMS client partition protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from ..run_pretrained_head_benchmark import sha256_file
from ..stats import paired_interval


def _holm_adjust(rows: list[dict], key: str) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index][key])
        running_max = max(running_max, adjusted)
        rows[index][key.replace("unadjusted", "holm")] = running_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--balanced", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    joint_path = Path(args.joint)
    balanced_path = Path(args.balanced)
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    balanced = json.loads(balanced_path.read_text(encoding="utf-8"))
    for key in ("source_archive_sha256", "split_indices_sha256"):
        if joint[key] != balanced[key]:
            raise ValueError(f"The two runs disagree on {key}")

    joint_by_method = {
        method: {int(row["seed"]): row for row in joint["results"] if row["method"] == method}
        for method in joint["configuration"]["strategies"]
    }
    balanced_by_method = {
        method: {int(row["seed"]): row for row in balanced["results"] if row["method"] == method}
        for method in balanced["configuration"]["strategies"]
    }
    if set(joint_by_method) != set(balanced_by_method):
        raise ValueError("The two runs do not contain the same methods")

    rows = []
    for method in joint_by_method:
        shared_seeds = sorted(
            set(joint_by_method[method]) & set(balanced_by_method[method])
        )
        if not shared_seeds:
            raise ValueError(f"No matched seeds for {method}")
        left = [
            joint_by_method[method][seed]["capacity_weighted_test_macro_f1"]
            for seed in shared_seeds
        ]
        right = [
            balanced_by_method[method][seed]["capacity_weighted_test_macro_f1"]
            for seed in shared_seeds
        ]
        interval = paired_interval(left, right)
        p_value = float(ttest_rel(right, left).pvalue) if len(shared_seeds) > 1 else 1.0
        rows.append({
            "method": method,
            "seeds": shared_seeds,
            "balanced_minus_joint_macro_f1": interval,
            "paired_t_test_p_unadjusted": p_value if np.isfinite(p_value) else 1.0,
        })
    _holm_adjust(rows, "paired_t_test_p_unadjusted")

    partition_cv = {}
    for label, payload in (("joint", joint), ("balanced", balanced)):
        partition_cv[label] = {
            method: float(summary["mean_client_partition_size_cv"])
            for method, summary in (
                (row["method"], row)
                for row in payload["federated_summary"]
            )
        }

    result = {
        "title": "Paired SMS partition-protocol sensitivity",
        "joint_run": str(joint_path),
        "balanced_run": str(balanced_path),
        "joint_run_sha256": sha256_file(joint_path),
        "balanced_run_sha256": sha256_file(balanced_path),
        "source_archive_sha256": joint["source_archive_sha256"],
        "split_indices_sha256": joint["split_indices_sha256"],
        "paired_seeds": sorted(set(joint["configuration"]["seeds"]) & set(balanced["configuration"]["seeds"])),
        "quantity_skew_cv_by_mode_and_method": partition_cv,
        "effect_direction": "balanced-size test macro-F1 minus joint label-and-quantity Dirichlet test macro-F1",
        "family_correction": "Holm adjustment across the nine paired method-by-protocol contrasts",
        "comparisons": rows,
    }
    result["analysis_script_sha256"] = sha256_file(Path(__file__))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
