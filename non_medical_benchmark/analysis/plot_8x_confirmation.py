"""Plot the fresh ten-seed 8x four-device confirmation block."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..stats import confidence_interval


METHODS = ("fedavg", "heterofl", "fjord")
LABELS = {"fedavg": "FedAvg", "heterofl": "HeteroFL", "fjord": "FjORD"}
COLORS = {"fedavg": "#38598c", "heterofl": "#2a9d8f", "fjord": "#e76f51"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=".benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json",
    )
    parser.add_argument(
        "--output",
        default="results_graphs/four_device_8x_confirmation_10seeds.png",
    )
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records = data["seed_records"]
    grouped = {
        method: [row for row in records if row["strategy"] == method]
        for method in METHODS
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    x = np.arange(len(METHODS), dtype=float)
    shift = 0.13
    for index, method in enumerate(METHODS):
        rows = grouped[method]
        capacity = confidence_interval(
            (row["score_macro_f1"] for row in rows), bounds=(0, 1)
        )
        full = confidence_interval(
            (row["full_supernet_macro_f1"] for row in rows), bounds=(0, 1)
        )
        ax.errorbar(
            x[index] - shift,
            capacity["mean"],
            yerr=[[capacity["mean"] - capacity["ci95_low"]], [capacity["ci95_high"] - capacity["mean"]]],
            fmt="o",
            color=COLORS[method],
            capsize=3,
            markersize=6,
            linewidth=1.4,
            label="Capacity-selected client model" if index == 0 else None,
        )
        ax.errorbar(
            x[index] + shift,
            full["mean"],
            yerr=[[full["mean"] - full["ci95_low"]], [full["ci95_high"] - full["mean"]]],
            fmt="s",
            color=COLORS[method],
            markerfacecolor="white",
            capsize=3,
            markersize=6,
            linewidth=1.4,
            label="Full shared model" if index == 0 else None,
        )
    ax.set_xticks(x, [LABELS[name] for name in METHODS])
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Accuracy depends on the evaluated model")
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)

    ax = axes[1]
    for index, method in enumerate(METHODS):
        seconds = [row["simulation_wall_clock_seconds"] for row in grouped[method]]
        median = statistics.median(seconds)
        lower, upper = np.percentile(seconds, [25, 75])
        ax.errorbar(
            median,
            index,
            xerr=[[median - lower], [upper - median]],
            fmt="o",
            color=COLORS[method],
            capsize=4,
            markersize=7,
            linewidth=2,
        )
    ax.set_yticks(np.arange(len(METHODS)), [LABELS[name] for name in METHODS])
    ax.invert_yaxis()
    ax.set_xlabel("Median wall-clock seconds / 8 rounds")
    ax.set_title("Single-host CPU simulation time")
    ax.grid(axis="x", color="#d9dee7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("4-client, 8× HAR benchmark — 10 fresh paired seeds", fontsize=13, fontweight="semibold")
    fig.text(
        0.5,
        0.015,
        "Server MLP: 2.81M parameters. Heterogeneous clients: 29,958 active parameters on average. "
        "Error bars: 95% seed-level t intervals (macro-F1); interquartile range (time).",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.92))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
