"""Plot the matched 8x/16x four-worker HAR scaling results."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    8: ROOT / ".benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json",
    16: ROOT / ".benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_16_8r_5seeds.json",
}
METHODS = ("fedavg", "heterofl", "fjord", "cc_efl")
LABELS = {"fedavg": "FedAvg", "heterofl": "HeteroFL", "fjord": "FjORD", "cc_efl": "CC-EFL"}
COLORS = {"fedavg": "#4c78a8", "heterofl": "#54a24b", "fjord": "#e45756", "cc_efl": "#1f9e89"}


def main() -> None:
    summaries: dict[tuple[int, str], dict] = {}
    records: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for scale, path in RUNS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.update({(int(row["scale_factor"]), row["strategy"]): row for row in payload["summary"]})
        for row in payload["seed_records"]:
            records[(int(row["scale_factor"]), row["strategy"])].append(row)

    x = np.arange(2)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    fig.suptitle(
        "Four CPU-affinity-limited clients · UCI HAR · 8 rounds · five seeds",
        fontsize=15,
        fontweight="bold",
    )

    for method in METHODS:
        color = COLORS[method]
        time_values = [
            statistics.median(float(r["simulation_wall_clock_seconds"]) for r in records[(scale, method)])
            for scale in (8, 16)
        ]
        axes[0, 0].plot(x, time_values, marker="o", linewidth=2.2, color=color, label=LABELS[method])
        axes[0, 0].annotate(f"{time_values[-1]:.0f}s", (x[-1], time_values[-1]), xytext=(5, 2), textcoords="offset points", color=color, fontsize=8)

        means = [summaries[(scale, method)]["macro_f1_mean"] for scale in (8, 16)]
        lows = [summaries[(scale, method)]["macro_f1_ci95_low"] for scale in (8, 16)]
        highs = [summaries[(scale, method)]["macro_f1_ci95_high"] for scale in (8, 16)]
        yerr = np.array([np.array(means) - lows, np.array(highs) - means])
        axes[0, 1].errorbar(x, means, yerr=yerr, marker="o", capsize=4, linewidth=2.0, color=color, label=LABELS[method])

        full_means = [
            statistics.mean(float(r["full_supernet_macro_f1"]) for r in records[(scale, method)])
            for scale in (8, 16)
        ]
        axes[1, 0].plot(x, full_means, marker="o", linewidth=2.0, color=color, label=LABELS[method])


    fedavg_active = [summaries[(scale, "fedavg")]["mean_active_client_parameters"] for scale in (8, 16)]
    subnet_active = [summaries[(scale, "cc_efl")]["mean_active_client_parameters"] for scale in (8, 16)]
    axes[1, 1].plot(x, fedavg_active, marker="o", linewidth=2.2, color=COLORS["fedavg"], label="Full-model FedAvg")
    axes[1, 1].plot(x, subnet_active, marker="o", linewidth=2.2, color=COLORS["cc_efl"], label="Subnet methods")

    for ax in axes.flat:
        ax.set_xticks(x, ["8×\n2.81M server params", "16×\n10.33M server params"])
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_title("Median eight-round simulation time")
    axes[0, 0].set_ylabel("Seconds (worker startup excluded)")
    axes[0, 1].set_title("Capacity-selected deployment macro-F1")
    axes[0, 1].set_ylabel("Test macro-F1 (95% seed CI)")
    axes[0, 1].set_ylim(0.55, 1.0)
    axes[1, 0].set_title("Full shared-supernet macro-F1")
    axes[1, 0].set_ylabel("Test macro-F1 (seed mean)")
    axes[1, 0].set_ylim(0.65, 1.0)
    axes[1, 1].set_title("Mean active parameters per client-round")
    axes[1, 1].set_ylabel("Trainable parameters (log scale)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(1e4, 2e7)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    axes[1, 1].legend(frameon=False, loc="upper left")
    fig.text(
        0.5,
        -0.01,
        "Virtual workers on one Intel Core Ultra 5 125H host; no per-client RAM cap, physical-device, energy, or network measurements.",
        ha="center",
        fontsize=9,
    )
    output = ROOT / "results_graphs/four_device_cpu_split_8x_16x_followup.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
