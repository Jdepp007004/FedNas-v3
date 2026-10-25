"""Create a compact report and figure from the four-device CPU scaling run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "maxnet", "cc_efl")
LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "heterofl": "HeteroFL",
    "fjord": "FjORD",
    "fedrolex": "FedRolex",
    "maxnet": "MaxNet",
    "cc_efl": "CC-EFL",
}
COLORS = {
    "fedavg": "#4c78a8",
    "fedprox": "#f58518",
    "heterofl": "#54a24b",
    "fjord": "#e45756",
    "fedrolex": "#72b7b2",
    "maxnet": "#b279a2",
    "cc_efl": "#1f9e89",
}


def _registry_cpu_model() -> str | None:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(value).strip()
    except (ImportError, OSError):
        return None


def _mean(rows: list[dict], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def _fmt_score(summary: dict) -> str:
    return (
        f"{summary['macro_f1_mean']:.3f} "
        f"({summary['macro_f1_ci95_low']:.3f}–{summary['macro_f1_ci95_high']:.3f}); "
        f"acc {summary['accuracy_mean']:.3f}"
    )


def make_report(source: Path, report_path: Path, plot_path: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    summaries = {
        (int(row["scale_factor"]), row["strategy"]): row
        for row in data["summary"]
    }
    records: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in data["seed_records"]:
        records[(int(row["scale_factor"]), row["strategy"])].append(row)
    scales = sorted(int(item) for item in data["scales"])

    host = data["hardware"]
    cpu_name = _registry_cpu_model() or host.get("processor", "unknown CPU")
    memory_gib = float(host.get("memory_bytes") or 0) / (1024**3)
    lines = [
        "# Four-device CPU-split model-scaling benchmark",
        "",
        f"Raw output: {source.as_posix()}",
        "",
        "## Setup",
        "",
        (
            f"Dataset: UCI HAR, {data['train_samples']} train / "
            f"{data['validation_samples']} validation / {data['test_samples']} test "
            f"examples, six activity classes. The benchmark ran {len(scales)} server "
            f"scales × {len(METHODS)} methods × {len(data['seeds'])} paired seeds "
            f"({len(scales) * len(METHODS) * len(data['seeds'])} runs), eight rounds "
            "per run, with four clients."
        ),
        "",
        (
            f"Host: {cpu_name}; {host.get('physical_core_count')} physical cores, "
            f"{host.get('logical_cpu_count')} logical processors, {memory_gib:.1f} GiB "
            f"RAM; Windows; PyTorch {host.get('pytorch')}; CUDA unavailable."
        ),
        "",
        (
            f"Client workers were affinity-limited to 1, 2, 4, and 8 logical processors "
            f"(15 total); the coordinator used logical processors "
            f"{', '.join(str(item) for item in data['coordinator_logical_cpu_ids'])}. "
            f"Verified worker masks: {data['worker_affinity_verifications']}."
        ),
        "",
        (
            f"Four device profiles use capacity tiers 0, 0, 1, and 2. At the 8× scale "
            f"these are depth/width 1/24, 1/24, 2/48, and 5/72. FedAvg and FedProx "
            f"train the full server model on every client; the subnet strategies train "
            f"the assigned client model. The coordinator's held-out endpoint is selected "
            f"by validation macro-F1."
        ),
        "",
        (
            f"The four worker processes were started once for the matrix; startup took "
            f"{data['client_worker_pool_startup_seconds']:.1f} seconds and is reported "
            "separately. Per-run times below are medians across five seeds. The first "
            "cold run is retained in the raw output but does not set the median."
        ),
        "",
        "## Four virtual device specifications",
        "",
        "| Device | CPU allocation | Logical processor IDs | Capacity tier | 8× client model | FP32 weights |",
        "|---|---:|---|---:|---:|---:|",
    ]

    reference_records = records[(scales[-1], "cc_efl")]
    device_specs = reference_records[0]["client_device_specs"]
    for device in device_specs:
        lines.append(
            f"| {device['device_id']} ({device['profile_name']}) | "
            f"{device['logical_processors']} logical processor(s), "
            f"{device['torch_threads']} PyTorch thread(s) | "
            f"{', '.join(str(item) for item in device['logical_cpu_ids'])} | "
            f"{device['capacity_tier']} | depth {device['capacity_depth']}, "
            f"width {device['capacity_width']}, {device['training_parameters']:,} "
            f"parameters | {device['fp32_parameter_mib']:.3f} MiB |"
        )
    lines.extend(
        [
            "",
            "These are four virtual CPU-limited client processes on one host, not four physical machines. CPU affinity limits where a worker may run; it does not reserve physical cores against other Windows processes. The benchmark does not enforce per-device RAM limits.",
            "",
            "## Median end-to-end simulation time",
            "",
            "Seconds for one eight-round run, including local training, server aggregation, validation, and final evaluation. The shared worker-pool startup is excluded above. Times are not real-device or network latency.",
            "",
            "| Method | 1× / 96 wide | 4× / 384 wide | 8× / 768 wide |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        cells = []
        for scale in scales:
            times = [float(row["simulation_wall_clock_seconds"]) for row in records[(scale, method)]]
            cells.append(f"{statistics.median(times):.1f} s")
        lines.append(f"| {LABELS[method]} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Capacity-selected client deployment scores",
            "",
            "Cells show test macro-F1 with an unadjusted 95% seed-level t interval, followed by accuracy. These metrics average the validation-selected models available under the participating client capacity tiers.",
            "",
            "| Method | 1× | 4× | 8× |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        cells = [_fmt_score(summaries[(scale, method)]) for scale in scales]
        lines.append(f"| {LABELS[method]} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Full shared supernet scores and client cost",
            "",
            "Full-model scores are test macro-F1 / accuracy. Active parameters are the mean number trained per client per round; coordinate coverage is the fraction of server parameter coordinates touched during the eight rounds.",
            "",
            "| Scale | Method | Full-model F1 / accuracy | Active parameters / client | Coordinate coverage |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for scale in scales:
        for method in METHODS:
            rows = records[(scale, method)]
            f1 = _mean(rows, "full_supernet_macro_f1")
            accuracy = _mean(rows, "full_supernet_accuracy")
            summary = summaries[(scale, method)]
            lines.append(
                f"| {scale}× | {LABELS[method]} | {f1:.3f} / {accuracy:.3f} | "
                f"{summary['mean_active_client_parameters']:,.0f} | "
                f"{summary['coordinate_coverage'] * 100:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Per-device local training time at 8×",
            "",
            "Mean seconds per local round across methods' five seeds; this is local model construction, data-loader setup, and client training, excluding aggregation and inter-process serialization.",
            "",
            "| Device | Full FedAvg local seconds / round | CC-EFL local seconds / round | CC-EFL trainable parameters |",
            "|---|---:|---:|---:|",
        ]
    )
    fedavg_records = records[(scales[-1], "fedavg")]
    cc_records = records[(scales[-1], "cc_efl")]
    for device_id, device in enumerate(device_specs):
        baseline = statistics.mean(
            float(row["client_device_specs"][device_id]["mean_local_training_seconds_per_round"])
            for row in fedavg_records
        )
        subnet = statistics.mean(
            float(row["client_device_specs"][device_id]["mean_local_training_seconds_per_round"])
            for row in cc_records
        )
        lines.append(
            f"| {device['device_id']} ({device['profile_name']}) | {baseline:.2f} s | "
            f"{subnet:.2f} s | "
            f"{cc_records[0]['client_device_specs'][device_id]['training_parameters']:,} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            (
                f"At 8×, the full server MLP has {summaries[(scales[-1], 'fedavg')]['full_supernet_parameters']:,.0f} "
                f"parameters, while CC-EFL clients train an average of "
                f"{summaries[(scales[-1], 'cc_efl')]['mean_active_client_parameters']:,.0f} "
                "parameters each. The average active client parameter and estimated FLOP "
                "reductions versus full FedAvg are both "
                f"{summaries[(scales[-1], 'cc_efl')]['active_parameter_reduction_vs_full_fedavg'] * 100:.1f}%."
            ),
            "",
            (
                f"At 8×, CC-EFL's capacity-selected test macro-F1 is "
                f"{summaries[(scales[-1], 'cc_efl')]['macro_f1_mean']:.3f} "
                f"({summaries[(scales[-1], 'cc_efl')]['macro_f1_ci95_low']:.3f}–"
                f"{summaries[(scales[-1], 'cc_efl')]['macro_f1_ci95_high']:.3f}), "
                f"versus {summaries[(scales[-1], 'fedavg')]['macro_f1_mean']:.3f} "
                "for capacity-selected FedAvg. Its full shared-model macro-F1 is "
                f"{_mean(cc_records, 'full_supernet_macro_f1'):.3f}, versus "
                f"{_mean(fedavg_records, 'full_supernet_macro_f1'):.3f} for full FedAvg. "
                f"CC-EFL touches only {summaries[(scales[-1], 'cc_efl')]['coordinate_coverage'] * 100:.1f}% "
                "of the enlarged supernet coordinates in eight rounds, so these data "
                "support small-client participation but do not show that the entire "
                "expanded supernet has converged."
            ),
            "",
            "The largest model here is a 2.81-million-parameter MLP, not an LLM-scale model. Weight sizes and FLOPs are estimates. No physical-device peak RAM, energy, real network time, or per-device memory limits were measured. Reported intervals are exploratory and are not corrected across all method-by-scale comparisons.",
            "",
            "![Four-device CPU-split scaling results](../results_graphs/four_device_cpu_split_scaling.png)",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _make_plot(data, summaries, records, scales, plot_path)


def _make_plot(data: dict, summaries: dict, records: dict, scales: list[int], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(len(scales))

    for method in METHODS:
        medians = [
            statistics.median(
                float(row["simulation_wall_clock_seconds"])
                for row in records[(scale, method)]
            )
            for scale in scales
        ]
        axes[0, 0].plot(x, medians, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
    axes[0, 0].set_title("Median eight-round simulation time")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].set_xticks(x, [f"{scale}×" for scale in scales])
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)

    for method in METHODS:
        means = [summaries[(scale, method)]["macro_f1_mean"] for scale in scales]
        lows = [means[i] - summaries[(scale, method)]["macro_f1_ci95_low"] for i, scale in enumerate(scales)]
        highs = [summaries[(scale, method)]["macro_f1_ci95_high"] - means[i] for i, scale in enumerate(scales)]
        axes[0, 1].errorbar(
            x, means, yerr=np.array([lows, highs]), marker="o", capsize=3,
            linewidth=1.8, color=COLORS[method], label=LABELS[method],
        )
    axes[0, 1].set_title("Capacity-selected deployment macro-F1")
    axes[0, 1].set_ylabel("Test macro-F1")
    axes[0, 1].set_ylim(0.45, 1.0)
    axes[0, 1].set_xticks(x, [f"{scale}×" for scale in scales])
    axes[0, 1].grid(axis="y", alpha=0.25)

    for method in ("fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "maxnet", "cc_efl"):
        means = [
            _mean(records[(scale, method)], "full_supernet_macro_f1")
            for scale in scales
        ]
        axes[1, 0].plot(x, means, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
    axes[1, 0].set_title("Full shared supernet macro-F1")
    axes[1, 0].set_ylabel("Test macro-F1")
    axes[1, 0].set_ylim(0.45, 1.0)
    axes[1, 0].set_xticks(x, [f"{scale}×" for scale in scales])
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(ncol=2, fontsize=8)

    strategies = list(METHODS)
    params = [
        summaries[(scales[-1], method)]["mean_active_client_parameters"]
        for method in strategies
    ]
    axes[1, 1].bar(
        [LABELS[method] for method in strategies],
        params,
        color=[COLORS[method] for method in strategies],
    )
    axes[1, 1].set_title("Mean trainable parameters per client at 8×")
    axes[1, 1].set_ylabel("Parameters")
    axes[1, 1].set_yscale("log")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Four CPU-limited virtual clients · HAR · five seeds · 8 rounds",
        fontsize=15,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=".benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json",
    )
    parser.add_argument(
        "--report",
        default="non_medical_benchmark/FOUR_DEVICE_BENCHMARK.md",
    )
    parser.add_argument(
        "--plot",
        default="results_graphs/four_device_cpu_split_scaling.png",
    )
    args = parser.parse_args()
    make_report(Path(args.input), Path(args.report), Path(args.plot))
    print(f"Saved benchmark report: {args.report}")
    print(f"Saved benchmark figure: {args.plot}")


if __name__ == "__main__":
    main()
