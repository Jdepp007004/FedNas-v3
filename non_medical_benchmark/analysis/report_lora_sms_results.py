"""Audit partitions and derive correct adapter-update resource accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..run_pretrained_head_benchmark import (
    DEFAULT_ROOT,
    ensure_sms_archive,
    read_sms_corpus,
    sha256_file,
    split_indices,
)
from ..run_lora_sms_benchmark import make_sms_client_partitions
from ..stats import confidence_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=".benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1.json",
    )
    parser.add_argument(
        "--output",
        default=".benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reported.json",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_ROOT / "data"))
    args = parser.parse_args()

    source_path = Path(args.input)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    archive_path = ensure_sms_archive(
        Path(args.data_dir) / "sms+spam+collection.zip"
    )
    archive_hash = sha256_file(archive_path)
    if archive_hash != payload["source_archive_sha256"]:
        raise ValueError("SMS source archive hash does not match the benchmark output")
    texts, labels = read_sms_corpus(archive_path)
    train_idx, val_idx, test_idx = split_indices(texts, labels)
    split_hash = hashlib.sha256(
        np.concatenate((train_idx, val_idx, test_idx)).astype("<i8").tobytes()
    ).hexdigest()
    if split_hash != payload["split_indices_sha256"]:
        raise ValueError("Duplicate-grouped split hash does not match the run")

    train_y = labels[train_idx]
    configuration = payload["configuration"]
    clients = int(configuration["clients"])
    alpha = float(configuration["dirichlet_alpha"])
    hidden_size = 128
    bytes_per_parameter = 4
    corrected_rows = []
    for row in payload["results"]:
        partitions = make_sms_client_partitions(
            train_y,
            clients,
            alpha=alpha,
            seed=10_000 + int(row["seed"]),
            partition_mode=configuration.get(
                "partition_mode", "dirichlet_label_and_quantity"
            ),
        )
        sizes = [len(part) for part in partitions]
        digest = hashlib.sha256(
            b"".join(np.asarray(part, dtype="<i8").tobytes() for part in partitions)
        ).hexdigest()
        if sizes != row["client_partition_sizes"]:
            raise ValueError("Reconstructed client sizes do not match the run")
        if digest != row["client_partition_sha256"]:
            raise ValueError("Reconstructed client partition hash does not match the run")
        row["client_class_counts"] = [
            {
                "ham": int(np.sum(train_y[part] == 0)),
                "spam": int(np.sum(train_y[part] == 1)),
            }
            for part in partitions
        ]
        active_counts = []
        payload_coordinates = []
        for round_entry in row["round_manifest"]:
            round_count = 0
            for window in round_entry["windows"]:
                rank = int(window["rank"])
                active_counts.append(4 * hidden_size * rank + 2 * hidden_size + 2)
                round_count += 4 * hidden_size * rank + 2 * hidden_size + 2
            payload_coordinates.append(round_count)
        mean_bytes = (
            float(np.mean(payload_coordinates)) * bytes_per_parameter
        )
        row["mean_active_trainable_parameters_per_client"] = float(
            np.mean(active_counts)
        )
        row["mean_parameter_coordinates_uploaded_per_round"] = float(
            np.mean(payload_coordinates)
        )
        row["upload_bytes_per_round_float32"] = mean_bytes
        row["download_bytes_per_round_float32"] = mean_bytes
        row["upload_bytes_per_run_float32"] = (
            mean_bytes * int(row["rounds"])
        )
        row["download_bytes_per_run_float32"] = (
            mean_bytes * int(row["rounds"])
        )
        row["client_partition_size_cv"] = float(
            np.std(sizes, ddof=0) / np.mean(sizes)
        )
        corrected_rows.append(row)

    summaries = []
    for method in configuration["strategies"]:
        rows = [row for row in corrected_rows if row["method"] == method]
        summaries.append({
            "method": method,
            "seeds": len(rows),
            "capacity_weighted_test_macro_f1": confidence_interval(
                [row["capacity_weighted_test_macro_f1"] for row in rows],
                bounds=(0.0, 1.0),
            ),
            "capacity_weighted_test_accuracy": confidence_interval(
                [row["capacity_weighted_test_accuracy"] for row in rows],
                bounds=(0.0, 1.0),
            ),
            "mean_active_trainable_parameters_per_client": float(np.mean([
                row["mean_active_trainable_parameters_per_client"] for row in rows
            ])),
            "full_rank_trainable_parameters": (
                4 * hidden_size * int(payload["encoder"]["lora_supernet_rank"])
                + 2 * hidden_size + 2
            ),
            "mean_parameter_coordinates_uploaded_per_round": float(np.mean([
                row["mean_parameter_coordinates_uploaded_per_round"] for row in rows
            ])),
            "mean_upload_bytes_per_round_float32": float(np.mean([
                row["upload_bytes_per_round_float32"] for row in rows
            ])),
            "mean_download_bytes_per_round_float32": float(np.mean([
                row["download_bytes_per_round_float32"] for row in rows
            ])),
            "mean_client_partition_size_cv": float(np.mean([
                row["client_partition_size_cv"] for row in rows
            ])),
            "mean_rank_coordinate_coverage": float(np.mean([
                row["rank_coordinate_coverage_fraction"] for row in rows
            ])),
            "mean_wall_clock_seconds": float(np.mean([
                row["wall_clock_seconds"] for row in rows
            ])),
        })
    fedavg_payload = next(
        row["mean_upload_bytes_per_round_float32"]
        for row in summaries if row["method"] == "fedavg"
    )
    for row in summaries:
        row["upload_payload_reduction_vs_fedavg"] = (
            1.0 - row["mean_upload_bytes_per_round_float32"] / fedavg_payload
        )

    payload["federated_summary"] = summaries
    payload["results"] = corrected_rows
    payload["resource_accounting"] = {
        "dtype": "float32",
        "counts": "only trainable LoRA and classifier-head coordinates in each recorded client window",
        "transport_metadata_included": False,
        "frozen_backbone_update_bytes": 0,
        "resource_scope": "adapter/head communication only; the frozen BERT-Tiny backbone remains stored and executed",
        "correction": "FedAvg and FedProx update full rank-4 adapters; heterogeneous strategies update ranks recorded in each round manifest",
    }
    payload["client_partition_note"] = (
        "Seeded Dirichlet partitioning induces both label skew and substantial "
        "quantity skew; per-client ham/spam counts and size CV are reported per run."
    )
    payload["postprocessing"] = {
        "source_raw_output": str(source_path),
        "source_raw_output_sha256": sha256_file(source_path),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "purpose": "verify data partition hashes and correct active-update accounting from saved per-round rank windows",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "federated_summary": summaries,
        "paired_inference": payload["paired_inference"],
    }, indent=2))


if __name__ == "__main__":
    main()
