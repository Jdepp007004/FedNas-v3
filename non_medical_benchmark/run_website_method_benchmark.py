"""Four-client benchmark of the exact self-serve tabular website method.

Four persistent CPU-affinity-limited workers train the same budget-conditioned
prefixes used by the web client. The baseline trains the full shared model on
every worker. RAM values are planner inputs, not enforced process memory
limits; all four workers run on one host.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
CLIENT = ROOT / "client"
for path in (ROOT, SERVER, CLIENT):
    if str(path) not in os.sys.path:
        os.sys.path.insert(0, str(path))


def _profiles(cpu_ids: list[int]) -> list[dict]:
    partitions = (1, 2, 4, 8)
    names = ("edge-1t", "low-2t", "mid-4t", "high-8t")
    ram = (0.5, 2.0, 4.0, 8.0)
    profiles = []
    cursor = 0
    for name, count, ram_gb in zip(names, partitions, ram):
        assigned = cpu_ids[cursor:cursor + count]
        cursor += count
        profiles.append({
            "name": name,
            "logical_cpu_ids": assigned,
            "cpu_threads": count,
            "dedicated_ram_gb": ram_gb,
            "available_ram_gb": ram_gb,
            "available_cpu_cores": count,
            "dedicated_cpu_cores": count,
            "gpu_available": False,
            "resource_source": "benchmark profile; RAM not OS-limited",
        })
    return profiles


def _current_cpu_ids() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
        ]
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        ok = kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
        return [index for index in range(ctypes.sizeof(ctypes.c_size_t) * 8)
                if process_mask.value & (1 << index)]
    raise RuntimeError("CPU affinity inspection is unsupported on this platform")


def _set_cpu_affinity(cpu_ids: list[int]) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_ids))
        return
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        mask = sum(1 << int(index) for index in cpu_ids)
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
        return
    raise RuntimeError("CPU affinity control is unsupported on this platform")


def _host_ram_gib() -> float | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024 ** 3), 2)
    if hasattr(os, "sysconf"):
        try:
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3), 2)
        except (ValueError, OSError):
            pass
    return None


def _dirichlet_indices(labels: np.ndarray, seed: int, clients: int = 4, alpha: float = 0.5):
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(clients)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(clients, alpha, dtype=float))
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            buckets[client_id].extend(indices[start:start + count].tolist())
            start += count
    if any(len(bucket) < 30 for bucket in buckets):
        return _dirichlet_indices(labels, seed + 1, clients, alpha)
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def _client_worker(connection, profile: dict, frame_json: str, task_config: dict, model_config: dict):
    """Keep one device process alive and train its local CSV partition."""
    import torch
    from io import StringIO
    from tabular_fl import build_loaders_from_frame, train_client_round

    _set_cpu_affinity(profile["logical_cpu_ids"])
    torch.set_num_threads(profile["cpu_threads"])
    try:
        torch.set_num_interop_threads(max(1, min(profile["cpu_threads"], 4)))
    except RuntimeError:
        pass
    local_frame = pd.read_json(StringIO(frame_json), orient="split")
    loaders = build_loaders_from_frame(local_frame, task_config)
    connection.send({
        "ready": True,
        "affinity": _current_cpu_ids(),
        "local_rows": int(len(local_frame)),
    })

    while True:
        payload = connection.recv()
        if payload is None:
            break
        torch.manual_seed(int(payload["seed"]))
        started = time.perf_counter()
        result = train_client_round(
            "", task_config, model_config, payload["weights"],
            int(payload["depth"]), int(payload["width"]),
            epochs=int(payload["epochs"]), data_loaders=loaders,
        )
        connection.send({"result": result, "seconds": time.perf_counter() - started})
    connection.close()


def _make_client_processes(profiles, frames, task_config, model_config):
    context = mp.get_context("spawn")
    processes, connections = [], []
    for profile, frame in zip(profiles, frames):
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_client_worker,
            args=(child, profile, frame.to_json(orient="split"), task_config, model_config),
        )
        process.start()
        child.close()
        processes.append(process)
        connections.append(parent)
    ready = [connection.recv() for connection in connections]
    for profile, info in zip(profiles, ready):
        if info.get("affinity") != profile["logical_cpu_ids"]:
            raise RuntimeError(f"CPU affinity mismatch for {profile['name']}: {info}")
    return processes, connections, ready


def _close_client_processes(processes, connections):
    for connection in connections:
        connection.send(None)
    for connection in connections:
        connection.close()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()


def _evaluate(weights, task_config, model_config, loader, depth, width):
    import torch
    from tabular_fl import ElasticTabularSubnet, load_weights

    model = ElasticTabularSubnet(model_config, depth, width)
    load_weights(model, weights)
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in loader:
            predictions.extend(model(features, depth, width).numpy())
            targets.extend(labels.numpy())
    predicted = np.asarray(predictions).argmax(axis=1)
    actual = np.asarray(targets, dtype=int)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted, labels=list(range(len(task_config["classes"]))),
            average="macro", zero_division=0,
        )),
    }


def _evaluate_full(weights, task_config, model_config, loader):
    return _evaluate(weights, task_config, model_config, loader,
                     model_config["max_depth"], model_config["max_width"])


def _coverage(global_weights: dict, updates: list[dict]) -> float:
    covered = {key: np.zeros_like(np.asarray(value), dtype=bool) for key, value in global_weights.items()}
    for update in updates:
        for key, value in update.items():
            if key not in covered:
                continue
            shape = np.asarray(value).shape
            covered[key][tuple(slice(0, size) for size in shape)] = True
    denominator = sum(mask.size for mask in covered.values())
    return float(sum(mask.sum() for mask in covered.values()) / denominator) if denominator else 0.0


def _run(seed, rounds, epochs, profiles, train_frames, test_frame, host_validation_frame, task_config, model_config, method):
    import torch
    from aggregation import aggregate_partial_fedavg
    from tabular_fl import (
        ElasticTabularMLP,
        build_eval_loader_from_frame,
        load_weights,
        project_subnet_weights,
    )

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = ElasticTabularMLP(model_config)
    global_weights = {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }
    test_loader = build_eval_loader_from_frame(test_frame, task_config)
    host_validation_loader = build_eval_loader_from_frame(host_validation_frame, task_config)
    if method in {"website_budget_elastic", "website_capacity_weighted"}:
        from resource_planner import depth_for_profile, width_for_profile
        assignments = [
            (depth_for_profile(profile, model_config["max_depth"]),
             width_for_profile(profile, model_config["max_width"]))
            for profile in profiles
        ]
    else:
        assignments = [(model_config["max_depth"], model_config["max_width"])] * len(profiles)

    processes, connections, ready = _make_client_processes(
        profiles, train_frames, task_config, model_config
    )
    local_seconds, round_seconds = [[] for _ in profiles], []
    last_updates = []
    try:
        for round_id in range(rounds):
            projected = [
                project_subnet_weights(global_weights, model_config, depth, width)
                for depth, width in assignments
            ]
            round_started = time.perf_counter()
            for connection, profile, (depth, width), weights in zip(
                connections, profiles, assignments, projected
            ):
                connection.send({
                    "seed": seed * 10_000 + round_id * 100 + profile["cpu_threads"],
                    "epochs": epochs,
                    "depth": depth,
                    "width": width,
                    "weights": weights,
                })
            responses = [connection.recv() for connection in connections]
            round_seconds.append(time.perf_counter() - round_started)
            updates = [response["result"]["weights"] for response in responses]
            counts = [response["result"]["num_samples"] for response in responses]
            if method == "website_capacity_weighted":
                capacities = [response["result"]["metrics"]["active_parameters"] for response in responses]
                max_capacity = max(capacities)
                counts = [
                    count * np.sqrt(capacity / max_capacity)
                    for count, capacity in zip(counts, capacities)
                ]
            global_weights = aggregate_partial_fedavg(global_weights, updates, counts)
            for client_id, response in enumerate(responses):
                local_seconds[client_id].append(float(response["seconds"]))
            last_updates = updates
    finally:
        _close_client_processes(processes, connections)

    client_metrics = [
        _evaluate(global_weights, task_config, model_config, test_loader, depth, width)
        for depth, width in assignments
    ]
    host_validation_metrics = [
        _evaluate(global_weights, task_config, model_config, host_validation_loader, depth, width)
        for depth, width in assignments
    ]
    full_metrics = _evaluate_full(global_weights, task_config, model_config, test_loader)
    full_host_validation = _evaluate_full(global_weights, task_config, model_config, host_validation_loader)
    active_parameters = [
        int(sum(value.size for value in project_subnet_weights(
            global_weights, model_config, depth, width
        ).values()))
        for depth, width in assignments
    ]
    payload_bytes = [count * np.dtype(np.float32).itemsize for count in active_parameters]
    device_results = []
    for index, profile in enumerate(profiles):
        device_results.append({
            "profile": profile["name"],
            "logical_cpu_ids": profile["logical_cpu_ids"],
            "cpu_threads": profile["cpu_threads"],
            "dedicated_ram_gb_planner_input": profile["dedicated_ram_gb"],
            "local_rows": ready[index]["local_rows"],
            "assigned_depth": assignments[index][0],
            "assigned_width": assignments[index][1],
            "active_parameters": active_parameters[index],
            "fp32_download_bytes": payload_bytes[index],
            "mean_local_seconds_per_round": float(np.mean(local_seconds[index])),
            "test_metrics": client_metrics[index],
        })
    return {
        "seed": seed,
        "method": method,
        "rounds": rounds,
        "local_epochs": epochs,
        "wall_seconds_rounds": float(sum(round_seconds)),
        "median_round_seconds": float(np.median(round_seconds)),
        "mean_local_seconds_per_round_by_client": [
            float(np.mean(values)) for values in local_seconds
        ],
        "assigned_depth_width": [
            {"depth": depth, "width": width} for depth, width in assignments
        ],
        "active_parameters_per_client": active_parameters,
        "fp32_download_bytes_per_client": payload_bytes,
        "coordinate_coverage": _coverage(global_weights, last_updates),
        "capacity_selected_test_accuracy": float(np.mean([item["accuracy"] for item in client_metrics])),
        "capacity_selected_test_macro_f1": float(np.mean([item["macro_f1"] for item in client_metrics])),
        "per_client_test_metrics": client_metrics,
        "device_results": device_results,
        "full_global_test_metrics": full_metrics,
        "mean_client_host_validation_metrics": {
            key: float(np.mean([item[key] for item in host_validation_metrics]))
            for key in host_validation_metrics[0]
        },
        "full_global_host_validation_metrics": full_host_validation,
        "worker_ready": ready,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-width", type=int, default=64)
    parser.add_argument("--dataset", choices=("project_vital_status", "breast_cancer"), default="project_vital_status")
    parser.add_argument("--partition", choices=("dirichlet", "source_clients"), default="dirichlet")
    parser.add_argument("--capacity-weighting-ablation", action="store_true")
    parser.add_argument("--output", default=".benchmarks/tabular_web_method/project_vital_status_4client.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    physical = _current_cpu_ids()
    if len(physical) < 15:
        raise RuntimeError(f"Need at least 15 available logical CPUs for the 1/2/4/8 split; found {len(physical)}")
    profiles = _profiles(physical[:15])

    if args.dataset == "project_vital_status":
        source_files = sorted((ROOT / "split_data").glob("client_*.csv"))
        if len(source_files) != 4:
            raise FileNotFoundError("Expected four local project CSV partitions in split_data/.")
        target_column = "vital_status"
        source_frames = []
        for path in source_files:
            source_frame = pd.read_csv(path, low_memory=False)
            source_frame[target_column] = source_frame[target_column].astype(str).str.strip().str.lower()
            source_frames.append(
                source_frame[source_frame[target_column].isin({"alive", "dead"})].reset_index(drop=True)
            )
        frame = pd.concat(source_frames, ignore_index=True)
        dataset_name = "project clinical CSV partitions; vital_status alive/dead"
    else:
        if args.partition == "source_clients":
            raise ValueError("source_clients partitioning is available only for the project CSV dataset.")
        dataset = load_breast_cancer(as_frame=True)
        frame = dataset.frame.copy()
        target_column = "diagnosis"
        frame[target_column] = frame["target"].map({0: "malignant", 1: "benign"})
        frame = frame.drop(columns=["target"])
        dataset_name = "Wisconsin Diagnostic Breast Cancer (scikit-learn bundled copy)"
    if args.partition == "source_clients":
        client_frames, setup_parts, test_parts = [], [], []
        for index, source_frame in enumerate(source_frames):
            client_frame, remaining = train_test_split(
                source_frame, test_size=0.4, random_state=2026 + index,
                stratify=source_frame[target_column],
            )
            setup_part, test_part = train_test_split(
                remaining, test_size=0.5, random_state=2027 + index,
                stratify=remaining[target_column],
            )
            client_frames.append(client_frame.reset_index(drop=True))
            setup_parts.append(setup_part)
            test_parts.append(test_part)
        client_pool = pd.concat(client_frames, ignore_index=True)
        host_setup_frame = pd.concat(setup_parts, ignore_index=True)
        test_frame = pd.concat(test_parts, ignore_index=True)
    else:
        client_pool, setup_and_test = train_test_split(
            frame, test_size=0.4, random_state=2026, stratify=frame[target_column]
        )
        host_setup_frame, test_frame = train_test_split(
            setup_and_test, test_size=0.5, random_state=2027,
            stratify=setup_and_test[target_column],
        )
    from tabular_fl import fit_task_config, split_host_reference_validation
    reference_frame, host_validation_frame, inferred_labels = split_host_reference_validation(
        host_setup_frame, target_column, "classification"
    )
    task_config = fit_task_config(
        reference_frame, target_column, "classification",
        class_values=inferred_labels,
        max_depth=args.max_depth, max_width=args.max_width,
    )
    model_config = {
        "input_dim": task_config["input_dim"],
        "max_depth": task_config["max_depth"],
        "max_width": task_config["max_width"],
        "output_dim": task_config["output_dim"],
    }

    results = []
    for seed in [int(part.strip()) for part in args.seeds.split(",") if part.strip()]:
        if args.partition == "source_clients":
            train_frames = client_frames
        else:
            client_indices = _dirichlet_indices(client_pool[target_column].to_numpy(), seed)
            train_frames = [client_pool.iloc[indices].reset_index(drop=True) for indices in client_indices]
        profile_order = np.random.default_rng(seed + 90_000).permutation(len(profiles))
        seed_profiles = [profiles[int(index)] for index in profile_order]
        methods = ["fedavg_full", "website_budget_elastic"]
        if args.capacity_weighting_ablation:
            methods.append("website_capacity_weighted")
        for method in methods:
            print(f"seed={seed} method={method} rounds={args.rounds} epochs={args.epochs}", flush=True)
            started = time.perf_counter()
            result = _run(
                seed, args.rounds, args.epochs, seed_profiles, train_frames, test_frame,
                host_validation_frame, task_config, model_config, method,
            )
            result["elapsed_including_worker_startup_seconds"] = time.perf_counter() - started
            results.append(result)
            print(json.dumps({key: result[key] for key in (
                "method", "seed", "wall_seconds_rounds", "capacity_selected_test_accuracy",
                "capacity_selected_test_macro_f1",
                "full_global_test_metrics", "coordinate_coverage",
            )}, sort_keys=True), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "exact self-serve website budget-elastic tabular method",
        "dataset": dataset_name,
        "client_training_rows": int(len(client_pool)),
        "host_setup_rows": int(len(host_setup_frame)),
        "host_reference_rows": int(len(reference_frame)),
        "host_validation_rows": int(len(host_validation_frame)),
        "test_rows": int(len(test_frame)),
        "features": len(task_config["feature_columns"]),
        "classes": task_config["classes"],
        "partition_method": (
            "four source client partitions retained"
            if args.partition == "source_clients"
            else "Dirichlet alpha=0.5, four clients, seed-specific; min 30 rows per client"
        ),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus_available": len(physical),
            "ram_gib": _host_ram_gib(),
            "cuda_disabled": True,
        },
        "device_profiles": profiles,
        "model_config": model_config,
        "settings": {
            "seeds": [int(part.strip()) for part in args.seeds.split(",") if part.strip()],
            "rounds": args.rounds, "epochs": args.epochs,
            "max_depth": args.max_depth, "max_width": args.max_width,
            "capacity_weighting_ablation": args.capacity_weighting_ablation,
        },
        "limitations": [
            "Four virtual clients are separate CPU-affinity-limited processes on one host, not four physical devices.",
            "RAM values are resource-planner inputs; per-process memory limits were not imposed.",
            "Round wall time excludes website, encryption, HTTP, and network overhead.",
            "The bundled small breast-cancer dataset is a functional clinical tabular workload, not evidence for LLM-scale training.",
        ],
        "runs": results,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
