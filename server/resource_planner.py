"""Resource-aware subnet planning shared by the host and client APIs.

The browser can only expose estimates of machine capacity.  Native clients
can replace those estimates with psutil/CUDA values.  The planner keeps the
decision visible and deterministic so every participant can see why a round
was assigned a particular subnet depth.
"""

from __future__ import annotations

import math
from typing import Any

from shared.model_schema import MODEL_CONFIG


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def normalise_hardware_profile(profile: dict | None) -> dict:
    """Return a safe, UI-friendly hardware profile.

    ``available_*`` are observations, while ``dedicated_*`` are the user's
    explicit contribution limits.  Dedicated values are always clamped to
    the available values so the round planner cannot promise more than the
    client offered.
    """

    source = profile or {}
    available_ram = max(0.5, _number(source.get("available_ram_gb", source.get("ram_gb")), 4.0))
    available_cpu = max(1, int(_number(source.get("available_cpu_cores", source.get("cpu_cores")), 2)))
    default_ram = max(0.5, round(available_ram * 0.5, 2))
    default_cpu = max(1, int(math.ceil(available_cpu * 0.5)))
    dedicated_ram = min(available_ram, max(0.5, _number(source.get("dedicated_ram_gb"), default_ram)))
    dedicated_cpu = min(available_cpu, max(1, int(_number(source.get("dedicated_cpu_cores"), default_cpu))))
    local_data_size = max(0, int(_number(source.get("local_data_size"), 0)))

    return {
        "available_ram_gb": round(available_ram, 2),
        "available_cpu_cores": available_cpu,
        "dedicated_ram_gb": round(dedicated_ram, 2),
        "dedicated_cpu_cores": dedicated_cpu,
        "gpu_available": bool(source.get("gpu_available", source.get("gpu", False))),
        "local_data_size": local_data_size,
        "resource_source": source.get("resource_source", "browser estimate"),
    }


def depth_for_profile(profile: dict | None, max_depth: int | None = None) -> int:
    """Choose a subnet depth from the resources a client dedicated."""

    normalised = normalise_hardware_profile(profile)
    upper = max(2, int(max_depth or MODEL_CONFIG["max_depth"]))

    # The coefficients intentionally remain simple enough to explain in the
    # dashboard.  GPU availability helps, but never overrides a RAM/CPU cap.
    ram_layers = math.floor(normalised["dedicated_ram_gb"] / 1.5)
    cpu_layers = math.floor(normalised["dedicated_cpu_cores"] / 1.25)
    gpu_bonus = 1 if normalised["gpu_available"] else 0
    depth = max(2, min(upper, min(ram_layers + 1, cpu_layers + 1) + gpu_bonus))
    return int(depth)


def build_round_plan(project: dict, users_by_id: dict[str, dict] | None = None) -> dict:
    """Build the explainable round plan shown to host and participants."""

    users_by_id = users_by_id or {}
    max_depth = max(2, int(project.get("max_depth", MODEL_CONFIG["max_depth"])))
    profiles = project.get("client_profiles", {}) or {}
    connected = project.get("connected_clients", []) or []
    costs = []
    for layer in range(1, max_depth + 1):
        costs.append({
            "layer": layer,
            "name": "input_projection" if layer == 1 else f"backbone_{layer - 1}",
            "estimated_ram_gb": round(0.75 + (layer - 1) * 0.55, 2),
            "estimated_cpu_cores": round(0.75 + (layer - 1) * 0.35, 2),
            "estimated_flops_m": round(18 + (layer - 1) * 14, 1),
        })

    clients = []
    for user_id in connected:
        record = profiles.get(user_id, {}) or {}
        hardware = normalise_hardware_profile(record.get("hardware_profile", record))
        depth = depth_for_profile(hardware, max_depth)
        user = users_by_id.get(user_id, {})
        selected_layers = [cost["name"] for cost in costs if cost["layer"] <= depth]
        if depth >= max_depth:
            reason = "Full depth fits the dedicated RAM/CPU budget."
        elif hardware["gpu_available"]:
            reason = f"GPU-assisted budget supports layers 1–{depth}; deeper layers exceed the CPU/RAM cap."
        else:
            reason = f"Layers 1–{depth} fit the dedicated {hardware['dedicated_ram_gb']:.1f} GB / {hardware['dedicated_cpu_cores']} core budget."
        clients.append({
            "user_id": user_id,
            "username": user.get("username", user_id[:8]),
            "display_name": user.get("hospital_name", user.get("username", user_id[:8])),
            "hardware": hardware,
            "selected_depth": depth,
            "selected_layers": selected_layers,
            "reason": reason,
            "dataset_rows": int((record.get("dataset_meta", {}) or {}).get("rows", hardware.get("local_data_size", 0)) or 0),
        })

    return {
        "round": int(project.get("current_round", 0)),
        "max_rounds": int(project.get("max_rounds", 20)),
        "objective": "Select the deepest subnet that fits each client's explicitly dedicated resources.",
        "formula": "depth = clamp(min(floor(dedicated_ram/1.5)+1, floor(dedicated_cpu/1.25)+1) + GPU bonus, 2, max_depth)",
        "max_depth": max_depth,
        "layer_costs": costs,
        "clients": clients,
    }
