"""In-memory federated benchmark loop with shape-safe subnet aggregation."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from .config import DatasetSpec
from .data import DatasetBundle, make_client_loader, make_client_partitions
from .models import (
    build_fixed_subnet,
    build_model,
    extract_subnet_state,
    load_numpy_state,
    load_subnet_state,
    state_to_numpy,
    subnet_flops,
    subnet_parameters,
)
from .security import (
    PairwiseSecureAggregator,
    clip_and_noise_vector,
    conservative_gaussian_epsilon,
    run_process_isolated_secure_round,
)


@dataclass(frozen=True)
class ClientArchitecture:
    depth: int
    width: int
    # FedRolex-style rolling extraction uses a contiguous channel window.
    # Prefix methods keep offset=0, preserving the original contract.
    offset: int = 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def client_architecture_for_round(
    spec: DatasetSpec,
    client_id: int,
    round_id: int,
    policy: str = "elastic",
) -> ClientArchitecture:
    """Return a reproducible capacity assignment for one client and round."""
    if policy == "fedavg":
        return ClientArchitecture(spec.max_depth, spec.hidden_dim, 0)

    widths = tuple(spec.widths)
    candidates = [
        ClientArchitecture(1, widths[0]),
        ClientArchitecture(max(1, spec.max_depth // 2), widths[min(1, len(widths) - 1)]),
        ClientArchitecture(spec.max_depth, widths[-1]),
    ]
    # Add intermediate candidates when the dataset has a wider search space.
    if len(widths) > 2:
        candidates.insert(2, ClientArchitecture(spec.max_depth, widths[-2]))
    if policy == "static":
        return candidates[client_id % len(candidates)]
    if policy in {"elastic", "elastic_scaffold"}:
        # Rotation is the key difference from a permanently nested prefix:
        # upper layers and wider channels receive regular training coverage.
        return candidates[(client_id + round_id) % len(candidates)]
    if policy == "fedrolex":
        # FedRolex rolls a contiguous submodel over the global channels.  A
        # prefix is used when the tier fills the model; otherwise the window
        # advances deterministically and covers the full supernet over time.
        architecture = candidates[(client_id + round_id) % len(candidates)]
        max_offset = max(0, spec.hidden_dim - architecture.width)
        offset = 0 if max_offset == 0 else (round_id * architecture.width + client_id) % (max_offset + 1)
        return ClientArchitecture(architecture.depth, architecture.width, offset)
    raise ValueError(
        "policy must be 'fedavg', 'static', 'elastic', 'elastic_scaffold', or 'fedrolex'"
    )


def _state_slices(
    key: str,
    base_shape: tuple[int, ...],
    architecture: ClientArchitecture,
):
    """Return the global tensor coordinates touched by a client subnet."""
    depth, width, offset = architecture.depth, architecture.width, architecture.offset
    if key.startswith("layers.") or key.startswith("norms."):
        parts = key.split(".")
        layer_index = int(parts[1])
        if layer_index >= depth:
            return None
        if key.startswith("layers.") and key.endswith("weight") and layer_index > 0:
            return (
                slice(offset, offset + width),
                slice(offset, offset + width),
                *tuple(slice(None) for _ in base_shape[2:]),
            )
        if key.startswith("layers.") and key.endswith("weight"):
            return (
                slice(offset, offset + width),
                slice(0, base_shape[1]),
                *tuple(slice(None) for _ in base_shape[2:]),
            )
        if key.startswith("layers.") and key.endswith("bias"):
            return (slice(offset, offset + width),)
        if key.startswith("norms.") and len(base_shape) == 1:
            return (slice(offset, offset + width),)
        if len(base_shape) == 1:
            return (slice(offset, offset + width),)
        return tuple(slice(0, size) for size in base_shape)
    if key == "classifier.weight":
        return (slice(0, base_shape[0]), slice(offset, offset + width))
    if key == "classifier.bias":
        return tuple(slice(0, size) for size in base_shape)
    return None


def _update_slices(
    key: str,
    base_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    architecture: Optional[ClientArchitecture] = None,
):
    if architecture is not None:
        slices = _state_slices(key, base_shape, architecture)
        if slices is not None and tuple(
            len(range(*item.indices(limit))) for item, limit in zip(slices, base_shape)
        ) == value_shape:
            return slices
    if len(base_shape) != len(value_shape):
        return None
    if any(target > source for target, source in zip(value_shape, base_shape)):
        return None
    return tuple(slice(0, size) for size in value_shape)


def _prefix_slices(source_shape: tuple[int, ...], target_shape: tuple[int, ...]):
    if len(source_shape) != len(target_shape):
        return None
    if any(target > source for target, source in zip(target_shape, source_shape)):
        return None
    return tuple(slice(0, size) for size in target_shape)


def _zero_float_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create a server control-variate state for trainable floating tensors."""
    return {
        key: np.zeros_like(value, dtype=np.float32)
        for key, value in state.items()
        if np.asarray(value).dtype.kind == "f"
    }


def _control_slice(
    control: dict[str, np.ndarray],
    key: str,
    shape: tuple[int, ...],
    architecture: Optional[ClientArchitecture] = None,
) -> np.ndarray:
    source = control.get(key)
    if source is None:
        return np.zeros(shape, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    slices = _update_slices(key, tuple(source.shape), shape, architecture)
    if slices is None:
        return np.zeros(shape, dtype=np.float32)
    return source[slices]


def _apply_control_deltas(
    server_control: dict[str, np.ndarray],
    deltas: list[dict[str, np.ndarray]],
    sample_counts: list[int],
    architectures: Optional[list[ClientArchitecture]] = None,
) -> dict[str, np.ndarray]:
    """Average partial control-variate deltas into the full server state."""
    result = {key: np.asarray(value).copy() for key, value in server_control.items()}
    for key, base_value in result.items():
        base = np.asarray(base_value)
        numerator = np.zeros_like(base, dtype=np.float64)
        denominator = np.zeros_like(base, dtype=np.float64)
        for index, (delta, samples) in enumerate(zip(deltas, sample_counts)):
            if key not in delta:
                continue
            value = np.asarray(delta[key], dtype=np.float32)
            architecture = architectures[index] if architectures is not None else None
            slices = _update_slices(key, tuple(base.shape), tuple(value.shape), architecture)
            if slices is None:
                continue
            weight = float(max(samples, 1))
            numerator[slices] += value.astype(np.float64) * weight
            denominator[slices] += weight
        mask = denominator > 0
        updated = base.astype(np.float64, copy=True)
        updated[mask] += numerator[mask] / denominator[mask]
        result[key] = updated.astype(np.float32)
    return result


def _state_payload_bytes(
    global_state: dict[str, np.ndarray],
    architecture: ClientArchitecture,
) -> int:
    """Approximate bytes sent to one client for a prefix subnet."""
    total = 0
    for key, value in global_state.items():
        array = np.asarray(value)
        if key.startswith("layers.") or key.startswith("norms."):
            parts = key.split(".")
            layer_index = int(parts[1])
            if layer_index >= architecture.depth:
                continue
            if key.startswith("layers.") and key.endswith("weight"):
                shape = (
                    architecture.width,
                    array.shape[1] if layer_index == 0 else architecture.width,
                )
            elif array.ndim == 1:
                shape = (architecture.width,)
            else:
                shape = array.shape
            total += int(np.prod(shape)) * array.dtype.itemsize
        elif key == "classifier.weight":
            total += int(np.prod((array.shape[0], architecture.width))) * array.dtype.itemsize
        elif key == "classifier.bias":
            total += int(array.size) * array.dtype.itemsize
    return int(total)


def _float_state_keys(state: dict[str, np.ndarray]) -> list[str]:
    return [key for key, value in state.items() if np.asarray(value).dtype.kind == "f"]


def _dense_delta_vector(
    update: dict[str, np.ndarray],
    global_state: dict[str, np.ndarray],
    keys: Optional[list[str]] = None,
    architecture: Optional[ClientArchitecture] = None,
) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], tuple[slice, ...] | tuple]]]:
    """Pack a possibly partial update into a deterministic dense delta vector."""
    keys = keys or _float_state_keys(global_state)
    parts = []
    layout = []
    for key in keys:
        base = np.asarray(global_state[key], dtype=np.float32)
        if key not in update:
            value = np.zeros_like(base)
            slices = tuple(slice(0, size) for size in base.shape) if base.ndim else ()
            layout.append((key, tuple(base.shape), slices))
            parts.append(value.reshape(-1))
            continue
        value = np.asarray(update[key], dtype=np.float32)
        slices = _update_slices(key, tuple(base.shape), tuple(value.shape), architecture)
        if slices is None:
            slices = tuple(slice(0, size) for size in base.shape) if base.ndim else ()
            value = np.zeros_like(base)
        dense = np.zeros_like(base)
        if base.ndim == 0:
            dense = value.reshape(()) - base.reshape(())
        else:
            dense[slices] = value - base[slices]
        layout.append((key, tuple(base.shape), slices))
        parts.append(dense.reshape(-1))
    vector = np.concatenate(parts).astype(np.float64, copy=False) if parts else np.zeros(0, dtype=np.float64)
    return vector, layout


def _apply_dense_delta_vector(
    update: dict[str, np.ndarray],
    global_state: dict[str, np.ndarray],
    vector: np.ndarray,
    keys: Optional[list[str]] = None,
    architecture: Optional[ClientArchitecture] = None,
) -> dict[str, np.ndarray]:
    """Apply a dense delta to only the coordinates present in ``update``."""
    keys = keys or _float_state_keys(global_state)
    result = {key: np.asarray(value).copy() for key, value in update.items()}
    offset = 0
    for key in keys:
        base = np.asarray(global_state[key], dtype=np.float32)
        size = int(base.size) if base.ndim else 1
        dense_delta = np.asarray(vector[offset : offset + size], dtype=np.float32).reshape(base.shape)
        offset += size
        if key not in update:
            continue
        value = np.asarray(update[key], dtype=np.float32)
        slices = _update_slices(key, tuple(base.shape), tuple(value.shape), architecture)
        if slices is None:
            continue
        result[key] = (base[slices] + dense_delta[slices]).astype(np.float32)
    return result


def _global_from_dense_delta(
    global_state: dict[str, np.ndarray],
    vector: np.ndarray,
    keys: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """Apply a dense full-model delta while preserving non-floating state."""
    keys = keys or _float_state_keys(global_state)
    result = {key: np.asarray(value).copy() for key, value in global_state.items()}
    offset = 0
    for key in keys:
        base = np.asarray(global_state[key], dtype=np.float32)
        size = int(base.size) if base.ndim else 1
        delta = np.asarray(vector[offset : offset + size], dtype=np.float32).reshape(base.shape)
        offset += size
        result[key] = (base + delta).astype(np.asarray(global_state[key]).dtype, copy=False)
    if offset != len(vector):
        raise ValueError("dense delta has the wrong dimension")
    return result


def _sanitize_update(
    update: dict[str, np.ndarray],
    global_state: dict[str, np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    architecture: Optional[ClientArchitecture] = None,
) -> tuple[dict[str, np.ndarray], float]:
    keys = _float_state_keys(global_state)
    vector, _ = _dense_delta_vector(update, global_state, keys, architecture)
    sanitized, clipped_norm = clip_and_noise_vector(
        vector, clip_norm=clip_norm, noise_multiplier=noise_multiplier, rng=rng
    )
    return _apply_dense_delta_vector(update, global_state, sanitized, keys, architecture), clipped_norm


def _poison_update(
    update: dict[str, np.ndarray],
    global_state: dict[str, np.ndarray],
    scale: float = 5.0,
    architecture: Optional[ClientArchitecture] = None,
) -> dict[str, np.ndarray]:
    """Sign-flip/model-replacement attack applied to a local model state."""
    poisoned = {}
    for key, value in update.items():
        if key not in global_state:
            poisoned[key] = np.asarray(value).copy()
            continue
        base = np.asarray(global_state[key])
        value = np.asarray(value)
        slices = _update_slices(key, tuple(base.shape), tuple(value.shape), architecture)
        if slices is None:
            poisoned[key] = value.copy()
            continue
        poisoned[key] = base[slices] - float(scale) * (value - base[slices])
    return poisoned


def _flame_keep_indices(
    updates: list[dict[str, np.ndarray]],
    global_state: dict[str, np.ndarray],
    architectures: Optional[list[ClientArchitecture]] = None,
    threshold: float = 0.6,
) -> list[int]:
    """Largest-cosine-cluster filter, operating on dense client deltas."""
    if len(updates) < 2:
        return list(range(len(updates)))
    keys = _float_state_keys(global_state)
    matrix = np.stack([
        _dense_delta_vector(
            update,
            global_state,
            keys,
            architectures[index] if architectures is not None else None,
        )[0]
        for index, update in enumerate(updates)
    ])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    clusters = [{index} for index in range(len(updates))]
    changed = True
    while changed:
        changed = False
        for left in list(clusters):
            for right in list(clusters):
                if left is right:
                    continue
                if max(similarity[i, j] for i in left for j in right) >= threshold:
                    left.update(right)
                    clusters.remove(right)
                    changed = True
                    break
            if changed:
                break
    largest = max(clusters, key=len)
    # A highly non-IID round can produce several small honest clusters.  A
    # pure largest-cluster rule would then discard nearly every client and
    # mistake heterogeneity for malice.  Retain at least half the round and
    # fill from the clients most similar to the selected cluster.
    minimum_keep = max(1, int(math.ceil(len(updates) / 2)))
    if len(largest) < minimum_keep:
        remaining = [index for index in range(len(updates)) if index not in largest]
        remaining.sort(
            key=lambda index: max(float(similarity[index, member]) for member in largest),
            reverse=True,
        )
        largest.update(remaining[: minimum_keep - len(largest)])
    return sorted(largest)


def _trigger_inputs(X: np.ndarray) -> np.ndarray:
    triggered = np.asarray(X, dtype=np.float32).copy()
    triggered[:, : min(3, triggered.shape[1])] = 5.0
    return triggered


def aggregate_subnet_updates(
    global_state: dict[str, np.ndarray],
    updates: list[dict[str, np.ndarray]],
    sample_counts: list[int],
    priorities: Optional[list[float]] = None,
    defense: str = "none",
    trim_ratio: float = 0.1,
    architectures: Optional[list[ClientArchitecture]] = None,
) -> dict[str, np.ndarray]:
    """Average arbitrary nested/rolling subnet slices into a full state.

    Unlike the production aggregator, this function explicitly handles
    partial tensor shapes and FedRolex-style channel offsets. Coordinates not
    touched in a round retain their prior global values, while each updated
    coordinate is normalized by the weight of the clients that contributed
    that coordinate.
    """
    if not updates or len(updates) != len(sample_counts):
        raise ValueError("updates and sample_counts must be non-empty and aligned")
    if priorities is None:
        priorities = [1.0] * len(updates)
    if len(priorities) != len(updates):
        raise ValueError("priorities and updates must be aligned")
    if architectures is not None and len(architectures) != len(updates):
        raise ValueError("architectures and updates must be aligned")
    if defense not in {"none", "trimmed_mean"}:
        raise ValueError("defense must be 'none' or 'trimmed_mean'")
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")

    result = {}
    for key, base_value in global_state.items():
        base = np.asarray(base_value)
        numerator = np.zeros_like(base, dtype=np.float64)
        denominator = np.zeros_like(base, dtype=np.float64)
        contributions = []
        for index, (update, samples, priority) in enumerate(zip(updates, sample_counts, priorities)):
            if key not in update:
                continue
            value = np.asarray(update[key])
            if value.ndim != base.ndim:
                if value.ndim == 0 and base.ndim == 0:
                    slices = ()
                else:
                    continue
            elif value.ndim == 0:
                slices = ()
            elif all(size <= limit for size, limit in zip(value.shape, base.shape)):
                slices = _update_slices(
                    key,
                    tuple(base.shape),
                    tuple(value.shape),
                    architectures[index] if architectures is not None else None,
                )
            else:
                continue
            if slices is None:
                continue
            weight = float(max(samples, 1)) * float(priority)
            contributions.append((slices, value, weight))
            numerator[slices] += value.astype(np.float64) * weight
            denominator[slices] += weight
        merged = base.astype(np.float64, copy=True)
        if defense == "trimmed_mean" and len(contributions) >= 4:
            # Coordinate-wise trimming is applied only when the clients
            # contributed the same tensor slice.  Partial subnet slices fall
            # back to the shape-safe weighted mean below.
            first_slices = contributions[0][0]
            first_shape = contributions[0][1].shape
            if all(item[0] == first_slices and item[1].shape == first_shape for item in contributions):
                values = np.stack([item[1].astype(np.float64) for item in contributions], axis=0)
                trim = min(
                    max(1, int(len(values) * trim_ratio)) if trim_ratio > 0 else 0,
                    (len(values) - 1) // 2,
                )
                if trim > 0:
                    values = np.sort(values, axis=0)[trim : len(values) - trim]
                merged[first_slices] = np.mean(values, axis=0)
                result[key] = merged.astype(base.dtype, copy=False)
                continue
        mask = denominator > 0
        merged[mask] = numerator[mask] / denominator[mask]
        result[key] = merged.astype(base.dtype, copy=False)
    return result


def _proximal_penalty(model, snapshot, mu: float):
    if mu <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for local_param, global_param in zip(model.parameters(), snapshot):
        penalty = penalty + (local_param - global_param).pow(2).sum()
    return 0.5 * mu * penalty


def _train_client(
    model,
    loader: DataLoader,
    architecture: ClientArchitecture,
    spec: DatasetSpec,
    global_state: Optional[dict[str, np.ndarray]] = None,
    server_control: Optional[dict[str, np.ndarray]] = None,
    client_control: Optional[dict[str, np.ndarray]] = None,
    drift_correction: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    poison_mode: str = "none",
    num_classes: int = 2,
    proximal_mu: float = 0.0,
    ordered_dropout: bool = False,
    ordered_widths: Optional[tuple[int, ...]] = None,
    drift_lr_multiplier: float = 5.0,
):
    snapshot = [param.detach().clone() for param in model.parameters()]
    # SCAFFOLD's control-variate update is derived for additive SGD steps.
    # Keep the original Adam optimizer for the baseline/elastic strategies,
    # but use a conservative SGD step when the correction is enabled so the
    # client-control estimate remains numerically meaningful.
    local_lr = spec.learning_rate * (
        float(drift_lr_multiplier) if drift_correction else 1.0
    )
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=local_lr, momentum=0.9)
        if drift_correction
        else torch.optim.Adam(model.parameters(), lr=local_lr)
    )
    model.train()
    last_loss = 0.0
    steps = 0
    if poison_mode not in {"none", "label_flip", "backdoor"}:
        raise ValueError("poison_mode must be 'none', 'label_flip', or 'backdoor'")
    for _ in range(spec.local_epochs):
        for X, y in loader:
            if poison_mode == "label_flip":
                y = (num_classes - 1 - y).clamp_min(0)
            elif poison_mode == "backdoor":
                X = X.clone()
                X[:, : min(3, X.shape[1])] = 5.0
                y = torch.zeros_like(y)
            optimizer.zero_grad()
            active_width = architecture.width
            if ordered_dropout:
                choices = tuple(
                    value for value in (ordered_widths or (architecture.width,))
                    if 1 <= value <= architecture.width
                )
                active_width = int(random.choice(choices))
            teacher_logits = None
            if hasattr(model, "hidden_dim") and model.max_depth == architecture.depth:
                # FixedElasticMLP is the actual edge-device path.  Its
                # optional active width implements FjORD's ordered dropout
                # without allocating a second (larger) client model.
                try:
                    if ordered_dropout and active_width < architecture.width:
                        with torch.no_grad():
                            teacher_logits = model(X, architecture.depth, architecture.width)
                    logits = model(X, architecture.depth, active_width)
                except TypeError:
                    logits = model(X)
            else:
                logits = model(X, architecture.depth, architecture.width)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            if ordered_dropout and teacher_logits is not None:
                temperature = 2.0
                distillation = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss = 0.9 * loss + 0.1 * distillation
            loss = loss + _proximal_penalty(model, snapshot, proximal_mu)
            loss.backward()

            if drift_correction and global_state is not None:
                server_control = server_control or {}
                client_control = client_control or {}
                # SCAFFOLD-style correction: add c - c_i to each local
                # gradient.  Only the active subnet tensors are touched.
                for key, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    server_slice = _control_slice(
                        server_control, key, tuple(parameter.shape), architecture
                    )
                    client_slice = _control_slice(
                        client_control, key, tuple(parameter.shape), architecture
                    )
                    correction = torch.as_tensor(
                        server_slice - client_slice,
                        dtype=parameter.grad.dtype,
                        device=parameter.grad.device,
                    )
                    parameter.grad.add_(correction)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach().item())
            steps += 1

    subnet_state = extract_subnet_state(
        model, architecture.depth, architecture.width, architecture.offset
    )
    if not drift_correction or global_state is None:
        return subnet_state, last_loss, {}, {}

    # Update each participating client's control variate using the standard
    # SCAFFOLD correction, but export only the active prefix to the server.
    old_control = client_control or {}
    new_control = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in old_control.items()
    }
    control_delta = {}
    denom = float(max(steps, 1) * max(local_lr, 1e-12))
    for key, local_value in subnet_state.items():
        if key not in global_state or np.asarray(global_state[key]).dtype.kind != "f":
            continue
        local_array = np.asarray(local_value, dtype=np.float32)
        global_array = np.asarray(global_state[key], dtype=np.float32)
        slices = _update_slices(
            key,
            tuple(global_array.shape),
            tuple(local_array.shape),
            architecture,
        )
        if slices is None:
            continue
        old_full = np.asarray(
            old_control.get(key, np.zeros_like(global_array, dtype=np.float32)),
            dtype=np.float32,
        )
        old_slice = old_full[slices]
        server_slice = _control_slice(
            server_control or {}, key, tuple(local_array.shape), architecture
        )
        new_slice = old_slice - server_slice + (global_array[slices] - local_array) / denom
        updated_full = old_full.copy()
        updated_full[slices] = new_slice
        new_control[key] = updated_full
        control_delta[key] = new_slice - old_slice

    return subnet_state, last_loss, new_control, control_delta


def _evaluate(model, X: np.ndarray, y: np.ndarray, architecture: ClientArchitecture, batch_size: int):
    model.eval()
    # Evaluation is inference-only and these benchmark splits fit comfortably
    # in CPU memory.  A single tensor pass removes the repeated DataLoader
    # setup that otherwise dominates the many-tier confidence-interval runs.
    inputs = torch.from_numpy(np.asarray(X))
    with torch.no_grad():
        predictions = model(inputs, architecture.depth, architecture.width).argmax(dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
    }


def evaluate_candidates(
    model,
    bundle: DatasetBundle,
    spec: DatasetSpec,
    include_test: bool = True,
):
    scores = []
    for depth in range(1, spec.max_depth + 1):
        for width in spec.widths:
            if width > spec.hidden_dim:
                continue
            architecture = ClientArchitecture(depth, width)
            metrics = _evaluate(model, bundle.X_val, bundle.y_val, architecture, spec.batch_size)
            test_metrics = (
                _evaluate(model, bundle.X_test, bundle.y_test, architecture, spec.batch_size)
                if include_test
                else {"accuracy": None, "macro_f1": None}
            )
            scores.append(
                {
                    "depth": depth,
                    "width": width,
                    "parameters": subnet_parameters(bundle.input_dim, spec.num_classes, depth, width),
                    "flops": subnet_flops(bundle.input_dim, spec.num_classes, depth, width),
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                }
            )
    return scores


def select_best_candidate(
    scores: list[dict],
    tolerance: float = 0.002,
    metric: str = "val_accuracy",
) -> dict:
    """Select the smallest subnet within tolerance of the best validation metric."""
    if not scores:
        raise ValueError("scores must not be empty")
    if metric not in scores[0]:
        raise ValueError(f"unknown candidate metric: {metric}")
    best_metric = max(item[metric] for item in scores)
    eligible = [item for item in scores if item[metric] >= best_metric - tolerance]
    return min(eligible, key=lambda item: (item["flops"], item["parameters"]))


def run_federated_experiment(
    bundle: DatasetBundle,
    spec: DatasetSpec,
    variant: str = "elastic",
    rounds: int = 3,
    seed: int = 42,
    num_clients: Optional[int] = None,
    strategy: str = "elastic",
    drift_correction: Optional[bool] = None,
    dropout_rate: float = 0.0,
    straggler_rate: float = 0.0,
    secure_aggregation: bool = False,
    secure_aggregation_isolated: bool = False,
    dp_noise_multiplier: float = 0.0,
    dp_clip_norm: float = 1.0,
    dp_delta: float = 1e-5,
    attack_fraction: float = 0.0,
    attack_type: str = "none",
    attack_scale: float = 5.0,
    defense: str = "none",
    trim_ratio: float = 0.1,
) -> dict:
    """Run a reproducible FL experiment and return paper-ready metrics.

    Strategies are intentionally easy to compare:

    ``fedavg``
        Every client trains the full model.  This is the accuracy/compute
        reference point.
    ``fedprox``
        Full-model FedProx, with the proximal coefficient from the dataset
        specification.
    ``heterofl``
        HeteroFL-style fixed nested prefixes assigned by client tier.
    ``fjord``
        FjORD-style ordered-dropout prefixes with local self-distillation.
    ``fedrolex``
        FedRoleX-style rolling contiguous submodel extraction.
    ``maxnet``
        SuperFedNAS/MaxNet-style coverage weighting without SCAFFOLD.
    ``elastic``
        Rotating capacities plus MaxNet-inspired coverage weighting.
    ``elastic_scaffold``
        ``elastic`` plus a subnet-aware SCAFFOLD control-variate correction.
    ``cc_efl``
        The integrated method: rolling extraction, MaxNet coverage weighting,
        and FedProx regularization. SCAFFOLD remains a separate reference
        because its correction is not stable with this rolling fixed-subnet
        backbone.

    ``dropout_rate`` and ``straggler_rate`` simulate clients missing a round
    deadline.  They affect only participation, not the local data itself, so
    the experiment remains deterministic and inexpensive.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    allowed_strategies = {
        "fedavg", "fedprox", "static", "heterofl", "fjord", "fedrolex",
        "elastic", "maxnet", "scaffold", "elastic_scaffold", "cc_efl",
    }
    if strategy not in allowed_strategies:
        raise ValueError("unknown strategy")
    if not 0.0 <= dropout_rate <= 1.0:
        raise ValueError("dropout_rate must be in [0, 1]")
    if not 0.0 <= straggler_rate <= 1.0:
        raise ValueError("straggler_rate must be in [0, 1]")
    if secure_aggregation and strategy not in {"fedavg", "fedprox", "scaffold"}:
        raise ValueError("the audited secure-aggregation path currently requires a full-model strategy")
    if secure_aggregation and (dropout_rate > 0 or straggler_rate > 0):
        raise ValueError("secure aggregation requires zero dropout/straggler rates until recovery is implemented")
    if secure_aggregation_isolated and not secure_aggregation:
        raise ValueError("secure_aggregation_isolated requires secure_aggregation=True")
    if dp_noise_multiplier < 0:
        raise ValueError("dp_noise_multiplier must be non-negative")
    if dp_clip_norm <= 0:
        raise ValueError("dp_clip_norm must be positive")
    if not 0.0 < dp_delta < 1.0:
        raise ValueError("dp_delta must be in (0, 1)")
    if not 0.0 <= attack_fraction <= 1.0:
        raise ValueError("attack_fraction must be in [0, 1]")
    if attack_type not in {"none", "sign_flip", "label_flip", "backdoor"}:
        raise ValueError("unknown attack_type")
    if attack_scale <= 0:
        raise ValueError("attack_scale must be positive")
    if defense not in {"none", "trimmed_mean", "flame"}:
        raise ValueError("unknown defense")
    if secure_aggregation and (attack_type != "none" or defense != "none"):
        raise ValueError("secure aggregation is benchmarked separately from attack defenses")
    if defense == "flame" and secure_aggregation:
        raise ValueError("FLAME-style filtering requires visible client updates")
    _seed_everything(seed)
    started_at = time.perf_counter()
    clients = num_clients or spec.clients
    attacker_start = max(0, clients - int(math.ceil(clients * attack_fraction)))
    attacker_ids = set(range(attacker_start, clients))
    partitions = make_client_partitions(
        bundle.y_train, clients, alpha=spec.dirichlet_alpha, seed=seed
    )
    model = build_model(variant, bundle.input_dim, spec.num_classes, spec.max_depth, spec.hidden_dim)
    global_state = state_to_numpy(model)
    server_control = _zero_float_state(global_state)
    client_controls: dict[int, dict[str, np.ndarray]] = {}
    history = []
    availability_rng = np.random.default_rng(seed + 1729)
    use_drift_correction = (
        strategy in {"scaffold", "elastic_scaffold"}
        if drift_correction is None else bool(drift_correction)
    )
    if strategy in {"fedavg", "fedprox", "scaffold"}:
        policy = "fedavg"
    elif strategy in {"static", "heterofl", "fjord"}:
        policy = "static"
    elif strategy in {"fedrolex", "cc_efl"}:
        policy = "fedrolex"
    else:
        policy = "elastic"
    proximal_mu = spec.fedprox_mu if strategy in {"fedprox", "cc_efl"} else 0.0
    ordered_dropout = strategy == "fjord"
    secure_keys = _float_state_keys(global_state)
    class_weights = None
    if spec.class_weighting == "balanced":
        class_counts = np.bincount(bundle.y_train, minlength=spec.num_classes).astype(np.float32)
        class_weights = len(bundle.y_train) / np.maximum(class_counts * spec.num_classes, 1.0)
        class_weights = torch.from_numpy(class_weights.astype(np.float32))

    for round_id in range(rounds):
        updates = []
        counts = []
        architectures = []
        losses = []
        control_deltas = []
        dropped_clients = []
        stragglers = []
        upload_bytes = []
        download_bytes = []
        local_parameter_counts = []
        active_client_ids = []
        clipped_norms = []
        for client_id, indices in enumerate(partitions):
            architecture = client_architecture_for_round(spec, client_id, round_id, policy=policy)
            if client_id != 0 and availability_rng.random() < dropout_rate:
                dropped_clients.append(client_id)
                continue
            if client_id != 0 and availability_rng.random() < straggler_rate:
                stragglers.append(client_id)
                continue

            if variant == "elastic":
                # This is the deployable path: the client allocates exactly
                # the requested subnet, never the full supernet.
                local_model = build_fixed_subnet(
                    variant,
                    bundle.input_dim,
                    spec.num_classes,
                    architecture.depth,
                    architecture.width,
                    offset=architecture.offset,
                )
                load_subnet_state(local_model, global_state)
            else:
                local_model = build_model(
                    variant, bundle.input_dim, spec.num_classes, spec.max_depth, spec.hidden_dim
                )
                load_numpy_state(local_model, global_state)

            old_control = client_controls.get(client_id, {})
            poison_mode = (
                attack_type if client_id in attacker_ids and attack_type in {"label_flip", "backdoor"}
                else "none"
            )
            update, loss, new_control, control_delta = _train_client(
                local_model,
                make_client_loader(bundle, indices, spec.batch_size),
                architecture,
                spec,
                global_state=global_state,
                server_control=server_control,
                client_control=old_control,
                drift_correction=use_drift_correction,
                class_weights=class_weights,
                poison_mode=poison_mode,
                num_classes=spec.num_classes,
                proximal_mu=proximal_mu,
                ordered_dropout=ordered_dropout,
                ordered_widths=tuple(spec.widths),
                drift_lr_multiplier=1.0 if strategy == "cc_efl" else 5.0,
            )
            if dp_noise_multiplier > 0:
                update, clipped_norm = _sanitize_update(
                    update,
                    global_state,
                    clip_norm=dp_clip_norm,
                    noise_multiplier=dp_noise_multiplier,
                    rng=np.random.default_rng(seed + 100_003 * (round_id + 1) + client_id),
                    architecture=architecture,
                )
                clipped_norms.append(clipped_norm)
            if attack_type == "sign_flip" and client_id in attacker_ids:
                update = _poison_update(
                    update, global_state, scale=attack_scale, architecture=architecture
                )
            updates.append(update)
            counts.append(len(indices))
            architectures.append(architecture)
            losses.append(loss)
            control_deltas.append(control_delta)
            active_client_ids.append(client_id)
            client_controls[client_id] = new_control
            upload_bytes.append(sum(np.asarray(value).nbytes for value in update.values()))
            download_bytes.append(_state_payload_bytes(global_state, architecture))
            if variant == "elastic":
                local_parameter_counts.append(
                    subnet_parameters(bundle.input_dim, spec.num_classes, architecture.depth, architecture.width)
                )
            else:
                local_parameter_counts.append(sum(parameter.numel() for parameter in local_model.parameters()))

        # Avoid an empty round in an extreme dropout simulation.  This keeps
        # the experiment useful while recording that the deadline policy was
        # too aggressive for that round.
        if not updates:
            client_id = 0
            architecture = client_architecture_for_round(spec, client_id, round_id, policy=policy)
            if variant == "elastic":
                local_model = build_fixed_subnet(
                    variant, bundle.input_dim, spec.num_classes,
                    architecture.depth, architecture.width, offset=architecture.offset
                )
                load_subnet_state(local_model, global_state)
            else:
                local_model = build_model(
                    variant, bundle.input_dim, spec.num_classes, spec.max_depth, spec.hidden_dim
                )
                load_numpy_state(local_model, global_state)
            update, loss, new_control, control_delta = _train_client(
                local_model,
                make_client_loader(bundle, partitions[0], spec.batch_size),
                architecture,
                spec,
                global_state=global_state,
                server_control=server_control,
                client_control=client_controls.get(0, {}),
                drift_correction=use_drift_correction,
                class_weights=class_weights,
                poison_mode=(
                    attack_type if 0 in attacker_ids and attack_type in {"label_flip", "backdoor"}
                    else "none"
                ),
                num_classes=spec.num_classes,
                proximal_mu=proximal_mu,
                ordered_dropout=ordered_dropout,
                ordered_widths=tuple(spec.widths),
                drift_lr_multiplier=1.0 if strategy == "cc_efl" else 5.0,
            )
            if dp_noise_multiplier > 0:
                update, clipped_norm = _sanitize_update(
                    update,
                    global_state,
                    clip_norm=dp_clip_norm,
                    noise_multiplier=dp_noise_multiplier,
                    rng=np.random.default_rng(seed + 100_003 * (round_id + 1)),
                    architecture=architecture,
                )
                clipped_norms.append(clipped_norm)
            if attack_type == "sign_flip" and 0 in attacker_ids:
                update = _poison_update(
                    update, global_state, scale=attack_scale, architecture=architecture
                )
            updates.append(update)
            counts.append(len(partitions[0]))
            architectures.append(architecture)
            losses.append(loss)
            control_deltas.append(control_delta)
            active_client_ids.append(0)
            client_controls[0] = new_control
            upload_bytes.append(sum(np.asarray(value).nbytes for value in update.values()))
            download_bytes.append(_state_payload_bytes(global_state, architecture))
            local_parameter_counts.append(
                subnet_parameters(bundle.input_dim, spec.num_classes, architecture.depth, architecture.width)
                if variant == "elastic"
                else sum(parameter.numel() for parameter in local_model.parameters())
            )

        defense_removed = 0
        if defense == "flame":
            keep_indices = _flame_keep_indices(updates, global_state, architectures)
            if not keep_indices:
                keep_indices = [int(np.argmax(counts))]
            defense_removed = len(updates) - len(keep_indices)
            updates = [updates[index] for index in keep_indices]
            counts = [counts[index] for index in keep_indices]
            architectures = [architectures[index] for index in keep_indices]
            losses = [losses[index] for index in keep_indices]
            control_deltas = [control_deltas[index] for index in keep_indices]
            upload_bytes = [upload_bytes[index] for index in keep_indices]
            download_bytes = [download_bytes[index] for index in keep_indices]
            local_parameter_counts = [local_parameter_counts[index] for index in keep_indices]
            active_client_ids = [active_client_ids[index] for index in keep_indices]
            if clipped_norms:
                clipped_norms = [clipped_norms[index] for index in keep_indices]

        # MaxNet-inspired priority: give the largest active subnet more weight
        # early, then reduce the bias toward it with a cosine schedule so small
        # subnets are not starved later in training.  FedAvg/static are kept
        # unweighted so they remain clean baselines.
        beta = 0.2 + 0.7 * 0.5 * (1.0 + math.cos(math.pi * round_id / max(rounds - 1, 1)))
        max_capacity = max(item.depth * item.width for item in architectures)
        if strategy in {"elastic", "maxnet", "elastic_scaffold", "cc_efl"}:
            priorities = [
                beta if item.depth * item.width == max_capacity else (1.0 - beta)
                for item in architectures
            ]
        else:
            priorities = [1.0] * len(architectures)
        secure_report = None
        unprotected_upload_bytes = int(np.sum(upload_bytes)) if upload_bytes else 0
        if secure_aggregation:
            # The audited path uses full-model FedAvg updates, so the secure
            # sum has one well-defined denominator and no unmasked coordinate
            # metadata.  Dropout recovery is intentionally not hidden here:
            # the simulator requires the announced participant set to finish.
            secure_vectors = [
                _dense_delta_vector(update, global_state, secure_keys, architecture)[0] * float(samples)
                for update, samples, architecture in zip(updates, counts, architectures)
            ]
            if secure_aggregation_isolated:
                summed_delta, secure_report = run_process_isolated_secure_round(
                    dict(zip(active_client_ids, secure_vectors)), seed + round_id
                )
            else:
                secure_aggregator = PairwiseSecureAggregator(
                    active_client_ids,
                    vector_dim=len(secure_vectors[0]) if secure_vectors else 1,
                    round_seed=seed + round_id,
                )
                masked = {
                    client_id: secure_aggregator.mask(client_id, vector)
                    for client_id, vector in zip(active_client_ids, secure_vectors)
                }
                summed_delta = secure_aggregator.aggregate(masked)
                secure_report = secure_aggregator.report()
            global_state = _global_from_dense_delta(
                global_state,
                summed_delta / float(max(sum(counts), 1)),
                secure_keys,
            )
        else:
            global_state = aggregate_subnet_updates(
                global_state,
                updates,
                counts,
                priorities,
                defense="trimmed_mean" if defense == "trimmed_mean" else "none",
                trim_ratio=trim_ratio,
                architectures=architectures,
            )
        if use_drift_correction:
            server_control = _apply_control_deltas(
                server_control, control_deltas, counts, architectures
            )
        load_numpy_state(model, global_state)

        candidate_scores = evaluate_candidates(model, bundle, spec, include_test=False)
        best = select_best_candidate(candidate_scores, metric=spec.selection_metric)
        selected_test = _evaluate(
            model,
            bundle.X_test,
            bundle.y_test,
            ClientArchitecture(best["depth"], best["width"]),
            spec.batch_size,
        )
        best = dict(best)
        best["test_accuracy"] = selected_test["accuracy"]
        best["test_macro_f1"] = selected_test["macro_f1"]
        backdoor_asr = None
        if attack_type == "backdoor":
            triggered_test = _evaluate(
                model,
                _trigger_inputs(bundle.X_test),
                np.zeros(len(bundle.y_test), dtype=np.int64),
                ClientArchitecture(best["depth"], best["width"]),
                spec.batch_size,
            )
            backdoor_asr = triggered_test["accuracy"]
        active_flops = [
            subnet_flops(bundle.input_dim, spec.num_classes, item.depth, item.width)
            for item in architectures
        ]
        history.append(
            {
                "round": round_id + 1,
                "mean_client_loss": float(np.mean(losses)) if losses else 0.0,
                "maxnet_beta": float(beta),
                "strategy": strategy,
                "drift_correction": use_drift_correction,
                "selection_metric": spec.selection_metric,
                "class_weighting": spec.class_weighting,
                "participating_clients": len(updates),
                "dropped_clients": len(dropped_clients),
                "stragglers": len(stragglers),
                "participation_rate": float(len(updates) / max(clients, 1)),
                "mean_upload_bytes": float(
                    (secure_report.total_bytes / max(len(updates), 1))
                    if secure_report is not None
                    else (np.mean(upload_bytes) if upload_bytes else 0.0)
                ),
                "mean_download_bytes": float(np.mean(download_bytes)) if download_bytes else 0.0,
                "total_upload_bytes": int(
                    secure_report.total_bytes if secure_report is not None else unprotected_upload_bytes
                ),
                "unprotected_upload_bytes": unprotected_upload_bytes,
                "total_download_bytes": int(np.sum(download_bytes)) if download_bytes else 0,
                "secure_aggregation_enabled": bool(secure_aggregation),
                "secure_aggregation_payload_bytes": (
                    secure_report.payload_bytes if secure_report is not None else 0
                ),
                "secure_aggregation_key_exchange_bytes": (
                    secure_report.key_exchange_bytes if secure_report is not None else 0
                ),
                "secure_aggregation_total_bytes": (
                    secure_report.total_bytes if secure_report is not None else 0
                ),
                "dp_enabled": bool(dp_noise_multiplier > 0),
                "dp_noise_multiplier": float(dp_noise_multiplier),
                "dp_clip_norm": float(dp_clip_norm),
                "dp_delta": float(dp_delta),
                "dp_epsilon_upper_bound": (
                    conservative_gaussian_epsilon(dp_noise_multiplier, dp_delta, round_id + 1)
                    if dp_noise_multiplier > 0 else None
                ),
                "mean_clipped_update_norm": float(np.mean(clipped_norms)) if clipped_norms else None,
                "attack_type": attack_type,
                "malicious_clients": len(attacker_ids),
                "active_malicious_clients": sum(client_id in attacker_ids for client_id in active_client_ids),
                "defense": defense,
                "defense_removed_clients": defense_removed,
                "mean_active_flops": float(np.mean(active_flops)) if active_flops else 0.0,
                "mean_client_parameters": float(np.mean(local_parameter_counts)) if local_parameter_counts else 0.0,
                "selected_depth": best["depth"],
                "selected_width": best["width"],
                "selected_val_accuracy": best["val_accuracy"],
                "selected_test_accuracy": best["test_accuracy"],
                "selected_test_macro_f1": best["test_macro_f1"],
                "selected_backdoor_asr": backdoor_asr,
                "active_architectures": [
                    {"depth": item.depth, "width": item.width, "offset": item.offset}
                    for item in architectures
                ],
                "tier_counts": {
                    f"d{depth}_w{width}": sum(
                        candidate.depth == depth and candidate.width == width
                        for candidate in architectures
                    )
                    for depth, width in sorted(
                        set((candidate.depth, candidate.width) for candidate in architectures)
                    )
                },
                "candidate_scores": candidate_scores,
            }
        )

    # Evaluate every deployable tier once on the untouched test split.  The
    # test set is never used for per-round selection; this table is reserved
    # for final reporting and makes the device/accuracy trade-off auditable.
    tier_scores = evaluate_candidates(model, bundle, spec, include_test=True)
    final = dict(history[-1])
    final["tier_scores"] = tier_scores
    history[-1] = final

    return {
        "dataset": bundle.name,
        "variant": variant,
        "seed": int(seed),
        "input_dim": bundle.input_dim,
        "num_classes": spec.num_classes,
        "clients": clients,
        "rounds": rounds,
        "strategy": strategy,
        "proximal_mu": float(proximal_mu),
        "drift_correction": use_drift_correction,
        "dropout_rate": dropout_rate,
        "straggler_rate": straggler_rate,
        "secure_aggregation": bool(secure_aggregation),
        "secure_aggregation_isolated": bool(secure_aggregation_isolated),
        "secure_aggregation_server_plaintext_visible_in_simulator": bool(secure_aggregation),
        "secure_aggregation_private_keys_server_side": bool(
            secure_aggregation and not secure_aggregation_isolated
        ),
        "client_data_shared_with_server": False,
        "dp_noise_multiplier": float(dp_noise_multiplier),
        "dp_clip_norm": float(dp_clip_norm),
        "dp_delta": float(dp_delta),
        "dp_epsilon_upper_bound": (
            conservative_gaussian_epsilon(dp_noise_multiplier, dp_delta, rounds)
            if dp_noise_multiplier > 0 else None
        ),
        "attack_fraction": float(attack_fraction),
        "attack_type": attack_type,
        "attack_scale": float(attack_scale),
        "defense": defense,
        "trim_ratio": float(trim_ratio),
        "history": history,
        "final": final,
        "tier_scores": tier_scores,
        "wall_clock_seconds": float(time.perf_counter() - started_at),
    }
