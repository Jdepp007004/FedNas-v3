"""Function-preserving width transforms for elastic MLP experiments."""

from __future__ import annotations

import numpy as np


def widen_replicated_state(
    state: dict[str, np.ndarray],
    source_width: int,
    target_width: int,
) -> dict[str, np.ndarray]:
    """Replicate a plain LayerNorm MLP by an integer width factor.

    This transform is exact up to LayerNorm's epsilon when the model has no
    residual skips. Hidden-to-hidden matrices and biases are divided by the
    replication factor so each replicated hidden unit computes the same
    normalized activation. The classifier columns are divided by the same
    factor so summing the replicas preserves the output.
    """
    source_width = int(source_width)
    target_width = int(target_width)
    if source_width < 1 or target_width < source_width:
        raise ValueError("target_width must be at least source_width")
    if target_width % source_width:
        raise ValueError("target_width must be an integer multiple of source_width")
    factor = target_width // source_width
    if factor == 1:
        return {key: np.asarray(value).copy() for key, value in state.items()}

    index = np.tile(np.arange(source_width, dtype=np.int64), factor)
    grown: dict[str, np.ndarray] = {}
    for key, raw in state.items():
        value = np.asarray(raw)
        if key == "classifier.weight":
            grown[key] = value[:, index].copy() / factor
        elif key == "classifier.bias":
            grown[key] = value.copy()
        elif key.startswith("layers.") and key.endswith(".weight"):
            layer_index = int(key.split(".")[1])
            if layer_index == 0:
                grown[key] = value[index, :].copy()
            else:
                grown[key] = value[np.ix_(index, index)].copy() / factor
        elif key.startswith("layers.") and key.endswith(".bias"):
            layer_index = int(key.split(".")[1])
            # Summed replicated inputs recover the parent preactivation before
            # adding bias, so the internal bias is copied without rescaling.
            grown[key] = value[index].copy()
        elif key.startswith("norms.") and value.ndim == 1:
            grown[key] = value[index].copy()
        else:
            grown[key] = value.copy()
    return grown
