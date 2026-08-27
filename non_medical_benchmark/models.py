"""Legacy and revised elastic MLPs used by the benchmark.

The legacy model mirrors the current project's prefix-depth idea: stacked
fully-connected layers with BatchNorm and a classifier head.  The revised
model keeps the nested supernet/subnet contract but adds ordered width,
LayerNorm, residual connections, and a shape-safe subnet export path.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_choice(depth: int, width: int, max_depth: int, max_width: int) -> None:
    if not 1 <= depth <= max_depth:
        raise ValueError(f"depth={depth} must be in [1, {max_depth}]")
    if not 1 <= width <= max_width:
        raise ValueError(f"width={width} must be in [1, {max_width}]")


class _BaseElasticMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, max_depth: int, hidden_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.max_depth = int(max_depth)
        self.hidden_dim = int(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(max_depth)
            ]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _linear_prefix(self, layer: nn.Linear, x: torch.Tensor, width: int, in_width: int):
        return F.linear(x[..., :in_width], layer.weight[:width, :in_width], layer.bias[:width])

    def forward(self, x: torch.Tensor, active_depth: int, active_width: int) -> torch.Tensor:
        raise NotImplementedError

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, self.max_depth, self.hidden_dim)


class ElasticMLP(_BaseElasticMLP):
    """Revised depth-and-width elastic MLP for non-IID federated clients."""

    def __init__(self, input_dim: int, num_classes: int, max_depth: int, hidden_dim: int):
        super().__init__(input_dim, num_classes, max_depth, hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max_depth)])

    def forward(self, x: torch.Tensor, active_depth: int, active_width: int) -> torch.Tensor:
        _validate_choice(active_depth, active_width, self.max_depth, self.hidden_dim)
        h = x.float()
        for i in range(active_depth):
            in_width = self.input_dim if i == 0 else active_width
            residual = h[..., :active_width] if i > 0 else None
            h = self._linear_prefix(self.layers[i], h, active_width, in_width)
            norm = self.norms[i]
            h = F.layer_norm(
                h,
                (active_width,),
                weight=norm.weight[:active_width],
                bias=norm.bias[:active_width],
                eps=norm.eps,
            )
            h = F.gelu(h)
            if residual is not None:
                h = h + residual
        return F.linear(h, self.classifier.weight[:, :active_width], self.classifier.bias)


class FixedElasticMLP(nn.Module):
    """A materialized elastic subnet for genuinely small clients.

    The server may keep the full supernet, but a client receives and trains
    only this fixed ``(depth, width)`` model.  This is important for the edge
    claim: slicing a full supernet during the forward pass still allocates the
    full model and therefore does not represent microcontroller deployment.

    Parameter names intentionally match the corresponding prefix of
    :class:`ElasticMLP`, which lets the server aggregate fixed subnet updates
    into one nested global state.
    """

    def __init__(self, input_dim: int, num_classes: int, depth: int, width: int, offset: int = 0):
        super().__init__()
        if depth < 1 or width < 1:
            raise ValueError("depth and width must be positive")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.max_depth = int(depth)
        self.hidden_dim = int(width)
        self.subnet_offset = int(offset)
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else width, width)
                for i in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.classifier = nn.Linear(width, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        active_depth: int | None = None,
        active_width: int | None = None,
    ) -> torch.Tensor:
        active_depth = self.max_depth if active_depth is None else int(active_depth)
        active_width = self.hidden_dim if active_width is None else int(active_width)
        _validate_choice(active_depth, active_width, self.max_depth, self.hidden_dim)
        h = x.float()
        for i in range(active_depth):
            layer = self.layers[i]
            norm = self.norms[i]
            in_width = self.input_dim if i == 0 else active_width
            residual = h[..., :active_width] if i > 0 else None
            h = F.linear(h[..., :in_width], layer.weight[:active_width, :in_width], layer.bias[:active_width])
            h = F.layer_norm(
                h,
                (active_width,),
                weight=norm.weight[:active_width],
                bias=norm.bias[:active_width],
                eps=norm.eps,
            )
            h = F.gelu(h)
            if residual is not None:
                h = h + residual
        return F.linear(h, self.classifier.weight[:, :active_width], self.classifier.bias)


class LegacyMLP(_BaseElasticMLP):
    """Current-style baseline with BatchNorm and no residual/width training."""

    def __init__(self, input_dim: int, num_classes: int, max_depth: int, hidden_dim: int):
        super().__init__(input_dim, num_classes, max_depth, hidden_dim)
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(max_depth)])

    def forward(self, x: torch.Tensor, active_depth: int, active_width: int) -> torch.Tensor:
        _validate_choice(active_depth, active_width, self.max_depth, self.hidden_dim)
        h = x.float()
        for i in range(active_depth):
            in_width = self.input_dim if i == 0 else active_width
            h = self._linear_prefix(self.layers[i], h, active_width, in_width)
            norm = self.norms[i]
            h = F.batch_norm(
                h,
                running_mean=norm.running_mean[:active_width],
                running_var=norm.running_var[:active_width],
                weight=norm.weight[:active_width],
                bias=norm.bias[:active_width],
                training=self.training,
                momentum=norm.momentum,
                eps=norm.eps,
            )
            h = F.relu(h)
        return F.linear(h, self.classifier.weight[:, :active_width], self.classifier.bias)


def build_model(variant: str, input_dim: int, num_classes: int, max_depth: int, hidden_dim: int):
    if variant == "elastic":
        return ElasticMLP(input_dim, num_classes, max_depth, hidden_dim)
    if variant == "legacy":
        return LegacyMLP(input_dim, num_classes, max_depth, hidden_dim)
    raise ValueError("variant must be 'elastic' or 'legacy'")


def build_fixed_subnet(
    variant: str,
    input_dim: int,
    num_classes: int,
    depth: int,
    width: int,
    offset: int = 0,
) -> nn.Module:
    """Build only the parameters required by one client subnet.

    ``legacy`` remains available as the original BatchNorm baseline.  The
    deployable fixed path is used for the revised elastic model because its
    LayerNorm contract is independent of cross-client running statistics.
    """
    if variant == "elastic":
        return FixedElasticMLP(input_dim, num_classes, depth, width, offset=offset)
    if variant == "legacy":
        return LegacyMLP(input_dim, num_classes, depth, width)
    raise ValueError("variant must be 'elastic' or 'legacy'")


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def load_numpy_state(model: nn.Module, state: Mapping[str, np.ndarray]) -> None:
    """Load a full global state while preserving the destination tensor dtypes."""
    current = model.state_dict()
    tensors = {}
    for key, value in state.items():
        if key not in current or tuple(np.asarray(value).shape) != tuple(current[key].shape):
            continue
        tensors[key] = torch.as_tensor(np.asarray(value)).to(dtype=current[key].dtype)
    model.load_state_dict(tensors, strict=False)


def load_subnet_state(
    model: nn.Module,
    global_state: Mapping[str, np.ndarray],
) -> None:
    """Load the matching prefix of a full supernet into a fixed subnet.

    The function is deliberately shape-driven, so it also works for future
    elastic modules whose parameter names follow the nested-prefix contract.
    It never allocates a full client-side model.
    """
    current = model.state_dict()
    offset = int(getattr(model, "subnet_offset", 0))
    width = int(getattr(model, "hidden_dim", 0))
    tensors = {}
    for key, target in current.items():
        if key not in global_state:
            continue
        source = np.asarray(global_state[key])
        target_shape = tuple(target.shape)
        source_shape = tuple(source.shape)
        if source_shape == target_shape:
            sliced = source
        elif source.ndim == target.ndim and all(
            target_size <= source_size
            for target_size, source_size in zip(target_shape, source_shape)
        ):
            parts = key.split(".")
            if key.startswith("layers.") and key.endswith("weight") and int(parts[1]) > 0:
                slices = (slice(offset, offset + target_shape[0]), slice(offset, offset + target_shape[1]))
            elif key.startswith("layers.") and key.endswith("weight"):
                slices = (slice(offset, offset + target_shape[0]), slice(0, target_shape[1]))
            elif key.startswith("classifier.") and key.endswith("weight"):
                slices = (slice(0, target_shape[0]), slice(offset, offset + target_shape[1]))
            elif key.startswith("norms.") and source.ndim == 1:
                slices = (slice(offset, offset + target_shape[0]),)
            elif key.startswith("layers.") and source.ndim == 1:
                slices = (slice(offset, offset + target_shape[0]),)
            else:
                slices = tuple(slice(0, size) for size in target_shape)
            sliced = source[slices]
        else:
            continue
        tensors[key] = torch.as_tensor(sliced).to(dtype=target.dtype)
    model.load_state_dict(tensors, strict=False)


def extract_subnet_state(
    model: nn.Module,
    depth: int,
    width: int,
    offset: int = 0,
) -> dict[str, np.ndarray]:
    """Export only the prefix slices exercised by one client subnet."""
    _validate_choice(depth, width, model.max_depth, model.hidden_dim)
    state = model.state_dict()
    update: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if key.startswith("layers.") or key.startswith("norms."):
            parts = key.split(".")
            layer_index = int(parts[1])
            if layer_index >= depth:
                continue
            if key.endswith("weight") and parts[0] == "layers":
                in_width = model.input_dim if layer_index == 0 else width
                out_start = offset if isinstance(model, ElasticMLP) else 0
                in_start = 0 if layer_index == 0 else out_start
                sliced = value[out_start : out_start + width, in_start : in_start + in_width]
            elif key.endswith("bias") and parts[0] == "layers":
                out_start = offset if isinstance(model, ElasticMLP) else 0
                sliced = value[out_start : out_start + width]
            elif parts[0] == "norms" and value.ndim == 1:
                out_start = offset if isinstance(model, ElasticMLP) else 0
                sliced = value[out_start : out_start + width]
            else:
                # BatchNorm running statistics and counters are included for
                # the legacy comparison; aggregation remains shape-aware.
                sliced = value[:width] if value.ndim == 1 else value
            update[key] = sliced.detach().cpu().numpy().copy()
        elif key == "classifier.weight":
            out_start = offset if isinstance(model, ElasticMLP) else 0
            update[key] = value[:, out_start : out_start + width].detach().cpu().numpy().copy()
        elif key == "classifier.bias":
            update[key] = value.detach().cpu().numpy().copy()
    return update


def subnet_parameters(input_dim: int, num_classes: int, depth: int, width: int) -> int:
    """Number of trainable parameters actually used by a subnet."""
    total = width * (input_dim + 1)
    if depth > 1:
        total += (depth - 1) * width * (width + 1)
    total += depth * (2 * width)
    total += num_classes * width + num_classes
    return int(total)


def subnet_flops(input_dim: int, num_classes: int, depth: int, width: int) -> int:
    """Approximate multiply-add FLOPs for the active subnet."""
    total = 2 * input_dim * width
    total += max(0, depth - 1) * 2 * width * width
    total += 2 * num_classes * width
    return int(total)
