"""CIFAR-10/100 image benchmark for the same elastic FL contract.

The tabular track uses an elastic MLP; this module keeps the federated
protocol and nested prefix idea but replaces only the task model with a small
residual CNN.  Clients materialize a fixed subnet, while the server evaluates
the full supernet and aggregates shape-compatible prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .data import make_client_partitions
from .federated import ClientArchitecture, aggregate_subnet_updates
from .models import load_numpy_state, load_subnet_state, state_to_numpy


@dataclass
class CifarBundle:
    name: str
    num_classes: int
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_labels: np.ndarray


class CifarElasticCNN(nn.Module):
    """Small nested CNN supernet with ordered depth and channel prefixes."""

    def __init__(self, num_classes: int, max_depth: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = 3
        self.num_classes = int(num_classes)
        self.max_depth = int(max_depth)
        self.hidden_dim = int(hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    3 if index == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for index in range(max_depth)
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_dim) for _ in range(max_depth)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, active_depth: int, active_width: int, offset: int = 0
    ) -> torch.Tensor:
        if not 1 <= active_depth <= self.max_depth:
            raise ValueError("active_depth is outside the supernet")
        if not 1 <= active_width <= self.hidden_dim:
            raise ValueError("active_width is outside the supernet")
        h = x.float()
        for index in range(active_depth):
            in_width = 3 if index == 0 else active_width
            layer = self.layers[index]
            residual = h if index > 0 else None
            out_start = int(offset)
            in_start = 0 if index == 0 else int(offset)
            h = F.conv2d(
                h[:, :in_width] if index == 0 else h[:, in_start : in_start + in_width],
                layer.weight[out_start : out_start + active_width, :in_width]
                if index == 0
                else layer.weight[
                    out_start : out_start + active_width,
                    in_start : in_start + in_width,
                ],
                layer.bias[out_start : out_start + active_width],
                padding=1,
            )
            norm = self.norms[index]
            h = F.group_norm(
                h,
                1,
                weight=norm.weight[out_start : out_start + active_width],
                bias=norm.bias[out_start : out_start + active_width],
                eps=norm.eps,
            )
            h = F.gelu(h)
            if residual is not None:
                h = h + residual[:, :active_width]
            if index == 1:
                h = F.avg_pool2d(h, kernel_size=2)
        pooled = F.adaptive_avg_pool2d(h, output_size=1).flatten(1)
        return F.linear(
            pooled,
            self.classifier.weight[:, int(offset) : int(offset) + active_width],
            self.classifier.bias,
        )


class FixedCifarCNN(nn.Module):
    """Materialized client subnet; it allocates only active channels/layers."""

    def __init__(self, num_classes: int, depth: int, width: int, offset: int = 0):
        super().__init__()
        self.input_dim = 3
        self.num_classes = int(num_classes)
        self.max_depth = int(depth)
        self.hidden_dim = int(width)
        self.subnet_offset = int(offset)
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    3 if index == 0 else width,
                    width,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for index in range(depth)
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(1, width) for _ in range(depth)])
        self.classifier = nn.Linear(width, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        active_depth: int | None = None,
        active_width: int | None = None,
    ) -> torch.Tensor:
        active_depth = self.max_depth if active_depth is None else int(active_depth)
        active_width = self.hidden_dim if active_width is None else int(active_width)
        h = x.float()
        for index in range(active_depth):
            layer, norm = self.layers[index], self.norms[index]
            residual = h if index > 0 else None
            in_width = 3 if index == 0 else active_width
            h = F.conv2d(
                h[:, :in_width],
                layer.weight[:active_width, :in_width],
                layer.bias[:active_width],
                padding=1,
            )
            h = F.group_norm(
                h,
                1,
                weight=norm.weight[:active_width],
                bias=norm.bias[:active_width],
            )
            h = F.gelu(h)
            if residual is not None:
                h = h + residual
            if index == 1:
                h = F.avg_pool2d(h, kernel_size=2)
        pooled = F.adaptive_avg_pool2d(h, output_size=1).flatten(1)
        return F.linear(
            pooled,
            self.classifier.weight[:, :active_width],
            self.classifier.bias,
        )


def build_cifar_subnet(
    num_classes: int, depth: int, width: int, offset: int = 0
) -> FixedCifarCNN:
    return FixedCifarCNN(num_classes, depth, width, offset=offset)


def extract_cifar_subnet_state(
    model: CifarElasticCNN | FixedCifarCNN,
    depth: int,
    width: int,
    offset: int = 0,
) -> dict[str, np.ndarray]:
    state = model.state_dict()
    result: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if key.startswith("layers.") or key.startswith("norms."):
            index = int(key.split(".")[1])
            if index >= depth:
                continue
            if key.startswith("layers.") and key.endswith("weight"):
                in_width = 3 if index == 0 else width
                out_start = offset if isinstance(model, CifarElasticCNN) else 0
                in_start = 0 if index == 0 else out_start
                sliced = value[
                    out_start : out_start + width,
                    :in_width if index == 0 else in_start + in_width,
                ]
                if index > 0:
                    sliced = sliced[:, in_start : in_start + in_width]
            elif key.startswith("layers.") and key.endswith("bias"):
                out_start = offset if isinstance(model, CifarElasticCNN) else 0
                sliced = value[out_start : out_start + width]
            elif value.ndim == 1:
                out_start = offset if isinstance(model, CifarElasticCNN) else 0
                sliced = value[out_start : out_start + width]
            else:
                sliced = value
            result[key] = sliced.detach().cpu().numpy().copy()
        elif key == "classifier.weight":
            out_start = offset if isinstance(model, CifarElasticCNN) else 0
            result[key] = value[:, out_start : out_start + width].detach().cpu().numpy().copy()
        elif key == "classifier.bias":
            result[key] = value.detach().cpu().numpy().copy()
    return result


def cifar_parameters(num_classes: int, depth: int, width: int) -> int:
    total = width * (3 * 3 * 3 + 1)
    total += max(0, depth - 1) * width * (width * 3 * 3 + 1)
    total += depth * 2 * width
    total += num_classes * width + num_classes
    return int(total)


def cifar_flops(num_classes: int, depth: int, width: int, image_size: int = 32) -> int:
    total = 0
    for index in range(depth):
        spatial = image_size * image_size if index < 2 else (image_size // 2) ** 2
        input_width = 3 if index == 0 else width
        total += 2 * spatial * (3 * 3 * input_width * width)
    total += 2 * num_classes * width
    return int(total)


def _select_indices(labels: np.ndarray, limit: Optional[int], seed: int) -> np.ndarray:
    indices = np.arange(len(labels), dtype=np.int64)
    if limit is None or limit >= len(labels):
        return indices
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(indices, labels))
    return np.asarray(sorted(indices[selected]), dtype=np.int64)


def load_cifar_bundle(
    name: str,
    data_dir: str | Path = "data/non_medical/cifar",
    download: bool = False,
    max_train: Optional[int] = 5000,
    max_test: Optional[int] = 2000,
    seed: int = 42,
) -> CifarBundle:
    if name not in {"cifar10", "cifar100"}:
        raise ValueError("name must be 'cifar10' or 'cifar100'")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    root = str(Path(data_dir))
    train_full = dataset_cls(root=root, train=True, transform=transform, download=download)
    test_full = dataset_cls(root=root, train=False, transform=transform, download=download)
    train_targets = np.asarray(train_full.targets, dtype=np.int64)
    test_targets = np.asarray(test_full.targets, dtype=np.int64)
    train_selected = _select_indices(train_targets, max_train, seed)
    test_selected = _select_indices(test_targets, max_test, seed + 1)
    train_selected, val_selected = train_test_split(
        train_selected,
        test_size=max(1, int(round(0.1 * len(train_selected)))),
        random_state=seed,
        stratify=train_targets[train_selected],
    )
    return CifarBundle(
        name=name,
        num_classes=10 if name == "cifar10" else 100,
        train_dataset=Subset(train_full, train_selected.tolist()),
        val_dataset=Subset(train_full, val_selected.tolist()),
        test_dataset=Subset(test_full, test_selected.tolist()),
        train_labels=train_targets[train_selected],
    )


def make_cifar_loader(dataset: Dataset, indices: Optional[np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    selected = dataset if indices is None else Subset(dataset, indices.tolist())
    return DataLoader(selected, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate_cifar(
    model: CifarElasticCNN,
    dataset: Dataset,
    architecture: ClientArchitecture,
    batch_size: int,
) -> float:
    model.eval()
    loader = make_cifar_loader(dataset, None, batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            prediction = model(
                images,
                architecture.depth,
                architecture.width,
                architecture.offset,
            ).argmax(dim=1)
            correct += int((prediction == labels).sum().item())
            total += len(labels)
    return float(correct / max(total, 1))


def _train_cifar_client(
    model: nn.Module,
    loader: DataLoader,
    architecture: ClientArchitecture,
    learning_rate: float,
    local_epochs: int,
    ordered_dropout: bool = False,
    ordered_widths: tuple[int, ...] = (),
    proximal_mu: float = 0.0,
) -> tuple[dict[str, np.ndarray], float]:
    snapshot = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_value = 0.0
    for _ in range(local_epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            active_width = architecture.width
            teacher_logits = None
            if ordered_dropout:
                choices = tuple(value for value in ordered_widths if value <= architecture.width)
                active_width = int(np.random.choice(choices or (architecture.width,)))
            if isinstance(model, FixedCifarCNN):
                if ordered_dropout and active_width < architecture.width:
                    with torch.no_grad():
                        teacher_logits = model(images, architecture.depth, architecture.width)
                logits = model(images, architecture.depth, active_width)
            else:
                logits = model(images, architecture.depth, active_width, architecture.offset)
            loss = F.cross_entropy(logits, labels)
            if teacher_logits is not None:
                temperature = 2.0
                loss = 0.9 * loss + 0.1 * F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
            if proximal_mu > 0:
                loss = loss + 0.5 * proximal_mu * sum(
                    (local - global_).pow(2).sum()
                    for local, global_ in zip(model.parameters(), snapshot)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = float(loss.detach().item())
    if isinstance(model, FixedCifarCNN):
        state = extract_cifar_subnet_state(
            model, architecture.depth, architecture.width, architecture.offset
        )
    else:
        state = extract_cifar_subnet_state(
            model, architecture.depth, architecture.width, architecture.offset
        )
    return state, loss_value


def _architecture_for_round(
    client_id: int,
    round_id: int,
    strategy: str,
    max_depth: int,
    widths: tuple[int, ...],
) -> ClientArchitecture:
    if strategy in {"fedavg", "fedprox"}:
        return ClientArchitecture(max_depth, widths[-1], 0)
    candidates = (
        ClientArchitecture(1, widths[0], 0),
        ClientArchitecture(max_depth // 2, widths[len(widths) // 2 - 1], 0),
        ClientArchitecture(max_depth, widths[-1], 0),
    )
    architecture = candidates[(client_id + round_id) % len(candidates)]
    if strategy in {"fedrolex", "cc_efl"}:
        max_offset = max(0, widths[-1] - architecture.width)
        offset = 0 if max_offset == 0 else (round_id * architecture.width + client_id) % (max_offset + 1)
        return ClientArchitecture(architecture.depth, architecture.width, offset)
    return architecture


def run_cifar_experiment(
    bundle: CifarBundle,
    strategy: str = "elastic",
    rounds: int = 5,
    seed: int = 42,
    clients: int = 4,
    batch_size: int = 64,
    local_epochs: int = 1,
    learning_rate: float = 1e-3,
    widths: tuple[int, ...] = (16, 32, 48, 64),
    max_depth: int = 4,
) -> dict:
    allowed = {"fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "elastic", "maxnet", "cc_efl"}
    if strategy not in allowed:
        raise ValueError(f"strategy must be one of {sorted(allowed)}")
    if rounds < 1 or clients < 1:
        raise ValueError("rounds and clients must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    started_at = time.perf_counter()
    partitions = make_client_partitions(bundle.train_labels, clients, alpha=0.5, seed=seed)
    model = CifarElasticCNN(bundle.num_classes, max_depth=max_depth, hidden_dim=widths[-1])
    global_state = state_to_numpy(model)
    history = []
    for round_id in range(rounds):
        updates = []
        counts = []
        architectures = []
        losses = []
        for client_id, indices in enumerate(partitions):
            architecture = _architecture_for_round(client_id, round_id, strategy, max_depth, widths)
            local_model = build_cifar_subnet(
                bundle.num_classes,
                architecture.depth,
                architecture.width,
                architecture.offset,
            )
            load_subnet_state(local_model, global_state)
            update, loss = _train_cifar_client(
                local_model,
                make_cifar_loader(bundle.train_dataset, indices, batch_size, shuffle=True),
                architecture,
                learning_rate,
                local_epochs,
                ordered_dropout=strategy == "fjord",
                ordered_widths=widths,
                proximal_mu=1e-3 if strategy == "fedprox" else 0.0,
            )
            updates.append(update)
            counts.append(len(indices))
            architectures.append(architecture)
            losses.append(loss)
        capacity_weighting = strategy in {"elastic", "maxnet", "cc_efl"}
        beta = 0.2 + 0.7 * 0.5 * (1.0 + np.cos(np.pi * round_id / max(rounds - 1, 1)))
        max_capacity = max(item.depth * item.width for item in architectures)
        priorities = [
            beta if item.depth * item.width == max_capacity else 1.0 - beta
            for item in architectures
        ] if capacity_weighting else [1.0] * len(architectures)
        global_state = aggregate_subnet_updates(
            global_state,
            updates,
            counts,
            priorities=priorities,
            architectures=architectures,
        )
        load_numpy_state(model, global_state)
        candidates = []
        for depth in range(1, max_depth + 1):
            for width in widths:
                candidate_architecture = ClientArchitecture(depth, width, 0)
                val_accuracy = _evaluate_cifar(
                    model, bundle.val_dataset, candidate_architecture, batch_size
                )
                candidates.append(
                    {
                        "depth": depth,
                        "width": width,
                        "parameters": cifar_parameters(bundle.num_classes, depth, width),
                        "flops": cifar_flops(bundle.num_classes, depth, width),
                        "val_accuracy": val_accuracy,
                        "test_accuracy": None,
                    }
                )
        best_val = max(item["val_accuracy"] for item in candidates)
        eligible = [item for item in candidates if item["val_accuracy"] >= best_val - 0.002]
        best = min(eligible, key=lambda item: (item["flops"], item["parameters"]))
        best = dict(best)
        best["test_accuracy"] = _evaluate_cifar(
            model,
            bundle.test_dataset,
            ClientArchitecture(best["depth"], best["width"], 0),
            batch_size,
        )
        history.append(
            {
                "round": round_id + 1,
                "strategy": strategy,
                "mean_client_loss": float(np.mean(losses)),
                "maxnet_beta": float(beta),
                "participating_clients": len(updates),
                "participation_rate": 1.0,
                "total_upload_bytes": int(sum(sum(value.nbytes for value in update.values()) for update in updates)),
                "total_download_bytes": int(
                    sum(
                        cifar_parameters(bundle.num_classes, architecture.depth, architecture.width) * 4
                        for architecture in architectures
                    )
                ),
                "mean_active_flops": float(np.mean([
                    cifar_flops(bundle.num_classes, item.depth, item.width)
                    for item in architectures
                ])),
                "mean_client_parameters": float(
                    np.mean([
                        cifar_parameters(bundle.num_classes, item.depth, item.width)
                        for item in architectures
                    ])
                ),
                "selected_depth": best["depth"],
                "selected_width": best["width"],
                "selected_val_accuracy": best["val_accuracy"],
                "selected_test_accuracy": best["test_accuracy"],
                "active_architectures": [
                    {"depth": item.depth, "width": item.width, "offset": item.offset}
                    for item in architectures
                ],
                "candidate_scores": candidates,
            }
        )
    load_numpy_state(model, global_state)
    tier_scores = []
    for depth in range(1, max_depth + 1):
        for width in widths:
            architecture = ClientArchitecture(depth, width, 0)
            tier_scores.append(
                {
                    "depth": depth,
                    "width": width,
                    "parameters": cifar_parameters(bundle.num_classes, depth, width),
                    "flops": cifar_flops(bundle.num_classes, depth, width),
                    "test_accuracy": _evaluate_cifar(
                        model, bundle.test_dataset, architecture, batch_size
                    ),
                }
            )
    final = dict(history[-1])
    final["tier_scores"] = tier_scores
    history[-1] = final
    return {
        "dataset": bundle.name,
        "variant": "elastic_cnn",
        "seed": int(seed),
        "num_classes": bundle.num_classes,
        "clients": clients,
        "rounds": rounds,
        "strategy": strategy,
        "history": history,
        "final": final,
        "tier_scores": tier_scores,
        "wall_clock_seconds": float(time.perf_counter() - started_at),
    }
