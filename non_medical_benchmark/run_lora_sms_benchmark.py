"""Federated LoRA fine-tuning of BERT-Tiny on a duplicate-grouped SMS split.

The BERT-Tiny backbone is frozen except for rank-window LoRA adapters on the last
Transformer block's query/value projections. A small classifier head is also
trained. This is a process-level FL simulation, not a privacy protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

from .data import make_client_partitions
from .run_pretrained_head_benchmark import (
    DEFAULT_ROOT,
    SMS_URL,
    ensure_sms_archive,
    read_sms_corpus,
    sha256_file,
    split_indices,
)
from .stats import confidence_interval, paired_interval


CAPACITY_TIERS = (1, 2, 3)
LORA_MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
LORA_MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
SUPPORTED_METHODS = (
    "fedavg",
    "fedprox",
    "heterofl",
    "fedrolex",
    "maxnet",
    "cc_efl",
    "cc_efl_no_prox",
    "cc_efl_uniform",
    "cc_efl_random_window",
)
PARTITION_MODES = (
    "dirichlet_label_and_quantity",
    "quantity_balanced_label_dirichlet",
)


def make_sms_client_partitions(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    partition_mode: str,
) -> list[np.ndarray]:
    """Build the original joint-skew partition or a size-balanced control."""
    if partition_mode == "dirichlet_label_and_quantity":
        return make_client_partitions(
            labels, num_clients, alpha=alpha, seed=seed, min_samples=4
        )
    if partition_mode != "quantity_balanced_label_dirichlet":
        raise ValueError(f"Unsupported SMS partition mode: {partition_mode}")
    if num_clients < 2 or alpha <= 0:
        raise ValueError("num_clients must be >= 2 and alpha must be positive")

    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Quantity-balanced SMS partitions require binary labels")
    rng = np.random.default_rng(seed)
    quotient, remainder = divmod(len(labels), num_clients)
    capacities = np.full(num_clients, quotient, dtype=np.int64)
    capacities[:remainder] += 1
    if np.any(capacities < 2):
        raise ValueError("Each client needs at least two examples")

    positive = np.flatnonzero(labels == classes[-1])
    negative = np.flatnonzero(labels == classes[0])
    if min(len(positive), len(negative)) < num_clients:
        raise ValueError("Each class needs at least one example per client")
    rng.shuffle(positive)
    rng.shuffle(negative)

    # Draw a Dirichlet target for the positive class, then project it onto
    # exact per-client capacities while retaining at least one example/class.
    probabilities = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    target = probabilities * len(positive)
    lower = np.ones(num_clients, dtype=np.int64)
    upper = capacities - 1
    counts = np.clip(rng.multinomial(len(positive), probabilities), lower, upper)
    while int(counts.sum()) != len(positive):
        delta = len(positive) - int(counts.sum())
        if delta > 0:
            eligible = np.flatnonzero(counts < upper)
            if not len(eligible):
                raise RuntimeError("Cannot place all positive-class examples")
            index = eligible[np.argmax(target[eligible] - counts[eligible])]
            counts[index] += 1
        else:
            eligible = np.flatnonzero(counts > lower)
            if not len(eligible):
                raise RuntimeError("Cannot remove excess positive-class examples")
            index = eligible[np.argmax(counts[eligible] - target[eligible])]
            counts[index] -= 1

    partitions = []
    positive_start = 0
    negative_start = 0
    for client_id, capacity in enumerate(capacities):
        positive_count = int(counts[client_id])
        negative_count = int(capacity - positive_count)
        client_indices = np.concatenate((
            positive[positive_start : positive_start + positive_count],
            negative[negative_start : negative_start + negative_count],
        ))
        positive_start += positive_count
        negative_start += negative_count
        partitions.append(np.sort(client_indices.astype(np.int64)))
    return partitions


class LoRALinear(nn.Module):
    """Frozen linear transform plus independently selectable LoRA components."""

    def __init__(self, base: nn.Linear, max_rank: int, scale: float) -> None:
        super().__init__()
        self.base = base
        self.max_rank = max_rank
        self.scale = scale
        self.lora_A = nn.Parameter(torch.empty(max_rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, max_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self.active_start = 0
        self.active_width = max_rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def set_window(self, start: int, width: int) -> None:
        if width < 1 or width > self.max_rank or start < 0:
            raise ValueError("Invalid LoRA rank window")
        if start + width > self.max_rank:
            raise ValueError("LoRA rank windows are bounded and may not wrap")
        self.active_start = start
        self.active_width = width

    def active_indices(self, device: torch.device) -> torch.Tensor:
        return torch.arange(
            self.active_start,
            self.active_start + self.active_width,
            device=device,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = self.active_indices(self.lora_A.device)
        low_rank = F.linear(inputs, self.lora_A.index_select(0, indices))
        update = F.linear(low_rank, self.lora_B.index_select(1, indices))
        return self.base(inputs) + update * self.scale


class TinyBertLoRAClassifier(nn.Module):
    """BERT-Tiny with LoRA on the final block's query/value projections."""

    def __init__(self, max_rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            LORA_MODEL_ID, revision=LORA_MODEL_REVISION
        )
        self.base_encoder_parameters = sum(
            parameter.numel() for parameter in self.encoder.parameters()
        )
        layer_index = self.encoder.config.num_hidden_layers - 1
        module_map = dict(self.encoder.named_modules())
        self.lora_modules: list[LoRALinear] = []
        for projection in ("query", "value"):
            suffix = f"encoder.layer.{layer_index}.attention.self.{projection}"
            matches = [
                name for name in module_map
                if name.endswith(suffix) and isinstance(module_map[name], nn.Linear)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one final-layer {projection} projection, found {matches}"
                )
            name = matches[0]
            pieces = name.split(".")
            parent: nn.Module = self.encoder
            for piece in pieces[:-1]:
                parent = getattr(parent, piece)
            base = getattr(parent, pieces[-1])
            wrapped = LoRALinear(base, max_rank=max_rank, scale=1.0 / max_rank)
            setattr(parent, pieces[-1], wrapped)
            self.lora_modules.append(wrapped)

        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for module in self.lora_modules:
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    def set_window(self, start: int, width: int) -> None:
        for module in self.lora_modules:
            module.set_window(start, width)

    def named_trainable_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def forward(
        self, prefix_hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        extended_mask = _extended_attention_mask(attention_mask, prefix_hidden.dtype)
        last_layer = self.encoder.encoder.layer[-1]
        hidden = last_layer(
            prefix_hidden,
            attention_mask=extended_mask,
        )
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        pooled = (
            self.encoder.pooler(hidden)
            if self.encoder.pooler is not None
            else hidden[:, 0]
        )
        return self.classifier(self.dropout(pooled))


def _extended_attention_mask(
    attention_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return (
        1.0 - attention_mask[:, None, None, :].to(dtype)
    ) * torch.finfo(dtype).min


def _encode_frozen_prefix(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    encoder: nn.Module,
) -> torch.Tensor:
    """Cache the frozen prefix layers; this is exact for final-layer LoRA."""
    encoder.eval()
    outputs = []
    with torch.no_grad():
        for begin in range(0, len(input_ids), batch_size):
            ids = input_ids[begin : begin + batch_size]
            mask = attention_mask[begin : begin + batch_size]
            extended_mask = _extended_attention_mask(mask, encoder.dtype)
            hidden = encoder.embeddings(
                input_ids=ids,
                token_type_ids=torch.zeros_like(ids),
            )
            for layer in encoder.encoder.layer[:-1]:
                hidden = layer(hidden, attention_mask=extended_mask)
                if isinstance(hidden, (tuple, list)):
                    hidden = hidden[0]
            outputs.append(hidden.cpu())

    return torch.cat(outputs, dim=0)


def _assert_prefix_equivalence(
    model: TinyBertLoRAClassifier,
    prefix_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    """Check that cached prefix features reproduce the unadapted full encoder."""
    count = min(8, len(input_ids))
    model.encoder.eval()
    with torch.no_grad():
        expected = model.encoder(
            input_ids=input_ids[:count],
            attention_mask=attention_mask[:count],
        ).last_hidden_state
        hidden = model.encoder.encoder.layer[-1](
            prefix_hidden[:count],
            attention_mask=_extended_attention_mask(
                attention_mask[:count], prefix_hidden.dtype
            ),
        )
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
    if not torch.allclose(expected, hidden, atol=2e-5, rtol=2e-4):
        raise RuntimeError(
            "Cached frozen prefix does not reproduce full BERT-Tiny output"
        )


def _capture_state(model: TinyBertLoRAClassifier) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_trainable_parameters()
    }


def _load_state(
    model: TinyBertLoRAClassifier, state: dict[str, torch.Tensor]
) -> None:
    with torch.no_grad():
        for name, parameter in model.named_trainable_parameters():
            parameter.copy_(state[name].to(parameter.device))


def _window_mask(
    tensor: torch.Tensor, name: str, start: int, width: int
) -> torch.Tensor:
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    indices = torch.arange(start, start + width)
    if name.endswith("lora_A"):
        mask[indices, :] = True
    elif name.endswith("lora_B"):
        mask[:, indices] = True
    else:
        mask.fill_(True)
    return mask


def _beta(round_id: int, rounds: int) -> float:
    return 0.2 + 0.35 * (
        1.0 + np.cos(np.pi * round_id / max(rounds - 1, 1))
    )


def _client_window(
    method: str,
    round_id: int,
    client_id: int,
    width: int,
    max_rank: int,
    rng: np.random.Generator,
) -> int:
    max_start = max_rank - width
    if max_start == 0:
        return 0
    if method in {"fedrolex", "cc_efl", "cc_efl_no_prox", "cc_efl_uniform"}:
        return (round_id + client_id) % (max_start + 1)
    if method == "cc_efl_random_window":
        return int(rng.integers(0, max_start + 1))
    return 0


def _priority(method: str, round_id: int, rounds: int, width: int) -> float:
    if method not in {"maxnet", "cc_efl", "cc_efl_no_prox", "cc_efl_random_window"}:
        return 1.0
    beta = _beta(round_id, rounds)
    if width == max(CAPACITY_TIERS):
        return beta
    return 1.0 - beta


def _evaluate(
    model: TinyBertLoRAClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: np.ndarray,
    rank: int,
    start: int,
    batch_size: int,
) -> dict[str, float]:
    model.set_window(start, rank)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(labels), batch_size):
            end = min(begin + batch_size, len(labels))
            logits = model(input_ids[begin:end], attention_mask[begin:end])
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }


def _select_deployments(
    model: TinyBertLoRAClassifier,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_y: np.ndarray,
    max_rank: int,
    batch_size: int,
) -> dict[int, dict]:
    choices: dict[int, dict] = {}
    for capacity in CAPACITY_TIERS:
        candidates = []
        for rank in CAPACITY_TIERS:
            if rank > capacity:
                continue
            for start in range(max_rank - rank + 1):
                score = _evaluate(
                    model, val_ids, val_mask, val_y, rank, start, batch_size
                )["macro_f1"]
                candidates.append((rank, start, score))
        best_score = max(row[2] for row in candidates)
        eligible = [row for row in candidates if row[2] >= best_score - 0.002]
        selected = min(eligible, key=lambda row: (row[0], -row[2], row[1]))
        choices[capacity] = {
            "capacity_rank": capacity,
            "selected_rank": selected[0],
            "selected_start": selected[1],
            "validation_macro_f1": selected[2],
            "best_feasible_validation_macro_f1": best_score,
        }
    return choices


def _class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    return torch.as_tensor(
        len(labels) / np.maximum(counts * 2, 1.0), dtype=torch.float32
    )


def _train_local_epoch(
    model: TinyBertLoRAClassifier,
    global_state: dict[str, torch.Tensor],
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_y: np.ndarray,
    indices: np.ndarray,
    class_weights: torch.Tensor,
    rank: int,
    start: int,
    learning_rate: float,
    proximal_mu: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.set_window(start, rank)
    _load_state(model, global_state)
    model.train()
    parameters = [parameter for _, parameter in model.named_trainable_parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    order = np.random.permutation(indices)
    for begin in range(0, len(order), batch_size):
        selected = order[begin : begin + batch_size]
        batch_indices = torch.as_tensor(selected, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            train_ids.index_select(0, batch_indices),
            train_mask.index_select(0, batch_indices),
        )
        targets = torch.as_tensor(train_y[selected], dtype=torch.long)
        loss = F.cross_entropy(logits, targets, weight=class_weights)
        if proximal_mu > 0:
            prox = torch.zeros((), dtype=loss.dtype)
            for name, parameter in model.named_trainable_parameters():
                reference = global_state[name].to(parameter.device)
                prox = prox + torch.sum((parameter - reference) ** 2)
            loss = loss + (proximal_mu / 2.0) * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return _capture_state(model)


def _run_federated(
    model: TinyBertLoRAClassifier,
    method: str,
    seed: int,
    rounds: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_y: np.ndarray,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_y: np.ndarray,
    test_ids: torch.Tensor,
    test_mask: torch.Tensor,
    test_y: np.ndarray,
    clients: int,
    alpha: float,
    partition_mode: str,
    max_rank: int,
    learning_rate: float,
    proximal_mu: float,
    batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    partitions = make_sms_client_partitions(
        train_y,
        clients,
        alpha=alpha,
        seed=10_000 + seed,
        partition_mode=partition_mode,
    )
    rng = np.random.default_rng(20_000 + seed)
    capacities = [CAPACITY_TIERS[index % len(CAPACITY_TIERS)] for index in range(clients)]
    global_state = _capture_state(model)
    started = time.perf_counter()
    round_manifest = []
    client_class_weights = [_class_weights(train_y[part]) for part in partitions]

    for round_id in range(rounds):
        local_states = []
        windows = []
        priorities = []
        for client_id, indices in enumerate(partitions):
            rank = max_rank if method in {"fedavg", "fedprox"} else capacities[client_id]
            start = _client_window(method, round_id, client_id, rank, max_rank, rng)
            mu = proximal_mu if method in {
                "fedprox", "cc_efl", "cc_efl_random_window", "cc_efl_uniform"
            } else 0.0
            local_state = _train_local_epoch(
                model=model,
                global_state=global_state,
                train_ids=train_ids,
                train_mask=train_mask,
                train_y=train_y,
                indices=indices,
                class_weights=client_class_weights[client_id],
                rank=rank,
                start=start,
                learning_rate=learning_rate,
                proximal_mu=mu,
                batch_size=batch_size,
            )
            local_states.append(local_state)
            windows.append({"client": client_id, "rank": rank, "start": start})
            priorities.append(_priority(method, round_id, rounds, rank))

        updated = {name: tensor.clone() for name, tensor in global_state.items()}
        for name, reference in global_state.items():
            numerator = torch.zeros_like(reference)
            denominator = torch.zeros_like(reference, dtype=torch.float64)
            for client_id, state in enumerate(local_states):
                rank = windows[client_id]["rank"]
                start = windows[client_id]["start"]
                weight = float(len(partitions[client_id])) * priorities[client_id]
                mask = _window_mask(reference, name, start, rank)
                numerator += state[name] * mask.to(reference.dtype) * weight
                denominator += mask.to(torch.float64) * weight
            covered = denominator > 0
            updated[name] = torch.where(
                covered,
                numerator / denominator.clamp_min(1e-12).to(reference.dtype),
                reference,
            )
        global_state = updated
        round_manifest.append({"round": round_id, "windows": windows})

    _load_state(model, global_state)
    selected = _select_deployments(
        model, val_ids, val_mask, val_y, max_rank, batch_size
    )
    by_capacity = {}
    for capacity, deployment in selected.items():
        by_capacity[str(capacity)] = _evaluate(
            model, test_ids, test_mask, test_y,
            deployment["selected_rank"], deployment["selected_start"], batch_size,
        )
    client_counts = {
        str(rank): sum(1 for value in capacities if value == rank)
        for rank in CAPACITY_TIERS
    }
    total_clients = sum(client_counts.values())
    capacity_weighted_macro_f1 = sum(
        client_counts[str(rank)] * by_capacity[str(rank)]["macro_f1"]
        for rank in CAPACITY_TIERS
    ) / total_clients
    capacity_weighted_accuracy = sum(
        client_counts[str(rank)] * by_capacity[str(rank)]["accuracy"]
        for rank in CAPACITY_TIERS
    ) / total_clients
    hidden_size = int(model.encoder.config.hidden_size)
    active_update_parameters = [
        4 * hidden_size * window["rank"] + (2 * hidden_size + 2)
        for window in round_manifest[0]["windows"]
    ]
    parameter_coordinates_by_round = [
        sum(
            4 * hidden_size * window["rank"] + (2 * hidden_size + 2)
            for window in entry["windows"]
        )
        for entry in round_manifest
    ]
    bytes_per_round = 4 * float(np.mean(parameter_coordinates_by_round))
    max_active_rank_coordinates = set()
    for entry in round_manifest:
        for window in entry["windows"]:
            max_active_rank_coordinates.update(
                range(window["start"], window["start"] + window["rank"])
            )
    return {
        "dataset": "sms_spam_bert_tiny_lora",
        "method": method,
        "seed": seed,
        "rounds": rounds,
        "clients": clients,
        "client_capacity_ranks": capacities,
        "client_partition_sizes": [len(part) for part in partitions],
        "client_class_counts": [
            {
                "ham": int(np.sum(train_y[part] == 0)),
                "spam": int(np.sum(train_y[part] == 1)),
            }
            for part in partitions
        ],
        "client_partition_sha256": hashlib.sha256(
            b"".join(np.asarray(part, dtype="<i8").tobytes() for part in partitions)
        ).hexdigest(),
        "selected_deployments_by_capacity": selected,
        "test_metrics_by_capacity": by_capacity,
        "capacity_weighted_test_macro_f1": capacity_weighted_macro_f1,
        "capacity_weighted_test_accuracy": capacity_weighted_accuracy,
        "mean_active_trainable_parameters_per_client": float(np.mean(active_update_parameters)),
        "mean_parameter_coordinates_uploaded_per_round": float(
            np.mean(parameter_coordinates_by_round)
        ),
        "upload_bytes_per_round_float32": bytes_per_round,
        "download_bytes_per_round_float32": bytes_per_round,
        "upload_bytes_per_run_float32": bytes_per_round * rounds,
        "download_bytes_per_run_float32": bytes_per_round * rounds,
        "full_rank_trainable_parameters": (
            4 * hidden_size * max_rank + (2 * hidden_size + 2)
        ),
        "base_encoder_parameters": model.base_encoder_parameters,
        "rank_coordinate_coverage_fraction": len(max_active_rank_coordinates) / max_rank,
        "round_manifest": round_manifest,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _run_centralized(
    model: TinyBertLoRAClassifier,
    seed: int,
    rounds: int,
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    train_y: np.ndarray,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    val_y: np.ndarray,
    test_ids: torch.Tensor,
    test_mask: torch.Tensor,
    test_y: np.ndarray,
    max_rank: int,
    learning_rate: float,
    batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.set_window(0, max_rank)
    model.train()
    parameters = [parameter for _, parameter in model.named_trainable_parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    weights = _class_weights(train_y)
    started = time.perf_counter()
    for _ in range(rounds):
        order = np.random.permutation(len(train_y))
        for begin in range(0, len(order), batch_size):
            selected = order[begin : begin + batch_size]
            indices = torch.as_tensor(selected, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train_ids.index_select(0, indices),
                train_mask.index_select(0, indices),
            )
            targets = torch.as_tensor(train_y[selected], dtype=torch.long)
            loss = F.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
    validation = _evaluate(model, val_ids, val_mask, val_y, max_rank, 0, batch_size)
    test = _evaluate(model, test_ids, test_mask, test_y, max_rank, 0, batch_size)
    return {
        "seed": seed,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "validation": validation,
        "test": test,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _paired_inference(results: list[dict], methods: list[str]) -> list[dict]:
    by_method = {
        method: {
            row["seed"]: row["capacity_weighted_test_macro_f1"]
            for row in results if row["method"] == method
        }
        for method in methods
    }
    rows = []
    for method in methods:
        if method == "fedavg":
            continue
        seeds = sorted(set(by_method["fedavg"]) & set(by_method[method]))
        reference = np.asarray([by_method["fedavg"][seed] for seed in seeds])
        candidate = np.asarray([by_method[method][seed] for seed in seeds])
        interval = paired_interval(reference, candidate)
        p_value = float(ttest_rel(candidate, reference).pvalue) if len(seeds) > 1 else 1.0
        rows.append({
            "reference": "fedavg",
            "method": method,
            "seeds": seeds,
            "paired_macro_f1_delta_mean": interval["mean"],
            "paired_macro_f1_delta_ci95_low": interval["ci95_low"],
            "paired_macro_f1_delta_ci95_high": interval["ci95_high"],
            "paired_t_test_p_unadjusted": p_value if np.isfinite(p_value) else 1.0,
        })
    order = sorted(range(len(rows)), key=lambda i: rows[i]["paired_t_test_p_unadjusted"])
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["paired_t_test_p_unadjusted"])
        running_max = max(running_max, adjusted)
        rows[index]["paired_t_test_p_holm"] = running_max
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "sms_bert_tiny_lora_5r_3seeds.json"))
    parser.add_argument("--data-dir", default=str(DEFAULT_ROOT / "data"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument(
        "--partition-mode", choices=PARTITION_MODES,
        default="dirichlet_label_and_quantity",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--strategies", nargs="+", default=list(SUPPORTED_METHODS))
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--proximal-mu", type=float, default=1e-3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1 or args.clients < 2 or args.torch_threads < 1:
        parser.error("rounds and torch-threads must be positive; clients must be at least two")
    if args.max_rank != 4:
        parser.error("This experiment currently supports --max-rank 4")
    if any(method not in SUPPORTED_METHODS for method in args.strategies):
        parser.error(f"strategies must be selected from {SUPPORTED_METHODS}")
    torch.set_num_threads(args.torch_threads)

    data_dir = Path(args.data_dir)
    archive_path = ensure_sms_archive(data_dir / "sms+spam+collection.zip")
    archive_hash = sha256_file(archive_path)
    texts, labels = read_sms_corpus(archive_path)
    train_idx, val_idx, test_idx = split_indices(texts, labels)
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_MODEL_ID, revision=LORA_MODEL_REVISION
    )
    token_started = time.perf_counter()
    encoded = tokenizer(
        texts,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    tokenization_seconds = time.perf_counter() - token_started
    all_ids, all_mask = encoded["input_ids"], encoded["attention_mask"]
    train_input_ids, val_input_ids, test_input_ids = (
        all_ids[train_idx], all_ids[val_idx], all_ids[test_idx]
    )
    train_mask, val_mask, test_mask = (
        all_mask[train_idx], all_mask[val_idx], all_mask[test_idx]
    )
    train_y, val_y, test_y = labels[train_idx], labels[val_idx], labels[test_idx]
    prefix_started = time.perf_counter()
    feature_encoder = AutoModel.from_pretrained(
        LORA_MODEL_ID, revision=LORA_MODEL_REVISION
    )
    feature_encoder.eval()
    val_ids = _encode_frozen_prefix(
        val_input_ids, val_mask, batch_size=64, encoder=feature_encoder
    )
    test_ids = _encode_frozen_prefix(
        test_input_ids, test_mask, batch_size=64, encoder=feature_encoder
    )
    holdout_prefix_cache_seconds = time.perf_counter() - prefix_started

    results = []
    centralized = []
    base_encoder_parameter_count = None
    prefix_cache_seconds_by_seed = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        partitions = make_sms_client_partitions(
            train_y,
            args.clients,
            alpha=args.alpha,
            seed=10_000 + seed,
            partition_mode=args.partition_mode,
        )
        train_ids = torch.empty(
            (len(train_y), train_mask.shape[1], feature_encoder.config.hidden_size),
            dtype=feature_encoder.dtype,
        )
        client_prefix_start = time.perf_counter()
        for indices in partitions:
            train_ids[indices] = _encode_frozen_prefix(
                train_input_ids[indices],
                train_mask[indices],
                batch_size=64,
                encoder=feature_encoder,
            )
        prefix_cache_seconds_by_seed[str(seed)] = (
            time.perf_counter() - client_prefix_start
        )
        model = TinyBertLoRAClassifier(max_rank=args.max_rank)
        _assert_prefix_equivalence(
            model, train_ids, train_input_ids, train_mask
        )
        if base_encoder_parameter_count is None:
            base_encoder_parameter_count = model.base_encoder_parameters
        initial_state = _capture_state(model)
        centralized.append(_run_centralized(
            model, seed, args.rounds, train_ids, train_mask, train_y,
            val_ids, val_mask, val_y, test_ids, test_mask, test_y,
            args.max_rank, args.learning_rate, args.batch_size,
        ))
        for method in args.strategies:
            torch.manual_seed(seed)
            np.random.seed(seed)
            _load_state(model, initial_state)
            results.append(_run_federated(
                model, method, seed, args.rounds, train_ids, train_mask, train_y,
                val_ids, val_mask, val_y, test_ids, test_mask, test_y,
                args.clients, args.alpha, args.partition_mode, args.max_rank, args.learning_rate,
                args.proximal_mu, args.batch_size,
            ))
            print(f"completed seed={seed} method={method}", flush=True)
        del model

    summaries = []
    for method in args.strategies:
        rows = [row for row in results if row["method"] == method]
        macro = confidence_interval(
            [row["capacity_weighted_test_macro_f1"] for row in rows],
            bounds=(0.0, 1.0),
        )
        accuracy = confidence_interval(
            [row["capacity_weighted_test_accuracy"] for row in rows],
            bounds=(0.0, 1.0),
        )
        summaries.append({
            "method": method,
            "seeds": len(rows),
            "capacity_weighted_test_macro_f1": macro,
            "capacity_weighted_test_accuracy": accuracy,
            "mean_active_trainable_parameters_per_client": float(np.mean([
                row["mean_active_trainable_parameters_per_client"] for row in rows
            ])),
            "full_rank_trainable_parameters": (
                4 * 128 * args.max_rank + (2 * 128 + 2)
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
            1.0
            - row["mean_upload_bytes_per_round_float32"] / fedavg_payload
        )
    text_sets = [
        set(texts[index] for index in split)
        for split in (train_idx, val_idx, test_idx)
    ]
    payload = {
        "dataset": "sms_spam_bert_tiny_lora",
        "source_url": SMS_URL,
        "source_archive_sha256": archive_hash,
        "encoder": {
            "model_id": LORA_MODEL_ID,
            "revision": LORA_MODEL_REVISION,
            "total_parameters": base_encoder_parameter_count,
            "trainable_scope": "rank-window LoRA on final transformer block query/value projections plus 2-class head",
            "base_encoder_parameters_frozen": True,
            "max_sequence_length": args.max_length,
            "lora_rank_tiers": list(CAPACITY_TIERS),
            "lora_supernet_rank": args.max_rank,
            "lora_modules": ["final attention query", "final attention value"],
        },
        "split_protocol": "grouped stratified 20-fold split: 14 folds train, 3 validation, 3 test; exact duplicate messages kept in one split",
        "split_sizes": {
            "train": len(train_y),
            "validation": len(val_y),
            "test": len(test_y),
            "exact_text_overlap": {
                "train_validation": len(text_sets[0] & text_sets[1]),
                "train_test": len(text_sets[0] & text_sets[2]),
                "validation_test": len(text_sets[1] & text_sets[2]),
            },
        },
        "configuration": {
            "clients": args.clients,
            "dirichlet_alpha": args.alpha,
            "partition_mode": args.partition_mode,
            "rounds": args.rounds,
            "seeds": args.seeds,
            "strategies": args.strategies,
            "batch_size": args.batch_size,
            "local_epochs": 1,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "proximal_mu": args.proximal_mu,
            "torch_threads": torch.get_num_threads(),
            "validation_selection": "for each maximum client rank, select the smallest feasible rank/window within 0.002 macro-F1 of best validation macro-F1",
            "training_feature_cache": "frozen encoder prefix features computed after seeded client partitioning and cached separately per client partition; reused across strategies",
        },
        "source_script_sha256": sha256_file(Path(__file__)),
        "split_indices_sha256": hashlib.sha256(
            np.concatenate((train_idx, val_idx, test_idx)).astype("<i8").tobytes()
        ).hexdigest(),
        "centralized_results": centralized,
        "centralized_test_macro_f1": confidence_interval(
            [row["test"]["macro_f1"] for row in centralized],
            bounds=(0.0, 1.0),
        ),
        "federated_summary": summaries,
        "paired_inference": _paired_inference(results, args.strategies),
        "results": results,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": "cpu",
            "tokenization_seconds": tokenization_seconds,
            "heldout_frozen_prefix_cache_seconds": holdout_prefix_cache_seconds,
            "client_partition_prefix_cache_seconds_by_seed": prefix_cache_seconds_by_seed,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "centralized_test_macro_f1": payload["centralized_test_macro_f1"],
        "federated_summary": payload["federated_summary"],
        "paired_inference": payload["paired_inference"],
    }, indent=2))


if __name__ == "__main__":
    main()
