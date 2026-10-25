"""Self-serve tabular federated learning with budget-conditioned subnetworks.

The host defines one task and a deterministic feature map from a held-out CSV.
Clients apply the same map to their local CSVs, then train a nested prefix of a
shared MLP. Small clients send only the tensors they trained; the coordinator
updates each coordinate using only clients that actually covered it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset


INPUT_DIM = 512
MAX_DEPTH = 6
MAX_WIDTH = 512


def _normalise_value(value) -> str:
    if pd.isna(value):
        return "__missing__"
    text = str(value).strip().lower()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text or "__missing__"


def split_host_reference_validation(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None]:
    """Separate the host's preprocessing reference rows from validation rows."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the CSV.")
    if task_type == "classification":
        usable = frame.loc[frame[target_column].notna()].copy()
    elif task_type == "regression":
        numeric_target = pd.to_numeric(frame[target_column], errors="coerce")
        usable = frame.loc[numeric_target.notna()].copy()
    else:
        raise ValueError("Task type must be classification or regression.")
    indices = np.arange(len(usable))
    if len(indices) < 20:
        raise ValueError("The host CSV needs at least 20 usable target rows for setup and a 20% validation split.")
    class_values = None
    if task_type == "classification":
        labels = usable[target_column].map(_normalise_value)
        class_values = sorted(set(labels.tolist()))
        _, counts = np.unique(labels, return_counts=True)
        val_count = max(1, int(math.ceil(len(indices) * validation_fraction)))
        stratify = labels.to_numpy() if (
            len(counts) >= 2 and counts.min() >= 2
            and len(counts) <= val_count and len(counts) <= len(indices) - val_count
        ) else None
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_fraction, random_state=42, stratify=stratify
        )
    else:
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_fraction, random_state=42
        )
    return usable.iloc[train_idx].copy(), usable.iloc[val_idx].copy(), class_values


def _hash_bucket(text: str, dimensions: int = INPUT_DIM) -> tuple[int, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"fl-tab-v1").digest()
    raw = int.from_bytes(digest, "little")
    return raw % dimensions, (1.0 if (raw >> 32) & 1 else -1.0)


def fit_task_config(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    class_values: list[str] | None = None,
    max_depth: int = MAX_DEPTH,
    max_width: int = MAX_WIDTH,
) -> dict:
    """Fit a portable, deterministic preprocessing specification."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the CSV.")
    if task_type not in {"classification", "regression"}:
        raise ValueError("Task type must be classification or regression.")
    max_depth = int(max_depth)
    max_width = int(max_width)
    if not 2 <= max_depth <= MAX_DEPTH:
        raise ValueError(f"Maximum model depth must be between 2 and {MAX_DEPTH}.")
    if max_width < 16 or max_width > MAX_WIDTH or max_width % 16:
        raise ValueError(f"Maximum model width must be a multiple of 16 between 16 and {MAX_WIDTH}.")
    feature_columns = [column for column in frame.columns if column != target_column]
    if not feature_columns:
        raise ValueError("The CSV needs at least one feature column besides the target.")
    if len(feature_columns) > 500:
        raise ValueError("This starter model supports up to 500 feature columns.")
    target = frame[target_column].dropna()
    if len(target) < 10:
        raise ValueError("The host validation CSV needs at least 10 non-empty target rows.")

    encoders = {}
    for column in feature_columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        if numeric:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std()) if values.notna().any() else 1.0
            encoders[column] = {
                "kind": "numeric",
                "median": median,
                "mean": mean,
                "scale": std if math.isfinite(std) and std > 1e-8 else 1.0,
            }
        else:
            encoders[column] = {"kind": "categorical"}

    config = {
        "task_type": task_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "feature_encoders": encoders,
        "input_dim": INPUT_DIM,
        "max_depth": max_depth,
        "max_width": max_width,
        "split_seed": 42,
    }
    if task_type == "classification":
        classes = (
            [_normalise_value(v) for v in class_values]
            if class_values
            else sorted({_normalise_value(v) for v in target.tolist()})
        )
        if len(set(classes)) != len(classes):
            raise ValueError("Classification labels must be unique after case/whitespace normalization.")
        if len(classes) < 2:
            raise ValueError("Classification needs at least two target classes in the host CSV.")
        if len(classes) > 32:
            raise ValueError("This starter model supports up to 32 target classes.")
        config["classes"] = classes
        config["output_dim"] = len(classes)
    else:
        values = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 10:
            raise ValueError("Regression target must contain at least 10 numeric values.")
        config["target_mean"] = float(values.mean())
        scale = float(values.std())
        config["target_scale"] = scale if math.isfinite(scale) and scale > 1e-8 else 1.0
        config["output_dim"] = 1
    return config


def encode_frame(frame: pd.DataFrame, task_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same hash and numeric maps on the host and every client."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    required = list(task_config["feature_columns"]) + [task_config["target_column"]]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing project columns: {', '.join(missing[:12])}")

    dimensions = int(task_config.get("input_dim", INPUT_DIM))
    features = np.zeros((len(frame), dimensions), dtype=np.float32)
    for column in task_config["feature_columns"]:
        encoder = task_config["feature_encoders"][column]
        bucket, sign = _hash_bucket("column:" + column, dimensions)
        if encoder["kind"] == "numeric":
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            values = values.fillna(float(encoder["median"])).to_numpy(dtype=np.float32)
            values = (values - float(encoder["mean"])) / float(encoder["scale"])
            features[:, bucket] += np.clip(values, -5.0, 5.0).astype(np.float32) / 5.0
        else:
            values = frame[column].map(_normalise_value).to_numpy()
            for row, value in enumerate(values):
                category_bucket, category_sign = _hash_bucket("category:" + column + "=" + value, dimensions)
                features[row, category_bucket] += category_sign

    target_column = task_config["target_column"]
    if task_config["task_type"] == "classification":
        class_to_index = {name: index for index, name in enumerate(task_config["classes"])}
        normalized = frame[target_column].map(_normalise_value)
        has_target = frame[target_column].notna()
        unknown = sorted(set(normalized[has_target]) - set(class_to_index))
        if unknown:
            raise ValueError(f"Target contains labels not in the host task: {', '.join(unknown[:8])}")
        valid = has_target & normalized.isin(class_to_index)
        targets = normalized[valid].map(class_to_index).to_numpy(dtype=np.int64)
    else:
        values = pd.to_numeric(frame[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        scale = float(task_config["target_scale"])
        targets = ((values[valid].to_numpy(dtype=np.float32) - float(task_config["target_mean"])) / scale)
    return features[valid.to_numpy()], np.asarray(targets)


def validate_csv(path: str, task_config: dict) -> dict:
    """Check columns and labels before a client requests access."""
    return validate_frame(pd.read_csv(path, low_memory=False), task_config)


def validate_frame(frame: pd.DataFrame, task_config: dict) -> dict:
    """Check an already-loaded local CSV before a client requests access."""
    X, y = encode_frame(frame, task_config)
    if len(y) < 10:
        raise ValueError("The local CSV needs at least 10 usable target rows.")
    return {"rows": int(len(y)), "features": int(X.shape[1]), "classes": int(len(np.unique(y))) if task_config["task_type"] == "classification" else None}


def build_loaders_from_frame(frame: pd.DataFrame, task_config: dict, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    X, y = encode_frame(frame, task_config)
    if len(y) < 10:
        raise ValueError("The CSV needs at least 10 usable target rows to create a local split.")
    indices = np.arange(len(y))
    if task_config["task_type"] == "classification":
        _, counts = np.unique(y, return_counts=True)
        val_count = max(1, int(math.ceil(len(y) * 0.2)))
        if len(counts) >= 2 and counts.min() >= 2 and len(counts) <= val_count and len(counts) <= len(y) - val_count:
            train_idx, val_idx = next(StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=int(task_config.get("split_seed", 42))
            ).split(indices, y))
        else:
            train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=int(task_config.get("split_seed", 42)))
    else:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=int(task_config.get("split_seed", 42)))
    y_tensor = torch.from_numpy(y.astype(np.int64 if task_config["task_type"] == "classification" else np.float32))
    dataset = TensorDataset(torch.from_numpy(X), y_tensor)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx.tolist()), batch_size=max(1, min(batch_size, len(train_idx))), shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx.tolist()), batch_size=max(1, min(batch_size, len(val_idx))), shuffle=False)
    return train_loader, val_loader


def build_loaders(path: str, task_config: dict, batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    return build_loaders_from_frame(pd.read_csv(path, low_memory=False), task_config, batch_size)


def build_eval_loader_from_frame(frame: pd.DataFrame, task_config: dict, batch_size: int = 128) -> DataLoader:
    X, y = encode_frame(frame, task_config)
    if len(y) == 0:
        raise ValueError("The host validation CSV has no usable target rows.")
    y_dtype = np.int64 if task_config["task_type"] == "classification" else np.float32
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(y_dtype)))
    return DataLoader(dataset, batch_size=max(1, min(batch_size, len(dataset))), shuffle=False)


def build_eval_loader(path: str, task_config: dict, batch_size: int = 128) -> DataLoader:
    return build_eval_loader_from_frame(pd.read_csv(path, low_memory=False), task_config, batch_size)


class ElasticTabularMLP(nn.Module):
    """Prefix-elastic MLP; narrower clients activate fewer coordinates."""

    def __init__(self, model_config: dict):
        super().__init__()
        self.input_dim = int(model_config["input_dim"])
        self.max_depth = int(model_config["max_depth"])
        self.max_width = int(model_config["max_width"])
        self.output_dim = int(model_config["output_dim"])
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim if layer == 0 else self.max_width, self.max_width)
            for layer in range(self.max_depth)
        ])
        self.head = nn.Linear(self.max_width, self.output_dim)

    def forward(self, x: torch.Tensor, active_depth: int | None = None, active_width: int | None = None):
        depth = max(1, min(int(active_depth or self.max_depth), self.max_depth))
        width = max(8, min(int(active_width or self.max_width), self.max_width))
        value = x
        for index in range(depth):
            layer = self.layers[index]
            input_width = self.input_dim if index == 0 else width
            value = F.linear(value, layer.weight[:width, :input_width], layer.bias[:width])
            value = F.relu(value)
        return F.linear(value, self.head.weight[:, :width], self.head.bias)


class ElasticTabularSubnet(nn.Module):
    """Client-side model containing only the assigned prefix parameters."""

    def __init__(self, model_config: dict, active_depth: int, active_width: int):
        super().__init__()
        self.input_dim = int(model_config["input_dim"])
        self.max_depth = max(1, min(int(active_depth), int(model_config["max_depth"])))
        self.max_width = max(8, min(int(active_width), int(model_config["max_width"])))
        self.supernet_width = int(model_config["max_width"])
        self.output_dim = int(model_config["output_dim"])
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim if index == 0 else self.max_width, self.max_width)
            for index in range(self.max_depth)
        ])
        self.head = nn.Linear(self.max_width, self.output_dim)

    def forward(self, x: torch.Tensor, active_depth: int | None = None, active_width: int | None = None):
        value = x
        # The coordinator initializes each layer for max_width fan-in. A
        # narrower prefix sees fewer inputs, so compensate each hidden input
        # by sqrt(max_width / active_width). This keeps activation scale
        # comparable across capacity tiers while preserving shared weights.
        fan_in_scale = math.sqrt(self.supernet_width / max(1, self.max_width))
        for index, layer in enumerate(self.layers):
            if index:
                value = value * fan_in_scale
            value = F.relu(layer(value))
        value = value * fan_in_scale
        return self.head(value)


def project_subnet_weights(weights: dict, model_config: dict, active_depth: int, active_width: int) -> dict:
    """Select the coordinates a client needs from the coordinator supernet."""
    input_dim = int(model_config["input_dim"])
    depth = max(1, min(int(active_depth), int(model_config["max_depth"])))
    width = max(8, min(int(active_width), int(model_config["max_width"])))
    selected = {}
    for index in range(depth):
        input_width = input_dim if index == 0 else width
        for suffix in ("weight", "bias"):
            key = f"layers.{index}.{suffix}"
            if key not in weights:
                continue
            value = weights[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            selected[key] = value[:width, :input_width] if suffix == "weight" else value[:width]
    for key in ("head.weight", "head.bias"):
        if key not in weights:
            continue
        value = weights[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        selected[key] = value[:, :width] if key == "head.weight" else value
    return selected


def subnet_weights(model: ElasticTabularMLP | ElasticTabularSubnet, active_depth: int, active_width: int) -> dict:
    state = model.state_dict()
    result = {}
    for index in range(active_depth):
        input_width = model.input_dim if index == 0 else active_width
        result[f"layers.{index}.weight"] = state[f"layers.{index}.weight"][:active_width, :input_width].detach().cpu().numpy()
        result[f"layers.{index}.bias"] = state[f"layers.{index}.bias"][:active_width].detach().cpu().numpy()
    result["head.weight"] = state["head.weight"][:, :active_width].detach().cpu().numpy()
    result["head.bias"] = state["head.bias"].detach().cpu().numpy()
    return result


def load_weights(model: ElasticTabularMLP | ElasticTabularSubnet, weights: dict) -> None:
    state = model.state_dict()
    with torch.no_grad():
        for key, value in weights.items():
            if key not in state:
                continue
            tensor = torch.as_tensor(value, dtype=state[key].dtype)
            if tensor.shape == state[key].shape:
                state[key].copy_(tensor)
            elif tensor.ndim == state[key].ndim:
                sizes = tuple(min(source, target) for source, target in zip(tensor.shape, state[key].shape))
                slices = tuple(slice(0, size) for size in sizes)
                state[key][slices].copy_(tensor[slices])


def _evaluate(model, loader, task_config, device, depth, width) -> dict:
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device), depth, width).cpu()
            predictions.extend(logits.numpy())
            targets.extend(y.numpy())
    if task_config["task_type"] == "classification":
        predicted = np.asarray(predictions).argmax(axis=1)
        actual = np.asarray(targets, dtype=int)
        return {
            "validation_accuracy": float(accuracy_score(actual, predicted)),
            "validation_macro_f1": float(f1_score(
                actual,
                predicted,
                labels=list(range(len(task_config["classes"]))),
                average="macro",
                zero_division=0,
            )),
        }
    pred = np.asarray(predictions).reshape(-1) * float(task_config["target_scale"]) + float(task_config["target_mean"])
    actual = np.asarray(targets).reshape(-1) * float(task_config["target_scale"]) + float(task_config["target_mean"])
    return {"validation_rmse": float(mean_squared_error(actual, pred) ** 0.5)}


def train_client_round(
    csv_path: str,
    task_config: dict,
    model_config: dict,
    global_weights: dict,
    active_depth: int,
    active_width: int,
    epochs: int = 3,
    batch_size: int = 128,
    data_loaders: tuple[DataLoader, DataLoader] | None = None,
) -> dict:
    """Train the assigned prefix locally and return only its coordinates."""
    train_loader, val_loader = data_loaders or build_loaders(csv_path, task_config, batch_size)
    model = ElasticTabularSubnet(model_config, active_depth, active_width)
    load_weights(model, global_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() if task_config["task_type"] == "classification" else nn.MSELoss()
    model.train()
    losses = []
    for _ in range(max(1, min(int(epochs), 20))):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(X, active_depth, active_width)
            if task_config["task_type"] == "regression":
                prediction = prediction.squeeze(1)
                y = y.float()
            loss = criterion(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    metrics = _evaluate(model, val_loader, task_config, device, active_depth, active_width)
    metrics.update({
        "local_loss": float(np.mean(losses)) if losses else 0.0,
        "active_width": int(active_width),
        "active_depth": int(active_depth),
        "active_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    })
    return {
        "weights": subnet_weights(model, int(active_depth), int(active_width)),
        "num_samples": int(len(train_loader.dataset)),
        "metrics": metrics,
    }


def validate_global_model(global_weights: dict, task_config: dict, model_config: dict, loader: DataLoader) -> dict:
    model = ElasticTabularMLP(model_config)
    load_weights(model, global_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    depth = int(model_config["max_depth"])
    width = int(model_config["max_width"])
    metrics = _evaluate(model, loader, task_config, device, depth, width)
    return {f"global_{key.removeprefix('validation_')}": value for key, value in metrics.items()}


def validate_client_subnet(
    global_weights: dict,
    task_config: dict,
    model_config: dict,
    loader: DataLoader,
    active_depth: int,
    active_width: int,
) -> dict:
    """Evaluate the model a participant at this resource tier can run."""
    model = ElasticTabularSubnet(model_config, active_depth, active_width)
    load_weights(model, global_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = _evaluate(model, loader, task_config, device, active_depth, active_width)
    return {key.removeprefix("validation_"): value for key, value in metrics.items()}
