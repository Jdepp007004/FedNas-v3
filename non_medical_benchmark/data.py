"""Dataset loading and deterministic federated partitioning.

Only the Digits benchmark is bundled with scikit-learn.  The three UCI
datasets are downloaded only when ``download=True`` is explicitly passed to
``load_dataset`` or the command-line runner is invoked with ``--download``.
This keeps smoke tests offline and avoids silently pulling data into the
workspace.
"""

from __future__ import annotations

import os
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import DatasetSpec, get_dataset_spec


UCI_URLS = {
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "bank_marketing": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    "har": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
}


@dataclass
class DatasetBundle:
    """Preprocessed train/validation/test arrays and dataset metadata."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_val.max(), self.y_test.max()) + 1)


def _finalize_bundle(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    feature_names: Iterable[str],
    label_names: Iterable[str],
) -> DatasetBundle:
    """Split the supplied holdout into validation and final test sets."""
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        random_state=42,
        stratify=y_holdout,
    )
    return DatasetBundle(
        name=name,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_val=np.asarray(X_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=tuple(feature_names),
        label_names=tuple(label_names),
    )


def _cap_stratified(X: np.ndarray, y: np.ndarray, max_samples: Optional[int]):
    if max_samples is None or len(X) <= max_samples:
        return X, y
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=42)
    indices, _ = next(splitter.split(X, y))
    return X[indices], y[indices]


def _load_digits_bundle(spec: DatasetSpec) -> DatasetBundle:
    data = load_digits()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        data.data.astype(np.float32),
        data.target.astype(np.int64),
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_holdout = scaler.transform(X_holdout).astype(np.float32)
    return _finalize_bundle(
        spec.name,
        X_train,
        y_train,
        X_holdout,
        y_holdout,
        (f"pixel_{i}" for i in range(X_train.shape[1])),
        (str(i) for i in sorted(np.unique(data.target))),
    )


def _read_zip_member(zip_path: Path, member: str, **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(member)]
        if not matches:
            raise FileNotFoundError(f"{member!r} not found in {zip_path}")
        with archive.open(matches[0]) as stream:
            return pd.read_csv(stream, **kwargs)


def _prepare_tabular(
    X_train_df: pd.DataFrame,
    y_train_raw: pd.Series,
    X_holdout_df: pd.DataFrame,
    y_holdout_raw: pd.Series,
):
    """One-hot encode categoricals and standardize numeric columns train-only."""
    X_train_df = X_train_df.copy()
    X_holdout_df = X_holdout_df.copy()
    numeric_cols = list(X_train_df.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train_df.columns if c not in numeric_cols]

    for col in numeric_cols:
        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
        X_holdout_df[col] = pd.to_numeric(X_holdout_df[col], errors="coerce")
        fill = X_train_df[col].median()
        fill = 0.0 if pd.isna(fill) else fill
        X_train_df[col] = X_train_df[col].fillna(fill)
        X_holdout_df[col] = X_holdout_df[col].fillna(fill)

    if numeric_cols:
        scaler = StandardScaler().fit(X_train_df[numeric_cols].to_numpy(dtype=np.float32))
        train_num = scaler.transform(
            X_train_df[numeric_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        holdout_num = scaler.transform(
            X_holdout_df[numeric_cols].to_numpy(dtype=np.float32)
        ).astype(np.float32)
    else:
        train_num = np.empty((len(X_train_df), 0), dtype=np.float32)
        holdout_num = np.empty((len(X_holdout_df), 0), dtype=np.float32)

    for col in categorical_cols:
        X_train_df[col] = X_train_df[col].fillna("unknown").astype(str).str.strip().str.lower()
        X_holdout_df[col] = X_holdout_df[col].fillna("unknown").astype(str).str.strip().str.lower()
        mode = X_train_df[col].mode()
        fill = mode.iloc[0] if len(mode) else "unknown"
        X_train_df[col] = X_train_df[col].replace({"?": fill, "nan": fill})
        X_holdout_df[col] = X_holdout_df[col].replace({"?": fill, "nan": fill})

    if categorical_cols:
        combined = pd.concat(
            [X_train_df[categorical_cols], X_holdout_df[categorical_cols]], ignore_index=True
        )
        encoded = pd.get_dummies(combined, columns=categorical_cols, dtype=np.float32)
        train_cat = encoded.iloc[: len(X_train_df)].to_numpy(dtype=np.float32)
        holdout_cat = encoded.iloc[len(X_train_df) :].to_numpy(dtype=np.float32)
        feature_names = tuple(encoded.columns)
    else:
        train_cat = np.empty((len(X_train_df), 0), dtype=np.float32)
        holdout_cat = np.empty((len(X_holdout_df), 0), dtype=np.float32)
        feature_names = tuple(numeric_cols)

    X_train = np.hstack([train_num, train_cat]).astype(np.float32)
    X_holdout = np.hstack([holdout_num, holdout_cat]).astype(np.float32)

    label_encoder = LabelEncoder().fit(
        pd.concat([y_train_raw.astype(str), y_holdout_raw.astype(str)], ignore_index=True)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )
    y_train = label_encoder.transform(
        y_train_raw.astype(str).str.strip().str.lower().str.rstrip(".")
    )
    y_holdout = label_encoder.transform(
        y_holdout_raw.astype(str).str.strip().str.lower().str.rstrip(".")
    )
    return X_train, y_train, X_holdout, y_holdout, feature_names, tuple(label_encoder.classes_)


def _load_adult(zip_path: Path, spec: DatasetSpec) -> DatasetBundle:
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
    ]
    with zipfile.ZipFile(zip_path) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("adult.data"))
        test_name = next(name for name in archive.namelist() if name.endswith("adult.test"))
        with archive.open(data_name) as stream:
            train_df = pd.read_csv(stream, header=None, names=columns, na_values="?", skipinitialspace=True)
        with archive.open(test_name) as stream:
            holdout_df = pd.read_csv(
                stream, header=None, names=columns, na_values="?", skipinitialspace=True, skiprows=1
            )
    prepared = _prepare_tabular(
        train_df.drop(columns="income"), train_df["income"],
        holdout_df.drop(columns="income"), holdout_df["income"],
    )
    X_train, y_train, X_holdout, y_holdout, names, labels = prepared
    X_train, y_train = _cap_stratified(X_train, y_train, spec.max_samples)
    return _finalize_bundle(spec.name, X_train, y_train, X_holdout, y_holdout, names, labels)


def _load_bank_marketing(zip_path: Path, spec: DatasetSpec) -> DatasetBundle:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        direct = next((name for name in names if name.endswith("bank-full.csv")), None)
        if direct is not None:
            with archive.open(direct) as stream:
                df = pd.read_csv(stream, sep=";")
        else:
            # The current UCI archive stores the CSV inside a nested
            # ``bank.zip``.  Keep the outer archive as the cached download and
            # read the inner member without creating a second on-disk copy.
            nested_name = next(name for name in names if name.endswith("bank.zip"))
            with archive.open(nested_name) as nested_stream:
                nested_bytes = io.BytesIO(nested_stream.read())
            with zipfile.ZipFile(nested_bytes) as nested:
                inner_name = next(name for name in nested.namelist() if name.endswith("bank-full.csv"))
                with nested.open(inner_name) as stream:
                    df = pd.read_csv(stream, sep=";")
    X = df.drop(columns="y")
    y = df["y"]
    X_train_df, X_holdout_df, y_train, y_holdout = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    prepared = _prepare_tabular(X_train_df, y_train, X_holdout_df, y_holdout)
    X_train, y_train, X_holdout, y_holdout, names, labels = prepared
    X_train, y_train = _cap_stratified(X_train, y_train, spec.max_samples)
    return _finalize_bundle(spec.name, X_train, y_train, X_holdout, y_holdout, names, labels)


def _load_har(zip_path: Path, spec: DatasetSpec) -> DatasetBundle:
    def read_archive(archive: zipfile.ZipFile):
        def read_txt(member: str):
            name = next(name for name in archive.namelist() if name.endswith(member))
            with archive.open(name) as stream:
                return np.loadtxt(stream)

        X_train = read_txt("train/X_train.txt").astype(np.float32)
        y_train = read_txt("train/y_train.txt").astype(np.int64) - 1
        X_holdout = read_txt("test/X_test.txt").astype(np.float32)
        y_holdout = read_txt("test/y_test.txt").astype(np.int64) - 1
        try:
            feature_name = next(name for name in archive.namelist() if name.endswith("features.txt"))
            with archive.open(feature_name) as stream:
                features = tuple(
                    line.decode("utf-8").strip().split(maxsplit=1)[-1]
                    for line in stream
                )
        except (StopIteration, UnicodeDecodeError):
            features = tuple(f"feature_{i}" for i in range(X_train.shape[1]))
        return X_train, y_train, X_holdout, y_holdout, features

    with zipfile.ZipFile(zip_path) as outer:
        if any(name.endswith("train/X_train.txt") for name in outer.namelist()):
            X_train, y_train, X_holdout, y_holdout, features = read_archive(outer)
        else:
            nested_name = next(name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip"))
            with outer.open(nested_name) as nested_stream:
                nested_bytes = io.BytesIO(nested_stream.read())
            with zipfile.ZipFile(nested_bytes) as nested:
                X_train, y_train, X_holdout, y_holdout, features = read_archive(nested)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_holdout = scaler.transform(X_holdout).astype(np.float32)
    X_train, y_train = _cap_stratified(X_train, y_train, spec.max_samples)
    return _finalize_bundle(
        spec.name, X_train, y_train, X_holdout, y_holdout, features, (str(i) for i in range(6))
    )


def _ensure_archive(name: str, data_dir: Path, download: bool) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"{name}.zip"
    if archive_path.exists():
        return archive_path
    if not download:
        raise FileNotFoundError(
            f"Dataset archive not found at {archive_path}. Re-run with download=True "
            f"to fetch it from {UCI_URLS[name]}"
        )
    urllib.request.urlretrieve(UCI_URLS[name], archive_path)
    return archive_path


def load_dataset(
    name: str,
    data_dir: str | os.PathLike[str] = "data/non_medical",
    download: bool = False,
    max_samples: Optional[int] = None,
) -> DatasetBundle:
    """Load a named benchmark, optionally downloading its UCI archive."""
    spec = get_dataset_spec(name)
    if name == "digits":
        bundle = _load_digits_bundle(spec)
        if max_samples is not None and max_samples < len(bundle.X_train):
            X, y = _cap_stratified(bundle.X_train, bundle.y_train, max_samples)
            bundle.X_train, bundle.y_train = X, y
        return bundle

    archive_path = _ensure_archive(name, Path(data_dir), download)
    if name == "adult":
        bundle = _load_adult(archive_path, spec)
    elif name == "bank_marketing":
        bundle = _load_bank_marketing(archive_path, spec)
    elif name == "har":
        bundle = _load_har(archive_path, spec)
    else:  # pragma: no cover - guarded by get_dataset_spec
        raise ValueError(f"No loader registered for {name}")
    if max_samples is not None and max_samples < len(bundle.X_train):
        X, y = _cap_stratified(bundle.X_train, bundle.y_train, max_samples)
        bundle.X_train, bundle.y_train = X, y
    return bundle


def make_client_partitions(
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 2,
) -> list[np.ndarray]:
    """Create deterministic label-skewed client partitions with no empty clients."""
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    rng = np.random.default_rng(seed)
    min_samples = min(min_samples, max(1, len(y) // num_clients))
    for _ in range(80):
        partitions = [[] for _ in range(num_clients)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(indices), proportions)
            start = 0
            for client_id, count in enumerate(counts):
                partitions[client_id].extend(indices[start : start + count].tolist())
                start += count
        if min(len(part) for part in partitions) >= min_samples:
            return [np.asarray(sorted(part), dtype=np.int64) for part in partitions]

    # Extremely small or highly skewed datasets can defeat a Dirichlet draw;
    # guarantee a usable partition while preserving as much skew as possible.
    shuffled = np.arange(len(y))
    rng.shuffle(shuffled)
    fallback = [chunk.astype(np.int64) for chunk in np.array_split(shuffled, num_clients)]
    return fallback


def make_client_loader(
    bundle: DatasetBundle,
    indices: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Build a client-local loader without dropping small client batches."""
    dataset = TensorDataset(
        torch.from_numpy(bundle.X_train[indices]),
        torch.from_numpy(bundle.y_train[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
