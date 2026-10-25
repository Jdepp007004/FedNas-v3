"""Small helpers for reproducible benchmark run manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _source_tree_sha256() -> str:
    source_root = Path(__file__).resolve().parent
    files = sorted(source_root.glob("*.py"))
    manifest = source_root / "EXPERIMENT_MANIFEST.json"
    if manifest.is_file():
        files.append(manifest)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _dataset_digest(dataset: Any) -> str:
    digest = hashlib.sha256()
    digest.update(type(dataset).__qualname__.encode("utf-8"))
    digest.update(str(len(dataset)).encode("ascii"))
    for index in range(len(dataset)):
        item = dataset[index]
        if isinstance(item, (tuple, list)):
            values = item
        else:
            values = (item,)
        for value in values:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(repr(value.shape).encode("ascii"))
                digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def source_file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_array_run_manifest(
    dataset: str,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    arguments: dict[str, Any],
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    split_hashes = {
        name: _array_digest([features, labels])
        for name, (features, labels) in splits.items()
    }
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split_sizes": {
            name: int(len(labels)) for name, (_, labels) in splits.items()
        },
        "split_sha256": split_hashes,
        "source_archive_sha256": source_file_sha256(source_path),
        "benchmark_source_sha256": _source_tree_sha256(),
        "arguments": arguments,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "xpu_available": bool(
                hasattr(torch, "xpu") and torch.xpu.is_available()
            ),
        },
        "git": _git_state(),
    }


def build_dataset_run_manifest(
    dataset: str,
    splits: dict[str, Any],
    arguments: dict[str, Any],
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    split_hashes = {name: _dataset_digest(value) for name, value in splits.items()}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "split_sizes": {name: int(len(value)) for name, value in splits.items()},
        "split_sha256": split_hashes,
        "source_archive_sha256": source_file_sha256(source_path),
        "benchmark_source_sha256": _source_tree_sha256(),
        "arguments": arguments,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "xpu_available": bool(
                hasattr(torch, "xpu") and torch.xpu.is_available()
            ),
        },
        "git": _git_state(),
    }
