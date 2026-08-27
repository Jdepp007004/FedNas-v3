"""Create four reproducible, schema-compatible client partitions.

The checked-in ``data/full_dataset.csv`` is preferred over the raw
``SEER_cleaned.csv`` because it already contains the feature names expected by
the platform's client data loader.  Partitions are balanced by vital status
when that column is present and never leave the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_dataset(source: str | Path = "data/full_dataset.csv", output_dir: str | Path = "split_data", parts: int = 4) -> list[Path]:
    if parts != 4:
        raise ValueError("This collaboration setup expects exactly 4 client parts.")
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")
    frame = pd.read_csv(source_path, low_memory=False)
    if frame.empty:
        raise ValueError("The source dataset is empty.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    buckets = [[] for _ in range(parts)]
    if "vital_status" in frame.columns:
        for _, group in frame.groupby("vital_status", dropna=False, sort=True):
            indices = group.index.to_numpy()
            rng.shuffle(indices)
            for part, chunk in enumerate(np.array_split(indices, parts)):
                buckets[part].extend(chunk.tolist())
    else:
        indices = frame.index.to_numpy()
        rng.shuffle(indices)
        buckets = [chunk.tolist() for chunk in np.array_split(indices, parts)]

    paths = []
    for number, indices in enumerate(buckets, start=1):
        part = frame.loc[indices].sample(frac=1, random_state=42 + number).reset_index(drop=True)
        destination = output_path / f"client_{number}.csv"
        part.to_csv(destination, index=False)
        paths.append(destination)

    manifest = {
        "source": str(source_path),
        "parts": len(paths),
        "seed": 42,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "files": [{"name": path.name, "rows": int(len(pd.read_csv(path)))} for path in paths],
        "privacy_note": "Client CSV files are intended to remain on their owner's machine; only model updates and metadata are sent.",
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the platform-ready dataset into four client CSVs.")
    parser.add_argument("--source", default="data/full_dataset.csv")
    parser.add_argument("--output-dir", default="split_data")
    args = parser.parse_args()
    for path in split_dataset(args.source, args.output_dir):
        print(f"[+] {path}")
