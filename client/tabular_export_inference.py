"""Run predictions from an exported self-serve tabular model bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _normalise_value(value) -> str:
    if pd.isna(value):
        return "__missing__"
    text = str(value).strip().lower()
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text or "__missing__"


def _hash_bucket(text: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"fl-tab-v1").digest()
    raw = int.from_bytes(digest, "little")
    return raw % dimensions, (1.0 if (raw >> 32) & 1 else -1.0)


def encode_features(frame: pd.DataFrame, task_config: dict) -> np.ndarray:
    """Apply the exact shared feature map without requiring a target column."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    feature_columns = list(task_config["feature_columns"])
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {', '.join(missing[:12])}")

    dimensions = int(task_config["input_dim"])
    features = np.zeros((len(frame), dimensions), dtype=np.float32)
    for column in feature_columns:
        encoder = task_config["feature_encoders"][column]
        bucket, _ = _hash_bucket("column:" + column, dimensions)
        if encoder["kind"] == "numeric":
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            values = values.fillna(float(encoder["median"])).to_numpy(dtype=np.float32)
            values = (values - float(encoder["mean"])) / float(encoder["scale"])
            features[:, bucket] += np.clip(values, -5.0, 5.0).astype(np.float32) / 5.0
        elif encoder["kind"] == "categorical":
            values = frame[column].map(_normalise_value).to_numpy()
            for row, value in enumerate(values):
                category_bucket, category_sign = _hash_bucket(
                    "category:" + column + "=" + value, dimensions
                )
                features[row, category_bucket] += category_sign
        else:
            raise ValueError(f"Unsupported encoder for column {column!r}.")
    return features


def predict(frame: pd.DataFrame, bundle_dir: Path) -> pd.DataFrame:
    task_config = json.loads((bundle_dir / "task_config.json").read_text(encoding="utf-8"))
    model_config = json.loads((bundle_dir / "model_config.json").read_text(encoding="utf-8"))
    features = encode_features(frame, task_config)
    if not len(features):
        raise ValueError("Input CSV has no rows to predict.")

    with np.load(bundle_dir / "weights.npz", allow_pickle=False) as archive:
        weights = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}

    X = torch.from_numpy(features)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(X), 4096):
            value = X[start:start + 4096]
            for layer in range(int(model_config["max_depth"])):
                value = F.relu(F.linear(
                    value,
                    weights[f"layers.{layer}.weight"],
                    weights[f"layers.{layer}.bias"],
                ))
            predictions.append(F.linear(value, weights["head.weight"], weights["head.bias"]))
    output = frame.copy()
    logits = torch.cat(predictions, dim=0)

    if task_config["task_type"] == "classification":
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        indexes = probabilities.argmax(axis=1)
        classes = task_config["classes"]
        output["prediction"] = [classes[index] for index in indexes]
        for index, label in enumerate(classes):
            output[f"probability_{label}"] = probabilities[:, index]
    else:
        values = logits.squeeze(1).cpu().numpy()
        output["prediction"] = (
            values * float(task_config["target_scale"]) + float(task_config["target_mean"])
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing the trained model's feature columns")
    parser.add_argument("--output", default="predictions.csv", help="Destination CSV path")
    args = parser.parse_args()
    bundle_dir = Path(__file__).resolve().parent
    frame = pd.read_csv(args.input, low_memory=False)
    result = predict(frame, bundle_dir)
    output_path = Path(args.output)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result):,} predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
