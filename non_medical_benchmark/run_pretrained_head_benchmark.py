"""Federated classifier-head adaptation on frozen MiniLM SMS embeddings.

This is deliberately a frozen-encoder experiment: only the elastic MLP head
is optimized by the centralized/federated benchmark. It must not be described
as transformer or LoRA fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel

from .config import DatasetSpec
from .data import DatasetBundle
from .federated import run_federated_experiment
from .models import build_model
from .provenance import build_array_run_manifest, source_file_sha256
from .run_benchmark import _summary
from .stats import confidence_interval, paired_interval


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_PARAMETER_COUNT = 22_713_216
SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_ROOT = Path(__file__).resolve().parent / ".benchmarks" / "pretrained_sms"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_sms_archive(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(
            SMS_URL, headers={"User-Agent": "federated-research-repro/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return path


def read_sms_corpus(path: Path) -> tuple[list[str], np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("SMSSpamCollection")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one SMSSpamCollection member, found {candidates}")
        raw = archive.read(candidates[0]).decode("utf-8", errors="replace")
    texts: list[str] = []
    labels: list[int] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        label, separator, text = line.partition("\t")
        if not separator or label not in {"ham", "spam"}:
            raise ValueError(f"Malformed SMS corpus row {line_number}")
        texts.append(text.strip())
        labels.append(int(label == "spam"))
    if len(texts) != 5574:
        raise ValueError(f"Expected the official 5,574-row corpus, found {len(texts)}")
    return texts, np.asarray(labels, dtype=np.int64)


def split_indices(
    texts: list[str], labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make approximate 70/15/15 splits without exact-message leakage."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal length")
    text_labels: dict[str, int] = {}
    for text, label in zip(texts, labels):
        previous = text_labels.setdefault(text, int(label))
        if previous != int(label):
            raise ValueError("An exact duplicate SMS has conflicting labels")

    # Twenty stratified group folds permit a 14/3/3 fold assignment while
    # keeping every exact duplicate text in exactly one split.
    fold_id = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    dummy_X = np.zeros((len(labels), 1), dtype=np.uint8)
    for fold, (_, held_out) in enumerate(
        splitter.split(dummy_X, labels, groups=np.asarray(texts, dtype=object))
    ):
        fold_id[held_out] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("Grouped split failed to assign every row")
    train_idx = np.flatnonzero(fold_id < 14)
    val_idx = np.flatnonzero((fold_id >= 14) & (fold_id < 17))
    test_idx = np.flatnonzero(fold_id >= 17)
    split_texts = [set(texts[index] for index in split) for split in (train_idx, val_idx, test_idx)]
    if any(
        split_texts[left] & split_texts[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Exact SMS text leakage across grouped splits")
    return train_idx, val_idx, test_idx


def load_or_encode(
    texts: list[str],
    labels: np.ndarray,
    archive_sha256: str,
    cache_dir: Path,
) -> tuple[DatasetBundle, dict]:
    embedding_started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"embeddings_{archive_sha256[:12]}_{MODEL_REVISION[:12]}_grouped_split42.npz"
    )
    train_idx, val_idx, test_idx = split_indices(texts, labels)
    split_text_sets = [
        set(texts[index] for index in split)
        for split in (train_idx, val_idx, test_idx)
    ]
    text_overlap_counts = {
        "train_validation": len(split_text_sets[0] & split_text_sets[1]),
        "train_test": len(split_text_sets[0] & split_text_sets[2]),
        "validation_test": len(split_text_sets[1] & split_text_sets[2]),
    }
    legacy_cache_path = cache_dir / (
        f"embeddings_{archive_sha256[:12]}_{MODEL_REVISION[:12]}_split42.npz"
    )
    cache_migration = False
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            X_train, X_val, X_test = (
                cached["X_train"], cached["X_val"], cached["X_test"]
            )
            y_train, y_val, y_test = (
                cached["y_train"], cached["y_val"], cached["y_test"]
            )
        encoder_parameters = MODEL_PARAMETER_COUNT
        encoder_file_hashes = None
        cache_hit = True
    elif legacy_cache_path.exists():
        # Reassemble the already-computed row-wise embeddings from the earlier
        # random split, then apply the exact-text grouped split. Embeddings do
        # not depend on labels or on other examples, so this is split-safe.
        old_train, remaining = train_test_split(
            np.arange(len(labels)),
            test_size=0.30,
            random_state=42,
            stratify=labels,
        )
        old_val, old_test = train_test_split(
            remaining,
            test_size=0.50,
            random_state=43,
            stratify=labels[remaining],
        )
        with np.load(legacy_cache_path, allow_pickle=False) as cached:
            old_X_train = cached["X_train"]
            old_X_val = cached["X_val"]
            old_X_test = cached["X_test"]
        if (len(old_train), len(old_val), len(old_test)) != (
            len(old_X_train), len(old_X_val), len(old_X_test)
        ):
            raise ValueError("Legacy embedding cache does not match the expected split")
        embeddings = np.empty(
            (len(labels), old_X_train.shape[1]), dtype=np.float32
        )
        embeddings[old_train] = old_X_train
        embeddings[old_val] = old_X_val
        embeddings[old_test] = old_X_test
        X_train, X_val, X_test = (
            embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
        )
        y_train, y_val, y_test = (
            labels[train_idx], labels[val_idx], labels[test_idx]
        )
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        encoder_parameters = MODEL_PARAMETER_COUNT
        encoder_file_hashes = None
        cache_hit = True
        cache_migration = True
    else:
        encoder = SentenceTransformer(
            MODEL_ID, revision=MODEL_REVISION, device="cpu"
        )
        encoder.max_seq_length = 128
        encoder.eval()
        encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
        if encoder_parameters != MODEL_PARAMETER_COUNT:
            raise RuntimeError(
                f"Pinned encoder parameter count changed: {encoder_parameters} "
                f"!= {MODEL_PARAMETER_COUNT}"
            )
        embeddings = encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        try:
            from huggingface_hub import try_to_load_from_cache

            model_files = {}
            for filename in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "modules.json",
                "sentence_bert_config.json",
            ):
                local_file = try_to_load_from_cache(
                    MODEL_ID, filename, revision=MODEL_REVISION
                )
                if isinstance(local_file, str):
                    model_files[filename] = sha256_file(Path(local_file))
            encoder_file_hashes = model_files
        except Exception:
            encoder_file_hashes = {}

        X_train, X_val, X_test = (
            embeddings[train_idx], embeddings[val_idx], embeddings[test_idx]
        )
        y_train, y_val, y_test = (
            labels[train_idx], labels[val_idx], labels[test_idx]
        )
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        del encoder
        cache_hit = False

    # Record file hashes on a cache hit too, without loading the encoder.
    if encoder_file_hashes is None:
        from huggingface_hub import try_to_load_from_cache

        encoder_file_hashes = {}
        for filename in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "modules.json",
            "sentence_bert_config.json",
        ):
            local_file = try_to_load_from_cache(
                MODEL_ID, filename, revision=MODEL_REVISION
            )
            if isinstance(local_file, str):
                encoder_file_hashes[filename] = sha256_file(Path(local_file))

    bundle = DatasetBundle(
        name="sms_spam_minilm_frozen_embeddings",
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_val=np.asarray(X_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        feature_names=tuple(f"embedding_{i}" for i in range(X_train.shape[1])),
        label_names=("ham", "spam"),
    )
    info = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "encoder_frozen": True,
        "encoder_parameters": encoder_parameters,
        "encoder_parameters_trainable_during_federated_runs": 0,
        "embedding_dimension": int(bundle.input_dim),
        "max_sequence_length": 128,
        "embedding_normalization": False,
        "encoder_files_sha256": encoder_file_hashes,
        "embedding_cache_path": str(cache_path),
        "embedding_cache_sha256": sha256_file(cache_path),
        "cache_hit": cache_hit,
        "cache_migrated_from_pre_grouped_split": cache_migration,
        "embedding_generation_seconds": (
            0.0 if cache_hit else time.perf_counter() - embedding_started
        ),
        "split_indices_sha256": hashlib.sha256(
            np.concatenate((train_idx, val_idx, test_idx)).astype("<i8").tobytes()
        ).hexdigest(),
        "exact_text_overlap_counts": text_overlap_counts,
    }
    return bundle, info


def fit_centralized_head(bundle: DatasetBundle, spec: DatasetSpec, rounds: int, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    model = build_model("elastic", bundle.input_dim, spec.num_classes, spec.max_depth, spec.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    counts = np.bincount(bundle.y_train, minlength=spec.num_classes).astype(np.float32)
    weights = torch.as_tensor(
        len(bundle.y_train) / np.maximum(counts * spec.num_classes, 1.0),
        dtype=torch.float32,
    )
    X = torch.from_numpy(bundle.X_train)
    y = torch.from_numpy(bundle.y_train)
    batch_size = spec.batch_size
    model.train()
    for _ in range(rounds * spec.local_epochs):
        for indices in torch.randperm(len(y)).split(batch_size):
            optimizer.zero_grad()
            logits = model.forward_full(X[indices])
            loss = torch.nn.functional.cross_entropy(logits, y[indices], weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        val_pred = model.forward_full(torch.from_numpy(bundle.X_val)).argmax(1).numpy()
        test_pred = model.forward_full(torch.from_numpy(bundle.X_test)).argmax(1).numpy()
    return {
        "seed": seed,
        "training_epochs": rounds * spec.local_epochs,
        "val_accuracy": float(accuracy_score(bundle.y_val, val_pred)),
        "val_macro_f1": float(f1_score(bundle.y_val, val_pred, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(bundle.y_test, test_pred)),
        "test_macro_f1": float(f1_score(bundle.y_test, test_pred, average="macro", zero_division=0)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _paired_family(
    results: list[dict],
    reference: str,
    comparisons: list[str],
    family_name: str,
) -> list[dict]:
    by_method = {
        method: {
            row["seed"]: row["final"]["selected_test_macro_f1"]
            for row in results
            if row["strategy"] == method
        }
        for method in [reference, *comparisons]
    }
    rows = []
    for method in comparisons:
        shared = sorted(set(by_method[reference]) & set(by_method[method]))
        baseline = [by_method[reference][seed] for seed in shared]
        candidate = [by_method[method][seed] for seed in shared]
        interval = paired_interval(baseline, candidate)
        p_value = (
            float(ttest_rel(candidate, baseline).pvalue)
            if len(shared) > 1
            else 1.0
        )
        if not np.isfinite(p_value):
            p_value = 1.0
        rows.append(
            {
                "family": family_name,
                "reference": reference,
                "method": method,
                "seeds": shared,
                "paired_macro_f1_delta_mean": interval["mean"],
                "paired_macro_f1_delta_ci95_low": interval["ci95_low"],
                "paired_macro_f1_delta_ci95_high": interval["ci95_high"],
                "paired_t_test_p_unadjusted": p_value,
            }
        )
    # Holm step-down family-wise error correction.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["paired_t_test_p_unadjusted"])
    running_max = 0.0
    count = len(rows)
    for rank, index in enumerate(order):
        raw = rows[index]["paired_t_test_p_unadjusted"]
        adjusted = min(1.0, (count - rank) * raw)
        running_max = max(running_max, adjusted)
        rows[index]["paired_t_test_p_holm"] = running_max
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "sms_minilm_head_10r_5seeds.json"))
    parser.add_argument("--data-dir", default=str(DEFAULT_ROOT / "data"))
    parser.add_argument("--cache-dir", default=str(DEFAULT_ROOT / "cache"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "fedavg", "fedprox", "heterofl", "fjord", "fedrolex", "maxnet",
            "cc_efl", "cc_efl_no_prox", "cc_efl_uniform", "cc_efl_random_window",
        ],
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.clients < 2 or args.torch_threads < 1:
        parser.error("rounds and torch-threads must be positive; clients must be at least two")
    torch.set_num_threads(args.torch_threads)

    data_dir = Path(args.data_dir)
    archive_path = ensure_sms_archive(data_dir / "sms+spam+collection.zip")
    archive_hash = source_file_sha256(archive_path)
    texts, labels = read_sms_corpus(archive_path)
    bundle, encoder = load_or_encode(
        texts, labels, archive_hash or "missing", Path(args.cache_dir)
    )
    spec = DatasetSpec(
        name=bundle.name,
        description="UCI SMS Spam binary text classification with frozen MiniLM embeddings.",
        source_url=SMS_URL,
        task="binary text classification",
        num_classes=2,
        max_samples=None,
        max_depth=3,
        hidden_dim=128,
        widths=(32, 64, 96, 128),
        clients=args.clients,
        dirichlet_alpha=args.alpha,
        batch_size=128,
        local_epochs=1,
        learning_rate=1e-3,
        fedprox_mu=1e-3,
        selection_metric="val_macro_f1",
        class_weighting="balanced",
    )

    results = []
    centralized = []
    for seed in args.seeds:
        centralized.append(fit_centralized_head(bundle, spec, args.rounds, seed))
        for strategy in args.strategies:
            results.append(
                run_federated_experiment(
                    bundle,
                    spec,
                    variant="elastic",
                    rounds=args.rounds,
                    seed=seed,
                    num_clients=args.clients,
                    strategy=strategy,
                    optimizer_name="adam",
                )
            )

    manifest = build_array_run_manifest(
        dataset=bundle.name,
        splits={
            "train": (bundle.X_train, bundle.y_train),
            "validation": (bundle.X_val, bundle.y_val),
            "test": (bundle.X_test, bundle.y_test),
        },
        source_path=archive_path,
        arguments={
            "dataset_spec": spec.__dict__,
            "encoder": encoder,
            "rounds": args.rounds,
            "clients": args.clients,
            "dirichlet_alpha": args.alpha,
            "seeds": args.seeds,
            "strategies": args.strategies,
            "torch_intraop_threads": torch.get_num_threads(),
            "centralized_head_training_epochs": args.rounds * spec.local_epochs,
            "fine_tuning_scope": "classifier head only; sentence encoder frozen",
        },
    )
    federated_summary = _summary(results)
    full_fedavg_head_parameters = next(
        row["full_model_parameters_mean"]
        for row in federated_summary
        if row["strategy"] == "fedavg"
    )
    for row in federated_summary:
        selected_head_parameters = row["selected_deployed_parameters_mean"]
        row["selected_head_parameters_mean"] = selected_head_parameters
        row["frozen_encoder_parameters"] = MODEL_PARAMETER_COUNT
        row["total_selected_model_parameters_including_encoder"] = (
            MODEL_PARAMETER_COUNT + selected_head_parameters
        )
        row["total_model_parameter_saving_vs_full_fedavg_including_encoder"] = (
            1.0
            - (MODEL_PARAMETER_COUNT + selected_head_parameters)
            / (MODEL_PARAMETER_COUNT + full_fedavg_head_parameters)
        )
        row["resource_saving_scope"] = (
            "head only; frozen encoder inference/storage is excluded from train-side metrics"
        )

    comparison_methods = [method for method in args.strategies if method != "fedavg"]
    paired_inference = {
        "primary_endpoint": "capacity-selected test macro-F1",
        "confidence_intervals": "unadjusted two-sided paired Student-t intervals over seeds",
        "p_values": "paired two-sided Student-t tests with Holm family-wise correction",
        "family_alpha": 0.05,
        "families": [
            _paired_family(results, "fedavg", comparison_methods, "all strategies vs FedAvg"),
            _paired_family(
                results,
                "cc_efl",
                [
                    method
                    for method in ("cc_efl_no_prox", "cc_efl_uniform", "cc_efl_random_window")
                    if method in args.strategies
                ],
                "CC-EFL component/window ablations",
            ),
        ],
    }
    centralized_macro_f1 = [item["test_macro_f1"] for item in centralized]

    payload = {
        "dataset": bundle.name,
        "source_url": SMS_URL,
        "source_archive_sha256": archive_hash,
        "encoder": encoder,
        "data_counts": {
            "total": len(texts),
            "train": len(bundle.y_train),
            "validation": len(bundle.y_val),
            "test": len(bundle.y_test),
            "class_counts_total": {
                "ham": int(np.sum(labels == 0)),
                "spam": int(np.sum(labels == 1)),
            },
            "class_counts_by_split": {
                name: {
                    "ham": int(np.sum(split_labels == 0)),
                    "spam": int(np.sum(split_labels == 1)),
                }
                for name, split_labels in (
                    ("train", bundle.y_train),
                    ("validation", bundle.y_val),
                    ("test", bundle.y_test),
                )
            },
        },
        "split_protocol": "grouped stratified 20-fold split: folds 0-13 train, 14-16 validation, 17-19 test; exact duplicate SMS texts are kept in one split; client partitions are seeded Dirichlet label-skew draws from train only",
        "metric": "macro-F1 primary; accuracy secondary",
        "centralized_full_head_results": centralized,
        "centralized_full_head_macro_f1_summary": confidence_interval(
            centralized_macro_f1, bounds=(0.0, 1.0)
        ),
        "federated_summary": federated_summary,
        "paired_inference": paired_inference,
        "results": results,
        "run_manifest": manifest,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
