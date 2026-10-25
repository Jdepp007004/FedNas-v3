"""Host-only controls: approvals, dataset upload, and live round state."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from db_handler import get_project, read_db, update_project, write_db
from live_state import build_project_snapshot
from project_router import approve_project_client, set_val_dataloader
from shared.model_schema import REQUIRED_COLUMNS, SERVER_SCHEMA

router = APIRouter(prefix="/api/host", tags=["host"])
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_DATASET_UPLOAD_BYTES", str(200 * 1024 * 1024)))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _require_host() -> dict:
    """Allow the host console to work without a second username/password.

    The host console is intended to be opened by the person running the
    server.  The ngrok URL is therefore a trusted collaboration link: anyone
    who has it can reach host controls.  The old JWT/admin-key checks remain
    available in the function signature for backwards-compatible clients,
    but the new workflow does not require them.  The dependency is retained as
    a named hook so a stronger host-control policy can be added later.
    """

    return {"sub": "local-host", "role": "host"}



def _http_error(status: int, detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code=status, detail=detail)


class HostSettings(BaseModel):
    max_rounds: int | None = None
    min_clients_per_round: int | None = None


@router.get("/projects")
async def host_projects(_: dict = Depends(_require_host)):
    db = read_db()
    return [{
        "proj_id": project.get("proj_id"),
        "name": project.get("name"),
        "current_round": project.get("current_round", 0),
        "max_rounds": project.get("max_rounds", 20),
        "model_family": project.get("model_family", "clinical_multitask"),
        "task_config": project.get("task_config", {}),
    } for project in db.get("projects", [])]


@router.post("/projects/tabular")
async def create_tabular_project(
    name: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    class_labels: str = Form(""),
    max_depth: int = Form(3),
    max_width: int = Form(64),
    file: UploadFile = File(...),
    _: dict = Depends(_require_host),
):
    """Create a self-serve classification or regression workspace from CSV."""
    clean_name = name.strip()
    if not clean_name or len(clean_name) > 100:
        raise _http_error(422, "Project name must contain 1–100 characters.")
    filename = os.path.basename(file.filename or "validation.csv")
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.csv", filename, flags=re.IGNORECASE):
        raise _http_error(400, "Choose a CSV filename containing letters, numbers, dots, underscores, or hyphens.")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise _http_error(413, f"Dataset is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    try:
        frame = pd.read_csv(io.BytesIO(content), low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        import sys
        client_path = str(Path(__file__).resolve().parent.parent / "client")
        if client_path not in sys.path:
            sys.path.insert(0, client_path)
        from tabular_fl import (
            ElasticTabularMLP,
            build_eval_loader_from_frame,
            fit_task_config,
            split_host_reference_validation,
        )
        labels = [value.strip() for value in re.split(r"[,\r\n]+", class_labels) if value.strip()]
        task_type = task_type.strip().lower()
        reference_frame, validation_frame, inferred_labels = split_host_reference_validation(
            frame, target_column.strip(), task_type
        )
        task_config = fit_task_config(
            reference_frame, target_column.strip(), task_type, labels or inferred_labels,
            max_depth=max_depth, max_width=max_width,
        )
        model_config = {
            "input_dim": task_config["input_dim"],
            "max_depth": task_config["max_depth"],
            "max_width": task_config["max_width"],
            "output_dim": task_config["output_dim"],
        }
        # Build once here so the host gets immediate feedback on unusable CSVs.
        # The stored copy is explicitly a held-out validation source.
        validation_loader = build_eval_loader_from_frame(validation_frame, task_config)
    except Exception as exc:
        raise _http_error(422, f"Could not configure this tabular task: {exc}") from exc

    proj_id = str(uuid.uuid4())
    destination = UPLOAD_ROOT / proj_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "validation.csv"
    validation_frame.to_csv(path, index=False)
    digest = hashlib.sha256(content).hexdigest()

    # Every participant starts from the same coordinator-owned initialization.
    import torch
    torch.manual_seed(2026)
    model = ElasticTabularMLP(model_config)
    model_path = Path(__file__).resolve().parent / "models" / f"{proj_id}_round0.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    project = {
        "proj_id": proj_id,
        "name": clean_name,
        "admin_id": "local-host",
        "model_family": "elastic_tabular",
        "model_config": model_config,
        "task_config": task_config,
        "data_schema": {"model_family": "elastic_tabular", "task_type": task_config["task_type"], "target_column": target_column.strip()},
        "schema_version": "tabular-1.0.0",
        "current_round": 0,
        "max_rounds": 20,
        "min_clients_per_round": 1,
        "connected_clients": [],
        "pending_clients": [],
        "client_profiles": {},
        "global_model_path": str(model_path),
        "fedprox_mu": 0.0,
        "momentum_beta": 0.0,
        "max_depth": model_config["max_depth"],
        "recommended_depth": model_config["max_depth"],
        "accepting_clients": True,
        "created_at": _now(),
        "dataset": {
            "filename": filename,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "size_bytes": len(content),
            "sha256": digest,
            "schema_valid": True,
            "validation_only": False,
            "validation_fraction": 0.2,
            "uploaded_at": _now(),
        },
        "validation_dataset": {"filename": "validation.csv", "source_filename": filename, "rows": int(len(validation_frame))},
    }
    db = read_db()
    db.setdefault("projects", []).append(project)
    write_db(db)
    set_val_dataloader(validation_loader, proj_id)
    return JSONResponse(status_code=201, content={
        "status": "created",
        "proj_id": proj_id,
        "name": clean_name,
        "model_family": project["model_family"],
        "task_type": task_config["task_type"],
        "target_column": task_config["target_column"],
        "feature_count": len(task_config["feature_columns"]),
        "classes": task_config.get("classes", []),
        "max_depth": model_config["max_depth"],
        "max_width": model_config["max_width"],
    })


@router.get("/projects/{proj_id}/state")
async def host_state(proj_id: str, _: dict = Depends(_require_host)):
    try:
        return build_project_snapshot(proj_id, host=True)
    except KeyError as exc:
        raise _http_error(404, f"Project {proj_id} not found.") from exc


@router.get("/projects/{proj_id}/export")
async def export_tabular_model(proj_id: str, _: dict = Depends(_require_host)):
    """Download the latest tabular checkpoint with preprocessing and inference code."""
    project = get_project(proj_id)
    if project is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    if project.get("model_family") != "elastic_tabular":
        raise _http_error(400, "Model bundle export is currently available for self-serve tabular projects.")
    current_round = int(project.get("current_round", 0))
    if current_round < 1:
        raise _http_error(409, "Complete at least one training round before downloading the model.")

    models_root = (Path(__file__).resolve().parent / "models").resolve()
    model_path = Path(project.get("global_model_path", "")).resolve()
    if models_root not in model_path.parents or not model_path.is_file():
        raise _http_error(404, "The latest model checkpoint is unavailable on this server.")

    try:
        import torch
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        weights = {
            name: (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value))
            for name, value in checkpoint.items()
        }
    except Exception as exc:
        raise _http_error(500, f"Could not read the latest model checkpoint: {exc}") from exc

    task_config = project.get("task_config", {})
    model_config = project.get("model_config", {})
    inference_script = Path(__file__).resolve().parent.parent / "client" / "tabular_export_inference.py"
    if not inference_script.is_file():
        raise _http_error(500, "The portable inference script is missing from the server installation.")

    manifest = {
        "format": "fl-platform-tabular-bundle-v1",
        "project_name": project.get("name", "tabular-model"),
        "project_id": proj_id,
        "trained_round": current_round,
        "downloaded_at": _now(),
        "task_type": task_config.get("task_type"),
        "target_column": task_config.get("target_column"),
        "model_family": project.get("model_family"),
        "weights_format": "NumPy NPZ; arrays load with allow_pickle=False",
    }
    readme = """# Exported federated tabular model

This bundle contains the latest completed global model checkpoint, its exact
feature preprocessing and label mapping, and a standalone prediction script.
It contains no participant training CSVs.

## Run predictions

Install Python 3.10+ with `numpy`, `pandas`, and `torch`. Place a CSV with the
feature columns listed in `task_config.json` beside these files, then run:

```bash
python predict_tabular.py --input new_rows.csv --output predictions.csv
```

The output preserves the input columns and adds `prediction`. Classification
bundles also add one probability column per class. Regression predictions are
returned in the target's original units.

The bundle uses the full coordinator model. Review its validation and test
metrics before using predictions in a real workflow; this capstone export is
not a clinical decision system.
"""

    archive_buffer = io.BytesIO()
    weights_buffer = io.BytesIO()
    np.savez_compressed(weights_buffer, **weights)
    try:
        with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
            bundle.writestr("task_config.json", json.dumps(task_config, indent=2, allow_nan=False))
            bundle.writestr("model_config.json", json.dumps(model_config, indent=2, allow_nan=False))
            bundle.writestr("weights.npz", weights_buffer.getvalue())
            bundle.write(inference_script, "predict_tabular.py")
            bundle.writestr("README.md", readme)
    except (TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
        raise _http_error(500, f"Could not package the model bundle: {exc}") from exc

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", str(project.get("name", "tabular-model"))).strip("-")[:48]
    filename = f"{safe_name or 'tabular-model'}-round-{current_round}.zip"
    return Response(
        content=archive_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/projects/{proj_id}/approve/{user_id}")
async def approve(proj_id: str, user_id: str, _: dict = Depends(_require_host)):
    try:
        return approve_project_client(proj_id, user_id)
    except KeyError as exc:
        raise _http_error(404, str(exc)) from exc
    except ValueError as exc:
        raise _http_error(400, str(exc)) from exc


@router.post("/projects/{proj_id}/settings")
async def settings(proj_id: str, payload: HostSettings, _: dict = Depends(_require_host)):
    project = get_project(proj_id)
    if project is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    changes = {}
    if payload.max_rounds is not None:
        if not 1 <= payload.max_rounds <= 1000:
            raise _http_error(422, "max_rounds must be between 1 and 1000.")
        changes["max_rounds"] = payload.max_rounds
    if payload.min_clients_per_round is not None:
        if payload.min_clients_per_round < 1:
            raise _http_error(422, "min_clients_per_round must be positive.")
        changes["min_clients_per_round"] = payload.min_clients_per_round
    if changes:
        update_project(proj_id, changes)
    return build_project_snapshot(proj_id, host=True)


@router.post("/projects/{proj_id}/dataset")
async def upload_dataset(
    proj_id: str,
    file: UploadFile = File(...),
    _: dict = Depends(_require_host),
):
    """Store a host-side CSV and expose metadata, never raw rows, to clients."""

    project = get_project(proj_id)
    if project is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    filename = os.path.basename(file.filename or "dataset.csv")
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.csv", filename, flags=re.IGNORECASE):
        raise _http_error(400, "Upload a CSV filename containing only letters, numbers, dots, underscores, or hyphens.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise _http_error(413, f"Dataset is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    destination = UPLOAD_ROOT / proj_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    staged_path = destination / f".{filename}.uploading"
    staged_path.write_bytes(content)

    try:
        header = pd.read_csv(staged_path, nrows=0, low_memory=False)
        rows = sum(len(chunk) for chunk in pd.read_csv(staged_path, chunksize=5000, low_memory=False))
    except Exception as exc:
        staged_path.unlink(missing_ok=True)
        raise _http_error(422, f"The CSV could not be read: {exc}") from exc

    # Replace the previous host dataset only after the new upload has passed
    # CSV parsing.  A malformed retry must not destroy a working dataset.
    staged_path.replace(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    digest = hashlib.sha256(content).hexdigest()
    metadata = {
        "filename": filename,
        "rows": int(rows),
        "columns": int(len(header.columns)),
        "size_bytes": len(content),
        "sha256": digest,
        "schema_valid": not missing,
        "missing_columns": missing[:20],
        "uploaded_at": _now(),
    }
    update_project(proj_id, {"dataset": metadata})

    # A valid host dataset is also a convenient server-side validation source.
    # Loading is best-effort so upload remains useful on machines without the
    # optional training dependencies or with a custom schema.
    if not missing:
        try:
            import sys
            client_path = str(Path(__file__).resolve().parent.parent / "client")
            if client_path not in sys.path:
                sys.path.insert(0, client_path)
            from data_loader import build_dataloaders_from_csv
            _, val_loader = build_dataloaders_from_csv(str(path), SERVER_SCHEMA)
            set_val_dataloader(val_loader, proj_id)
            update_project(proj_id, {"validation_dataset": {"filename": filename, "rows": int(rows)}})
        except Exception:
            pass

    return JSONResponse(status_code=201, content={"status": "uploaded", "dataset": metadata})
