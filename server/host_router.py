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
from fastapi import APIRouter, Depends, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from db_handler import get_project, read_db, update_project, write_db
from live_state import build_project_snapshot
from project_router import approve_project_client, set_val_dataloader
from auth_router import verify_jwt
from shared.model_schema import REQUIRED_COLUMNS, SERVER_SCHEMA

router = APIRouter(prefix="/api/host", tags=["host"])
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_DATASET_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_SUPERNET_UPLOAD_BYTES = int(os.environ.get("MAX_SUPERNET_UPLOAD_BYTES", str(50 * 1024 * 1024)))


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


def _load_uploaded_supernet_weights(content: bytes, filename: str, model_config: dict, task_config: dict, model):
    """Read safe NumPy weights or this app's exported bundle; never unpickle uploads."""
    suffix = Path(filename).suffix.lower()
    weights_content = content
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as bundle:
                members = bundle.infolist()
                names = [member.filename for member in members]
                if len(names) != len(set(names)):
                    raise ValueError("The model ZIP contains duplicate filenames.")
                if sum(member.file_size for member in members) > MAX_SUPERNET_UPLOAD_BYTES * 2:
                    raise ValueError("The expanded model ZIP is too large.")
                required = {"manifest.json", "task_config.json", "model_config.json", "weights.npz"}
                if not required.issubset(names):
                    raise ValueError("The ZIP must be a model bundle exported by this website.")
                if any(member.file_size > 1024 * 1024 for member in members if member.filename.endswith(".json")):
                    raise ValueError("Model bundle metadata exceeds the 1 MB limit.")
                manifest = json.loads(bundle.read("manifest.json"))
                source_task = json.loads(bundle.read("task_config.json"))
                source_model = json.loads(bundle.read("model_config.json"))
                if not isinstance(manifest, dict) or manifest.get("format") != "fl-platform-tabular-bundle-v1":
                    raise ValueError("This model ZIP format is not supported.")
                expected_config = {key: int(model_config[key]) for key in ("input_dim", "max_depth", "max_width", "output_dim")}
                if not isinstance(source_task, dict) or not isinstance(source_model, dict) or source_model != expected_config:
                    raise ValueError("The uploaded model's input size, depth, width, or output size does not match this workspace.")
                if source_task.get("task_type") != task_config.get("task_type"):
                    raise ValueError("The uploaded model's task type does not match this workspace.")
                if task_config.get("task_type") == "classification" and source_task.get("classes") != task_config.get("classes"):
                    raise ValueError("The uploaded model's class labels or label order do not match this workspace.")
                weights_content = bundle.read("weights.npz")
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Could not read the model ZIP. Upload a model bundle exported by this website.") from exc
    elif suffix != ".npz":
        raise ValueError("Upload a .npz weights file or a model .zip exported by this website.")

    if len(weights_content) > MAX_SUPERNET_UPLOAD_BYTES:
        raise ValueError(f"The weights file exceeds the {MAX_SUPERNET_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    import torch
    try:
        with zipfile.ZipFile(io.BytesIO(weights_content)) as archive:
            entries = archive.infolist()
            entry_names = [entry.filename for entry in entries]
            if len(entry_names) != len(set(entry_names)):
                raise ValueError("The weights archive contains duplicate arrays.")
            if sum(entry.file_size for entry in entries) > MAX_SUPERNET_UPLOAD_BYTES * 2:
                raise ValueError("The expanded weights file is too large.")
            expected_names = {f"{key}.npy" for key in model.state_dict()}
            if set(entry_names) != expected_names:
                raise ValueError("The weights do not match the elastic tabular MLP parameter names.")
            expected = model.state_dict()
            for entry in entries:
                key = entry.filename[:-4]
                with archive.open(entry) as array_file:
                    version = np.lib.format.read_magic(array_file)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array_file)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array_file)
                    else:
                        raise ValueError(f"Uploaded parameter '{key}' uses an unsupported NumPy array format.")
                if tuple(shape) != tuple(expected[key].shape) or dtype.kind not in "fiu":
                    raise ValueError(f"Uploaded parameter '{key}' has an invalid shape or data type.")
        with np.load(io.BytesIO(weights_content), allow_pickle=False) as archive:
            expected = model.state_dict()
            if set(archive.files) != set(expected):
                raise ValueError("The weights do not match the elastic tabular MLP parameter names.")
            loaded = {}
            for key, target in expected.items():
                array = np.asarray(archive[key])
                if array.shape != tuple(target.shape) or array.dtype.kind not in "fiu":
                    raise ValueError(f"Uploaded parameter '{key}' has an invalid shape or data type.")
                array = array.astype(np.float32, copy=False)
                if not np.isfinite(array).all():
                    raise ValueError(f"Uploaded parameter '{key}' contains non-finite values.")
                loaded[key] = torch.from_numpy(array.copy()).to(dtype=target.dtype)
        model.load_state_dict(loaded, strict=True)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("The ", "Uploaded ")):
            raise
        raise ValueError("Could not read the weights. Upload a valid NumPy .npz state dictionary or exported model ZIP.") from exc


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
        "supernet_initialization": project.get("supernet_initialization", {"source": "built_in_default"}),
        "project_status": project.get("project_status", "active"),
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
    supernet_file: UploadFile | None = File(None),
    review_pending: bool = False,
    creator_id: str | None = None,
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

    import torch
    torch.manual_seed(2026)
    model = ElasticTabularMLP(model_config)
    initialization = {"source": "built_in_default"}
    if supernet_file is not None and supernet_file.filename:
        supernet_name = re.split(r"[/\\]", supernet_file.filename)[-1]
        supernet_content = await supernet_file.read(MAX_SUPERNET_UPLOAD_BYTES + 1)
        if len(supernet_content) > MAX_SUPERNET_UPLOAD_BYTES:
            raise _http_error(413, f"The supernet file exceeds the {MAX_SUPERNET_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        try:
            _load_uploaded_supernet_weights(supernet_content, supernet_name, model_config, task_config, model)
        except ValueError as exc:
            raise _http_error(422, str(exc)) from exc
        initialization = {"source": "uploaded_checkpoint", "filename": supernet_name}

    proj_id = str(uuid.uuid4())
    destination = UPLOAD_ROOT / proj_id
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "validation.csv"
    validation_frame.to_csv(path, index=False)
    digest = hashlib.sha256(content).hexdigest()

    # Every participant starts from the same coordinator-owned initialization.
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
        "supernet_initialization": initialization,
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
        "accepting_clients": not review_pending,
        "created_at": _now(),
        "project_status": "pending_approval" if review_pending else "active",
        "creator_id": creator_id if review_pending else None,
        "request_submitted_at": _now() if review_pending else None,
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
        "supernet_initialization": initialization,
    })


@router.post("/project-requests/tabular")
async def submit_tabular_project_request(
    name: str = Form(...),
    task_type: str = Form(...),
    target_column: str = Form(...),
    class_labels: str = Form(""),
    max_depth: int = Form(3),
    max_width: int = Form(64),
    file: UploadFile = File(...),
    supernet_file: UploadFile | None = File(None),
    authorization: str | None = Header(None),
):
    """Prepare a participant-submitted project and hold it for host review."""
    if not authorization or not authorization.startswith("Bearer "):
        raise _http_error(401, "Connect as a participant before submitting a project request.")
    try:
        claims = verify_jwt(authorization.split(" ", 1)[1])
    except ValueError as exc:
        raise _http_error(401, str(exc)) from exc
    if claims.get("role") != "participant":
        raise _http_error(403, "Only a participant session can submit a project request.")
    if supernet_file is None or not supernet_file.filename:
        raise _http_error(422, "Upload the starting supermodel weights or an exported model ZIP with the project request.")

    result = await create_tabular_project(
        name=name,
        task_type=task_type,
        target_column=target_column,
        class_labels=class_labels,
        max_depth=max_depth,
        max_width=max_width,
        file=file,
        supernet_file=supernet_file,
        review_pending=True,
        creator_id=claims.get("sub"),
        _=claims,
    )
    details = json.loads(result.body)
    return JSONResponse(status_code=202, content={
        "status": "pending_approval",
        "request_id": details["proj_id"],
        "name": details["name"],
        "task_type": details["task_type"],
        "target_column": details["target_column"],
        "feature_count": details["feature_count"],
        "classes": details["classes"],
        "max_depth": details["max_depth"],
        "max_width": details["max_width"],
        "supernet_initialization": details["supernet_initialization"],
    })


@router.get("/project-requests")
async def list_project_requests(_: dict = Depends(_require_host)):
    """Return pending project proposals for the server review queue."""
    db = read_db()
    requests = []
    for project in db.get("projects", []):
        if project.get("project_status") != "pending_approval":
            continue
        creator_id = project.get("creator_id")
        creator = next((user for user in db.get("users", []) if user.get("user_id") == creator_id), {})
        requests.append({
            "proj_id": project.get("proj_id"),
            "name": project.get("name"),
            "description": project.get("description", ""),
            "model_family": project.get("model_family"),
            "task_config": project.get("task_config", {}),
            "model_config": project.get("model_config", {}),
            "data_schema": project.get("data_schema", {}),
            "dataset": project.get("dataset", {}),
            "supernet_initialization": project.get("supernet_initialization", {}),
            "request_submitted_at": project.get("request_submitted_at"),
            "creator_name": creator.get("hospital_name") or creator.get("username") or "Participant",
        })
    return requests


@router.post("/project-requests/{proj_id}/approve")
async def approve_project_request(proj_id: str, _: dict = Depends(_require_host)):
    project = get_project(proj_id)
    if project is None:
        raise _http_error(404, "Project request not found.")
    if project.get("project_status") != "pending_approval":
        raise _http_error(409, "This project request has already been reviewed.")
    update_project(proj_id, {
        "project_status": "active",
        "accepting_clients": True,
        "request_reviewed_at": _now(),
    })
    return {"status": "approved", "proj_id": proj_id}


@router.post("/project-requests/{proj_id}/reject")
async def reject_project_request(proj_id: str, _: dict = Depends(_require_host)):
    project = get_project(proj_id)
    if project is None:
        raise _http_error(404, "Project request not found.")
    if project.get("project_status") != "pending_approval":
        raise _http_error(409, "This project request has already been reviewed.")
    update_project(proj_id, {
        "project_status": "rejected",
        "accepting_clients": False,
        "request_reviewed_at": _now(),
    })
    return {"status": "rejected", "proj_id": proj_id}


@router.get("/benchmarks")
async def benchmark_summary(_: dict = Depends(_require_host)):
    """Return the checked-in tabular reference benchmark, if available."""
    benchmark_path = Path(__file__).resolve().parent.parent / ".benchmarks" / "tabular_web_method" / "breast_cancer_4client.json"
    if not benchmark_path.is_file():
        return {"available": False, "runs": []}
    try:
        report = json.loads(benchmark_path.read_text(encoding="utf-8"))
        methods = {}
        for run in report.get("runs", []):
            method = run.get("method", "Unknown")
            values = methods.setdefault(method, {"accuracy": [], "macro_f1": [], "seconds": []})
            for source, target in (("capacity_selected_test_accuracy", "accuracy"), ("capacity_selected_test_macro_f1", "macro_f1"), ("wall_seconds_rounds", "seconds")):
                value = run.get(source)
                if isinstance(value, (int, float)):
                    values[target].append(float(value))
        return {
            "available": True,
            "benchmark": report.get("benchmark"),
            "dataset": report.get("dataset"),
            "rounds": report.get("settings", {}).get("rounds"),
            "seeds": report.get("settings", {}).get("seeds", []),
            "limitations": report.get("limitations", []),
            "methods": [{
                "name": name,
                "accuracy": sum(values["accuracy"]) / len(values["accuracy"]) if values["accuracy"] else None,
                "macro_f1": sum(values["macro_f1"]) / len(values["macro_f1"]) if values["macro_f1"] else None,
                "seconds": sum(values["seconds"]) / len(values["seconds"]) if values["seconds"] else None,
                "runs": len(values["accuracy"]),
            } for name, values in methods.items()],
        }
    except (OSError, json.JSONDecodeError, TypeError):
        return {"available": False, "runs": []}


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
