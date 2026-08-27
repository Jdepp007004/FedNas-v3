"""Host-only controls: approvals, dataset upload, and live round state."""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import re
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, Header, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth_router import verify_host_jwt
from db_handler import get_project, read_db, update_project
from live_state import build_project_snapshot
from project_router import approve_project_client, set_val_dataloader
from shared.model_schema import REQUIRED_COLUMNS, SERVER_SCHEMA

router = APIRouter(prefix="/api/host", tags=["host"])
UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_DATASET_UPLOAD_BYTES", str(200 * 1024 * 1024)))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _require_host(
    authorization: str | None = Header(None),
    x_admin_key: str | None = Header(None),
) -> dict:
    """Accept the new host JWT and preserve the legacy local admin key."""

    if authorization and authorization.startswith("Bearer "):
        try:
            claims = verify_host_jwt(authorization.split(" ", 1)[1])
            return claims
        except ValueError as exc:
            raise _http_error(401, str(exc)) from exc

    expected = os.environ.get("JWT_SECRET", "dev_secret_change_in_production")
    if x_admin_key and hmac.compare_digest(x_admin_key, expected):
        return {"sub": "legacy-host", "role": "host"}
    raise _http_error(401, "Host login required.")


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
    } for project in db.get("projects", [])]


@router.get("/projects/{proj_id}/state")
async def host_state(proj_id: str, _: dict = Depends(_require_host)):
    try:
        return build_project_snapshot(proj_id, host=True)
    except KeyError as exc:
        raise _http_error(404, f"Project {proj_id} not found.") from exc


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
            set_val_dataloader(val_loader)
            update_project(proj_id, {"validation_dataset": {"filename": filename, "rows": int(rows)}})
        except Exception:
            pass

    return JSONResponse(status_code=201, content={"status": "uploaded", "dataset": metadata})
