"""
server/project_router.py
M3: /api/projects/* endpoints + background round lifecycle.
Owner: Sunishka Sarkar
"""

import os
import sys
import threading
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import APIRouter, Depends, BackgroundTasks, Header, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from db_handler import read_db, write_db, get_project, update_project, append_round_history  # noqa: E402
from auth_router import verify_jwt  # noqa: E402
from aggregation import aggregate_fedavg, update_with_momentum, validate_global_model, EmptyRoundError  # noqa: E402
from nas_controller import evaluate_architecture_candidates  # noqa: E402
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA  # noqa: E402
from resource_planner import depth_for_profile, normalise_hardware_profile, build_round_plan  # noqa: E402

# ── Phase 2 infrastructure (graceful degradation when not configured) ─────────
try:
    from redis_state import get_state as _get_redis_state
    _redis_state = _get_redis_state()
except Exception:  # noqa: BLE001
    _redis_state = None

try:
    from round_state import RoundState, RoundStateMachine
    _HAVE_STATE_MACHINE = True
except Exception:  # noqa: BLE001
    _HAVE_STATE_MACHINE = False

try:
    from metrics import ACTIVE_CLIENTS, PENDING_UPDATES
    _HAVE_METRICS = True
except Exception:  # noqa: BLE001
    _HAVE_METRICS = False

# ── Path to models directory ──────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ── In-memory buffers (keyed by proj_id) ────────────────────────────────────
_pending_updates: dict = {}   # proj_id → list of update dicts
_velocity_state:  dict = {}   # proj_id → velocity dict for momentum
_buffer_lock = threading.Lock()

# ── Server-side validation dataloader (created at startup by main.py) ────────
_val_dataloader = None


def set_val_dataloader(dl):
    global _val_dataloader
    _val_dataloader = dl


def _dispatch_round(
    proj_id: str,
    updates_snapshot: list,
    db_snapshot: dict,
    background_tasks,
) -> None:
    """
    Route the round_lifecycle call to Celery (if configured) or
    FastAPI BackgroundTask (fallback — existing behaviour).
    """
    if os.getenv("CELERY_BROKER_URL"):
        try:
            from tasks import dispatch_round as _celery_task
            _celery_task.delay(proj_id, updates_snapshot, db_snapshot)
            return
        except Exception as e:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "Celery dispatch failed, falling back to BackgroundTask: %s", e
            )
    # ── Fallback: FastAPI BackgroundTask (original behaviour) ─────────────────
    background_tasks.add_task(round_lifecycle, proj_id, updates_snapshot, db_snapshot)



# ─── JWT Dependency ───────────────────────────────────────────────────────────

def _get_current_user(authorization: str = Header(None)) -> dict:  # noqa: B008
    if not authorization or not authorization.startswith("Bearer "):
        raise _http_error(401, "Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        return verify_jwt(token)
    except ValueError as e:
        raise _http_error(401, str(e))


def _http_error(status: int, detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code=status, detail=detail)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class JoinRequest(BaseModel):
    hardware_profile: dict


class UpdateRequest(BaseModel):
    round_id:     int
    active_depth: int
    weights:      dict   # encrypted payload
    num_samples:  int
    metrics:      dict


class ResourceUpdateRequest(BaseModel):
    hardware_profile: dict


class DatasetMetaRequest(BaseModel):
    filename: str
    rows: int = 0
    columns: int = 0
    size_bytes: int = 0
    sha256: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("")
async def list_projects(current_user: dict = Depends(_get_current_user)):  # noqa: B008
    """GET /api/projects"""
    db = read_db()
    user_id = current_user["sub"]
    user = next((u for u in db["users"] if u["user_id"] == user_id), None)
    approved = set(user.get("approved_projects", [])) if user else set()

    visible = []
    for proj in db.get("projects", []):
        if proj.get("accepting_clients") or proj["proj_id"] in approved:
            entry = {k: proj[k] for k in proj if k != "global_model_path"}
            entry["i_am_connected"] = user_id in proj.get("connected_clients", [])
            entry["i_am_pending"] = user_id in proj.get("pending_clients", [])
            visible.append(entry)
    return JSONResponse(status_code=200, content=visible)


@router.get("/{proj_id}")
async def get_project_details(proj_id: str, current_user: dict = Depends(_get_current_user)):  # noqa: B008
    """GET /api/projects/{proj_id} — project details for a logged-in client."""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    entry = {k: proj[k] for k in proj if k not in {"global_model_path", "client_profiles"}}
    user_id = current_user["sub"]
    entry["i_am_connected"] = user_id in proj.get("connected_clients", [])
    entry["i_am_pending"] = user_id in proj.get("pending_clients", [])
    return JSONResponse(status_code=200, content=entry)


@router.post("/{proj_id}/join")
async def join_project(
    proj_id: str,
    payload: JoinRequest,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """POST /api/projects/{proj_id}/join"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    if not proj.get("accepting_clients", True):
        raise _http_error(403, "Project is not accepting new clients.")

    user_id = current_user["sub"]
    hardware_profile = normalise_hardware_profile(payload.hardware_profile)
    recommended_depth = depth_for_profile(hardware_profile, MODEL_CONFIG["max_depth"])

    # Add to pending_clients if not already there or in connected
    pending = proj.get("pending_clients", [])
    connected = proj.get("connected_clients", [])
    profile_map = proj.get("client_profiles", {}) or {}
    previous = profile_map.get(user_id, {}) or {}
    profile_map[user_id] = {
        **previous,
        "hardware_profile": hardware_profile,
        "last_seen": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if user_id not in pending and user_id not in connected:
        pending.append(user_id)
    update_project(proj_id, {"pending_clients": pending, "client_profiles": profile_map})

    return JSONResponse(status_code=200, content={
        "status":           "pending_approval",
        "recommended_depth": recommended_depth,
        "required_schema":  proj.get("data_schema", SERVER_SCHEMA),
        "schema_version":   proj.get("schema_version", "1.0.0"),
    })


@router.get("/{proj_id}/model")
async def get_global_model(
    proj_id: str,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """GET /api/projects/{proj_id}/model"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    user_id = current_user["sub"]
    if user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "You are not an approved participant in this project.")

    model_path = proj.get("global_model_path")
    if model_path and os.path.exists(model_path):
        weights_raw = torch.load(model_path, map_location="cpu")
        weights_json = {k: (v.numpy().tolist() if hasattr(v, 'numpy') else v.tolist())
                        for k, v in weights_raw.items()}
    else:
        # Return empty weights dict for the first round
        weights_json = {}

    # Per-client depth assignment.  The same explainable plan is returned to
    # the browser console, while this endpoint returns only this client's
    # assigned depth.
    db = read_db()
    profiles = proj.get("client_profiles", {}) or {}
    own_profile = profiles.get(user_id, {}).get("hardware_profile", {})
    active_depth = depth_for_profile(own_profile, MODEL_CONFIG["max_depth"])
    users = {item.get("user_id"): item for item in db.get("users", [])}
    round_plan = build_round_plan(proj, users)

    return JSONResponse(status_code=200, content={
        "round":        proj.get("current_round", 0),
        "active_depth": active_depth,
        "weights":      weights_json,
        "round_plan":   round_plan,
    })


@router.post("/{proj_id}/resources")
async def update_resources(
    proj_id: str,
    payload: ResourceUpdateRequest,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """Update observed capacity and the client's explicit contribution cap."""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    user_id = current_user["sub"]
    if user_id not in proj.get("pending_clients", []) and user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "Request access to the project before setting resources.")
    profiles = proj.get("client_profiles", {}) or {}
    previous = profiles.get(user_id, {}) or {}
    profiles[user_id] = {
        **previous,
        "hardware_profile": normalise_hardware_profile(payload.hardware_profile),
        "last_seen": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    update_project(proj_id, {"client_profiles": profiles})
    return JSONResponse(status_code=200, content={
        "status": "updated",
        "hardware_profile": profiles[user_id]["hardware_profile"],
    })


@router.post("/{proj_id}/dataset-meta")
async def save_dataset_meta(
    proj_id: str,
    payload: DatasetMetaRequest,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """Record local dataset metadata without transferring CSV rows."""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    user_id = current_user["sub"]
    if user_id not in proj.get("pending_clients", []) and user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "Request access to the project before attaching a dataset.")
    profiles = proj.get("client_profiles", {}) or {}
    record = profiles.get(user_id, {}) or {}
    record["dataset_meta"] = {
        "filename": os.path.basename(payload.filename),
        "rows": max(0, payload.rows),
        "columns": max(0, payload.columns),
        "size_bytes": max(0, payload.size_bytes),
        "sha256": payload.sha256,
        "local_only": True,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    record["last_seen"] = dt.datetime.now(dt.timezone.utc).isoformat()
    profiles[user_id] = record
    update_project(proj_id, {"client_profiles": profiles})
    return JSONResponse(status_code=200, content={"status": "metadata_saved", "dataset_meta": record["dataset_meta"]})


@router.post("/{proj_id}/heartbeat")
async def heartbeat(proj_id: str, current_user: dict = Depends(_get_current_user)):  # noqa: B008
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    user_id = current_user["sub"]
    profiles = proj.get("client_profiles", {}) or {}
    if user_id in profiles:
        profiles[user_id]["last_seen"] = dt.datetime.now(dt.timezone.utc).isoformat()
        update_project(proj_id, {"client_profiles": profiles})
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.post("/{proj_id}/update")
async def post_model_update(
    proj_id: str,
    payload: UpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """POST /api/projects/{proj_id}/update"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    user_id = current_user["sub"]
    if user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "Not an approved participant.")

    # Round ID check
    current_round = proj.get("current_round", 0)
    if payload.round_id != current_round:
        return JSONResponse(status_code=409, content={
            "detail": "Round ID mismatch.",
            "expected_round": current_round,
        })

    # Decrypt weights
    try:
        from shared.encryption import decrypt_weights
        decrypted = decrypt_weights(payload.weights)
    except Exception as e:
        raise _http_error(400, f"Weight decryption failed: {e}")

    update_entry = {
        "user_id":     user_id,
        "weights":     decrypted,
        "num_samples": payload.num_samples,
        "active_depth": payload.active_depth,
        "metrics":     payload.metrics,
    }

    with _buffer_lock:
        _pending_updates.setdefault(proj_id, []).append(update_entry)
        submitted = len(_pending_updates[proj_id])

    expected = len(proj.get("connected_clients", []))
    min_clients = proj.get("min_clients_per_round", 1)
    trigger = submitted >= min(expected, min_clients)
    update_project(proj_id, {"round_progress": {
        "round": current_round,
        "submitted": submitted,
        "expected": expected,
        "last_update_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }})

    if trigger:
        db_snapshot = read_db()
        updates_snapshot = list(_pending_updates.get(proj_id, []))
        _dispatch_round(proj_id, updates_snapshot, db_snapshot, background_tasks)
        with _buffer_lock:
            _pending_updates[proj_id] = []

    return JSONResponse(status_code=202, content={
        "status":               "received",
        "clients_submitted":    submitted,
        "clients_expected":     expected,
        "aggregation_triggered": trigger,
    })


@router.get("/{proj_id}/history")
async def get_round_history(
    proj_id: str,
    current_user: dict = Depends(_get_current_user),  # noqa: B008
):
    """GET /api/projects/{proj_id}/history"""
    db = read_db()
    history = [r for r in db.get("rounds_history", []) if r.get("proj_id") == proj_id]
    return JSONResponse(status_code=200, content=history)


@router.post("/{proj_id}/approve/{user_id_to_approve}")
async def approve_client(
    proj_id: str,
    user_id_to_approve: str,
    request: Request,
):
    """
    POST /api/projects/{proj_id}/approve/{user_id} — dashboard admin action.
    Protected by X-Admin-Key header (must match JWT_SECRET env var).
    No user JWT required — this is called directly from the server dashboard.
    """
    import os
    admin_key = request.headers.get("X-Admin-Key", "")
    expected  = os.environ.get("JWT_SECRET", "dev_secret_change_in_production")
    if admin_key != expected:
        raise _http_error(401, "Invalid or missing X-Admin-Key header.")

    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    try:
        return JSONResponse(status_code=200, content=approve_project_client(proj_id, user_id_to_approve))
    except ValueError as exc:
        raise _http_error(400, str(exc)) from exc


def approve_project_client(proj_id: str, user_id_to_approve: str) -> dict:
    """Approve a pending participant; shared by legacy and host endpoints."""
    proj = get_project(proj_id)
    if proj is None:
        raise KeyError(f"Project {proj_id} not found.")
    pending = proj.get("pending_clients", [])
    connected = proj.get("connected_clients", [])
    if user_id_to_approve not in pending:
        raise ValueError("User is not in pending_clients list.")
    pending.remove(user_id_to_approve)
    if user_id_to_approve not in connected:
        connected.append(user_id_to_approve)
    update_project(proj_id, {"pending_clients": pending, "connected_clients": connected})

    db = read_db()
    for user in db["users"]:
        if user["user_id"] == user_id_to_approve:
            if proj_id not in user.get("approved_projects", []):
                user.setdefault("approved_projects", []).append(proj_id)
            if proj_id in user.get("pending_projects", []):
                user["pending_projects"].remove(proj_id)
    write_db(db)
    return {"status": "approved", "user_id": user_id_to_approve}


# ─── Background: Round Lifecycle ─────────────────────────────────────────────

def round_lifecycle(proj_id: str, updates_buffer: list, db_snapshot: dict) -> None:
    """
    Full federated round pipeline (runs as FastAPI BackgroundTask):
      1. aggregate_fedavg
      2. update_with_momentum
      3. validate_global_model
      4. evaluate_architecture_candidates (if depth diversity)
      5. Save .pt, update DB, increment round
    """
    proj = next((p for p in db_snapshot.get("projects", []) if p["proj_id"] == proj_id), None)
    if proj is None:
        return

    try:
        # ── Load current global weights ──────────────────────────────────────
        model_path = proj.get("global_model_path")
        if model_path and os.path.exists(model_path):
            current_global = {k: v.numpy() for k, v in torch.load(model_path, map_location="cpu").items()}
        else:
            current_global = {}

        weight_dicts = [u["weights"] for u in updates_buffer]
        sample_counts = [u["num_samples"] for u in updates_buffer]

        # ── Step 1: FedAvg ───────────────────────────────────────────────────
        fedavg_result = aggregate_fedavg(weight_dicts, sample_counts)

        # ── Step 2: Momentum ─────────────────────────────────────────────────
        velocity = _velocity_state.get(proj_id, {})
        momentum = proj.get("momentum_beta", 0.9)
        new_global, new_velocity = update_with_momentum(current_global, fedavg_result, momentum, velocity)
        _velocity_state[proj_id] = new_velocity

        # ── Step 3: Validate ─────────────────────────────────────────────────
        metrics = {}
        if _val_dataloader is not None:
            metrics = validate_global_model(new_global, _val_dataloader, MODEL_CONFIG)

        # ── Step 4: NAS ──────────────────────────────────────────────────────
        updates_by_depth = {}
        for u in updates_buffer:
            d = u.get("active_depth", MODEL_CONFIG["max_depth"])
            updates_by_depth.setdefault(d, []).append(u)

        if len(updates_by_depth) > 1:
            recommended_depth = evaluate_architecture_candidates(updates_by_depth, current_global)
        else:
            recommended_depth = proj.get("recommended_depth", MODEL_CONFIG["max_depth"])

        # ── Step 5: Save model ───────────────────────────────────────────────
        new_round = proj.get("current_round", 0) + 1
        pt_path = os.path.join(MODELS_DIR, f"{proj_id}_round{new_round}.pt")
        torch.save({k: torch.from_numpy(np.array(v)) for k, v in new_global.items()}, pt_path)

        # ── Update DB ────────────────────────────────────────────────────────
        update_project(proj_id, {
            "current_round":     new_round,
            "global_model_path": pt_path,
            "recommended_depth": recommended_depth,
            "round_progress": {
                "round": new_round,
                "submitted": 0,
                "expected": len(proj.get("connected_clients", [])),
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        })

        # Append round history
        record = {
            "proj_id": proj_id,
            "round":   new_round,
            "participants": [u.get("user_id") for u in updates_buffer],
            "depths": {str(u.get("user_id")): int(u.get("active_depth", MODEL_CONFIG["max_depth"])) for u in updates_buffer},
            **metrics,
        }
        append_round_history(record)

    except EmptyRoundError as e:
        print(f"[round_lifecycle] EmptyRoundError for {proj_id}: {e}")
    except Exception as e:
        print(f"[round_lifecycle] ERROR for {proj_id}: {e}")
