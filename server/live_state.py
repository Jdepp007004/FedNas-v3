"""Safe, serialisable project snapshots for the host/client live views."""

from __future__ import annotations

from db_handler import read_db
from resource_planner import build_round_plan


def build_project_snapshot(proj_id: str, viewer_id: str | None = None, host: bool = False) -> dict:
    db = read_db()
    project = next((item for item in db.get("projects", []) if item.get("proj_id") == proj_id), None)
    if project is None:
        raise KeyError(proj_id)

    users = {item.get("user_id"): item for item in db.get("users", [])}
    plan = build_round_plan(project, users)
    profile_map = project.get("client_profiles", {}) or {}
    connected = set(project.get("connected_clients", []) or [])
    pending = set(project.get("pending_clients", []) or [])
    clients = []
    for user_id in sorted(connected | pending):
        user = users.get(user_id, {})
        record = profile_map.get(user_id, {}) or {}
        clients.append({
            "user_id": user_id,
            "username": user.get("username", user_id[:8]),
            "display_name": user.get("hospital_name", user.get("username", user_id[:8])),
            "contact_email": user.get("contact_email", ""),
            "status": "approved" if user_id in connected else "pending",
            "hardware_profile": record.get("hardware_profile", {}),
            "dataset_meta": record.get("dataset_meta", {}),
            "last_seen": record.get("last_seen"),
        })

    visible_project = {key: value for key, value in project.items() if key not in {"global_model_path", "client_profiles"}}
    visible_project["connected_count"] = len(connected)
    visible_project["pending_count"] = len(pending)
    visible_project["round_progress"] = project.get("round_progress", {
        "round": project.get("current_round", 0),
        "submitted": 0,
        "expected": len(connected),
    })

    if not host:
        clients = [item for item in clients if item["user_id"] == viewer_id]
        plan["clients"] = [item for item in plan["clients"] if item["user_id"] == viewer_id]

    return {
        "project": visible_project,
        "clients": clients,
        "dataset": project.get("dataset", {}),
        "round_plan": plan,
        "history": [item for item in db.get("rounds_history", []) if item.get("proj_id") == proj_id][-50:],
        "server_time": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
