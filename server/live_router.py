"""Live project snapshots and WebSocket stream for the two browser consoles."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect

from auth_router import verify_jwt
from live_state import build_project_snapshot

router = APIRouter(tags=["live"])


def _token_from_header(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    try:
        return verify_jwt(authorization.split(" ", 1)[1])
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/api/projects/{proj_id}/live")
async def client_live_state(proj_id: str, authorization: str | None = Header(None)):
    claims = _token_from_header(authorization)
    try:
        return build_project_snapshot(proj_id, viewer_id=claims.get("sub"), host=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Project {proj_id} not found.") from exc


@router.websocket("/ws/projects/{proj_id}")
async def project_stream(websocket: WebSocket, proj_id: str):
    token = websocket.query_params.get("token", "")
    try:
        claims = verify_jwt(token)
        is_host = claims.get("role") == "host"
        viewer_id = claims.get("sub")
        build_project_snapshot(proj_id, viewer_id=viewer_id, host=is_host)
    except (ValueError, KeyError):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            snapshot = build_project_snapshot(proj_id, viewer_id=viewer_id, host=is_host)
            await websocket.send_json(snapshot)
            await asyncio.sleep(1.5)
    except (WebSocketDisconnect, RuntimeError):
        return
