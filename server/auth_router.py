from __future__ import annotations
"""
server/auth_router.py
M3: /api/auth/* endpoints — register and login.
Owner: Sunishka Sarkar
"""

import uuid
import datetime
import hmac
from datetime import timezone as _tz

import bcrypt
import jwt as pyjwt
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db_handler import read_db, write_db, get_user  # noqa: F401

# Environment variable key for JWT signing
import os
JWT_SECRET = os.environ.get("JWT_SECRET", "dev_secret_change_in_production")
JWT_ALGO = "HS256"
JWT_EXP_HOURS = 24

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username:      str
    password:      str
    hospital_name: str
    contact_email: str


class LoginRequest(BaseModel):
    username: str
    password: str


class HostLoginRequest(BaseModel):
    username: str
    password: str


class GuestLoginRequest(BaseModel):
    display_name: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def create_jwt(user_id: str, role: str | None = None) -> str:
    exp = datetime.datetime.now(_tz.utc) + datetime.timedelta(hours=JWT_EXP_HOURS)
    payload = {"sub": user_id, "exp": exp}
    if role:
        payload["role"] = role
    return pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_jwt(token: str) -> dict:
    """Decode and validate a JWT; returns the payload dict."""
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except pyjwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except pyjwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")


def verify_host_jwt(token: str) -> dict:
    """Decode a host token and reject ordinary participant tokens."""
    payload = verify_jwt(token)
    if payload.get("role") != "host":
        raise ValueError("This token is not a host token.")
    return payload


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register")
async def register_user(payload: RegisterRequest):
    """POST /api/auth/register — create a new hospital account."""
    db = read_db()

    # Uniqueness check
    existing = next((u for u in db["users"] if u["username"] == payload.username), None)
    if existing:
        return JSONResponse(
            status_code=409,
            content={"detail": f"Username '{payload.username}' is already taken."},
        )

    # Hash password
    pw_hash = bcrypt.hashpw(payload.password.encode(), bcrypt.gensalt()).decode()

    user_id = str(uuid.uuid4())
    now = datetime.datetime.now(_tz.utc).isoformat() + "Z"
    new_user = {
        "user_id":          user_id,
        "username":         payload.username,
        "password_hash":    pw_hash,
        "hospital_name":    payload.hospital_name,
        "contact_email":    payload.contact_email,
        "approved_projects": [],
        "pending_projects":  [],
        "created_at":       now,
        "last_active":      now,
    }
    db["users"].append(new_user)
    write_db(db)

    return JSONResponse(
        status_code=201,
        content={"user_id": user_id, "username": payload.username},
    )


@router.post("/login")
async def login_user(payload: LoginRequest):
    """POST /api/auth/login — validate credentials and issue JWT."""
    db = read_db()
    user = next((u for u in db["users"] if u["username"] == payload.username), None)

    if user is None or not bcrypt.checkpw(
        payload.password.encode(), user["password_hash"].encode()
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid username or password."},
        )

    # Update last_active
    user["last_active"] = datetime.datetime.now(_tz.utc).isoformat() + "Z"
    write_db(db)

    token = create_jwt(user["user_id"])
    return JSONResponse(
        status_code=200,
        content={
            "access_token":      token,
            "token_type":        "bearer",
            "approved_projects": user.get("approved_projects", []),
            "user_id":           user["user_id"],
        },
    )


@router.post("/host-login")
async def login_host(payload: HostLoginRequest):
    """Issue a host token using credentials held in environment variables."""
    expected_username = os.environ.get("HOST_USERNAME", "host")
    expected_password = os.environ.get("HOST_PASSWORD", "hostpass")
    configured_hash = os.environ.get("HOST_PASSWORD_HASH", "")
    if configured_hash:
        try:
            valid_password = bcrypt.checkpw(payload.password.encode(), configured_hash.encode())
        except ValueError:
            valid_password = False
    else:
        valid_password = hmac.compare_digest(payload.password, expected_password)
    if payload.username != expected_username or not valid_password:
        return JSONResponse(status_code=401, content={"detail": "Invalid host credentials."})

    token = create_jwt("host", role="host")
    return JSONResponse(status_code=200, content={
        "access_token": token,
        "token_type": "bearer",
        "role": "host",
        "username": expected_username,
    })


@router.post("/guest")
async def login_guest(payload: GuestLoginRequest):
    """Create a passwordless participant identity for a trusted collaboration."""
    display_name = payload.display_name.strip()
    if not 1 <= len(display_name) <= 80:
        return JSONResponse(status_code=422, content={"detail": "Name must be between 1 and 80 characters."})

    db = read_db()
    now = datetime.datetime.now(_tz.utc).isoformat() + "Z"
    # Reuse the same guest identity when the native worker is started after
    # the browser request.  This prevents the host from seeing two separate
    # approvals for the same named participant.
    user = next((item for item in db.setdefault("users", [])
                 if item.get("anonymous") and item.get("username") == display_name), None)
    if user is None:
        user_id = str(uuid.uuid4())
        # Keep the existing project/update pipeline compatible while ensuring
        # no real password is ever requested or stored for guest participants.
        guest_secret = bcrypt.hashpw(os.urandom(32), bcrypt.gensalt()).decode()
        user = {
            "user_id": user_id,
            "username": display_name,
            "password_hash": guest_secret,
            "hospital_name": display_name,
            "contact_email": "",
            "approved_projects": [],
            "pending_projects": [],
            "anonymous": True,
            "created_at": now,
            "last_active": now,
        }
        db["users"].append(user)
        status_code = 201
    else:
        user_id = user["user_id"]
        user["last_active"] = now
        status_code = 200
    write_db(db)
    token = create_jwt(user_id, role="participant")
    return JSONResponse(status_code=status_code, content={
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "display_name": display_name,
    })
