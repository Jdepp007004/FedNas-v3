"""Runnable host entry point for the federated-learning platform.

Run from the repository root with ``python host_app.py``.  The host serves
the API, host console, client console, and optional ngrok tunnel.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from local_env import load_env_file

load_env_file(os.environ.get("FL_ENV_FILE", os.path.join(ROOT, ".env")))

import uvicorn

from main import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVER_PORT", "8000")),
        reload=False,
    )
