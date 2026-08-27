"""Runnable host entry point for the federated-learning platform.

Run from the repository root with ``python host_app.py``.  The host serves
the API, host console, client console, and optional ngrok tunnel.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
import webbrowser

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
    from client.local_agent import start_local_agent_background, stop_local_agent

    host_agent, host_agent_url = start_local_agent_background()
    os.environ["FL_HOST_AGENT_URL"] = host_agent_url
    port = int(os.environ.get("SERVER_PORT", "8000"))

    def open_host_console_when_ready():
        url = f"http://127.0.0.1:{port}/host"
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=0.5).close()
                webbrowser.open(url)
                return
            except Exception:
                time.sleep(0.5)

    threading.Thread(target=open_host_console_when_ready, name="open-host-console", daemon=True).start()
    try:
        uvicorn.run(
            app,
            host=os.environ.get("SERVER_HOST", "0.0.0.0"),
            port=port,
            reload=False,
        )
    finally:
        stop_local_agent(host_agent)
