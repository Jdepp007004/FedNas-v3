"""Convenience entry point for the native participant trainer.

This wrapper lets teammates run ``python client_app.py ...`` after receiving
the repository and their local CSV split.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIENT_DIR = os.path.join(ROOT, "client")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if CLIENT_DIR not in sys.path:
    sys.path.insert(0, CLIENT_DIR)

from local_env import load_env_file

_default_env = os.path.join(ROOT, ".env")
if not os.path.exists(_default_env):
    _default_env = os.path.join(ROOT, "client.env")
load_env_file(os.environ.get("FL_ENV_FILE", _default_env))

from client.client_app import main


if __name__ == "__main__":
    # With no arguments, open the simple client onboarding page.  Supplying
    # arguments runs the real native worker for local training.
    if len(sys.argv) == 1:
        from client.local_agent import start_local_client_ui
        start_local_client_ui()
    else:
        main()
