"""Local companion used by ``client_app.py`` for command-free training.

The browser cannot start a Python process or reveal a selected file's path.
This small localhost service bridges those two browser restrictions without
ever sending the CSV to the federated server.  It only accepts requests from
the local client page, stores the selected CSV locally, and starts the native
worker after the page has completed the host approval flow.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


CLIENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CLIENT_DIR.parent
UI_PATH = CLIENT_DIR / "client.html"
RUNTIME_DIR = CLIENT_DIR / "runtime_data"
MAX_DATASET_BYTES = 500 * 1024 * 1024

_state_lock = threading.Lock()
_state: dict = {
    "worker": None,
    "server_url": "",
    "name": "",
    "dataset_path": "",
    "message": "Waiting for the client setup page.",
}


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _response_state() -> dict:
    with _state_lock:
        process = _state.get("worker")
        returncode = process.poll() if process is not None else None
        if process is not None and returncode is not None:
            _state["worker"] = None
            if returncode == 0:
                _state["message"] = "Worker stopped normally."
            else:
                _state["message"] = f"Worker stopped with code {returncode}. Check the client terminal."
        return {
            "running": _state.get("worker") is not None,
            "pid": process.pid if process is not None and returncode is None else None,
            "server_url": _state.get("server_url", ""),
            "name": _state.get("name", ""),
            "dataset_path": _state.get("dataset_path", ""),
            "message": _state.get("message", ""),
            "returncode": returncode,
        }


def _resources() -> dict:
    try:
        import psutil

        available_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
        available_cpu_cores = psutil.cpu_count(logical=True) or 2
        source = "native psutil"
    except Exception:
        available_ram_gb = 8.0
        available_cpu_cores = 4
        source = "conservative fallback"
    gpu_available = False
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "available_ram_gb": max(0.5, available_ram_gb),
        "available_cpu_cores": max(1, int(available_cpu_cores)),
        "gpu_available": gpu_available,
        "source": source,
    }


def _multipart_file(content_type: str, body: bytes) -> tuple[str, bytes]:
    header = BytesParser(policy=email_default).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    boundary = header.get_boundary()
    if not boundary:
        raise ValueError("The selected file upload was not valid multipart data.")
    delimiter = b"--" + boundary.encode("utf-8")
    for part in body.split(delimiter):
        if b"filename=" not in part or b"\r\n\r\n" not in part:
            continue
        part = part.lstrip(b"\r\n")
        header_bytes, payload = part.split(b"\r\n\r\n", 1)
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        filename_match = re.search(rb'filename="([^"]*)"', header_bytes)
        if filename_match is None:
            filename_match = re.search(rb"filename=([^;\r\n]+)", header_bytes)
        filename = filename_match.group(1).decode("utf-8", errors="replace") if filename_match else "dataset.csv"
        return Path(filename).name, payload
    raise ValueError("No CSV file was found in the upload.")


class _LocalHandler(BaseHTTPRequestHandler):
    server_version = "FLClientLocalAgent/1.0"

    def log_message(self, format: str, *args):  # noqa: A002
        return

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _send_json(self, value: dict, status: int = 200) -> None:
        body = _json_bytes(value)
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid request length.") from exc
        if length < 0 or length > MAX_DATASET_BYTES + 2_000_000:
            raise ValueError("The selected file is too large for the local client.")
        return self.rfile.read(length)

    def do_OPTIONS(self):  # noqa: N802
        self._headers(204, "text/plain", 0)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/client.html"}:
            try:
                body = UI_PATH.read_bytes()
            except OSError:
                self._send_json({"detail": "Client UI file is missing."}, 500)
                return
            self._headers(200, "text/html; charset=utf-8", len(body))
            self.wfile.write(body)
            return
        if path == "/agent/status":
            self._send_json(_response_state())
            return
        if path == "/agent/resources":
            self._send_json(_resources())
            return
        self._send_json({"detail": "Local client endpoint not found."}, 404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/agent/dataset":
                filename, content = _multipart_file(self.headers.get("Content-Type", ""), body)
                if not filename.lower().endswith(".csv"):
                    raise ValueError("Choose a CSV file.")
                if len(content) > MAX_DATASET_BYTES:
                    raise ValueError("The selected CSV is larger than 500 MB.")
                RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                destination = RUNTIME_DIR / f"{uuid.uuid4().hex}_{filename}"
                destination.write_bytes(content)
                with _state_lock:
                    _state["dataset_path"] = str(destination)
                    _state["message"] = "CSV saved locally. Waiting for host approval."
                self._send_json({"status": "saved", "path": str(destination), "filename": filename})
                return

            if path == "/agent/start":
                payload = json.loads(body.decode("utf-8"))
                server_url = str(payload.get("server_url", "")).strip().rstrip("/")
                name = str(payload.get("name", "")).strip()
                raw_csv_path = Path(str(payload.get("csv_path", "")))
                csv_path = (raw_csv_path if raw_csv_path.is_absolute() else REPO_ROOT / raw_csv_path).resolve()
                if not re.match(r"^https?://", server_url, flags=re.IGNORECASE):
                    raise ValueError("The host URL must start with http:// or https://.")
                if not name:
                    raise ValueError("A participant name is required.")
                if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
                    raise ValueError("Choose a local CSV before starting the worker.")
                with _state_lock:
                    process = _state.get("worker")
                    worker_already_running = process is not None and process.poll() is None
                if worker_already_running:
                    self._send_json(_response_state())
                    return
                command = [
                    sys.executable,
                    str(CLIENT_DIR / "client_app.py"),
                    "--server", server_url,
                    "--name", name,
                    "--csv", str(csv_path),
                    "--no-ui",
                ]
                for key, flag in (("available_ram_gb", "--ram"), ("available_cpu_cores", "--cores"),
                                  ("dedicated_ram_gb", "--dedicated-ram"), ("dedicated_cpu_cores", "--dedicated-cores")):
                    if payload.get(key) not in (None, ""):
                        command.extend([flag, str(payload[key])])
                if payload.get("gpu_available"):
                    command.append("--gpu")
                process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=os.environ.copy())
                with _state_lock:
                    _state.update({
                        "worker": process,
                        "server_url": server_url,
                        "name": name,
                        "dataset_path": str(csv_path),
                        "message": "Native worker started. It will train after host approval.",
                    })
                self._send_json(_response_state(), 201)
                return

            self._send_json({"detail": "Local client endpoint not found."}, 404)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"detail": str(exc)}, 400)


def start_local_agent_background(port: int = 8765) -> tuple[ThreadingHTTPServer, str]:
    """Start the localhost bridge without opening a browser."""

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), _LocalHandler)
    except OSError:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _LocalHandler)
    host, port = httpd.server_address
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=httpd.serve_forever, name="fl-local-agent", daemon=True)
    thread.start()
    return httpd, url


def stop_local_agent(httpd: ThreadingHTTPServer | None) -> None:
    """Stop a background agent and any worker it launched."""

    if httpd is None:
        return
    with _state_lock:
        process = _state.get("worker")
    if process is not None and process.poll() is None:
        process.terminate()
    httpd.shutdown()
    httpd.server_close()


def start_local_client_ui() -> None:
    """Serve the client page locally, open it, and keep the agent alive."""

    import webbrowser

    httpd, url = start_local_agent_background()
    print(f"Client UI: {url}")
    print("Enter your name and the host ngrok link. The native worker starts after approval.")
    webbrowser.open(url)
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop_local_agent(httpd)


if __name__ == "__main__":
    start_local_client_ui()
