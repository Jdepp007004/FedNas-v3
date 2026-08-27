"""
client/client_app.py
Client entry point — launches local FL training loop.
Owner: Nikhil Garuda (M4 orchestration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from local_env import load_env_file  # noqa: E402

load_env_file(os.environ.get("FL_ENV_FILE", os.path.join(os.path.dirname(__file__), "..", ".env")))

import time  # noqa: E402
import argparse  # noqa: E402

from api_client import APIClient, ServerUnreachableError, AuthError  # noqa: E402
from supernet import Supernet, load_global_weights  # noqa: E402
from train_loop import run_local_training, TrainConfig  # noqa: E402
from data_loader import build_dataloaders_from_csv  # noqa: E402
from schema_validator import validate_schema, format_validation_report  # noqa: E402
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA, DEFAULT_BATCH_SIZE  # noqa: E402


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning Client")
    p.add_argument("--server",    required=True, help="ngrok server URL, e.g. https://xxx.ngrok.io")
    p.add_argument("--name",      required=True, help="Name shown to the host for approval")
    p.add_argument("--csv",       required=True, help="Path to local TCGA CSV file")
    p.add_argument("--proj",      default=None, help=argparse.SUPPRESS)
    p.add_argument("--ram",       type=float, default=None, help="Override detected available RAM (GB)")
    p.add_argument("--cores",     type=int,   default=None, help="Override detected CPU cores")
    p.add_argument("--dedicated-ram", type=float, default=None, help="RAM cap to contribute (GB)")
    p.add_argument("--dedicated-cores", type=int, default=None, help="CPU core cap to contribute")
    p.add_argument("--gpu",       action="store_true")
    p.add_argument("--no-ui",     action="store_true", help="Disable matplotlib dashboard")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Native clients can report real machine capacity.  The browser console
    # uses conservative estimates because browsers do not expose free RAM.
    import hashlib
    try:
        import psutil
        detected_ram = round(psutil.virtual_memory().available / (1024 ** 3), 2)
        detected_cores = psutil.cpu_count(logical=True) or 2
    except Exception:
        detected_ram = 8.0
        detected_cores = 4
    available_ram = max(0.5, float(args.ram or detected_ram))
    available_cores = max(1, int(args.cores or detected_cores))
    dedicated_ram = min(available_ram, float(args.dedicated_ram or max(0.5, available_ram * 0.5)))
    dedicated_cores = min(available_cores, int(args.dedicated_cores or max(1, (available_cores + 1) // 2)))
    gpu_available = bool(args.gpu)

    client = APIClient(args.server)

    # ── Connectivity check ────────────────────────────────────────────────────
    print(f"[*] Connecting to server: {args.server}")
    try:
        status = client.check_status()
        print(f"[+] Server OK — version {status.get('server_version', '?')}")
    except ServerUnreachableError as e:
        print(f"[!] Cannot reach server: {e}")
        sys.exit(1)

    # ── Passwordless participant session ─────────────────────────────────────
    print(f"[*] Joining as {args.name}…")
    try:
        client.guest_login(args.name)
        print(f"[+] Temporary participant session created for {args.name}")
    except Exception as e:
        print(f"[!] Could not create participant session: {e}")
        sys.exit(1)

    # The default project is selected automatically.  ``--proj`` remains an
    # internal/backwards-compatible override but is not needed by teammates.
    if not args.proj:
        try:
            projects = client.list_projects()
            project = next((item for item in projects if item.get("accepting_clients", True)), None)
            if project is None:
                raise RuntimeError("The host has no project accepting clients yet.")
            args.proj = project["proj_id"]
            print(f"[+] Project selected automatically: {project.get('name', args.proj)}")
        except Exception as e:
            print(f"[!] Could not find an open project: {e}")
            sys.exit(1)

    # ── Schema Validation ─────────────────────────────────────────────────────
    print(f"[*] Validating CSV: {args.csv}")
    import pandas as pd
    df_check = pd.read_csv(args.csv, low_memory=False)
    val_result = validate_schema(df_check, SERVER_SCHEMA)
    print(format_validation_report(val_result))
    if not val_result.passed:
        print("[!] Schema validation failed. Please fix errors before joining.")
        sys.exit(1)

    # ── Join Project ──────────────────────────────────────────────────────────
    hw_profile = {
        "available_ram_gb": available_ram,
        "available_cpu_cores": available_cores,
        "dedicated_ram_gb": dedicated_ram,
        "dedicated_cpu_cores": dedicated_cores,
        "gpu_available": gpu_available,
        "local_data_size": df_check.shape[0],
        "resource_source": "native psutil",
    }
    print(f"[*] Joining project: {args.proj}")
    try:
        join_resp = client.join_project(args.proj, hw_profile)
        active_depth = join_resp.get("recommended_depth", 4)
        schema = join_resp.get("required_schema", SERVER_SCHEMA)
        print(f"[+] Join request submitted. Recommended depth: {active_depth}")
        print(f"[*] Status: {join_resp.get('status', '?')} — waiting for admin approval…")
        client.update_resources(args.proj, hw_profile)
        with open(args.csv, "rb") as dataset_handle:
            dataset_digest = hashlib.sha256(dataset_handle.read()).hexdigest()
        client.save_dataset_meta(args.proj, {
            "filename": os.path.basename(args.csv),
            "rows": int(df_check.shape[0]),
            "columns": int(df_check.shape[1]),
            "size_bytes": os.path.getsize(args.csv),
            "sha256": dataset_digest,
        })
    except Exception as e:
        print(f"[!] Join failed: {e}")
        sys.exit(1)

    # ── Wait for approval ─────────────────────────────────────────────────────
    print("[*] Polling for approval (press Ctrl-C to abort)…")
    approved = False
    while not approved:
        try:
            projects = client.list_projects()
            proj = next((p for p in projects if p.get("proj_id") == args.proj), None)
            if proj and proj.get("i_am_connected"):
                approved = True
                break
            # Fallback: try fetching model — 403 means still pending, 200 means approved
            model_resp = client.fetch_global_model(args.proj)  # noqa: F841
            approved = True
        except AuthError:
            pass  # still pending
        except Exception:
            pass
        if not approved:
            time.sleep(10)

    print("[+] Approved! Starting federated training loop…")

    # ── Build model ───────────────────────────────────────────────────────────
    supernet = Supernet(**MODEL_CONFIG)

    # ── Init visualizer ───────────────────────────────────────────────────────
    fig, axes = None, None
    if not args.no_ui:
        try:
            from visualizer import init_metrics_dashboard
            fig, axes = init_metrics_dashboard()
        except Exception as e:
            print(f"[!] Visualizer init failed: {e}. Running headless.")

    # ── Federated training loop ───────────────────────────────────────────────
    round_history = []
    while True:
        try:
            client.heartbeat(args.proj)
        except Exception:
            pass
        # R1: Fetch global model
        print("[*] Fetching global model…")
        model_data = client.fetch_global_model(args.proj)
        current_round = model_data["round"]
        active_depth = model_data.get("active_depth", active_depth)
        global_weights = model_data["weights"]
        print(f"[+] Round {current_round} | Active depth: {active_depth}")

        # R2: Load global weights
        load_global_weights(supernet, global_weights, strict=False)

        # R3: Load & preprocess data
        print("[*] Preparing data loaders…")
        train_loader, val_loader = build_dataloaders_from_csv(
            args.csv, schema, batch_size=DEFAULT_BATCH_SIZE
        )

        # R4: Local training
        print("[*] Starting local training…")
        cfg = TrainConfig(active_depth=active_depth)
        result = run_local_training(supernet, (train_loader, val_loader), cfg, axes=axes)
        print(f"[+] Training done | Loss: {result['metrics']['loss']:.4f} | "
              f"RMSE: {result['metrics']['val_rmse']:.4f} | "
              f"ToxAcc: {result['metrics']['val_acc_tox']:.4f} | "
              f"AUC: {result['metrics']['val_auc']:.4f}")

        # R6: Post update
        print("[*] Posting model update to server…")
        post_resp = client.post_local_update(
            proj_id=args.proj,
            weights=result["weights"],
            num_samples=result["num_samples"],
            metrics=result["metrics"],
            round_id=current_round,
            active_depth=active_depth,
        )
        print(f"[+] Update received. Clients submitted: "
              f"{post_resp.get('clients_submitted')}/{post_resp.get('clients_expected')} | "
              f"Aggregation triggered: {post_resp.get('aggregation_triggered')}")

        # R13: Update dashboard
        round_history = client.get_round_history(args.proj)
        if axes and round_history:
            try:
                from visualizer import update_global_metrics
                update_global_metrics(axes, round_history)
            except Exception:
                pass

        # Wait before next round
        print("[*] Waiting for next round… (sleeping 5s)")
        time.sleep(5)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        from local_agent import start_local_client_ui
        start_local_client_ui()
    else:
        main()
