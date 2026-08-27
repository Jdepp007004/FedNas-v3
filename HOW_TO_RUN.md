# FL Platform — Complete How-To Guide

## Prerequisites

```powershell
# Verify Python 3.10+
python --version

# Install all dependencies
pip install requests pandas scikit-learn fastapi uvicorn pyngrok \
            torch cryptography PyJWT bcrypt pyngrok aiofiles \
            python-multipart Jinja2
```

---

## 1. Environment Setup

Run once from the **repo root** (`fl_platform/`):

```powershell
# Generate encryption key
python -c "import base64,os; print(base64.b64encode(os.urandom(32)).decode())"
# → copy the output, e.g. 9m6IJiKDiSL1+nA3OH3SZxZUqCF7SanGObrCnZWpMpQ=

# Set environment variables (PowerShell — re-run every new terminal session)
$env:FL_ENCRYPTION_KEY = "PASTE_OUTPUT_HERE"
$env:JWT_SECRET        = "any_string_you_want"
$env:NGROK_AUTH_TOKEN  = "your_ngrok_token_from_ngrok.com"
```

> **JWT_SECRET** can be any string for local dev. Only change before going to production.

---

## 2. Download the TCGA Dataset & Split for Clients

```powershell
# From repo root
python download_and_split.py --max-cases 10000 --n-clients 4
```

**What it does:**
- Pulls up to 10,000 clinical records from the GDC (NIH) public API — no account needed
- Cleans and maps columns to the FL Platform schema
- Saves to `data/`:
  ```
  data/
    full_dataset.csv   ← complete dataset (keep as backup)
    client_1.csv       ← send to Teammate 1
    client_2.csv       ← send to Teammate 2
    client_3.csv       ← send to Teammate 3
    client_4.csv       ← send to Teammate 4
  ```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--max-cases` | 10000 | Number of patient records to download |
| `--n-clients` | 4 | How many splits to create |
| `--output-dir` | `data/` | Where to save the CSV files |

---

## 3. Start the Server

```powershell
# From repo root — set env vars first (see Section 1)
cd c:\path\to\fl_platform

$env:FL_ENCRYPTION_KEY = "9m6IJiKDiSL1+nA3OH3SZxZUqCF7SanGObrCnZWpMpQ="
$env:JWT_SECRET        = "my_dev_secret"
$env:NGROK_AUTH_TOKEN  = "your_ngrok_token"

cd server
python main.py
```

**Expected output:**
```
[+] ngrok tunnel active: https://abc123.ngrok-free.app   ← share this URL
[+] Created default project: 193b8223-....
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Useful URLs (while server is running):**

| URL | Purpose |
|-----|---------|
| http://localhost:8000/host | Passwordless host control console |
| http://localhost:8000/client | Client name/link connection page |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:8000/api/status | Health check (JSON) |

---

## 4. Get the Project ID

The current workflow is passwordless. The host console does not require a
login, and a client only enters a display name plus the host's ngrok link.
The older account endpoints are retained only for compatibility with older
clients.

After the server starts, the project ID is printed to the console. It is also
shown in the host console's project selector. For a script-based check, use:

```powershell
# Host project listing does not require a second host account.
Invoke-WebRequest -Uri "http://localhost:8000/api/host/projects" |
  Select-Object -ExpandProperty Content
```

Or just read it from the server console output or `server/database.json`.

---

## 5. Client Setup (each teammate)

### What to send each teammate
1. The `fl_platform` folder (zip it or share via GitHub)
2. Their `client_X.csv` file from `data/`
3. The ngrok URL (e.g. `https://abc123.ngrok-free.app`)
4. The project ID (from Step 4)

### Install client dependencies (one time)
```powershell
cd fl_platform
pip install -r client/requirements.txt
```

---

## 6. Run the Client

### Option A — Command Line (recommended)

**Teammate 1:**
```powershell
python client/client_app.py `
  --server   https://abc123.ngrok-free.app `
  --name     "Client 1" `
  --csv      data/client_1.csv `
  --proj     193b8223-311e-4de4-809d-68d431da46ab
```

**Teammate 2:**
```powershell
python client/client_app.py `
  --server   https://abc123.ngrok-free.app `
  --name     "Client 2" `
  --csv      data/client_2.csv `
  --proj     193b8223-311e-4de4-809d-68d431da46ab
```

> **Teammates 3 and 4:** Same pattern, change the displayed name and CSV split.

### Option B — Browser UI (no Python knowledge needed)

1. Run `python client_app.py` to open the client page, or open
   `https://YOUR_NGROK_URL/client` in any browser.
2. Enter a display name and the ngrok URL, then click **Connect and request access**.
3. Select the project and local `client_X.csv` split, then save the resource allocation.
4. Wait for the host to approve that name in the host console.
5. Run the native `python client_app.py ...` command shown on the client page. The browser page is the live control plane; the Python worker performs the real local training.

**All CLI flags:**

| Flag | Required | Description |
|------|----------|-------------|
| `--server` | ✅ | ngrok HTTPS URL |
| `--name` | ✅ | Name shown to the host for approval |
| `--csv` | ✅ | Path to their client CSV file |
| `--proj` | ✅ | Project UUID |
| `--ram` | ❌ | RAM in GB (for NAS depth, default 8) |
| `--cores` | ❌ | CPU cores (default 4) |
| `--gpu` | ❌ | Flag: GPU available |
| `--no-ui` | ❌ | Disable live Matplotlib charts |

---

## 7. Approve Clients (Server Admin)

1. Open **http://localhost:8000/host**
2. Under the **Projects** table, **Pending Approval** column shows yellow badges
3. Click the badge to approve that client
4. Client automatically moves to **Connected** and training begins

> The host page auto-refreshes through its live stream. Anyone who has the ngrok link can reach host controls, so share it only with trusted collaborators.

---

## 8. Federated Training Round Flow

```
Client                               Server
  │── GET /api/projects/{id}/model ─▶│  Download global weights
  │◀─ {round, depth, weights} ───────│
  │   [Local training on client CSV]  │
  │── POST /api/projects/{id}/update ▶│  Send encrypted update
  │◀─ 202 Accepted ──────────────────│
  │                                   │  [FedAvg → Momentum → Validate → NAS]
  │── GET /api/projects/{id}/history ▶│  Poll for new metrics
  │◀─ [round records] ───────────────│
        ↑ repeats until max_rounds
```

---

## 9. Run Tests

```powershell
# From repo root
pip install -r tests/requirements-test.txt

pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 10. Docker (Production)

```bash
# From repo root
docker compose -f server/docker-compose.yml up --build -d

# View logs
docker compose -f server/docker-compose.yml logs -f

# Stop
docker compose -f server/docker-compose.yml down
```

Create a `.env` file at repo root for Docker:
```
FL_ENCRYPTION_KEY=your_base64_key
JWT_SECRET=your_secret
NGROK_AUTH_TOKEN=your_ngrok_token
```

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: shared` | Run commands from repo root, not from `server/` |
| `pyngrok is not installed` | `pip install pyngrok` — import is `from pyngrok import ngrok` |
| `ngrok tunnel failed` | Check `NGROK_AUTH_TOKEN` is set correctly |
| `403 Not an approved participant` | Admin must click approve on the dashboard |
| `409 Round ID mismatch` | Client is behind — it will auto-retry |
| Schema validation fails | Ensure CSV was generated by `download_and_split.py` |
| Dashboard shows Jinja2 error | Pull latest `dashboard.html` (template was fixed) |
