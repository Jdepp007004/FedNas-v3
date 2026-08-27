# Run the real host/client platform

This is the runnable code path for the platform. It is separate from the
demo/site project and uses the existing FastAPI server plus the native Python
client trainer.

## Dataset recommendation

Use the checked-in `data/full_dataset.csv`. It is the cleaned clinical dataset
already shaped for this platform's feature schema. The raw `SEER_cleaned.csv`
has different column names and would need a separate preprocessing adapter.

The four ready-to-share partitions are in `split_data/`:

```text
split_data/client_1.csv
split_data/client_2.csv
split_data/client_3.csv
split_data/client_4.csv
```

They were created deterministically with `prepare_split_data.py`, balanced by
`vital_status`, and contain 1,031 / 1,029 / 1,029 / 1,027 rows. Keep each file
on its owner's laptop. The client UI sends only filename, row count, schema
metadata, and model updates; it does not upload client rows.

## One-time install

From the repository root:

```powershell
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

The repository now has a local `.env` file for this workspace. It is ignored by
Git and is loaded automatically by `host_app.py` and `client_app.py`. For a
fresh clone, create `.env` from `.env.example` and fill in the values below.
The host/client pages no longer ask anyone for an account or password:

```powershell
$env:JWT_SECRET = "replace-with-a-long-secret"
$env:FL_ENCRYPTION_KEY = (python -c "from shared.encryption import generate_key_b64; print(generate_key_b64())")
$env:NGROK_AUTH_TOKEN = "your-ngrok-auth-token"
```

`NGROK_AUTH_TOKEN` may be left unset for local-only testing. Keep the ngrok URL
private to the friends who should be able to see the host controls.

## Start the host

From the repository root:

```powershell
python host_app.py
```

The host console opens automatically at `http://localhost:8000/host`. The
terminal also prints the public `https://...ngrok...` URL when ngrok is
enabled. Share that URL with friends. The client control page is available at:

```text
https://YOUR_NGROK_URL/client
```

The host page opens directly—there is no host sign-in step and no project ID to
enter.

## Client workflow

1. Run only `python client_app.py`. The local client page opens automatically.
2. Enter a name (for example `Client 1`) and the host's ngrok URL, then click
   **Connect and request access**. The project is selected automatically.
3. The page shows the client's real available RAM/CPU when the local agent is
   running. Enter the RAM and CPU amount to dedicate.
4. Select the assigned `split_data/client_N.csv`. The file is copied only to
   this laptop; it is never uploaded to the host.
5. The host approves the request by the displayed name in `/host`.
6. After approval, the native worker starts automatically. No project ID,
   generated command, username, or password is needed.

The native client detects available RAM/CPU with `psutil` when installed and
uses the contribution values entered in the page. It trains locally, encrypts
model weights, and sends only the update to the host.

For a teammate's machine, create an untracked `client.env` containing the
same `FL_ENCRYPTION_KEY` from the host's `.env` (share that one value privately).
The client launcher automatically loads `client.env` when `.env` is absent, so
the teammate can then run:

```powershell
python client_app.py
```

The ngrok token and host JWT secret do not need to be shared with clients.

## Make the host a participant

In the host console, enter the host's available and dedicated resources and
click **Start host participant**. It uses `split_data/client_1.csv`, appears in
the approval table as `Host`, and starts real local training after approval.

## What the host sees

The host page updates over a WebSocket stream and shows:

- pending and approved participants;
- each participant's available and dedicated resources;
- local dataset metadata without raw client rows;
- the current round's per-client depth and selected layers;
- the exact resource-to-depth formula;
- submitted/expected updates and completed-round history;
- host validation dataset upload and round settings.

The underlying round remains the existing FedAvg + momentum + optional server
validation pipeline. If fewer clients are available, set the minimum updates
per round in the host console to the number you want to wait for.
