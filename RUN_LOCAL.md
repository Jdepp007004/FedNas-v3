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
fresh clone, create `.env` from `.env.example` and fill in the same values.
Replace the development secrets before using this beyond a trusted test:

```powershell
$env:JWT_SECRET = "replace-with-a-long-secret"
$env:HOST_USERNAME = "host"
$env:HOST_PASSWORD = "replace-with-a-host-password"
$env:FL_ENCRYPTION_KEY = (python -c "from shared.encryption import generate_key_b64; print(generate_key_b64())")
$env:NGROK_AUTH_TOKEN = "your-ngrok-auth-token"
```

`NGROK_AUTH_TOKEN` may be left unset for local-only testing. The configured host
login for this workspace is `host` / `FedNasHost_2026!`.

## Start the host

From the repository root:

```powershell
python host_app.py
```

Open `http://localhost:8000/host` on the host machine. The terminal prints the
public `https://...ngrok...` URL when ngrok is enabled. Share that URL with
friends. The client control page is available at:

```text
https://YOUR_NGROK_URL/client
```

Friends can also open `client/client.html` locally and enter the same URL.

The default development host login is `host` / `hostpass` unless you set
`HOST_USERNAME` and `HOST_PASSWORD`.

## Client workflow

1. Enter the host's ngrok URL and connect.
2. Register and log in.
3. Choose the project.
4. Select the assigned `split_data/client_N.csv`. The browser reads it locally
   and sends metadata only.
5. Set available and dedicated RAM/CPU, then request access.
6. The host approves the request in `/host`.
7. Run the native trainer command shown by the client page.

Example for a teammate:

```powershell
python client_app.py `
  --server https://YOUR_NGROK_URL `
  --username friend_1 `
  --password choose-a-password `
  --hospital "Friend 1 laptop" `
  --email friend1@example.com `
  --csv split_data\client_2.csv `
  --proj PROJECT_ID `
  --dedicated-ram 4 `
  --dedicated-cores 2
```

The native client detects available RAM/CPU with `psutil` when installed and
posts its explicit contribution cap. It trains locally, encrypts model
weights, and sends only the update to the host.

For a teammate's machine, create an untracked `client.env` containing the
same `FL_ENCRYPTION_KEY` from the host's `.env` (share that one value privately)
and run:

```powershell
$env:FL_ENV_FILE = "client.env"
python client_app.py ...
```

The ngrok token and host JWT secret do not need to be shared with clients.

## Make the host a participant

The host console includes a ready-made command. Run it in a second host
terminal after approving the host participant:

```powershell
python client_app.py `
  --server http://localhost:8000 `
  --username host_client `
  --password choose-a-password `
  --hospital "Host machine" `
  --email host@local `
  --csv split_data\client_1.csv `
  --proj PROJECT_ID `
  --dedicated-ram 4 `
  --dedicated-cores 2
```

This is a real local training worker, not a browser simulation. Approve it in
the host console like any other participant.

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
