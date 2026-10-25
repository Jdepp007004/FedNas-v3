# Run the real host/client platform

This is the runnable code path for the platform. It is separate from the
demo/site project and uses the existing FastAPI server plus the native Python
client trainer.

To let users define a new CSV prediction task, open `/create` from the host
machine after starting the server. That flow creates a tabular workspace with
an elastic model; it does not replace the original clinical workspace.

## Dataset recommendation

Use the four checked-in files in `split_data/`. They are the standalone client
inputs and are already shaped for this platform's feature schema. The optional
full dataset is not required to run the host or clients.

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
pip install -r requirements.txt
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
   **Connect**. Select the intended workspace in the project list. A self-serve
   task shows its type and target column.
3. The page shows the client's real available RAM/CPU when the local agent is
   running. Enter the RAM and CPU amount to dedicate.
4. Select your local CSV and click **Request access / save allocation**. The
   file is copied only to this laptop; it is never uploaded to the host.
5. The host approves the request by the displayed name in `/host`.
6. The native worker starts for the selected workspace and trains after
   approval. No project ID, generated command, username, or password is needed.
7. After the final round, an approved participant can choose a separate labeled
   test CSV in the client page and calculate accuracy/macro-F1 or RMSE locally.
   The test file stays on that device; the client downloads only its assigned
   tabular model subnet.

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

## Create and train a user-defined tabular task

1. On the host, open `/create` (or choose **Create a tabular model** in `/host`).
2. Enter a workspace name, choose classification or regression, type the exact
   target column, and upload a host setup CSV. The platform uses a fixed 80/20
   split: reference rows set shared numeric statistics and class vocabulary,
   while the held-out slice reports global validation metrics. Keep this file
   separate from participant training rows.
3. Share the host URL and encryption key with participants as described above.
   Each participant chooses this workspace, attaches a local CSV with matching
   column names and target labels, and requests access.
4. Approve the participants in `/host`. Set the minimum updates per round to
   the number of clients you want the coordinator to wait for (for example, 4
   for a four-device round).
5. Watch the live assignment and round history. The validation table reports
   accuracy and macro-F1 for classification, or RMSE for regression.

The generated MLP has six maximum layers, 512 hashed input features, and a
maximum hidden width of 512 (about 1.6 million parameters for a 32-class task).
CPU/RAM budgets select nested prefixes; narrower clients send only the weights
for the layers and width they trained, while the coordinator preserves
uncovered coordinates. The self-serve mode supports up to 500 input columns,
32 classification labels, and 200 MB host CSVs. It handles numeric and
categorical input columns with one shared deterministic feature map. It is a
tabular starter model, not an arbitrary PyTorch model upload service.
