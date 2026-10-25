# Self-serve tabular federated training

The host can create a task at `/create` without editing Python code. It chooses
classification or regression, names the target column, and supplies a setup
CSV that is separate from participant training data. A deterministic 80/20
split uses the reference portion for shared preprocessing statistics and keeps
the 20% validation slice on the host for coordinator-side metrics. The
coordinator initializes the global model before any participant joins. The
built-in elastic tabular supernet is loaded by default. The host can optionally
upload a compatible `.npz` state dictionary or a model ZIP exported by this
website to use its weights as the starting checkpoint. Uploads are validated
against the selected model dimensions and task; classification bundles must
also have the same ordered class labels. Uploaded executable model code and
other architectures are not supported.
Participants choose that workspace, attach their own CSV locally, request
access, and wait for host approval.

## Method: budget-conditioned elastic prefixes

Each task uses one shared MLP whose maximum depth and width the host chooses
when creating the workspace. The web defaults are three layers and 64 hidden
units; the supported range is two to six layers and 16 to 512 units. A six-layer,
512-wide model has about 1.6 million parameters at 32 output classes.
The preprocessing map is fixed by the host task: numeric columns use shared
center/scale values; categorical columns use deterministic signed feature
hashing into a 512-dimensional vector. The target vocabulary for classification
comes from the host's held-out CSV, so local class indices stay consistent.

For a participant's explicitly dedicated resources, the planner assigns a
depth and width prefix. For the elastic tabular model, width is rounded down to
a multiple of 16 using `min(16 × dedicated RAM in GB, 32 × dedicated CPU
cores)`, with a minimum width of 16 and maximum of the host-selected width.
Depth uses the existing resource-aware depth rule. A client executes only its assigned prefix and
returns the tensors for those coordinates. The coordinator averages each
coordinate over the sample-weighted clients that actually trained it and keeps
the previous global value for uncovered coordinates. This lets heterogeneous
clients contribute to one persistent model without uploading an entire
client-side model.

The coordinator also sends only the assigned prefix to a tabular participant.
The worker constructs a compact subnet containing only those parameters, not a
full local copy of the supernet. To keep narrow prefixes trainable when the
supernet was initialized for a wider fan-in, the worker scales each hidden-layer
input after the first layer, and the head input, by
`sqrt(max_width / active_width)`. This fan-in compensation preserves the
coordinator's shared parameter representation while keeping activation scale
comparable across client widths. The host dashboard reports both full-supernet
validation and the mean validation score of the participant-sized models; the
two endpoints should not be conflated when coordinate coverage is low.

This is an implemented systems method and capstone feature. It should not be
described as a proven novel optimizer or as evidence that every large model can
be trained on a phone. The current generated model is a compact tabular MLP;
the experimental HAR benchmark is a separate evaluation track.

## Data and metrics

- Participant CSV rows remain on the participant machine. The server receives
  dataset metadata and encrypted model updates.
- Only the held-out 20% host validation slice is retained for coordinator-side
  metrics. The upload is used transiently to fit preprocessing statistics on
  the other 80%; keep the complete host file separate from participant
  training data.
- The host setup CSV is uploaded to the coordinator, and its held-out rows are
  stored there for validation. For the capstone demo, use public or synthetic
  data for this setup file; do not describe this path as keeping every host
  record on premises.
- Every participant CSV must contain the configured target and the same
  feature column names. Classification labels must occur in the host vocabulary.
- Classification history reports full-supernet accuracy/macro-F1 and the mean
  score of participant-sized deployments separately. Regression history
  reports both corresponding RMSE values in the target's original units.
- The current starter supports up to 500 feature columns, 32 classification
  labels, and the configured 200 MB host upload limit. It does not execute
  uploaded model code or support image, text, or sequence models.
- The native worker limits PyTorch compute threads to the dedicated CPU count.
  RAM controls the assigned subnet size, but the process has no operating
  system memory quota and materializes its local CSV/features in RAM; keep
  client partitions sized for their machines. The worker's model weights and
  optimizer state are limited to the assigned subnet.
- Encryption protects model-update transport in this project, but the server
  decrypts updates to aggregate them. This is not secure aggregation, formal
  differential privacy, or a production privacy certification.

## Four-device session

To wait for all four participants before aggregating a round, approve all four
and set **Minimum client updates before aggregation** to `4` in `/host`. For a
smaller group, set the count to the number of active participants. The live
planner exposes each device's reported/declared RAM and CPU budget, assigned
depth and width, and the completed-round validation metrics.

The five-seed benchmark for this website method, including exact CPU profiles,
independent-test scores, timing definitions, and limits, is documented in
[`WEBSITE_METHOD_BENCHMARK.md`](../non_medical_benchmark/WEBSITE_METHOD_BENCHMARK.md).

After a self-serve task completes at least one round, the host dashboard can
download a ZIP bundle with the latest global weights, preprocessing metadata,
and a portable prediction script. See [MODEL_EXPORT.md](MODEL_EXPORT.md).

After the final round, an approved participant can reopen the page through
`client_app.py`, select a separate labeled test CSV, and calculate test
metrics. The local agent downloads only that participant's assigned tabular
subnet and computes classification accuracy/macro-F1 or regression RMSE on the
participant machine. The test CSV is never sent to the coordinator. This
reports the participant-sized deployment, not the full coordinator model.
