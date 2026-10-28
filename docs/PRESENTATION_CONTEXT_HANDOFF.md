# Presentation context handoff: Federated Learning Platform

This document is a briefing for preparing a presentation about this repository. It collects the product story, architecture, benchmark evidence, demo flow, and claims that need careful wording. Use it as the presentation source of truth alongside the linked benchmark reports.

## Project in one sentence

The Federated Learning Platform is a web-based prototype for coordinating tabular federated learning across participants with different CPU and memory budgets, so each participant can train a smaller model on local CSV rows and contribute model updates to a shared coordinator.

## Presentation message

Show the practical system and its trade-off: smaller, resource-sized participant models can train and contribute useful updates, but the current prefix schedule does not adequately cover a large shared model. Present the product workflow and the measured results separately from claims about privacy, clinical deployment, or large language models.

Suggested title, retained from the Phase 3 review: **E2E Federated Learning Platform for Heterogeneous Systems**.

Motivation: hospitals or other data owners may want to collaborate on a prediction task without pooling their participant training rows. This is a motivating scenario, not a claim that the system has been deployed at hospitals.

## Product and architecture

The repository has two related workflows:

- A fixed clinical workflow based on a shared schema and participant CSV partitions.
- A self-serve tabular workflow where a host creates a classification or regression task, chooses a target and model size, then invites participants with matching local CSV schemas.

Main components:

1. **Host/coordinator:** FastAPI service and host dashboard. The host creates a project, uploads a setup CSV for preprocessing and validation, reviews participant access, chooses round settings, monitors updates, and can export a trained tabular model bundle.
2. **Participant workspace:** Browser UI served through the local client launcher. Participants connect to a host, inspect projects and schemas, select their local CSV, declare resources, and request access.
3. **Native client worker:** Python process that reads the selected CSV locally, receives its assigned model prefix, trains locally, and returns an encrypted model update. It can use local system information through psutil when available.
4. **Resource planner and model:** A shared elastic multilayer perceptron (MLP) supports depth and width prefixes. Participants receive smaller prefixes according to the resources they dedicate.
5. **Aggregation and evaluation:** The coordinator sample-weights each shared coordinate using only the clients that updated it, retains previous values for uncovered coordinates, and reports full-model and participant-sized validation metrics separately.

The client-side CSV stays on the participant machine in the documented workflow. The host does receive metadata such as schema and row count, plus model updates. For self-serve tasks, the host setup CSV is uploaded: an 80% reference split is used to create shared preprocessing statistics and the class vocabulary, and the remaining 20% is retained for host-side validation. Use public or synthetic setup data for the demo.

## Self-serve tabular method

- The host selects classification or regression, the target column, and the shared model's maximum depth and width. The web defaults are 3 layers and width 64; supported ranges are 2–6 layers and width 16–512.
- Numeric columns use shared center/scale values. Categorical columns use deterministic signed feature hashing into 512 input features.
- For classification, the host data defines a stable class vocabulary so clients map labels consistently.
- Each participant trains only its assigned depth/width prefix. The worker constructs a compact subnet instead of a full local copy of the supernet.
- The server aggregates only coordinates actually returned by clients, weighted by local sample count. Uncovered coordinates retain their prior global value.
- Fan-in compensation scales narrower subnet inputs so they can use weights initialized for the maximum width.
- The host dashboard distinguishes full-supernet validation from the average score of participant-sized models. These metrics are different endpoints and must not be conflated.
- A completed tabular model can be exported with weights, preprocessing configuration, class mapping, manifest, and a portable prediction script. The bundle excludes participant training CSVs.

The current starter supports tabular CSVs; it does not execute uploaded model code or accept arbitrary model architectures. Documented limits include up to 500 feature columns, 32 classification labels, and a 200 MB host upload.

## Privacy and security wording

Use precise statements:

- Participant training rows are processed locally; the participant client sends schema/row metadata and model updates.
- The application encrypts model updates with AES-256-GCM for transport. The coordinator decrypts updates in order to aggregate them.
- The separate test CSV evaluation path runs locally on the participant device.
- The host setup CSV is uploaded, and its held-out validation slice is kept at the coordinator.

Do **not** claim that this product workflow provides secure aggregation, formal differential privacy, HIPAA compliance/certification, or an operational hospital deployment. The current workflow's encryption does not hide individual updates from the coordinator. Although the repository contains Phase 3 security modules and a HIPAA-oriented design document, the capstone workflow documentation explicitly says there is no production privacy certification. Treat compliance/security modules as implementation material, not proof of audited compliance.

## Primary capstone benchmark: website method

This is the most directly relevant quantitative result for a presentation about the working self-serve website.

**Task and protocol**

- Predict cancer dataset target vital_status (alive/dead).
- Four existing participant partitions: 2,451 participant training rows total.
- Host setup data: 817 rows, split into 653 reference rows and 164 host-validation rows.
- Independent test set: 820 rows.
- There are 66 original input columns, mapped into 512 features; the target has two classes.
- 6-layer, width-512 shared MLP: 1,576,962 parameters.
- Four worker processes, 8 rounds, 3 local epochs, 5 seeds (1–5).
- One Intel Core Ultra 5 125H workstation, Windows 11, CPU-only PyTorch 2.9.0. The client processes used CPU affinity masks to simulate 1-, 2-, 4-, and 8-thread capacity tiers. They were not four physical devices.
- RAM inputs influenced the resource planner but were not enforced as process memory limits.

**Assigned participant capacities**

| Virtual profile | CPU threads | RAM planner input | Assigned model | Parameters |
|---|---:|---:|---:|---:|
| Edge | 1 | 0.5 GB | 2 layers × 16 | 8,514 |
| Low | 2 | 2 GB | 2 layers × 32 | 17,538 |
| Mid | 4 | 4 GB | 3 layers × 64 | 41,282 |
| High | 8 | 8 GB | 6 layers × 128 | 148,482 |

The full shared model has 1,576,962 parameters. The smallest participant model has 99.46% fewer parameters than the full model; the largest has 90.59% fewer. These are parameter counts and derived weight sizes, not measurements of peak RAM use.

**Independent test results**

| Method and endpoint | Accuracy, mean ± SD | Macro-F1, mean ± SD | Client dispatch/wait sum for 8 rounds, mean ± SD |
|---|---:|---:|---:|
| Full-model FedAvg, full model on every client | 0.868 ± 0.005 | 0.854 ± 0.006 | 10.56 ± 1.18 s |
| Website method, mean of four capacity-sized deployments | 0.811 ± 0.011 | 0.794 ± 0.009 | 5.39 ± 0.62 s |
| Website method, complete 6×512 global model | 0.700 ± 0.159 | 0.686 ± 0.159 | Not applicable |

The website method used about 49% less measured client dispatch/wait time across eight rounds. This timer sends local work to four worker processes and waits for them. It excludes worker startup, aggregation, validation, website/API overhead, encryption, and network transfer. The broader harness elapsed time was 26.61 seconds for the website method versus 29.75 seconds for full-model FedAvg; this noisy measurement does not establish a 49% end-to-end speedup.

The 6×512 global model had only 9.42% coordinate coverage after eight rounds and unstable test macro-F1 across seeds (0.443, 0.816, 0.808, 0.610, 0.754). The client deployment score of 0.794 and full-model FedAvg score of 0.854 are different endpoints; their 0.059 difference is descriptive, not a non-inferiority result. The five-seed intervals are exploratory and not adjusted for multiple comparisons.

The exploratory 6×128 shared-width check reached 100% coordinate coverage, but participant deployment macro-F1 averaged 0.746 versus 0.837 for full-model FedAvg; dispatch/wait time was 4.90 versus 5.12 seconds. This is a configuration check, not the primary result.

**Suggested spoken result**

“In a five-seed, four-worker simulation, the resource-sized client models reached 0.79 average macro-F1 while using up to 99.46% fewer parameters than the full model. The measured client dispatch-and-wait sum was about half of the full-model reference. The large shared model was sparsely covered, so these results support useful small-client deployments, not convergence of the complete supernet.”

## Optional separate research-track evidence

The repository also has a non-medical benchmark track on UCI Human Activity Recognition (HAR). It uses an isolated research harness and is not the primary website-method benchmark. Do not combine its figures with the cancer tabular results.

A fresh ten-seed confirmation compared full-model FedAvg, HeteroFL, and FjORD at the 8× HAR server scale (2,806,278 parameters), using four CPU-affinity-limited worker processes and eight rounds:

| Method | Capacity-selected client test macro-F1 | Full shared-model test macro-F1 | Median 8-round simulation time | Coordinate coverage |
|---|---:|---:|---:|---:|
| FedAvg | 0.660 (95% CI 0.602–0.717) | 0.908 | 30.52 s | 100% |
| HeteroFL | 0.848 (0.807–0.888) | 0.813 | 7.94 s | 2.23% |
| FjORD | 0.905 (0.885–0.926) | 0.811 | 9.42 s | 2.23% |

This is separate evidence that small-client deployment models can score well even when the complete supernet is not trained. These are simulated workers on one host, and the low coordinate coverage remains an important limitation. The report also contains other experiments, including Digits, whose results do not support a broad generalization claim.

## Recommended slide sequence

1. **Title and motivation:** E2E Federated Learning Platform for Heterogeneous Systems; collaboration without pooling participant training rows.
2. **Problem:** Conventional federated training can assume similar compute budgets; smaller clients may not run the full shared model.
3. **Goal and product:** A host creates a tabular project and participants join using locally held CSVs and their declared CPU/RAM budgets.
4. **System architecture:** Host/coordinator, participant UI, local Python worker, resource planner, elastic model, partial-update aggregation.
5. **Training method:** Show how resources select depth/width prefixes, how only touched coordinates are returned, and how the coordinator merges updates.
6. **Demo flow:** Create a vital_status classification workspace; use public/synthetic host setup CSV; add four participant partitions; approve participants; complete rounds; inspect resource assignments and both validation metrics; export model and run a prediction.
7. **Benchmark setup:** State 4 virtual CPU profiles, one workstation, 8 rounds × 3 local epochs, five seeds, and the full-model FedAvg reference.
8. **Results:** Show client parameter counts, capacity-sized client performance, and the measured dispatch/wait sum. Keep client deployments and the complete global model in separate rows.
9. **Limitations:** Simulated devices; declared RAM not enforced; one primary tabular task; no real network, energy, or peak-memory measurement; incomplete full-supernet coverage; no claim of secure aggregation or compliance certification.
10. **Conclusion and next work:** The system demonstrates resource-sized tabular participation. Improve coordinate coverage and validate with multiple datasets and separate physical devices with measured memory, network, and energy.

## Demo preparation checklist

- Start the host with python host_app.py and open /host; use /create for a self-serve task.
- Start participant workers using python client_app.py.
- Use a public or synthetic host setup CSV. Do not use participant CSVs as the host setup upload.
- Use the four files under split_data/ as the documented ready-to-share client partitions. Keep each participant file on its owner's machine.
- Approve all four participants and set minimum updates per aggregation round to four if the goal is to wait for every client.
- Show per-client assigned capacity and both participant-sized and full-model validation results.
- If the session has not completed rounds, present the benchmark report as measured research evidence rather than implying those results came from the live demo.

## Current working-tree changes to account for

At the time this handoff was prepared, the existing uncommitted work updates six files:

- client/client.html: expanded participant workspace and project views.
- server/templates/host.html: updated host console interface.
- server/templates/demo.html: guided walkthrough presentation page.
- server/main.py: /walkthrough route; /demo points to it; /dashboard serves the host console and the prior summary has moved to /dashboard/summary.
- server/host_router.py: participant-submitted tabular project request/review endpoints and benchmark summary.
- server/project_router.py: optional dataset metadata and required-column validation on project join.

These changes had not been tested as part of preparing this handoff. The handoff should describe them as current repository changes unless they are verified separately.

## Source files for further detail

- docs/CAPSTONE_FINAL_STORY.md: intended capstone narrative, slide order, and guardrails.
- docs/SELF_SERVE_TABULAR_TRAINING.md: self-serve workflow, model method, data and privacy contract.
- docs/MODEL_EXPORT.md: downloaded model bundle and prediction usage.
- RUN_LOCAL.md: host/client setup and demo flow.
- non_medical_benchmark/WEBSITE_METHOD_BENCHMARK.md: exact primary capstone benchmark protocol and results.
- non_medical_benchmark/FOUR_DEVICE_BENCHMARK.md: separate HAR research-track results and limitations.
- PHASE_3_MASTER.md and SECURITY.md: security modules and project security design; do not interpret these documents alone as an audited compliance claim.
