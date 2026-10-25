# Non-medical federated supernet benchmark

This is an isolated research track for the existing federated-learning platform. It uses the same overall workflow—local client training, partial model updates, server aggregation, and resource-aware subnet selection—without clinical data.

## What is being tested

The current production prototype uses a depth-only prefix of a fully connected supernet, `BatchNorm1d`, partial updates, and a NAS score based on update magnitude. That combination is useful as a baseline, but it has four accuracy risks:

1. BatchNorm running statistics are not generally safe to average across non-IID clients.
2. A client that trains depth 2 always updates the first two layers, so deeper layers can be stale or underrepresented.
3. The current server NAS proxy rewards large parameter movement rather than measured validation performance.
4. A subnet is selected by hardware rules, but the current API returns the project-wide recommended depth rather than a persisted per-client architecture.

The `legacy` benchmark reproduces the first design pattern with a single classification head. The `elastic` benchmark keeps the nested supernet/subnet idea but adds:

- ordered depth and width prefixes;
- `LayerNorm`, which has no cross-client running statistics;
- residual connections for the deeper path;
- shape-safe aggregation for partial tensors;
- fixed client capability profiles shared across comparison methods;
- channel-window rotation within each client's fixed capability;
- MaxNet-inspired largest-subnet priority with cosine decay;
- validation-based subnet choice with a FLOP tie-breaker.

The benchmark is intentionally small and in-memory. The revised elastic path
now materializes only the selected fixed subnet on a client; the full
supernet remains server-side. It is for architecture and systems evidence.
The security switches below are auditable simulations for comparative runs,
not a production cryptographic implementation: secure aggregation derives
pairwise masks from X25519 shared secrets and currently requires all
announced clients to finish, while DP reports a deliberately conservative
sequential Gaussian bound without sampling amplification.
Use `--secure-aggregation-isolated` to move private keys into separate local
client processes for a stronger protocol harness; the federated benchmark
coordinator still has plaintext updates because local training itself remains
in-process.

The paper track exposes the reference mechanisms as separate, auditable
strategies: `fedavg`, `fedprox`, `heterofl`, `fjord`, `fedrolex`, `maxnet`,
and the integrated `cc_efl` method. SCAFFOLD is evaluated separately with an
optimizer-matched SGD comparison (`fedavg`, `fedprox`, `scaffold`); it is not
mixed into the primary Adam comparison. Matched controls include
`cc_efl_no_prox`, `cc_efl_uniform`, and `cc_efl_random_window`; the last keeps
the same per-client model sizes while randomizing channel offsets. The
`cc_efl_fjord` extension adds FjORD-style ordered-width distillation to
CC-EFL and remains exploratory until its matched results are reviewed.
`cc_efl_warmup_fjord` uses a 20% stable-prefix warmup before rolling windows,
with the same local distillation; it is also exploratory. The
`cc_efl_dynamic` variant preserves the rotating-capacity schedule as a
separate diagnostic, not as the fixed-device primary setting. The older names
`static`, `elastic`, and `elastic_scaffold` remain available for compatibility.
Every run records its client capacity map, parameter-coordinate coverage,
per-capacity validation-selected models, confidence intervals, software
versions, and split fingerprints. FLOPs are per-example forward estimates;
payload-byte and local wall-clock measurements are reported separately.
Result summaries separate full-supernet test performance from fixed-capacity,
validation-selected deployment performance. A sliced FedAvg subnet is a
deployment reference, distinct from the full-model FedAvg score.

The implementation is faithful to the algorithmic mechanisms, not a claim of
bit-for-bit reproduction of the papers' original CNN/RNN code. Our shared
backbone is deliberately the existing small nested MLP/CNN so that every
comparison uses the same data, optimizer budget, and hardware accounting.
See [`ARCHITECTURES.md`](ARCHITECTURES.md) for the exact mapping and limits.

## Dataset matrix

| Dataset | Task and stress case | Size | Default model configuration |
|---|---|---:|---|
| Digits | 10-class image classification; offline smoke test | 1,797 | input 64, depth 1–4, widths 16/32/48/64, 6 clients |
| UCI Adult | Binary income classification; mixed categorical/numeric features and demographic label skew | 48,842 | capped at 12,000 train rows, depth 1–4, widths 32/64/96/128, 8 clients |
| UCI Bank Marketing | Binary subscription prediction; mixed business features and severe class imbalance | 45,211 | capped at 12,000 train rows, depth 1–4, widths 32/64/96/128, 8 clients |
| UCI HAR | Six-class smartphone activity recognition; 561 numeric sensor features and client label skew | 10,299 | capped at 12,000 train rows, depth 1–5, widths 24/48/72/96, 8 clients |
| CIFAR-10 | 10-class natural-image classification; CNN capacity and label skew | 60,000 | capped image subset, depth 1–4, channels 16/32/48/64, 4 clients |
| CIFAR-100 | 100-class natural-image classification; harder fine-grained image task | 60,000 | same small residual CNN search space |

Dataset sources: [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult), [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing), and [UCI Human Activity Recognition](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones). UCI reports the corresponding instance counts and task descriptions. Digits is loaded from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

CIFAR is loaded through [torchvision's CIFAR datasets](https://pytorch.org/vision/stable/datasets.html). The default image run uses a fixed 2,000-image training cap and 1,000-image test cap for a fast architecture check; increase both caps for the final paper rerun.

The tabular harness uses deterministic Dirichlet label-skew partitions; the
CIFAR harness applies the same partition rule to image labels. HAR includes
subject identifiers in the source archive, but this first pass does not expose
them as a partition key; a subject-held-out split should be added if the
benchmark is used to study feature or subject shift specifically.

The UCI archives are not checked into this repository. Use `--download` for an explicit download, or run Digits first for an offline smoke test.

## Security and robustness switches

The runner can produce controlled threat-model evidence without changing the
ordinary accuracy path:

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 10 --strategies fedavg --seeds 1 2 3 --attack-fraction 0.25 --attack-type sign_flip
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 10 --strategies fedavg --seeds 1 2 3 --attack-fraction 0.25 --attack-type sign_flip --defense trimmed_mean
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 10 --strategies fedavg --seeds 1 2 3 --secure-aggregation --dp-noise-multiplier 0.5 --dp-clip-norm 1.0
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 10 --strategies fedavg --seeds 1 2 3 --secure-aggregation --secure-aggregation-isolated
```

`trimmed_mean` is a coordinate-wise baseline for same-shape FedAvg updates;
`flame` is a compact cosine-cluster filtering analogue. The latter two are
robustness baselines, not claims that the platform has solved Byzantine FL.
Backdoor runs additionally report attack success rate on a fixed trigger.
The isolated secure-sum flag runs one client worker per local process. It
isolates client private keys from the coordinator in the protocol harness,
but it is still not a production network deployment and does not implement
dropout recovery.

## CIFAR commands

```text
python -m non_medical_benchmark.run_cifar_benchmark --dataset cifar10 --download --rounds 20 --max-train 10000 --max-test 5000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/cifar10_paper.json
python -m non_medical_benchmark.run_cifar_benchmark --dataset cifar100 --download --rounds 20 --max-train 10000 --max-test 5000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/cifar100_paper.json
```

## Server-model width scaling

The scaling runner compares an expanded server supernet with a fixed client
capacity menu. In `fixed` mode, server width grows while client widths remain
at the dataset's original capability tiers. `proportional` mode grows both;
that mode does not test a constant client budget. Results include capacity-
selected utility, per-client active parameters and FLOPs, tensor payload,
supernet coordinate coverage, and simulator wall-clock time. Weight bytes and
FLOPs are estimates, and local wall time is not an edge-device measurement.

```text
python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 1 4 --capacity-scaling fixed --rounds 8 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex maxnet cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fixedclient_widthscale_1_4_8r_5seeds.json
python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 8 --capacity-scaling fixed --rounds 8 --seeds 1 2 3 --strategies fedavg heterofl fjord fedrolex maxnet cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fixedclient_widthscale_8_8r_3seeds.json
```

The current exploratory results and coverage limitation are summarized in
[`RESULTS.md`](RESULTS.md).

## Reproducible environment

The benchmark was run with Python 3.13.7, NumPy 2.2.6, pandas 2.3.3,
scikit-learn 1.7.2, PyTorch 2.9.0 CPU, torchvision 0.24.0 CPU,
cryptography 46.0.3, and pytest 8.4.2. The pip-installable pins are in
`non_medical_benchmark/requirements-lock.txt`; select the matching CPU
PyTorch wheel index for the machine before installing the two PyTorch pins.
The repository test configuration uses a workspace-local fixture factory and
disables pytest's cache provider to avoid machine-specific temporary-directory
permission failures.

## Research evidence

### Supernet/subnet and federated model heterogeneity

- [Once-for-All (OFA), ICLR 2020](https://arxiv.org/abs/1908.09791) trains one elastic network and specializes depth, width, kernel size, and resolution after training. It reports up to 4 percentage points over MobileNetV3 at comparable edge settings and substantially lower search/training cost. The relevant lesson is progressive shrinking and post-training specialization—not simply turning off later layers.
- [HeteroFL, ICLR 2021](https://arxiv.org/abs/2010.01264) sends different-width client subnetworks and aggregates the corresponding parameter portions into a larger global model. It demonstrates five computation levels across three architectures and three datasets. It establishes feasibility and communication savings, but the static prefix can leave upper layers trained less often.
- [FjORD, NeurIPS 2021](https://arxiv.org/abs/2102.13451) uses ordered dropout to tailor model width to client capacity and adds self-distillation. Its paper reports smoother convergence and accuracy gains over federated dropout across CNN and RNN tasks; it also explicitly studies keeping smaller submodels accurate.
- [FedRolex, NeurIPS 2022](https://arxiv.org/abs/2212.01548) rotates which portions of the server model each client receives. Its selective averaging is per-parameter based on the clients that updated that parameter. In the reported high-heterogeneity table, FedRolex reaches 69.44% vs. HeteroFL's 63.90% on CIFAR-10 and 56.57% vs. 52.38% on CIFAR-100; the low-heterogeneity figures are 84.45% vs. 73.19% and 58.73% vs. 57.44%, respectively. This is the strongest direct argument against always training only the first prefix.
- [Once-for-All Federated Learning, KDD FL4Data-Mining 2023](https://openreview.net/pdf?id=aJhe-VC0Ue) adapts progressive pruning and local joint optimization to FL. The authors report that optimizing all subnet sizes together gives the highest average subnet accuracy and the lowest variance in their MNIST/CIFAR-10 studies. Its key warning is that smaller subnets can be neglected unless the local loss explicitly trains them.
- [SuperFedNAS / MaxNet, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10165.pdf) is the closest match to this track. It samples smallest, largest, and random subnets, prioritizes coverage of shared weights, and uses a decaying largest-subnet aggregation weight. On its CIFAR-10/100/CINIC-10 table, the 0.45–0.95B MAC results are 89.42/56.35/73.12% for SuperFedNAS versus 85.25/43.19/61.76% for FedAvg. The paper reports up to 37.7% higher accuracy at the same MACs, up to 8.13x fewer MACs at the same accuracy, and an 11x reduction in training cost for 20 deployment targets. It also shows naive supernet FL can converge more slowly and underperform FedAvg, so the supernet alone is not enough.

### Pretrained-text adapter benchmark

An additional UCI SMS experiment uses [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) with rank-window LoRA adapters on the final attention block and a trainable classifier head. The 4.39M-parameter base encoder remains frozen; the frozen prefix is cached locally per client partition. In five seeds × nine strategies, no strategy comparison survives Holm correction. Heterogeneous methods reduce estimated float32 adapter/head update payload by 44.4% per round relative to rank-4 FedAvg, excluding the frozen base. The initial Dirichlet split has substantial quantity skew (mean client-size CV 0.927). A post-hoc balanced-size sensitivity run reduces this CV to 0.00072 but still finds no corrected strategy or protocol effect; because it also changes class allocation, it is not a pure quantity-skew intervention. See RESULTS.md and RESEARCH_PROGRESS.md for full results, limitations, and reproduction commands.

### Normalization warning

[FedBN, ICLR 2021](https://arxiv.org/abs/2102.07623) shows why the current `BatchNorm1d` choice deserves an ablation: it keeps BN parameters local and averages only the other parameters under feature-shift non-IID data. The revised benchmark uses LayerNorm instead, avoiding running-stat synchronization entirely. If a later CNN track needs BatchNorm, use a FedBN-style local-stat policy or compare GroupNorm/LayerNorm explicitly.

## Run it

From the repository root:

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 3
python -m non_medical_benchmark.run_benchmark --dataset adult --download --rounds 3
python -m non_medical_benchmark.run_benchmark --dataset bank_marketing --download --rounds 3
python -m non_medical_benchmark.run_benchmark --dataset har --download --rounds 3
```

For a faster smoke check:

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 1 --clients 3 --variants elastic --max-samples 600
```

Paper-style multi-seed run on the offline dataset:

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 20 --seeds 1 2 3 4 5 --output .benchmarks/digits_paper.json
```

Deadline stress test:

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 20 --seeds 1 2 3 --dropout-rate 0.2 --straggler-rate 0.2 --output .benchmarks/digits_deadlines.json
```

Each run reports validation and final-test accuracy for every depth/width candidate, selected subnet size, macro-F1, approximate FLOPs, parameter count, client loss, and MaxNet beta. The final test score is reported for auditability but is not used to choose the subnet; selection uses validation accuracy and then chooses the smallest subnet within 0.2 percentage points of the best validation result.

## Decision rule before production integration

Run each dataset with five seeds and compare `legacy` and `elastic` on the same partitions. Accept the revised architecture only if it improves mean validation accuracy or macro-F1 without a material increase in communication/compute, and if the smallest selected subnet stays within 1 percentage point of the selected full model. If it wins on only one dataset, keep the production model unchanged and treat the result as dataset-specific evidence.

If it wins consistently, the next integration step is to adapt the existing API payload to carry `{depth, width, subnet_version}` and move `aggregate_subnet_updates` into the server aggregation layer. Do not reuse the current update-magnitude NAS score as the production selector. See `PAPER_PLAN.md` for the paper hypothesis, threat model, metrics, and acceptance rule.
