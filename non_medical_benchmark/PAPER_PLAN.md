# Paper track: resource-aware elastic federated learning

## Working claim

### Proposed paper framing: CC-EFL

We call the integrated system **Coverage-Controlled Elastic Federated
Learning (CC-EFL)**. Each client receives a fixed capability tier for a run;
the channel window rotates within that tier so different regions of the
server model receive updates. Coverage weights protect the largest available
client tier from being starved, while FedProx stabilizes residual drift under
non-IID data. The full-width model remains a separate reference baseline.
Secure aggregation, clipping/DP, and robust aggregation are explicit system
layers around this training protocol, with their utility and overhead reported
separately.
The paper should present this as a systems integration and measurement claim,
not as a claim that any individual ingredient is new.

An elastic federated model can improve the accuracy/communication/compute
trade-off under non-IID data and heterogeneous client budgets when four
conditions are enforced together:

1. Clients materialize only their assigned subnet.
2. Each client's depth/width budget stays fixed across rounds, while its
   channel window rotates without exceeding that budget.
3. Coverage weighting gives the largest subnet enough signal early while
   preserving smaller tiers later.
4. FedProx regularization stabilizes local optimization under the remaining
   non-IID drift; SCAFFOLD is reported as a separate ablation.

The claim is intentionally narrower than “solving federated learning.”
Encryption, secure aggregation, differential privacy, and Byzantine defense
are evaluated as system layers with their own threat models and costs.

## Experimental factors

### Strategies

| Strategy | Purpose |
|---|---|
| `fedavg` | Full-model accuracy and compute reference. |
| `fedprox` | Full-model proximal baseline. |
| `heterofl` | Static heterogeneous nested prefixes. |
| `fjord` | Ordered dropout and self-distillation. |
| `fedrolex` | Rolling channel-window extraction. |
| `maxnet` | Coverage-weighted elastic supernet training. |
| `scaffold` | Full-model SCAFFOLD control variates; compare separately with FedAvg and FedProx under matched SGD (`--optimizer sgd`). |
| `cc_efl` | Rolling + coverage weighting + FedProx regularization. |
| `cc_efl_no_prox` | CC-EFL without the proximal term. |
| `cc_efl_uniform` | CC-EFL with uniform aggregation weights. |
| `cc_efl_random_window` | Same per-client tier and resource budget as CC-EFL, with random channel offsets. |
| `cc_efl_fjord` | Exploratory CC-EFL extension adding FjORD-style ordered-width local self-distillation. |
| `cc_efl_warmup_fjord` | Exploratory extension with a 20% prefix warmup, systematic rolling windows, and FjORD-style local distillation. |
| `cc_efl_dynamic` | Historical rotating-capacity schedule; diagnostic only. |

### Datasets

The default matrix is Digits, UCI Adult, UCI Bank Marketing, UCI HAR, and
CIFAR-10/100. Digits is the offline smoke test; the UCI and CIFAR datasets
must be downloaded explicitly. Each run uses deterministic preprocessing and
Dirichlet label skew. CIFAR uses a small residual CNN with the same nested
depth/width client contract, so comparisons to HeteroFL, FjORD, FedRolex, and
SuperFedNAS are not limited to tabular/MLP tasks.

### Required reporting

- mean, sample standard deviation, and two-sided 95% Student-t confidence
  intervals over at least five seeds;
- paired per-seed deltas against FedAvg, not only unpaired averages;
- accuracy and macro-F1 on the final held-out test set;
- per-client and per-capacity-tier accuracy and macro-F1;
- training parameters and per-example forward FLOPs by client tier;
- upload/download payload bytes per round and total per run;
- selected deployed parameters and per-example inference FLOPs by capacity;
- parameter-coordinate coverage and update-frequency distribution;
- convergence rounds and wall-clock time;
- participation, simulated deadline misses, and recovery behavior;
- ablations for normalization, residuals, width, systematic versus random
  windows, MaxNet weighting, FedProx, control variates, and fixed-subnet
  materialization.
- a pre-registered run manifest containing software versions, seeds, caps,
  partition alpha, learning rates, local epochs, and selection metric;

The primary elastic and baseline matrix uses Adam. SCAFFOLD is excluded from
that matrix because its control-variate update requires SGD; run a separate
three-way `fedavg`, `fedprox`, and `scaffold` comparison with
`--optimizer sgd` so all three methods share the same learning rate and
momentum.

Report two accuracy views separately: the full-supernet test score (the
FedAvg/FedProx full-model reference) and the validation-selected deployment
score averaged over the fixed client-capacity profile map. A capacity-selected
FedAvg score is a post-training subnet slice of the full model; label it as
such, and do not call it the full-model FedAvg baseline. Report paired
capacity-selected deltas separately from comparisons against the full model.

## Threat model and privacy boundary

The base benchmark assumes an honest-but-curious server and honest clients.
The server must not receive raw client data, but local training alone does not
prevent update inference. The implemented security axis adds client-level
clipping/noise and an auditable pairwise-mask secure-sum simulator, then
reports the conservative sequential Gaussian epsilon bound, model bytes, and
key-exchange overhead. An optional process-isolated harness keeps each private
key inside a client worker. It deliberately does not claim a production
threshold protocol, tight subsampling accounting, or dropout recovery: all
announced participants must finish in this simulator. In the ordinary
in-memory FL run, the coordinator still has plaintext updates before invoking
the mask simulator; that run is a functional secure-sum test, not a privacy
proof. Secure aggregation reduces server visibility of individual updates in a
real deployment; DP is still needed to limit information released by the
aggregate model. Participation patterns and repeated rounds remain part of the
privacy discussion.

The robustness axis adds sign-flip, label-flip, and fixed-trigger backdoor
attacks, with coordinate-trimmed mean and a compact FLAME-style cosine-cluster
filter as baselines. These experiments quantify failure modes; they do not
constitute a claim of complete Byzantine resilience.

## Closest prior work

- [OFA](https://arxiv.org/abs/1908.09791): progressive shrinking for one
  elastic network across depth, width, kernels, and resolution.
- [HeteroFL](https://arxiv.org/abs/2010.01264): heterogeneous-width client
  models and partial global aggregation.
- [FjORD](https://arxiv.org/abs/2102.13451): ordered dropout and
  self-distillation for heterogeneous federated clients.
- [FedRolex](https://arxiv.org/abs/2212.01548): rolling submodel extraction
  to cover parameters that static prefixes neglect.
- [SuperFedNAS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10165.pdf):
  MaxNet sampling and largest-subnet prioritization for federated supernets.
- [SCAFFOLD](https://arxiv.org/abs/1910.06378): control variates for client
  drift under non-IID data.
- [Practical Secure Aggregation](https://arxiv.org/abs/1611.04482):
  threshold/dropout-tolerant secure summation for federated updates.
- [FLAME](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen):
  clustering, clipping, and noise for backdoor mitigation.

## Acceptance rule

The revised architecture is not accepted into the production clinical path
just because one smoke run improves. It must improve mean test accuracy or
macro-F1 across the agreed datasets/seeds while keeping the selected small
subnet within one percentage point of the selected full model and reducing
client-side parameters/communication. Any privacy or robustness claim must
also have a corresponding attack or protocol-cost experiment, and the paper
must state the simulator limitations above.

## IJCNN 2027 conference track (2026-10-25)

### Target and current claim

Target IJCNN 2027. INNS lists IJCNN meetings dating to 1989 and predecessor
IEEE neural-network conferences in 1987. The current call includes federated,
distributed, scalable, and resource-constrained learning. The regular-paper
deadline is January 31, 2027; initial submissions are anonymous IEEE conference
papers of 4--6 pages. Recheck the live call and template before submission.

The conference draft is
`submission/IJCNN2027_submission.tex`. Frame the work as a controlled
resource--utility benchmark, not a novel algorithm. The core experiment is 105
paired runs on UCI HAR: four affinity-limited workers (1/2/4/8 logical CPUs),
three server widths (96/384/768), seven methods, five seeds, and eight rounds.
At 8x, subnet methods train about 29,958 active parameters per client-round
against a 2,806,278-parameter shared MLP. CC-EFL's capacity-selected macro-F1
is 0.828, while FjORD reaches 0.902 and HeteroFL 0.862. CC-EFL has 2.6%
coordinate coverage and full-supernet macro-F1 0.829 versus 0.905 for
full-model FedAvg. The allowed claim is that CPU-limited processes can
participate in short, single-host subnet training; this is not physical-device
validation or evidence that the full expanded model converged.

A 20-run follow-up (five seeds each for FedAvg, HeteroFL, FjORD, and CC-EFL)
extends the shared MLP to 10,331,142 parameters while keeping average subnet
size at 29,958. The follow-up preserves the same result pattern: HeteroFL and
FjORD exceed CC-EFL on capacity-selected utility, and CC-EFL full-supernet F1
is 0.825 versus 0.904 for full-model FedAvg, at 0.70% coordinate coverage.
Use the `submission/README.md` and `FOUR_DEVICE_BENCHMARK.md` values for the
final paper; do not imply the 16x run repeats all seven methods.

### Fresh confirmation and contribution boundary

The 8x HAR result is now checked on ten fresh paired seeds (6--15) for
FedAvg/HeteroFL/FjORD, with a five-seed Digits workload check (seeds 6--10).
For the smallest 1-thread HAR profile, FjORD's validation-selected
13,686-parameter deployment reaches test macro-F1 0.889 (95% CI 0.855--0.923)
and accuracy 0.896; mean local round time is 0.048 s versus 1.874 s for
full-model FedAvg on that worker. Across the four profiles, FjORD deployment
macro-F1 is 0.905 (0.885--0.926), median eight-round time is 9.42 s, and
active training averages 29,958 parameters against a 2.81M-parameter server
model. Its full-supernet macro-F1 is only 0.811 at 2.23% coordinate coverage.
The descriptive deployment-versus-full-FedAvg paired difference is -0.003
(95% CI -0.021 to +0.015); it is not a formal non-inferiority test. Digits
shows the same separation more sharply: client FjORD macro-F1 0.810, versus
0.623 for its full supernet and 0.964 for full-model FedAvg.

These results strengthen the short-run small-client participation claim, but
do not establish full-global-model training or physical-device performance.
An attempted Adult CPU-affinity extension was stopped after one FedAvg run
took 104.4 s; no aggregate output is included. Inspired alternatives were
also tested but did not produce a successful method: function-preserving
growth pilots and a server replica ensemble are documented in
`submission/EXPLORATORY_METHODS.md`; layer-pair coverage routing is in
`submission/EXPLORATORY_ABLATION.md`. The paper should claim controlled
measurement and per-capacity evidence, not a novel optimizer or a fabricated
improvement.

### Failed exploratory method

An exploratory layer-pair coverage schedule was tested in a residual-free
MLP. At 8, 30, and 121 rounds it raised coordinate coverage to 18.2%, 39.5%,
and 100%, respectively, but mean deployment macro-F1 was below the paired
CC-EFL runs; the 121-round paired difference was -0.099 (95% CI -0.218 to
+0.020), with about 2.1x the median runtime. Keep this as a negative ablation,
not as a proposed contribution. Details are in
`submission/EXPLORATORY_ABLATION.md` and the raw JSON outputs.

### Submission readiness

An anonymous IEEE conference-format source and reproduction guide are in
`submission/`. The built-in LaTeX compiler returned a platform directory
error, so successful compilation and final page count remain unverified. The
CPU-affinity evidence covers HAR and a smaller Digits workload, with one host,
no per-client RAM cap, no network or energy measurements, and an MLP with at
most 10.33M parameters. The short-run global-model coverage failure remains a
central limitation. This is an empirical conference submission candidate,
not an acceptance guarantee; physical-device validation, additional full-scale
workloads, and a successful venue-template build remain open. Recheck current
author instructions and AI disclosure requirements before submission.
