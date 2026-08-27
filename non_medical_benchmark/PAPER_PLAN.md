# Paper track: resource-aware elastic federated learning

## Working claim

### Proposed paper framing: CC-EFL

We call the integrated system **Coverage-Controlled Elastic Federated
Learning (CC-EFL)**. Its single design principle is to treat a heterogeneous
federated supernet as a coverage problem: every client trains only the model
slice its hardware can afford, the slice position rotates so the server model
is fully covered, coverage weights protect the largest deployable subnet from
being starved, while FedProx stabilizes the residual drift caused by non-IID
clients. Secure aggregation, clipping/DP, and robust aggregation are explicit
system layers around this training protocol, with their utility and overhead
reported separately.
The paper should present this as a systems integration and measurement claim,
not as a claim that any individual ingredient is new.

An elastic federated model can improve the accuracy/communication/compute
trade-off under non-IID data and heterogeneous client budgets when four
conditions are enforced together:

1. Clients materialize only their assigned subnet.
2. Subnet assignments rotate so upper layers and wider channels are not
   permanently starved.
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
| `scaffold` | Full-model SCAFFOLD control variates. |
| `cc_efl` | Rolling + coverage weighting + FedProx regularization. |

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
- per-client and per-capacity-tier accuracy;
- total uploaded/downloaded bytes per round;
- active FLOPs, parameter count, and peak memory on each client tier;
- convergence rounds and wall-clock time;
- participation, simulated deadline misses, and recovery behavior;
- ablations for normalization, residuals, width, rotation, MaxNet weighting,
  control variates, and fixed-subnet materialization.
- a pre-registered run manifest containing software versions, seeds, caps,
  partition alpha, learning rates, local epochs, and selection metric;

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
