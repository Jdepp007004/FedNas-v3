# Architecture and method mapping

This file fixes the terminology used in the paper experiments. The project
does not claim to reproduce the original authors' entire vision or language
models; it reproduces the cited training mechanisms on the project's existing
elastic MLP and small CNN. That is the appropriate comparison for an ablation
paper because the backbone, data split, optimizer budget, and resource
measurement stay controlled.

## Reference mechanisms

| Strategy | Mechanism implemented here | What it isolates |
|---|---|---|
| `fedavg` | Full model on every client; weighted average | Accuracy/compute reference |
| `fedprox` | Full model plus `mu/2 ||w-w_t||²` during local training | Proximal stabilization under non-IID data |
| `heterofl` | Fixed per-client tiers, ordered nested width/depth prefixes, coordinate-wise partial aggregation | Heterogeneous local models without channel rotation |
| `fjord` | Ordered prefix dropout over widths on each tier and temperature-2 self-distillation from the tier's largest ordered subnet | Ordered knowledge preservation for small targets |
| `fedrolex` | Fixed per-client tier plus contiguous rolling channel windows and coordinate-wise aggregation | Systematic coverage of parameters neglected by static prefixes |
| `maxnet` | Fixed per-client tiers plus a cosine-decayed largest-tier aggregation priority | Coverage-weighted supernet training |
| `scaffold` | Full-model server/client control variates and corrected local gradients | Client-drift correction alone |
| `cc_efl` | Fixed per-client tiers + FedRolex rolling windows + MaxNet-inspired coverage weighting + FedProx | Integrated method under study |
| `cc_efl_no_prox` | CC-EFL without its proximal penalty | Isolates FedProx's contribution |
| `cc_efl_uniform` | CC-EFL with uniform rather than largest-tier-prioritized aggregation | Isolates coverage weighting |
| `cc_efl_random_window` | CC-EFL with random channel offsets at the same fixed client tiers | Matched-budget control for systematic window rotation |
| `cc_efl_fjord` | CC-EFL plus FjORD-style ordered-width dropout and local self-distillation | Exploratory extension testing whether within-client distillation improves the CC-EFL accuracy/resource trade-off |
| `cc_efl_warmup_fjord` | CC-EFL plus a 20% prefix-window warmup, followed by systematic rolling and FjORD-style distillation | Exploratory schedule testing whether stable early features help while retaining later parameter coverage |
| `cc_efl_dynamic` | Historical rotating-capacity schedule, reported separately | Sensitivity to changing tier assignments across rounds |

`static`, `elastic`, and `elastic_scaffold` are compatibility aliases for the
older experiment names. Pure `fedavg` now has `mu=0`; FedProx is reported
separately so the baseline cannot silently benefit from regularization.
The current experiments keep SCAFFOLD as a distinct ablation: its correction
was unstable when combined with rolling fixed-subnet windows on this backbone,
so CC-EFL does not make an unsupported client-drift claim.

## Hardware path

The server owns the full supernet. Each client's capability tier is assigned
once per seed and reused across methods and rounds. Heterogeneous-client tiers
are capped below the full supernet width; the full model remains a reference
baseline. A client receives a `FixedElasticMLP` or `FixedCifarCNN` containing
only its assigned depth/width tier. This means the
reported parameter, FLOP, and payload savings correspond to an actually
materialized smaller model, not merely a full model with masked computation.
For FedRolex, each client keeps the same tensor size; only its channel window
moves over the server model. The random-window control preserves each client's
depth and width schedule and changes only those window positions.

## Fidelity limits

The original papers use larger CNN/RNN families, hardware-aware search
spaces, or production-grade distributed systems. The following are therefore
not claimed here: the full SuperFedNAS architecture-search controller,
original paper-specific data preprocessing, secure-aggregation dropout
recovery, or Byzantine robustness guarantees. Those are separate future
systems contributions. The present paper claim is the measured
accuracy/resource trade-off of the combined mechanisms on a controlled shared
backbone.

## Primary references

- [HeteroFL, ICLR 2021](https://arxiv.org/abs/2010.01264)
- [FjORD, NeurIPS 2021](https://arxiv.org/abs/2102.13451)
- [FedRolex, NeurIPS 2022](https://arxiv.org/abs/2212.01548)
- [SuperFedNAS, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10165.pdf)
- [SCAFFOLD, ICML 2020](https://arxiv.org/abs/1910.06378)
- [FedProx, MLSys 2020](https://arxiv.org/abs/1812.06127)
- [Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning, arXiv 2026](https://arxiv.org/abs/2608.07157)
