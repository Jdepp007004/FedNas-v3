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
| `heterofl` | Static client tiers, ordered nested width/depth prefixes, coordinate-wise partial aggregation | Heterogeneous local models without rotation |
| `fjord` | Ordered prefix dropout over widths on each tier and temperature-2 self-distillation from the tier's largest ordered subnet | Ordered knowledge preservation for small targets |
| `fedrolex` | Contiguous rolling channel windows with coordinate-wise placement during load, update, clipping, and aggregation | Coverage of parameters neglected by static prefixes |
| `maxnet` | Rotating capacities plus a cosine-decayed largest-subnet aggregation priority | Coverage-weighted supernet training |
| `scaffold` | Full-model server/client control variates and corrected local gradients | Client-drift correction alone |
| `cc_efl` | FedRolex rolling windows + MaxNet coverage weighting + FedProx regularization | Integrated method proposed for this project |

`static`, `elastic`, and `elastic_scaffold` are compatibility aliases for the
older experiment names. Pure `fedavg` now has `mu=0`; FedProx is reported
separately so the baseline cannot silently benefit from regularization.
The current experiments keep SCAFFOLD as a distinct ablation: its correction
was unstable when combined with rolling fixed-subnet windows on this backbone,
so CC-EFL does not make an unsupported client-drift claim.

## Hardware path

The server owns the full supernet. A client receives a `FixedElasticMLP` or
`FixedCifarCNN` containing only its assigned depth/width tier. This means the
reported parameter, FLOP, and payload savings correspond to an actually
materialized smaller model, not merely a full model with masked computation.
For FedRolex, the local tensor is still the same size; only its channel window
is moved over the server model.

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
