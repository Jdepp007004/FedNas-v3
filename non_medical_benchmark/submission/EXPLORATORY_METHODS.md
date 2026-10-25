# Exploratory method attempts (not paper claims)

The conference-preparation work tested alternatives after the user asked for a
stronger result and a nontrivial idea. None produced sufficiently strong,
multi-seed evidence to support a new algorithm claim. The paper therefore
reports the measured heterogeneous-learning baselines and the capacity-resolved
benchmark, rather than renaming a combination of existing techniques.

## Function-preserving progressive growth

This experiment was inspired by function-preserving network widening in
[Net2Net](https://arxiv.org/abs/1511.05641) and federated progressive growth in
[FedNet2Net](https://arxiv.org/abs/2207.09568). The parent-to-expanded-state
conversion passed a numerical function-preservation check (maximum logit error
about $10^{-6}$), so the initialization itself behaved as intended. Training
quality did not:

| Pilot | Schedule | Capacity-selected macro-F1 | Full-supernet macro-F1 | Median time |
|---|---|---:|---:|---:|
| v3 | 4 parent rounds + 5 expanded rounds | 0.704 | 0.713 | 50.3 s |
| warm-8 | 8 parent rounds + 5 expanded rounds | 0.846 | 0.725 | 35.3 s |

Each row is one seed and is hypothesis-generating only. Raw files are
`.benchmarks/model_scaling/har_progressive_growth_scale15_9r_seed1_pilot_v3.json`
and
`.benchmarks/model_scaling/har_progressive_growth_scale15_13r_seed1_warm8_pilot.json`.
Progressive growth is not included in the paper's main result.

## Server-side replica ensemble

This pilot explored a server ensemble of small client-derived subnetworks,
motivated by ensemble aggregation ideas and the need to use updates from
capacity-limited clients. The 20-member, one-seed, 28-round pilot reached
capacity-selected macro-F1 0.727, below the same-pipeline HeteroFL reference
of 0.886 and a full-model reference of 0.837. It is not a successful result or
a fair multi-seed comparison. Raw output:
`.benchmarks/model_scaling/har_replica_ensemble_20members_28r_seed1_pilot.json`.

Related work already addresses importance-aware client submodels; see
[PriSM](https://arxiv.org/abs/2208.13141). The pilot should not be presented as
an original ensemble method.

## Coverage-focused schedules

The layer-pair routing ablation raises parameter coverage but reduces client
utility and increases runtime; see `EXPLORATORY_ABLATION.md`. Coverage is a
diagnostic, not an objective to maximize without regard to model quality.

## Decision

Keep these experiments and raw outputs for auditability. They do not justify
a new-method claim. The strongest current contribution is an empirical result:
under the tested HAR partition and one-host CPU simulation, a one-thread FjORD
client selects a 13,686-parameter model with test macro-F1 0.889, while the
complete shared FjORD supernet remains under-covered. Additional physical
device and workload experiments are needed to broaden that claim.
