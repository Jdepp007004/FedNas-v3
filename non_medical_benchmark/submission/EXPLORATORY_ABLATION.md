# Exploratory layer-pair coverage ablation

## Motivation and implementation

The main scaling benchmark found low coordinate coverage for fixed-width subnet methods at 8x. To test whether layer-specific window routing would improve shared-model coverage, an exploratory schedule was implemented as `cc_efl_layercover`. It routes different bounded channel windows through adjacent hidden layers. Since layerwise channel mappings are incompatible with residual skip connections in this implementation, the experiment uses a residual-free MLP (`elastic_plain`). The schedule was compared at identical client parameter budgets against plain-MLP CC-EFL, with paired seeds.

## Results

All values below are mean capacity-selected test macro-F1 and coordinate coverage for the 8x, 2,806,278-parameter HAR MLP. Intervals are unadjusted Student-t 95% intervals over five seeds. A negative delta is layer-pair routing minus CC-EFL.

| Rounds | CC-EFL F1 | Layer-pair F1 | Paired difference (95% CI) | Coverage: CC-EFL / layer-pair | Median time: CC-EFL / layer-pair |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.709 | 0.359 | -0.349 (-0.447, -0.252) | 2.6% / 18.2% | 5.8 / 5.8 s |
| 30 | 0.762 | 0.315 | -0.448 (-0.524, -0.372) | 3.5% / 39.5% | 30.2 / 28.9 s |
| 121 | 0.708 | 0.609 | -0.099 (-0.218, +0.020) | 7.2% / 100.0% | 117.4 / 246.3 s |

The 121-round difference is uncertain at five seeds and its interval includes zero. At 121 rounds the schedule does reach full coordinate coverage, but its median runtime is about 2.1 times the CC-EFL runtime and its point estimate is lower. At 8 and 30 rounds, it has substantially higher coverage but much lower deployment utility. More coverage alone did not provide a useful short-run model in this experiment.

## Interpretation

Do not present the layer-pair schedule as a contribution or claim that it improves training. Keep it as a negative exploratory result that motivates reporting coverage together with utility. The main paper remains focused on the 105-run CPU-affinity comparison of the established heterogeneous baselines. These results are from a separate plain-MLP condition and should not be pooled with the residual-MLP scaling matrix.

Raw outputs:

- `.benchmarks/model_scaling/har_layercover_scale8_plain_8r_5seeds_v1.json`
- `.benchmarks/model_scaling/har_layercover_scale8_plain_30r_5seeds_v1.json`
- `.benchmarks/model_scaling/har_layercover_scale8_plain_121r_5seeds_v1.json`
