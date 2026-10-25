# Measured non-medical benchmark results

> **Protocol status (2026-09-28):** corrected four-task v3 matrix and inference are complete. Grouped SMS experiments now include both 20-round MiniLM head adaptation and a five-round BERT-Tiny final-block LoRA adapter benchmark (five seeds, nine strategies). Raw and audited LoRA outputs are preserved separately. The BERT-Tiny experiment trains adapters and a classifier head while freezing/caching the lower encoder; it is not full-model fine-tuning or a distributed privacy test.

## Evidence audit — 2026-09-28

The table below identifies **legacy** five-seed source files previously cited by the draft. They are not results for the current protocol: the old runner queried test metrics during training, tabular preprocessing used holdout categories, capacity tiers changed across rounds, and reporting selected one global model rather than a feasible model for each fixed client capacity. Current four-task evidence is reported in the v3 tables below.

The files named `*_architectures_5seeds.json` are separate comparison sweeps. They are not a reliable replacement for every dedicated run: for example, the Digits sweep records `proximal_mu=0.0` for its `cc_efl` rows, while the intended Digits CC-EFL configuration uses `proximal_mu=0.001`. Preserve those files as historical diagnostics, but do not mix them into the primary CC-EFL results.

| Dataset | Legacy source file for cited CC-EFL row | Selection metric | Audit status |
|---|---|---|---|
| Digits | `.benchmarks/digits_cc_final_5seeds.json` | Validation accuracy | Legacy audit only; v3 results below |
| Adult | `.benchmarks/adult_architectures_5seeds.json` | Validation macro-F1 | Legacy audit only; v3 results below |
| Bank Marketing | `.benchmarks/bank_cc_final_5seeds.json` | Validation macro-F1 | Legacy audit only; v3 results below |
| HAR | `.benchmarks/har_cc_final_5seeds.json` | Validation macro-F1 | Legacy audit only; v3 results below |

The older CIFAR pilot uses 900 training, 100 validation, and 500 test examples, not 1,000 training examples. It remains a pipeline pilot; a clean large-data run is required only if the paper makes a CIFAR performance claim.

## Corrected v3 results — 2026-09-28

The tables below use the bounded unit-stride channel-window schedule, fixed client capability tiers, five paired seeds, validation-only capacity selection, and final held-out evaluation. The v2 files are preserved as audit artifacts; v3 is the current four-task evidence. Resource percentages are parameter-count estimates from the harness, not measured energy or real network traffic. Confidence intervals are unadjusted two-sided paired or seed-level Student-t intervals, as identified.

### Digits — 20 rounds, five seeds

The full-supernet FedAvg accuracy is 96.98% (95.17–98.78); the FedAvg row below is its validation-selected capacity-sliced deployment, not its full-supernet score. The primary endpoint is capacity-selected accuracy.

| Strategy | Selected endpoint (mean, 95% CI) | Paired delta vs capacity-sliced FedAvg (95% CI) | Active client parameter reduction | Selected deployment parameter saving vs full FedAvg |
|---|---:|---:|---:|---:|
| FedAvg | 65.81% (63.52–68.09) | Reference | 0.0% | 76.2% |
| FedProx | 65.99% (63.65–68.32) | +0.18 pp (-0.06, +0.42) | 0.0% | 74.3% |
| HeteroFL | 85.27% (79.72–90.83) | +19.47 pp (+14.80, +24.13) | 70.3% | 80.8% |
| FjORD | 88.86% (86.38–91.34) | +23.05 pp (+20.06, +26.05) | 70.3% | 82.5% |
| FedRolex | 76.83% (72.64–81.02) | +11.02 pp (+6.68, +15.36) | 70.3% | 77.9% |
| MaxNet | 85.66% (80.72–90.60) | +19.85 pp (+15.68, +24.02) | 70.3% | 80.3% |
| CC-EFL | 74.55% (70.73–78.37) | +8.74 pp (+4.37, +13.11) | 70.3% | 77.9% |
| CC-EFL no proximal | 74.49% (70.65–78.32) | +8.68 pp (+4.26, +13.11) | 70.3% | 77.9% |
| CC-EFL uniform weighting | 76.98% (73.03–80.92) | +11.17 pp (+7.06, +15.28) | 70.3% | 76.9% |
| CC-EFL random window | 80.41% (74.04–86.79) | +14.61 pp (+9.19, +20.02) | 70.3% | 77.4% |
| CC-EFL + FjORD distillation | 78.01% (75.27–80.76) | +12.21 pp (+9.86, +14.55) | 70.3% | 82.5% |

### Adult — 10 rounds, five seeds

The full-supernet FedAvg macro-F1 is 0.6856 (0.5933–0.7779); the FedAvg row below is its validation-selected capacity-sliced deployment, not its full-supernet score. The primary endpoint is capacity-selected macro-F1.

| Strategy | Selected endpoint (mean, 95% CI) | Paired delta vs capacity-sliced FedAvg (95% CI) | Active client parameter reduction | Selected deployment parameter saving vs full FedAvg |
|---|---:|---:|---:|---:|
| FedAvg | 0.6935 (0.6423–0.7447) | Reference | 0.0% | 87.9% |
| FedProx | 0.6744 (0.6134–0.7354) | -0.0191 (-0.0361, -0.0021) | 0.0% | 87.9% |
| HeteroFL | 0.6439 (0.5489–0.7390) | -0.0496 (-0.1430, +0.0439) | 72.0% | 83.9% |
| FjORD | 0.6594 (0.5600–0.7589) | -0.0341 (-0.1187, +0.0505) | 72.0% | 86.3% |
| FedRolex | 0.6441 (0.5315–0.7567) | -0.0494 (-0.1706, +0.0717) | 72.0% | 82.1% |
| MaxNet | 0.6401 (0.5428–0.7373) | -0.0535 (-0.1465, +0.0396) | 72.0% | 83.4% |
| CC-EFL | 0.6459 (0.5328–0.7590) | -0.0476 (-0.1689, +0.0737) | 72.0% | 81.2% |
| CC-EFL no proximal | 0.6428 (0.5292–0.7563) | -0.0508 (-0.1704, +0.0688) | 72.0% | 82.2% |
| CC-EFL uniform weighting | 0.6469 (0.5351–0.7588) | -0.0466 (-0.1686, +0.0754) | 72.0% | 82.1% |
| CC-EFL random window | 0.7270 (0.6690–0.7849) | +0.0334 (-0.0353, +0.1021) | 72.0% | 81.9% |
| CC-EFL + FjORD distillation | 0.6455 (0.5198–0.7712) | -0.0480 (-0.1700, +0.0741) | 72.0% | 83.9% |

### Bank Marketing — 10 rounds, five seeds

The full-supernet FedAvg macro-F1 is 0.6716 (0.6029–0.7402); the FedAvg row below is its validation-selected capacity-sliced deployment, not its full-supernet score. The primary endpoint is capacity-selected macro-F1.

| Strategy | Selected endpoint (mean, 95% CI) | Paired delta vs capacity-sliced FedAvg (95% CI) | Active client parameter reduction | Selected deployment parameter saving vs full FedAvg |
|---|---:|---:|---:|---:|
| FedAvg | 0.6236 (0.5747–0.6725) | Reference | 0.0% | 84.4% |
| FedProx | 0.6227 (0.5787–0.6666) | -0.0009 (-0.0144, +0.0125) | 0.0% | 84.9% |
| HeteroFL | 0.6792 (0.6367–0.7216) | +0.0556 (-0.0044, +0.1156) | 74.7% | 88.5% |
| FjORD | 0.6769 (0.6440–0.7098) | +0.0533 (-0.0039, +0.1105) | 74.7% | 94.5% |
| FedRolex | 0.6727 (0.6310–0.7143) | +0.0490 (-0.0056, +0.1037) | 74.7% | 86.5% |
| MaxNet | 0.6704 (0.6100–0.7309) | +0.0468 (-0.0262, +0.1199) | 74.7% | 87.8% |
| CC-EFL | 0.6756 (0.6336–0.7176) | +0.0520 (+0.0009, +0.1031) | 74.7% | 85.8% |
| CC-EFL no proximal | 0.6681 (0.6077–0.7286) | +0.0445 (-0.0235, +0.1126) | 74.7% | 87.1% |
| CC-EFL uniform weighting | 0.6764 (0.6442–0.7086) | +0.0528 (+0.0045, +0.1011) | 74.7% | 86.5% |
| CC-EFL random window | 0.6073 (0.4853–0.7294) | -0.0163 (-0.1542, +0.1216) | 74.7% | 83.0% |
| CC-EFL + FjORD distillation | 0.6841 (0.6441–0.7242) | +0.0605 (-0.0028, +0.1239) | 74.7% | 90.6% |

### Human Activity Recognition (HAR) — 8 rounds, five seeds

The full-supernet FedAvg macro-F1 is 0.9107 (0.8819–0.9395); the FedAvg row below is its validation-selected capacity-sliced deployment, not its full-supernet score. The primary endpoint is capacity-selected macro-F1.

| Strategy | Selected endpoint (mean, 95% CI) | Paired delta vs capacity-sliced FedAvg (95% CI) | Active client parameter reduction | Selected deployment parameter saving vs full FedAvg |
|---|---:|---:|---:|---:|
| FedAvg | 0.6903 (0.6171–0.7634) | Reference | 0.0% | 63.4% |
| FedProx | 0.6782 (0.6039–0.7526) | -0.0120 (-0.0356, +0.0116) | 0.0% | 63.4% |
| HeteroFL | 0.8835 (0.8500–0.9169) | +0.1932 (+0.1398, +0.2466) | 61.8% | 71.4% |
| FjORD | 0.8965 (0.8707–0.9224) | +0.2063 (+0.1509, +0.2617) | 61.8% | 80.6% |
| FedRolex | 0.8630 (0.8211–0.9048) | +0.1727 (+0.1093, +0.2361) | 61.8% | 68.9% |
| MaxNet | 0.8791 (0.8349–0.9234) | +0.1889 (+0.1477, +0.2300) | 61.8% | 69.6% |
| CC-EFL | 0.8618 (0.8231–0.9006) | +0.1716 (+0.1061, +0.2370) | 61.8% | 67.4% |
| CC-EFL no proximal | 0.8610 (0.8213–0.9007) | +0.1707 (+0.1063, +0.2352) | 61.8% | 67.4% |
| CC-EFL uniform weighting | 0.8618 (0.8192–0.9044) | +0.1715 (+0.1113, +0.2318) | 61.8% | 69.0% |
| CC-EFL random window | 0.8103 (0.7362–0.8845) | +0.1201 (+0.0234, +0.2168) | 61.8% | 69.0% |
| CC-EFL + FjORD distillation | 0.8639 (0.8401–0.8878) | +0.1737 (+0.1121, +0.2352) | 61.8% | 69.9% |

### Cross-task primary inference

For the four CC-EFL versus FedAvg task comparisons, paired deltas use capacity-selected scores from each method; the FedAvg endpoint is a capacity-sliced deployment. Holm adjustment is across the four task-level tests; the displayed 95% confidence intervals are unadjusted.

| Task | Endpoint | CC-EFL paired delta | Unadjusted paired 95% CI | Unadjusted p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| digits | capacity-selected accuracy | +8.74 pp | (+4.37, +13.11) pp | 0.0051 | 0.0154 |
| adult | capacity-selected macro-F1 | -0.0476 | (-0.1689, +0.0737) | 0.3372 | 0.3372 |
| bank_marketing | capacity-selected macro-F1 | +0.0520 | (+0.0009, +0.1031) | 0.0475 | 0.0949 |
| har | capacity-selected macro-F1 | +0.1716 | (+0.1061, +0.2371) | 0.0019 | 0.0076 |

After correction, Digits and HAR have positive differences from capacity-sliced FedAvg; Adult is uncertain and Bank does not meet the adjusted 0.05 threshold. FjORD has the highest mean endpoint on Digits and HAR; HeteroFL has the highest mean macro-F1 on Bank. CC-EFL therefore does not lead the comparator set across tasks. Exact per-seed comparisons, all-strategy Holm families, and CC-EFL ablation families are in `.benchmarks/phase2_v3_inference.json`; regenerate them with `python -m non_medical_benchmark.analysis.summarize_v3_inference`.

## Fixed-client-budget supernet scaling — 2026-09-28

Artifacts: `.benchmarks/model_scaling/har_fixedclient_widthscale_1_4_8r_5seeds.json` and `.benchmarks/model_scaling/har_fixedclient_widthscale_8_8r_3seeds.json`. This exploratory HAR experiment grows the server supernet's maximum width from 96 to 384 and 768 while keeping the heterogeneous client tier menu fixed at widths 24/48/72 (same depth profiles, eight clients, eight rounds). The full-width FedAvg baseline trains the expanded width on every client. The held-out utility endpoint is the validation-selected, capacity-bounded client deployment macro-F1. All methods share the same splits, client partitions, and capacity map per seed.

| Server max width | Full supernet parameters | FP32 supernet weights | Heterogeneous mean active client parameters | Active parameter / estimated FLOP reduction vs full FedAvg | CC-EFL capacity-selected macro-F1 (95% CI) | CC-EFL / FedAvg full-supernet macro-F1 | CC-EFL paired delta vs capacity-selected FedAvg (95% CI) | CC-EFL coordinate coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 (1×) | 92,742 | 0.354 MiB | 31,974 | 65.5% / 65.4% | 0.8321 (0.7734–0.8908) | 0.8612 / 0.9059 | +0.1440 (+0.0274, +0.2607) | 81.4% |
| 384 (4×) | 813,318 | 3.103 MiB | 31,974 | 96.1% / 96.1% | 0.8036 (0.7502–0.8570) | 0.8216 / 0.8956 | +0.1294 (+0.0939, +0.1649) | 9.3% |
| 768 (8×) | 2,806,278 | 10.705 MiB | 31,974 | 98.9% / 98.9% | 0.7459 (0.6825–0.8093) | 0.7300 / 0.8824 | +0.1197 (−0.0019, +0.2413) | 2.7% |

The 1× and 4× rows use five seeds; 8× uses three, so its intervals are especially tentative. The paired intervals are descriptive and were not adjusted across the five method comparisons; this was a scaling probe, not a registered non-inferiority test. At 4×, CC-EFL's mean deployment macro-F1 is below HeteroFL (0.8418), FjORD (0.8572), and MaxNet (0.8466). At 8× it is below HeteroFL (0.7825), FjORD (0.8282), and MaxNet (0.7913). All subnet methods use the same active client parameter/FLOP budget under these fixed tiers, so the savings relative to full FedAvg are not unique to CC-EFL.

This verifies that the simulator can execute an eight-round run over a 2.81M-parameter server supernet while each heterogeneous client still materializes a fixed small subnet averaging 31,974 parameters (0.122 MiB of FP32 weights). It does **not** establish a useful fully trained 2.81M-parameter model: CC-EFL's full-supernet macro-F1 is 0.7300 versus 0.8824 for full-width FedAvg, and touched-coordinate coverage is only 2.7%. The capacity-selected smaller client deployments retain higher utility than the full-FedAvg sliced reference in these runs, but that result does not mean the entire expanded server model is well-trained. The parameter-derived storage and FLOP values are estimates; the CPU simulation did not measure peak memory on an isolated edge device, energy, or real network behavior.

### Matched random-window control at fixed client budgets

Artifacts: `.benchmarks/model_scaling/har_fixedclient_widthscale_4_randomcontrol_8r_5seeds.json` and `.benchmarks/model_scaling/har_fixedclient_widthscale_8_randomcontrol_8r_3seeds.json`. CC-EFL and its random-window control use identical client sizes, active parameters, FLOPs, and payload. Random offsets touch more of the expanded supernet but have lower capacity-selected macro-F1 in this short schedule:

| Server width | Method | Macro-F1 (95% CI) | Coordinate coverage | Paired delta vs CC-EFL (95% CI) |
|---:|---|---:|---:|---:|
| 384 | CC-EFL | 0.8036 (0.7502–0.8570) | 9.3% | Reference |
| 384 | Random-window CC-EFL | 0.5845 (0.4026–0.7663) | 45.3% | −0.2191 (−0.4133, −0.0249) |
| 768 | CC-EFL | 0.7459 (0.6825–0.8093) | 2.7% | Reference |
| 768 | Random-window CC-EFL | 0.4636 (0.1281–0.7991) | 23.6% | −0.2823 (−0.6681, +0.1034) |

These paired intervals are exploratory and unadjusted. They suggest that maximizing coordinate coverage with independent random windows can hurt client-tier utility under eight rounds; the 8× result is highly uncertain. This does not resolve the global-coverage limitation because neither schedule adequately trains the full expanded model at the tested round budget.

The proportional-width control, `.benchmarks/model_scaling/har_widthscale_1_4_8r_5seeds.json`, grows client widths along with the supernet and must not be used as evidence for a fixed client budget. In that run, mean active client parameters rose from 31,974 at 1× to 200,454 at 4×.

Reproduce the fixed-budget experiment with:

    python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 1 4 --capacity-scaling fixed --rounds 8 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex maxnet cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fixedclient_widthscale_1_4_8r_5seeds.json
    python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 8 --capacity-scaling fixed --rounds 8 --seeds 1 2 3 --strategies fedavg heterofl fjord fedrolex maxnet cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fixedclient_widthscale_8_8r_3seeds.json

## Pretrained-text head-adaptation results — 2026-09-28

Artifact: .benchmarks/pretrained_sms/sms_minilm_head_20r_10seeds_v3.json. It uses UCI SMS Spam (5,574 messages; archive SHA-256 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3) and pinned sentence-transformers/all-MiniLM-L6-v2 revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. The UCI source page lists the dataset and CC BY 4.0 terms: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). The [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) identifies a 384-dimensional sentence encoder and Apache-2.0 model license.

The encoder has 22,713,216 parameters and is frozen. The benchmark trains/communicates only an elastic MLP classifier head (maximum 83,330 parameters, depth 3, hidden dimension 128, widths 32/64/96/128). Embeddings were computed or loaded in one local preprocessing stage before the federated simulation. This is federated head adaptation over frozen representations, not encoder fine-tuning, LoRA, private on-device encoding, or an end-to-end federated language model. Head-only payload reductions must not be described as whole-model communication or storage savings.

The grouped stratified split assigns 14/20 folds to train, 3/20 to validation, and 3/20 to test. Exact duplicate texts remain in one partition; measured train-validation, train-test, and validation-test overlaps are all zero. Counts are 3,908 train (3,397 ham / 511 spam), 837 validation (721 / 116), and 829 test (709 / 120). The client partitions are six seeded Dirichlet label-skew shards (alpha 0.3). Training uses 20 rounds, seeds 1–10, Adam at 0.001, batch size 128, one local epoch per round, balanced class weights, and FedProx mu 0.001 where applicable. The primary endpoint is validation-selected capacity test macro-F1. Confidence intervals are unadjusted two-sided Student-t intervals over seeds; paired p-values use Holm correction within the nine-strategy-vs-FedAvg family.

Centralized full-head training reaches test macro-F1 0.9691 (95% CI 0.9657–0.9726). Federated results:

| Strategy | Capacity-selected test macro-F1 (95% CI) | Paired delta vs FedAvg (95% CI) | Holm-adjusted p |
|---|---:|---:|---:|
| FedAvg | 0.8591 (0.7531–0.9652) | Reference | — |
| FedProx | 0.8461 (0.7471–0.9451) | -0.0131 (-0.0248, -0.0013) | 0.2944 |
| HeteroFL | 0.8674 (0.7668–0.9680) | +0.0083 (-0.0887, +0.1053) | 1.0000 |
| FjORD | 0.8102 (0.6652–0.9552) | -0.0489 (-0.1172, +0.0193) | 1.0000 |
| FedRolex | 0.8633 (0.7560–0.9707) | +0.0042 (-0.0837, +0.0921) | 1.0000 |
| MaxNet | 0.8667 (0.7674–0.9660) | +0.0076 (-0.0912, +0.1063) | 1.0000 |
| CC-EFL | 0.8565 (0.7480–0.9650) | -0.0026 (-0.1020, +0.0968) | 1.0000 |
| CC-EFL no proximal | 0.8667 (0.7605–0.9728) | +0.0075 (-0.0969, +0.1119) | 1.0000 |
| CC-EFL uniform weighting | 0.8663 (0.7663–0.9662) | +0.0071 (-0.0821, +0.0963) | 1.0000 |
| CC-EFL random window | 0.8915 (0.8545–0.9284) | +0.0323 (-0.0636, +0.1283) | 1.0000 |

No comparison survives Holm correction. The random-window control has the highest mean, but its paired interval includes zero. CC-EFL is statistically indistinguishable from FedAvg, with a wide interval. On the communicated head, CC-EFL has an estimated 62.5% per-round payload reduction versus full-head FedAvg. Including the frozen encoder, CC-EFL's total model parameter count is only about 0.27% below the FedAvg total. This experiment therefore gives no evidence of substantial whole-model size savings.

The earlier random-row-split artifact sms_minilm_head_10r_5seeds.json is invalid for final reporting because exact duplicate texts crossed split boundaries. The 10-round grouped five-seed run is superseded by the 20-round/10-seed result above. Both remain available for audit.

Reproduce from the repository root with:

    python -m non_medical_benchmark.run_pretrained_head_benchmark --rounds 20 --clients 6 --alpha 0.3 --seeds 1 2 3 4 5 6 7 8 9 10 --torch-threads 1 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window --output .benchmarks/pretrained_sms/sms_minilm_head_20r_10seeds_v3_reproduce.json

## BERT-Tiny final-block LoRA adapter benchmark — 2026-09-28

Raw run: .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1.json. Audited report: .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reported.json. The postprocessor verifies the source archive, grouped split, and each seeded client partition against the raw run, then derives communication counts from the saved rank windows. It corrects a raw-run reporting defect: FedAvg and FedProx trained rank-4 adapters, not deployment-rank adapters. Scores are unchanged by this correction.

The encoder is google/bert_uncased_L-2_H-128_A-2, pinned to revision 30b0a37ccaaa32f332884b96992754e246e48c5f (4,385,920 total parameters). The two-layer BERT-Tiny model has LoRA updates on the query and value projections in its final transformer block; rank-1/2/3 windows are assigned to heterogeneous clients from a rank-4 adapter supernet. The 2-class head is also trainable. The trainable count is 2,306 parameters at rank 4; this includes the rank-4 adapter matrices and classifier. The base encoder and remaining transformer weights stay frozen. Inputs to the trainable final block are computed by a frozen prefix after client partitioning, cached separately per client partition, and reused across methods. Thus this is adapter fine-tuning in a single-process FL simulation, not full encoder training or a test of private on-device inference.

The same duplicate-grouped UCI SMS split as the head-only experiment is used (3,908 train, 837 validation, 829 test; zero exact text overlap for all partition pairs). The run uses six clients, a seeded Dirichlet alpha of 0.3, five communication rounds, seeds 1–5, one local epoch per client per round, batch size 128, maximum sequence length 64, Adam at 0.0002, and FedProx mu 0.001 where enabled. Each client receives a fixed rank capacity in {1,2,3}; FedAvg and FedProx update rank 4. Each method-seed pair is evaluated once on the same held-out test split after per-capacity validation selection. In total, this is 45 federated runs plus five centralized controls. The originally run Dirichlet protocol samples class allocations independently, which changes both label proportions and total client sizes.

Centralized rank-4 training reaches test macro-F1 0.8547 (95% CI 0.8242–0.8853). Federated capacity-selected results are:

| Strategy | Capacity-selected test macro-F1 (95% CI) | Paired delta vs FedAvg (95% CI) | Holm-adjusted p | Adapter/head upload bytes per round | Reduction vs FedAvg |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.6307 (0.4166–0.8448) | Reference | — | 55,344 | 0.0% |
| FedProx | 0.6307 (0.4166–0.8448) | 0.0000 (0.0000, 0.0000) | 1.0000 | 55,344 | 0.0% |
| HeteroFL | 0.6307 (0.4161–0.8453) | −0.000003 (−0.00147, +0.00146) | 1.0000 | 30,768 | 44.4% |
| FedRolex | 0.6312 (0.4174–0.8450) | +0.00048 (−0.00041, +0.00138) | 1.0000 | 30,768 | 44.4% |
| MaxNet | 0.6223 (0.3979–0.8467) | −0.00838 (−0.02668, +0.00992) | 1.0000 | 30,768 | 44.4% |
| CC-EFL | 0.6227 (0.3958–0.8495) | −0.00802 (−0.02776, +0.01173) | 1.0000 | 30,768 | 44.4% |
| CC-EFL without proximal term | 0.6227 (0.3958–0.8495) | −0.00802 (−0.02776, +0.01173) | 1.0000 | 30,768 | 44.4% |
| CC-EFL uniform weighting | 0.6312 (0.4174–0.8450) | +0.00048 (−0.00041, +0.00138) | 1.0000 | 30,768 | 44.4% |
| CC-EFL random window | 0.6235 (0.3961–0.8509) | −0.00719 (−0.02689, +0.01250) | 1.0000 | 30,768 | 44.4% |

FedAvg in this table trains all clients at rank 4 and reports test utility from validation-selected capacity deployments. No paired strategy difference survives Holm correction; every adjusted p-value is 1.0. CC-EFL does not outperform FedAvg in this experiment, and this run does not establish its superiority over heterogeneous baselines. The confidence intervals are wide because outcomes vary greatly across only five partition seeds. Several methods have exactly matching test scores on some seeds; those are retained as observed, not interpreted as equivalence.

The Dirichlet partitioner produces both label skew and strong client-size imbalance. Across seeds, the mean coefficient of variation of the six client sizes is 0.927. Per-client ham/spam counts and partition hashes are included in the reported JSON. This limits attribution: the initial experiment does not isolate label skew from quantity skew.

The 44.4% payload reduction counts only float32 trainable adapter and classifier coordinates uploaded/downloaded each round (7,692 versus 13,836 coordinates per round). It excludes transport metadata and the frozen 4.39M-parameter base, which remains stored and executed on the client. It is not a total-model or network measurement. The method's rank-coordinate coverage averages 1.0 for FedRolex/CC-EFL variants, versus 0.75 for HeteroFL and MaxNet in this five-round schedule.

Model source: [official BERT-Tiny model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) and [pinned model revision](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f). Dataset source and terms are documented above.

### Post-hoc quantity-balanced partition sensitivity — 2026-09-28

This follow-up was added after observing the very high client-size CV above; treat it as exploratory, not as a pre-registered confirmatory result. Raw output: .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1.json. Audited output: .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1_reported.json. Paired protocol analysis: .benchmarks/pretrained_sms/sms_bert_tiny_lora_partition_comparison_v1.json. It uses the same five seeds, split hash, model, rounds, and strategy set. Each client receives 651 or 652 messages (mean size CV 0.00072). Spam counts are drawn using a Dirichlet allocation projected to client capacities with at least one example of each class per client; ham fills the remaining client slots. This is a quantity-balanced alternative that remains label-skewed, but it also changes the class-allocation procedure, so it is not a pure intervention on client-size CV.

| Strategy | Capacity-selected test macro-F1 (95% CI) | Paired delta vs FedAvg (95% CI) | Holm-adjusted p |
|---|---:|---:|---:|
| FedAvg | 0.5914 (0.3109–0.8719) | Reference | — |
| FedProx | 0.5914 (0.3109–0.8719) | 0.0000 (0.0000, 0.0000) | 1.0000 |
| HeteroFL | 0.5914 (0.3104–0.8724) | −0.000002 (−0.00225, +0.00225) | 1.0000 |
| FedRolex | 0.5912 (0.3102–0.8722) | −0.00022 (−0.00082, +0.00039) | 1.0000 |
| MaxNet | 0.5874 (0.3007–0.8741) | −0.00397 (−0.01268, +0.00475) | 1.0000 |
| CC-EFL | 0.5877 (0.3008–0.8746) | −0.00372 (−0.01237, +0.00492) | 1.0000 |
| CC-EFL without proximal term | 0.5877 (0.3008–0.8746) | −0.00372 (−0.01237, +0.00492) | 1.0000 |
| CC-EFL uniform weighting | 0.5912 (0.3102–0.8722) | −0.00022 (−0.00082, +0.00039) | 1.0000 |
| CC-EFL random window | 0.5877 (0.3008–0.8746) | −0.00372 (−0.01237, +0.00492) | 1.0000 |

Every strategy-versus-FedAvg comparison is Holm-adjusted p=1.0. In the paired cross-protocol comparison, the CC-EFL change (balanced minus original protocol) is −0.0350 macro-F1 (95% CI −0.3676 to +0.2975; Holm-adjusted p=1.0); all nine strategy-by-protocol comparisons have adjusted p=1.0. Client-size CV falls from 0.9272 to 0.00072, but the five-seed score spread does not narrow: FedAvg's across-seed standard deviation rises from 0.1725 to 0.2260 and CC-EFL's from 0.1827 to 0.2311. This small sensitivity check does not show that client-size imbalance alone explains the instability; severe label skew and the low seed count remain plausible contributors. The balanced condition also yields no evidence of a CC-EFL gain.

Reproduce the balanced run and its audit with:

    python -m non_medical_benchmark.run_lora_sms_benchmark --partition-mode quantity_balanced_label_dirichlet --rounds 5 --clients 6 --alpha 0.3 --seeds 1 2 3 4 5 --strategies fedavg fedprox heterofl fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window --batch-size 128 --max-length 64 --torch-threads 1 --output .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1_reproduce.json
    python -m non_medical_benchmark.analysis.report_lora_sms_results --input .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1_reproduce.json --output .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1_reproduce_reported.json
    python -m non_medical_benchmark.analysis.compare_lora_partition_modes --joint .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reported.json --balanced .benchmarks/pretrained_sms/sms_bert_tiny_lora_balanced_5r_5seeds_v1_reported.json --output .benchmarks/pretrained_sms/sms_bert_tiny_lora_partition_comparison_reproduce.json

Reproduce from the repository root with:

    python -m non_medical_benchmark.run_lora_sms_benchmark --rounds 5 --clients 6 --alpha 0.3 --seeds 1 2 3 4 5 --strategies fedavg fedprox heterofl fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window --batch-size 128 --max-length 64 --torch-threads 1 --output .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reproduce.json
    python -m non_medical_benchmark.analysis.report_lora_sms_results --input .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reproduce.json --output .benchmarks/pretrained_sms/sms_bert_tiny_lora_5r_5seeds_v1_reproduce_reported.json

## Historical scores — audit only

The following tables are historical legacy evidence. They are not results
for the corrected protocol, and must not be used as the paper's current
four-task headline. Corrected Digits, Adult, Bank Marketing, and HAR results
are reported above.

## Legacy integrated method versus FedAvg

`cc_efl` means FedRoleX rolling channel coverage + MaxNet-style largest-tier
weighting + dataset-tuned FedProx regularization. It is a systems integration
claim, not a claim of a new standalone optimizer.

| Dataset / rounds | FedAvg accuracy (95% CI) | CC-EFL accuracy (95% CI) | CC-EFL macro-F1 | Paired accuracy delta | Client resource reduction |
|---|---:|---:|---:|---:|---:|
| Digits / 20 / 5 seeds | 96.53% (94.91–98.16) | 96.53% (95.34–97.73) | 96.50% | 0.00 pp (−1.23, +1.23) | 50.6% parameters, 50.8% FLOPs/bytes |
| Adult / 10 / 5 seeds | 82.83% (80.36–85.30) | 82.93% (80.26–85.61) | 72.18% | +0.11 pp (−1.07, +1.28) | 56.2% parameters, 56.4% FLOPs/bytes |
| Bank Marketing / 10 / 5 seeds | 87.94% (85.27–90.61) | 88.77% (87.63–89.92) | 71.95% | +0.83 pp (−2.16, +3.82) | 58.6% parameters, 58.9% FLOPs/bytes |
| HAR / 8 / 5 seeds | 91.55% (90.60–92.49) | 92.17% (90.16–94.18) | 92.07% | +0.62 pp (−0.82, +2.07) | 45.1% parameters, 45.1% FLOPs/bytes |

The result supports a resource/utility trade-off across these four tabular
tasks: no statistically significant accuracy loss is visible in these paired
intervals, and the integrated method saves 45–59% of client-side resources.
This is not evidence of universal improvement; the intervals overlap zero.

## Legacy reference architecture comparison

The full five-seed Digits architecture sweep is in
`.benchmarks/digits_architectures_5seeds.json`. The method names correspond to
the mechanisms documented in `ARCHITECTURES.md`.

| Strategy | Accuracy | 95% CI | Resource reduction vs FedAvg |
|---|---:|---:|---:|
| FedAvg | 97.07% | 95.21–98.92 | 0% |
| HeteroFL | 90.58% | 88.46–92.69 | 63.9% |
| FjORD | 92.53% | 88.82–96.24 | 63.9% |
| FedRolex | 96.36% | 94.59–98.12 | 50.6% |
| MaxNet | 96.00% | 93.76–98.24 | 50.6% |

The direct comparison says why the integrated design is needed: static
prefixes are cheap but lose accuracy, while FedRoleX recovers most of the
FedAvg accuracy by training neglected channel regions. FjORD is the strongest
small-tier result on Adult (84.06% accuracy and 77.34% macro-F1 in the
five-seed sweep, with about 65% parameter/FLOP reduction), so it remains a
reference option for especially constrained devices.

## Security and privacy measurements

These are five-round Digits runs with 600 training samples and six clients.
Accuracy intervals use the same seed-level reporting. “Backdoor success” is
attack success rate (ASR): the fraction of triggered test examples classified
as the attacker's target label.

| Condition | Accuracy | Macro-F1 / ASR | Interpretation |
|---|---:|---:|---|
| Clean FedAvg | 84.15% | 83.43% F1 | Reference |
| 33% sign-flip clients | 19.56% | 11.47% F1 | Severe Byzantine failure |
| Sign-flip + trimmed mean | 28.00% | 17.30% F1 | Partial recovery only |
| Sign-flip + FLAME-style filter | 45.33% | 40.50% F1 | Unstable partial recovery |
| 33% backdoor clients | 66.67% | 98.67% ASR | Clean accuracy can hide a successful trigger |
| Backdoor + FLAME-style filter | 48.30% | 98.96% ASR | Current filter does not mitigate this attack |
| Secure aggregation, no DP | unchanged | unchanged | Pairwise masks cancel exactly; +192 key bytes/round |
| Secure aggregation + DP, noise 0.05, clip 5 | 24.15% | ε≈484.48, δ=10⁻⁵ | Large utility loss at this calibration |

The ordinary secure-aggregation run is an in-memory functional simulator: the
coordinator still holds plaintext updates before applying the masks. The
process-isolated harness keeps each private key in a client worker and verified
mask cancellation with max absolute sum error below 1e−14, but it is still not
a production network protocol and has no dropout recovery or threshold key
reconstruction.

## Corrected CIFAR pilot

The corrected CIFAR-10 pilot uses five rounds, 900 training images, 100
validation images, 500 test images, and two seeds. It is useful for checking the CNN and tier plumbing, not
for the final image accuracy claim.

| Strategy | Accuracy | 95% CI | Resource reduction |
|---|---:|---:|---:|
| FedAvg | 21.60% | 6.35–36.85 | 0% |
| FjORD | 19.10% | 15.29–22.91 | 70.2% parameters |
| FedRolex | 17.00% | 0–42.41 | 70.2% parameters |
| CC-EFL | 16.30% | 7.41–25.19 | 70.2% parameters |

These low values are expected from the tiny cap, short training budget, small
CNN, and CPU-only run. The paper must use the larger CIFAR command in the
README before making a vision accuracy claim.

## Reproduction commands

Run these from the repository root. The output names are new and do not overwrite legacy artifacts. Each core command reproduces the five-seed Adam comparison, including the component controls and matched random-window/FjORD extension; UCI datasets require `--download` on the first run.

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 20 --clients 6 --seeds 1 2 3 4 5 --torch-threads 1 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window cc_efl_fjord --output .benchmarks/phase2_digits_adam_20r_5seeds_v3_reproduce.json
python -m non_medical_benchmark.run_benchmark --dataset adult --download --rounds 10 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --torch-threads 1 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window cc_efl_fjord --output .benchmarks/phase2_adult_adam_10r_5seeds_v3_reproduce.json
python -m non_medical_benchmark.run_benchmark --dataset bank_marketing --download --rounds 10 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --torch-threads 1 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window cc_efl_fjord --output .benchmarks/phase2_bank_adam_10r_5seeds_v3_reproduce.json
python -m non_medical_benchmark.run_benchmark --dataset har --download --rounds 8 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --torch-threads 1 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl cc_efl_no_prox cc_efl_uniform cc_efl_random_window cc_efl_fjord --output .benchmarks/phase2_har_adam_8r_5seeds_v3_reproduce.json
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 20 --clients 6 --optimizer sgd --seeds 1 2 3 4 5 --torch-threads 1 --strategies fedavg fedprox scaffold --output .benchmarks/phase2_digits_scaffold_sgd_matched_reproduce.json
```

The five-seed, 20-round CIFAR-10 run is optional and was not completed as part of the corrected four-task core matrix. Run it only if the paper makes an image-model claim; the existing tiny pilot is not evidence for that claim. The security protocol harness is separate and does not establish deployed privacy or Byzantine robustness.

## Four-device CPU-split scaling benchmark — 2026-10-25

Ran 105 paired runs on HAR across 1×/4×/8× server widths, seven methods, five seeds, and eight rounds. Four affinity-limited virtual clients were assigned 1, 2, 4, and 8 logical processors, leaving three logical processors for the coordinator. The host is an Intel Core Ultra 5 125H with 14 physical cores, 18 logical processors, 16 GB RAM, and CPU-only PyTorch 2.9.0. Worker-pool startup (23.1 seconds) is reported separately; per-run times below are five-seed medians.

At 8× (2.81M server parameters), full FedAvg takes 36.4 seconds per eight-round run. CC-EFL takes 17.9 seconds, HeteroFL 10.2 seconds, MaxNet 13.1 seconds, FedRolex 13.4 seconds, and FjORD 16.9 seconds. CC-EFL trains an average 29,958 parameters per client (98.9% fewer than full FedAvg). Its capacity-selected deployment macro-F1 is 0.828 (95% CI 0.785–0.871) with 0.849 accuracy, versus 0.658 macro-F1 / 0.695 accuracy for capacity-selected FedAvg. On the full shared model, CC-EFL reaches 0.829 macro-F1 / 0.848 accuracy versus 0.905 / 0.907 for full FedAvg; only 2.6% of expanded coordinates are touched in eight rounds. The result supports participation by smaller client subnets, but the full 8× supernet remains under-covered.

These are affinity-limited worker processes on one Windows host, not four physical devices. Affinity is not an exclusive CPU reservation. The benchmark does not measure physical-device memory limits, energy, or network latency; parameter memory and FLOPs are estimates. The main matrix reaches 2.81M parameters; the separate follow-up below reaches 10.33M. Neither is an LLM-scale result. Confidence intervals are exploratory and unadjusted across the method/scale matrix.

The [full benchmark report](FOUR_DEVICE_BENCHMARK.md) lists all 21 main-matrix method/scale results and the four device specifications. Raw run data: `.benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json`. Reproduce from the repository root with:

```text
python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 1 4 8 --capacity-scaling fixed --rounds 8 --clients 4 --client-cpu-threads 1 2 4 8 --reserve-cpu-threads 3 --seeds 1 2 3 4 5 --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json
```

### 16× server-scale follow-up

A separate 20-run extension tests FedAvg, HeteroFL, FjORD, and CC-EFL at width 1,536 (10,331,142 parameters; 39.41 MiB of FP32 weights), retaining the same client CPU allocations, four fixed subnet tiers, and five paired seeds. Median eight-round times are 56.9, 20.1, 20.3, and 30.0 seconds, respectively. Capacity-selected macro-F1 / accuracy are 0.737 / 0.753 for FedAvg, 0.857 / 0.869 for HeteroFL, 0.882 / 0.890 for FjORD, and 0.801 / 0.828 for CC-EFL. HeteroFL, FjORD, and CC-EFL average 29,958 active parameters per client-round (99.71% fewer than full-model FedAvg). CC-EFL full-supernet macro-F1 is 0.825 versus 0.904 for full-model FedAvg, with 0.70% coordinate coverage after eight rounds. This extends the single-host simulation to 10.33M parameters, but does not demonstrate physical-device operation or full-supernet convergence.

Raw output: `.benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_16_8r_5seeds.json`. Reproduce with:

```text
python -m non_medical_benchmark.run_model_scaling_benchmark --dataset har --scales 16 --capacity-scaling fixed --rounds 8 --clients 4 --client-cpu-threads 1 2 4 8 --reserve-cpu-threads 3 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord cc_efl --torch-threads 1 --output .benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_16_8r_5seeds.json
```

### Fresh 8× confirmation and second workload — 2026-10-25

A new paired block on HAR uses seeds 6–15 and repeats FedAvg, HeteroFL, and
FjORD for eight rounds each (30 runs). It reuses the four affinity-limited
profiles and 2,806,278-parameter server MLP. This is a ten-seed confirmation of
three strategies, not a rerun of all seven methods in the main matrix.

| Strategy | Median seconds / 8 rounds | Capacity-selected test macro-F1 (95% CI) / accuracy | Full-supernet macro-F1 / accuracy | Active parameters / client-round |
|---|---:|---:|---:|---:|
| FedAvg | 30.52 | 0.660 (0.602–0.717) / 0.689 | 0.908 / 0.911 | 2,806,278 |
| HeteroFL | 7.94 | 0.848 (0.807–0.888) / 0.866 | 0.813 / 0.834 | 29,958 |
| FjORD | 9.42 | 0.905 (0.885–0.926) / 0.909 | 0.811 / 0.832 | 29,958 |

The paired FjORD deployment-minus-capacity-sliced-FedAvg macro-F1 difference
is +0.245 (95% CI +0.188 to +0.303). The median per-seed FedAvg/FjORD
simulation-time ratio is 2.97. The descriptive cross-endpoint difference
between FjORD's capacity-selected deployment and full FedAvg is −0.003
(95% CI −0.021 to +0.015); it is not a formal non-inferiority test. Full FjORD
macro-F1 is 0.811 (0.726–0.895), and its paired difference from full FedAvg is
−0.098 (−0.188 to −0.008), with only 2.23% coordinate coverage. Therefore the
strong client endpoint does not establish convergence of the full shared
model.

The smallest FjORD device (one CPU thread) trains 13,686 active parameters per
round and selects a 13,686-parameter model with test macro-F1 0.889
(0.855–0.923), accuracy 0.896, and mean local training time 0.048 seconds per
round. Its full-model FedAvg counterpart trains 2,806,278 parameters, with
mean local time 1.874 seconds per round. The 2-thread profile shares the same
capacity and selected score; the 4- and 8-thread profiles reach macro-F1
0.919 (0.904–0.933) and 0.924 (0.915–0.934). Full profile, accuracy, and
per-device timing details are in `FOUR_DEVICE_BENCHMARK.md`.

A five-seed Digits extension (seeds 6–10, 15 runs) gives capacity-selected
macro-F1 0.810 (0.743–0.877) for FjORD using 4,282 active parameters, compared
with 0.290 (0.245–0.334) for capacity-sliced FedAvg. Median eight-round times
are 5.64 and 11.98 seconds. Full-model FedAvg reaches 0.964, while the FjORD
full supernet reaches 0.623. This small-workload replication supports the
client-capacity endpoint but also reinforces the global-model coverage limit.

Raw HAR and Digits outputs are `.benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json` and
`.benchmarks/model_scaling/digits_fourdevice_8x_confirmation_5seeds.json`.
Per-device and endpoint analysis is reproducible with
`non_medical_benchmark.analysis.summarize_cpu_split_confirmation`.
