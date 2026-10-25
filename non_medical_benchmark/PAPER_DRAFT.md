# Coverage-Controlled Elastic Federated Learning: A Controlled Empirical Study of Resource–Utility Trade-offs

**Authors:** [Add author names and affiliations]

**Corresponding author:** [Add email]
**Draft status:** Evidence-complete working draft for internal review; not yet submission-ready. Author names, affiliations, corresponding-author email, venue formatting, final source-version freeze, and artifact packaging remain open. The text experiments separately evaluate a frozen MiniLM head and final-block BERT-Tiny LoRA adapters; neither updates the full encoder or simulates a real private network.

## Abstract

Federated learning must accommodate clients with unequal compute and storage budgets and non-identically distributed local data. We evaluate Coverage-Controlled Elastic Federated Learning (CC-EFL), an integration of fixed-capacity subnet materialization, bounded rotating channel windows, largest-tier aggregation priority, and FedProx regularization. A corrected benchmark compares it with FedAvg, FedProx, HeteroFL, FjORD, FedRolex, MaxNet-style sampling, and matched controls on four classification tasks using paired five-seed runs. CC-EFL exceeds capacity-sliced FedAvg after Holm correction on Digits and Human Activity Recognition, is inconclusive on Adult and Bank Marketing, and does not lead the strongest baseline on any task. A grouped UCI SMS Spam experiment tests rank-window LoRA adapters in BERT-Tiny's final attention block; no strategy difference survives Holm adjustment, and the backbone remains frozen. A four-worker CPU-affinity study compares 1-, 2-, 4-, and 8-thread virtual clients. In a fresh ten-seed 8× HAR confirmation, FjORD's one-thread client selects a 13,686-parameter model with test macro-F1 0.889 (95% CI 0.855–0.923); across the four profiles, FjORD reaches 0.905 (0.885–0.926) in a median 9.42 s per eight rounds, versus 30.52 s for full-model FedAvg. Active client models average 29,958 parameters versus 2,806,278 for the server model. Yet the full FjORD supernet scores 0.811 versus 0.908 for full FedAvg and updates only 2.23% of coordinates. A five-seed Digits check likewise finds a strong client endpoint (FjORD macro-F1 0.810) but a weak full supernet (0.623 versus FedAvg's 0.964). A separate 16× follow-up reaches 10.33M server parameters but updates under 0.71% of coordinates after eight rounds. These are single-host simulations, not physical-device measurements. The evidence supports a bounded resource–utility benchmark, not a novel optimizer or a claim that the expanded global model converges.

**Keywords:** federated learning; heterogeneous clients; elastic models; resource-aware learning; non-IID data; reproducibility

## 1. Introduction

Federated learning trains a shared model from data retained by multiple clients. In practice, clients can differ in compute and memory budgets, while their local data distributions can be non-identical. Full-model training can exceed a constrained client's budget. Training only fixed prefixes of a larger model can also give some parameter coordinates little or no training.

Prior methods address these challenges through heterogeneous-width models, ordered dropout, rolling subnet extraction, architecture-aware sampling, or client-drift regularization. Combining these mechanisms can obscure which one explains an observed score or resource cost. We therefore present a controlled implementation-level comparison using shared small backbones, fixed client capability tiers, consistent splits, and validation-only capacity selection.

The paper studies CC-EFL as an integration, not as a newly invented optimizer. Its rotating coverage is related to FedRolex, its largest-tier priority is inspired by MaxNet, and its local proximal term is FedProx. Our main research question is empirical: under fixed heterogeneous client budgets and label skew, how do these mechanisms compare in task utility and modeled client cost, and does their combination outperform matched alternatives?

### Contributions

1. A corrected evaluation harness with fixed per-client capability profiles, train-only tabular preprocessing, final-only held-out test evaluation, validation-only per-capacity model selection, and run provenance.
2. A paired five-seed comparison of CC-EFL with established heterogeneous-FL baselines and matched controls on four non-medical tasks.
3. Two grouped-duplicate UCI SMS experiments: a 10-seed frozen-MiniLM head benchmark and a five-seed BERT-Tiny final-block LoRA adapter benchmark.
4. A negative and mixed-results account that identifies where CC-EFL does and does not outperform baselines and separates estimated parameter/payload costs from measured system performance.

## 2. Related Work and Contribution Boundary

HeteroFL trains heterogeneous-width client models and aggregates corresponding model portions. FjORD uses ordered dropout and self-distillation. FedRolex rotates the submodels extracted from a global model. SuperFedNAS/MaxNet uses architecture-aware sampling and gives priority to a large subnet. FedProx adds a proximal term to address local client drift. These works establish the components that motivate our benchmark; our implementation is not a bit-for-bit reproduction of each paper.

Recent work makes novelty claims based only on combining rotating coverage and adaptive allocation particularly difficult to defend. The 2026 preprints “Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning” and “FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients” are close in problem framing and adaptive submodel allocation. They should be checked again at submission time for peer-reviewed versions and newly published related work. We therefore frame CC-EFL as a controlled empirical study and corrected benchmark rather than a new algorithmic principle.

References:
- McMahan et al., Federated Averaging, AISTATS 2017. https://arxiv.org/abs/1602.05629
- Li et al., FedProx, MLSys 2020. https://arxiv.org/abs/1812.06127
- Li et al., SCAFFOLD, ICML 2020. https://arxiv.org/abs/1910.06378
- Diao et al., HeteroFL, ICLR 2021. https://arxiv.org/abs/2010.01264
- Horvath et al., FjORD, NeurIPS 2021. https://arxiv.org/abs/2102.13451
- Alam et al., FedRolex, NeurIPS 2022. https://arxiv.org/abs/2212.01548
- SuperFedNAS / MaxNet, ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10165.pdf
- Moayedikia and Troncoso Lora, Capacity Confounds and Coverage Guarantees in Adaptive Sub-model Federated Learning, arXiv:2608.07157 (2026 preprint). https://arxiv.org/abs/2608.07157
- FEAST: Federated Shared-Space Training for Resource-Heterogeneous Clients, arXiv:2608.09250 (2026 preprint). https://arxiv.org/abs/2608.09250

## 3. Method

### 3.1 Setting

There are K clients. Client i holds data D_i and receives a fixed depth/width capability tier for the entire run. The server maintains an elastic supernet. At each communication round, each participating client materializes a subnet that fits its fixed tier, trains it locally, and returns updates for the coordinates it used. Client partitions are seeded Dirichlet label-skew draws. The benchmark reports a full-supernet score separately from each capacity-selected deployment score.

### 3.2 CC-EFL

CC-EFL combines four implemented choices:

1. Fixed capability tiers are assigned per client and seed and remain unchanged through training.
2. Bounded contiguous channel windows rotate over the hidden dimension. For round r, client id c, subnet width w, and supernet width H, the start offset is (r+c) mod (H-w+1). For a fixed tier this enumerates all legal non-wrapping start positions in a cycle of H-w+1 rounds.
3. The largest active tier, measured by depth times width, receives priority beta_r; other active tiers receive 1-beta_r, with beta_r = 0.2 + 0.35(1 + cos(pi*r/max(R-1,1))). This is a MaxNet-inspired weighting choice, not a claim of new theory.
4. FedProx local training adds mu/2 times the squared distance from the round-start server parameters to the local objective. Task-specific values are listed in the run manifests.

For a parameter coordinate j, aggregation includes only clients that updated j. If C_r(j) is that set, n_i is the local sample count, and p_i^r is the implemented client priority, then the server update is the weighted mean over C_r(j); coordinates with no update retain their previous value. The exact source and all outputs are preserved in the run manifests.

A previous schedule advanced by the subnet width and could have a shorter modular cycle than the number of legal starts. Those v2 results are retained as protocol-audit artifacts. All headline results below use the corrected bounded unit-stride schedule.

### 3.3 Comparison methods and controls

The primary comparison includes FedAvg, FedProx, HeteroFL, FjORD, FedRolex, MaxNet, CC-EFL, CC-EFL without the proximal term, CC-EFL with uniform tier weighting, and CC-EFL with random channel windows. A CC-EFL + FjORD distillation extension is included as an exploratory comparator in the four-task matrix. SCAFFOLD is a separate matched-SGD diagnostic and is not mixed into the Adam primary matrix.

The random-window control preserves the same client depth, width, sample counts, and valid offset range, changing only the channel-window schedule. It is included to test whether systematic rotation is needed in this harness.

### 3.4 Model selection

For every fixed capability tier, candidate subnet performance is measured on validation data. The smallest feasible tier within 0.2 percentage points of the best validation score is selected. Digits uses validation accuracy; Adult, Bank Marketing, HAR, and SMS use validation macro-F1. The test set is not used for round-by-round decisions. After model selection is frozen, the selected deployments and predeclared full-model diagnostic are evaluated on held-out test data.

## 4. Experimental Setup

### 4.1 Four-task matrix

| Dataset | Task and source | Client count | Dirichlet alpha | Rounds | Seeds |
|---|---|---:|---:|---:|---:|
| Digits | 1,797 handwritten 8x8 digit images from scikit-learn | 6 | 0.5 | 20 | 1–5 |
| Adult | UCI Adult income classification; at most 6,000 training examples per run | 6 | 0.3 | 10 | 1–5 |
| Bank Marketing | UCI Bank Marketing subscription prediction; at most 6,000 training examples per run | 6 | 0.3 | 10 | 1–5 |
| HAR | UCI smartphone Human Activity Recognition; at most 6,000 training examples per run | 6 | 0.5 | 8 | 1–5 |

Numeric transforms and categorical vocabularies for tabular tasks are learned from training partitions only; unseen holdout categories map to all-zero encodings for their field. Capability tiers are fixed within a run. Each task uses its configured elastic MLP, balanced class weighting where recorded, and Adam in the primary matrix. Exact parameters, splits, and source hashes are stored with the output artifacts.

### 4.2 Frozen MiniLM head-adaptation experiment

The source is the UCI SMS Spam Collection (5,574 messages; source archive SHA-256 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3). Exact duplicate messages are assigned to one split using a stratified grouped 20-fold partition: folds 0–13 train, 14–16 validation, 17–19 test. Counts are 3,908 train, 837 validation, and 829 test; the split has zero exact-text overlap between every pair of partitions. The split manifest hash and class counts are in the output JSON.

We use sentence-transformers/all-MiniLM-L6-v2 at pinned revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, with 384-dimensional, unnormalized embeddings and maximum sequence length 128. The 22,713,216-parameter encoder is frozen. The actual federated model is a maximum 83,330-parameter elastic MLP head (depth 3, hidden dimension 128, widths 32/64/96/128). Only this head is trained and communicated. The embeddings were generated or loaded in one local preprocessing stage before federated simulation, so this experiment does not test private, on-device encoder execution or encoder fine-tuning.

The head experiment uses six clients, Dirichlet alpha 0.3, 20 rounds, ten seeds (1–10), Adam with learning rate 0.001, batch size 128, one local epoch per round, balanced class weights, and FedProx mu 0.001 where applicable. Centralized head training uses the same number of epochs. The source dataset is available from https://archive.ics.uci.edu/dataset/228/sms+spam+collection; the encoder model card is https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.

### 4.3 BERT-Tiny LoRA adapter experiment

We use google/bert_uncased_L-2_H-128_A-2 at pinned revision 30b0a37ccaaa32f332884b96992754e246e48c5f (4,385,920 base parameters). The final transformer's query and value projections receive rank-window LoRA adapters, and a two-class head is trained. The remaining encoder weights, including the first transformer block and the non-adapter weights in the final block, remain frozen. At maximum rank 4, adapters plus head have 2,306 trainable parameters; heterogeneous client ranks are 1, 2, or 3. The frozen prefix is evaluated after client partitioning and cached separately for each client partition, then reused across methods. This reduces repeated computation in this simulation but does not test end-to-end private local encoding or communication of the backbone.

This experiment uses the same grouped split and archive as Section 4.2: 3,908 train, 837 validation, and 829 test messages, with zero exact-text overlap between partitions. It runs six clients, seeded Dirichlet alpha 0.3, five rounds, seeds 1–5, one local epoch per client per round, batch size 128, maximum sequence length 64, Adam at 0.0002, and FedProx mu 0.001 where enabled. Client rank capacities are fixed at 1, 2, or 3; FedAvg and FedProx update rank 4. The local objective uses class-weighted cross entropy. The run comprises 45 federated method-seed executions and five centralized controls. Per-client partition sizes and ham/spam counts are recorded because the chosen Dirichlet process induces quantity as well as label skew.

The model source is the [official BERT-Tiny model card](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), pinned to [revision 30b0a37ccaaa32f332884b96992754e246e48c5f](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f). The UCI dataset and terms are described in Section 4.2.

### 4.4 Outcomes and inference

The primary endpoint is validation-selected capacity deployment test score: accuracy for Digits, macro-F1 for the other tasks. We report mean and two-sided Student-t 95% confidence intervals over paired seeds. Paired comparisons use the same task seed and split; p-values use two-sided paired t-tests and Holm family-wise adjustment. The four task-level CC-EFL versus capacity-sliced FedAvg tests form one family. Within each task, comparisons of the remaining strategies against FedAvg form a separate family; CC-EFL ablation comparisons form another family. Confidence intervals shown are unadjusted.

Resource reporting separates active client parameter counts, selected deployed parameter counts, per-example forward-FLOP estimates, simulated upload/download payload bytes, wall-clock time, and parameter-coordinate coverage. These are harness estimates. They do not establish measured latency, energy, peak memory, real-network throughput, or total device storage unless the encoder is explicitly included.

## 5. Results

### 5.1 Four-task primary result

| Task | CC-EFL selected score (95% CI) | Paired difference vs capacity-sliced FedAvg (95% CI) | Holm-adjusted p | Highest mean comparator |
|---|---:|---:|---:|---|
| Digits accuracy | 74.55% (70.73–78.37) | +8.74 pp (+4.37, +13.11) | 0.0154 | FjORD, 88.86% |
| Adult macro-F1 | 0.6459 (0.5328–0.7590) | −0.0476 (−0.1689, +0.0737) | 0.3372 | CC-EFL random window, 0.7270; paired evidence uncertain |
| Bank Marketing macro-F1 | 0.6756 (0.6336–0.7176) | +0.0520 (+0.0009, +0.1031) | 0.0949 | HeteroFL, 0.6792 |
| HAR macro-F1 | 0.8618 (0.8231–0.9006) | +0.1716 (+0.1061, +0.2371) | 0.0076 | FjORD, 0.8965 |

Holm correction across the four task-level CC-EFL/FedAvg comparisons leaves positive differences on Digits and HAR, but not on Adult or Bank Marketing. The highest mean results on Digits and HAR belong to FjORD; HeteroFL leads on Bank. Thus CC-EFL does not lead the comparator set across tasks. Full per-strategy tables, all paired intervals, and the resource estimates are in RESULTS.md and the four v3 JSON files.

Across these runs, CC-EFL active client parameter counts are 61.8–74.7% lower than the full FedAvg model. Selected deployment parameter savings relative to the full FedAvg model are task-dependent. These modeled reductions are not real hardware measurements and the full-model reference is distinct from the capacity-sliced FedAvg score used for utility comparison.

### 5.2 Fixed-client-budget supernet scaling

To test whether a larger shared model can be trained without increasing the heterogeneous clients' model-size tiers, we expanded the HAR server supernet from width 96 to widths 384 and 768 while keeping client widths fixed at 24/48/72. All runs used eight rounds. The width-96 and width-384 comparisons used five seeds; width 768 used three. The client subnets are physically materialized. The reported FP32 weight sizes count model parameters only; they exclude gradients, optimizer state, activations, and runtime overhead.

| Server max width | Full supernet parameters | Mean active client parameters | Active-parameter reduction vs full FedAvg | CC-EFL capacity-selected macro-F1 (95% CI) | CC-EFL / FedAvg full-supernet macro-F1 | Paired delta vs capacity-selected FedAvg (95% CI) | CC-EFL coordinate coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 96 | 92,742 | 31,974 | 65.5% | 0.8321 (0.7734–0.8908) | 0.8612 / 0.9059 | +0.1440 (+0.0274, +0.2607) | 81.4% |
| 384 | 813,318 | 31,974 | 96.1% | 0.8036 (0.7502–0.8570) | 0.8216 / 0.8956 | +0.1294 (+0.0939, +0.1649) | 9.3% |
| 768 | 2,806,278 | 31,974 | 98.9% | 0.7459 (0.6825–0.8093) | 0.7300 / 0.8824 | +0.1197 (−0.0019, +0.2413) | 2.7% |

The paired intervals are descriptive; this exploratory scaling sweep did not use a predeclared non-inferiority margin or adjust the scaling comparisons for multiple testing. CC-EFL's mean is below the strongest heterogeneous comparator at each scale: FjORD scores 0.8859, 0.8572, and 0.8282 at widths 96, 384, and 768. HeteroFL and MaxNet also exceed CC-EFL at widths 384 and 768. The fixed client tiers produce the same active parameter/FLOP budget for all subnet methods, so this reduction demonstrates the benefit of subnet training over full-model FedAvg, not a resource advantage unique to CC-EFL.

The server model grows by about 30× in parameter count from width 96 to width 768, while the average client subnet remains 31,974 parameters (about 0.122 MiB for FP32 weights). The full CC-EFL supernet macro-F1 falls to 0.8216 at width 384 and 0.7300 at width 768, compared with full-width FedAvg at 0.8956 and 0.8824. Coverage also drops: after eight rounds, CC-EFL touches only 9.3% of width-384 coordinates and 2.7% at width 768. The simulator completes the larger runs with unchanged client tiers, but the expanded global model is sparsely updated and does not retain full-model utility. The run was CPU-only and did not measure peak memory or performance on an edge device. See the full baseline tables and reproducibility commands in RESULTS.md.

A matched random-window CC-EFL control uses the same client parameters, FLOPs, and payload. At width 384, it raises coordinate coverage from 9.3% to 45.3% while reducing capacity-selected macro-F1 from 0.8036 to 0.5845; the paired random-minus-systematic difference is −0.2191 (95% CI −0.4133 to −0.0249, unadjusted). At width 768, coverage rises from 2.7% to 23.6%, while macro-F1 is 0.4636 versus 0.7459; the three-seed paired interval is wide and includes zero. These controls show a coverage–utility trade-off under the short schedule, but they do not solve the low global coverage at larger widths.

### 5.2.1 Four-client CPU-affinity scaling follow-up

We ran a separate five-seed follow-up with four concurrent virtual client processes, affinity-limited to 1, 2, 4, and 8 logical processors on one Intel Core Ultra 5 125H host. The clients received fixed depth/width capacities 1/24, 1/24, 2/48, and 5/72; the eight-thread worker used the largest tier. The coordinator used three other logical processors. Across 1×, 4×, and 8× server widths, seven methods, five seeds, and eight rounds, the matrix contains 105 runs. At 8×, the shared HAR MLP has 2,806,278 parameters while the mean active subnet for heterogeneous methods has 29,958 parameters per client (98.9% fewer; 0.114 MiB of FP32 weights, excluding runtime state).

| 8× method | Median time / 8 rounds | Capacity-selected test macro-F1 / accuracy | Full-supernet test macro-F1 / accuracy |
|---|---:|---:|---:|
| FedAvg | 36.39 s | 0.658 / 0.695 | 0.905 / 0.907 |
| HeteroFL | 10.15 s | 0.862 / 0.874 | 0.824 / 0.834 |
| FjORD | 16.94 s | 0.902 / 0.908 | 0.762 / 0.793 |
| CC-EFL | 17.92 s | 0.828 / 0.849 | 0.829 / 0.848 |

The 8× CC-EFL capacity-selected macro-F1 difference versus capacity-selected FedAvg is +0.171 (unadjusted paired 95% CI +0.112 to +0.229); FjORD has the highest capacity-selected score, and HeteroFL has the shortest time. CC-EFL's full-supernet score remains below full-model FedAvg, with only 2.6% coordinate coverage, so the evidence supports small-subnet participation in a short simulation, not successful training of the entire expanded model. The four profiles, CPU assignments, all 21 method/scale accuracy and time results, and reproducibility command are in FOUR_DEVICE_BENCHMARK.md. The experiment is a CPU-affinity simulation on one host: affinity is not an exclusive hardware reservation, and it measures neither physical-device memory/energy nor network latency.

A separate 16× follow-up adds five paired seeds for FedAvg, HeteroFL, FjORD, and CC-EFL (20 runs total), with the same client tiers and eight rounds. The maximum server MLP has 10,331,142 parameters (39.41 MiB of FP32 weights); subnet clients still average 29,958 active parameters. Median simulation times are 56.9 s for FedAvg, 20.1 s for HeteroFL, 20.3 s for FjORD, and 30.0 s for CC-EFL. Capacity-selected macro-F1 is 0.737, 0.857, 0.882, and 0.801, respectively. At this scale, CC-EFL reaches only 0.825 full-supernet macro-F1 versus 0.904 for full-model FedAvg, after 0.70% coordinate coverage. The 10.33M-parameter result broadens the size sweep but remains a single-host CPU simulation with sparse global updates, not physical-device validation or evidence of full-model convergence. See the separate 16× table in FOUR_DEVICE_BENCHMARK.md and raw output in `.benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_16_8r_5seeds.json`.

### 5.2.2 Fresh ten-seed confirmation and per-device utility

To estimate seed sensitivity more directly, we ran FedAvg, HeteroFL, and FjORD on the same 8× HAR workload for ten new paired seeds (6–15; 30 runs). The shared model has 2,806,278 parameters, and four affinity-limited workers use 1/2/4/8 CPU threads. FjORD has a capacity-selected deployment macro-F1 of 0.905 (95% CI 0.885–0.926) and accuracy 0.909, with a median simulation time of 9.42 seconds per eight rounds; full-model FedAvg takes 30.52 seconds and has full-model macro-F1 0.908. The paired median runtime ratio is 2.97. FjORD averages 29,958 active parameters per client-round, 98.93% fewer than full-model FedAvg.

The smallest worker's selected 13,686-parameter deployment has macro-F1 0.889 (0.855–0.923) and accuracy 0.896. Its mean local time is 0.048 seconds per round, compared with 1.874 seconds for full-model FedAvg on the same 1-thread worker. The two lowest CPU profiles share a model tier; the 4-thread and 8-thread FjORD deployments reach macro-F1 0.919 and 0.924, respectively. These per-profile results are validation-selected and are reported with unadjusted seed-level intervals.

This is a client-deployment result, not a global-supernet convergence result. FjORD's full supernet scores 0.811 (0.726–0.895), with only 2.23% coordinate coverage; full FedAvg scores 0.908 (0.892–0.925). FjORD deployment versus full FedAvg has a descriptive paired difference of −0.003 (95% CI −0.021 to +0.015), but this compares different endpoints and is not a formal non-inferiority test. On a second workload, a five-seed Digits run gives FjORD deployment macro-F1 0.810 (0.743–0.877) from an average of 4,282 active parameters, while its full supernet reaches 0.623 versus 0.964 for full FedAvg. This corroborates the endpoint separation and shows that HAR's near-full deployment score does not generalize as full-model quality. Results, per-device tables, and reproduction commands are in FOUR_DEVICE_BENCHMARK.md; raw outputs are in `.benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json` and `.benchmarks/model_scaling/digits_fourdevice_8x_confirmation_5seeds.json`.

### 5.3 SMS head adaptation

Centralized head training achieves test macro-F1 0.9691 (95% CI 0.9657–0.9726). Federated method results over ten seeds are:

| Strategy | Capacity-selected macro-F1 (95% CI) | Paired difference vs FedAvg (95% CI) | Holm-adjusted p |
|---|---:|---:|---:|
| FedAvg | 0.8591 (0.7531–0.9652) | Reference | — |
| FedProx | 0.8461 (0.7471–0.9451) | −0.0131 (−0.0248, −0.0013) | 0.2944 |
| HeteroFL | 0.8674 (0.7668–0.9680) | +0.0083 (−0.0887, +0.1053) | 1.0000 |
| FjORD | 0.8102 (0.6652–0.9552) | −0.0489 (−0.1172, +0.0193) | 1.0000 |
| FedRolex | 0.8633 (0.7560–0.9707) | +0.0042 (−0.0837, +0.0921) | 1.0000 |
| MaxNet | 0.8667 (0.7674–0.9660) | +0.0076 (−0.0912, +0.1063) | 1.0000 |
| CC-EFL | 0.8565 (0.7480–0.9650) | −0.0026 (−0.1020, +0.0968) | 1.0000 |
| CC-EFL no proximal | 0.8667 (0.7605–0.9728) | +0.0075 (−0.0969, +0.1119) | 1.0000 |
| CC-EFL uniform weighting | 0.8663 (0.7663–0.9662) | +0.0071 (−0.0821, +0.0963) | 1.0000 |
| CC-EFL random window | 0.8915 (0.8545–0.9284) | +0.0323 (−0.0636, +0.1283) | 1.0000 |

No strategy comparison survives the nine-comparison Holm family. The random-window control has the highest observed mean but its paired interval includes zero. CC-EFL's paired interval is also broad and includes zero. The head-only reductions must not be presented as whole-model cost reductions: including the frozen encoder, CC-EFL saves approximately 0.27% of total parameter count relative to full-head FedAvg, while its roughly 62.5% payload reduction is for communicated head updates only. The encoder is frozen and centrally used to create embeddings, so no privacy or on-device fine-tuning conclusion follows.

### 5.4 BERT-Tiny LoRA adapter benchmark

Centralized rank-4 adapter training reaches test macro-F1 0.8547 (95% CI 0.8242–0.8853). The five-seed federated capacity-selected results are:

| Strategy | Capacity-selected test macro-F1 (95% CI) | Paired delta vs FedAvg (95% CI) | Holm-adjusted p | Adapter/head upload bytes per round |
|---|---:|---:|---:|---:|
| FedAvg | 0.6307 (0.4166–0.8448) | Reference | — | 55,344 |
| FedProx | 0.6307 (0.4166–0.8448) | 0.0000 (0.0000, 0.0000) | 1.0000 | 55,344 |
| HeteroFL | 0.6307 (0.4161–0.8453) | −0.000003 (−0.00147, +0.00146) | 1.0000 | 30,768 |
| FedRolex | 0.6312 (0.4174–0.8450) | +0.00048 (−0.00041, +0.00138) | 1.0000 | 30,768 |
| MaxNet | 0.6223 (0.3979–0.8467) | −0.00838 (−0.02668, +0.00992) | 1.0000 | 30,768 |
| CC-EFL | 0.6227 (0.3958–0.8495) | −0.00802 (−0.02776, +0.01173) | 1.0000 | 30,768 |
| CC-EFL without proximal term | 0.6227 (0.3958–0.8495) | −0.00802 (−0.02776, +0.01173) | 1.0000 | 30,768 |
| CC-EFL uniform weighting | 0.6312 (0.4174–0.8450) | +0.00048 (−0.00041, +0.00138) | 1.0000 | 30,768 |
| CC-EFL random window | 0.6235 (0.3961–0.8509) | −0.00719 (−0.02689, +0.01250) | 1.0000 | 30,768 |

FedAvg in this table trains all clients at rank 4 and reports test utility from validation-selected capacity deployments, as described in Section 4.3. No method comparison survives Holm correction. CC-EFL has neither a significant gain over FedAvg nor a lead over the baselines. The very wide seed-level intervals and identical observed scores for some method/seed pairs warrant a larger replication before ranking methods. The mean client-size coefficient of variation is 0.927 across the seeded partitions, demonstrating strong quantity imbalance in addition to label skew.

A post-hoc quantity-balanced sensitivity run uses the same five seeds and dataset split but assigns 651–652 examples per client (mean size CV 0.00072). It draws spam allocations from a Dirichlet distribution, projects counts to client capacities with at least one example per class, and fills remaining slots with ham. This changes the class-allocation procedure as well as client sizes, so it is not a pure quantity-skew intervention. CC-EFL's mean macro-F1 is 0.5877 (95% CI 0.3008–0.8746); the paired balanced-minus-original difference is −0.0350 (−0.3676 to +0.2975), Holm-adjusted p=1.0. All nine strategy-by-protocol tests are non-significant after Holm correction. Seed variation does not shrink for FedAvg or CC-EFL in this small control, so the cause of instability remains unresolved.

The estimated 44.4% per-round payload reduction for heterogeneous methods counts only float32 trainable adapter/head coordinates (7,692 versus 13,836 coordinates per round). It excludes metadata and the frozen BERT-Tiny backbone, which remains stored and executed. It is not a whole-model size, energy, latency, or real-network measurement. The source raw output and corrected resource/partition report are preserved in RESULTS.md. The correction fixes resource accounting from recorded per-round ranks and does not change model scores.

### 5.5 Security and image pilots

Small security simulations show failures under the tested sign-flip and backdoor settings and a poor privacy–utility result for the tested differential privacy calibration. The secure aggregation code is a functional simulator, not a deployed network protocol. These results are limitations, not claims of security or privacy guarantees. The tiny CIFAR-10 pilot is plumbing validation only and is excluded from headline accuracy claims. Details and exact boundaries are documented in RESULTS.md.

## 6. Discussion

The most stable result is that client-side parameter counts can be reduced by materializing a model within a fixed capability tier. Whether that yields an acceptable utility point depends on dataset and method. On the corrected four-task matrix, CC-EFL exceeds capacity-sliced FedAvg after task-family correction on Digits and HAR, but not on Adult or Bank Marketing. Stronger means from FjORD and HeteroFL show that the integration does not dominate established alternatives. The grouped frozen-MiniLM and BERT-Tiny adapter experiments also show no adjusted strategy differences. Balancing SMS client sizes reduces size CV but does not narrow the five-seed score variation; label skew or other sources of instability remain.

These findings support a benchmark and empirical-analysis contribution only if the paper foregrounds the corrected protocol, fair comparisons, and negative results. They do not substantiate a new optimization method. Any claim of a novel mathematical allocation or coverage rule needs a formal definition, proof of distinction from existing schedules, ablations that isolate the rule, matched-budget random controls, and stronger independent workloads.

The CPU-affinity follow-up shows that four client processes with different CPU and model-size budgets can participate in training rounds for a shared model with 2.81M parameters. It does not establish deployment on four small devices: the processes share one workstation, memory is not capped per client, and no edge latency, energy, or communication costs were measured. At the largest scale, capacity-selected client utility is materially higher for subnet methods than for post-hoc FedAvg slices, but global-model coordinate coverage remains very low and the full CC-EFL score trails full-model FedAvg.

## 7. Limitations and Remaining Work

1. The corrected four-task matrix has five seeds; the new CPU-split HAR confirmation has ten seeds for three methods, and the Digits CPU-split check has five. The additional CPU-affinity extension for Adult and Bank Marketing was not completed. Both text experiments use one public dataset; the BERT-Tiny adapter run has five seeds and five rounds.
2. The four-client scaling follow-up uses affinity-limited processes on one workstation, not physical devices. It does not impose per-client RAM limits or measure edge latency, energy, intermittent connectivity, or protocol overhead.
3. BERT-Tiny updates LoRA adapters only in the final attention block plus a classifier head; its backbone remains frozen. It is not full encoder fine-tuning.
4. The frozen prefix is cached centrally in this single-process simulation; this does not validate private, on-device encoding or production federated communication.
5. The joint-skew SMS partitions exhibit strong quantity skew, while the post-hoc balanced-size control uses a different label-allocation procedure. Both conditions retain severe label skew, one public dataset, and only five seeds; they do not isolate the causes of score variation.
6. The chosen results do not show a universal accuracy advantage or CC-EFL superiority over the best method.
7. In the fixed-client-budget scaling probes, coordinate coverage falls to 2.23% on the 8× HAR confirmation and below 0.71% at 16× after eight rounds. Larger-supernet training needs a coverage-aware schedule and more evaluation before it supports a global-model scale-up claim.
8. Security/DP results do not justify deployment guarantees. CIFAR results are only a tiny pilot.
9. The current contribution boundary must be rechecked against current related work before submission. Author metadata, venue-specific formatting, artifact packaging, and source-version freeze are open.

## 8. Conclusion

We evaluated CC-EFL, an integration of fixed-capacity subnet materialization, bounded rotating channel windows, largest-tier weighting, and FedProx, against heterogeneous federated-learning baselines. Corrected paired results are mixed, and stronger mean baselines exist. A fresh ten-seed CPU-affinity confirmation shows that a 1-thread client can train a 13,686-parameter FjORD deployment with macro-F1 0.889 while participating in an eight-round simulation against a 2.81M-parameter server model; FjORD's four-profile deployment averages 0.905 and runs about three times faster than full-model FedAvg on this host. The full shared FjORD model, however, scores 0.811 after only 2.23% coordinate coverage. A five-seed Digits check also separates client deployment quality from full-supernet quality. Thus the evidence supports short-run participation by small virtual clients and a capacity-resolved resource–utility comparison, not physical-device performance, convergence of the enlarged global model, or a new federated optimizer. The text and security experiments retain their stated limits.

## Appendix A. Information for venue matching

This work concerns federated learning, heterogeneous model capacity, and reproducible ML systems evaluation. The current draft is empirical and does not claim new theory or a novel optimizer. The institution's S-grade rule provided by the user requires a conference series age of at least ten years. A venue recommendation should separately verify fit, series start year, next actual deadline, notification timing, archival peer-reviewed proceedings, indexing, and whether the required grade criteria accept that publication type. Historical venue-wide acceptance rates are not the probability that this particular paper will be accepted. The BERT-Tiny experiment adds trainable final-block adapters but does not train the full encoder or establish an algorithmic novelty claim.
