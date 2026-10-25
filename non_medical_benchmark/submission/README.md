# IJCNN 2027 submission package

## Target

IJCNN 2027 is an established conference series whose official history lists IJCNN events back to 1989 (and predecessor IEEE neural-network conferences in 1987). Its current call includes federated, distributed, scalable, and resource-constrained learning. The regular-paper deadline is January 31, 2027. Initial submissions must be anonymous IEEE two-column papers of 4--6 pages, including references.

Official pages:

- [IJCNN 2027 call for papers](https://ijcnn.org/2027/authors/call-for-papers)
- [IJCNN 2027 initial author instructions](https://ijcnn.org/2027/authors/initial-author-instructions)
- [INNS IJCNN past conferences](https://www.inns.org/ijcnn-past-conferences)

## Draft and evidence

- `IJCNN2027_submission.tex` is the anonymous conference-length draft. It uses the IEEE conference class and distinguishes capacity-selected client deployment scores from full-supernet scores.
- `../FOUR_DEVICE_BENCHMARK.md` documents the 105-run, four-worker, three-scale benchmark, worker specifications, times, scores, and limitations.
- `../../.benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json` contains the raw main-benchmark records and run manifest.
- `../../.benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_16_8r_5seeds.json` contains the separate 20-run 16x follow-up (FedAvg, HeteroFL, FjORD, and CC-EFL; five seeds each).
- `../../.benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json` contains the fresh 30-run, ten-seed 8x confirmation for FedAvg, HeteroFL, and FjORD (seeds 6–15), with raw per-device validation selections, test scores, CPU timings, and hardware provenance.
- `../../.benchmarks/model_scaling/digits_fourdevice_8x_confirmation_5seeds.json` contains a separate 15-run, five-seed Digits workload check (seeds 6–10).
- `../run_model_scaling_benchmark.py` is the benchmark runner.
- `../analysis/summarize_cpu_split_confirmation.py` regenerates the per-method and per-device confidence intervals and paired comparisons from a raw result file.
- `../../results_graphs/four_device_cpu_split_scaling.png` is the existing benchmark figure.
- `../../results_graphs/four_device_cpu_split_8x_16x_followup.png` compares time, deployment F1, full-supernet F1, and active parameters at 8x and 16x.
- `../../results_graphs/four_device_8x_confirmation_10seeds.png` plots client-deployment versus full-supernet macro-F1 and the 10-seed runtime distribution.
- `../analysis/plot_four_device_16x_followup.py` regenerates that comparison figure from the two raw result files.
- `EXPLORATORY_ABLATION.md` records the attempted layer-pair coverage schedule and why it is not presented as a successful method.
- `EXPLORATORY_METHODS.md` records the function-preserving growth and replica-ensemble pilots; neither supports a new-method claim.

The strongest bounded result is the fresh ten-seed HAR confirmation: on a 2,806,278-parameter shared MLP, FjORD's four-profile capacity-selected macro-F1 is 0.905 (95% CI 0.885–0.926) with median 9.42 seconds per eight rounds, compared with full-model FedAvg's 30.52 seconds. The 1-thread client selects a 13,686-parameter model with macro-F1 0.889 (0.855–0.923), accuracy 0.896, and mean local time 0.048 seconds per round. FjORD averages 29,958 active client parameters per round. Its complete supernet, however, scores 0.811 with 2.23% coordinate coverage, versus full FedAvg's 0.908. The Digits check likewise shows a deployment/global-model gap. These results support small-client participation in short CPU simulations; they do not establish physical-device deployment, measured memory/energy/network savings, or convergence of the expanded global model.

## Reproduce the main benchmark

From the repository root, use the same software versions captured in the raw JSON run manifest (Python 3.13.7, NumPy 2.2.6, scikit-learn 1.7.2, PyTorch 2.9.0+cpu). The HAR archive must be available under the configured data directory; `--download` asks the runner to fetch it if needed.

```powershell
python -m non_medical_benchmark.run_model_scaling_benchmark `
  --dataset har --download --scales 1 4 8 --capacity-scaling fixed `
  --rounds 8 --clients 4 --client-cpu-threads 1 2 4 8 `
  --reserve-cpu-threads 3 --seeds 1 2 3 4 5 --torch-threads 1 `
  --strategies fedavg fedprox heterofl fjord fedrolex maxnet cc_efl `
  --output .benchmarks/model_scaling/har_fourdevice_cpu_split_widthscale_1_4_8_8r_5seeds.json
```

The runner records split and source hashes, affinity masks, software versions, hardware metadata, raw run records, and summary statistics. Because execution time depends on host load and CPU scheduling, a reproduction should report its own measurements rather than overwrite the archived output.

The 16x extension can be reproduced with the same options, changing `--scales 1 4 8` to `--scales 16`, changing the strategy list to `fedavg heterofl fjord cc_efl`, and writing to a new output file. It is an additional 20-run scale check, not a full repetition of the seven-method matrix.

The fresh 8x HAR confirmation and Digits workload check can be reproduced
with these commands. They use ten and five fresh paired seeds, respectively.

```powershell
python -m non_medical_benchmark.run_model_scaling_benchmark `
  --dataset har --download --scales 8 --capacity-scaling fixed `
  --rounds 8 --clients 4 --client-cpu-threads 1 2 4 8 `
  --reserve-cpu-threads 3 --seeds 6 7 8 9 10 11 12 13 14 15 --torch-threads 1 `
  --strategies fedavg heterofl fjord `
  --output .benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json

python -m non_medical_benchmark.run_model_scaling_benchmark `
  --dataset digits --scales 8 --capacity-scaling fixed `
  --rounds 8 --clients 4 --client-cpu-threads 1 2 4 8 `
  --reserve-cpu-threads 3 --seeds 6 7 8 9 10 --torch-threads 1 `
  --strategies fedavg heterofl fjord `
  --output .benchmarks/model_scaling/digits_fourdevice_8x_confirmation_5seeds.json

python -m non_medical_benchmark.analysis.summarize_cpu_split_confirmation `
  --input .benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds.json `
  --output .benchmarks/model_scaling/har_fourdevice_8x_confirmation_10newseeds_analysis.json
```

## Before submission

1. Check the draft against the IEEE template and verify the final page count and PDF output. The built-in LaTeX compiler was invoked for this source but returned `Unable to find standard directories for platform`; this is an environment/compiler limitation, so a successful compile and page count are still unverified.
2. Add final author metadata only after review; keep `Anonymous Authors` in the submission PDF. Confirm all authors before submission because IJCNN says new authors cannot be added after acceptance.
3. Retain the AI-use disclosure and citation required by IJCNN's current instructions. The draft includes an anonymous acknowledgment and citation; authors should confirm it complies with the final IEEE guidance.
4. Recheck the live call, IEEE template, page limit, and deadline immediately before submission.
5. Decide whether the contribution is strong enough after independent review. The present evidence is a controlled benchmark on one small classification workload, not a new method or a physical-device systems evaluation.

This is a review-ready starting package, not a submitted or accepted paper. No conference submission has been made.
