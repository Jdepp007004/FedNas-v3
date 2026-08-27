# Measured non-medical benchmark results

These are current runs from the isolated research track. UCI runs use six
clients and a 6,000-row train cap; Digits uses the full dataset. Every number
is final held-out test performance after validation-only tier selection. The
intervals are two-sided 95% Student-t intervals over shared random seeds;
paired deltas compare each method with FedAvg on the same seeds.

## Integrated method versus FedAvg

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

## Reference architecture comparison

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

The corrected CIFAR-10 pilot uses five rounds, 1,000 training images, 500 test
images, and two seeds. It is useful for checking the CNN and tier plumbing, not
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

```text
python -m non_medical_benchmark.run_benchmark --dataset digits --rounds 20 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex maxnet cc_efl --output .benchmarks/digits_architectures_5seeds.json
python -m non_medical_benchmark.run_benchmark --dataset adult --download --rounds 10 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/adult_architectures_5seeds.json
python -m non_medical_benchmark.run_benchmark --dataset bank_marketing --download --rounds 10 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/bank_architectures_5seeds.json
python -m non_medical_benchmark.run_benchmark --dataset har --download --rounds 8 --clients 6 --max-samples 6000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/har_architectures_5seeds.json
python -m non_medical_benchmark.run_cifar_benchmark --dataset cifar10 --download --rounds 20 --max-train 10000 --max-test 5000 --seeds 1 2 3 4 5 --strategies fedavg heterofl fjord fedrolex cc_efl --output .benchmarks/cifar10_paper.json
python -m non_medical_benchmark.run_security_protocol --clients 6 --dimension 53431 --seed 7
```
