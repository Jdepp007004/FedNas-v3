# Capstone final story

## Keep the project note

Keep the title from the Phase-3 review:

> **E2E Federated Learning Platform for Heterogeneous Systems**

Keep the hospital/cancer collaboration example as motivation: multiple data
owners train together while participant training rows remain local. Present
the self-serve tabular workspace as the working product feature added to that
same system. The concrete demonstration task is predicting `vital_status`
(alive/dead) from the project's clinical CSV partitions.

The Phase-3 deck itself marks end-to-end integration as upcoming, adaptive
subnet integration as in progress, and shows a failed client fetch plus a
one-client dashboard. Final slides should replace those interim states with
the working create-workspace flow, four-client session, completed rounds, and
the benchmark table. Keep the original system narrative; update the evidence
and project status.

## Final claim to present

The website lets a host define a tabular classification or regression task,
choose a shared MLP size, and invite participants whose CSV training rows stay
on their own client machines. A resource planner assigns each participant a
smaller depth/width prefix. Clients train compact subnetworks, return only the
parameters they touched, and the coordinator sample-weights updates for
overlapping coordinates. Fan-in compensation keeps narrower prefixes on the
same shared parameter scale. The host sees full-model and client-sized
validation metrics separately.

On the project's cancer vital-status task, a 6×512 shared model was evaluated
against full-model FedAvg using four CPU-affinity-limited worker processes and
five seeds. The website clients trained 8,514–148,482 parameters each, versus
1,576,962 in the full model. Their average capacity-sized test macro-F1 was
0.794, compared with 0.854 for the full-model FedAvg reference. Summed
client-round dispatch/wait time was 5.39 s versus 10.56 s for eight rounds.
The complete 6×512 global model reached only 9.42% coordinate coverage and
had unstable test macro-F1; report this as a limitation and design finding.

If asked why the benchmark uses a 6×512 maximum instead of a 6×128 model, note
the five-seed size check: 6×128 reached full coordinate coverage but average
client macro-F1 was 0.746 and client-round dispatch/wait time was 4.90 s,
compared with 0.794 and 5.39 s for 6×512. That smaller configuration is not a
better result; it exposes the trade-off between small-client quality and
coverage in the current prefix rule.

Use this as the main spoken result:

> “The platform lets heterogeneous CPU clients join training with models sized
> to their declared resources. In our four-worker simulation, client models
> reached 0.79 average macro-F1 while using up to 99.46% fewer parameters than
> the full model and about half the measured client-round dispatch/wait time.
> The full 6×512 supernet was not fully trained, so we report the client
> deployment scores separately and treat full-model coverage as the next
> research problem.”

Do not say this proves that phones train LLMs or that a 1.58M-parameter global
model converged. The four devices were virtual CPU-affinity profiles on one
Intel Core Ultra 5 125H host; RAM planner values were not enforced. The task
was a modest tabular clinical classification workload, not an operational
hospital deployment.

## Presentation order

1. **Title and need** — preserve heterogeneous federated learning and the
   cancer collaboration motivation from the existing deck.
2. **Problem** — conventional FL assumes similar client capability; small
   participants may be unable to run a full shared model.
3. **Goal** — make the workflow usable through the website and support CPU
   clients with different budgets.
4. **System** — participant browser/workspace, local worker, FastAPI
   coordinator, resource planner, partial-update aggregation, host dashboard.
5. **Method** — explain model setup, resource-derived depth/width, compact
   prefixes, fan-in compensation, and coordinate-wise sample-weighted merge.
6. **Website demo** — create classification task, select `vital_status`, use
   a public/synthetic setup CSV, join four participant partitions, approve
   clients, complete rounds, show individual resource assignments and both
   validation endpoints, then download the trained model bundle and run a
   prediction on a held-out CSV.
7. **Benchmark design** — 4 virtual CPU clients, host specs, dataset split,
   8 rounds × 3 local epochs, five seeds, full-model FedAvg reference.
8. **Results** — show capacity-sized client scores, active parameter counts,
   and measured round dispatch/wait time; state the full 6×512 global score
   separately.
9. **Limitations** — simulated devices, unenforced RAM profiles, one tabular
   workload, no network/energy/secure-aggregation measurement, incomplete
   full-supernet coverage.
10. **Conclusion and next step** — the platform demonstrates low-budget
    participation today; improve coordinate coverage and validate on real
    heterogeneous hardware next.

## Demo checklist

- Start the coordinator and show `/create`.
- Select classification and `vital_status`, choose the 6-layer/512-width
  shared model, and use public or synthetic host setup data.
- Start four clients with the four supplied participant partitions. For a
  single-laptop demo, describe them as four worker processes; call them four
  physical devices only if the team actually runs four separate machines.
- Set the host minimum updates to four and complete enough rounds to display
  the dashboard history.
- Point out each client's assigned model size and the separate client-mean and
  full-model metrics.
- Show the benchmark chart and explain that its time is the local worker
  dispatch/wait sum, not internet or end-to-end product latency.

## What remains after the capstone

For a research submission, the next experiment must address the observed
9.42% coordinate coverage at 6×512. Test a coverage-aware channel schedule or
another explicit mechanism that lets small clients update different parts of
the shared model over time, then compare full-model convergence and
client-sized quality on multiple datasets. Repeat on separate physical
devices with enforced memory limits and measured network/energy. Establish
novelty against prior work before claiming a new FL algorithm. The current
results are a credible capstone systems prototype, not yet evidence for an
LLM-scale or general-purpose large-model training claim.

Detailed benchmark protocol and per-profile scores: [WEBSITE_METHOD_BENCHMARK.md](../non_medical_benchmark/WEBSITE_METHOD_BENCHMARK.md).
