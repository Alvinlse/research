# Core 2×2 experiment pre-registration

Frozen before any `core-2x2-v1` scheduler result is generated.

## Design

- Factor A: reasoning structure — Single Referee vs Demand + Supply + Referee.
- Factor B: policy family — non-market vs market-based.
- The four cells use the same model, prompt limits, parser, categorical sizing space, workload
  windows, deadlines, simulator configuration, selection interval, and paired inference seeds.
- Each family exposes four global policies. Non-market: FCFS, least laxity, ageing fair-share,
  priority-tier FCFS. Market: the same uniform-price auction with wait, deadline, fair-share, or
  priority valuations. Every bid uses scheduler-visible state only.
- Sizing actions are REQ, ADAPT, GREEDY, and RCON. The response has one default plus optional
  job-ID overrides; code expands this to exactly one action for every active job. The action governs
  both admission and subsequent resize points. RCON's benefit threshold is fixed at 300 seconds.

## Workloads and outcomes

- Forty-eight complete 24-hour windows with 12 hours of warm-up are selected with seed 20260912
  using workload characteristics only. Intervals do not overlap, all contain at least 100 scored
  jobs, and all legacy frozen-2x2 intervals are excluded.
- Split: 24 train, 12 validation, 12 test. Zero-shot selectors do not train; validation selects the
  best fixed policy in each family. The test split is not used for selection.
- Fixed capacity: 80 GPUs. All jobs are malleable with bounds `[1, min(80, 4 × requested)]`.
  Serial fraction is 0.7 and there are ten resize points. Resize overhead is not modeled and is
  reported as zero; resize count and moved GPUs are reported.
- Synthetic deadline: `arrival + 1.0 × requested-walltime estimate`; missing walltime uses the
  fixed 86,400-second estimate. Deadlines are written once into each world.
- Primary outcome: deadline violation rate. Secondary descriptive outcomes: bounded slowdown,
  mean/median wait, useful and raw utilization, completions, resize activity, fairness, model calls,
  tokens, latency, malformed/invalid responses, and fallbacks.

## Statistical analysis

- Inferential unit: workload window. The three seeded responses are averaged within window first.
- Three two-sided paired contrasts on test only: market main effect, multi-agent main effect, and
  the difference-in-differences interaction.
- Report paired mean differences and paired-t 95% confidence intervals.
- Exact sign-flip randomization p-values are Holm-corrected over exactly these three contrasts.
- Secondary metrics are descriptive. No test-window policy, deadline, threshold, or prompt tuning.

## Execution gate

`pins.verify_core_2x2` must validate hashes, split/non-overlap, build parameters, deadlines,
malleable bounds, Amdahl anchoring, and provenance before the sweep can be armed. Legacy partial
results are stored separately and are not eligible for this analysis.

## Amendment 1 (2026-09-13): assertion-only fix re-frozen

`assert_invariants` in `pins/elastisim_bench.py` excluded only `COMPLETED` jobs from
simultaneous-ownership accounting. ElastiSim keeps a **killed** job's assignment in the Python
mirror for its callback as well, while already offering those nodes as free, so a killed job could
trip a false double-ownership assertion and abort an otherwise valid run. The fix adds `KILLED` to
that exclusion.

This touches the feasibility **checker** only. It changes no ordering, no sizing, no bid, no
deadline, no window, no split and no reported metric, so results produced before and after it are
directly comparable. It does not relax the checks that matter: allocations are still bounded by
`[num_nodes_min, num_nodes_max]`, the pool is still a hard cap, and no GPU may be held by two
**active** jobs.

Because the gate hashes the implementation, that one-line change invalidated the frozen hash and
`pins.verify_core_2x2` refused to run, which is the gate behaving correctly. The manifest's
`implementation.sha256` is therefore re-frozen to the fixed code, and the superseded hash is kept in
`implementation.accepted_sha256` so the 483 rows generated before the fix remain verifiable instead
of being silently re-stamped. No design parameter was re-frozen: windows, trace, deadlines, build
configuration, policy families, sizing actions, seeds and the analysis plan are unchanged.
