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

## Amendment 2 (2026-09-13): one inference seed instead of three

Measured from 483 completed rows, the multi-agent cell costs 1,703 s per window run (median 1,426 s),
the single referee 251 s and a fixed policy 8 s. Completing the frozen design needed 123 further
multi-agent runs, about 64 hours serial, which does not fit before the submission target. The
inference seed set is therefore reduced from `[17, 29, 43]` to `[17]`, cutting the remaining
multi-agent work to roughly 11 hours.

What this costs, stated plainly: the analysis plan said the three seeded responses are averaged
within each window first, so within-window response noise was separable from between-window
variance. With one response per window it is not. The inferential unit is unchanged (the window),
the three contrasts are unchanged, the paired sign-flip test and Holm correction are unchanged, and
the test split is still untouched by any selection. Decoding stays temperature 0.1, which is not
greedy, so a single response is a draw from a distribution rather than a deterministic answer, and
per-window differences now carry that sampling noise. This makes the tests more conservative for
detecting a real effect, not less.

No other parameter is re-frozen. `implementation.sha256` is updated because the prereg file and the
tick script are part of the hashed implementation set; the superseded hashes stay in
`accepted_sha256` and `accepted_manifest_sha256` so all rows generated before this amendment, at
either seed count, remain verifiable. Rows already collected at seeds 29 and 43 are kept on disk and
are excluded from the primary analysis by the manifest's seed list rather than deleted.

## Amendment 3 (2026-09-13): two diagnostic flags, both inert by default

Two flags were added to the scheduler while diagnosing why the role-separated referee settled on one
policy: `demand_v2` selects a demand-analyst prompt that defines urgency against the median laxity and
licenses the empty answer, and `packet_v2` moves the analyst statements ahead of the state, adds a
held-intervals counter, relabels the decision-history block, and trims the action menu to names.

Both default to off and the frozen behaviour is unchanged, which is verified rather than asserted: the
state packet rendered by the default path is byte-identical to the packet rendered by the code at the
previous frozen hash. Only the implementation hash moved, so the superseded hash joins
`accepted_sha256` and every row generated before this amendment remains verifiable.

The diagnostic runs those flags drive are training-window only and are reported as diagnostics, never
as protocol results. Their findings, for the record: the anchor fix cut urgency flagging from 32 of 34
epochs to 11 of 34 but made the policy choice *more* concentrated, not less, and the packet changes
diversified choices while costing nine times the mean waiting on the same window. Neither is carried
into the protocol.

## Amendment 4 (2026-09-13): held-out safety floor and interleaved execution

Before the first selector run on a test window, the invalid-action fallback is changed from “hold the
last valid model action” to the independently selected fixed action in the same family. The floor is
selected once from the complete validation fixed-policy menu: FAIR+ADAPT for non-market and
Auction-Deadline+ADAPT for market. An invalid answer is still counted, but it cannot silently prolong
an earlier model error. This uses no test outcome and is applied identically to Single and Multi.

The local inference path already requests Ollama's JSON output mode. The strict parser additionally
rejects unknown ordering names, sizing names, job IDs, and off-family actions; we retain that tested
contract rather than introducing an unpiloted provider-specific schema immediately before test.
The original semantic menu is retained (`packet_v2=false`, `demand_v2=false`).

Execution is moved to an append-only driver that orders the four conditions within each window:
Single-NM, Multi-NM, Single-MKT, Multi-MKT. This changes no simulated action or analysis; it ensures
that an interrupted run leaves complete paired windows rather than an unbalanced collection of cells.
The already disclosed one-seed amendment remains in force. No prompt, model, fallback, window, metric,
or analysis change is permitted after the first test selector row is written.
