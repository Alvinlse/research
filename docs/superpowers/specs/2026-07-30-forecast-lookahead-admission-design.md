# Forecast-lookahead admission — MECHANISM DESIGN (not a pre-registration)

**Date:** 2026-07-30  **Branch:** `referee_allocator`
**Status:** design only. **No experiment is pre-registered here and no code is authorised by this
document.** It describes a mechanism whose turn has not come: it is gated behind
`2026-07-30-exp63-admission-lever-design.md`, which is itself written and unrun.

## 1. Why this document exists, and what it is not

The idea it captures — *forecast the near-future cluster state and let that shape the decision* —
came up as "feed a forecast to the debate so the LLM can plan long-term." Two findings redirect it,
and both are recorded here so the redirection is not re-derived later:

- **The LLM is the wrong consumer.** Exp 92 (n=32) showed the winning escalation shape
  (`make_policy_corrected`) is a literal no-op in-sim — 843 escalations, 0 LLM calls, 0 proposals,
  0 ticks changed, allocation identical to `market` digit for digit — because the v2020 replay
  carries no text channel. Exp 94 then closed the generated-channel route. A numeric forecast handed
  to a reasoning layer in this sim would be consumed by a layer that has already been measured
  inert. The deterministic core is the consumer.
- **The existing forecaster cannot be reused.** `pins/forecast/` is a *per-job telemetry*
  forecaster (`CHANNELS = [gpu_util, gpu_mem_gb, cpu_util, mem_gb]`, `BIN_S = 10`, `HORIZON = 30`)
  trained on MIT Supercloud samples. It needs a lookback window of gridded samples, and v2020 has
  no such input — `pins/build_worker_holdout.py:7` records that `pai_sensor_table` has no timestamp
  (one row per worker), and `job_usage.csv` is per-job mean/max. Its *static* counterpart is
  already in the tree and already measured: `pins/eval/pred_job_usage.csv` ships `p10,p50,p90,truth`
  and `trace_replay.load_usage_quanta(quantile="prod-p90")` already feeds the P90 hedge (Exp 36).
  The source of the Supercloud model was trimmed at commit `1534db2` and survives only as `.pyc`;
  restore via `git show 1534db2^:pins/forecast/<file>` if ever needed. **This design does not use
  it.**

What this design proposes is a *cluster-load* forecast — how contended the pool will be over the
next several ticks — feeding `admission_plan`. That signal is derivable from v2020, which does carry
arrivals: `data/alibaba-gpu-v2020/replay_jobs.csv` is `job_name,arrival,dur,quanta`.

## 2. The gap in the current lever

`negotiation_protocol.admission_plan(waiting, free, reserve=0)` is myopic by construction:

```python
rank = sorted(waiting, key=lambda w: (w["tier"] != "prod", ADMIT_ORDER.get(w["deadline"], 1),
                                      w["base_gpus"], w["jid"]))
priority, defer, left = {}, [], max(0, free - reserve)
```

Both halves are present-tick only. The budget is `free - reserve` *now*; the ranking keys on
`(tier, deadline, base_gpus, jid)` — no term refers to anything future. The docstring's own
justification for the lever is a present-tick argument: concentration, so admitted jobs run at full
rate instead of every job trickling.

The failure mode a lookahead would address is concrete and follows from the ranking: *smaller base
first* fills the pool with cheap best-effort jobs, and a prod burst arriving two ticks later finds
nothing free. The myopic plan cannot see the burst. Holding capacity for it is exactly what the
`reserve` scalar does — but as a fixed number, not one sized to predicted demand.

So the mechanism claim is narrow and should be stated that way: **a lookahead lets `reserve` be
sized by prediction instead of set as a constant.** That is the whole of it. It is not a new
scheduler.

## 3. Position in the queue

This is gated, in order:

1. **Exp 63** runs as specified (`2026-07-30-exp63-admission-lever-design.md`). Unchanged, no new
   code. Its H4 manipulation check decides whether the lever engages at all.
2. **Exp 63 decision-rule branch 2** — if H2 pays and H3 is null, `admission_plan` gets wired into
   the market arm. Already pre-named there; needs no spec of its own.
3. **This design** becomes runnable only if (2) shows the deterministic lever pays on the market
   arm. If concentration buys ~0 there, a better-informed budget has nothing to improve and this
   document is closed unrun.

Skipping to (3) would also break the pre-registration discipline the Exp 63 spec sets: it bounds
its own conclusion to the negotiation-era arms precisely so the market wiring stays a declared
follow-up rather than an unblinded choice.

## 4. Mechanism

One change, expressed as one additional argument:

```
admission_plan(waiting, free, reserve=0, lookahead=None)
```

`lookahead` supplies predicted contention over the next *H* ticks. It affects the **budget only** —
the ranking keys are untouched, so a null `lookahead` reproduces today's behaviour byte for byte
(the same default-off discipline `--admit` itself follows).

The budget becomes `left = max(0, free - max(reserve, predicted_prod_demand_within_H))`: hold back
whichever is larger, the static reserve or the capacity predicted to be claimed by arriving prod
work inside the horizon. Deferred jobs are still only ever jobs that have never started — the
replay world does not preempt, and that invariant is not touched.

*H* is a free parameter and must be fixed before any run, not swept post hoc.

## 5. Arms

Mirrors the oracle/predicted pattern `make_trace_workload` already uses, for the same reason:

- **`oracle-lookahead`** — reads the window's real future arrivals from the trace. A **ceiling**,
  reported as such, never as a result. Its purpose is to bound the headroom: if the oracle buys
  nothing, no forecaster can.
- **`forecast-lookahead`** — predicts from observable history only (arrival rate and mix over a
  trailing window). The honest arm, and the only one that can be claimed.

The distinction is load-bearing because `replay_jobs.csv` makes peeking trivial: reading `arrival`
ahead of the clock is an oracle, and would be very easy to report as a forecast by accident.

Run the oracle first. It is cheap, needs no model, and its result decides whether the forecaster is
worth building — the same de-risking logic that puts Exp 63 ahead of this document.

## 6. Threats to validity

- **The competing mechanism is already there.** `reserve` is static headroom for incoming prod. The
  comparison must therefore be against a **tuned** static reserve, not against `reserve=0`;
  otherwise a lookahead win is just the discovery that some reserve beats none, which Exp 22–42
  already established.
- **The rigid world bounds the effect.** `two_sided_sim` never preempts base allocations, so a
  lookahead can only change *start* decisions, never reclaim capacity from a running job. The
  reachable effect size is correspondingly small, and a null result here is weaker evidence against
  lookahead-in-general than it looks.
- **The censoring guard carries over verbatim** from Exp 63 §5: deferral can manufacture SLA by
  never starting the jobs that would have missed. Any SLA gain accompanied by a significant drop in
  `finished` or a rise in `starved`/`wait_max` is a censoring artefact, not a win. A lookahead
  budget defers *more* than the myopic one by design, so this guard binds harder here.
- **`wait` vs `wait_full`.** The marginal-job partial-allocation rule flatters `wait`. As in Exp 63,
  `wait_full` is the mechanism metric and `wait` may not be read alone.
- **No LLM anywhere.** Both arms are deterministic, consistent with §4.1/§4.2 and with Exp 92. A
  positive result is an efficiency result for the deterministic core; it does **not** bear on the
  flexibility thesis, which lives in the hard-case suite.
- **Horizon *H* is a researcher degree of freedom.** Fix it in the pre-registration that this design
  would eventually feed, and report it.

## 7. Open questions to settle before any pre-registration

1. What *H* (in ticks), and justified how — job-duration quantile, or the trigger's own cadence?
2. Does predicted contention size the reserve only, or also break ties in the ranking? (§4 proposes
   budget-only, as the smaller claim.)
3. What trailing window does `forecast-lookahead` observe, and is a rate estimator enough, or is a
   learned model needed at all? A moving average that matches the oracle would be the best possible
   outcome and should be tried first.
4. Which operating point — Exp 63 runs `--pools 8 --seeds 32`; `--horizon` / `--slack-mult`
   (`trace_replay.py:1093,1098`) are the flags that set the deadline regime, and the choice must be
   inherited from whatever cell branch 2 lands on, not re-picked here.
