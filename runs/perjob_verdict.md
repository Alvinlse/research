# Per-JOB policy-selection headroom: verdict (2026-09-12)

Input: `runs/labels_perjob` — 589 states x 15 policies x 155,331 per-job outcomes over 54 windows.
Analysis: `pins/perjob_analyse.py`, report `runs/perjob_report.json`. Costs are the balanced
reward's own job-level terms (sla10, bsd, wait), so `mean_j cost_j` IS the window reward minus its
two window-level terms. Lower is better. Safe menu (14 actions; the deadlocking pair excluded).

Four ceilings, each strictly above the last:

  fixed  (best single policy, pooled)       1.0599     wait 12625 s   sla10 7.44%   bsd 4.80
  state  (best policy per state)            0.9079     wait 11669 s   sla10 6.21%   bsd 4.04   -14.3%
  group  (best per state x job group)       0.7525     wait  9429 s   sla10 5.40%   bsd 3.51   -29.0%
  job    (best per state x job)             0.6148     wait  7143 s   sla10 4.98%   bsd 3.09   -42.0%

THE RESULT: of the job-level gain, a per-STATE choice already collects 34%. The other 66% is
per-job REDISTRIBUTION. Split each state's matrix as cost[p,j] = m[p] + d[p,j]; by construction
sum_j d[p,j] = 0, so the per-job minimum of d (-0.79, against a common-mode spread of only -0.37)
cannot be collected for all jobs at once. One job's earlier start is another's later one. The
window-level selector was not blind to this headroom; two thirds of it does not exist.

Group ceilings (cost): size 0.7989 > tier 0.8659 > queued 0.8860 > elastic 0.9079 (= state, and
inert BY CONSTRUCTION: every job in the holdout is elastic=1). Job SIZE is the only axis that
separates. Per-job regret of the best fixed policy is concentrated, not broad: mean 0.8150, 71% of
jobs hurt at all, top 10% hold 77% of it and the top 1% hold 21%.

The implementable slice — ordering is global by nature, sizing is applied per job at placement:

  fixed 1.0599 | state 0.9079 | STATIC RULE 0.9250 | per-group 0.8193 | per-job 0.8105
  wait: fixed 12625 s | state 11669 s | STATIC RULE 11514 s | per-group 10447 s | per-job 10289 s

The static rule is 3 buckets and no model: gpus=1 -> as_requested, 2-8 -> adaptive, >8 -> adaptive
(greedy only for already-queued >8 jobs under the cost objective, 37 such jobs in total), global
ordering least_laxity (fairness under the wait objective). Tier does not change the rule anywhere
and neither does queue state. It captures 89% of what the per-state ORACLE gets on cost and beats
that oracle outright on wait, at zero tokens, because it is heterogeneous and the oracle cannot be.

CAVEATS, all load-bearing:
1. NOT COMPOSABLE. Every rollout applies one policy to the whole cluster, so a heterogeneous
   schedule's real interactions are unpriced and the static-rule numbers above are assembled, not
   measured. They must be verified by an actual per-job-sizer run; if the run disagrees, this
   estimate is what is wrong.
2. IN-SAMPLE. The rule was fitted on the same 54 windows it is scored on. Needs a window-held-out
   refit before any paper number.
3. WRONG OBJECTIVE for the submission. This scores the balanced reward's job terms; the paper's
   primary metric is the deadline violation rate. The excluded window-level terms (fairness, resize
   churn) are 54% of the total cost and flip the per-state argmax on 49% of states, so do NOT quote
   "best fixed = fairness+adaptive" as a baseline — that ranking is an artifact of this cost.

---

# COMPOSABILITY TEST (2026-09-12): the ranking replicates, the effect size does not

Driver `pins/run_frozen.py`, rows `runs/frozen_sizer/rows.jsonl`: all 55 frozen windows, three cells,
paired, whole-window runs from t=0. `by_size` is the rule above, implemented as `--sizer by_size`.

```
cell                          n    wait_s  p90_wait   sla10   bsd    util   d wait vs as_requested
least_laxity+as_requested    55     12268     29343    7.46   4.85   0.798            +0
least_laxity+adaptive        55     12251     25482    6.45   4.18   0.819           -18
least_laxity+by_size         55     12141     25186    6.34   4.25   0.810          -127
```

**The labels' ranking is confirmed.** They predicted `by_size` < `adaptive` < `as_requested` on mean
wait and the simulator reproduces exactly that order, with `by_size` also lowest on p90 wait and on
sla10. The rule is the best of the three cells on every service metric, at zero tokens.

**The magnitude is not.** The labels predicted **-3083 s** against `as_requested`; the run measures
**-127 s**, about a twenty-fourth of it, with a median of 0 and 25 of 55 windows improving. Against
the better homogeneous comparator (`adaptive`) the rule is worth -110 s on a 12,268 s baseline, i.e.
under 1%, though consistently so (34/55 windows). So the direction is real and the size is small:
this is a tie-breaker, not the 89%-of-oracle result the assembled number suggested.

**Why the assembled number was too large.** Reading job j's wait out of the `as_requested` run when j
is small and out of the `adaptive` run when j is large takes each job's outcome from a schedule the
other jobs were not in, and the jobs that win under one sizing cannot all win alongside those that
win under the other. Mechanically the rule is also mostly a no-op: 11,115 of 13,281 scored jobs (84%)
ask for a single GPU, so `by_size` IS `as_requested` for most of the trace, which is why the median
window moves by exactly zero. (A first peek at 3 windows had the sign the other way; at n=3 one
heavy-load window dominated the mean. The 55-window set is the result.)

**What the labels DO predict at full size.** Homogeneous policies, at the earliest switch point where
the choice holds for ~95% of the window: labels put `adaptive` 207 s ahead of `as_requested`, the
simulator 18 s ahead -- same sign, both a tie. The sweep is quantitatively sound for the question it
was built for (one policy for the whole cluster) and only over-reads assemblies across rollouts.

**Status of the four ceilings above:** `fixed` and `state` are measured facts about homogeneous
policies and stand. `group` and `job` remain upper bounds, now known to be loose by more than an
order of magnitude where anyone tried to collect them. Per-job grain is real but thin.

No inferential test is reported. No pre-registration pins a test, scoring rule or sidedness for this
comparison, and `.claude/agents/pins-analyst.md` forbids inventing one; the paired distribution above
is descriptive. A confirmatory run needs the axes pinned first.
