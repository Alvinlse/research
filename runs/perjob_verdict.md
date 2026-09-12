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
