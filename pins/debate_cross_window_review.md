# Cross-window review of policy debate v1

**Completed:** 2026-09-15. All 24 training windows have a complete `policy_select` /
`policy_debate` pair (48 phase-one arm-runs), and the eight outcome-selected `policy_bo3` controls
declared in Amendment 1 are also complete. This is the pre-registered exploratory training-split
analysis. It makes no claim about the reserved validation or test windows.

## Result by completed window

The primary difference is debate minus single, in percentage points of deadline violations;
negative favours debate. `Fixed D` and `Fixed F` are diagnostic zero-LLM replays of the same frozen
world using adaptive sizing and, respectively, deadline or fairness ordering throughout. `Harm` is
a job that met its deadline under single but missed under debate; `save` is the reverse.

| Window | Load | Single | Debate v1 | Difference | Fixed D | Fixed F | Harm / save | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| d48h23 | 4.99 | 24.1 | 14.3 | -9.8 | 10.1 | 22.3 | 3 / 46 | Deadline ordering is decisively better; debate's deadline bias helps. |
| d111h22 | 3.76 | 26.9 | 12.7 | -14.2 | 8.7 | 27.8 | 4 / 68 | Deadline ordering is decisively better; debate tracks the right regime. |
| d22h6 | 3.53 | 31.5 | 32.0 | +0.5 | 32.4 | 30.5 | 102 / 96 | The fixed policies are close and fairness is slightly better; many offsetting job flips leave a near tie. |
| d118h14 | 3.01 | 25.8 | 22.6 | -3.2 | 21.9 | 25.4 | 0 / 9 | Deadline ordering is moderately better and debate causes no adverse deadline flips. |
| d223h11 | 2.90 | 28.1 | 34.8 | +6.7 | 33.8 | 22.9 | 105 / 60 | Fairness is decisively better, but debate stays deadline-heavy. The 105 harmed jobs are concentrated in four users; 90 arrived in hours 30--36, and their median additional wait under debate is 22.4 hours. |
| d188h6 | 2.74 | 4.8 | 7.7 | +2.9 | 5.9 | 5.0 | 51 / 10 | Fairness is slightly better; all 51 harmed jobs belong to one user, the clearest concentrated-burst failure. |
| d202h19 | 2.68 | 34.7 | 24.0 | -10.7 | 22.8 | 25.2 | 97 / 206 | Deadline ordering is better and the saved jobs outnumber harmed jobs by more than two to one. |
| d160h17 | 2.54 | 7.7 | 7.7 | 0.0 | 8.2 | 9.4 | 0 / 0 | Neither selector changes any job's deadline-status classification; this window is uninformative for the primary contrast. |
| d207h17 | 2.46 | 5.5 | 5.3 | -0.2 | 5.3 | 6.3 | 0 / 1 | Deadline ordering is slightly better for the primary metric; fairness has lower mean wait, showing the objective trade-off. |
| d102h8 | 1.88 | 8.5 | 9.5 | +1.0 | 9.5 | 8.5 | 7 / 3 | Debate tracks the inferior fixed-deadline result; all seven harmed jobs belong to one user. |
| d109h7 | 1.82 | 28.8 | 26.8 | -2.0 | 25.6 | 26.0 | 0 / 5 | The fixed orderings are close, but debate saves five jobs without harming one. |
| d190h0 | 1.78 | 10.6 | 10.0 | -0.6 | 10.0 | 11.7 | 0 / 2 | Debate matches fixed deadline and saves two jobs. |
| d185h6 | 1.69 | 5.1 | 4.7 | -0.4 | 4.7 | 4.7 | 0 / 1 | Fixed ordering is uninformative; debate saves one job through the remaining policy path. |
| d225h0 | 1.47 | 12.8 | 7.8 | -5.0 | 7.5 | 14.7 | 1 / 17 | Deadline ordering is decisively better despite a batch-dominated concentrated burst. |
| d58h19 | 1.35 | 13.0 | 13.0 | 0.0 | 13.0 | 13.0 | 0 / 0 | Every compared policy has the same primary result. |
| d209h23 | 1.32 | 5.2 | 6.8 | +1.6 | 6.8 | 7.1 | 5 / 0 | Debate loses even though fixed deadline is marginally better; four of five harmed jobs share a user, so ordering alone does not explain this loss. |
| d221h21 | 1.23 | 2.1 | 2.1 | 0.0 | 1.9 | 2.1 | 0 / 0 | No job-level deadline-status flips; the rounded fixed-policy difference is too small to matter here. |
| d211h16 | 1.19 | 8.5 | 8.5 | 0.0 | 8.5 | 9.0 | 0 / 0 | No primary-outcome flips despite a small fixed-deadline advantage. |
| d98h19 | 0.91 | 0.1 | 0.0 | -0.1 | 0.0 | 0.1 | 0 / 1 | Debate saves one job and matches fixed deadline. |
| d71h7 | 0.88 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0 / 0 | Every compared policy has the same primary result. |
| d218h4 | 0.62 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | Floor effect: no policy misses a deadline. |
| d116h2 | 0.58 | 29.6 | 5.6 | -24.0 | 4.8 | 24.8 | 0 / 30 | The largest gain occurs at low aggregate load: deadline ordering saves 30 jobs without harm, proving offered load alone is not a safe regime selector. |
| d26h21 | 0.49 | 6.2 | 6.2 | 0.0 | 6.2 | 6.2 | 0 / 0 | Every compared policy has the same primary result. |
| d63h11 | 0.32 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 / 0 | Floor effect: no policy misses a deadline. |

Across all 24 pairs, the mean difference is **-2.40 points** (paired-t 95% CI -5.11 to +0.31):
debate is better in 11, tied in eight, and worse in five. The two-sided exact paired sign-flip test
gives p=0.0739. In the eight pre-declared high-contention windows (offered load at least 2.5), the
mean difference is -3.48 points and p=0.2188. Holm-adjusted p-values are 0.1477 and 0.2188,
respectively. The exploratory sweep therefore does **not** reject either null.

## What actually predicts the win or loss

The debate does not discover a distinct policy each window. Across 1,576 debate epochs:

- Demand opened with `auction_deadline` in 1,576/1,576 epochs.
- Supply also opened with `auction_deadline` in 1,244/1,576 epochs (78.9%); its only alternative was
  fairness.
- The executed debate ordering was deadline in 1,326/1,576 epochs (84.1%), fairness in 230, wait in
  two, and 18 invalid decisions used the configured deadline/adaptive floor.
- By contrast, single selected fairness in 1,151/1,587 epochs (72.5%).
- The two complete openings agreed on both policy components in 894/1,576 epochs (56.7%), which v1
  ratified without Referee review. Another 53 disagreements converged after rebuttal; the Referee
  supplied 611 valid final decisions.

This makes the per-window outcome largely an ordering-regime test. Fixed deadline beats fixed
fairness in 14 windows, fairness wins four, and six tie. The four fairness windows are `d22h6`,
`d223h11`, `d188h6`, and `d102h8`; debate loses in all four. The large debate gains generally track
a fixed-deadline advantage, especially `d48h23`, `d111h22`, `d225h0`, and `d116h2`. The exception
`d209h23` shows that sizing and switching history can still overturn the fixed-ordering comparison.
Most of the eight reported ties are literal zero job-level flips, not balanced churn.

The model is not mainly failing JSON or feasibility. Only 18 of 1,576 epochs fell back (1.1%), and
there were no recorded inference errors. The primary defect is the decision contract:

1. Demand's v1 red line activates as soon as one waiting job has negative laxity. With the frozen
   synthetic deadline equal to submit time plus the requested-runtime estimate, any positive wait
   tends to make this value negative. The signal therefore does not distinguish broad deadline
   danger from an ordinary non-empty queue.
2. Supply's red line constrains only sizing (`greedy` under pressure), not ordering. It has no
   machine-enforced basis to reject deadline ordering when a same-user burst calls for fairness.
3. The packet reports queue summaries but not the distribution needed to distinguish these cases:
   top-user job/wait share and whether deadline pressure is spread across users.
4. Direct consensus ratification turns correlated advocate defaults into a final decision. The
   Referee is skipped precisely when an independent check would be most useful.
5. Ordering and sizing are coupled into one proposal. The diagnostic replays show that the large
   reversals are caused by ordering, so a useful Referee should be able to choose an advocate's
   ordering and the other advocate's sizing independently within a bounded set.

## Revised protocol: `policy_debate_v2`

V2 is a separately named exploratory arm. It does not edit v1 or enter its current sweep.

### Demand rule

- Ordering jurisdiction: `auction_deadline` or `auction_priority` only.
- Sizing jurisdiction: `as_requested` or `adaptive` only.
- Defend deadline only when deadline-budget consumption is severe across a substantial part of the
  queue and is distributed across users. One waiting or negative-laxity job is not sufficient.
- Defend priority only when production work is materially represented.
- State the specific service harm that the capacity/fairness view may miss, citing supplied state.
- In v2.2, a queue of at least eight jobs with at least 10% production work starts a persistent
  Demand episode. Demand's proposed ordering becomes binding until the backlog drops below eight;
  the Referee still decides sizing.

### Supply rule

- Ordering jurisdiction: `auction_fairness` or `auction_wait` only, so Supply cannot simply echo
  Demand's deadline default.
- Sizing jurisdiction: `adaptive` or `rcon` only.
- Defend fairness for concentrated top-user job/wait share, concentrated threatened jobs, or a
  policy held while the queue grows.
- Defend accumulated wait when age is broadly distributed and neither concentration nor deadline
  evidence dominates.
- Treat adaptive as the ordinary contention response; require an explicit resize-stability reason
  for `rcon`.
- In v2.2, outside a Demand episode, a queue of at least eight jobs where one user owns at least
  half the queued jobs or accumulated wait starts a persistent Supply episode. Supply must propose
  fairness, and that ordering remains binding until the backlog drops below eight. Releasing on
  drainage rather than instantaneous dilution preserves the objection after other users arrive.

### Referee rule

- Always adjudicate when both openings validate; opening agreement is not a final decision.
- Decide ordering before sizing and address both objections.
- Prefer deadline for severe, user-distributed deadline-budget consumption; fairness for
  concentrated user bursts or sustained queue growth; priority for material production pressure;
  accumulated wait when neither deadline nor concentration is decisive.
- Choose only from the code-generated cross-product of the two proposed orderings and two proposed
  sizing actions (at most four candidates). This permits component-wise judgment without allowing
  an invented policy or GPU count.
- During an active Demand or Supply episode, filter that cross-product to the active advocate's
  ordering. This is a machine-enforced objection, not a prose preference. If the Referee violates
  it, code executes the episode ordering with deterministic fallback sizing.
- If either advocate is invalid, do not accept the remaining side unopposed; execute the configured
  deterministic floor. The shared validator still computes all per-job sizing and enforces family
  membership and allocation bounds.

### Added online state

V2 supplies queue/arrival pressure, wait p90, deadline-budget-used p50/p90, threatened-job count,
the top user's share of threatened jobs, top-user job and wait shares, waiting-user count, declared
walltime and production shares, the last queue delta, queue-growth streak, and current-policy
tenure. Every field is available at scheduling time. V2 does not expose true runtime, future
arrivals, a window label, or realised deadline outcomes.

### Call budget

V1 used 5,145 calls over these 1,576 epochs because its path varies between two, four, and five calls.
V2 uses three calls for every epoch with two valid openings and two calls if an opening fails, so its
maximum on the same 1,576 epochs is 4,728 calls: 417 fewer (8.1%), not an increase.

## Budget-matched control

The outcome-selected `policy_bo3` control finished on all eight Amendment-1 windows. Debate beats
bo3 on every selected window, by 8.11 points on average, while using 1,932 versus 1,953 calls. This
rules out call count as the explanation *within these deliberately favourable windows*. It is not an
unbiased effect estimate: the windows were selected after observing where debate gained most.

## Two real-window development checks

These checks were selected after inspecting the nine-window training results. They are development
diagnostics, not a pre-registered comparison and not evidence that v2.2 generalises.

| Window | Single | Debate v1 | Full LLM revision run | Post-hoc episode replay | What it tested |
|---|---:|---:|---:|---:|---|
| d223h11 | 28.1 | 34.8 | v2.0: 33.8 | Supply episode: **22.5** | Batch-only majority-user burst; fairness must persist after the burst is diluted. |
| d118h14 | 25.8 | 22.6 | v2.1: 25.4 | Demand episode: **21.9** | The same concentration heuristic is unsafe when production is 29.1% at the trigger. |

The d223 v2.0 run used 318 calls over 106 epochs with no malformed replies or model errors. Demand
proposed deadline 106/106 times, Supply proposed fairness 106/106, but the Referee still selected
deadline 77 times. At the start of the critical burst it described 3.9% p90 deadline-budget use as
substantial even though `deadline_pressure_jobs` was zero, while one user accounted for 91.5% of
jobs and 96.7% of accumulated wait. The result improved v1 by only one point and still lost to
single. This falsified the assumption that a stronger prompt alone could protect Supply's objection.

The persistent Supply episode replay switched from deadline to fairness at the observed burst and
held it for the remaining backlog. It changed the job-level balance relative to single from v1's
105 harmed / 60 saved to 20 harmed / 58 saved, and scored 22.5%. A short fairness interval ending as
soon as instantaneous concentration diluted scored 33.8%, exactly the fixed-deadline result; episode
memory, not merely detecting the initial burst, caused the improvement.

The first persistent guard was deliberately challenged on d118, a window where deadline ordering
was known to help. Full v2.1 latched Supply for 42/58 epochs and selected fairness in all 58, scoring
25.4%: marginally better than single but worse than v1. Its trigger had 74.5% top-user job share but
also 29.1% production work, versus zero production at the d223 trigger. A symmetric replay that
latched Demand at that production episode scored 21.9%. That result motivates v2.2's role precedence:
material production activates Demand; only a batch-dominated majority-user burst activates Supply.

Both episode replays used the exact frozen real-window inputs and adaptive sizing, made zero LLM
calls, and changed only the scripted ordering interval. Because the switch rules were derived after
viewing these outcomes, their scores are optimistic mechanism checks. The final v2.2 code has contract
tests for trigger precedence, persistence, drainage release, constrained candidates, and deterministic
guard enforcement, but has **not** been measured as a full LLM arm. It requires a new frozen protocol
and unseen evaluation before any performance claim.

The completed sweep adds a stronger warning: the v2.2 majority-user trigger would also activate in
many deadline-favouring windows, including `d225h0` where fixed deadline beats fixed fairness by 7.2
points and `d116h2` where the gap is 20.0 points. Concentration is therefore useful evidence for a
Supply objection but not a safe machine veto by itself. The defensible next revision is to preserve
the disjoint openings and mandatory Referee, keep deadline as the deterministic failure floor, and
require Supply to establish sustained concentration *plus* deterioration under the current service
ordering before any binding escalation. Demand should establish material production or severe,
user-distributed deadline-budget consumption. Those escalation conditions must be frozen and tested
on unseen windows; the present v2.2 guard is an intentionally retained, rejected development
prototype rather than the recommended final policy.

## Provenance and isolation

- Observed selector outcomes and policy counts come from
  `runs/sweep_train_debate/rows.jsonl` and each world's saved `sw_*_policy_log.json` and
  `sw_*_job_statistics.csv`.
- Fixed D/F diagnostics come from each frozen world's saved `fx_auction_deadline` and
  `fx_auction_fairness` adaptive runs, summarised in `runs/sweep_train_debate/fixed_ref.jsonl`; they
  require no LLM calls and do not modify the sweep row file.
- Full v2.0/v2.1 LLM runs and their post-hoc episode replays used isolated copies under
  `runs/debate_v2_pilot`; SHA-256 checks confirmed that each copied `jobs.json` matched its source.
- The sweep completed 48/48 phase-one arm-runs and 8/8 Amendment-1 controls, then removed its cron
  entry. Its existing script invoked only the unchanged `policy_debate` arm; none of the v2 work
  modified the live row chain.
