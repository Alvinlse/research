# PINS research progress — complete summary (Exp 1–33, June 6 – July 7, 2026)

> Condensed from `research_progress.md` (the full log, with per-experiment tables, caveats and
> reproduce blocks). This file is the track-level view: what each method was and what it achieved.

**The thesis system:** an HPC/GPU scheduler where *LLMs reason and explain, deterministic code
decides and guarantees*. Stage 1 predicts each job's resource demand with uncertainty; Stage 2
rations GPUs through a two-sided agent negotiation cleared by an auction, with an ILP
feasibility layer. One sentence: *LLMs reason about value under uncertainty; the auction decides
the allocation; the ILP guarantees feasibility.*

---

## Track 1 — Stage-1 static prediction: peak GPU memory (Exp 1–7)

**Method arc:** predict a CNN's peak training memory from its metadata, ground truth measured on
the real A100 (`torch.cuda.max_memory_allocated`). Four designs, each removing one more number
from the LLM: (1) raw LLM number → (2) LLM extracts facts, formula computes → (3) LLM does
free-text layer-walk reasoning → (4) LLM emits only per-layer **shapes**; deterministic code does
all arithmetic (`param_term + a·activation_raw + b`, leave-one-out calibrated).

**Achieved:** each step that moved a number out of the LLM cut error ~an order of magnitude:
MAE 283 GB → 3.9 → 2.6 → **0.04 GB (1.8% MAPE, ~40× better than the params heuristic)**. Robust
to fp16/bf16 (the precision-blind heuristic collapses instead), and generalizes to ResNets and
Transformers once a deterministic attention term (`layers·nhead·seq²`) closes the long-context
gap (Exp 7: MAE 0.28 GB across 3 families, where the heuristic's correlation goes *negative*).
Crucially, even 14b produced perfect shapes but couldn't sum consistently — **scaling the LLM was
not the fix; removing it from the arithmetic was.** The project's governing principle, proven
empirically first.

## Track 2 — Stage-1 dynamic forecasting + uncertainty (Exp 8, 16A)

**Method:** a small Transformer encoder on MIT Supercloud telemetry (lookback 30 → forecast 30
steps × 4 channels), predicting the *residual from persistence*; extended with a P10/P50/P90
pinball-loss head and split-conformal (CQR) calibration.

**Achieved:** beats persistence exactly where a forecaster should — the dynamic channels
(gpu_util MAE −21%) — while the residual anchor protects flat channels. Adding quantiles cost no
accuracy, and conformal calibration lifted interval coverage 0.67 → 0.75 (fixing gpu_mem
0.52 → 0.77), yielding a genuine per-job uncertainty signal (median 0.16, well spread) that
feeds Stage 2.

## Track 3 — Stage-1 on real traces: what the LLM can and cannot predict (Exp 19–21)

**Method:** take prediction to real data — Supercloud runtimes, Alibaba v2018 DAGs, minted GPU
labels via rank-matched CNNs executed on real DAG topologies.

**Achieved (mostly negative for the LLM, decisively):** per-model **retrieval quantiles beat the
14b LLM** on runtime (MAE 172 vs 236 min; LLM coverage 0.16 — badly overconfident) → LLM removed
from the numeric prediction path. DAG topology **substitutes for missing co-request features**
(MAE −68%, ρ +0.48 → +0.86) via GBT quantiles. Peak concurrent GPU is predicted by a one-line
heaviest-level rule to ~3% MAPE; co-location slowdown is k-determined. Together with Track 1:
**the LLM's measured value is judgment and justification, never calibrated numbers** —
deterministic rules and GBTs own the numeric layer.

## Track 4 — Stage-2 mechanism design: how to ration GPUs (Exp 9–13)

**Method:** pure-Python simulator; on one job stream, compare the PINS sealed-bid marginal
auction (`mechanism.clear`, uniform price, anti-thrashing gate) against value-blind baselines on
SLA-violation rate.

**Achieved — a negative result that shaped everything:** value-max per-round auctions **lose SLA
to greedy-FIFO** (pool 8: 74 vs 47%) because diminishing-returns bids *spread* GPUs thin and
re-pricing *reorders* every round; SLA rewards concentration + a stable order (Exp 9–10). The
fix (Exp 11): the **committed auction** — bid once on arrival, freeze priority, serialize
run-to-completion. It matches greedy on raw SLA and **halves prod-tier SLA violations** (pool 8:
23 vs 54%). An LLM then sets that frozen priority as an ordinal class with a one-line
justification (Exp 12) and *keeps* the win — the interpretable version matches the deterministic
one. Exp 13 exposed the mechanism's trust problem (Track 9).

## Track 5 — the two agents' levers + the guarantee layer (Exp 14–18, 26)

- **Supply agent, headroom reservation (Exp 14–15):** reserving idle GPUs for incoming prod jobs
  pays *only* against rigid (non-preemptable) incumbents at moderate contention (prodSLA
  26.8 → 18.8); a malleability-aware variant recovers the utilization cost, Pareto-dominating
  blind reservation, and the agent's value scales with the rigid fraction. First decision where
  model size had stakes: 3b over-reserves (its justification misreads "scarce"), 14b matches the
  hand-built oracle.
- **Demand agent, uncertainty-sized margin (Exp 16B–17):** jobs bid base + margin GPUs sized by
  their calibrated interval width; true work can spike within the uncertainty bound.
  **Uncertainty-sizing is insurance whose value grows with tail severity** (spike 2.0: prodSLA
  28.1 → 18.4), while a blanket margin is worse than none at mild tails — the *signal* is the
  value, not the headroom. The LLM hedge at 7b+ beats the deterministic margin in both regimes
  *once given a spike-risk signal* — again, the cure for an LLM blind spot was a missing signal,
  not scale.
- **ILP (Exp 18):** on a 1-D pool the ILP exactly ties the auction at ~150× the cost — worthless
  there — but on 2-D node placement, where the count-only auction strands up to 5.4 GPUs/round to
  fragmentation, the ILP plans count+node jointly: fragmentation → 0, util +7–12 pts. Locates
  precisely where the guarantee layer earns its keep.
- **Reflection (Exp 26, negative):** a 3b agent reflecting on outcome feedback enters an exact
  period-2 limit cycle and never converges — naive reflection does not substitute for scale; the
  deterministic protocol does (Track 6).

## Track 6 — integration: the locked pipeline (Exp 22–25)

**Method:** merge everything into one world (`two_sided_sim`): demand margins ⇄ supply reserve
negotiated by a bounded concession protocol (provable termination), against a no-LLM floor,
isolated agents, and the must-have **single-LLM-with-both-objectives** baseline; then compose the
full pipeline (`pipeline_sim`): negotiate → committed-auction rations → ILP places on nodes.

**Achieved:**
- **Exp 24 (after fixing a real bug — non-negotiable base demand polluting the margin table):**
  negotiation fallback 96/74/49% → **0%**; model size finally matters; and the headline *flipped*:
  **negotiated@14b beats or ties single-llm@14b at every pool.** The lone agent over-commits with
  no brake; the concession ladder *is* the brake.
- **Exp 25 (full pipeline, 2×2 ablation):** the ILP halves fragmentation loss inside the full
  pipeline; 14b bids *more aggressive* margins that backfire under count-only placement
  (nego+sticky worse than the floor) — and **the ILP rescues them** (6×8: prodSLA 7.5 → 2.3),
  pulling the pipeline back to the frontier. The guarantee layer is what makes LLM over-demand
  *safe* — the design hinge demonstrated end-to-end.
- **Exp 23 (affinity placement):** the LLM classifying task bottlenecks is a knife-edge — 3b is
  *worse than no hint*, 14b matches the oracle — but the ILP's hard constraints absorbed every
  bad hint (no infeasible placement ever).

## Track 7 — realism and statistical honesty (Exp 27–30)

**Method:** replace every synthetic input with real data, then replace anecdotes with paired
statistics: real Stage-1 predicted GPU caps at quarter-GPU quantum (Exp 27); full trace replay of
606k real Alibaba v2020 jobs with arrivals/durations/demand jointly real on one clock (Exp 28);
32-seed paired 95% CIs (Exp 29); Stage-1 *predicted* demand with its real errors in the
negotiation loop, with an oracle control on matched windows (Exp 30).

**Achieved — the thesis's load-bearing claims, with error bars:**
- **Prod-SLA protection is real:** negotiated − floor significant at pools 6/8 in every tier
  (−4 to −8 pts*) at *no measurable overall-SLA or slowdown cost*. ("Beats the floor on overall
  SLA" was exposed as 8-seed noise and dropped — the honesty pass mattered.)
- **The protocol substitutes for scale as *sufficiency*:** negotiated rule ≈ 3b ≈ 7b ≈ 14b, all
  statistically indistinguishable — the brake makes a small model as good as a big one — while
  the un-braked single-llm still needs scale and still loses at every size.
- **Prediction error costs everyone** (+4..+10 SLA pts* vs oracle), yet prod protection
  **survives it** at every pool, and at slack the negotiation measurably *cushions* the error
  (diff-of-diffs −2.7*/−4.7* vs the floor) — the Exp-16 "uncertainty is insurance" story
  reappearing at system level, unprompted.
- Constant regime lesson: every lever in the project (margins, reserve, hedges, cushioning)
  needs slack; at saturation nothing helps.

## Track 8 — uncertainty sizes the request (Exp 31 + addendum)

**Method:** agents request a chosen quantile of the Stage-1 [P10, P90] interval instead of the
point P50 (`--quantile`), then a per-job newsvendor rule (`prod-p90`: only prod jobs hedge). All
tiers seed-paired with the oracle.

**Achieved:** the request quantile is violently asymmetric — P10 is catastrophic (+5..+17 SLA
pts*), P90 nearly free — and the **P90 hedge recovers the entire prod-tier prediction-error
cost** (pred ≈ oracle on prodSLA everywhere), at a price in best-effort slowdown and hoarded
GPUs. The targeted `prod-p90` rule **keeps the full recovery at a collapsed price** (slowdown
−1..−2*, util −3..−7* vs uniform), and restores the negotiated reserve's significance — hedge and
reserve are *substitutes* when the hedge is blanket, *complements* when it's targeted. Thesis
line, now measured: *size the hedge from the interval, but only for the tier you protect; let the
negotiation insure the rest.*

## Track 9 — incentives: closing the oldest open problem (Exp 13 → 32 → 33)

**Method & achieved, in three steps:**
- **Exp 13 (the vulnerability):** the committed auction trusts declared classes; best-effort jobs
  lying 'critical' collapse prod protection to the greedy floor, and a flat per-job budget
  provably can't fix it (a cost on the class can't distinguish a liar from an honest declarer).
- **Exp 32 (the mechanism):** per-**user** budgets (jobs share one purse) + pay-your-own-claim
  pricing charged only on *contested served ticks*, insolvency demoting the whole portfolio.
  Measured as a best-response test (one deviating user, paired by seed): unpriced lying pays
  −14..−24* net; at the 120 budget the gain is **statistically zero at every pool**, because the
  lie's cost lands on the deviator's *own prod jobs* (+15*). The price: +3..+9 prodSLA pts of
  honest-world rationing — a monotone deterrence-vs-rationing frontier, which *is* the operator's
  fairness-policy menu. Boundaries stated honestly: spend level can't identify liars
  (honest-heavy users overlap), and no anonymous tariff can rescue an all-liars world. Two
  rejected designs recorded (uniform second-price socializes the lie; flat budget).
- **Exp 33 (the behavior):** an LLM playing the self-interested *user* — told only "maximize your
  own jobs" — **exploits the unpriced mechanism unprompted** (3b −12..−14*, 14b −5..−7* vs honest
  play; the smaller model games harder, 14b's alignment priors leak normative language), and
  **the tariff flips both to honesty-equivalence** (14b exactly zero at every pool), with 14b's
  justification explicitly citing the purse externality. The deterrence works through *legible
  rules*, which is the interpretability edge the thesis claims over RL.

---

## The through-lines (what the whole body of work shows)

1. **"LLM reasons, code decides" is not a slogan — it's the empirically optimal division at every
   layer:** shapes vs arithmetic (Exp 4), class vs weight (Exp 12), hedge level vs GPU count
   (Exp 17), bottleneck label vs placement (Exp 23), declaration vs clearing (Exp 32–33).
2. **The LLM never wins as a number predictor** (Exp 1, 19–21) and never needs to: its measured
   value is judgment over rich context + an auditable justification per decision.
3. **Deterministic structure substitutes for model scale:** the concession ladder makes 3b ≈ 14b
   (Exp 27/29); reflection does not (Exp 26); missing *signals* fix LLM blind spots, not bigger
   models (Exp 7, 17).
4. **The guarantee layers make LLM aggression safe:** ILP absorbs over-demand (Exp 25), the
   tariff absorbs self-interest (Exp 32–33), the interval hedge absorbs prediction error
   (Exp 31).
5. **Everything needs slack**, and honest statistics reshaped the claims twice (Exp 29 killed one
   headline; Exp 24/27 exposed two artifacts) — the surviving claims all carry 32-seed paired
   CIs.

**Still open:** NVLink/3-class placement (d), a slack-regime showcase run where negotiation
SLA-win and ILP placement show together (f), and the EASY-backfilling/DRL baseline scope decision
(k) — plus the thesis write-up itself, for which the full log is now essentially the skeleton.
