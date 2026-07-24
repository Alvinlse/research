# Exp 89 — Powered confirmatory run for the debate-structure claim

**Status:** PRE-REGISTRATION. Written and committed BEFORE any new case is authored and before
any LLM/debate/boN arm is run on the new cases. Nothing below was chosen after seeing a result
on the new batch.

**Date:** 2026-07-24  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b

## 1. Why this experiment exists

Exp 88 is the only budget-controlled test of the project's one surviving pro-LLM result. On the
round-3 PRIMARY 31 (STRICT, qwen2.5:14b):

    market 0/31  ->  single-pkt 9/31  ->  single-pkt-boN 10/31  ->  debate-pkt 14/31

`single-pkt-boN` is the single LLM given debate's full budget (K = 2N+3 samples/case, majority
vote, 217 calls to debate's 217). It reached 10/31 — flat vs single at 31 calls (b=1 c=1... =1,
p=1.0). Debate reached 14/31 (vs boN: b=5 c=1, one-sided McNemar p≈0.11, two-sided 0.22).

**The point estimate says debate's edge is STRUCTURE, not budget.** But at n=31 the test is
underpowered by design (Exp 88 was pre-declared a screening run): 6 discordant pairs cannot reach
α=0.05. Exp 88's own decision rule said an outcome in the 9-10 bucket "→ structure survives; a
powered confirmatory run (more CASES, not more models) is worth the authoring cost." This is that
run.

## 2. Hypotheses (declared in advance)

- **H1 (the claim):** `debate-pkt` handles more PRIMARY cases than `single-pkt-boN`. The rebuttal
  round buys something more samples at the same budget cannot.
- **H0:** `debate-pkt` ≈ `single-pkt-boN`. Debate is sampling, not argument; the multi-agent line
  closes on evidence.

**Primary test:** one-sided exact McNemar on the per-case handled indicator, `debate-pkt` vs
`single-pkt-boN`, α = 0.05, on the POOLED PRIMARY set (round-3's 31 + round-4's 50 = **81**).

**Companion confirmatory test (the clean number):** the identical test on the **new round-4
PRIMARY only (n=50)**. Reported alongside the pooled result from the same single run. If pooled
and new-batch agree, the claim is robust; if they diverge, that is itself the finding.

**Secondary (reported, not the claim):**
- `debate-pkt` vs `single-pkt` (the non-budget-matched arm) — the raw structure+budget effect.
- handled vs `ilp` / `rule` / `market` — the tail claim itself (rigid floor must stay ≈0).
- over-award count per arm (the LLM arms' known failure mode).
- citation rate of `must_cite` (faithfulness).
- CONTROLS handled-rate — the boundary condition (see §5).

## 3. Power

Simulation under Exp 88's point estimate (discordant rate 0.194, conditional split 0.833):

| new N | pooled n | pooled power | new-batch-only power |
|------|---------|-------------|---------------------|
| 50   | 81      | ~0.83       | ~0.57               |

Chosen: **author 50 new PRIMARY cases.** Pooled power ~0.83 at the point estimate is the target.
Honest caveats, declared now:
- The 5:1 split is estimated from 6 pairs. If the true effect is weaker (split 0.75, rate 0.16),
  pooled power falls to ~0.47 — the run can come back INCONCLUSIVE, and that is a legitimate,
  pre-declared outcome, not a failure to be re-rolled.
- The new-batch-only number stays ~0.57 and is reported as SUPPORTING, not decisive.

## 4. The authoring rule (round-3's rule, sharpened)

Every round-4 PRIMARY case must satisfy BOTH:

1. **Text-dependence (inherited from round-3).** Deleting the exception sentence leaves a scene
   whose numbers imply a DIFFERENT, perfectly reasonable answer. This makes the no-text arm a
   real ablation, not a harder version of the same question. Cases where the prose merely
   restates the numbers are rejected.

2. **Systematic-first-reading (new, the mechanistic target).** The case is built so a single pass
   is pulled toward a specific, *predictable, consistent* wrong reading — not a random error. The
   distinction is load-bearing for the test:
   - **best-of-N fixes STOCHASTIC error** — it majority-votes away noise. It CANNOT fix a
     systematic first-reading error, because all N samples make the same mistake and the majority
     vote confirms it.
   - **a rebuttal round can fix SYSTEMATIC error** — the second perspective challenges the
     reading a single pass locks onto.
   So a case engineered this way is exactly where structure should beat matched-budget sampling.
   This sharpens the thesis to: *structure helps where a first reading systematically misleads.*

   Concretely, a good round-4 case has a salient surface cue that points one way and a subtler
   governing fact (in the prose) that points the other; the surface cue is what a single reading
   grabs and what N samples will agree on.

**Composition (proportional to round-3 PRIMARY, to stay blind to which categories won):**
nl_policy 13, unmodeled 13, corrupt 8, contradiction 6, ambiguous 5, infeasible 5 = **50**.
Plus **8 new CONTROLS** (placebo/confirm) so the boundary check keeps pace; pooled with round-3's
9 gives 17.

## 5. Controls / boundary condition (declared in advance)

CONTROLS must be TEXT-INDEPENDENT: prose that is decorative, expired, or confirms the numeric
default. The text effect on controls must be ≈0. If an arm "wins" the controls too, the effect is
a response to text VOLUME, not CONTENT, and H1 is not supported however the primary test comes
out. Placebo keyword-traps (a trigger word inside a note that negates it) and confirm cases (the
prose confirms the default, or asks to break a rule and be refused) are included as in round-3.

## 6. Pre-registration discipline (the safeguards)

- **Blind authoring.** The author does NOT open Exp 88's per-case results (`results_exp88_*.json`)
  to see which specific cases or categories debate beat boN on. Only the aggregate (14 vs 10) and
  the authoring rule guide the work. This is what the proportional composition protects.
- **Rigid-only iteration.** Each case is iterated against the RIGID arms only — `ilp` (fixed
  static objective in `hardcase_eval.py`, never tuned per case) and `_rule_referee`. Acceptance
  criterion for a PRIMARY case: **both rigid arms FAIL the predicate** (do not do the defensible
  thing). Acceptance for a CONTROL: **both rigid arms PASS** (the default is defensible). No LLM,
  debate, or boN arm is ever run during authoring.
- **Full pre-registration per case.** Each `HardCase` ships `predicate` (the defensible answer),
  `rationale`, `expect` (the pre-registered rigid-arm failure), `must_cite`.
- **This document is committed before authoring begins.**

## 7. The run

Reuse `exp88_budget_control.py` + `exp88_analyse.py`, extended to load the pooled r3+r4 suite and
print the two strata (pooled n=81, new-batch n=50). Config unchanged from Exp 88: qwen2.5:14b,
boN temperature 0.8 (the declared best-of-N confound, biasing toward the control), K = 2N+3,
majority vote on `action_ids` with ties → fewer actions then lexicographic. Detached run on the
login node (one sweep, no mid-run edits — the CPU-reaper rule). Results JSON written at the end.

Harness reproduction check retained: `debate-pkt` vs `single-pkt` on the round-3 31 must still
give b=6 c=1 to match Exp 83; a mismatch is a finding, reported not accepted.

## 8. Decision rule (written down beforehand)

- **Pooled p < 0.05 AND new-batch direction agrees (debate > boN):** structure beats matched
  budget on evidence. The multi-agent sub-claim is confirmed; it earns its place in the paper as
  a result, not a point estimate.
- **Pooled p ≥ 0.05 but direction holds:** INCONCLUSIVE — consistent with structure, underpowered.
  Report the point estimate and the achieved power; do not overclaim.
- **debate-pkt ≈ boN (D small, both directions):** structure is sampling. The multi-agent line
  closes; "spend more inference on hard cases" survives as the cheaper mechanism.
- **Controls show a text effect in any arm:** the primary result is discounted as text-volume, not
  content, regardless of the primary p-value.

**The referee-flexibility thesis is untouched either way** — it rests on market 0/31 → LLM
9-14/31 with zero harm on controls. Exp 89 can only strengthen or close the multi-agent sub-claim.
