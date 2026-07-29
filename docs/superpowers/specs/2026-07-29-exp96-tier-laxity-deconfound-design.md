# Exp 96 — the tier/laxity de-confound (PRE-REGISTRATION)

**Date:** 2026-07-29  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b
**Status:** written **before any code**. No flag implemented, no pilot run, no data seen.

## 1. The question

Claim 2 of the state-of-claims table says *the reasoning layer's entire reproducible contribution is
production-tier protection, and it is captured by a single reserve scalar* (Exp 70, 71, 73).

The workload generator makes that claim unfalsifiable as stated. `pins/trace_replay.py:211-214`:

```python
urgency  = rng.uniform(0.6, 2.2)
slack    = max(1.15, min(2.4, 2.5 - 0.65 * urgency)) * slack_mult
deadline = arrival + int(round(work * slack))
tier     = "prod" if urgency >= 1.667 else "besteffort"
```

**Tier and deadline tightness are the same random variable.** `prod` ⟺ `urgency ≥ 1.667` ⟺
`slack ∈ [1.15, 1.42]`, so every prod job carries one of the tightest deadlines in its window and no
best-effort job ever does. "Protecting the prod tier" and "protecting the tightest deadlines" are
therefore not separable in any experiment run to date. In a real cluster, tier is an account
property and deadline tightness is a job property; they are not one number.

This experiment separates them and asks which one the mechanism was actually protecting.

## 2. The change

New flag `--decorrelate` (default OFF, tier suffix `+decorr`): draw `tier` **independently** of
`urgency`, with `P(prod) = 1/3` — the exact marginal the current rule produces, since
`P(urgency ≥ 1.667) = 0.533 / 1.6 = 1/3`. `slack`, `deadline` and the bid keep running off `urgency`
unchanged.

**The tier draw uses its own RNG stream** (`random.Random(f"tier-{seed}")`), never the shared one.
If it advanced the shared stream every window would change and the comparison against the correlated
world would be unpaired — the Exp 59 trap. With a separate stream the two worlds share
**byte-identical** jobs, arrivals, durations, caps and deadlines, and differ only in which jobs carry
the `prod` label. The diff-in-diff is then exact rather than statistical.

Marginals preserved, correlation broken, minimum diff. Every existing tier stays byte-identical with
the flag off.

## 3. New metric

`tight_sla` — deadline-violation rate over the **tightest-laxity tercile** of each window, laxity
measured as `(deadline − arrival) / work`. Computed alongside `sla`/`prod_sla` in
`two_sided_sim.py` (both `simulate` and `simulate_backfill`) and added to `METRICS`.

In the **correlated** world `tight_sla` should nearly duplicate `prod_sla`. That is the manipulation
check: it demonstrates the confound rather than asserting it. In the **decorrelated** world the two
separate, and `tight_sla` becomes the measurement that matters.

## 4. Arms

Exactly Exp 73's configuration, so the replication is faithful:

```
--referee --market --composed --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32
```

rows: `no-llm` (floor), `referee`, `negotiated`, `market`, `composed`. Both laws (`amdahl`, `sat`),
run with and without `--decorrelate`. n=32 paired seeds.

## 5. Hypotheses

- **H1 (primary).** `dprodSLA(composed − floor)` in the decorrelated world, paired by seed; and the
  **diff-in-diff** `[composed − floor]_decorr − [composed − floor]_correlated`, which the shared RNG
  stream makes an exact within-seed contrast.
- **H2 (decisive).** `d tight_sla(arm − floor)` for every arm, both worlds.
- **H3 (manipulation check).** Correlation between tier and laxity: deterministic in the correlated
  world, ≈ 0 in the decorrelated one. Plus `tight_sla ≈ prod_sla` in the correlated world only.

Two-sided throughout; no direction is pre-declared for H1 or H2. Significance is the house rule
(95% CI excludes 0, paired by seed), Holm across the vs-floor family as `trace_replay` already
reports.

**Predicted outcome: protection SURVIVES but SHRINKS.** The supply agent's input is
`bridge.reserve_ctx(contention, incoming_prod)` — literally a count of incoming *prod* jobs — so the
mechanism is structurally tier-driven and will still withhold headroom. But once prod jobs no longer
carry the tightest deadlines, that headroom should convert into fewer saved deadlines. `tight_sla` is
predicted flat everywhere, because nothing in the system keys on laxity at all.

## 6. Decision rule

1. **Protection undiminished** (diff-in-diff ≈ 0, prodSLA still significant) → claim 2 **strengthens**:
   tier is genuinely what is protected, and the correlation was incidental.
2. **Shrinks materially but stays significant** → claim 2 must be **restated** as partly
   laxity-driven, with both worlds' numbers reported side by side. *Predicted.*
3. **Vanishes** (prodSLA null in the decorrelated world) → claim 2 was tight-deadline protection
   mislabelled as tier protection. §4 of the paper needs rewriting and the headline weakens. This
   branch is pinned now precisely because it is the expensive one to admit later.
4. **`tight_sla` flat in every arm** → nothing in the system serves laxity, which motivates
   least-laxity grant ordering as the next experiment (`two_sided_sim.py:454` currently orders by
   frozen bid value). *Predicted, and independent of branches 1-3.*

No outcome changes claims 1, 4, 5 or 9. Branch 3 changes claim 2 and paper §4.

## 7. Threats to validity

- **Prod count variance.** An independent Bernoulli(1/3) draw has binomial variance where the old
  rule's did too (it was also a per-job draw), so windows with 2 or 9 prod jobs occur in both worlds.
  `prod_sla` over a small denominator is noisy; report the per-seed prod count alongside.
- **Tier still drives grant precedence.** `two_sided_sim.py:456-457` serves prod before best-effort
  structurally, independent of the bid. That is retained deliberately — it is the realistic part of
  the tier concept — so this experiment de-confounds the *deadline*, not the *precedence*.
- **Urgency still drives the bid.** Decorrelating tier leaves `Job.urgency` feeding `bid()` and the
  frozen grant priority. So prod jobs lose their implicit bid advantage in the new world. This is
  intended (tier precedence is structural, urgency is per-job) but must be stated: two things change
  for a prod job, its deadline tightness and its bid rank.
- **This tests the world, not the mechanism.** A shrinking effect does not mean the mechanism is
  broken; it means the old world flattered it.

## 8. Reproduce

To be filled at implementation. Analysis through the **`pins-analyst`** subagent against this
document.
