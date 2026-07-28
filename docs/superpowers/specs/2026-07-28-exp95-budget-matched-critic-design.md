# Exp 95 — the budget-matched critic (PRE-REGISTRATION)

**Date:** 2026-07-28  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b  **Suite:** r34
**Status:** written **before the arm exists**. `single-pkt-boNc` is not implemented, no pilot has
been run, no data seen. (Exp 93's pre-reg was written mid-run and said so; this one is not.)

## 1. Why this experiment exists

Exp 93 ran a reconstructed critic — one objection-only reviewer per text-bearing job — and scored
**34/81** against `single-pkt`'s **27/81**, one-sided exact McNemar **p=0.0461**. Under the two-test
Holm correction pinned in that pre-registration, 0.0461 > 0.05/2, so **H2 was not significant** and
the result stands as suggestive only.

It is also **budget-confounded**, which Exp 93 §5 flagged in advance: critic spends
`n_text_jobs + 1` calls per case (244 over the suite) against `single-pkt`'s 1 (81). Exp 93's own
follow-up rule was explicit — *"a win there is a **cost** result … the follow-up is a budget-matched
critic arm."* This is that arm.

Exp 88 already established that budget alone buys nothing **at debate's budget** (7×, p=0.250) and
Exp 89 confirmed it at n=81 (boN 29/81 vs debate 43/81, p=0.00066). That does not settle the
question at **critic's** budget: 3× is a different point on the curve, and the whole interest in
critic is that it might be the cheap structure.

## 2. The control arm

**`single-pkt-boNc`** — identical to the existing `single-pkt-boN` in every respect except `k`:

| arm | k (samples/case) | matched to |
|---|---|---|
| `single-pkt-boN` (exists, Exp 88/89) | `2·n_jobs + 3` | debate |
| **`single-pkt-boNc`** (new) | **`n_text_jobs + 1`** | **critic** |

`n_text_jobs + 1` is critic's per-case call count *exactly* — `_critic_signed` issues one `_ask`
per job with a non-empty note, and the referee call adds one. Matching is **per case, not in
total**, as in Exp 88.

Everything else is inherited unchanged and deliberately not re-litigated: same packet, same
`SYSTEM_PACKET_SINGLE`, same `_vote` (modal action-id set, ties toward fewer actions per packet
rule 2, then lexicographic), same temperature 0.8, same per-sample cache tags.

**The temperature confound is inherited and re-declared.** `correction._ask` pins temperature 0, so
best-of-N is unbuildable at k>1 without it. The control therefore differs from critic on
temperature as well as on structure. This is intrinsic to best-of-N, was declared in Exp 88 before
that run, and biases **in the control's favour** — the conservative direction here, since the
result that would most interest us is critic surviving.

## 3. Hypothesis

**H1 (one-sided).** `critic-pkt` > `single-pkt-boNc` on STRICT handled, POOLED n=81.

This is the **only** pre-registered test, so **no multiplicity correction applies** — unlike Exp 93,
where H2 died to Holm. That is a consequence of asking one question, not a relaxation of the
standard, and it is recorded here so the difference cannot later look opportunistic.

Secondary, reported always, never substituted for the primary:
- TOST equivalence at the project ±3-case margin (Exp 55/88 precedent), for the null read;
- `single-pkt-boNc` vs `single-pkt`, which isolates *pure sampling* at critic's budget;
- `critic-pkt` vs `single-pkt`, which must reproduce Exp 93's **b=10 c=3** or the harness is
  suspect;
- bare `handled` beside STRICT; r4-only n=50 as the blind stratum; CONTROLS n=17, never pooled.

## 4. Pinned analysis axes

Inherited verbatim from Exp 93 §4 / Exp 89: STRICT (handled AND feasible) headline, exact McNemar
on discordant pairs, one-sided for H1, α=0.05, POOLED n=81 headline, r4-only reported, controls
separate, all arms paired within case on an identical packet.

## 5. Power, declared in advance

At n=81 with critic ~34 and the control somewhere in 27–34, the discordant count will plausibly be
~10–20 pairs. That supports detecting a large effect and **not** a small one, and the ±3 TOST margin
will likely FAIL for the same reason it failed in Exp 93 (m too small). **Both an inconclusive
McNemar and a failed TOST are declared possible now**, so neither can afterwards be read as support
for either side. The decisive quantity in practice is the point estimate.

## 6. Decision rule

- **critic > boNc, p<0.05** → a structure cheaper than debate survives its budget control. Critic
  becomes a paper-worthy arm and the natural next question is critic-vs-debate at matched budget.
- **critic ≈ boNc** (either ns with a small D, or TOST PASS) → critic's 34/81 is **sampling, not
  objection**. Combined with Exp 93's H1 tie, the reading becomes: *on this suite, extra passes help
  and their content does not* — and the multi-agent story stays exactly where Exp 89 left it,
  sequential debate alone.
- **boNc > critic** → the objection channel actively costs something relative to plain resampling
  at the same spend; report it, do not bury it.

Because Exp 93's H2 was **not** established, no outcome here can retro-fit significance onto it.
This run can only characterise a suggestive lead — kill it, or promote it to a real claim.

## 7. Disclosures

1. `critic-pkt` is **re-run, not replayed**, though it is deterministic at temperature 0 — the same
   guard Exp 88 applied to debate. A mismatch against Exp 93's 34/81 is itself a finding and must be
   reported, not silently accepted.
2. `critic-pkt` remains a **reconstruction** of Exp 67's arm, whose original was never committed.
   Results attach to this implementation.
3. The suite is **authored**, not sampled from operations (paper §5).
4. Nothing about Exp 89 / claim 5 is under test here; this experiment can only add or remove a
   cheaper alternative.

## 8. Reproduce

```
PINS_RESULTS=pins/results_exp95_qwen2514b.json \
  .venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b --suite r34 \
    --arms market,single-pkt,single-pkt-boNc,critic-pkt
.venv/bin/python -m pins.exp95_analyse pins/results_exp95_qwen2514b.json
```
