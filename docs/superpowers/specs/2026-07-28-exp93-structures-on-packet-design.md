# Exp 93 — the untested structures, on the strong packet (PRE-REGISTRATION)

**Date:** 2026-07-28  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b  **Suite:** r34

> **Written mid-run, and that is disclosed.** The run was launched before this document existed —
> a process failure, not a design choice. At the time of writing, 7 of 98 cases had completed and
> **no result table had been produced**. One earlier disclosure below is material: a 4-case smoke
> test was run and seen (see §7). Everything pinned here is pinned before any primary result
> table exists.

## 1. Why this experiment exists

Exp 65–67 (2026-07-22) tested three reasoning structures — self-consistency, an argumentation
ablation, and a critic arm — and found every pairwise comparison null (min p=0.238 across ~30
tests). Those verdicts are **confounded**: all three ran one day before the decision packet
(`b77d3fa`, 2026-07-23), which took infeasible rulings from 11–16/31 to 0/31. They are therefore
*untested*, not refuted — the same trap that made Exp 79 read "parallel multi-agent is dead"
before the interface was repaired and debate reached 43/81 (Exp 89).

## 2. Scope correction made before launch

**`selfcons` is NOT re-run.** `single-pkt-boN` already *is* self-consistency: it samples `k` times
at temperature and takes the **modal** action set (`_vote`), which is majority-vote
self-consistency. It has a strong-packet result at n=81 (Exp 89: 29/81 vs debate 43/81,
p=0.0007). Re-implementing it under its old name would have duplicated an existing finding. This
is recorded because "we re-ran all three" would otherwise be the natural reading.

Two arms are therefore new:

- **`debate-noarg-pkt`** — the Exp 66 argumentation ablation on the packet. Full gather+debate,
  then `evidence` is stripped from proposals before `build_packet`. The referee sees WHO proposed
  WHAT but not WHY. Isolates whether debate's win is argument *content* or merely the second pass.
- **`critic-pkt`** — Exp 67's critic. The original implementation was never committed (only its
  result JSON survives); this is a reconstruction from those transcripts, which carry a `problems`
  list and no deltas. One objection-only reviewer per text-bearing job; objections reach the
  referee via `reviewer_proposals`.

## 3. Hypotheses

- **H1 (argument content).** `debate-noarg-pkt` < `debate-pkt` on strict handled. If stripping the
  reasons costs nothing, debate's win is structural (a second pass), not argumentative.
- **H2 (critic).** `critic-pkt` > `single-pkt` on strict handled. An objection-only reviewer is
  the cheapest structure that could carry debate's benefit.

Both are **secondary** to the confirmed Exp 89 result; neither can overturn it. This experiment
can only add structures, never subtract from `debate-pkt`'s standing.

## 4. Pinned analysis axes

Inherited unchanged from the Exp 89 pre-registration
(`2026-07-24-exp89-round3-extension-design.md`), which is the covering pre-reg for this harness:

| axis | value |
|---|---|
| scoring | **STRICT** — handled AND feasible (over-award penalised) |
| test | exact McNemar on discordant pairs |
| sidedness | **one-sided** for H1 and H2, in the directions stated in §3 |
| alpha | 0.05 |
| strata | POOLED primary n=81 headline; r4-only n=50 reported as the blind batch |
| controls | n=17, never pooled with primary; reported for specificity cost |
| pairing | within case, all arms on an identical packet |

**Nothing new is invented.** If a comparison this document does not name is wanted later, it is
exploratory and must be labelled so.

## 5. The budget confound — pinned in advance

Call counts differ by construction and are **not** matched:

| arm | calls/case (approx) |
|---|---|
| `single-pkt` | 1 |
| `critic-pkt` | n_text_jobs + 1 (~3) |
| `debate-pkt`, `debate-noarg-pkt` | 2·n_jobs + 3 (~7) |

Exp 88 established that budget alone buys nothing on this suite (best-of-N at 7× the calls:
p=0.250 at n=31, p=0.250 at n=81). The interpretation rule, fixed now:

- **`critic-pkt` > `single-pkt`** counts as a structure result *only* alongside its call count,
  reported in the same table. It is cheaper than debate, so a win there is a **cost** result and
  must be framed as such, not as "critic beats debate".
- **`critic-pkt` vs `debate-pkt`** is reported but is **confounded by budget** and may not be
  claimed as a structure comparison in either direction. If critic ties or beats debate at ~40%
  of the calls, the honest claim is efficiency, and the follow-up is a budget-matched critic arm.
- **`debate-noarg-pkt` vs `debate-pkt`** is the one clean contrast here: identical call counts,
  identical pipeline, differing only in whether `evidence` survives into the packet.

## 6. Decision rule

- H1 rejected (noarg ≈ debate, TOST or ns with a small point estimate) → debate's benefit is the
  second pass, not the arguments. That **simplifies** the architecture: strip the justifications
  and keep the structure.
- H1 supported (noarg < debate\*) → argument content is load-bearing; the transcript is not
  decoration, which strengthens the interpretability claim in §5 of the paper.
- H2 supported → a cheaper structure than debate exists; schedule a budget-matched critic arm
  before any paper claim.
- Both null → the Exp 65–67 verdicts survive the interface fix for these two arms, and the
  multi-agent story stays exactly where Exp 89 left it: sequential debate wins, nothing else does.

Any outcome is publishable; none changes §4.4, which rests on Exp 89.

## 7. Disclosures

1. **This document was written after launch**, at 7/98 cases, with no result table produced.
2. **A 4-case smoke test was run and its output seen**, to verify the arms execute:
   `market 0/4, single-pkt 1/4, debate-pkt 1/4, debate-noarg-pkt 1/4, critic-pkt 2/4`. Those four
   are round-3 primary cases, so ~5% of the primary set has been observed. At n=4 no comparison is
   interpretable, but the peek is recorded rather than discarded.
3. `critic-pkt` is a **reconstruction**, not the Exp 67 original, which is unrecoverable. Results
   attach to this implementation and must be described that way.
4. The suite is **authored**, not sampled from operations (paper §5). This limits transfer of the
   effect size to a real workload; it does not bias between arms, which all read identical text.

## 8. Reproduce

```
PINS_RESULTS=pins/results_exp93_qwen2514b.json \
  .venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b --suite r34 \
    --arms market,single-pkt,debate-pkt,debate-noarg-pkt,critic-pkt
.venv/bin/python -m pins.exp89_analyse pins/results_exp93_qwen2514b.json
```

Harness: commit `9e5b223`. Analysis to be run through the `pins-analyst` subagent, which is the
gate this experiment initially bypassed.
