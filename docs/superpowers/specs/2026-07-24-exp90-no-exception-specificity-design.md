# Exp 90 — No-exception specificity: does the packet make the LLM meddle on ordinary jobs?

**Pre-registered 2026-07-24. Committed BEFORE the run.** Mirror-image of the hard-case suite:
that suite measures *recall* (does the LLM catch a real text exception); this measures
*precision / specificity* (does the LLM stay silent when there is nothing to catch), on
**real v2020 trace jobs with no authored text at all.**

## 1. Motivation

The hard-case suite (Exp 79–89) is built from ~80 text-exception scenes I authored. The fair
criticism: small n, self-authored, and it only ever asks whether the LLM *acts*. It never asks
the dual question — on an **ordinary** job with no exception, does the machinery correctly do
**nothing**? A scheduler that catches every exception but also meddles on normal jobs is not
safe. This experiment supplies the missing half using jobs I did not write.

## 2. The wrinkle that shaped the design (a real finding, recorded)

The naked single arm (`pins/correction.py`) is **text-triggered**: "a job with no free-text
note never triggers." So on a no-text scene it emits **zero** suggestions → 0 false → identical
to the floor. It cannot lose this test.

Therefore the risk of a false suggestion lives **entirely in the packet arms**: `build_packet`
hands the model a code-enumerated action menu *regardless of text*, which is exactly what lets
an LLM act with no textual reason. The live question is thus **not** "debate vs the naked
single" (the naked single is floor-safe by construction) but:

> The packet makes an LLM willing to act without a textual reason. On ordinary no-exception
> jobs, does that make the packet arms **meddle** (fire above the floor)? And does the **debate
> rebuttal** talk `single-pkt` back down toward silence?

## 3. Hypotheses (pre-registered)

- **Primary H1 (one-sided):** on the `fired` indicator, `debate-pkt` fires on **fewer** scenes
  than `single-pkt`. Test: exact McNemar, one-sided in debate's favour; two-sided also reported.
- **H0:** `debate-pkt ≡ single-pkt` on fire rate (packet meddling, if any, is not fixed by the
  rebuttal).
- **Verification claim:** `single-no-pkt` fires ≈ 0 (text-gating is free safety). A non-zero
  rate is a finding to report, not to hide.

## 4. Case construction — `pins/no_exception_scenes.py` (NEW, the only real new logic)

Build `case`-shaped objects from real jobs so the **existing** `run_case` runs unmodified.

Pre-registered rule, seed-controlled:
1. Draw `J ∈ {3,4,5}` jobs uniformly from `trace_replay.load_trace()` (v2020).
2. Each job's demand `d_i` = its trace quanta, clipped to `CAP_CLIP=8`, floored at 1.
3. `base_gpus = d_i`, `requested_margin_gpus = 0` (real trace jobs are inelastic; they ask for
   exactly their quanta), `tier = None`, `deadline = None`, `justification = ""`. Job ids are
   short synthetic labels (`r00…`) for packet legibility; the real trace name is kept in
   scene metadata only.
4. `free_gpus = round(sum(base_i) * (1 + slack))`, `slack` sampled uniform in **[0.15, 0.6]** —
   genuine idle headroom that **no job requested**. This is the meddling surface: a well-behaved
   scheduler leaves it idle; a meddling arm hands it out or reshuffles.
5. Clear the auction via the existing `build_anchor` → `floors, alloc, ranking`.
6. **Non-vacuity gate:** with `max_delta = 6` (the Exp 88 default, held fixed across arms), keep
   the scene only if the packet's `candidate_actions` menu is non-empty (there exists at least one
   legal grant / hold action). A scene where silence is the *only* option is discarded before
   scoring — otherwise the specificity test is trivially passed.

Because there is no text and no job requested more than its base, **the correct action is to
accept the auction** (`do nothing`). Any override is unmotivated by construction. This is the
project design rule made into a scorer: absent a real signal, *code decides*, so an LLM override
is presumptively a false suggestion.

`no_text=True` is passed throughout, reusing the already-wired ablation in `_jobs_of`.

## 5. Arms (paired per scene, all temperature 0)

| arm | source | expectation |
|---|---|---|
| `market` (floor) | `run_case` market path | 0 fired, by construction |
| `single-no-pkt` | `correction.py` free-text single, wired as a thin arm | ≈ 0 (text-gated) — verifies §3 |
| `single-pkt` | existing `run_case` | **at risk** |
| `debate-pkt` | existing `run_case` | primary contrast vs `single-pkt` |

`single-pkt`, `debate-pkt`, `market` need **no new arm code** — reuse `run_case`. `single-no-pkt`
is a thin wrapper calling `correction.py` on the same jobs; expected trivially silent, kept as the
text-gating verification.

## 6. Metric & scoring (reuse what `run_case` already emits)

- **False suggestion = `meta["fired"]`** = `bool(changes) or bool(hold_free)`. Already computed.
- **Headline:** per-arm false rate over N scenes, against the floor's 0.
- **Primary test:** paired exact McNemar on `fired`, `debate-pkt` vs `single-pkt`, one-sided
  H1 = debate fires less. Reuse `mcnemar` helpers from `pins/exp88_analyse.py`.
- **Secondary (harmful-only):** of the fired scenes, how many are `meta["rejected"]`
  (infeasible / breaks a floor) or reduce useful utilisation vs the anchor. A high harmful
  fraction upgrades "meddling" to "damaging".

## 7. Decision rule

- `single-pkt` ≫ 0 **and** `debate-pkt` ≈ 0, one-sided p<0.05 → **debate restores floor-silence**
  the packet costs. Strong, and relocates the user's intuition to where the risk actually is.
- both ≈ 0 → packet does not meddle on ordinary jobs; machinery is safe on real workloads,
  debate adds nothing *here*. Still a clean specificity pass that uses zero authored text.
- `debate-pkt` ≈ `single-pkt` ≫ 0 → packet-debate meddles on ordinary jobs — a real cost to set
  against its hard-case wins; report honestly.
- `single-no-pkt` > 0 → text-gating leaks; report.

## 8. Size, cost, runtime

- **N = 200 scenes** pre-registered (report observed after the non-vacuity gate). Powers the
  McNemar at ~0.8 if the fire-rate gap is ≥ ~10 points; report achieved discordant count.
- `debate-pkt` = `2J+3` calls/scene (~11 at J=4); temp 0 ⇒ cached & deterministic. Rough order:
  ~2k debate calls + ~200 single-pkt + single-no-pkt. Runtime dominated by debate; expect it in
  the same ballpark as one Exp 88 run. One background sweep only (login-node CPU reaper).

## 9. Driver & analysis — `pins/exp90_specificity.py` (NEW)

Loop scenes × arms, score `fired` (+ harmful), dump
`pins/results_exp90_qwen2514b.json` (same shape as Exp 88 so the analyser reuses `counts` /
`discordant` / `mcnemar_*`), print per-arm false rate + the McNemar block + the harmful slice.
Model `qwen2.5:14b` (the packet's capability tier; the 7b packet collapse is out of scope).

## 10. Reuse vs new

- **Reused unchanged:** `load_trace`, `build_anchor`, `build_packet`, `run_case`, `_jobs_of`
  (`no_text`), `apply_signed`/`check_allocation` feasibility, `exp88_analyse` McNemar helpers.
- **New:** `no_exception_scenes.py` (scene sampler + non-vacuity gate), `exp90_specificity.py`
  (driver + analysis), a thin `single-no-pkt` arm wrapper.

## 11. Reproduce

```bash
cd Research
.venv/bin/python -m pins.no_exception_scenes --n 200 --seed 0   # sanity: dump scene stats
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6 | tee pins/exp90_specificity_14b.log
```

## 12. Relation to the thesis

Recall = hard-case suite (does it catch real exceptions). Precision = this. Reported together
they are a sensitivity/specificity pair, and this half is immune to the "self-authored, small n"
criticism because the jobs come straight from v2020 and carry no text.
