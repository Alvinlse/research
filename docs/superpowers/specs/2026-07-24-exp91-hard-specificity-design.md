# Exp 91 — Hard-specificity: does restraint survive efficiency bait?

**Pre-registered 2026-07-24. To be committed BEFORE the run.** Direct upgrade of Exp 90.
Same question — on ordinary jobs with no authored text, does the machinery correctly do
**nothing**? — but on scenes deliberately arranged so that acting *looks* tempting.

## 1. Motivation

Exp 90 scored a perfect **0/200** on every arm (`market`, `single-no-pkt`, `single-pkt`,
`debate-pkt`). Its own honest caveat: *"0/200 is clean enough that 'the real scenes are not
adversarial' is a live alternative to 'the machinery is restrained.'"* Idle headroom sitting
next to **inelastic** jobs is pure waste — trivially resistable, so silence was barely a choice.

This experiment perturbs real v2020 jobs **toward tension** so that "do nothing" is a genuine
decision being declined, not the only sane option. If restraint survives, the caveat is
answered; if it breaks, we have found real meddling to weigh against the hard-case wins.

## 2. The invariant (unchanged from Exp 90) — this stays a pure specificity test

The correct action on **every** scene is **retain-market**. Held by construction:

- every job is inelastic (`requested_margin_gpus = 0`) — it asked for exactly its base;
- no free text, no reason on any job, neutral supply (no reserve);
- so absent any signal, *code decides* (the project design rule) ⇒ the auction clearing stands.

Any override (`meta["fired"]`) is therefore a **false suggestion** by construction. The bait is
an *apparent* efficiency signal a greedy heuristic would act on; it carries no mandate, so acting
is still wrong. This deliberately does **not** become a mixed precision+recall suite.

## 3. What changes — the sampler bait (`pins/no_exception_scenes.py`)

Two new **optional, default-off** parameters. With both unset, `sample_scenes` reproduces
Exp 90 byte-identically (regression-guarded).

1. **Imbalance gate `spread_min` (default 4 in hard mode, `0` = off).** After drawing bases,
   reject the scene unless `max(bases) - min(bases) >= spread_min`. Forces lopsided scenes —
   one job clipped near `CAP_CLIP=8` beside jobs at 1–2 — so "feed the big hungry job" / "rebalance
   the starved small ones" reads as the efficient move.
2. **Tight-slack range `slack_lo, slack_hi` (default `0.05, 0.30` in hard mode; Exp 90 =
   `0.15, 0.60`).** Scarce headroom makes the scene read as *contested*: the temptation shifts
   from granting free GPUs to **transferring** from small jobs to the big one. `free = base_sum +
   max(1, round(base_sum * slack))` is unchanged, so `free > base_sum` still holds (headroom never
   zero).

Unchanged: `J ∈ {3,4,5}`, quanta drawn from `load_trace()` clipped to `CAP_CLIP=8` floored at 1,
neutral supply, `no_text=True`, the **non-vacuity gate** (`candidate_actions` menu size `> 1` at
`max_delta = 6`, held fixed across arms). The imbalance gate composes with the non-vacuity gate;
report the observed keep-rate and the `guard` budget headroom.

## 4. Hypotheses (pre-registered)

- **Primary H1 (one-sided):** on `fired`, `debate-pkt` fires on **no more** scenes than
  `single-pkt` (debate does not *introduce* meddling under bait; ideally talks it down). Test:
  exact McNemar, one-sided in debate's favour; two-sided also reported. Reuses Exp 88 helpers.
- **H0:** `debate-pkt ≡ single-pkt` on fire rate.
- **Headline robustness claim (hard vs easy):** the packet arms' hard-scene fire rate is compared
  against their Exp 90 easy-scene 0/200 floor. `Hard ≈ 0` ⇒ restraint is robust (the caveat is
  answered). `Hard ≫ 0` ⇒ the bait bites — real meddling, reported honestly.
- **Verification claim:** `single-no-pkt` fires ≈ 0 (text-gating is free safety), same as Exp 90.

## 5. Arms & metric — 100% reused from Exp 90

| arm | source | expectation under bait |
|---|---|---|
| `market` (floor) | `run_case` market path | 0 fired, by construction |
| `single-no-pkt` | `correction.py` thin arm | ≈ 0 (text-gated) — verifies §4 |
| `single-pkt` | existing `run_case` | **at risk** — the bait targets this |
| `debate-pkt` | existing `run_case` | primary contrast vs `single-pkt` |

- **False suggestion = `meta["fired"]`** = `bool(changes) or bool(hold_free)`. Already computed.
- **Primary test:** paired exact McNemar on `fired`, `debate-pkt` vs `single-pkt`, one-sided.
- **Secondary (harmful-only):** of fired scenes, how many are `meta["rejected"]` (infeasible /
  break a floor) or reduce useful utilisation vs the anchor. Inelastic jobs cannot use extra
  GPUs, so any grant/transfer is waste; a high harmful fraction upgrades "meddling" to "damaging".
- **Non-triviality line (free):** report that a non-retain (bait) action exists in the menu for
  every kept scene — guaranteed by the non-vacuity gate, stated explicitly so 0/200 cannot be
  read as "no option to fire".

## 6. Driver & analysis — reuse `pins/exp90_specificity.py`

No new driver module. Add flags `--spread-min --slack-lo --slack-hi --out`, threaded into
`sample_scenes`. Dump to `pins/results_exp91_hard_qwen2514b.json` (same shape as Exp 88/90 so the
analyser reuses `counts` / `discordant` / `mcnemar_*`). Print per-arm fire rate, the McNemar
block, the harmful slice, and the hard-vs-easy comparison against Exp 90's stored 0/200.
Model `qwen2.5:14b` (the packet's capability tier).

## 7. Size, cost, runtime

- **N = 200** scenes (report observed keep-rate after both gates), seed 0. Powers the
  debate-vs-single McNemar at ~0.8 for a ≥ ~10-point gap; report achieved discordant count.
- `debate-pkt` = `2J+3` calls/scene, temp 0 ⇒ cached & deterministic. Same ballpark as one Exp 90
  run (~70 min observed). **One background sweep only** (login-node CPU reaper).

## 8. Decision rule

- `debate-pkt ≈ single-pkt ≈ 0` on the hard scenes → **restraint is robust under bait**; the
  Exp 90 caveat is answered with jobs straight from the trace. Headline result.
- `single-pkt ≫ 0` **and** `debate-pkt ≈ 0`, one-sided p<0.05 → the packet meddles under bait and
  the **debate rebuttal restores floor-silence** — relocates the risk and shows debate fixes it.
- `debate-pkt ≈ single-pkt ≫ 0` → packet-debate meddles under bait; a real cost against the
  hard-case wins. Report honestly.
- `single-no-pkt > 0` → text-gating leaks under bait; report.

## 9. Reuse vs new

- **Reused unchanged:** `load_trace`, `build_anchor`, `build_packet`, `run_case`, `_jobs_of`
  (`no_text`), `candidate_actions`, `apply_signed`/`check_allocation`, `exp88_analyse` McNemar
  helpers, the whole `exp90_specificity.py` scoring loop.
- **New:** two optional params + the imbalance gate in `sample_scenes` (a few lines); four
  CLI flags + a hard-vs-easy print block in the driver; the Exp 90-reproduces-when-off regression
  test.

## 10. Reproduce

```bash
cd Research
# sanity: dump hard scene stats (spread + tight slack)
.venv/bin/python -m pins.no_exception_scenes --n 200 --seed 0 --spread-min 4 --slack-lo 0.05 --slack-hi 0.30
# the run
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6 \
  --spread-min 4 --slack-lo 0.05 --slack-hi 0.30 \
  --out pins/results_exp91_hard_qwen2514b.json | tee pins/exp91_hard_14b.log
```

## 11. Relation to the thesis & to Exp 90

Exp 90 = precision on ordinary jobs (0/200, but scenes possibly too easy). Exp 91 = precision on
jobs **perturbed toward tension** — the missing stress test. Paired with Exp 89's recall (43/81),
the sensitivity/specificity story now covers both easy and adversarial no-exception jobs, and does
so with jobs drawn straight from v2020 — immune to the "self-authored, small n" criticism.
