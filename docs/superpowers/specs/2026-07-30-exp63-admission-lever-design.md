# Exp 63 — the admission lever, finally measured (PRE-REGISTRATION)

**Date:** 2026-07-30  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b
**Status:** written **before any run**. The lever itself was built on 2026-07-22 (`9c1d127`,
`438df61`) and has never been measured; **no code is added by this experiment**, and no pilot has
been run. Any code change found necessary during the run is disclosed here before the run restarts.

## 1. The question

`paper/pins_gated_draft.md` §5 says:

> *"In this contended sim, overall SLA is governed by which jobs start, not which go faster… A
> reasoning gain on overall SLA therefore requires a text exception that moves admission or
> priority, not margin — a direction we scope for future work."*

The lever that moves admission is **already in the tree**. `--admit` gives the policy a `waiting`
table and lets its outcome carry `defer` (hold a job at ceiling 0, only if it has never run — the
rigid no-preemption invariant is preserved) and `priority` (reorder grants *within* a tier). Two
holders of that lever exist:

- **deterministic** — `negotiation_protocol.admission_plan(waiting, free, reserve)`: prod first,
  then behind-deadline, then smallest base; the marginal job takes a **partial** allocation rather
  than being deferred (the strict whole-base-or-wait version measured −8.3\* utilisation and was
  rejected during the build).
- **the referee (rule 7)** — `RULE7_ADMIT` is appended to `SYSTEM_REFEREE` only when a waiting
  table is passed; the ruling's `defer` / `admit_order` are parsed **unrepaired**, and deferring a
  prod job while admitting best-effort is a reported violation that floors the tick.

So the paper scopes as future work a mechanism that exists and is unmeasured. This experiment
measures it.

## 2. What this experiment can and cannot reach

`--admit` is wired to `referee`, `debate` and `negotiated` **only**. It does **not** reach `market`,
`composed`, `gated` or `corrected` — the arms claims 1–4 are built on. This is stated up front
because it bounds the conclusion: a positive result here is a result about the *negotiation-era*
arms and would justify wiring admission into the market as a follow-up; it is **not** by itself an
upgrade to the gated architecture. No such wiring is done in this experiment.

## 3. Arms and configuration

Two tiers, paired seed for seed, differing only in the flag:

```
.venv/bin/python -m pins.trace_replay --referee --llm --model qwen2.5:14b \
    --caps predicted --pools 8 --seeds 32 [--admit]
```

rows: `no-llm` (floor), `referee`, `negotiated`. amdahl law, v2020 replay, 16 jobs/window.
n=32 paired seeds. The floor row must be **byte-identical** across the two tiers (the build
verified this; it is re-checked here as a harness test, §7).

Pool 8 / 16 jobs is the affordable operating point on the login node. The more contended cell the
build had queued (pool 32 / slack 4 / 89 jobs) is **out of scope**: that is exactly the
configuration the reaper killed twice in Exp 64, and it needs a batch node.

## 4. Metrics

Primary metric is `sla`. The lever's *mechanism* metric is `wait_full` (first tick at full base,
horizon-censored) next to `wait` (first GPU) — a partial trickle-feed flatters `wait`, and partial
allocation is precisely what the marginal-job rule introduces, so the flattering metric may not be
read alone. `finished`, `starved` and `wait_max` are reported in every table.

## 5. Hypotheses (two-sided; no direction pre-declared)

- **H1 (primary).** `dSLA(referee+admit − referee)`, paired by seed.
- **H2 (the lever without judgement).** `dSLA(negotiated+admit − negotiated)`, paired by seed.
  Together with H1 this separates *having* the lever from *who holds* it.
- **H3 (judgement).** `referee+admit` vs `negotiated+admit`, head to head, paired. This is the
  flexibility thesis' own contrast: LLM judgement over WHO WAITS against a greedy code rule, on a
  workload with no text at all.
- **H4 (manipulation check).** `d wait_full` and `d wait` for both arms. If neither moves, the
  lever did not engage and H1–H3 are uninformative rather than negative — that reading is fixed
  now, before the numbers exist.

Significance is the house rule (95% CI excludes 0, paired by seed) with Holm over the vs-floor
family exactly as `trace_replay` prints it. H1 is the single primary test; H2–H4 are secondary.

**The censoring guard, declared in advance.** Deferral can buy SLA by never starting the jobs that
would have missed. An SLA gain is therefore **rejected** if it is accompanied by a significant drop
in `finished` or a significant rise in `starved`/`wait_max`. Unstarted jobs are censored at the
horizon by construction, so "defer forever" cannot look like a win on `wait_full`, but the guard is
pinned anyway because it is the one way this experiment could flatter itself.

**Predicted outcome.** The deterministic lever moves the queue a lot and the scoreboard little: the
build's own smoke showed `wait` 16.6 → 10.7 under `admission_plan`, so H4 should fire strongly while
H2 lands near zero on SLA — deferring trades one job's deadline for another's, and Exp 62 found no
regime in the sweep where the negotiation-era mechanism pays. H3 is predicted **null**: rule 7 is
judgement over a purely numeric waiting table, which is the setting where §4.1/§4.2 say code beats
the LLM, and Exp 92 showed the reasoning layer has nothing to add where there is no text. A
referee win on H3 would be the first in-sim judgement win in the project and is the reason this
experiment is worth running despite the prediction.

## 6. Decision rule

1. **H4 fires, H1 and H2 both null on SLA** *(predicted)* → the lever engages and does not pay.
   §5's future-work scope closes **negative**: admission control is available, moves queue delay,
   and does not convert into service quality on this workload. §4.3 is unaffected.
2. **H2 pays, H3 null** → admission belongs in the **deterministic** core, next to the market.
   Follow-up, pre-named now: wire `admission_plan` into `make_policy_market` and re-run against
   claim 1's cells. That would be a cheap SLA improvement, **not** a reasoning result.
3. **H3 pays (referee+admit > negotiated+admit)** → first in-sim judgement win; §5 gets rewritten
   and the immediate follow-up is *which* rulings differ (the `decisions` block carries them).
4. **Any SLA gain with fewer jobs finished** → not a win; reported as a censoring artefact.
5. **H4 does not fire** → the lever never engaged; this is a harness result, not a scheduling one,
   and the write-up says so instead of reporting a null.

No outcome changes claims 1, 4, 5 or 9. Claim 2 is untouched: this is a different lever from the
reserve scalar.

## 7. Threats to validity

- **No text.** Rule 7 rules over numbers here. This tests admission *judgement*, not the text
  channel; the text-exception version of the same lever remains unbuilt and unmeasured.
- **The in-sim referee does NOT use the packet.** *(Added 2026-07-31, after H3 was computed — it was
  missing from the original pre-registration and is disclosed here rather than silently folded in.)*
  `pins/referee.py` contains no reference to `pins/packet.py`; the packet is imported only by the
  hard-case harnesses (`exp82_packet_2x2`, `exp88_budget_control`, `no_exception_scenes`,
  `test_exp90_scenes`), and `trace_replay` exposes no flag to enable it. Every referee ruling scored
  in this experiment is therefore delivered through the **pre-Exp-82 interface** — the one whose
  repair moved the market arm 0/31 → 14/31 and collapsed hard-case infeasibility from 11–16 to 0.
  H3's negative is consequently a result about *the referee as it exists in the simulator*, not
  about admission judgement in general, and it must be worded that way.
  Pulling the other way, and why this is a qualification rather than a refutation: in-sim
  `fallback_rate` is only 2%, so rulings are already mostly feasible and the packet's headline fix
  (feasibility) is not the binding constraint here; and Exp 92 showed the full packet+signed
  architecture is a literal no-op in-sim on this trace. A packet-equipped in-sim referee is an
  **unbuilt arm**, and no result here speaks to it. Same family as the standing weak-packet
  confound over Exp 65–67.
- **Cache keys.** The scene key digests the waiting set (`|adm:` hash, jids in, waited_ticks out).
  Untouched arms must keep byte-identical prompts and cache keys; the floor byte-identity check in
  §3 is the tripwire.
- **Unpaired-comparison trap (Exp 59).** The two tiers must differ only by the flag. Same seeds,
  same windows, same law, one results file (`PINS_RESULTS=pins/results_exp63.json`) so the
  `+admit` tier cannot clobber or be compared across files.
- **Partial allocation confound.** The marginal job's partial grant is part of the lever, not a
  bug; `wait` vs `wait_full` is how it stays visible.
- **Reaper.** Per-tick referee at n=32 is ~45 min of wall clock on this node. One arm at a time,
  no concurrent LLM work.
- **`debate` is not run.** It also accepts `--admit`, but adding it would double the LLM cost and
  it answers no question H1–H3 does not.

## 8. Reproduce

```
PINS_RESULTS=pins/results_exp63.json .venv/bin/python -m pins.trace_replay \
    --referee --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32
PINS_RESULTS=pins/results_exp63.json .venv/bin/python -m pins.trace_replay \
    --referee --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32 --admit
```
Analysis against this document; the paired contrasts H1–H4 are computed from the two tiers'
`per_seed` blocks, the same shape `exp96_analyse` reads.
