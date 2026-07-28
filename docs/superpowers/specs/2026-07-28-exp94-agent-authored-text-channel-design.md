# Exp 94 — the agent-authored text channel (PRE-REGISTRATION)

**Date:** 2026-07-28  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b
**Status:** written **before any code**. No arm has been implemented, no pilot run, no data seen.

## 1. The question

Exp 92 showed the winning escalation is a **literal no-op** in-sim: 843 triggers, 0 LLM calls, 0
changes, allocation identical to `market` — because the v2020 replay has no text channel. §4.4
therefore rests on an authored suite, and §5 concedes the honest limitation.

This experiment asks whether the text channel can be **generated rather than fabricated**. When
the state changes materially between t−1 and t, the demand and supply agents each report, in plain
text, *why their situation changed*, alongside their numeric request. The referee then rules on
the packet as it already does.

This is the two-LLM design in `CLAUDE.md` closing its loop: the agents hold private state, and the
transcript is the interpretability edge. Until now they have emitted only numbers.

**What this is NOT.** It is not code synthesising operator notes from public numbers. §4.4 declines
that explicitly and it would measure a translation, not a channel. Here the text is authored by an
agent from state the market does not represent.

## 2. The gap this exploits, stated precisely

The market has **one-step allocation memory** and no causal memory:

```python
m = max(0, facts.get("held", base) - base)   # C_resize prices change from current holdings
```

It knows an allocation changed. It has no representation of **why** — a margin lost to a prod
arrival and a margin lost to a downward revision of predicted usage are the same number to it.

## 3. Arms

Three arms, paired within seed. Pool 8, `caps=predicted`, qwen2.5:14b, v2020 replay.

| arm | note content | LLM budget |
|---|---|---|
| `market` | none | 0 |
| `narrated` (**placebo**) | states **what** changed; forbidden by prompt from stating why | matched to `attributed` |
| `attributed` | states **why** it changed | matched to `narrated` |

**The placebo is the point of the design.** Both notes are authored by the same model at the same
call count on the same trigger; the arms differ *only* in whether a causal claim is permitted.
Without it, a win could not be separated from "the LLM was handed cross-tick history the market
lacks", which would be a finding about the market's objective, not about text.

Prompt symmetry (length, structure, effort) is to be reviewed and recorded **before** the run.

### Trigger

Tick-level: the **existing** `_trigger` from `pins/market.py` (`_make_trigger`), unchanged, so
escalation frequency is identical to Exp 87 and Exp 92 and the arms stay comparable to the
published gated result.

Per-job authoring condition — who gets to speak on a fired tick: the job's held margin changed
since t−1, **or** its deadline bucket changed, **or** it arrived this tick. Supply always speaks
on a fired tick.

### What the agents may see

Pinned, so "the LLM got extra state" cannot be argued after the fact: previous allocation,
previous free GPUs, previous deadline bucket, and the agent's own previous request. Nothing else.
Notes land in `build_packet`'s `demand_claims[].note` and `supply_claim.note` — the identical
fields the authored suite uses, so the mechanism validated in Exp 82–89 is unchanged.

## 4. Hypotheses

- **H1 (primary).** `attributed` − `narrated` on **overall SLA**. Two-sided: a negative result is
  as informative as a positive one, and no direction is predicted.
- **H2.** `attributed` − `market` and `narrated` − `market`, same metric.
- **Equivalence.** TOST at **±2.0 SLA points** on H1, so "no difference" is a positive claim rather
  than a failure to reject.

**Predicted outcome: equivalence.** In a simulator the causes of a change are largely determined by
state already logged, so causal attribution should carry little the numbers do not. Recording the
prediction here so that a null is a confirmed prediction, not a salvaged one.

## 5. Analysis axes

In-sim convention, matching every other trace-replay experiment (not the hard-case convention):

| axis | value |
|---|---|
| pairing | by seed, n=32, identical windows across arms |
| significance | \* = 95% CI excludes 0 |
| family correction | Holm across the vs-floor family, as `trace_replay` already reports |
| equivalence | TOST ±2.0 pts, SLA only |
| secondary metrics | prodSLA, util, useful, regret — reported, Holm-corrected, not headline |
| cost | tokens/seed and escalation count reported per arm, always |

A pilot at n=8 may be run to check the harness and estimate runtime. **The pilot may not be used
to choose the primary metric, the arms, or the direction** — those are fixed above. Headline is
n=32; TOST requires the full n.

## 6. Decision rule

1. **`attributed` ≈ `narrated` (TOST passes), both ≈ `market`** → the channel is inert in-sim.
   Consistent with Exp 92; strengthens §5's claim that the capability needs a workload whose
   causes are not determined by its numbers. *Predicted.*
2. **`attributed` > `narrated`\*** → causal explanation is load-bearing. First in-sim evidence of a
   real text channel. Would be a major result and therefore requires replication at a second seed
   block and a second model before any paper claim.
3. **Both > `market`\* but `attributed` ≈ `narrated`** → the gain is **history, not text**. The
   honest reading is that the market's objective is under-specified and should carry more
   cross-tick state; the follow-up is `BID_W`/`C_resize`, not the LLM. This branch is pinned now
   precisely because it would otherwise be tempting to report it as a text result.
4. **Either arm < `market`\*** → authoring cost or distraction; report as a cost of the channel.

No outcome changes §4.4, which rests on Exp 89. Outcome 3 would change §4.1.

## 7. Threats to validity

- **Memory confound** — addressed by the placebo; this is the reason it exists.
- **Cache dilution** — notes vary per tick, so LLM cache hit rate will be far below the referee
  arms'. Expect a substantially longer run than Exp 87; runtime to be measured in the pilot and
  the run detached (`setsid`), per the login-node reaper note.
- **Prompt asymmetry** — the two note prompts must not differ in effort or length; reviewed and
  recorded before running.
- **Simulator determinism** — the "why" is largely recoverable from logged state, which is why
  equivalence is predicted. This bounds what a null can claim: it is evidence about *this* world,
  not about text channels in general.
- **Authoring by the same model that rules** — demand/supply notes and the referee share a model.
  Not corrected for; recorded as a limitation.

## 8. Reproduce

To be filled at implementation. Harness will extend `pins/market.py` with a `make_policy_authored`
arm and a `--authored {narrated,attributed}` flag on `trace_replay`, additively, leaving the
`market`, `gated` and `corrected` arms byte-identical.

Analysis to be run through the **`pins-analyst`** subagent against this document.
