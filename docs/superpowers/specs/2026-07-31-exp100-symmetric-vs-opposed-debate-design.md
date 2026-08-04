# Exp 100 — symmetric vs opposed debate: is the demand/supply asymmetry load-bearing? (PRE-REGISTRATION)

**Date:** 2026-07-31  **Branch:** `referee_allocator`  **Model:** qwen2.5:14b
**Status:** written **before any code and any run**. The `debate-sym-pkt` arm does not exist yet;
§4 specifies it. No pilot has been run and no LLM call has been made under this design.

## 1. The question

The project's founding design assumption, stated in `CLAUDE.md` since June and never measured:

> Two agents negotiating is only meaningful if they want different things or know different
> things. **With symmetric objectives the "discussion" is theater.**

Everything built since rests on it: the demand agent advocates for its job, the supply agent
protects headroom, and the tension between them is supposed to be what produces a better ruling.
This experiment asks whether that asymmetry does any work at all.

## 2. Why this is the right experiment now

Three results box the question in, and none of them answers it:

- **Exp 79** — perspective splitting in *parallel* is inert: demand/supply reviewers who state
  positions independently and never read each other tie a single call across seven models
  (confirmatory pool b=3 c=2, p=0.500; all seven b=6 c=5, p=0.500). Two perspectives *alone* buy
  nothing.
- **Exp 83/89** — the same two perspectives *with a rebuttal round* win: 43/81 vs a budget-matched
  best-of-N 29/81, b=16 c=2, p=0.0007. So the cross-talk is doing something.
- **Exp 93** — but the *content* of the arguments is not what does it: stripping `evidence` from
  the packet (who proposed what, minus the why) costs **zero** cases, 43 = 43.

So the win is the second pass. What remains unknown is whether that second pass needs reviewers
who want **different things**, or merely a second reviewer at all.

`critic-pkt` (34/81) is a second pass with **one** neutral perspective, and `debate-pkt` (43/81)
is a second pass with **two opposed** ones — but that pair confounds *four* things: budget,
number of perspectives, opposed-ness, and cross-talk. The log's standing item ("critic vs debate
at matched budget") removes only the first. This experiment removes opposed-ness alone.

## 3. Arms

Two arms, differing **only** in the objective each reviewer is given.

| arm | reviewers | objective | calls/case |
|---|---|---|---|
| `debate-pkt` (existing) | per-job demand + supply | **opposed**: advocate for my job / protect headroom | $2n+3$ |
| `debate-sym-pkt` (new) | per-job reviewer + capacity reviewer | **shared**: both judge what best serves overall service quality | $2n+3$ |

**Budget is matched by construction, not by tuning.** Both arms issue one opening call per
text-bearing job, one supply/capacity opening, one rebuttal call per text-bearing job, one
supply/capacity rebuttal, and one referee call. The call-count expression `2*n_jobs + 3` in
`exp88_budget_control.py` is shared and unchanged.

Held identical: the suite, the packet builder, what each reviewer may see (`_opening_view` —
every other position and its evidence, never private state), the output schema
(`delta_gpus` / `hold_free_gpus`), the referee stage (`SYSTEM_PACKET_REFEREE`), `apply_signed`'s
feasibility check, temperature 0, and the cache-key discipline.

**The only difference is the system prompt's objective clause.** The symmetric prompts keep the
same structural roles (one reviewer owns one job's delta; one reviewer owns the hold-free number)
so that the output schema and call count cannot change — what is removed is the *advocacy*: no
"argue your corner", no "every extra GPU your job gets comes out of another job", no "do not
release capacity that a stated limit requires". Both reviewers are told to recommend what best
serves overall service quality across the whole pool.

## 4. Implementation (to be written after this document is committed)

`pins/correction_signed.py` gains an optional prompt-set parameter on `gather_signed` and
`debate_signed`, defaulting to the current opposed prompts, so **every existing arm replays
byte-identically**. A `SYMMETRIC` prompt set supplies the four neutral system messages.
`exp88_budget_control.py` gains `debate-sym-pkt` in `STRUCTURE_ARMS` and routes it through the
same `gather → debate → packet → referee` path as `debate-pkt`.

Byte-identity of `debate-pkt` against Exp 89's stored results is the harness tripwire (§7).

## 5. Hypotheses and predictions

- **H1 (primary).** `debate-sym-pkt` vs `debate-pkt` on the 81-case suite, strict metric
  (handled **and** feasible), one-sided exact McNemar on discordant pairs, both directions
  reported.
- **H2 (placement).** Both debate arms vs `critic-pkt` (34/81) and `single-pkt` (27/81), to place
  symmetric debate on the existing ladder.
- **H3 (controls).** The 17 no-exception control scenes. Neither debate arm may fire more than
  the other; a specificity difference would confound H1.

**Pre-registered prediction: symmetric ties opposed.** Exp 93 already showed the *content* of the
arguments is inert (43 = 43 with evidence stripped), and an advocacy framing is a property of that
content. If the second look is the mechanism, the objective clause should not matter. Stated
plainly because the opposite result is the one that would vindicate the project's founding
assumption, and pre-registering the sceptical prediction is what makes that vindication mean
something.

Equivalence, not just absence of difference, is the interesting outcome here, so H1 is also
reported as a TOST at $\pm 5$ cases alongside the McNemar.

## 6. Decision rule

1. **Symmetric $\equiv$ opposed** *(predicted)* → the demand/supply asymmetry is **scaffolding**:
   it produces a second reviewer, and the second reviewer is the mechanism. §4.4 must be reworded
   from "opposed advocates" to "a second pass with cross-talk", and "two-sided" should come out of
   the paper's framing. `CLAUDE.md`'s founding assumption is recorded as **measured false** in this
   venue.
2. **Opposed $>$ symmetric** → the asymmetry is load-bearing, the founding assumption is
   vindicated, and "two-sided" earns its place in the title. Immediate follow-up: whether the
   effect is the opposition itself or merely role *diversity*.
3. **Symmetric $>$ opposed** → advocacy is actively harmful (reviewers over-claim for their own
   job). Would reframe the mechanism as neutral review and make `critic-pkt` the arm to develop.
4. **Either debate arm fires significantly more on the 17 controls** → H1 is confounded by
   specificity and is reported as such rather than as a structure result.

No outcome changes §4.1–§4.3. This is a claim about *why* §4.4 works, not whether it does.

## 7. Threats to validity

- **Harness tripwire.** `debate-pkt` must reproduce Exp 89's 43/81 exactly. If it does not, the
  prompt-set refactor changed behaviour and the comparison is void until that is fixed.
- **Prompt confound.** The symmetric prompts must differ from the opposed ones *only* in the
  objective clause — same length band, same output schema, same JSON instruction, same
  "one sentence to the referee". Exp 94's follow-up (b) flagged output-length asymmetry
  (10.9 vs 28.2 words) as an uncontrolled variable in an earlier text comparison; here the two
  prompt sets are written to the same length and that is checked before the run.
  **Measured before the run (2026-07-31), opposed vs symmetric, in words:**
  demand 136/140, supply 106/114, demand-rebut 115/116, supply-rebut 82/92 — a maximum gap of
  12%, against Exp 94's 2.6× problem. The residual is the objective clause itself ("does OVERALL
  service quality improve if…" is inherently longer than "should this job get…"), i.e. it *is*
  the treatment; padding the opposed set to match would introduce a worse confound. Recorded
  rather than engineered away.
- **Cache-tag separation (verified before the run).** `correction._ask` keys on
  `f"{PROMPT_VERSION}|{tag}|{user}|{model}"` and does **not** include the system message, so a
  symmetric arm reusing the opposed tags would silently return the opposed arm's cached answers
  and measure nothing. The two sets share no tag (`dem-signed`/`sup-signed`/`dem-rebut`/`sup-rebut`
  vs `job-sym`/`cap-sym`/`job-sym-rebut`/`cap-sym-rebut`); checked programmatically, empty
  intersection.
- **One venue.** This is the authored hard-case suite. A null here does not establish that
  asymmetry is inert in a negotiation over real contested capacity (Exp 22–42's venue), where the
  agents hold genuinely private information rather than assigned framings.
- **Model-bound.** qwen2.5:14b only. The packet is capability-gated (+6 at 14b, −6 at 7b), so a
  smaller-model replication is not informative and is not run.
- **Reaper / concurrency.** One arm at a time, no concurrent LLM work. Exp 63 must be finished
  before this starts.

## 8. Reproduce

```
PINS_RESULTS=pins/results_exp100.json .venv/bin/python -m pins.exp88_budget_control \
    --suite r34 --model qwen2.5:14b --arms debate-pkt,debate-sym-pkt,critic-pkt,single-pkt
```
Analysis against this document; the McNemar and TOST are computed from the per-case outcome
vectors, the same shape `exp93_analyse.py` reads.

## 9. Analysis amendment (2026-08-04, added MID-RUN — read §9.1 first)

Two additions to §5's reporting, prompted by an external adversarial review of the Exp 93
premise this design rests on.

### 9.1 Provenance — exactly what had been observed when this was written

**This is not a blind amendment and must not be presented as one.** Written 2026-08-04T16:11+09:00,
after the run of §8 was launched. At that moment `pins/results_exp100.json` did not yet exist, no
aggregate score, discordant count, McNemar or TOST had been computed for any arm, and the log
contained **6 of 98 case lines** — the entirety of what had been seen:

```
P3-POLICY-01   nl_policy    debate-pkt=.* debate-sym-pkt=.* critic-pkt=.* single-pkt=.*
P3-POLICY-02   nl_policy    debate-pkt=.* debate-sym-pkt=.* critic-pkt=.* single-pkt=.*
P3-POLICY-03   nl_policy    debate-pkt=.* debate-sym-pkt=.* critic-pkt=H* single-pkt=.*
P3-POLICY-04   nl_policy    debate-pkt=H* debate-sym-pkt=H* critic-pkt=H* single-pkt=H*
P3-POLICY-05   nl_policy    debate-pkt=H* debate-sym-pkt=H* critic-pkt=H* single-pkt=H*
P3-POLICY-06   nl_policy    debate-pkt=.* debate-sym-pkt=.* critic-pkt=.* single-pkt=.*
```

Five of those six are concordant across all four arms; the sixth differs on `critic-pkt`, which is
not an H1 arm. Neither addition below is derivable from those lines. Recorded rather than
engineered away, in the same spirit as §7's prompt-length disclosure.

### 9.2 Report the ±3 bound alongside the pre-registered ±5

§5 registers TOST at ±5 cases. That bound is **looser than the ±3 bound Exp 93 already failed**
(90% CI [−4.91, +4.91], `research_progress.md:2635`), and Exp 89's headline debate-vs-best-of-N gap
is 14 cases. A ±5 pass is therefore compatible with a difference that is not practically negligible
on this suite.

- **Primary, pre-registered:** TOST at ±5. Unchanged; this is what §6's decision rule keys on.
- **Secondary, added here and labelled post-hoc in every write-up:** TOST at ±3, the Exp 93 bound.

A ±5 pass that fails at ±3 must be reported as *"equivalent within 5 cases, not shown equivalent
within 3"* — **not** as equivalence. Decision rule 1 ("symmetric ≡ opposed → the asymmetry is
scaffolding") requires the ±3 result to be stated next to it, because the wording change it
licenses in §4.4 is exactly the overclaim this amendment exists to prevent.

### 9.3 Report discordant pairs, never bare totals

Exp 93 was recorded as "43 = 43, zero cases changed". The totals tied; the **per-case vectors did
not** — 8 discordant pairs, 4 in each direction (`research_progress.md:2625`). Equal totals were
read downstream as "the argument content is inert", which is stronger than the data supports and
became the stated premise for running this experiment.

Every arm comparison here reports **b, c, and b+c** (the McNemar discordant counts) beside the
score, and no comparison in the write-up may cite matched totals without them. Two arms scoring
43/81 with 0 discordant pairs and two scoring 43/81 with 20 are different results.

### 9.4 What this experiment can and cannot conclude

Recorded before the numbers, to bound the write-up. Exp 100 varies the objective clause while
holding multiple roles, openings, cross-visible positions, rebuttals and the referee fixed
(§3, `exp88_budget_control.py:206`). Equivalence therefore licenses exactly:

> *"Opposed advocacy was not shown load-bearing within this authored suite and this architecture."*

It does **not** separate second-pass from role diversity, cross-talk from independent
reconsideration, multiple contexts from multiple agents, or repeated calls to one model from
genuinely distinct agents holding private information. It is not evidence about whether multi-LLM
negotiation beats a single LLM in general; §7's "one venue" and "model-bound" threats still apply.
