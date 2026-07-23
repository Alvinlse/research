# pins-analyst subagent — design

Date: 2026-07-23
Status: approved (design), pending implementation

## Goal

A reusable custom subagent that analyzes a finished PINS experiment's results and
reports back a table **plus the statistical method it chose and why**, so the number
can be audited before it is trusted. This offloads result-analysis — the researcher's
stated bottleneck — while keeping the heavy JSON crunching out of the main session's
context.

Design rule it embodies (project-wide): *the LLM reasons/explains; deterministic code
decides.* Here the subagent reasons about which test to use and explains it; the
researcher decides whether to trust it.

## What it is

A single agent-definition file: `Research/.claude/agents/pins-analyst.md`.
No Python, no dependencies. Invoked via the `Agent` tool with
`subagent_type: pins-analyst`.

## Contract

**Input (per invocation, supplied by caller):**
- which results file (e.g. `pins/results_backup_pre_exp87_gated.json`),
- which arms/conditions to compare,
- the question in plain terms (e.g. "does the gated arm beat bare market on SLA?").

**Mandatory grounding step (runs first, before any test is chosen):**
Search for a pre-registered analysis for this comparison:
- `pins/exp*_analyse.py` (e.g. `exp79_analyse.py`),
- pre-registration text in `research_progress.md`,
- `pins/hardcases_r3.py` and referee-prompt scoring rules.

Then branch:
- **Pre-registration exists → obey it.** Use its test, its scoring, its pooling,
  its one-sided/two-sided choice. Do not substitute its own judgement.
- **No pre-registration for this comparison → STOP and ask the caller** which test and
  scoring to use. It never silently improvises a test. (Explicit design decision.)

**Output (required block, every time):**
- the result table (the compared arms with their metrics),
- **Method used** — the exact test and correction,
- **Why** — the one-line justification (which pre-reg it obeyed, or which
  caller-approved choice it applied),
- **Scoring** — reported under STRICT (handled AND feasible) and bare `handled`,
- **Caveats** — e.g. unpaired comparison, overwritten rows, competence-bar exclusions.

## Baked-in discipline (agent system prompt)

- **Schema knowledge:** results JSON is
  `tiers → {arm-config key} → "per_seed" → "{pool}" → {arm} → [ per-seed dicts ]`,
  each per-seed dict has `sla, prod_sla, util, slowdown, finished, fallback_rate`.
- **Defaults it may use ONLY after the caller approves a method for an
  un-pre-registered comparison:** paired / matched-pair tests; Holm correction across a
  family of comparisons; report strict AND bare scoring side by side; one-sided tests
  only when the direction was pre-declared.
- **Feasibility / competence discipline:** flag arms that over-award (allocate >100 % of
  a pool); apply the <25 % infeasibility competence bar when it is the pre-registered
  rule; never let bare `handled` stand alone when over-award is possible.
- **Read-only:** it analyzes results files only. It never launches runs, never edits or
  overwrites results, never touches caches.

## Why a subagent, not a skill

The JSON crunching and intermediate statistical reasoning stay in the subagent's own
context; the main session receives only the final table + rationale. A skill would run
the same work in the main context, defeating the offload that motivated this.

## Verification of the workflow itself

Before relying on it, run it once against a comparison whose answer is already
published — e.g. Exp 79's flat null (interaction +0), or Exp 87's SLA-neutral result
(dSLA −0.2, ns). Acceptance: it reproduces the published number **and** names the
correct pre-registered test. If it invents a different test or misses the pre-reg, the
grounding instructions are fixed before use.

## Out of scope (YAGNI)

- No running of experiments (separate concern, not requested).
- No write-up / paper drafting.
- No new stats library. House style is stdlib-only exact tests (`exp79_analyse.py`
  computes exact McNemar with `math.comb`); the subagent reproduces the pre-reg
  script's own computation and prefers stdlib. `scipy` (1.15.3) is in `.venv` if a
  test genuinely needs it; `statsmodels` is NOT installed and must not be assumed.
- No orchestration of multiple analyses; one comparison per invocation.
