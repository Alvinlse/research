# IEEE Workshop Submission Plan — deadline 2026-09-21 (6 pages)

*Written 2026-09-02. Owner: Lay Kim Seng. Paper: `paper/ieee.tex` (IEEEtran, builds clean,
currently 4 pages). Supersedes the jlreq `main.tex` scope for this submission.*

## The paper

**Title:** Does the LLM Earn Its Cost in GPU Scheduling? A Gated Architecture That Separates
Efficiency from Exception Handling

Built only on results that survive the Exp 97 operating-point rebase:

| § | Content | Source | Status |
|---|---|---|---|
| IV-A | Market wins every efficiency metric at zero tokens; deadline effect exactly 0.0 | Exp 72 → 97 | rebased; re-verify at standard 8-GPU scale (amdahl row lands with the Exp 92 sweep; sat law to run) |
| IV-B | Least-laxity ordering: −16.4 ± 3.1\* SLA at zero tokens — largest effect in project | Exp 99 | measured at the standard point, done |
| IV-C | Text exceptions: debate 43/81 vs boN 29/81 at matched budget, p=0.0007; blind batch p=0.0032; budget-alone null p=0.25; specificity cost 12/17 vs 16/17 | Exp 89/90 | done, not in-sim, untouched by rebase |
| IV-D | Boundary: same escalation in-sim = literal no-op (0 calls, 0 changes, identical allocation) | Exp 92 | **re-run at standard point 2026-09-02**: n=2 probe confirms no-op holds (403 escalations, 0 calls); n=32 sweep running |

**Cut** (needed the killed Exp 97 Stage-2 rerun): the one-scalar/composed claim (old claim 2),
the referee frontier, the negotiation-era section (§5 of the old draft) — negotiation survives
only as a baseline row in Setup.

**Reviewer defences already baked into the tex:**
- Dataset table for the 98-case suite (categories, n, controls).
- The 0/81 rigid floor explicitly labeled an *inclusion criterion*, not a finding.
- Boundary condition (placebo/confirm controls) reported next to the headline.

## Compute in flight (2026-09-02)

1. **Exp 92 rebase, n=32** — `pins/exp92_rebase.log`, results
   `pins/results_exp92_rebase.json`. Fills Table "noop" in IV-D + the amdahl market row.
2. **Cross-family Exp 89 replication** — gemma2:9b then gemma2:27b, r34 suite, default arms
   (`pins/exp89x_gemma2*.log`, results `pins/results_exp89x_gemma2*_t0.8.json`).
   If it lands: one sentence in IV-C + threats update. If negative: report it as measured.
3. ~~**To queue next:** market-vs-floor at standard point, `--law sat`, n=32~~ *(done
   2026-09-02: no-op holds in both laws; Tables 1 and 5 complete.)*
4. **Frontier-model arm (NEW 2026-09-02).** All hard-case results are local models ≤27b; a
   reviewer will ask whether the debate effect is an artifact of small-model weakness. Test
   the same suite on a frontier API model (Claude Sonnet/Haiku via Anthropic API; optionally
   a GPT-5.x model via OpenAI API).
   - **Implementation:** adapter in `pins/correction.py:_ask` — route model names
     `claude-*`/`gpt-*` to the respective API instead of the Ollama client. Prompts, cache
     (key already includes model), arms, and scoring unchanged. `format="json"` has no
     Anthropic equivalent; prompts already demand JSON and `_parse` handles it — verify on a
     5-case smoke run first. Temperature must be honoured for the boN arm.
   - **Pre-registered read, declared before running:** primary contrast stays debate vs
     budget-matched boN, one-sided McNemar, per model, never pooled with qwen. Two admissible
     outcomes, both publishable: (a) the debate edge *persists* → structure is orthogonal to
     capability; (b) single-call *saturates* and the edge shrinks → extends the
     capability-gating result (claim 7: packet +6 at 14b, −6 at 7b) upward — debate is a
     capability prosthesis small local models need. Also record the 17 controls (does frontier
     fix the 12/17 specificity cost?) and citation faithfulness.
   - **Caveat to state in the paper either way:** cases were authored/iterated against rigid
     arms in the qwen era; a ceiling effect compresses discordant pairs, so a null at high
     scores is underpowered, not evidence of equivalence.
   - **Cost:** ~1,300 calls × ~2k tokens ≈ a few million tokens — single-digit dollars on
     Sonnet/Haiku. **Prerequisite: API key(s)** — confirm `ANTHROPIC_API_KEY` (and OpenAI if
     used) are available on the login node; runs are I/O-bound, reaper-safe.
   - **Paper placement:** one row block in Table "arms" or one paragraph in IV-C + Threats
     rewrite ("one model family at one scale" → measured across scale and provider).

## Schedule

### Days 1–4 · Sep 2–5 (Wed–Sat) — scope locked, long poles started
- [x] Lock 6-page scope (advisor already informed)
- [x] IEEEtran skeleton `paper/ieee.tex`, builds clean
- [x] Exp 92 re-run launched at standard point; probe confirms no-op
- [x] gemma2 9b/27b replication launched
- [x] sat-law market run — done 2026-09-02: no-op holds in both laws (5494/5517 escalations, 0 calls)
- [ ] **Recruit two raters** for predicate-agreement study: 30 sampled cases, scenes shown
  without predicates. *Only item nobody else can start — start it now.*
- [ ] Confirm co-author list (tex TODO)

### Days 5–10 · Sep 6–11 (Sun–Fri) — skeleton → full draft body
- [ ] **Frontier-model arm** (compute item 4): API adapter in `_ask` + 5-case smoke run,
  then full r34 run on Claude (Haiku first as the cheap canary, then Sonnet). *Adapter built + smoke-tested 2026-09-02 (`correction.py:_ask_api`); temperature rides extra_body (SDK 1.x dropped the kwarg — if the server rejects it, boN runs at API default sampling, documented).* Declare the
  per-model primary contrast before the full run. Blocked on API key confirmation — resolve
  that Day 5.
- [x] Figure 1: pipeline (market → validator → trigger → debate), TikZ in-tex *(done 2026-09-02)*
- [ ] Related work: ~25 real references (currently 6, 2 placeholders). Budget 2 full days.
  Needed: LLM-scheduling (2–3), debate/self-consistency (Du et al., Wang et al.),
  LLM-as-judge, DRL schedulers, trace papers, mechanism design.
- [ ] §I–III polish (intro, related, architecture)
- [ ] Collect rater results, compute agreement, fold into IV-C + Threats
- [ ] Fill IV-D table from finished n=32 run; fill gemma sentence

### Days 11–15 · Sep 12–16 (Sat–Wed) — evaluation complete
- [ ] IV-A–IV-D final numbers, all red TODOs cleared
- [ ] Threats section finalized (synthetic recipe, authored suite, single-annotator → rated,
  model coverage, calibration regime)
- [ ] Abstract rewrite against final content (claim 2 is out — abstract must not state it)

### Days 16–18 · Sep 17–19 (Thu–Sat) — review & trim
- [ ] Full draft to advisor **by end of Sep 17** (gives 2 review days)
- [ ] Trim to 6 pages. Cut order if over: Exp 84 architecture ablation → per-category
  exploratory split → Exp 83 docs-substitutes paragraph
- [ ] Reference check, clean `latexmk` build, page count on the real template

### Days 19–20 · Sep 20–21 (Sun–Mon) — submit
- [ ] Final polish, advisor sign-off incorporated
- [ ] **Submit Sep 20** — one day of slack for portal problems. Sep 21 is the fallback, not
  the target.

## Risks

| Risk | Mitigation |
|---|---|
| ~~Exp 92 no-op fails at new operating point~~ | **Retired** — probe confirms it holds |
| gemma replication comes back negative | Report honestly; headline stays qwen2.5:14b with a scoped claim |
| Frontier model saturates the suite (ceiling effect) | Pre-declared as outcome (b): extends capability-gating upward; report with the underpowered-null caveat, never as equivalence |
| No API key / budget on the node | Arm degrades to a stated limitation in Threats; paper does not depend on it |
| Rater agreement is low | Report the number, scope the claim to "author-defined defensibility"; still better than silence |
| 6-page overflow | Cut order pre-decided (above) |
| Advisor turnaround | Draft delivered Sep 17, not Sep 19 |
| Login-node reaper kills sweeps | One sweep per background shell, `--seed-start` sharding, `PINS_RESULTS` always set |

## Working rules for every run

- `PINS_RESULTS=<new file>` on **every** invocation — default paths hold published runs
  (results_hardcases.json was clobbered once already on 2026-09-02 and restored from git).
- Ollama: `OLLAMA_NUM_PARALLEL=1`, `PINS_NUM_CTX=8192`, verify `ollama ps` says 100% GPU.
- LLM runs sequential, never two models concurrently (GPU swap thrash).
