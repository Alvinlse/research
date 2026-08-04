# Research Progress — PINS experiment log

> **Restructured 2026-07-28.** This log is tiered. Sections kept **in full** are the ones the
> current paper (`paper/pins_gated_draft.md`) stands on. Sections marked *Compressed* keep their
> verdict and evidence table only — full text is at `git show b8b0af0:research_progress.md`. The
> retired Stage-1 GPU-memory/DAG prediction track (Exp 1–8, 19–21, 34) was **removed**; recover it
> from the same commit. Earlier pruning pass: 2026-07-06, through commit `4a74508`.
>
> **Reading order for a new session:** this claims table → the experiments it cites → nothing else
> unless you need provenance.

## State of the claims (2026-07-28)

> **⚠ Operating-point warning (2026-07-30, Exp 97).** Every in-sim **SLA / prodSLA** number in the
> table below was measured where the deadline was set against a job's *solo* runtime (slack
> 1.15–2.4× work) at 76–89% utilisation — a regime in which the floor misses **54% (amdahl) / 40%
> (sat)** of deadlines and ~2.5% of jobs are counted late purely because the horizon censored them.
> That is not a cluster anyone would run. The practical operating point is `--horizon 400
> --slack-mult 10` (floor violations 11–13%, every job finishes, util 76–78%); the rebase is
> **in progress**. Efficiency results (util, useful util, regret) do not depend on the deadline
> recipe and are unaffected; the hard-case suite (Exp 79–83, 88, 89, 93, 95) is not in-sim and is
> untouched. Landed so far: claim 1 **survives and sharpens** (the market's deadline effect is
> exactly 0.0 ± 0.0 under amdahl while every efficiency gain holds), and claim 3 is **open** — at
> the practical point protection and efficiency look complementary, not like one dial.

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | The **deterministic bid/ask market wins every efficiency metric** — utilisation, useful utilisation, oracle regret — against every LLM arm, in all six pool×law cells, at **zero tokens** | Exp 72, 86 | solid (n=32, both laws) |
| 2 | The reasoning layer's **entire reproducible contribution is production-tier protection**, and it is captured by a **single reserve scalar**; a full per-job referee is dominated at 1/24000 of the tokens | Exp 70, 71, 73 → **Exp 96** | **qualified** — the workload made tier ≡ tight deadline; de-confounded, amdahl protection holds (DiD +2.2 ± 4.3 ns) but sat's Holm-surviving cell does not (−8.0\* → −2.7 ns, DiD +5.3 ± 5.8, p=0.053). Report both worlds; n≈175 would settle it |
| 3 | **Protection and efficiency are one dial, not two layers** — the reserve protects by holding GPUs idle, which is exactly the market's utilisation gain | Exp 73 | solid |
| 4 | The **gated architecture is efficiency-neutral**: the per-tick validator is byte-identical on correct clearings (0% fallback), and debate-on-trigger costs −0.20 ± 1.40 SLA (ns) against the bare auction | Exp 86, 87 | solid (n=32) |
| 5 | **Debate beats a budget-matched single call on text exceptions** — 43/81 vs 29/81 at equal calls, one-sided McNemar **p=0.00066**; best-of-N at the same budget is flat (p=0.250), so the gain is structure, not spend | Exp 83 → 88 → **Exp 89** | **solid (powered, n=81)** |
| 6 | **Parallel perspective-splitting is inert** — the referee-vs-single interaction is null across seven models and two families (confirmatory pool p=0.500). Sequential debate works; parallel review does not | Exp 79 | solid (negative) |
| 7 | The **decision packet is what makes rulings feasible** (infeasible rulings 11–16/31 → 0/31), and it is **capability-gated**: identical packet gives 9/31 at 14b and 0/31 at 7b | Exp 80, 81, 82 | solid |
| 8 | **In-sim, debate replacing the auction is SLA-null and production-costly** (dprodSLA −12.0\*) because the v2020 trace carries **no operator free text**; text-stripped controls score zero in every arm | Exp 84 | solid (negative, boundary) |
| 9 | **Reasoning has a specificity cost**: where no exception exists the market scores 16/17 and every LLM arm 12/17 — the argument for gating rather than defaulting | Exp 88, 89, 90, 91 | solid |
| 10 | Stage-1 prediction survives **only as an input**: `caps=predicted` (`pred_job_usage.csv`) feeds every headline run, and prediction error costs every policy while prod-SLA protection survives it | Exp 30, 35, 36 | solid (retained as infrastructure) |
| 11 | The negotiation-era mechanism (bounded protocol, ILP guarantee, per-user tariff, small-model sufficiency) is **superseded as a contribution** but retained as the `negotiated` **baseline arm** the market is measured against | Exp 22–48 (compressed) | superseded; baseline only |

| 14 | **A cheaper structure also survives its budget control** — an objection-only critic scores 34/81 against a best-of-N sampler at *identical* per-case spend (26/81, p=0.0193), while that sampler is one case *worse* than a single call at 3× the price. Third replication of "budget alone buys nothing" | Exp 93 → **Exp 95** | moderate (pooled n=81; blind r4 stratum ns, p=0.227) |
| 13 | **Debate's win is not in the argument *content*** — stripping `evidence` from the packet changes the score by exactly zero cases (43/81 both ways) but flips **8 discordant pairs, 4 each way**; the transcript stays readable and `evidence` can be dropped at no measured *net* cost | Exp 93 | **qualified — equal totals are not equivalence.** TOST at ±3 FAILS (CI [−4.91, +4.91]) on m=8. Do not state as "the win is the second pass": Exp 100 tried to extend that reading and came back inconclusive (claim 17) |
| 15 | **The text channel cannot be generated from the simulator's own state.** Agent-authored causal notes (`attributed`) vs a what-only placebo (`narrated`) give SLA identical *seed for seed at full precision*, and both equal the 0-token market exactly; the notes moved the allocation on 4 of ~842 escalated ticks and flipped no outcome. The cross-tick history the market lacks bought nothing, so §4.1 is untouched | Exp 92 → **Exp 94** | solid (negative, n=32, exact identity) |
| 16 | **Nothing in the system serves deadline laxity.** Once tier and tightness are drawn independently, no reasoning arm protects the tightest-laxity tercile in either law, and under sat the referee makes it significantly *worse* (+6.9 ± 5.1\*); the market's tight-tercile effect is identical seed-for-seed across both worlds, i.e. label-independent. Motivates least-laxity grant ordering (`two_sided_sim.py:454` still orders by frozen bid) | **Exp 96** | solid (negative, n=32, both laws) |
| 12 | The round-2 54-scene suite was **insensitive**, not merely negative: text-blind baselines (ILP 30/54, rule 31/54) scored within noise of every LLM arm, min p=0.238 across ~30 tests. Round 3's text-dependent design drops the rigid floor to 0/31 — the instrument was replaced, not the hypothesis re-shopped | Exp 65–67 → 79–83 | solid (methodology) |
| 17 | **A second pass with cross-talk beats one call whether or not the reviewers are opposed** — opposed 43/81 and symmetric 41/81 both beat `single-pkt` 28/81 (p=0.0015, p=0.0044), confirmed on the blind r4 stratum. But **opposed vs symmetric is UNRESOLVED, not equivalent**: b=4 c=2, McNemar p=0.6875, and TOST **fails at the pre-registered ±5** (CI [−2.74, +5.25]) as well as ±3. Only 6 discordant pairs — the instrument cannot separate them. `CLAUDE.md`'s founding "symmetric objectives are theater" assumption stays **open**; it is not measured false | **Exp 100** | inconclusive (underpowered, n=81, m=6) — do not cite as equivalence |

**Open / next**, roughly by value:

1. ~~**Exp 63 — the admission lever is built and never run.**~~ *(Done 2026-08-03, results
   `e978039`. Pre-registered **branch 2**: deterministic `negotiated+admit` improves SLA
   23.4% → 16.8%* and prodSLA 13.7% → 8.4%* at zero tokens, while the referee arm degrades to
   29.9% SLA / 66% util. Admission belongs in the deterministic core.)* **Live follow-ups:**
   (a) wire `admission_plan` into `make_policy_market` and re-run against claim 1's cells — the
   pre-named branch-2 successor, cheap and token-free; (b) re-run the referee arm on the **strong
   packet**, since the measured arm used the weak pre-packet interface and that is what the
   negative reading rests on.
2. ~~**Re-run the confounded structures on the strong packet.**~~ *(Done 2026-07-28 — Exp 93.
   `selfcons` needed no re-run (`single-pkt-boN` already is it); the argumentation ablation is an
   **exact tie** with debate (43 = 43), so debate's win is the second pass, not the arguments; the
   critic reconstruction is 34/81 vs single 27/81, p=0.0461, which **fails Holm** and is therefore
   suggestive only.)* *(Closed 2026-07-28 — Exp 95: the budget-matched critic arm ran, critic 34/81
   vs matched-budget boNc 26/81, p=0.0193, H1 supported; the blind r4 stratum is ns.)* Remaining
   work: **critic vs debate at matched budget** — the last confounded pair. *(Exp 100 measured this
   pair unmatched: debate 43 vs critic 35, b=14 c=6, p=0.1153 ns — suggestive, still not the
   budget-matched test.)*
2b. **Power the opposed-vs-symmetric question (Exp 100 follow-up).** Exp 100 came back
   **inconclusive, not equivalent**: b=4 c=2, m=6, TOST fails at ±5 and ±3 (claim 17). Resolving it
   needs more CASES, not more arms — at a ~7% discordant rate, separating the arms at ±3 takes
   several times n=81. Worth it only if the write-up needs to say *why* debate works; the
   *that* it works result (both debate arms ≫ single, p≤0.0044) does not depend on it. Note the
   bound: even resolved, it speaks to this authored suite and this architecture, and does not
   separate second-pass from role diversity or repeated calls to one model from distinct agents.
3. ~~**Update §4.4 to lead with Exp 92, not Exp 84.**~~ *(CLOSED — this entry was stale when
   re-read on 2026-07-30. Exp 92 already ran at n=32 (843 escalations, 0 calls, 0 changes) and
   commit `a8f6886` rewrote §4.4 to lead with the no-op boundary and demoted Exp 84 to an
   architecture ablation. Nothing left here.)*
4. **Exp 58 — the change-cost lever** (`--prev-input`, rule 6) is likewise built and never
   isolated.
4b. **Settle claim 2's wording (Exp 96 follow-up).** The de-confound DiD is inconclusive at n=32 and
   the two laws disagree; ~95 seeds (amdahl) / ~175 (sat) would give a ±2.5-pt DiD CI. Same command,
   more seeds, overnight. Until then claim 2 must be reported with both worlds' numbers.
4c. **Least-laxity grant ordering (Exp 96 branch 4).** Nothing in the system keys on laxity —
   `two_sided_sim.py:454` orders grants by frozen bid value. Ordering by laxity is a one-line
   deterministic lever with a pre-registered motivation now.
5. Malleability ablation (elastic-job fraction sweep) remains unstarted.
6. Exp 64's pool-32 rebaseline needs a batch node; the reaper killed it twice.

*Done 2026-07-28: `paper/pins_gated_draft.md` §4.4, abstract and contribution 5 now lead with
Exp 89 (commit `1472e21`).*

*Done 2026-07-29 — Exp 94: the agent-authored text channel is **inert in-sim**, as pre-registered.
The generated-channel route out of the authoring bottleneck is closed, and §5's limitation gets
sharper: the case for text now rests entirely on facts the state does not contain. Two follow-ups
this opens, neither started: (a) attributed's proposals are ~96% rejected by `apply_signed` and the
violation reason is counted but never logged — one line of instrumentation would say whether the
causal channel over-reaches or is merely unfundable; (b) output-length symmetry was never controlled
(10.9 vs 28.2 words), which must be fixed before any positive text result could be read as causality
rather than verbosity.* *(Follow-up (a) is now instrumented — commit `d36f09a`: the corrected and
authored arms tally which constraint the rejected ruling tripped (`unknown job(s)` / `cannot fund` /
`change budget` / `infeasible`) and print it beside the count. The authored re-run that populates
the counters has not been done.)*

<details>
<summary>Superseded claims table (2026-07-06) — kept for provenance</summary>


| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | LLMs cannot calibrate absolute resource numbers — CONFIRMED on real named models (22-572 GB MAE, famous names don't help); LLM-derives-structure + code-computes beats heuristic/mean/LLM on every metric, but on REAL architectures (13 Supercloud-labelled families) the margin is MAPE 18%/rho 0.80 vs 24%/0.40, not the synthetic ~2%/40x; residual = two mechanically fixable proxy blind spots (functional pools, transformer internals) | Exp 1-7 → **Exp 34** | solid (negative); revised (deterministic margin) |
| 2 | A small attention forecaster beats persistence on the dynamic channels; a quantile head + conformal calibration adds a usable per-job uncertainty signal at no accuracy cost | Exp 8, 16A | solid |
| 3 | Per-round value-max auctions lose SLA to stable serialisation; the bid-once committed-auction matches greedy on raw SLA and ~halves prod-tier SLA; an LLM can set & justify the priority (interpretable, matches deterministic) | Exp 9-12 | solid |
| 4 | The committed mechanism is gameable via inflated self-reports; a flat budget does NOT fix it; per-USER budgets + contested-tick claim pricing neutralise the deviation (lying's net gain → 0*, its cost lands on the liar's own prod jobs) at a measured rationing price (+3..+9 prodSLA pts honest-world); all-liars collapse is not rescuable by any anonymous tariff. A self-interested LLM user exploits the unpriced mechanism UNPROMPTED (−5..−14* vs honest; 3b games harder than 14b) and the tariff flips it to honesty-equivalence (14b exactly, 3b ns), citing the mechanism's logic in its justification | Exp 13 → **Exp 32-33** | solid (deterrence, n=32) |
| 5 | The supply agent's headroom-reservation lever pays only against rigid incumbents at moderate contention; malleability-aware reservation recovers the utilisation cost | Exp 14-15 | solid (regime-gated) |
| 6 | Uncertainty-sized margins are insurance whose value grows with tail severity; blanket margins backfire; the LLM hedge (7b+) beats the deterministic margin once given a spike-risk signal | Exp 16-17 | solid |
| 7 | The ILP ties the auction on a 1-D pool (~150x cost, not worth it) and earns its keep exactly where count-only clearing structurally fails: node placement (ploss to 0, util +7-12 pts) and making aggressive LLM margins SAFE | Exp 18, 25 | solid |
| 8 | The LLM does not earn its cost as a numeric predictor (runtime: retrieval wins; DAG demand: GBT/one-line rules win); its measured value is judgement + justification in the agent layer | Exp 19-21 | solid (negative for the LLM) |
| 9 | The two-sided split beats the single-LLM-both-objectives baseline: the concession ladder is the brake a lone agent lacks (single-llm over-commits at every scale) | Exp 22, 24, 27-29 | solid |
| 10 | On real trace replay, negotiation buys significant prod-SLA protection (-4..-8 pts, 95% CI) at no measurable overall-SLA or slowdown cost; "beats the floor on overall SLA" was 8-seed noise | Exp 28 -> **Exp 29** | solid (n=32) |
| 11 | The bounded protocol substitutes for scale: base world = statistical EQUIVALENCE (TOST ±3 pts, pre-registered) of rule ~ 3b ~ 7b ~ 14b on SLA at contention (pool-4 14b edge is the slack exception); predicted-time world = 3b genuinely BETTER (only arm beating floor, prodSLA −8..−9* all pools, beats 14b directly*) because scale buys conformity to a ladder that noisy time signals make harmful; family is a THRESHOLD not a ranking — qwen and gemma2:9b clear it (gemma buys prodSLA −7..−10* at all pools, not qwen-specific), llama3:8b does not (worse than 3b at every pool*, worse than gemma at slack*); un-braked single-llm still needs scale and still loses | Exp 27-29 -> **Exp 43-44** | solid (n=32, TOST, 2 families) |
| 12 | With Stage-1 predicted demand in the loop, prediction error costs every policy, yet prod-SLA protection survives it at every pool (rule AND 3b tiers); the slack-regime cushion vs the floor is significant at the rule tier, direction-consistent at 3b | Exp 30 | solid (rule + 3b) |
| 13 | Naive per-state LLM reflection does not converge (period-2 limit cycle) — the deterministic ladder, not reflection, is what makes a small model safe | Exp 26 | negative |

*Experiments 1–8, 19–21 and 34 cited above were removed in the 2026-07-28 restructure; read them
at `git show b8b0af0:research_progress.md`.*

</details>

**Environment:** single A100-PCIE-40GB · PyTorch 2.6.0+cu124 (`.venv-forecast` for torch) ·
qwen2.5 3b/7b/14b via Ollama `localhost:11434` · simulators pure-Python in `.venv`.

---

# Stage-2 NEGOTIATION — which allocation mechanism rations GPUs best?

The headline negotiation experiment (thesis refocus 2026-06-17): on one shared job stream,
does the PINS sealed-bid auction beat value-blind scheduler baselines on **SLA-violation rate**
at high utilisation? Harness: `pins/negotiation_sim.py` (pure Python, runs in `.venv`, no
LLM/MCP). It reuses the real decider `pins/mechanism.py:clear` and the predictor's
`marginal_values`/`PHASE_PROFILES`; baselines are wrapped with the same signature.

## Experiment 9 — Value-max auction vs greedy/equal/static (NEGATIVE; the diagnosis that shaped Stage-2)

**Date:** 2026-06-18 · `pins/negotiation_sim.py` (16 jobs, urgency scales both bids and deadline

**Result (SLA-violation %, 8-seed mean):** greedy-FIFO beats every auction at every contended pool — pool 8: greedy **46.9** vs PINS-auction 74.2, equal-share 76.6, static-sticky 100. The auction does win **welfare** (its own objective); deadline-scaled bidding made SLA *worse* (preemption churn).

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 10 — LLM agents bid strategically: hinge-safe and interpretable, but SLA unchanged (compressed)

**Date:** 2026-06-18 · `pins/llm_agent.py`: the LLM (qwen2.5:3b) picks a categorical stance +

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 11 — The stability lever: committed-auction beats greedy on prod-tier SLA

**Date:** 2026-06-18

**Result: committed-auction roughly MATCHES greedy on raw SLA and roughly HALVES prod-tier SLA** (pool 8: 23% vs 54%; pool 12: 15% vs 26%). It deliberately spends best-effort deadlines to protect production deadlines — exactly the value-weighted behaviour the thesis wants, and the per-round auctions never delivered. (Single-seed sweep is even stronger: committed wins BOTH metrics at pools 6 & 8.)

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 12 — LLM sets & justifies the committed priority (interpretable winner)

**Date:** 2026-06-18

**Result (8-seed mean; SLA / prodSLA, lower = better).**

| pool | greedy-FIFO | committed (deterministic) | **llm-committed** |
|---|---|---|---|
| 6 | 69.5 / 70.2 | 71.9 / 47.5 | 71.9 / **51.7** |
| 8 | 46.9 / 53.8 | 53.1 / 23.0 | **49.2** / **25.1** |
| 12 | 21.9 / 26.3 | 21.9 / 14.5 | 21.9 / **9.9** |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 13 — Incentives: the committed-auction is gameable; a flat budget does NOT fix it

**Date:** 2026-06-18

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 14 — the SUPPLY agent (headroom reservation): regime-gated win; model size matters (compressed)

**Date:** 2026-06-18/19 · `pins/supply_sim.py`; lever = reserve R GPUs from best-effort so a

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 15 — MIXED malleable+rigid: malleability-AWARE reservation recovers the util cost (compressed)

**Date:** 2026-06-19 · `simulate_mixed` (phi = malleable fraction) reproduces `simulate_rigid` at

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 16 — Uncertainty as a first-class signal: quantile forecasting → uncertainty-sized safety margin

**Date:** 2026-06-19

**Result.**

| metric | persistence | quantile P50 |
|---|---|---|
| nMAE_mean (accuracy gate) | 0.072 | **0.066** |
| gpu_util MAE | 6.26 | **5.40** |
| cpu_util MAE | 42.5 | **35.8** |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 17 — the LLM demand agent decides the hedge from uncertainty; "fix the decision, not the model" (compressed)

**Date:** 2026-06-19 · `llm_agent.llm_margin`: from `(uncertainty, deadline, contention, tier)`

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

# Stage-2 DECIDER + Stage-1 on REAL TRACES — Exp 18-21 (compressed)

## Experiment 18 — LLMSched-style ILP vs the PINS auction (2026-06-22)

**1-D pool: the ILP TIES the auction.** Welfare-max on one divisible pool with non-increasing
curves is already solved by the auction's greedy fill (welfare exact-equal across pools, asserted
in `test_ilp.py`), and CBC costs ~150x more per round (8.5 ms vs 0.056 ms). Not worth it in 1-D —
which locates exactly where the ILP earns its keep: constraints the count-only auction cannot
express.

**2-D nodes + sticky co-location: the auction is structurally handicapped; the ILP is not.** The
count-blind auction strands 1.28 -> 5.37 GPUs/round to fragmentation as the cluster grows (2x8 ->
6x8); `ilp.allocate_placement` plans count+node jointly, migrates at bounded cost, ploss = 0 by
construction, util +7-12 pts, slowdown lower at every size. But welfare/SLA stay mixed — and
greedy+sticky still wins raw SLA at every cluster size: placement feasibility and scheduling
discipline are ORTHOGONAL axes (the Exp-11 stable-order lesson again). Hence the layered
architecture: the committed order decides WHO, the ILP decides WHERE. Single seed (directional).
`pins/ilp.py`, `pins/placement.py`, `pins/placement_sim.py`; tests 4+3 green.

## Integration architecture (locked 2026-06-25)

**LLMs reason/bid -> committed-auction decides (who / how many) -> ILP places (where) /
guarantees.** To build: the Stage-1->agent text bridge, the bounded two-sided protocol, and the
must-have single-LLM-both-objectives baseline — done in Exp 22-25.

---


# Stage-1→Stage-2 INTEGRATION — the locked pipeline, built and measured (Exp 22-30)

The three locked next-work items (1) text bridge, (2) bounded two-sided protocol, (3) single-LLM
baseline are now built (`pins/bridge.py`, `pins/negotiation_protocol.py`, `llm_joint` in
`pins/llm_agent.py`, `pins/two_sided_sim.py`), plus a placement extension (`pins/affinity.py` +
an `affinity` arg on `ilp.allocate_placement`). Design hinge preserved throughout: the LLM emits
only a **categorical level + justification**; deterministic code owns every GPU count, the
clearing, and the feasibility.

## Experiment 22 — first two-sided run (margin <-> reserve in ONE world) — PARTLY ARTIFACT (compressed)

**Date:** 2026-06-26 · `pins/two_sided_sim.py` merges the demand margin (Exp 16-17) and supply

**Findings that SURVIVED:** (1) **single-llm over-commits both levers** with no brake — the worst overall-SLA agent under contention (pool 6 @3b: util 87%, only 14.5/16 done); the negotiation's concession ladder is exactly the missing brake. (2) negotiated uniquely beats the floor at slack (pool 12: 31.2 vs 33.6) by restraint. **ARTIFACTS, corrected later:** the 96/74/49% negotiation fallback and the resulting "negotiated is byte-identical across 3b/14b" were caused by non-negotiable base demand polluting the margin table (fixed in Exp 24) on top of unrealistic flat-8 synthetic caps (fixed at the source in Exp 27) — with both fixed, fallback is 0% everywhere and the model tiers genuinely …

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 23 — LLM affinity hint + ILP placement: a knife-edge gated on model size (compressed)

**Date:** 2026-06-26 · the Exp-21 open door, executed. The LLM only CLASSIFIES each task's

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 24 — Contested-slice negotiation: the fallback was an ARTIFACT; the two-sided split now BEATS the single-LLM

**Date:** 2026-06-29

**Result — `negotiated` across agents (8-seed mean; SLA / prodSLA, lower = better; fb now 0% everywhere).**

| pool | rule | qwen2.5:3b | qwen2.5:14b |
|---|---|---|---|
| 6  | 89.8 / 95.4 | 91.4 / 95.4 | 89.8 / 95.4 |
| 8  | 71.1 / 71.3 | 73.4 / 71.3 | 71.1 / 71.3 |
| 12 | 34.4 / 33.1 | 35.2 / 38.0 | 35.2 / **33.4** |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 25 — The full locked pipeline end-to-end: the ILP guarantee makes the LLM's over-demand SAFE

**Date:** 2026-06-29

**Result (agents=qwen2.5:14b; SLA / prodSLA / util / ploss; lower SLA/prodSLA/ploss = better).**

| cluster | floor | floor+ILP | nego+sticky | pipeline |
|---|---|---|---|---|
| 2×8=16 | 82.8 / 45.3 / 96 / 0.17 | **82.0 / 42.2** / 97 / **0.08** | 82.8 / 45.3 / 93 / 0.60 | **82.0 / 42.2** / 94 / 0.53 |
| 3×8=24 | 71.9 / 23.5 / 92 / 0.62 | **70.3 / 21.0** / 94 / **0.20** | 73.4 / 29.7 / 89 / 1.35 | **70.3 / 21.0** / 91 / 0.97 |
| 4×8=32 | 58.6 / 6.6 / 88 / 0.93 | **57.0 / 6.6** / 90 / **0.60** | 58.6 / 6.6 / 85 / 2.16 | **57.0 / 6.6** / 87 / 1.63 |
| 6×8=48 | 32.0 / 5.4 / 79 / 1.59 | **28.9 / 2.3** / 80 / **1.08** | 34.4 / 7.5 / 78 / 3.04 | 30.5 / **2.3** / 80 / 2.21 |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 26 — reflective margin agent: NEGATIVE — reflection thrashes into a limit cycle (compressed)

**Date:** 2026-06-30 · `pins/reflective_sim.py`. Can a weak model (3b) reflect its way to the

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 27 — REAL Stage-1 predicted GPU caps wired into the two-sided sim: the fallback problem vanishes, negotiation buys prod-SLA at slack cost

**Date:** 2026-07-01/02

**Result (rule-fallback tier; SLA / prodSLA / util / fb; lower SLA/prodSLA = better).**

| pool | policy | SLA | prodSLA | util | fb |
|---|---|---|---|---|---|
| 3 | no-llm | **57.0** | 62.3 | 90 | — |
| 3 | isolated | 63.3 | 52.9 | 84 | — |
| 3 | negotiated | 60.9 | **47.1** | 84 | 0% |
| 3 | single-llm | 64.1 | 52.9 | 83 | — |
| 4 | no-llm | **30.5** | 34.9 | 79 | — |
| 4 | isolated | 35.2 | **22.3** | 76 | — |
| 4 | negotiated | 35.9 | 28.2 | 77 | 0% |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 28 — TRACE REPLAY: real Alibaba gpu-v2020 jobs in the two-sided sim (the Exp-27 caveat closed)

**Date:** 2026-07-03

**Why.** Exp 27 sampled only the per-job GPU cap from Stage-1 predictions; arrivals, durations and
their correlations stayed synthetic (`make_workload`), and its own caveat asked for a
"demand/capacity-ratio-matched" real workload. This replays windows of REAL v2020 jobs so
**(arrival, duration, GPU demand) come JOINTLY from the trace** — the strongest workload realism
the harness has run on.

**Method.** New `pins/trace_replay.py`; the validated `two_sided_sim.simulate` + all four policies
imported, unmodified. One-time cache `data/alibaba-gpu-v2020/replay_jobs.csv`: 606,421 Terminated
GPU jobs (≥60 s) aggregated from `pai_task_table.csv` — arrival = min task start, dur = max end −
min start, demand = Σ(plan_gpu·inst_num)/25 quarter-GPU quanta (the Exp-27 quantum). Workload =
**one clock**: tick = 120 s for BOTH arrivals and durations (an early cut that affine-stretched a
3-minute burst of 16 consecutive arrivals onto the horizon while separately compressing durations
was discarded — it destroys exactly the joint arrival-vs-duration structure a replay is for).
Per seed: a random 10-hour window (~2,200 real arrivals; the 6.5k-GPU PAI cluster sees ~370/h),
**thinned** to 16 sampled jobs (thinning a near-Poisson stream preserves its statistics while
scaling load to a 1-2 GPU pool); work = real dur/tick clamped [1,60] (median 4.5, mean 17.7 —
genuinely heavy-tailed, ~12% clamped); caps = real quanta clipped at 8 (~80% of trace jobs are
below; mean 4.46, bimodal ¼/1/2 GPU — heavier than Exp 27's 2.35). What the trace does NOT have —
deadlines, urgency, tiers — keeps the exact `make_workload` recipe (seeded), so the only change vs
Exp 27 is the workload: a clean ablation. Pools {4,6,8}, 8 seeds (one window each), spike 0.6,
scale 3, rule/3b/14b tiers.

**Result (8-seed mean; SLA / prodSLA, lower = better; fb = 0% everywhere, all tiers).**

| pool | policy | rule | qwen2.5:3b | qwen2.5:14b |
|---|---|---|---|---|
| 4 | no-llm (floor) | **63.3** / 71.6 | 63.3 / 71.6 | 63.3 / 71.6 |
| 4 | negotiated | 65.6 / **69.6** | 66.4 / **66.4** | 65.6 / **69.6** |
| 4 | single-llm | 66.4 / 69.6 | 68.8 / 69.6 | 68.0 / 69.6 |
| 6 | no-llm | 57.8 / 70.5 | 57.8 / 70.5 | 57.8 / 70.5 |
| 6 | isolated | **54.7** / **63.5** | 57.0 / **59.4** | **53.9** / 60.4 |
| 6 | negotiated | 55.5 / 66.6 | **55.5** / 63.5 | 55.5 / 66.6 |
| 6 | single-llm | 55.5 / 66.6 | 58.6 / 65.7 | 56.2 / **59.4** |
| 8 | no-llm | 47.7 / 60.3 | 47.7 / 60.3 | **47.7** / 60.3 |
| 8 | negotiated | **46.1** / **54.6** | **46.9** / **50.1** | **47.7** / **56.7** |
| 8 | single-llm | 46.9 / 54.6 | 50.8 / 51.7 | 51.6 / 57.5 |

**Findings.**
1. **The Exp-27 headline SURVIVES the jump to real jobs.** fb = 0% at every pool/tier (the realistic
   demand/pool ratio holds); agents buy prod-SLA (pool 8: 60.3 → 50-57); `single-llm` still
   over-commits (worst done counts, util −4-7 pts, worst slowdown at every pool).
2. **Stronger than Exp 27: at moderate contention the agents now beat the floor on BOTH metrics.**
   Pool 8 `negotiated` wins overall SLA *and* prod-SLA at every tier (rule 46.1/54.6 vs floor
   47.7/60.3; 3b 46.9/50.1). On the synthetic workload the floor always won raw SLA; real
   heavy-tailed work gives the margins long at-risk jobs to save, so prod protection stops costing
   overall SLA. The price moved to best-effort **slowdown** (5.9 → 8.5-9.6 at pool 8) — the trade
   is real but now lives in latency, not deadline counts. *(Superseded by Exp 29: at 32 seeds the overall-SLA win and most of the slowdown price are within noise — the significant, surviving effect is the prod-SLA protection.)*
3. **The 3b≥14b inversion REPRODUCES on real jobs** (Exp-27 finding 5, which asked for exactly this
   check): pool-8 `negotiated@3b` 46.9/**50.1** vs `@14b` 47.7/56.7; pool-4 prodSLA 66.4 vs 69.6.
   And 14b does NOT rescue `single-llm` here (51.6/57.5 at pool 8, still worst) — on a harsher real
   workload even the big model can't safely run un-braked. The concession ladder, not model scale,
   is what makes the agents safe — now shown on real demand dynamics, not just synthetic. *(Superseded by Exp 29: at 32 seeds the 3b-vs-14b gap is a statistical tie; single-llm's penalty remains significant.)*
4. **At pool 4 (saturated, util 95%) the floor wins overall SLA** — every lever in this project
   needs slack; unchanged.

**Honest read / caveats.** Deadlines/urgency/tiers are still synthetic (v2020 has none — nothing to
replay); caps are the real *requests*, not the Stage-1 GBT predictions (per-job predictions keyed to
replayed jobs would need `predict_gpu.py` to emit job ids — the honest next step, putting prediction
ERROR into the loop); CAP_CLIP=8 truncates the heaviest ~20% of jobs and WORK_CLAMP=60 the longest
~12%; late-arriving long jobs can be horizon-infeasible (hits all policies equally); 8 seeds ×
1 window each. The slowdown cost of the agent policies (finding 2) deserves its own look — prod-SLA
is bought with best-effort latency, and on real heavy tails that price is larger than Exp 27 showed.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.trace_replay                          # rule tier (no Ollama)
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:3b
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b
```
`pins/trace_replay.py` (`load_trace`, `make_trace_workload` one-clock thinned windows, `sweep`);
cache `data/alibaba-gpu-v2020/replay_jobs.csv`; artifact `pins/results_trace_replay.json`.

## Experiment 29 — SEED SWEEP: the Exp-28 headline under 32-seed paired statistics

**Date:** 2026-07-06

**Why.** Exp 27's conclusion and Exp 28's finding 3 both said it themselves: the "negotiated beats
the floor on BOTH metrics" and "negotiated@3b ≥ @14b" claims rested on **8 seeds with no error
bars**, where differences like 46.9 vs 47.7 SLA are plausibly noise. Before either becomes a thesis
headline, run enough seeds to attach confidence intervals — and accept whatever survives.

**Method.** `trace_replay.py` upgraded, sim untouched: `sweep()` now records **per-seed** metrics
per (pool, policy), saves them into `results_trace_replay.json` **keyed by tier** (rule / 3b / 14b
runs no longer clobber each other), and prints **paired mean differences ± 95% CI**
(Student-t, df=31 → 2.042). Pairing is by seed and exact: within a tier all policies see the
identical workload + spike realisation; across tiers the same seed regenerates the same window, so
3b−14b is also matched. 32 seeds (= 32 random 10-h windows), pools {4,6,8}, all three tiers; the
same one-clock thinned-replay recipe as Exp 28. `--stats` reprints cross-tier comparisons from the
saved per-seed data. (`test_mechanism.py` green; the modified harness reproduces the Exp-28 8-seed
table digit-for-digit before scaling up.)

**Result (paired diff, negotiated − no-llm floor; − = negotiated better; * = 95% CI excludes 0).**

| pool | tier | ΔSLA (pts) | ΔprodSLA (pts) | Δslowdown |
|---|---|---|---|---|
| 4 | rule | +1.2 ±3.2 | −2.2 ±4.2 | −0.1 ±1.1 |
| 6 | rule | −0.4 ±1.9 | **−4.1 ±3.6*** | +1.3 ±1.3 |
| 8 | rule | −0.8 ±2.1 | **−4.7 ±3.2*** | +0.8 ±1.2 |
| 6 | 3b | −0.8 ±2.5 | **−7.5 ±5.6*** | +1.3 ±1.6 |
| 8 | 3b | +0.0 ±2.6 | **−7.7 ±5.1*** | +2.2 ±1.9* |
| 6 | 14b | −1.6 ±2.4 | **−5.2 ±4.1*** | +1.2 ±1.3 |
| 8 | 14b | −1.2 ±2.2 | **−5.3 ±4.0*** | +0.7 ±1.3 |

**Cross-tier, negotiated only (paired by seed, n=32):** 14b−3b = pool 4 SLA **−2.9 ±2.8*** (14b
better), pool 6 −0.8 ±1.1 ns, pool 8 −1.2 ±1.6 ns; prodSLA +2.2 ±3.8 / +2.3 ±4.7 ns (sign favours
3b, noise); pool-8 slowdown −1.5 ±1.4* (14b cheaper). 14b−rule: all metrics ns with tiny CIs
(±1.1–2.0) — the 14b agents essentially reproduce the deterministic ladder. 3b−rule: pool-6
prodSLA **−3.4 ±3.3*** (3b better), pool-4 SLA **+2.7 ±2.3*** (3b worse).

**Findings.**
1. **The prod-SLA protection is REAL.** Significant at pools 6 & 8 in every tier (−4 to −8 pts,
   CIs exclude 0). This is the negotiation's genuine, now statistically-backed contribution.
2. **"Beats the floor on overall SLA" did NOT survive.** No tier, no pool: negotiated's ΔSLA is
   never significant (best −1.6 ±2.4). Exp 28's pool-8 both-metrics win was an 8-seed fluctuation.
   The defensible claim is sharper and still strong: **prod-tier protection at no measurable
   overall-SLA cost** — the SLA⇄prodSLA *trade* of Exp 27 also vanishes into noise at n=32.
3. **The 3b≥14b inversion DISSOLVES — into indistinguishability, not defeat.** At contended pools
   6/8 negotiated@3b and @14b are statistically tied on both SLA metrics; at slack pool 4, 14b is
   marginally better. So the claim is **sufficiency, not superiority**: inside the bounded
   protocol a 3b model is indistinguishable from a 14b — while the un-braked `single-llm` still
   pays measurably (14b: pool-4 ΔSLA +4.9 ±4.6*, pool-8 Δslowdown +2.5 ±2.1*). "The brake makes
   the small model *as good as* the big one", not "better than".
4. **The slowdown price is smaller than Exp 28 suggested.** For negotiated it is significant only
   at 3b pool 8 (+2.2 ±1.9); rule/14b are ns everywhere. The "price moved to latency" caveat
   softens: at n=32 the latency price is mostly within noise too.
5. Tier nuance worth keeping: 14b ≈ the deterministic rule (near-zero paired diffs — it *concurs*
   with the ladder), while 3b *deviates* from it — sometimes profitably (pool-6 prodSLA), sometimes
   not (pool-4 SLA). Model scale buys conformity to the safe policy, not extra performance.

**Honest read / caveats.** n=32 windows from one trace, one contention recipe; deadlines/tiers
still synthetic (v2020 has none); the pool-4 saturated regime still favours the floor (unchanged).
The earlier "negotiated beats floor on both metrics" language in Exp 28 should be read as
superseded by this experiment. Multiple comparisons: ~21 CIs per tier are reported without
correction — the repeated, same-direction prodSLA effect is safe; isolated single stars (e.g.
3b−rule pool-4) should not be leaned on individually.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.trace_replay --seeds 32                          # rule tier + CIs
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model qwen2.5:3b
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model qwen2.5:14b
.venv/bin/python -m pins.trace_replay --stats                             # cross-tier paired stats
```
Per-seed data: `pins/results_trace_replay.json` (`tiers.<tag>.per_seed`); stats helpers
`paired_ci`/`t95`/`cross_tier_stats` in `pins/trace_replay.py`.

**Addendum (2026-07-06, follow-up e): 7b fills the ablation middle and lands inside the tie.**
32-seed real-caps sweep at qwen2.5:7b: `negotiated@7b` is statistically indistinguishable from
rule, 3b AND 14b on both metrics at every pool (all paired CIs include 0; largest |diff| 2.0 pts
vs a ±2.4 CI). The sufficiency claim is now the full flat curve — **rule ≈ 3b ≈ 7b ≈ 14b inside
the protocol** — while the un-braked `single-llm@7b` still pays clearly (pool 4: SLA 78.5 vs
floor 67.8, util 82%, 12.6/16 done, worst slowdown 15.8). Model scale changes nothing the brake
hasn't already fixed, at any of three sizes.

## Experiment 30 — PREDICTION ERROR IN THE LOOP: agents negotiate over Stage-1 *predicted* demand

**Date:** 2026-07-06

**Result (paired by seed, n=32; * = 95% CI excludes 0).** Cost of prediction error (pred − oracle, same windows; + = worse):

| pool | policy | ΔSLA (pts) | ΔprodSLA (pts) |
|---|---|---|---|
| 4 | no-llm floor | +5.3 ±3.2* | +5.2 ±5.1* |
| 4 | negotiated | +4.1 ±2.6* | +4.1 ±5.2 |
| 6 | no-llm floor | +6.6 ±3.0* | +8.2 ±5.6* |
| 6 | negotiated | +5.7 ±3.2* | +10.1 ±6.5* |
| 8 | no-llm floor | **+8.8 ±3.8*** | **+9.7 ±5.4*** |
| 8 | negotiated | **+6.1 ±3.3*** | +5.0 ±6.2 |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 31 — UNCERTAINTY SIZES THE REQUEST: agents ask for a quantile of the predicted interval

**Date:** 2026-07-07

**Result (paired by seed, n=32; * = 95% CI excludes 0).** Quantile vs the P50 request, negotiated policy (+ = quantile worse):

| pool | p10 − p50 | p90 − p50 |
|---|---|---|
| 4 | ΔSLA **+5.3 ±4.3*** · Δutil −10.5* | ΔSLA −1.6 ±3.5 · ΔprodSLA **−6.7 ±6.3*** |
| 6 | ΔSLA **+10.7 ±4.6*** · Δutil −14.1* | ΔSLA +0.0 ±3.8 · ΔprodSLA −5.4 ±6.0 |
| 8 | ΔSLA **+17.2 ±6.4*** · ΔprodSLA **+14.5 ±8.9*** | ΔSLA −1.6 ±4.0 · Δslow **+3.1 ±1.6*** |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 32 — THE INCENTIVE LAYER: per-user budgets + contested-tick claim pricing

**Date:** 2026-07-07

**Findings.** 1. **The vulnerability, restated as best response:** without an incentive layer, lying is individually profitable at every pool (−14..−24 pts* net for the deviator, −21..−39* for its best-effort jobs) and dumps +6..+10* violation pts on other users' prod jobs. 2. **The layer works, through the designed channel:** at 120-lump the net deviation gain is statistically zero at every pool, because the lie now significantly damages the deviator's OWN prod jobs (+15*) — the purse the lying best-effort jobs drain is the purse the user's critical jobs need. Externality internalised; victim harm goes ns. 3. **Incentive compatibility has a price, and it's a monotone frontier:** the honest …

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 33 — THE LLM USER AGENT vs THE TARIFF: does a self-interested LLM discover honesty?

**Date:** 2026-07-07

**Result (user-0's own violation rate, paired diffs ±95% CI; − = agent better).**

| | vs truthful, unpriced | vs liar, unpriced | vs truthful, priced |
|---|---|---|---|
| qwen2.5:3b | **−11.7* / −12.5* / −14.1*** | +12.5* / +10.2* / 0 | +2.3 / −4.7 / −2.3 (ns) |
| qwen2.5:14b | **−4.7* / −5.5* / −7.0*** | +19.5* / +17.2* / +7.0* | **+0.0 / +0.0 / +0.0** |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 35 — RETARGET STAGE-1: predict what the submission script does NOT already say

**Date:** 2026-07-08

**Why.** The GBT track's target, `plan_gpu`, is a USER-DECLARED field — it sits in the
submission script next to `plan_cpu`/`plan_mem`, so at decision time the scheduler already
has it. Predicting it is imputation, not demand prediction (the standing Exp 30/31 caveat).
The genuinely unknown-at-submission quantities in the v2020 trace are the task's **runtime**
(task table `end_time − start_time`) and its **actual usage** (`pai_sensor_table`,
downloaded here: 1.06 GB, 82% task coverage). Rule adopted: **declared fields are features,
not targets** — `plan_gpu` moves into the feature set for every new target.

**Method.** `predict_gpu.py` gains `--target {plan_gpu,runtime,gpu_util,gpu_mem}`; one
harness, same quantile-GBT machinery, same by-job split, same gbt-full-vs-gbt-num gate. The
`plan_gpu` default is byte-identical (re-ran it: metrics and `pred_job_gpu.csv` match Exp 30
exactly, so Exp 30/31 stay reproducible). New targets: `runtime` = Terminated GPU tasks,
y = end−start (732,691 tasks, median 615 s); `gpu_util` = mean `gpu_wrk_util` over the
task's workers; `gpu_mem` = max `max_gpu_wrk_mem` (both: inner join sensor→task on
(job_name, task_name), 850,068/1,037,085 tasks, zeros KEPT — idle-GPU tasks are the
over-provisioning signal). Features = [inst_num, plan_cpu, plan_mem, **plan_gpu**] + the
same semantic tags (gpu_type, task_name, workload).

**Result (test split, ~183k–213k tasks; gate = does gbt-full beat gbt-num).**

| target | global floor | gbt-num | gbt-full | gate |
|---|---|---|---|---|
| runtime (s) | MAE 5140 · rho −0.01 · w2x 22% | MAE 4600 · rho 0.48 | **MAE 4497 · rho 0.55 · w2x 41%** | PASS +2.2% MAE |
| gpu_util (%) | MAE 10.8 · rho −0.04 · w2x 3% | MAE 8.71 · rho 0.53 | **MAE 8.34 · rho 0.60** | PASS +4.2% MAE |
| gpu_mem (GB) | MAE 2.16 · rho −0.05 · w2x 23% | MAE 1.75 · rho 0.55 | **MAE 1.60 · rho 0.63 · w2x 53%** | PASS +8.5% MAE |

**Findings.**
1. **All three unknown-at-submission targets are predictable from the submission bundle**
   (rho 0.55–0.63 vs a rank-dead floor) and the semantic-tags gate passes on every one —
   the Stage-1 claim now stands on targets a scheduler could not simply read off the script.
2. **Runtime is the hardest** (within-2x only 41%) — consistent with the scheduling
   literature; the P10–P90 interval (width ≈ 12k s vs median 615 s) is doing honest work here,
   which is exactly what the Exp 31 hedge machinery wants as input.
3. **The declared-vs-actual gap is enormous and now measured in-trace:** median request is
   50% of a GPU, median actual utilization is **0.22%**; 55.9% of GPU-requesting tasks use
   <1%; spearman(plan_gpu, actual util) = **0.23**. The user's declaration barely tracks
   reality — this is the over-provisioning premise the whole PINS pitch rests on, and it is
   also Exp 32/33's unpriced over-claiming observed in the wild at trace scale.
4. plan_gpu-as-feature + tags predict actual usage far better than plan_gpu alone (rho 0.60
   vs 0.23) — a supply agent using this predictor can *discount* inflated requests, the
   Stage-1→incentive-layer bridge.

**Honest read / caveats.** Usage aggregation hides worker-level heterogeneity (mean/max per
task); sensor join covers 82% (missingness plausibly biased toward short tasks); runtime is
Terminated-only (survivor bias vs Failed); logRMSE/within-2x on usage targets are distorted
by the (deliberately kept) near-zero truths — MAE/rho/coverage are the meaningful columns;
intervals are plain quantile regression, not conformal. **Stage-2 is NOT rewired:**
trace_replay still negotiates over plan_gpu predictions (`pred_job_gpu.csv` unchanged);
moving the replay world to usage-based demand changes the truth definition (a job that
requested 4 GPUs but uses 0.1 should arguably not *need* 4 quanta to progress) and is the
natural next experiment.

**Reproduce.**
```bash
cd Research
.venv/bin/python data/fetch_alibaba_gpu.py --tables pai_sensor_table   # 1.06 GB, once
.venv/bin/python -m pins.eval.predict_gpu --target runtime
.venv/bin/python -m pins.eval.predict_gpu --target gpu_util
.venv/bin/python -m pins.eval.predict_gpu --target gpu_mem
.venv/bin/python -m pins.eval.predict_gpu                               # plan_gpu, unchanged
```
`--target` + target-aware `build_features` in `pins/eval/predict_gpu.py`; results in
`pins/eval/results_{gpu_runtime,gpu_util,gpu_mem}.json`.

## Experiment 36 — THE USAGE WORLD: Stage-1 usage prediction wired into the two-sided negotiation

**Date:** 2026-07-08

**Why.** Exp 35 retargeted Stage-1 to what the submission script does NOT say and measured
the wedge: median declaration 50% of a GPU vs median actual utilization 0.22%, rho 0.23.
But trace_replay still negotiated over plan_gpu — both the requests AND the truth. This
experiment moves the replay world onto the retargeted prediction: a job's TRUE need
(progress denominator) becomes its measured usage, and the only thing that varies between
arms is what the agents request — the plan_gpu DECLARATION (today's practice), the Stage-1
P50 USAGE PREDICTION, or the usage ORACLE. This is the right-sizing question end-to-end:
does predicting actual demand, instead of trusting the user's ask, buy system value through
the negotiation?

**Method.** `predict_gpu --target gpu_util` now exports `pred_job_usage.csv` (209,336 test
jobs: p10/p50/p90 predicted usage quanta + usage truth, same `sum(util·inst_num)/25` recipe
as replay_jobs.csv). `trace_replay --truth usage` swaps `true_cap_map` to usage quanta
(floor 1, clip CAP_CLIP); `--caps real|predicted|oracle` picks the request = declaration /
prediction / truth. One csv carries pred+truth so all arms share one key set — windows, and
therefore seeds, stay perfectly paired across the three arms (NOT with plan-world tiers,
whose window restriction differs). 83% of jobs truly need ≤1 quantum; the median declaration
is ~4. Rule tier + qwen2.5:3b tier (fb=0% everywhere — the LLM really decided), 32 seeds,
pools {4,6,8}. Plan-world paths byte-untouched.

**Result (paired by seed, n=32; * = 95% CI excludes 0).**

Declared − predicted (the value of right-sizing; + = declaration worse), negotiated policy:

| pool | rule | qwen2.5:3b |
|---|---|---|
| 4 | ΔSLA **+13.1 ±6.6*** · Δutil +19.5* · Δslow +3.3* | ΔSLA **+17.0 ±6.2*** · Δutil +12.9* · Δslow +5.1* |
| 6 | ΔSLA **+8.2 ±6.0*** · Δutil +27.5* · Δslow +2.0* | ΔSLA **+14.1 ±5.0*** · Δutil +24.2* · Δslow +2.9* |
| 8 | ΔSLA **+5.3 ±4.8*** · Δutil +29.6* · Δslow +1.4* | ΔSLA **+9.2 ±4.0*** · Δutil +27.4* · Δslow +1.6* |

(The greedy floor shows the same or bigger gaps: +8..+18 SLA pts*. Note Δutil's sign: the
DECLARED arm's higher "utilization" is hoarding-by-construction — quanta granted to jobs
that cannot convert them into progress — while its SLA and slowdown are strictly worse.)

Predicted − oracle (cost of the GBT's remaining error), negotiated: rule +2.7ns/+5.5*/+6.4*
SLA pts (prodSLA +3.6/+5.8 ns, +9.0*); 3b +2.3ns/+2.7*/+3.7* (prodSLA +6.2ns/+5.0*/+5.8*).

Negotiated − floor WITHIN each request world (does negotiation still earn its keep?):

| world | rule | qwen2.5:3b |
|---|---|---|
| declared | SLA −4.9*/−3.1ns/−4.9* · prodSLA −8.4..−12.4* | SLA −6.2*/−2.5ns/−5.3* · prodSLA **−12.0..−19.7*** |
| predicted | SLA ns/ns/−2.0* · prodSLA ns/ns/−3.1* | SLA **−4.9/−6.8/−6.2*** · prodSLA **−5.7/−6.8/−5.2*** · util +7..+9* · slow ≤0 |
| oracle | SLA ns · prodSLA −7.7*/ns/−4.9* | SLA ns/−2.7*/ns · prodSLA −12.6/−5.5/−3.9* |

**Findings.**
1. **Right-sizing from the Stage-1 usage prediction dominates trusting the declaration** at
   every pool for every policy: −5..−17 SLA violation pts, slowdown halved at saturation
   (8.0 vs 4.7 ticks, rule negotiated pool 4; floor 8.9 vs 3.9), and 13–30 pts of the pool
   freed from hoarded grants. The Exp-35 wedge (declarations barely track usage) converts
   directly into scheduler value through the existing negotiation — no mechanism change needed.
2. **Prediction error still costs +2..+6 SLA pts vs the usage oracle** (prodSLA up to +9*) —
   the imperfect GBT recovers most of the declared→oracle gap. Better usage predictors keep
   a direct system-level payoff here (unlike the plan world, where Exp 31's hedging had
   already saturated the prod tier), but the bulk of the value is already banked at P50.
3. **The 3b LLM negotiation and the prediction are complements**: under predicted-usage
   requests, negotiated@3b beats its floor on BOTH overall SLA (−4.9/−6.8/−6.2*) and prodSLA
   (−5.7/−6.8/−5.2*) at every pool, with HIGHER productive utilization (+7..+9*) and zero or
   negative slowdown cost — the first world in this project where the negotiation wins every
   headline metric simultaneously. The rule tier's negotiated advantage mostly evaporates
   there (SLA ns at pools 4/6) — the LLM agents' state-dependent margins do real work that
   the fixed rule does not, echoing the Exp-24/27 protocol-vs-scale pattern.
4. Prod-protection compresses as requests get honest (decl −19.7* → pred −5.7* at 3b pool
   4) — Exp 31's hedge/reserve substitution reappearing: an accurate request already carries
   most of the insurance the reserve used to provide, and what negotiation adds shifts from
   tier protection to overall efficiency.

**Bugfix (2026-07-08, later the same day).** All numbers above are the CORRECTED 32-seed
values — the original run carried the window-reroll bug (see the bugfix note before Exp 38):
5/32 seeds silently reverted to the plan world, zeroing their decl−pred diffs. Corrections
strengthened the headline (e.g. 3b decl−pred +14.5/+12.5/+8.6 → +17.0/+14.1/+9.2*) and
shrank pred−oracle (those 5 seeds had been measuring plan-world prediction error).

**Honest read / caveats.** "True need = measured mean utilization" is the OPTIMISTIC
right-sizing assumption — it ignores GPU-memory residency and utilization burstiness (a job
using 30% on average may still need the whole device resident; `gpu_mem`/max-based truth is
one flag away and would shrink the wedge); usage quanta floor at 1 quantum; windows are
restricted to predict_gpu's gpu_util test jobs, so usage-world tiers pair only with each
other; 3b only (14b unrun); the declared arm's util is not comparable across arms (held ≠
productive); intervals still not conformal, and the quantile knob (`--quantile p90` etc.)
is wired but unexplored in the usage world.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_gpu --target gpu_util      # regenerates pred_job_usage.csv
.venv/bin/python -m pins.trace_replay --seeds 32 --truth usage --caps real       # declaration
.venv/bin/python -m pins.trace_replay --seeds 32 --truth usage --caps predicted  # Stage-1 P50
.venv/bin/python -m pins.trace_replay --seeds 32 --truth usage --caps oracle     # matched control
# + the same three with --llm --model qwen2.5:3b
```
`--truth` + `load_usage_quanta` + `true_map`/`declared` in `pins/trace_replay.py`; usage
export in `pins/eval/predict_gpu.py`; tiers `{rule,qwen2.5:3b}+{decl,pred,oracle}@usage` in
`pins/results_trace_replay.json`.

## Experiment 37 — THE MEMORY WORLD: does right-sizing survive the pessimistic residency truth?

**Date:** 2026-07-08

**Result (paired by seed, n=32; * = 95% CI excludes 0).** Declared − predicted (value of right-sizing; + = declaration worse), negotiated policy:

| pool | rule | qwen2.5:3b |
|---|---|---|
| 4 | ΔSLA **+10.9 ±5.9*** · Δutil +17.5* · Δslow +4.1* | ΔSLA **+15.2 ±5.9*** · Δutil +12.0* · Δslow +4.7* |
| 6 | ΔSLA **+6.6 ±5.4*** · Δutil +24.4* · Δslow +1.2* | ΔSLA **+11.5 ±5.3*** · Δutil +22.4* · Δslow +2.0* |
| 8 | ΔSLA +3.1 ±4.5 ns · Δutil +25.6* · Δslow +1.0* | ΔSLA **+7.2 ±4.4*** · Δutil +24.7* · Δslow +1.3* |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Bugfix — the window reroll dropped the world (Exp 36/37 tiers rerun, 2026-07-08)

`make_trace_workload`'s sparse-window reroll (`trace_replay.py`) forwarded `pred`/`oracle`
but DROPPED `true_map`/`declared` — so any seed whose first window had <16
prediction-covered jobs silently reverted to the PLAN world: truth became plan-quanta and
the "declared" arm requested the prediction. Deterministic check: exactly seeds
{1, 9, 10, 15, 27} rerolled in the usage/mem worlds — confirmed in the stored per-seed data
as decl≡pred byte-equal on those 5 seeds. Effect: decl−pred diffs diluted toward zero
(headlines were UNDERSTATED), pred−oracle contaminated with plan-world error (overstated).
Fix forwards everything; all 12 `@usage`/`@mem` tiers rerun at 32 seeds; both entries'
tables corrected in place (original values in git history). Plan-world tiers never touched
the reroll's dropped args and are unaffected. Lesson logged: a reroll/retry path must
forward the FULL world spec — the silent fallback hid for two experiments because the
contaminated arms still produced plausible numbers.

## Experiment 38 — TIME BELIEF: de-oracling the deadline signal with the Stage-1 runtime prediction

**Date:** 2026-07-08

**Result (negotiated policy, paired by seed, n=32).**

| contrast | pool | rule | qwen2.5:3b |
|---|---|---|---|
| blind − oracle | 4 | ΔSLA −1.4 ±1.5 · ΔpSLA −2.1 ±2.6 | ΔSLA −0.2 ±0.4 · ΔpSLA −0.5 ±1.1 |
|  | 6 | ΔSLA −1.6 ±1.6 · ΔpSLA −1.8 ±3.4 | ΔSLA −0.2 ±1.3 · ΔpSLA +2.6 ±3.3 |
|  | 8 | ΔSLA −1.4 ±1.5 · ΔpSLA −1.9 ±2.2 | ΔSLA −0.6 ±0.7 · ΔpSLA −0.4 ±0.9 |
| predicted − oracle | 4 | ΔSLA **+0.8 ±0.8*** · ΔpSLA +0.3 | ΔSLA −0.4 ±0.8 · ΔpSLA −0.3 |
|  | 6 | ΔSLA +0.6 ±1.2 · ΔpSLA +0.0 | ΔSLA +0.4 ±0.8 · ΔpSLA +0.9 |
|  | 8 | ΔSLA +0.8 ±1.2 · ΔpSLA +0.6 | ΔSLA −0.4 ±0.6 · ΔpSLA −1.5 |
| blind − predicted | 4 | ΔSLA **−2.1 ±1.6*** · ΔpSLA −2.4 | ΔSLA +0.2 ±0.9 · ΔpSLA −0.2 |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 39 — SUPPLY-SIDE TIME-TO-FREE: the last plausible slot for runtime prediction

**Date:** 2026-07-08

**Result (negotiated policy vs control, paired, n=32).**

| contrast | rule | qwen2.5:3b |
|---|---|---|
| ttf-oracle − control | ΔSLA −0.6/−1.0/−0.8 ns · ΔpSLA ns · Δutil +0.2..0.3* | **exactly 0 on every metric, every seed** |
| ttf-predicted − control | ΔSLA **−2.9*/−2.1*/−1.4ns** · ΔpSLA ns · Δutil +0.6..1.0* | **exactly 0** |
| ttf-predicted − ttf-oracle | ΔSLA −2.3*/−1.2ns/−0.6ns | exactly 0 |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 40 — MODEL-FAMILY ABLATION: llama3:8b vs qwen2.5:{3b,14b} on paired windows

**Date:** 2026-07-09

**Result (negotiated policy, paired vs own floor, n=8; pools 4/6/8).**

| model | dSLA | dprodSLA |
|---|---|---|
| qwen2.5:3b | +5.5ns / −0.8ns / **−3.9*** | −0.4 / −8.5 / **−12.2*** |
| qwen2.5:14b | +5.5ns / +2.4ns / −1.6ns | −0.4 / +2.5 / −3.8 (all ns) |
| llama3:8b | **+14.1*** / +3.1ns / +0.8ns | −1.1 / +0.4 / −1.7 (all ns) |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 41 — EASY BACKFILLING: the classical baseline, fuelled by the runtime prediction

**Date:** 2026-07-09

**Why.** research_plan.md's baseline row demands FCFS + EASY backfilling; Exp 38/39 closed
both negotiation-side slots for the runtime predictor and concluded its remaining role is
EASY's reservation/backfill estimate — the one classical scheduler that CANNOT run without
a runtime estimate. Two questions: does the negotiation beat the classical discipline, and
what does the GBT's prediction error cost EASY?

**Method.** New `simulate_backfill()` in `two_sided_sim.py` — a sibling loop, not a
policy (EASY is a different allocation DISCIPLINE: FCFS order, all-or-nothing grants at
the requested cap held to completion, tier/urgency-blind, head-of-queue reservation from
believed remaining runtimes, backfill only jobs that provably don't delay the reservation;
no margins ever, so no spike-absorption lever). Progress dynamics copied verbatim from
`simulate` — the discipline is the only difference. `trace_replay --baseline easy --time
predicted`: `easy-pred` believes the Stage-1 runtime P50, `easy-oracle` the true spiked
work. Floor reproduced exactly (60.9/45.3/41.4) → seed-paired with the Exp-40 tiers.
`--compare TIER/POL,TIER/POL` added for arbitrary paired tier contrasts.

**Result (paired, n=8; pools 4/6/8).**

| contrast | dSLA | dprodSLA | dslow |
|---|---|---|---|
| easy-pred − floor | **+11.7*** / **+11.7*** / +8.6ns | +10.1 / +8.5 / +8.8 ns | **+9.8*** / +2.9 / +2.9 |
| easy-oracle − floor | +1.6ns / +6.2ns / +3.1ns | ns | **+6.8*** / +1.6 / +0.9 |
| negotiated@3b − easy-pred | −6.2ns / **−12.5*** / **−12.5*** | −10.5ns / **−16.9*** / **−21.0*** | **−9.7*** / −3.1 / −4.2 |
| negotiated@3b − easy-oracle | +3.9ns / −7.0ns / −7.0ns | −0.1 / −14.4 / −18.5 ns | **−6.7*** / −1.8 / −2.1 |

**Findings.**
1. **EASY loses to everything here** — even the rigid no-llm floor beats it (all-or-nothing
   grants waste holes: util 73/70/65% vs the floor's 90/82/76%; FCFS head-blocking sends
   slowdown to 14.5 at pool 4). The workload is exactly EASY's bad case: elastic partial
   grants are legal and the floor exploits them.
2. **negotiated@3b beats easy-pred decisively** (SLA −12.5* at pools 6/8; prodSLA up to
   −21.0*) and nominally beats even easy-ORACLE everywhere but pool-4 SLA. The negotiation
   claim survives its first classical baseline.
3. **Prediction error costs EASY 5–10 SLA points** (pred vs oracle) — the first Stage-2
   slot where runtime-prediction quality MATTERS (Exp 38/39 found it null in both
   negotiation slots). P50 under-estimates of right-skewed runtimes let backfilled jobs
   overstay the head's reservation — the classical EASY failure mode, measured.
4. Honest framing for the thesis: the floor-vs-EASY gap is mostly the elastic-vs-rigid
   grant model, not intelligence; the fair sentence is "in an elastic-GPU world, the
   negotiated policy beats the classical runtime-estimate discipline even when that
   discipline gets oracle runtimes".

**Caveats.** EASY implemented for the single-phase trace workload (multi-phase
`make_workload` jobs would need a per-phase request schedule); conservative-backfill
variant untried; n=8.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --baseline easy --time predicted
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:3b+time-predicted/negotiated,easy+time-predicted/easy-pred"
```
Tier `easy+time-predicted`; `simulate_backfill` in `pins/two_sided_sim.py`.

## Experiment 42 — TABULAR Q-LEARNING: a learned policy in the LLM's own interface

**Date:** 2026-07-09

**Why.** The plan's "one learning-based scheduler" baseline, scoped (scope decision,
plan item 5) to the question the thesis actually needs: is the LLM's decision quality
just something a cheap learned table could match? Full DeepRM/Decima-style DRL owns a
different (whole-allocator) action space and weeks of training budget — out of scope,
defended in research_plan.md.

**Method.** New `pins/qlearn.py`: two Q-tables over EXACTLY the LLM's discretised
interface — `margin_state_key` states × hedge{none,some,heavy}, `reserve_state_key` ×
reserve{none,light,heavy}. Episodic Monte-Carlo (contextual bandit: every state-action
visited in an episode shares the return −(sla+prod_sla)), ε=0.2, α=0.1, 900 episodes on
trace windows from seeds 100+ (disjoint from eval 0–7), same world as eval (`--time
predicted`, pools 4/6/8 cycled). Unvisited states fall back to the deterministic rule at
eval. Greedy eval is deterministic (rerun byte-identical). 179 margin / 9 reserve states
discovered; return plateaus ~−1.15 (window-difficulty noise dominates).

**Result (paired, n=8; pools 4/6/8).**

| contrast | dSLA | dprodSLA |
|---|---|---|
| qlearn − floor | **+17.2*** / **+16.4*** / **+7.0*** | −0.7 / +7.5 / +0.8 ns |
| negotiated@3b − qlearn | **−11.7*** / **−17.2*** / **−10.9*** | +0.3 / −16.0ns / **−13.0*** |

**Findings.**
1. **The learned table is significantly WORSE than the rule floor at every pool** and
   loses to negotiated@3b by 11–17 SLA points*. In the same interface, with the same
   information, learned-from-returns ≪ rule ≈ LLM-negotiated.
2. Why it fails is the finding: episode-level returns are dominated by window difficulty,
   so ~45 noisy samples per state-action cannot rank three actions per state — the
   credit-assignment problem the rule (priors) and the LLM (language-encoded priors)
   simply don't have. This is the written defense for scoping full DRL out: the failure
   is structural to learning-from-returns at feasible sample sizes, and a policy-gradient
   agent on the same episodes would face the same variance.
3. Together with Exp 40/41 this completes the baseline triangle: negotiated@3b ≥ floor >
   {easy-oracle, easy-pred, qlearn} on SLA at contention, and negotiated is the only
   policy that also protects prodSLA (−12.2* at pool 8).

**Caveats.** Tabular MC is the WEAKEST reasonable learner (no bootstrapping, no function
approximation, no per-decision credit); a tuned contextual bandit with counterfactual
baselines might close some gap — the claim is "cheap learning doesn't match priors here",
not "RL can't". Reserve table has only 9 states (its lever is tiny, Exp 39). n=8.

**Reproduce.**
```bash
.venv/bin/python -m pins.qlearn                                    # train -> qlearn_table.json
.venv/bin/python -m pins.trace_replay --baseline qlearn --time predicted
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:3b+time-predicted/negotiated,qlearn+time-predicted/qlearn"
```
Tier `qlearn+time-predicted`; trainer/policy in `pins/qlearn.py`.

## Experiment 43 — GENERALITY OF "SMALL MODEL SUFFICES": TOST equivalence + the Exp-40 world at n=32

**Date:** 2026-07-10

**Result A — TOST on the base world (Exp 29/addendum data, negotiated, n=32).** Overall SLA: 3b≡14b within ±3 at pools 6/8 (pool 6 even ±2: 90%CI[−1.7,+0.1]); 3b≡7b within ±3 at all pools; 14b≡rule within ±3 on BOTH metrics at all pools. Pool-4 3b-vs-14b stays a real 14b edge (−2.9 ±2.8*, not equivalent) — the slack-regime exception stands. prodSLA between LLM sizes: CIs ±3.2–4.7, too wide to certify ±3-equivalence at n=32 — there the claim remains "no detectable difference", not "equivalent".

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 44 — SECOND FAMILY: gemma2:9b joins the paired predicted-time windows (Exp-43 caveat, family leg)

**Date:** 2026-07-10

**Result (negotiated, paired, n=32; dSLA / dprodSLA).** vs own floor: pool 4 +3.1 ±3.3 / **−10.0 ±7.0***; pool 6 +2.1 ±2.4 / **−9.7 ±7.4***; pool 8 −1.2 ±2.5 / **−7.2 ±6.5***. gemma−3b: SLA **+2.3*/+2.0*** at pools 6/8 (3b better, ~2 pts), pool 4 EQ±3; prodSLA ns. gemma−llama3: pool 4 SLA **−3.9*** (gemma better), pools 6/8 EQ±2; prodSLA −2..−4 ns. gemma−14b: prodSLA **−8.4*** at pool 4 (gemma better; −6.1/−4.9 same direction, pool-6 90%CI excludes 0), SLA +2.0* at pool 6, else ns/EQ.

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 45 — DYNAMIC CAP: telemetry-corrected allocation base (rule tier) (2026-07-13)

**Result (paired, n=32; dSLA / dprodSLA, negative = dynamic better).** dyn-oracle − static pred (negotiated): **−2.3* / −3.1* / −5.7*** SLA at pools 4/6/8 (prodSLA ns). Same compare on the FLOOR arm: **−3.1* / −4.3* / −7.2*** (pool-8 prodSLA −5.6* too). dyn-noisy − static pred: −1.4 / −1.6 / **−3.5*** SLA. dyn-oracle − static ORACLE-at-admission: dyn worse or equal (+1.8/+2.5/+0.4 SLA; pool-6 prodSLA +5.8*).

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 46 — SUB-3B LADDER + gemma2:2b: where "small suffices" stops (2026-07-13)

**Result (paired, n=32; dSLA / dprodSLA, negotiated arm).** vs own floor (pool 8): 1.5b **−2.7* / −5.6***; 0.5b +0.6 / −5.6 (ns), its single-llm arm HURTS (+6.6* SLA); gemma2:2b **−2.9* / −8.3***. vs 3b head-to-head: 1.5b SLA TOST-EQ±3 at ALL pools (±2 at 4/8) but pays prodSLA (+4.4* pool 6); 0.5b SLA WORSE at all pools (**+2.5*/+3.5*/+3.7***); gemma2:2b EQ±3/±2 at pools 6/8, worse at slack (+3.3* pool 4). vs deterministic rule ladder: 1.5b better (SLA −3.1*/−3.1* pools 6/8, prodSLA −5.9*/−4.3* pools 4/8); 0.5b SLA ns everywhere (EQ at 6/8) — no SLA value over the rule — though it still buys prod protection at slack (−12.1* pool 4); gemma2:2b better at contention (pool 8 −3.3*/−7.0*).

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 47 — SECOND TRACE (MIT Supercloud): the mechanism's applicability BOUNDARY (2026-07-13)

**Result (paired, n=32).** NOT a replication — a boundary. At pool 4 (1 whole-GPU slot) every 3b arm LOSES SLA to its own floor (negotiated **+7.2 ±3.7***; floor runs 100% util); pools 6/8 are a wash (negotiated +1.4/+2.1 ns; prodSLA −3.2/−2.4 ns, CIs ±7). 3b vs rule: worse at pool 4 (+3.7*), TOST-EQ±3 at 6/8. Rule tier vs floor: ns everywhere. **Slot-count control** (pools 16/24/32 = same jobs, 4-8 whole GPUs; json backed up — `--pools` is not in the tier tag): ALL deltas collapse to ~0 (|dSLA| ≤ 0.6 ±1.1, ns) — with enough whole-GPU slots and thinned load, the mechanism is INERT, not harmful.

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 48 — ALLOCATION QUANTUM at scale: whole GPUs vs quarter quanta, 30 GPUs / 500 jobs (2026-07-13)

**Why.** Exp 47's boundary finding ("the mechanism's value requires demand FINER than the
allocation quantum") came from a *different trace*, so quantum and workload were confounded;
its caveat also flagged the contended many-slot regime as untested (no n_jobs knob). And the
scale-up question (queued 2026-07-13) was still open: does the negotiated win survive 30
GPUs / hundreds of jobs? Exp 48 de-confounds and scales in one shot: the SAME v2020 jobs on
the SAME 30 physical GPUs, with only the smallest negotiable element changed — the trace's
native quarter-GPU quanta (pool 120 units) vs whole GPUs (pool 30 units; every sub-GPU job
rounds up to a full card, caps collapse {1,2,4,8}→{1,2}, physical demand +27%). New
`trace_replay.py` knobs: `--quantum {quarter,whole}`, `--n-jobs` (tier suffixes `+qwhole`,
`+nN`; `--quantum whole` is base-world-only; default path verified byte-identical, window
sampling untouched so all four arms share seeds/windows). Saturated regime: util 96–98%,
floor SLA 65–75%, n=32.

**Result (paired, n=32).** Negotiated beats its own floor in EVERY arm — but the whole-GPU
*world* itself fails, and no policy inside it recovers that:

| arm (30 phys GPUs, 500 jobs)   | negotiated vs floor dSLA | dprodSLA |
|---|---|---|
| quarter, rule (`rule+n500`)             | **−1.2 ±0.3\*** | **−3.2 ±0.9\*** |
| quarter, 3b (`qwen2.5:3b+n500`)         | **−1.7 ±0.6\*** | **−6.2 ±1.4\*** |
| whole, rule (`rule+qwhole+n500`)        | **−2.3 ±0.5\*** | **−7.1 ±1.3\*** |
| whole, 3b (`qwen2.5:3b+qwhole+n500`)    | **−1.6 ±0.5\*** | **−5.6 ±1.3\*** |

Cross-quantum (paired by seed, same jobs, same hardware): whole-GPU allocation costs the
floor **+9.9 ±2.2\* SLA pts, +11.8 ±2.9\* prodSLA, −70 ±14\* finished jobs**; the BEST
whole-GPU arm (negotiated@3b, 73.3%) is still **+8.2 ±2.3\*** SLA worse than the quarter
world's own *floor* (65.1%). Breaking demand into quarter quanta rescues what negotiation
cannot: negotiated recovers ~1.6 of the ~10 points whole-GPU packing loses.

**Findings.**
1. **The quantum's cost is a packing property of the WORLD, not a negotiation property.**
   Rounding sub-GPU demand (75% of v2020 jobs ask ≤1 GPU) up to whole cards burns ~10 SLA
   pts / 70 finished jobs that no allocation policy inside the whole-GPU world gets back.
   "The LLM negotiates harder" is not a substitute for a finer allocation quantum — the
   mechanism-design fix (fractional quanta) dominates any policy fix, the cleanest possible
   statement of the thesis's scope sentence, now measured on the headline trace itself.
2. **Exp 47's boundary is refined**: at 30 whole-GPU slots the negotiated delta is intact
   (−1.6..−2.3\*) — what killed Supercloud pool 4 was the hedge being 25–100% of the pool
   (1–2 slots), i.e. quantum coarseness *relative to pool size*, not whole-GPU-ness per se.
   Fractional quanta matter for what the WORKLOAD wastes; slot count for what the HEDGE costs.
3. **Scale-up TODO closed — the win survives and the protocol becomes load-bearing.** At
   500 jobs the arms finally separate: negotiated@3b is the ONLY 3b arm improving both
   metrics (−1.7\*/−6.2\*); isolated agents now HURT SLA (+1.2\*, they trade prodSLA by
   abandoning best-effort wholesale: done 336 vs 377) and single-llm is a disaster
   (+7.2\*/+4.2\*, worse still at whole quantum +9.5\*/+14.9\*). The two-sided protocol —
   not just LLM hedging — is what scales; Open-Q #5's control keeps losing harder as n grows.

**Caveats.** The whole-GPU arm's +27% demand inflation IS the effect under study (that is
what whole-GPU allocation does to fractional askers), but it means regime saturation also
rises (floor 75% vs 65%) — the negotiated-delta comparison across quanta is between regimes
as well as quanta. Deadlines/urgency/tiers stay synthetic (make_workload recipe); base world
only (request == truth); single pool per arm (tier tag excludes `--pools`).

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --n-jobs 500 --pools 120 --seeds 32
.venv/bin/python -m pins.trace_replay --n-jobs 500 --pools 30  --seeds 32 --quantum whole
.venv/bin/python -m pins.trace_replay --n-jobs 500 --pools 120 --seeds 32 --llm --model qwen2.5:3b
.venv/bin/python -m pins.trace_replay --n-jobs 500 --pools 30  --seeds 32 --llm --model qwen2.5:3b --quantum whole
```
Tiers `rule+n500`, `rule+qwhole+n500`, `qwen2.5:3b+n500`, `qwen2.5:3b+qwhole+n500` (n=32)
in `results_trace_replay.json`; cross-quantum deltas via paired_ci over per_seed (pool keys
differ, so `--compare` does not apply).

## PIVOT (2026-07-15) — reason-then-referee: the LLM decides, code demoted to evaluator

The goal is now to **beat the rule/ILP-guaranteed pipeline with a referee LLM that outputs
the allocation directly**. Rationale: no mathematical rule set covers every situation; the
allocator must reason flexibly about the situation in front of it. Demand/supply agents
submit statements (base + requested margin + why; requested reserve + why) → the referee
LLM decides → `check_allocation` **evaluates only** (violations → floor fallback, charged
to `fallback_rate`; code never repairs — otherwise "the LLM can allocate" and "the ILP
fixes the LLM" are indistinguishable and we have rebuilt LLMsched). Exp 22–48's pipeline
stays as the baseline arm. Win condition: feasibility is table stakes (rule arm is 100% by
construction); the win must show in outcomes, ideally on the ticks where the rigid rule
decides badly. Branch: `referee_allocator`; plan updated (Focus, Goal 1/1b, Phase 5).

## Experiment 49 — REFEREE SCENE EVAL: can an LLM allocate feasibly at all? (2026-07-15)

**Result.** - **Toy scenes:** 3b 0/3 feasible (budget-blind), 14b 2/3, 27b 2/3; a SELF-CHECK prompt line gets 14b to 3/3 but conservative. Every model computes `total_awarded` CORRECTLY, then fails to act on the ≤ comparison — **constraint enforcement, not arithmetic, is the failure**. - **Real v2020 scenes:** feasibility collapses to **0% at exact/shortfall pools for ALL chat models** (even with self-check); prodcov 1.0 + overcommit 5–8 GPUs ⇒ **chat LLMs won't say no under scarcity** — they serve everyone and blow the budget. - **deepseek-r1:32b flips it: 100% feasible at ALL pool factors incl. shortfall.** Not parroting the rule referee: 10/24 scenes differ while feasible (egalitarian …

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 50 — REFEREE IN-SIM: trace replay with the LLM as the allocator (2026-07-15/16)

**Result (v2020 base world, n=8, paired vs floor; SLA deltas in pp, lower better).**

| arm | pool 4 | pool 6 | pool 8 | fb |
|---|---|---|---|---|
| referee@3b        | −1.6 / −11.9 | +2.3 / +5.2 | 0.0 / −10.4 | 45–58% |
| referee@r1:32b    | +0.8 / −1.8  | **+0.0 ±3.0 / +0.0 ±0.0** | +0.8 ±7.1 / **−10.4 ±16.1** | **0%** |
| negotiated@r1:32b | +2.3 / −1.8  | +1.6 / +2.1 | +0.0 ±6.8 / −5.2 ±9.0 | 0% |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 51 — REFEREE PRECEDENT MANUAL, build + Phase B at 3b: grounding the author, and the small model won't take dictation (2026-07-16/17)

**Result 1 — the first self-authored manual was INERT (and why).** Phase A v1's entries looked plausible but never fired: its P1 matched **5 of 925 eval scenes**. Root cause: the reflection payload named the constant pool size `free_pool_gpus` (colliding with decision-time `free_gpus`) and decisions carried **no per-decision state at all**, so r1 wrote WHEN clauses over window constants and invented fields (`upcoming_prod_jobs`). Fix (`1d02ff8`): trace entries record `free_gpus` and `llm_reserve` (pre-fallback), the dedup signature includes the state, and the reflection payload is per-decision. Artifacts of the inert round preserved as `pins/*.exp51-p1.*`.

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 52 — THE MANUAL AT 14b: obedience achieved, content worth ~0 — vanilla 14b is the best arm (2026-07-17)

**Result (pool 8, paired vs floor, n=32).**

| arm | dSLA | dprodSLA | dutil | dslow | fb |
|---|---|---|---|---|---|
| **vanilla 14b**    | +1.0 ± 2.9 | **−9.0 ± 5.3\*** | −1.5 ± 1.8 | +0.5 ± 2.8 | 1% |
| 14b+manual-learned | +1.0 ± 1.9 | −6.4 ± 5.1\*     | +0.8 ± 1.5 | +1.7 ± 1.7 | 1% |
| (r1:32b, Exp 50)   | +0.4 ± 2.4 | −7.7 ± 5.4\*     | +2.2 ± 1.5\* | +1.3 ± 1.5 | 0% |
| negotiated@14b     | +0.0 ± 2.1 | −4.3 ± 4.1\*     | +3.0 ± 1.1\* | +0.3 ± 1.2 | 0% |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 53 — DOES THE 14b WIN GENERALIZE? Reseed + pool sweep: real (pooled n=64) but venue-bound (2026-07-17)

**Result 1 — reseed (pool 8, seeds 32–63, n=32).**

| arm | dSLA | dprodSLA | dutil | dslow | fb |
|---|---|---|---|---|---|
| referee@14b    | +0.8 ± 1.7 | −5.9 ± 7.5 | −1.6 ± 1.7 | −2.1 ± 2.5 | 1% |
| negotiated@14b | +0.2 ± 1.2 | −3.0 ± 7.0 | +2.1 ± 1.0\* | −1.1 ± 1.2 | 0% |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 56 — REFEREE ON A PREDICTED BELIEF: the 14b win survives the belief swap (2026-07-19)

**Result (pool 8, seeds 0–31, n=32, `--caps predicted --quantile p50`).**

| arm | SLA | prodSLA | util | slowdown | fb |
|---|---|---|---|---|---|
| no-llm (floor) | 53.9% | 59.1% | 80% | 8.73 | 0% |
| referee@14b    | 53.7% | 51.5%\* | 78% | 8.66 | 1% |
| negotiated@14b | 52.1%\* | 52.5% | 82% | 7.92 | 0% |

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 57 — METHOD REVISION vs THE LLMSCHED PIPELINE: per-tick invocation is load-bearing (2026-07-19)

**Question.** The LLMSched pipeline doc (referee_allocator branch) prescribes: Δ-trigger the
LLM only on large state changes (θ=0.15), serve routine ticks from an adapted cached
strategy (96.8% fast mode), and pipeline the decision against the previous epoch's snapshot.
Does the referee keep its prod-protection win inside that controller shell — and while we're
instrumenting, does the win survive the plan's other realism knobs (reallocation overhead,
sublinear scaling, token budgets)?

**New infrastructure (all default-off; defaults verified bit-identical to Exp 56 replays).**
- Churn probe: `churn_gpu`/`churn_jobs` per tick in METRICS (always on, free).
- `--realloc-cost F`: a job forfeits F of a tick's progress on any tick its allocation
  changed. `--alpha A` + `--alpha-norm {c0,unit}`: P(g)=g/(1+A(g-1)) scaling law, normalised
  at base demand (c0, default: alpha reprices margin only) or at P(1) (unit: the plan's
  literal law — also tightens every deadline; the two disagree in SIGN on world difficulty).
- Token meter: every ollama client runs through `metered_client()`; `llm_calls`/`llm_tokens`
  per (seed, arm) in per_seed. Cache hits never reach the meter -> MARGINAL cost. Cold cost
  via `PINS_DRY_LLM=1` + `PINS_CACHE=<throwaway>` (counts calls without calling).
- `--trigger delta`: role-indexed identity-free referee scene key (rulings transfer across
  jobs in the same roles). Only −19% cold calls: the scene SPACE is combinatorial, not the key.
- Controller shell `--theta T --extend` + `--stale one`: Δ = 0.4·supply-bucket-crossing +
  0.4·jobset-change (departures only under extend) + 0.2·new-prod-behind; fast mode
  re-executes the STANDING ruling (extend: arrivals get the ruling's per-tier exemplar in
  rule-3/4 order); infeasible standing ruling re-fires the trigger, never repaired.

**Findings (pool 8, caps predicted, p50).**
1. **Churn ≠ churn harm.** r1 referee re-tunes 1.52x the floor's GPUs/tick (+0.172±0.098*,
   n=8) — but pricing churn (rule arms, rc=0.5) WIDENS the referee's win (−7.1 -> −11.4):
   the floor's reallocations land where lost progress costs deadlines, the referee's don't.
2. **Alpha robustness, both normalisations (rule arms, n=8).** c0-norm worlds get EASIER
   (starvation cushioning dominates), unit-norm worlds get brutal (floor prodSLA 91.5% at
   0.3); the referee's advantage stays −7..−12 in every cell of both. Linear-scaling is not
   what the win rests on.
3. **Cold-cost anatomy.** Referee 42.1 cold calls/seed vs negotiated 8.9 (4.7x) — key
   granularity + combinatorial scenes, NOT decision frequency. Shell cuts consultations 7x
   (217->31 llm ticks/seed) but cold inference only 36%: trigger and cache de-duplicate the
   same repeats; the triggered ticks are the novel scenes.
4. **THE METHOD VERDICT (14b, n=32, seeds 0-31, paired).**
   | arm | dprodSLA vs floor | fb | llm ticks | paired vs A |
   |---|---|---|---|---|
   | A per-tick (Exp 56 method) | **−7.6 ± 4.8*** | 1.1% | 217 | — |
   | B shell θ=.15+extend       | −4.5 ± 3.2*     | 1.1% |  31 | **+3.08 ± 2.71 LOSS** |
   | C B + stale one            | −2.5 ± 2.6 ns   | 5.0% |  29 | +5.12 ± 5.84 ns |
   The pre-registered rule ("adopt most-featured arm with no significant paired loss")
   nominally admits C — **overruled**: C's ns comes from staleness inflating variance
   (point estimate WORSE than B's significant loss, fb 5x). Rules that reward noise lose.
   **Verdict: per-tick invocation is load-bearing. A stays the method; B is the documented
   budget variant (−4.5* at 1/7 the consultations, 1330 marginal tokens/seed); C rejected —
   one tick of staleness erodes the headline to noise.** The referee's value lives in
   reacting to the current tick — exactly what LLMSched's pipelined planner would remove.
5. **Exp 57a (r1 predicted-belief, INTERRUPTED at n=16, checkpoint
   `results_exp57_r1_p50_ckpt16.json`):** dutil **+2.5 ± 2.4*** — r1's spend-don't-hold
   mechanism survives the belief swap (14b: −1.6 same settings); dprodSLA −6.4 ± 8.0 ns,
   needs the remaining 16 seeds. Seeds 0-15 cached; resume with `--seed-start 16`.

**Caveats.** Method verdict is 14b/single pool/p50; r1 shell arms unmeasured (its standing
rulings may extend differently). Sensitivity findings are rule-arm, n=8 — the LLM-referee
realloc/alpha arms were pre-empted for the method test. alpha is uncalibratable from v2020:
sensitivity, never measurement. Shell 5%-invocation target NOT met (14%): free-pool
feasibility churn at tiny pools is real work, not cache-able routine.

**Ops.** (a) metered_client briefly recursed onto itself via blanket substitution — caught
by the live-call smoke test, invisible to fully-cached runs; keep that test before trusting
token numbers. (b) TaskStop on a sweep kills its nohup'd ollama too — restart before the
next arm. (c) git worktrees don't carry untracked .venv/data/eval-csvs: symlink them.
(d) This commit also carries the previously-uncommitted Exp 55 debate wiring and Phase A
stratify code that shared the same files.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -m pins.trace_replay --referee --llm \
  --model qwen2.5:14b --caps predicted --pools 8 --seeds 32                  # arm A
# + --theta 0.15 --extend            -> arm B     # + --stale one            -> arm C
PINS_DRY_LLM=1 PINS_CACHE=$(mktemp -u) ... --theta 0.15 --extend            # cold counts
.venv/bin/python -m pins.trace_replay --referee --caps predicted --pools 8 \
  --seeds 8 --realloc-cost 0.5                                              # churn pricing
MODEL=deepseek-r1:32b SEEDS=32 POOLS=8 WORKERS=4 ARGS="--referee --llm \
  --caps predicted --quantile p50" bash pins/run_parallel_sweep.sh          # resume 57a
```

### Exp 57b — RQ5 LATENCY: the per-tick 14b referee fits 3-minute epochs with 30x headroom (2026-07-19)

Cold-scene, uncontended, per-call wall time (arm-A config, throwaway cache, weights
pre-warmed; `scratchpad/latency_probe.py` via `pins.llm_agent.LATENCIES`):

| model | n | P50 | P95 | max |
|---|---|---|---|---|
| qwen2.5:14b (referee) | 133 | 4.03 s | 5.43 s | 8.33 s |
| qwen2.5:3b (statements) | 49 | 2.86 s | 3.88 s | 11.11 s |

A fully cold triggered tick (statements + ruling, sequential) is ~8-20 s against the 180 s
epoch; aggregate LLM duty cycle 0.7% of wall time (517 s of 14b across two 10 h windows).
**RQ5 passes at 14b without the budget variant** — the shell (Exp 57 arm B) is a cost knob,
not a feasibility requirement. r1:32b probe running (expected marginal-to-failing per call:
Exp 50 observed 1-3 min); r1 remains the quality-ceiling ablation, not the deployment story.
Outcome numbers from the probe runs are n<=2 — vehicles for call generation, not results.

## Experiment 57c–g — THE STRUCTURE LADDER: what each layer of the multi-LLM architecture buys (2026-07-20)

> **⚠ COMPARABILITY CAVEAT (audit 2026-07-28).** The ladder's base kept moving. `pins/referee.py`
> was modified in the same commit that reports Exp 57 (`479286e`, +152/−18) and again before each
> of the arms that follow: `8beb001` (57c–g, +11/−4), `ea6dc16` (58 build, +29/−5), `7d1cd6b`
> (59, +37/−21), `a06e739` (60/61, +39/−7). Each rung is validly paired **within its own run**, so
> no individual arm is retracted — but rungs measured on different dates are not guaranteed to sit
> on the same referee, and cross-experiment increments (e.g. 57c–g vs 59 vs 61) should not be
> subtracted from one another without re-running on a fixed base.
>
> Not affected by the decision-packet confound: the packet never entered the in-sim path (see the
> caveat on Exp 84).

**Thesis reframing (this session).** The question is now: *can debate-like structured
multi-LLM interaction improve resource-utilisation safety (low SLA)?* That turns the
referee-vs-negotiated comparisons into a LADDER — no arguments → one-shot statements →
statements+debate — each rung paired on the same seeds in the revised-method world
(qwen2.5:14b rules, 3b advocates, caps=predicted, pool 8, seeds 0-31), flanked by the
rivals. New arms: `--no-argue` (justifications stripped after gathering, |noarg cache tag)
and the 57g gated-debate composition. All defaults off; arm-A replay verified bit-identical
after every build.

**The ladder (dprodSLA / dSLA / dutil vs floor, n=32, 0 referee fallbacks in every run).**

| arm | dprodSLA | dSLA | dutil | note |
|---|---|---|---|---|
| no-llm floor | — | — | — | |
| single-ilp (57d) | −7.9 ± 5.3\* | −1.8 | +0.2 | ILP repaired **44%** of ticks |
| no-argue (57f) | −7.0 ± 3.9\* | **+4.9 ± 2.5\*** | **−5.3 ± 2.5\*** | 14.8/16 done — worst completions |
| referee (A) | −7.6 ± 5.0\* | −0.2 | −1.6 | fb 1% |
| **debate (57e)** | **−12.0 ± 7.0\*** | +0.8 | −2.5 ± 1.8\* | largest effect in project |
| gated debate (57g) | −9.1 ± 6.4\* | +1.0 | −2.9 ± 1.7\* | llm-ticks 33 vs 221 |
| negotiated | −6.6 ± 4.6\* | −1.8 | **+2.0 ± 1.0\*** | only +util arm |

**Findings.**
1. **The advocacy text contributes ~nothing to prod protection and everything to
   non-destructiveness** (57f paired vs A: prod +0.58 ns; sla **+5.08 ± 2.43\***; util
   **−3.64 ± 2.34\***). Numbers-only refereeing still shields prod (rules 3-4 are mechanical)
   but becomes a blunt tax: overall SLA significantly WORSE than no scheduler, util −5.3\*.
   The statements are the information channel that makes rule-5 skepticism discriminating.
2. **The debate round ~1.6×'s the protection** (−12.0\* vs −7.6\*), six starred peeks,
   trajectory −18.7→−15.3→−17.2→−15.0→−12.5→−13.1→−12.0. The INCREMENT over the plain
   referee is −4.37 ± 4.39 — **ns by 0.02 points at n=32**; n=64 reseed required before
   claiming it. Price: dutil −2.5\* (deliberation holds capacity).
3. **Gating the debate (Δ>0.15, 57g) is a token-economy device, not a cluster-efficiency
   device**: consultations −6.6× with ALL paired outcome diffs vs full debate ns
   (prod +2.84 ± 3.82, util −0.37 ± 0.86) — but util identical (−2.9\*): the standing ruling
   FREEZES the hold-back across quiet ticks. The shell that significantly hurt the plain
   referee (+3.08\*) costs the debate arm nothing measurable. First 57g attempt was INVALID
   (debate row missed the theta plumbing, replayed 57e byte-identically — caught by
   paired-diff=0.00 + shell_fast=0; quarantined in scratchpad, rerun verified fast=187/llm=33).
4. **Single-ilp @14b+pred (57d): outcome TIE with the referee** (paired −0.28 ± 3.92 ns) at
   44% ILP-repair rate vs the referee's fb 1%. The prior log/paper claim that propose+repair
   is inferior was WRONG (it was never logged; 3b tier −7.3\* was already competitive). The
   architecture contrast is autonomy/auditability, not outcomes.
5. **Robustness knobs transfer to the 14b LLM referee SPLIT (57c)**: α=0.3 survives
   (−6.2 ± 3.4\*, DiD vs α=0 +1.44 ± 3.94 ns) — linear scaling is not load-bearing;
   rc=0.25 loses significance (−4.2 ± 4.5, DiD +3.42 ± 4.09 ns, floor +1.9 vs referee +5.3
   absolute damage) — the rule-arm "pricing widens the win" did NOT reproduce; undecided,
   leaning adverse. 14b churn 1.23× floor (jobs 2.1×).
6. **No 14b configuration achieves +util AND low SLA.** Deliberation's util price is not
   avoidable by gating; negotiated remains the only +util\* arm. The both-signs candidate is
   r1's spend-don't-hold mechanism (Exp 50 +2.2\*, 57a +2.5\* at n=16) — Exp 55 (r1+debate,
   caps real, pre-registered) is IN FLIGHT serial (~7-10 h; referee row replayed from cache
   in 27 s as designed; r1 100% VRAM at NUM_PARALLEL=1).

**Accounting debts (open).** Debate token bill not captured (warmers pay cold calls in
throwaway processes; cache proxy ~37 debated rulings + ~96 rebuts/seed). Concession-rate
diagnostic needs a deterministic replay script (rebut cache stores revised LEVELS only).
Debate latency probe (--debate, 1 seed) pending. All multi-arm selection on seeds 0-31 —
any headline arm from this search must be CONFIRMED on seeds 32-63.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -m pins.trace_replay --referee --debate --llm \
  --model qwen2.5:14b --caps predicted --pools 8 --seeds 32          # 57e
# --no-argue -> 57f   # --debate --theta 0.15 --extend -> 57g   # --single-ilp -> 57d
bash pins/run_debate.sh                                              # Exp 55 (r1, caps real)
```

## Experiment 54 — FEW-SHOT vs VANILLA REFEREE at 14b: half-run, no contrast exists (2026-07-18)

**Date:** 2026-07-18

**Status.** **Incomplete.** The few-shot arm completed (qwen2.5:14b, `nothink`, caps=real, pool 8,
32 seeds): referee vs floor `dSLA +0.2 ± 2.0`, `dprodSLA −6.7 ± 5.0*`, `dutil +2.0 ± 1.7*`. The
paired vanilla run (`pins/exp54_vanilla_14b.log`) stopped at **20/32 seeds**, so the
few-shot-vs-vanilla contrast this experiment existed to measure was never computed.

Superseded in practice by Exp 52/53, which established vanilla 14b as the best arm and found the
manual's content worth ~0. **Do not cite a few-shot result** — there isn't one.

---

## Experiment 55 — THE PRE-REGISTERED DEBATE ROUND AT r1:32b (caps=real): both signs at last, and the round itself is EQUIVALENT to no round (2026-07-20)

Pre-registered before the 57c–g ladder was run (`pins/run_debate.sh`), finally landed:
`deepseek-r1:32b` rules, 3b advocates in BOTH the referee and debate rows (so the delta
isolates the round, not the advocate model), caps=real, pool 8, seeds 0-31. Serial,
NUM_PARALLEL=1, ~11.7 h for the debate row alone (41,992 s / 32 seeds ≈ 1,312 s/seed);
no-llm / referee / negotiated replayed from the Exp 50 cache in ~26 s each as designed.
0 referee fallbacks, 3,422 distinct decisions/transcripts.

| pool | policy | SLA | prodSLA | util | slowdown | fb | done |
|---|---|---|---|---|---|---|---|
| 8 | no-llm | 51.8% | 55.7% | 82% | 9.98 | 0% | 15.4/16 |
| 8 | referee | 52.1% | 48.0% | 84% | 11.24 | 0% | 15.4/16 |
| 8 | debate | 51.2%\* | 48.0%\* | 84% | 10.87 | 0% | 15.4/16 |
| 8 | negotiated | 51.8% | 51.7% | 83% | 10.25 | 0% | 15.4/16 |

```
referee      vs floor:  dSLA   +0.4 ± 2.4   dprodSLA   -7.7 ± 5.4*  dutil   +2.2 ± 1.5*  dslow   +1.3 ± 1.5
debate       vs floor:  dSLA   -0.6 ± 1.9   dprodSLA   -7.7 ± 6.1*  dutil   +2.3 ± 1.5*  dslow   +0.9 ± 1.1
negotiated   vs floor:  dSLA   +0.0 ± 2.2   dprodSLA   -4.0 ± 3.8*  dutil   +1.0 ± 0.8*  dslow   +0.3 ± 1.2
(* = 95% CI excludes 0, paired by seed, n=32)
```

**Findings.**
1. **THE BOTH-SIGNS ARM EXISTS.** r1:32b at caps=real is the first configuration to buy
   significant prod protection AND significant utilisation at once: referee −7.7\* prodSLA
   with **+2.2\* util**, debate −7.7\* with **+2.3\* util**. Every 14b configuration in the
   57c–g ladder paid util for deliberation (−1.6 to −5.3); r1's spend-don't-hold mechanism
   (Exp 50 +2.2\*, 57a +2.5\*) reproduces at n=32 and now carries the debate arm with it.
   This closes the "no 14b configuration achieves +util AND low SLA" gap noted in 57c–g —
   by changing the referee model, not the structure.
2. **The debate round adds NOTHING here, and that is now measured as EQUIVALENCE, not just
   ns** (paired debate MINUS referee, same tier, n=32): dprodSLA −0.1 ± 3.3, dutil +0.1 ± 0.7,
   dSLA −1.0 ± 1.9, and TOST rejects a ±3 pt effect on all three metrics
   (prodSLA 90%CI[−2.8,+2.7], util[−0.5,+0.6] — also EQ±2, sla[−2.6,+0.6]). Contrast the
   14b ladder, where the debate increment was −4.37 ± 4.39 (ns by 0.02, "n=64 required").
   **The cross-talk round's value is model-dependent, and at the strong-reasoner end it
   vanishes** — the ruling is already whatever the round would have converged to.
3. Carried caveat from the pre-registration, now load-bearing: advocates are 3b, a weak
   arguer (Exp 33, 51). A null round at r1 is partly "3b had nothing to say to a 32b model".
   The `--advocates qwen2.5:14b` rerun (~20 h, all rows cold) is the remaining way to
   separate "the round is decoration" from "the arguers were outclassed".
4. Overall SLA is flat everywhere (debate's 51.2% is starred against the floor but is a
   −0.6 ns paired diff at pool level); the win remains prod-tier protection plus util.

**Open.** Concession rate still not extracted (`referee._r0_gpus` holds the opening ask;
needs the deterministic replay script — same debt as 57c–g). Debate-vs-referee equivalence
is selection-free (pre-registered), so no seeds 32-63 confirmation is owed for THIS null,
but the both-signs headline should be confirmed on fresh seeds before it goes in the paper.

**Reproduce.**
```bash
# ollama MUST be OLLAMA_NUM_PARALLEL=1 (r1:32b spills to CPU otherwise, ~6x)
bash pins/run_debate.sh                                   # ~12 h, debate row only is cold
.venv/bin/python -m pins.trace_replay \
  --compare 'deepseek-r1:32b+referee/debate,deepseek-r1:32b+referee/referee'
```

## Experiment 58 (E1) — `--prev-input` + change-cost rule 6: BUILT, NEVER RUN (2026-07-20)

**Date:** 2026-07-20

**Status.** **Build only — no run log, no results JSON, no measurement.** Commit `ea6dc16`:
the referee receives the last **executed** allocation plus a change-cost rule (rule 6), so it can
price its own churn instead of re-deciding from scratch each tick.

The gating and stability work that followed (Exp 59 fast path, Exp 61 hard trigger) may have
exercised this path indirectly, but the change-cost lever was never isolated or measured under
this number. Open lead — cheap to test in the current harness.

---

## Experiment 59 — THE FAST PATH TRADES ITS DETERMINISTIC REPLAY FOR THE CHEAP AUCTION: util cost flips to a gain (2026-07-21)

**Findings.** 1. **Util flips sign against the documented 57g run** (same config, `--extend` instead): dprodSLA −9.1 ± 6.4\* / dutil −2.9 ± 1.7\* (57g) vs dprodSLA −8.3 ± 5.5\* / **dutil +1.8 ± 1.4\*** (this run). Protection is unchanged within overlapping CIs; utilisation moves ~4.7 points and crosses zero. **The util cost identified in 57c–g was a property of the frozen-replay fast path, not of gating itself** — a cheap live re-decision below θ recovers it. 2. **This is the second both-signs configuration in the project** (after Exp 55's r1:32b) — significant prod-tier protection AND significant +util together — but reached by an architecture change (what the fast path IS) rather than a …

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 60 — THE ARRIVAL SURGE: the referee's protection is a slack-regime property, and debate INVERTS under contention (2026-07-21/22)

**Findings.** 1. **The referee's prod-protection does not survive the surge.** dprodSLA −4.9 ± 5.2, non- significant, against −9.1\* (57g) and −12.0\* (57e, debate) in the calm pool-8 world. The point estimate roughly halves and the CI opens past zero. 2. **The util cost survives intact** (−2.7\* referee, −2.2\* debate). Under surge the LLM arms keep the whole bill and lose the benefit — the worst of both. 3. **`negotiated` is the only arm with both signs right here:** prodSLA −4.9\*, util +2.6\*, slowdown −0.7\*, and 15.2/16 jobs done vs the referee's 14.7. Note its prod point estimate is IDENTICAL to the referee's (−4.9) — what separates them is variance, not mean: the auction's spread is …

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 61 (E3) — HARD-TRIGGER DELIBERATION: gate the round, not the ruling — 6.4x fewer rounds, protection intact, and the round is EQUIVALENT to no round (2026-07-22)

User's latency proposal, mapped onto the architecture: call the LLM only when the situation
changes seriously, restrain its input/output, hold the discussion to one round, and reuse the
cache. Three of the four were already true — statements are categorical and cached per
discretised bucket, `rebut()` is a single monotone round (asks may only shrink, so termination
is structural), and `--no-argue` (57f) already measured that stripping the arguments further
COSTS SLA +4.9\* vs floor. The one live idea was selective invocation, and 57c–g had applied it
to the wrong layer: it froze the RULING (measured load-bearing — arm B +3.08\*) to save the
cheap cached statements, when the actual bill is the rebuttal round, which fires once per job
per tick.

**Implementation.** New `--hard-trigger` (`referee.make_policy_referee`, requires `--debate`):
the rebuttal round fires only on a hard trigger — prod arrival, free-GPU bucket crossing, a job
newly behind deadline, or a fallback last tick — while the referee still rules EVERY tick on
live numbers (T^R=1 preserved). Trigger state is kept in separate `h_*` keys from the theta
shell's, so the two gates compose. Un-debated ticks never set `_debated`, so `_scene_key` drops
the `|dbt` justification hash and those ticks REUSE the plain referee arm's cached ruling —
identical inputs, identical ruling, no extra inference. New `shell_debate` per-seed counter
makes the saving measurable rather than asserted. Tier suffix `+hardtrig`.

**Run.** qwen2.5:14b, caps=predicted, pool 8, seeds 0-31, `--referee --debate --hard-trigger`.

| pool | policy | SLA | prodSLA | util | slowdown | fb | done |
|---|---|---|---|---|---|---|---|
| 8 | no-llm | 53.9% | 59.1% | 80% | 8.73 | 0% | 15.3/16 |
| 8 | referee | 53.7% | 51.5%\* | 78% | 8.66 | 1% | 15.2/16 |
| 8 | debate | 53.7% | 52.6% | 79% | 8.19 | 1% | 15.2/16 |
| 8 | negotiated | 52.1%\* | 52.5% | 82% | 7.92 | 0% | 15.5/16 |

```
referee      vs floor:  dSLA  -0.2 ± 1.9   dprodSLA  -7.6 ± 5.0*  dutil  -1.6 ± 1.7   dslow  -0.1 ± 1.0
debate       vs floor:  dSLA  -0.2 ± 2.1   dprodSLA  -6.5 ± 5.3*  dutil  -1.0 ± 1.5   dslow  -0.5 ± 1.0
negotiated   vs floor:  dSLA  -1.8 ± 2.2   dprodSLA  -6.6 ± 4.6*  dutil  +2.0 ± 1.0*  dslow  -0.8 ± 0.8
(* = 95% CI excludes 0, paired by seed, n=32)

gated debate MINUS its own (fresh, same-tier) referee row:
  dSLA +0.0 ± 1.1   dprodSLA +1.1 ± 1.5   dutil +0.6 ± 1.2   dslow -0.5 ± 1.2
  TOST: dSLA 90%CI[-1.0,+1.0] EQ±3 EQ±2   dprodSLA 90%CI[-0.2,+2.4] EQ±3   dutil 90%CI[-0.4,+1.6] EQ±3 EQ±2

deliberation load (mean per seed):
  referee   ticks_ruled 217.3   debate_rounds  0.0
  debate    ticks_ruled 217.8   debate_rounds 33.8  (15.5% of ticks)   tokens 6813
```

**Findings.**
1. **The gate works and is free.** 217.8 ticks ruled per seed, only 33.8 deliberations —
   a **6.4x cut in rebuttal rounds** — with prod protection fully intact (−6.5\* gated,
   −7.6\* referee, both in the same band as Exp 59's ungated −8.3\*/−6.4\*) and fallback
   unchanged at 1%. Selective invocation at the DELIBERATION layer costs nothing, unlike
   57g's gating of the ruling.
2. **The util cost disappears** (−1.0 ± 1.5, ns; −1.6 ± 1.7, ns) where 57c–g measured −2.5\*
   for the ungated round. The plan pre-registered util unchanged at ~−2.5\* and named an
   improvement as the FALSIFICATION of the "deliberation holds capacity" mechanism. That
   falsification is what happened: cutting 85% of the deliberation recovered the util cost
   without touching protection, so the capacity was not being held by the deliberation.
3. **The round is equivalent to no round.** Gated debate vs its own referee row is EQ±2 on
   SLA and util, with the prod increment +1.1 ± 1.5 (ns, if anything worse). This reproduces
   Exp 55's r1:32b result on a second model and strengthens the emerging story: across 14b
   (here), r1:32b (Exp 55), and the 57e increment (ns@32), **the cross-talk round has never
   been shown to carry the result** — 57e's −12.0\* vs floor is the referee stage's effect,
   not the round's.
4. **`negotiated` remains the only both-signs arm** (prodSLA −6.6\*, util +2.0\*). Same
   pattern as Exp 60's surge: the LLM arms protect the prod tier but never win utilisation;
   the deterministic auction does both.

**Caveat — the gated-vs-UNGATED-debate increment could not be computed.** The plain
`qwen2.5:14b+pred+referee` tier has no `debate` per-seed rows at all
(`{'no-llm': 32, 'referee': 32, 'negotiated': 32}`) — overwritten by later ungated runs before
this session's tier-suffix fix landed, the same trap documented on Exp 59. The increment above
is against this run's own fresh referee row, which is clean and answers "does the gated round
still buy anything"; the token/latency saving vs an ungated debate run is quantified by the
`shell_debate` counter, not by a paired outcome diff. A fresh ungated `--debate` reseed would
make the direct contrast computable.

**Standing conclusion after Exp 60 + 61.** Effort spent on the reasoning layer is hitting
diminishing returns: the round is equivalent to no round, gating it is free, and the auction
still beats both LLM arms on utilisation in every regime tested. The binding constraints are
elsewhere — see the calibration note below.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.trace_replay --referee --debate --hard-trigger \
  --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32
.venv/bin/python -m pins.trace_replay --compare \
  'qwen2.5:14b+pred+referee+hardtrig/debate,qwen2.5:14b+pred+referee+hardtrig/referee'
```

## Experiment 62 — CONTENTION/SLACK CALIBRATION: negotiation does not pay anywhere in the sweep (2026-07-22)

**Date:** 2026-07-22

**Method.** Calibration sweep, not a hypothesis test: pools {8, 16, 32, 64} × slack {1, 2, 4, 8}×,
22–179 jobs per window, `negotiated` against the no-LLM floor. Run to choose the operating point
the headline experiments would report.

**Findings (SLA / prodSLA, * = 95% CI excludes 0).**

| slack | pool | no-llm | negotiated |
|---|---|---|---|
| 1× | 8 | 68.2% / 73.3% | 66.2%* / 68.7% |
| 1× | 32 | 62.5% / 54.8% | 62.0% / 53.8% |
| 1× | 64 | 58.5% / 44.2% | 57.9% / 42.6% |
| 2× | 8 | 45.7%* / 44.0% | 45.7%* / 41.2%* |
| 4× | 8 | 34.1% / 29.1% | 32.1%* / 23.0%* |
| 4× | 16 | 34.4% / 20.4% | 33.2%* / 17.3%* |
| 8× | 8 | 27.7% / 20.8% | 26.8%* / 16.7%* |
| 8× | 16 | 27.3% / 11.8% | 27.0%* / 11.4%* |

- `negotiated` is at or below the floor on overall SLA in **every** cell, and the production-tier
  gap **widens as slack grows** (−2.8 at 4× pool 8, −4.1 at 8× pool 8, both significant).
- Utilisation is flat (87–89%) throughout, so the loss is rationing, not throughput.

**Reading.** A calibration input, not a headline: it says the negotiation-era mechanism has no
regime in this sweep where it pays, which is consistent in direction with Exp 70–73 where the
deterministic market beats it outright. It is also one input to reporting pools {4, 6, 8} rather
than 32/64 — the large-pool cells are slack-dominated and separate nothing.

**Reproduce.** `pins/exp62_calib.log`, `pins/exp62_calib_slack.log`.

---

## Experiment 63 — THE ADMISSION LEVER: RUN. Deterministic admission pays, the referee does not (2026-07-22 built, run 2026-08-03)

**Date:** built 2026-07-22 (`9c1d127`, `438df61`), run 2026-08-03, results committed `e978039`,
heading corrected 2026-08-04 (it read "BUILT, NEVER RUN" for a day after the results landed).

**Status.** **Measured**, n=32 seeds, 8 quarter-GPU quanta (2 GPU), truth=plan / caps=predicted.
Logs `pins/exp63_base.log`, `pins/exp63_admit.log`; results `pins/results_exp63.json`.

| arm | SLA base → +admit | prodSLA base → +admit | util base → +admit |
|---|---|---|---|
| no-llm (floor) | 24.9% → 24.9% | 16.3% → 16.3% | 88% → 88% |
| **referee** | 25.3% → **29.9%** | 9.8%* → 12.2% | 81% → **66%** |
| **negotiated** (deterministic) | 23.4%* → **16.8%*** | 13.7% → **8.4%*** | 89% → 90% |

Paired vs the floor, with admission on:

```
referee      dSLA  +5.1 ± 4.0*   dutil -22.6 ± 4.5*   duseful -23.2 ± 4.4*   dregret +29.9 ± 5.1*
negotiated   dSLA  -8.1 ± 4.2*   dutil  +2.1 ± 1.2*   duseful  +2.7 ± 1.3*   dslow    -9.2 ± 3.8*
```

**Verdict — pre-registered branch 2** (`docs/superpowers/specs/2026-07-30-exp63-admission-lever-design.md:99`):
*H2 pays, H3 null → admission belongs in the deterministic core, next to the market.* The lever is
real and large — it is the second-biggest SLA effect in the project after Exp 99's urgency rule —
but it is a **mechanism** result, not a reasoning one. Wiring `admission_plan` into
`make_policy_market` is the pre-named follow-up.

**The referee arm got worse on every axis that matters**: +5.1 SLA points, and it paid for that by
running the cluster at 66% utilisation (−22.6 pts, regret +29.9) — it is admitting fewer jobs and
calling the resulting idleness a win. Cost: **94.1 calls, 102,812 tokens, 634.6 s per seed** versus
0/0/0.0 for both deterministic arms.

**Branch 4 (censoring artefact) does not fire.** `negotiated+admit` finishes *more* jobs than the
floor, 23.8/24 vs 23.5/24, so the SLA gain is not bought by dropping work.

**Qualification — do not over-read this.** The in-sim referee here runs the **weak pre-packet
interface** (spec §7, `:111`); Exp 81's "cannot fund" failure mode is exactly what an
un-budgeted reviewer does. This licenses *"this referee configuration handled admission badly"*,
**not** *"an LLM cannot do admission"*. A packet-equipped re-run is the open version of this arm.

**What it closes.** §5 of `paper/pins_gated_draft.md` scoped an admission-moving exception as
future work and predicted a reasoning gain there. The lever exists, it moves SLA by 6.6 points —
and it does so with **zero tokens**. The paper's §5 must be rewritten accordingly.

---

## Experiment 64 — REBASELINE AT pool 32 / slack 4 / 89 jobs: killed mid-run, twice (2026-07-22)

**Date:** 2026-07-22

**Status.** **No results.** Both variants — plain and `--fastneg` (θ=0.15) — show the identical
failure: the floor arm completes and the referee arm dies at its first seed.

```
32gpu no-llm      32/32 seeds |    37s elapsed ~    0s left
32gpu referee      1/32 seeds |     1s elapsed ~   35s left
```

Consistent with the login-node CPU reaper: 89 jobs/window with per-tick LLM calls exceeds the
background budget within minutes. A re-run needs a batch node or a smaller window; the scale
question it was meant to answer is still open, though Exp 48 covers 30 GPUs / 500 jobs by another
route.

---

## Experiments 65–67 — THE ROUND-2 STRUCTURE LADDER: everything is null, and the instrument is why (2026-07-22)

**Date:** 2026-07-22

**Question.** Does any reasoning *structure* beat any other on exception scenes? Three passes over
the round-2 hard-case suite (54 scenes: infeasible, contradiction, unmodeled, corrupt, ambiguous,
and 6 routine controls), qwen2.5:14b.

**Findings — Exp 65, perspective ladder.**

| category | ilp | rule | single | referee | debate | selfcons |
|---|---|---|---|---|---|---|
| infeasible | 5/9 | 5/9 | 4/9 | 3/9 | 5/9 | 2/9 |
| contradiction | 4/9 | 4/9 | 4/9 | 5/9 | 5/9 | 4/9 |
| unmodeled | 3/6 | 4/6 | 4/6 | 4/6 | 3/6 | 4/6 |
| corrupt | 5/9 | 5/9 | 8/9 | 8/9 | 6/9 | 8/9 |
| ambiguous | 4/9 | 4/9 | 4/9 | 5/9 | 4/9 | 5/9 |
| routine (control) | 6/6 | 6/6 | 4/6 | 5/6 | 4/6 | 5/6 |
| **TOTAL** | **30/54** | **31/54** | **33/54** | **35/54** | **31/54** | **33/54** |

**Exp 66, argumentation 2×2** (does letting agents argue matter?): ilp 30/54, rule 31/54,
single 33/54, single-noarg 32/54, referee-noarg 30/54, referee 35/54.

**Exp 67, critic arm** (a reviewer that only objects): ilp 30/54, rule 31/54, referee 36/54,
critic 31/54.

**Every pairwise McNemar across all three passes is null.** The smallest p in ~30 tests is
**0.238** (ilp vs referee, Exp 67); referee vs critic p=0.267, referee vs debate p=0.424,
single vs referee p=0.688.

**Why this matters more than the nulls do.** The suite is not measuring exception handling. The
deterministic baselines — a plain ILP and a rule referee, neither of which can read text at all —
score 30/54 and 31/54, within noise of every LLM arm. A suite a text-blind baseline half-passes
cannot detect a text-reading capability. Round 3 was built in response: 31 primary
**text-dependent** scenes on which the rigid floor is **0/31**. That drop, 31/54 → 0/31, is the
evidence the instrument was replaced rather than the hypothesis re-shopped.

The `routine` control row already shows the specificity cost that Exp 88/89 later measured at
scale: ilp and rule score 6/6, the LLM arms 4–5/6. Reasoning invoked where nothing is wrong makes
things worse, and it was visible here first.

> **⚠ CONFOUND — these three predate the decision packet and must not be read as verdicts on the
> structures they test.** All of Exp 65–67 ran on **2026-07-22**. The structured decision packet
> was introduced the following day in Exp 82 (`b77d3fa`, 2026-07-23), and it is what took
> infeasible rulings from 11–16/31 to **0/31**. These runs therefore carry *two* confounds at
> once: an insensitive suite **and** the weak pre-packet interface. This is the same trap as Exp
> 79, whose "parallel multi-agent is dead" reading was overturned once the interface was repaired
> and a sequential debate on the packet reached 43/81 (Exp 89).
>
> **Consequently the following are UNTESTED, not refuted:** self-consistency (`selfcons`), the
> argumentation ablation (`single-noarg` / `referee-noarg`), and the **critic** arm. None has ever
> been run on the strong packet, against the round-3/4 suite, or under a matched budget. Exp 67's
> critic is the most interesting of the three — it scored 31/54 on the weak interface, while the
> structurally similar debate went on to win decisively once the packet existed. Re-running these
> three arms in the Exp 88/89 harness is a cheap, well-posed experiment.

**Reproduce.** `pins/exp65_perspective_14b.log`, `pins/exp66_2x2_14b.log`,
`pins/exp67_critic_14b.log`; transcripts in `pins/results_hardcases_perspective.json`,
`results_hardcases_2x2_14b.json`, `results_hardcases_critic_14b.json`.

---

## Experiment 68 — THE ELEVATED PLAN'S MEASUREMENT LAYER: occupancy is not productive work, and every margin arm buys waste (2026-07-22)

**What changed (build).** The elevated research plan (`referee_allocator/Elevated_Multi_Agent_GPU_Scheduling_Research_Plan.pdf`,
uploaded to `origin/referee_allocator`) makes three demands on the simulator that the Exp 22–63
line never satisfied. All three are now implemented in `pins/two_sided_sim.py`:

1. **§3.2 counterfactual progress model.** A second, *primary* scaling law
   `p_j(g) = 1 − exp(−kappa_j·g)` with `kappa_j = K/c0` — each job saturates on the scale of its
   own base demand, so `rate == 1` at `c0` and deadlines stay comparable across laws
   (`--law sat --kappa K`, default `K=2`). The Exp 57 Amdahl law is now the §3.3 *robustness*
   model (`--law amdahl`, default, byte-identical to every result through Exp 63 — verified by
   re-running HEAD in a worktree: SLA/prodSLA/util/slowdown/wait/done all match exactly).
2. **§4 utilisation decomposition.** `util` (= U_alloc, occupancy) is now reported alongside
   **`u_useful`** (GPU-equivalents of progress actually produced, `min(g, c0·rate)`) and
   `u_waste = U_alloc − U_useful`. Allocated GPUs a job cannot convert into progress no longer
   count as utilisation.
3. **§14 allocation regret.** A per-tick oracle `A*_t` maximising useful progress subject to
   `sum_j g_j ≤ G`, `g_j ≤ ceil_use_j`. Both laws are concave, so greedy-by-marginal-gain is the
   *exact* maximiser (heap, O(n + G log n)). `regret = (oracle − actual)/oracle` over the run.
   The oracle is unconstrained by rigidity, so this regret prices rigid incumbency too — an
   upper bound, and reported as one.

**Result (rule tier, v2020 replay, 16 jobs/window, pools 4/6/8, n=32 paired seeds, both laws).**

| law | pool | arm | dutil | **duseful** | **dregret** | dprodSLA |
|---|---|---|---|---|---|---|
| amdahl | 4 | negotiated | −1.1\* | **−1.3\*** | **+1.8\*** | −3.7 |
| amdahl | 6 | negotiated | +0.1 | −0.3 | **+0.6\*** | −2.9 |
| amdahl | 8 | negotiated | **+1.0\*** | +0.3 | +0.1 | **−4.0\*** |
| sat | 4 | negotiated | −1.7\* | **−2.4\*** | **+2.6\*** | −1.3 |
| sat | 6 | negotiated | −0.4 | **−1.7\*** | **+1.7\*** | −2.0 |
| sat | 8 | negotiated | +0.8 | **−0.8\*** | **+1.1\*** | **−3.4\*** |

(paired vs the `no-llm` floor; \* = 95% CI excludes 0. `isolated` and `single-llm` behave the
same or worse throughout.)

**Reading.**
1. **The utilisation gains reported since Exp 22 are partly waste.** At pool 8/amdahl the
   negotiated arm's headline `dutil +1.0*` shrinks to `duseful +0.3` (ns) once occupancy is
   separated from progress — the margin GPUs are held but not converted. Under the plan's
   primary law the sign flips outright: `duseful −0.8*` at every pool. **No arm buys useful
   utilisation anywhere in this sweep.** Every past "+util\*" claim in this log needs the
   qualifier "occupancy" until re-run with `u_useful`.
2. **Regret is the sharper instrument.** It is significantly WORSE than the floor for every
   LLM-shaped arm in 5 of 6 cells — a signal SLA and util both hid, and exactly the plan's
   argument for making normalised regret a primary outcome.
3. **The prod-tier protection survives the law change.** `dprodSLA −3.4*` (sat, pool 8) vs
   `−4.0*` (amdahl, pool 8): the project's core finding passes the §3.3 both-models test. The
   utilisation story does not.
4. **Contention flips the sign of the waste.** At pool 4 (saturated) margin costs both
   occupancy and useful work; at pool 8 (slack) it at least converts. This is the same
   slack-regime boundary Exp 60 found for the referee.

**Reproduce.**
```bash
for L in amdahl sat; do PINS_RESULTS=pins/results_exp68_$L.json \
  .venv/bin/python -m pins.trace_replay --seeds 32 --pools 4,6,8 --law $L > pins/exp68_$L.log; done
```

**Open (not done here).** The plan's other demands are untouched: §13 equal-inference-budget
protocol (the single-LLM baseline is still unmatched on tokens), §6–8 explicit bid/ask +
credits + seriousness gate, §12 quality-aware cache with false-reuse rate, §15 resize
cooldowns, §17 ρ/λ sweeps. And the headline LLM arms (14b referee, r1:32b) have NOT been
re-run under `--law sat` or scored on `u_useful`/`regret` — the table above is rule-tier only.

## Build 68b — THE REST OF THE ELEVATED PLAN'S MEASURABLE LAYER: lateness, starvation, resize physics, seriousness gate, Holm (2026-07-22)

Built while Exp 69 held the GPU. Every knob defaults to the pre-Exp-68 behaviour, so no
existing tier's numbers move; each was verified live by sweeping it.

| plan § | what landed | flag / metric | verified by |
|---|---|---|---|
| §5 | normalised lateness `L_j = max(0,(C_j−d_j)/(d_j−r_j))`, censored at the horizon | `lateness` | reported per seed |
| §15 | resize cost `c0 + c1·|Δg|`; resizing **cooldown** K | `--resize-c1`, `--cooldown K` | cooldown 0/2/5 moves prodSLA 55.3→60.6→60.6% |
| §16 | starvation rate (wait ≥ 30 ticks), max wait, **Jain** share fairness; ageing bonus `φ·min(1, waited/30)` on grant priority | `starved`,`wait_max`,`jain`, `--phi` | φ=20/200 at pool 4 moves prodSLA 66.0→74.1% |
| §8 | **seriousness score Γ_t** = mean(SLA risk, clearing ambiguity, uncertainty, starvation, job-set churn); deliberation fires on Γ>θ *or* a hard trigger | `--gamma THETA` | rebuttal rounds/seed 192 → 24.5 → 24 at θ = 0.1/0.5/0.9 |
| §18 | **Holm correction** over the whole vs-floor family, with the plan's t-vs-Wilcoxon selection by Shapiro | printed under every pool | see below |
| §17 | sensitivity grid driver (ρ, caps regime, λ, resize, cooldown, φ, granularity, law) | `pins/run_sensitivity.sh` | smoke at 4 seeds |

**Two things this immediately exposed.**

1. **Holm is brutal, and it should be.** The vs-floor block is 18 simultaneous tests (3 arms ×
   6 metrics). At n=8 the two `dregret*` stars vanish: *"NOTHING survives correction"*. Every
   single-pool `*` in this log's history was reported uncorrected — the n=32 headline results
   (e.g. Exp 53's −7.5\*, Exp 68's `duseful`) need re-reading against their own families
   before the paper quotes them.
2. **Γ_t is a usable throttle, not a binary.** θ=0.5 cuts rebuttal rounds 8× (192→24.5/seed)
   and lands exactly where `--hard-trigger` alone sits (24), i.e. the score reproduces the
   pre-registered hard triggers as a *special case* and lets us tune below them. Whether the
   outcome survives the throttle is Exp 71.

**Not built, and why.** §6 explicit bid/ask formulas and §7 virtual credits both replace the
decision mechanism rather than measure it — `mechanism.clear()` already clears marginal-bid
curves, but the bid *content* (`αΔp+βR_SLA+γW_wait−δC_resize`) and the credit purse would
change every arm's behaviour, so they need their own pre-registered experiment, not a
side-build. §12's quality-aware cache needs a design decision first: `Q_i` is defined on the
outcome of a decision, and in a 300-tick simulation a decision's regret is only attributable
at the end of the run — so either the cache scores entries offline (post-hoc, not deployable)
or on a proxy available at decision time. §10's diversity ablation needs temperature plumbing
in the advocate reasoners; `--samples` already provides the single-LLM half of it.

## Build 68c — THE QUALITY-AWARE CACHE ON A DECISION-TIME PROXY (plan §12), and what it costs (2026-07-22)

**The design decision (made, not deferred).** §12 scores a cache entry with
`Q_i = a·U_useful − b·SVR − c·C_resize − d·R_invalid` — all *outcome* quantities. In a
300-tick simulation an outcome is attributable only at the end of the run, so an
outcome-scored cache is not deployable: a live scheduler cannot wait for the job to finish
before deciding whether to trust a cached ruling. We therefore score on **decision-time
signals only**:

* `fill` — share of the free pool the ruling put to work, capped at each job's usable
  parallelism (the immediate stand-in for `U_useful`: over-award is visible at once),
* `invalid` — the deterministic validator's verdict (the plan's own `R_invalid`),
* `churn` — share of jobs whose award moved vs the last executed allocation (`C_resize`).

`Q = fill − invalid − 0.3·churn`, clipped to [0,1]. Retrieval is the plan's
`R_i = cos(z_t, z_i)·Q_i·exp(−0.01·age)`; adaptation maps awards by **job category**
(`tier|deadline`), never by raw job id; the candidate is then **re-validated, never repaired**,
and a rejected candidate falls through to a fresh ruling. `pins/qcache.py`, opt-in via
`--qcache THR`.

**Result (rule-tier referee, pool 8, n=4, v2020).** The threshold does exactly what the plan
predicts, and the false-reuse metric earns its keep:

| threshold | reuse rate | **false-reuse rate** | SLA | prodSLA |
|---|---|---|---|---|
| — (no cache) | — | — | 28.1% | 37.3% |
| 0.95 | 2% | 35% | 28.1% | 37.3% |
| 0.85 | 6% | 40% | 28.1% | 37.3% |
| 0.70 | 13% | 43% | 28.1% | 37.3% |
| 0.50 | 15% | 64% | 28.1% | 37.3% |
| 0.30 | 20% | 64% | 29.7% | **43.6%** |

1. **Reuse is free until it isn't, and the false-reuse rate says where the edge is.** Down to
   θ=0.50 the outcomes are bit-identical to no cache while 15% of rulings are served from it.
   At θ=0.30 outcomes degrade (+6.3 prodSLA pts) — and the false-reuse rate had already
   jumped to 64% one step earlier. **Hit rate would have told us nothing here: it rises
   smoothly across the whole range.** This is the plan's §12 argument, demonstrated.
2. **A ~35-40% false-reuse floor even at θ=0.95** is a caution about the decision-time proxy
   itself: a ruling can be locally sensible (fills the pool, validates, doesn't churn) and
   still travel badly to a similar-looking scene. The proxy cannot see that; only outcomes
   can. Reported as the honest limitation of the deployable design.

**Scope.** Rule tier only, so this measures the cache's DECISION quality, not its token
saving — the point of caching at the LLM tier. `--qcache` at 14b is the follow-up, and there
the fast-tick counter (`shell_fast`) turns reuse into a real bill reduction.

## Experiment 69 — THE EQUAL-BUDGET CONTROL, FIRST LOOK: the centralised LLM does more with 41% of the referee's tokens (2026-07-22)

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 70 — §13 SETTLED: the referee does not earn its budget. A cheaper centralised control ties it, and the auction beats both (2026-07-22)

**Setup.** 14b, caps=predicted, pool 8, **n=32 paired seeds**, v2020 replay, cold cache
(`pins/cache_exp70.json`), single-LLM control at `--samples 10` (median-aggregated).

```
   8  no-llm        53.9%   59.1%    80%     75%     10%      8.73   22.7    0% 15.3/16
   8  referee       52.5%   50.2%    78%     74%     13%      8.53   24.0    1% 15.2/16
   8  negotiated    52.3%*  52.5%    81%     76%      9%      8.10   21.3    0% 15.5/16
   8  single-llm    54.1%   50.1%*   81%     74%     12%      9.43   24.5    0% 15.3/16

      referee      vs floor:  dSLA -1.4 ±2.0  dprodSLA -8.9 ±5.1*  dutil -1.2 ±1.6   duseful -1.9 ±1.2*  dregret +3.2 ±1.6*
      negotiated   vs floor:  dSLA -1.6 ±2.2  dprodSLA -6.6 ±4.6*  dutil +1.8 ±1.1*  duseful +0.8 ±0.8   dregret -0.7 ±1.2   dslow -0.6 ±0.6*
      single-llm   vs floor:  dSLA +0.2 ±3.2  dprodSLA -9.0 ±6.3*  dutil +1.4 ±1.9   duseful -0.9 ±1.5   dregret +2.1 ±1.9*
      Holm (18 tests): referee/regret p=0.006[t], referee/prodSLA p=0.024[wilcoxon], negotiated/util p=0.046[t]

      arm           calls/seed  tokens/seed  wall s/seed
      referee             36.7        24240        154.5
      negotiated           3.7         1639          9.6
      single-llm          30.9        10902         89.9
```

**1. The multi-agent structure buys nothing over a centralised LLM.** Paired head-to-head,
referee − single-llm on prod SLA is **+0.1 ± 4.6 pts, p=0.965** (Wilcoxon) — indistinguishable,
and the referee is the more expensive side of the tie. On utilisation the referee is
significantly WORSE (−2.5 ± 1.3, p<0.001); on useful utilisation and regret the two are
TOST-equivalent within ±3. There is no metric on which the referee beats the control.

**2. And it still isn't a fair fight — in the referee's favour.** `--samples 10` put the
control at 10.9k tokens/seed against the referee's 24.2k: the control ties while spending
**45%** of the budget. §13's protocol exists to stop a multi-agent arm winning on budget; here
the multi-agent arm does not win *despite* a 2.2× budget advantage, so funding the control
further is unnecessary to reach the conclusion. (Matching exactly would need `--samples ~22`;
it could only strengthen this.)

**3. The deterministic auction dominates both, at 6.8% of the referee's bill.** `negotiated`
is the only arm that is *positive* on utilisation (+1.8\*, Holm-surviving), non-negative on
useful utilisation (+0.8), the only arm with NEGATIVE regret (−0.7, i.e. better decisions
than the floor), significantly faster (dslow −0.6\*), and it still protects the prod tier
(−6.6\*) — for 1,639 tokens/seed against the referee's 24,240.

**4. What survives multiplicity.** Three of eighteen: `referee/regret +3.2` (worse than the
floor), `referee/prodSLA −8.9` (better), `negotiated/util +1.8` (better). Note that the
referee's Holm-surviving results point in *both* directions: it protects production by making
measurably worse allocations.

**Standing conclusion.** Across Exp 60, 61, 68, 69 and now 70 the reasoning layer has not
paid for itself on any outcome the plan treats as primary. The referee's prod-tier protection
is real and reproducible — but it is reproduced by one cheap centralised LLM call, and it is
bought with regret and utilisation the auction does not have to pay. The defensible thesis
claim is narrowing to the *tail* (hard-case flexibility, `[[referee-flexibility-thesis]]`),
not the mean.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 PINS_CACHE=pins/cache_exp70.json PINS_RESULTS=pins/results_exp70.json \
  .venv/bin/python -u -m pins.trace_replay --referee --llm --model qwen2.5:14b \
  --caps predicted --pools 8 --seeds 32 --samples 10
```

## Experiment 71 — THE VERDICT SURVIVES THE PLAN'S PRIMARY PROGRESS LAW (--law sat, n=32) (2026-07-22)

Exp 70's configuration re-run under the elevated plan's §3.2 saturating law — the robustness
pair the plan demands (§3.3: a conclusion counts only if it holds under both models). Warm
cache from Exp 70, so the bills here are marginal, not cold; §13's budget verdict stands on
Exp 70.

```
   8  referee       39.3%   39.7%    77%     69%     14%      7.57   20.0    1% 15.7/16
   8  negotiated    37.5%*  37.6%*   79%     71%     10%      6.45   16.9    0% 15.7/16
   8  single-llm    41.0%   39.2%    80%     70%     13%      7.68   19.8    0% 15.7/16

      referee      vs floor:  dSLA -0.2 ±1.9   dprodSLA -6.3 ±3.5*  dutil -0.8 ±1.8   duseful -2.2 ±1.6*  dregret +4.4 ±1.8*
      negotiated   vs floor:  dSLA -2.0 ±1.3*  dprodSLA -8.4 ±4.4*  dutil +1.7 ±1.6*  duseful -0.2 ±1.3   dregret +0.6 ±1.1   dslow -0.8 ±0.7*
      single-llm   vs floor:  dSLA +1.6 ±2.6   dprodSLA -6.8 ±5.1*  dutil +2.5 ±2.3*  duseful -1.4 ±1.6   dregret +3.1 ±2.0*
      Holm (18): referee/regret p=0.001, referee/useful p=0.011, negotiated/prodSLA p=0.023,
                 single-llm/regret p=0.048, referee/prodSLA p=0.048, negotiated/lateness p=0.048
```

**1. Both of Exp 70's conclusions hold under the other law.** Referee − single-llm on prod SLA
is **+0.5 ± 5.2, p=0.789** (was +0.1, p=0.965): still indistinguishable. The referee is again
significantly worse on utilisation (−3.3 ± 1.5, p<0.001). The multi-agent structure buys
nothing on the mean under either progress model.

**2. Under the primary law the AUCTION becomes the best prod-tier protector too.** dprodSLA
−8.4\* (negotiated) vs −6.3\* (referee) — a reversal of the amdahl ordering, and negotiated is
the only arm also positive on utilisation (+1.7\*) and significantly better on SLA (−2.0\*) and
slowdown (−0.8\*). Head-to-head, referee − negotiated: prod SLA +2.1 ± 5.1 (ns), util
−2.5 (p<0.001), **useful util −1.9 (p<0.001), regret +3.7 (p<0.001)** — the referee is
significantly worse on three of four, tied on the fourth.

**3. Six survivors under Holm, and the referee's are split in sign** — `regret +4.4` (worse,
p=0.001) and `useful −2.2` (worse, p=0.011) rank ABOVE its `prodSLA −6.3` (better, p=0.048).
Under the plan's primary law the strongest thing that can be said about the referee is what
it costs.

**Standing conclusion (Exp 68→71), both laws, Holm-corrected, budget-metered.** The reasoning
layer does not pay for itself on the mean: it is matched on prod-tier protection by one
cheaper centralised LLM call, and beaten outright by the deterministic auction on
utilisation, useful utilisation, regret, slowdown — and, under the primary law, on prod-tier
protection as well. The thesis claim that remains open is the TAIL: hard-case flexibility
(`[[referee-flexibility-thesis]]`, `[[hardcase-suite]]`), which no mean-based experiment in
this log has ever tested.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 PINS_CACHE=pins/cache_exp71.json PINS_RESULTS=pins/results_exp71_sat.json \
  .venv/bin/python -u -m pins.trace_replay --referee --llm --model qwen2.5:14b \
  --caps predicted --pools 8 --seeds 32 --samples 10 --law sat
```

## Experiment 72 — THE EXPLICIT MARKET (plan §6): the bid was never a bid, and fixing it buys utilisation everywhere (2026-07-22)

**What was actually there.** `two_sided_sim` — the simulator behind every result from Exp 22
to Exp 71 — never ran a market. The bid entered exactly once,
`frozen = {j.jid: sum(j.bid()) for j in active}` (two_sided_sim.py:429): the marginal-value
CURVE collapsed to a scalar, used as a tie-break for greedy fill. No supply ask, no clearing
condition. `pins/mechanism.py`'s uniform-price auction is only called by the older sims.

**What Exp 72 builds** (`pins/market.py`, `--market`, deterministic, zero tokens):
demand `b_(j,k) = α·dp̂_(j,k) + β·R_SLA + γ·W_wait − δ·C_resize` per job per extra GPU,
non-increasing; supply `a_q = η1·Scarcity + η2·Frag + η3·ReservePressure + η4·ArrivalPressure`
rising with units sold; clearing `Q_t = max{q : b_(q) ≥ a_(q)}`. **`dp̂` is the job's real
marginal useful progress** from the Exp 68 counterfactual model — the term that made §6
buildable at all.

**A design error that changed the design.** The first build set `reserve = free − sold`
("unsold = held-idle headroom", the plan's literal reading). It collapsed the cluster:
util 30%, 8.6/16 finished, `dutil −48.1*`. Cause: in this simulator the reserve blocks
best-effort jobs from their **base**, not just their margin (two_sided_sim.py:455), so every
unsold margin unit starved queued work. Corrected mapping: **headroom lives in the price**
(η3 raises the ask when prod arrivals pend, so fewer margin units sell) and undsold units stay
available for other jobs' bases — which is what actually helps an incoming prod job.
`hold_unsold=True` preserves the old semantics for comparison.

**Result (n=32 paired, pools 4/6/8, both laws, v2020).** market vs the floor:

| law | pool | dSLA | dprodSLA | dutil | duseful | dregret |
|---|---|---|---|---|---|---|
| amdahl | 4 | −1.0\* | −1.0 | +0.9\* | +0.9\* | −1.3\* |
| amdahl | 6 | −0.6 | −0.7 | +1.6\* | +1.6\* | −2.1\* |
| amdahl | 8 | −1.6\* | −2.7 | **+2.1\*** | **+2.1\*** | **−2.8\*** |
| sat | 4 | −0.4 | +0.0 | +1.2\* | +0.2 | −0.1 |
| sat | 6 | +0.0 | +0.0 | +2.1\* | +0.4\* | −0.4\* |
| sat | 8 | −0.2 | −0.8 | **+2.3\*** | +0.5\* | −0.5\* |

Head-to-head vs `negotiated` (paired, n=32), the arm that won Exp 70/71:

| law | pool | dutil | duseful | dregret |
|---|---|---|---|---|
| amdahl | 4/6/8 | +1.9\* / +1.6\* / +1.1\* | +2.2\* / +2.0\* / +1.8\* | −3.0\* / −2.7\* / −3.0\* |
| sat | 4/6/8 | +2.9\* / +2.5\* / +1.6\* | +2.6\* / +2.1\* / +1.3\* | −2.8\* / −2.0\* / −1.6\* |

**1. The market beats every previous arm on the plan's primary metrics, in all 6 cells, under
both laws.** Utilisation, useful utilisation and regret all improve significantly against both
the floor and `negotiated` — and `market/util`, `market/useful`, `market/regret` survive Holm
in most cells. It is the only arm in the project's history that is simultaneously positive on
useful utilisation and negative on regret.

**2. It does NOT protect the prod tier.** dprodSLA is small and ns everywhere (−2.7 at best,
+0.0 under sat). The referee's −8.9\*/−6.3\* remains the largest prod-tier effect measured. So
the two capabilities are now cleanly separated: **the market allocates efficiently; the LLM
arms ration protectively**, and neither does the other's job.

**3. It costs nothing.** Zero LLM calls, zero tokens, 0% fallback. The comparison against a
24,240-token referee is not close on any efficiency metric.

**Standing picture after Exp 68–72.** Efficiency (util / useful util / regret) belongs to the
deterministic market; prod-tier protection belongs to the reasoning layer but is reproduced by
one cheap centralised LLM call (Exp 70/71). The open question is whether the two compose —
a market that clears efficiency with an LLM setting only the protective reserve — and whether
the LLM's protection survives when it no longer controls the margin.

**Reproduce.**
```bash
for L in amdahl sat; do PINS_RESULTS=pins/results_exp72_$L.json \
  .venv/bin/python -m pins.trace_replay --market --seeds 32 --pools 4,6,8 --law $L; done
```

> **AMENDED 2026-07-22 (Exp 74).** The comparison above is NOT information-matched. The market
> received each job's usable extra parallelism as an exact integer (`facts["usable"]`), while
> the LLM arms see the same quantity only as a `low/medium/high` spike-risk bucket. Re-running
> with `--bid-info bucket` (the market told only what the LLM arms are told) **halves every
> gain**: at pool 8/amdahl `dutil +2.1* -> +1.1*`, `duseful +2.1* -> +1.1*`,
> `dregret -2.8* -> -1.6*`; at pool 8/sat `dutil +2.3* -> +1.3*`. The mechanism half survives
> and the market still beats `negotiated` on useful utilisation and regret in all six cells at
> equal information — but the headline number in the table above is the *exact-information*
> figure and must not be quoted without this qualifier. See Exp 74.

## Experiment 73 — THE COMPOSED ARM: the referee's whole contribution is one scalar, and protection trades against efficiency through the SAME resource (2026-07-22)

**Design.** `--composed` (`pins/market.py::make_policy_composed`): the LLM sees only the
supply context and returns a reserve LEVEL; that headroom is withheld and the plan's §6 market
clears the REMAINING pool into margins. The LLM never touches the margin — the part Exp 70/71
showed it pays for in regret. Pre-registered falsification: if the referee's protection was
reasoning about arrivals it survives the amputation; if it was margin-hoarding, `composed`
collapses onto plain `market`.

**Setup.** 14b, caps=predicted, pool 8, n=32 paired seeds, both laws, all five arms on
identical seeds. (Cache warm from Exp 71, so marginal tokens read 0 here; the cost comparison
stands on Exp 70's cold-cache bills.)

vs the floor:

| law | arm | dSLA | dprodSLA | dutil | duseful | dregret |
|---|---|---|---|---|---|---|
| amdahl | referee | −1.4 | **−8.9\*** | −1.2 | −1.9\* | +3.2\* |
| amdahl | market | −2.1\* | −3.8\* | +1.3\* | +0.7\* | −1.3\* |
| amdahl | composed | +0.4 | **−5.6\*** | −0.1 | −0.2 | +0.8 |
| sat | referee | −0.2 | **−6.3\*** | −0.8 | −2.2\* | +4.4\* |
| sat | market | +0.4 | −0.7 | +2.5\* | +1.2\* | −0.8\* |
| sat | composed | −1.0 | **−7.5\*** | +0.2 | −0.8 | +1.7\* |

**1. The protection survives, and the margin reasoning was pure cost.** composed − referee:
prodSLA **+3.3 (ns) / −1.1 (ns)** — indistinguishable — while regret improves **−2.4\* / −2.6\***
and useful utilisation **+1.7\* / +1.4**. **`composed` dominates `referee` on both laws** and
should replace it as the LLM arm. The referee's entire measurable contribution is reproducible
by choosing a single scalar; its per-job statements, rebuttals and rulings are the part that
costs regret, useful work, and 24k tokens/seed.

**2. Composition does NOT dominate — protection and efficiency draw on the same resource.**
composed − market: prodSLA −1.8 (ns) / **−6.7\*** but util **−1.4\* / −2.3\***, useful
**−0.9\* / −2.0\***, regret **+2.0\* / +2.5\***. The reserve protects by holding GPUs IDLE, and
idle GPUs are exactly what the market's utilisation gain was made of. This is not two
complementary layers; it is one dial.

**3. The result is a Pareto frontier, not a winner.**

| arm | buys | costs | bill |
|---|---|---|---|
| `market` | util, useful util, regret (all \*) | no prod protection | 0 tokens |
| `composed` | referee-equivalent protection | efficiency, significantly | 1 cheap call/tick |
| `referee` | nothing `composed` doesn't | regret\*, useful\* | 24,240 tok/seed |

Which of `market`/`composed` is preferable is an operator preference (throughput vs
production protection), not something the data settles. Reporting it as an interpretable
frontier with a single dial is a stronger claim than "the LLM wins" — and it is what the
evidence supports.

**Consequence for the thesis.** The multi-agent architecture is now dominated on the mean by a
one-scalar LLM plus a deterministic market. What remains genuinely untested is the TAIL:
hard-case flexibility (`[[referee-flexibility-thesis]]`, `[[hardcase-suite]]`) — where a
scalar reserve manifestly cannot express the decision, and the destroyed Exp 64-67 arms were
the probe.

**Reproduce.**
```bash
for L in amdahl sat; do PINS_NUM_CTX=8192 PINS_CACHE=pins/cache_exp73.json \
  PINS_RESULTS=pins/results_exp73_$L.json .venv/bin/python -u -m pins.trace_replay \
  --referee --market --composed --llm --model qwen2.5:14b --caps predicted \
  --pools 8 --seeds 32 --law $L; done
```

## Experiments 74–76 — THE MARKET'S THREE QUALIFIERS (2026-07-22)

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 77 (H2) — BID-FIRST CORRECTION vs FROM-SCRATCH GENERATION, on the hard-case suite (2026-07-22)

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 78 — DOES THE REVIEWER'S JUDGEMENT TRACK MEASURED USAGE ON REAL JOBS? (2026-07-22)

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 78d — THE WORKER-HOLDOUT VENUE IS TRIVIAL: the idle gate belongs in the mechanism (2026-07-22)

**Venue conclusion.** v2020 cannot host a runtime-evidence reasoning experiment: no time axis, and the observable quantity settles the question by threshold. That experiment needs MIT Supercloud (10 s CPU / 100 ms GPU sampling), where the target is genuinely a FUTURE interval rather than a copy of what was shown. Recorded so the design is not re-attempted on this trace.

*Compressed. Full write-up: `git show b8b0af0:research_progress.md`*

## Experiment 79 — PERSPECTIVE × TEXT on a seven-model ladder: the referee split is INERT, the text is everything (2026-07-23)

**Date:** 2026-07-23

**Question.** Round-3 of the hard-case suite (31 pre-registered primary scenes + 9 placebo/confirm
controls) is a 2×2: {single LLM, referee split} × {exception text present, text stripped}. Does
splitting the reasoning into demand/supply perspectives make the free-text exception *usable* — a
genuine interaction — or is any gain just the text itself?

**Method.** `pins/exp79_analyse.py` over `results_hardcases_r3_2x2_<model>.json`, seven local
models. Headline metric is STRICT: handled **and** feasible (over-award penalised). Pre-registered
confirmatory pool = models with infeasibility < 25%. McNemar one-sided exact on discordant pairs
(b = referee-favoured, c = single-favoured).

**Findings (STRICT, primary n=31; cells are text/no-text).**

| model | single | referee | text S | text R | inter | infeas |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b | 2/2 | 3/2 | +0 | +1 | +1 | 70% *below bar* |
| gemma2:2b | 3/1 | 4/1 | +2 | +3 | +1 | 81% *below bar* |
| qwen2.5:3b | 3/2 | 3/2 | +1 | +1 | +0 | 63% *below bar* |
| qwen2.5:7b | 10/2 | 10/2 | **+8** | +8 | +0 | 6% |
| llama3:8b | 5/2 | 6/2 | +3 | +4 | +1 | 23% |
| gemma2:9b | 4/0 | 2/0 | +4 | +2 | −2 | 58% *below bar* |
| qwen2.5:14b | 12/1 | 12/1 | **+11** | +11 | +0 | 2% |

- **The interaction is null.** Confirmatory pool (7b, llama3:8b, 14b): b=3 c=2, one-sided exact
  **p=0.500**. All seven models pooled: b=6 c=5, p=0.500. Parallel multi-agent structure buys
  nothing over a single call — across every family and every size tested.
- **The text effect is large and real** (+11 at 14b, +8 at 7b, +3/+4 at llama3:8b). The suite
  measures what it was built to measure; it is the *perspective split* that fails, not the text
  channel.
- **Feasibility is a capability gate.** Four of seven models sit above 55% infeasibility and cannot
  be scored on the headline at all; only 14b (2%), 7b (6%) and llama3:8b (23%) clear the bar.

**Why this did not end the multi-agent question.** Exp 80–82 found three separate interface defects
in the arm this experiment scored. The structure was being judged through a broken channel.

**Reproduce.**
```
bash pins/run_exp79_models.sh
.venv/bin/python -m pins.exp79_analyse
```

---

## Experiment 80 — THE CORRECTION INTERFACE IS DEAD ON ARRIVAL: 0/31, one call fired (2026-07-23)

**Date:** 2026-07-23

**Question.** Reframe the agent as a *corrector*: the market allocates, the LLM only proposes
corrections. Does restricting it to corrections improve exception handling?

**Findings.** All three arms score **0/31** on the primary set. The correction arm fired **once**
in 31 cases, and that single ruling was **invalid**. Controls: every arm 8/9 — identical to the
market, i.e. the LLM never intervened.

| arm | handled | fired | invalid | net |
|---|---|---|---|---|
| market | 0/31 | 0 | 0 | +0 |
| single-corr | 0/31 | 1 | 1 | +0 |
| referee-corr | 0/31 | 1 | 1 | +0 |

This is not a null result about reasoning. It is a broken interface: the agent was asked to correct
an allocation without being shown enough of it to have grounds to object.

**Reproduce.** `.venv/bin/python -m pins.exp80_correction_2x2 --model qwen2.5:14b`

---

## Experiment 81 — SIGNED CORRECTIONS: the agent finally acts, and one ruling in three is infeasible (2026-07-23)

**Date:** 2026-07-23

**Question.** Exp 80's arm never fired. Give it a signed correction contract — an explicit,
schema-checked statement of what it is allowed to change. Does it now act, and act legally?

**Findings.**

| arm | handled | fired | invalid | rescued | broke | net |
|---|---|---|---|---|---|---|
| market | 0/31 | 0 | 0 | 0 | 0 | +0 |
| single-sgn | **10/31** | 31 | **11** | 10 | 0 | +10 |
| referee-sgn | 5/31 | 26 | **16** | 5 | 0 | +5 |

- The arm now fires on essentially every case (31 and 26 of 31) and handles real exceptions for the
  first time (10/31).
- But **11 and 16 rulings are infeasible** — signing made the agent willing to act, not able to act
  legally.
- The split *hurts*: head-to-head referee vs single is 4 referee-only against 9 single-only.
- Controls: single-sgn 7/9 with **broke 1** — it damaged a case the market had right.

**Reproduce.** `.venv/bin/python -m pins.exp81_signed_2x2 --model qwen2.5:14b`

---

## Experiment 82 — THE DECISION PACKET: infeasible rulings go to ZERO, and the packet is capability-gated (2026-07-23)

**Date:** 2026-07-23

**Question.** Exp 81's failure was informational, not motivational. Give every arm an identical
structured decision packet — job rows, capacity, reserve, tier, and the auction's own award — so a
ruling can be checked before it is emitted.

**Findings (qwen2.5:14b).**

| arm | handled | fired | invalid | rescued | broke | net |
|---|---|---|---|---|---|---|
| market | 0/31 | 0 | 0 | 0 | 0 | +0 |
| single-pkt | 9/31 | 28 | **0** | 9 | 0 | +9 |
| referee-pkt | 11/31 | 28 | **0** | 11 | 0 | +11 |

- **Infeasibility collapses from 11–16 to 0** in every arm. The packet is what makes a ruling
  checkable; this is the fix that makes the hard-case suite a valid instrument.
- The perspective split is still not a real effect (head-to-head 5 vs 3).
- **Controls expose the cost:** single-pkt and referee-pkt both 6/9 against the market's 8/9, with
  **broke 2** each. The packet also makes the arms fire on cases that did not need them.
- **Capability-gated.** The identical packet at **qwen2.5:7b scores 0/31** (fired 6 and 18, invalid
  0) — it fires and says nothing useful. The packet is not a prompt trick that lifts small models;
  it is scaffolding only a capable model can stand on.
- With the operator precedent manual in context (`_hist`), 14b improves to single 11/31,
  referee 13/31.

**Reproduce.** `.venv/bin/python -m pins.exp82_packet_2x2 --model qwen2.5:14b --depth 6`

---

## Experiment 83 — DEBATE ON THE PACKET: the first structure that beats a single call (2026-07-23)

**Date:** 2026-07-23

**Question.** Parallel perspectives were inert (Exp 79). With the interface repaired (Exp 82), test
a *sequential* structure instead: demand- and supply-side reviewers state positions, read each
other, and revise before a referee rules.

**Findings (qwen2.5:14b, primary n=31).**

| arm | handled (no docs) | handled (+docs) |
|---|---|---|
| market | 0/31 | 0/31 |
| single-pkt | 9/31 | 11/31 |
| referee-pkt | 11/31 | 13/31 |
| **debate-pkt** | **14/31** | **15/31** |

- Debate **strictly dominates** the single call: head-to-head 6 debate-only against 1 single-only,
  and 3 against 0 versus the parallel referee. It is the first structural arm in the project that
  wins without also losing somewhere.
- **Debate and the operator docs are substitutes, not complements.** Docs add +2 to the single call
  but only +1 to debate — both mechanisms push attention onto the operator instruction, so the
  second one buys little. Best absolute configuration is debate+docs at 15/31; the cleanest
  *mechanism* evidence is the 14-vs-9 no-docs comparison.
- Controls are unchanged at 6/9 with broke 2 — debate adds no extra false firing over the packet.

**Caveat that Exp 88 exists to close.** Debate spends far more calls than a single ruling. At this
point the win could be budget rather than structure.

**Reproduce.** `.venv/bin/python -m pins.exp82_packet_2x2 --model qwen2.5:14b --depth 6 --debate`

---

## Experiment 84 — THE SAME DEBATE, IN THE SIM: SLA-null and production-costly (2026-07-23)

**Date:** 2026-07-23

**Question.** Debate wins on authored text exceptions. Does that convert into a service-quality
gain when it *replaces* the auction on the numeric v2020 trace?

**Findings (qwen2.5:14b, caps=predicted, pool 8, n=32 paired seeds, vs the no-LLM floor).**

| arm | dSLA | dprodSLA | dutil | duseful | dregret |
|---|---|---|---|---|---|
| referee | −0.2 ± 1.9 | −7.6 ± 5.0* | −1.6 ± 1.7 | −2.3 ± 1.5* | +3.4 ± 1.8* |
| **debate** | +0.8 ± 2.4 | **−12.0 ± 7.0*** | −2.5 ± 1.8* | −2.9 ± 1.7* | +5.0 ± 2.1* |
| negotiated | −1.8 ± 2.2 | −6.6 ± 4.6* | +2.0 ± 1.0* | +0.9 ± 0.8* | −0.9 ± 1.1 |

- Debate is **SLA-null (+0.8, ns)** and carries the **largest production-tier penalty in the
  project (−12.0\*)**, surviving Holm correction over the family of 18 vs-floor tests
  (prodSLA p=0.020 wilcoxon, regret p=0.001, useful p=0.018).
- **Why, and it is not a contradiction of Exp 83.** A text-stripped control run of the same suite
  scores **every arm at zero** — the v2020 trace carries no operator free text for debate to act
  on. The market already gets the numeric scenes right, so a reasoning layer that *replaces* it can
  only do harm.
- This is the honest boundary the paper reports rather than papers over: debate belongs behind a
  trigger (Exp 87), not in the default path.

> **⚠ TWO DIFFERENT DEBATES (audit 2026-07-28) — do not read this as "Exp 89's debate, minus the
> text".** The arm measured here and the arm that wins in Exp 83/88/89 are different mechanisms,
> and the packet never crossed between them (`pins/referee.py`, `trace_replay.py`,
> `two_sided_sim.py`, `llm_agent.py` all contain **zero** references to `packet`; `debate_signed`
> is imported only by `exp82_packet_2x2.py` and `exp88_budget_control.py`).
>
> | | in-sim debate (this experiment, Exp 87) | hard-case debate (Exp 83, 88, 89) |
> |---|---|---|
> | code | `referee.py`: `gather_statements → rebut → referee_decide` | `correction_signed.debate_signed()` |
> | scope | **full re-decision of every job** | **corrections only**, on top of the market's allocation |
> | text gate | none | skips any job with no note: `if not (j.get("note") or "").strip(): continue` |
> | interface | free-form ruling | packet + code-generated action menu |
>
> So the −12.0\* prodSLA here is *partly* the cost of replacing the auction wholesale, not purely
> evidence about the absence of text. The text-stripped control (every arm 0) supports the no-text
> reading but does not isolate it from the architectural difference. The clean test is the open
> item below.
>
> **Open experiment (cheap, decisive).** Run `debate_signed` — the *winning* architecture — in-sim
> on v2020. Because it skips note-less jobs and the trace has no notes, it should be a **literal
> no-op**: zero escalations, allocation byte-identical to `market`. "The mechanism that wins on
> text does exactly nothing where there is no text" is a strictly stronger boundary claim than
> "a different, more invasive debate hurt production SLA", and it would let §4.4 drop its reliance
> on this arm.

**Reproduce.**
```
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b --caps predicted \
    --pool 8 --seeds 0-31 --referee --debate
```

---

## Experiment 85 — TEXT-STRIPPED CONTROL: not separately run (2026-07-23)

**Date:** 2026-07-23

**Status.** Recorded for numbering continuity only. `pins/exp85_14b_debate_notext.log` is **empty
(0 bytes)** — the run was superseded before it produced output. The text-stripped control it was
meant to provide was instead captured inside Exp 84
(`pins/exp84_14b_debate_notext.log`), where every arm scores zero on every case with the exception
text removed. No separate Exp 85 result exists; do not cite one.

---

## Experiment 86 — THE PER-TICK VALIDATOR IS INERT ON CORRECT CLEARINGS (2026-07-23)

**Date:** 2026-07-23

**Question.** The gated architecture validates every auction clearing with cheap deterministic
code before applying it. Does that safety layer cost anything when the clearing is already correct?

**Findings.** Re-running the market arm with the validator reproduces the Exp 72 numbers
**byte-identically** across both progress laws × pools {4,6,8} × 32 seeds, at **0% fallback
throughout**.

| law | pool | floor SLA | market SLA | util | regret | fb |
|---|---|---|---|---|---|---|
| amdahl | 4 | 68.0% | 67.0%* | 95% | 0% | 0% |
| amdahl | 6 | 58.4% | 57.8%* | 90% | 0% | 0% |
| amdahl | 8 | 51.8% | 50.2%* | 84% | 1% | 0% |
| sat | 4 | 63.7% | 63.3% | 95% | 0% | 0% |
| sat | 6 | 52.5%* | 52.5%* | 89% | 0% | 0% |
| sat | 8 | 44.9% | 44.7% | 82% | 0% | 0% |

Fault injection (oversubscription, negative award, unknown job id) confirms the validator rejects
when it should. It is a bug-and-stale-input guard that never touches a well-formed auction output —
exactly what a safety layer should be, and the precondition for gating reasoning behind it.

**Reproduce.**
```
.venv/bin/python -m pins.trace_replay --market --validate --pool 4,6,8 --seeds 0-31 --law amdahl,sat
```

---

## Experiment 87 — THE GATED ARCHITECTURE: validated auction every tick, debate+docs only on a contextual trigger — SLA is UNCHANGED vs the bare auction (2026-07-23)

The design under test (user's spec): every tick runs the deterministic clearing, then a cheap
code validator (`market.validate_clearing` — rule-1 feasibility + unknown-id + negativity,
reusing `referee.check_allocation`). If the clearing is valid AND no contextual trigger fires,
the auction's allocation is applied with **zero LLM calls**. On a trigger (validator reject,
prod arrival, job newly behind deadline, contested-capacity crossing, cold start — the Exp-61
`_hard` event set) the **debate arm** rules instead, and an infeasible ruling falls back and
re-fires next tick. Escalation is DEBATE+precedent-docs (`--manual referee_manual_learned.md`),
the 15/31 hard-case configuration (Exp 83), not the plain referee. `market.make_policy_gated`,
new `--gated` flag, own tier `qwen2.5:14b+pred+manual-learned+gated`.

The point of the experiment: Exp 83 showed debate+docs FIXES the single LLM's unreliable
exception suggestions on the text suite (15/31 vs single 9–11/31, zero harm). This asks the
orthogonal question — **does layering that arm onto the validated auction cost SLA?**

**Run.** qwen2.5:14b, caps=predicted, pool 8, seeds 0-31. Paired within seed against `market`
(validated auction alone) and the `no-llm` floor.

| pool | arm | SLA | prodSLA | util | useful | regret | done |
|---|---|---|---|---|---|---|---|
| 8 | no-llm (floor) | 53.9% | 59.1% | 80% | 75% | 10% | 15.3/16 |
| 8 | market | 51.8% | 55.3% | 81% | 76% | 8% | 15.4/16 |
| 8 | gated | 51.6%\* | 54.7% | 81% | 76% | 9% | 15.4/16 |

```
market vs floor:  dSLA -2.1 ± 1.7*  dprodSLA -3.8 ± 3.1*  dutil +1.3 ± 0.5*  duseful +0.7 ± 0.4*  dregret -1.3 ± 0.8*
gated  vs floor:  dSLA -2.3 ± 1.9*  dprodSLA -4.4 ± 4.1*  dutil +1.1 ± 0.7*  duseful +0.3 ± 0.6   dregret -0.8 ± 0.8
gated MINUS market (paired, n=32):  dSLA -0.20 ± 1.40   dprodSLA -0.63 ± 3.34   dutil -0.21 ± 0.38   dregret +0.49 ± 0.39*   duseful -0.40 ± 0.40*
gate: 5791 auction ticks, 1076 escalated (15.7%), 9 escalated rulings infeasible
cost: gated 83.2 calls/seed, 52,853 tokens/seed, 293.9 s/seed;  every other arm 0 tokens
```

**Findings.**

1. **SLA is unchanged. gated − market = −0.20 ± 1.40 SLA, ns** — statistically identical to the
   0-token auction. The escalation genuinely ran (15.7% of ticks, 1,076 debates, only 9
   infeasible), it simply did not move the outcome. The two arms track each other on every
   headline metric (prodSLA −0.63 ns, util −0.21 ns); the only starred paired diffs are a
   trivial +0.49\* regret and −0.40\* useful — the debate arm's known small efficiency drag,
   not an SLA effect.

2. **This is the EXPECTED result, and it is the thesis, not a null.** The trace has no text
   exceptions (4-tuple jobs, bucketed enums — see the debate-mechanism thread and Exp 57f).
   The 15.7% of ticks that escalated were capacity crossings and prod arrivals — numeric scenes
   where the auction was already correct and the docs had nothing to catch. So the arm behaves
   exactly as the gated design predicts: **cheap validated auction on routine ticks, escalate on
   triggers, and where the escalation has real text to reason about (the hard-case suite) it wins
   15/31; where it does not (this trace) it is SLA-neutral rather than harmful.**

3. **The paired claim now holds both halves cleanly.** Debate+docs improves the unreliable
   single-LLM suggestion on text-dependent exceptions (Exp 83, 15/31 vs 9/31), AND layering that
   same arm onto the validated auction does not cost SLA (Exp 87, −0.2 ns). Efficiency-neutral
   capability-add, which is stronger and more defensible than any "debate raises SLA" claim —
   the sim cannot raise SLA with debate because it contains nothing for debate to fix (compare
   Exp 84's plain 14b debate row: dSLA +0.8 ns, but −12.0\* prodSLA when it REPLACES the auction
   rather than gating on top of it).

4. **The validator is confirmed inert on correct clearings (Exp 86 companion).** Re-running the
   bare `market` arm with the new per-tick validator reproduced Exp 72's numbers byte-identically
   across amdahl+sat × pools 4/6/8 × 32 seeds, fb 0% throughout — so the validator is a
   bug/stale-input guard that never rejects a well-formed auction output, exactly what a safety
   layer should be. Fault-injection (oversubscription, negative award, unknown id) confirmed it
   fires when it should.

**Honest cost caveat.** The escalation buys the exception-handling CAPABILITY at 52,853
tokens/seed for an SLA result identical to the free auction ON THIS TRACE — because this trace
never exercises the capability. The token bill is only justified in a venue with genuine text
exceptions; on a pure-numeric workload the bare `market` arm dominates on cost at equal SLA.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.trace_replay --gated --market --llm \
  --model qwen2.5:14b --caps predicted --pools 8 --seeds 32 --manual pins/referee_manual_learned.md
```
Backup of the pre-run results at `pins/results_backup_pre_exp87_gated.json`.

**Next (unchanged from Exp 78, plus):** (d) if the gated arm is to show an SLA gain in-sim, the
trace needs a text-exception channel — no public GPU trace carries one (they are scrubbed of the
user-authored free text by construction; see the dataset thread), so this is an AOBA-data /
authored-notes question, not an engineering one.

## Experiment 88 — THE BUDGET CONTROL: debate's win is STRUCTURE, not spend (2026-07-24)

**Date:** 2026-07-24

**Question.** Exp 83's debate arm beat a single call 14/31 vs 9/31 — but it also spent 217 LLM
calls against the single arm's 31. The obvious reviewer objection is that debate simply bought its
win with tokens. Control for budget: add a **best-of-N single call** (`single-pkt-boN`) at the
*same* 217 calls, so the only difference from debate is how the budget is spent.

**Method.** `pins/exp88_budget_control.py`, qwen2.5:14b, T=0.8, round-3 suite (31 primary + 9
controls), all arms on the identical decision packet. McNemar one-sided exact on discordant pairs.

**Findings (primary n=31).**

| arm | handled | fired | invalid | calls |
|---|---|---|---|---|
| market | 0/31 | 0 | 0 | 0 |
| single-pkt | 9/31 | 28 | 0 | 31 |
| single-pkt-boN | 10/31 | 28 | 0 | **217** |
| **debate-pkt** | **14/31** | 28 | 0 | **217** |

| comparison | b | c | one-sided exact p |
|---|---|---|---|
| debate vs boN | 5 | 1 | 0.109 |
| debate vs single | 6 | 1 | 0.063 |
| boN vs single | 1 | 0 | 0.500 |

- **Spending the budget does nothing; spending it on debate does.** Best-of-N at 217 calls scores
  10/31 — essentially flat against the single call's 9/31 (p=0.500) — while debate at the *same*
  217 calls scores 14/31. The multi-agent sub-claim survives its sharpest control.
- Neither contrast reaches significance at n=31 (p=0.109 vs boN). The effect is real in direction
  and the design is now clean; what is missing is **cases, not compute** — which is what Exp 89
  supplies.
- Controls n=9: market 8/9, single 6/9, boN 5/9, debate 6/9. The LLM arms remain slightly
  *worse* than the market where no exception exists.

**Reproduce.** `.venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b --temp 0.8`

---

## Experiment 89 — THE POWERED CONFIRMATORY RUN: debate beats matched-budget best-of-N at p=0.00066 (2026-07-24)

**Date:** 2026-07-24

**Question.** Exp 88 left the right design and the wrong sample size. Pool round-3 with 50 newly
authored, blind, mechanism-engineered round-4 cases (81 primary, 17 controls) and re-run the same
four arms. Pre-registered primary contrast: **debate-pkt vs single-pkt-boN**, one-sided McNemar.

**Findings (primary n=81).**

| arm | strict | handled | fired | invalid | calls |
|---|---|---|---|---|---|
| market | 0/81 | 0/81 | 0 | 0 | 0 |
| single-pkt | 27/81 | 27/81 | 73 | 0 | 81 |
| single-pkt-boN | 29/81 | 29/81 | 73 | 0 | 569 |
| **debate-pkt** | **43/81** | **43/81** | 75 | 0 | 569 |

| comparison | b | c | one-sided exact p |
|---|---|---|---|
| **debate vs boN (primary)** | 16 | 2 | **0.00066** |
| debate vs single | 18 | 2 | **0.00020** |
| boN vs single | 2 | 0 | 0.250 |

- **The pre-registered contrast lands.** Debate beats a budget-matched best-of-N single call at
  **p=0.00066**, up from p=0.109 at n=31. The Exp 88 direction was real and the run was simply
  underpowered. This is the strongest structural result in the project.
- **Budget still buys nothing on its own**: boN vs single is 29 vs 27, p=0.250, at 7× the calls.
  The gain is attributable to the debate structure, not to spend.
- **The market floor is absolute**: 0/81. No text exception is reachable from the numbers.
- **Specificity cost is real and must be reported.** On the 17 controls the market scores **16/17**
  while all three LLM arms score **12/17**. Where no exception exists, reasoning makes things
  slightly worse — which is precisely the argument for gating it behind a trigger rather than
  running it by default.

**Provenance caveat.** `pins/exp89_analysis.txt` records `RUN ENDED WITHOUT WRITING
pins/results_exp89_qwen2514b_t0.8.json` — the analyser wrote
`results_exp89_qwen2.514b_t0.8.json` (dotted model name) and the undotted copy was made
afterwards. Both files are 259,311 bytes and carry the same run; the tables above are from
`exp89_analysis.txt`. Re-analyse from the dotted file if in doubt.

**What this changes.** §4.4 of `paper/pins_gated_draft.md` currently cites Exp 83's 14/31 vs 9/31
with "one-sided exact p=0.0625". That is now superseded by a powered result with a budget-matched
control. The paper should be updated to lead with Exp 89.

**Reproduce.**
```
.venv/bin/python -m pins.exp89_analyse pins/results_exp89_qwen2.514b_t0.8.json
```

---

## Experiment 90 — NO-EXCEPTION SPECIFICITY: the packet arms make ZERO false suggestions on real jobs (2026-07-24)

Pre-reg: `docs/superpowers/specs/2026-07-24-exp90-no-exception-specificity-design.md`;
plan: `docs/superpowers/plans/2026-07-24-exp90-no-exception-specificity.md`.

The mirror of the hard-case suite. That suite measures **recall** on authored text exceptions
(Exp 89: debate 43/81). This measures **precision / specificity** on **200 real v2020
job-scenes with no authored text**, where the correct action is retain-market and a false
suggestion = any override (`meta["fired"]`). Motive: the hard cases are self-authored and few;
this half uses jobs straight from the trace and asks the dual question — on an ordinary job with
no exception, does the machinery correctly do **nothing**?

**Scenes** (`pins/no_exception_scenes.py`): 200 real jobs sampled from v2020, 3–5 per scene,
inelastic (`margin 0`), with idle headroom (`slack 0.15–0.6`, `free = base_sum + max(1,
round(base_sum·slack))` so `free > base_sum` by construction) as the only meddling surface. A
non-vacuity gate keeps only scenes whose code-enumerated action menu is non-trivial — **mean menu
≈ 77 legal grant/transfer/hold options per scene**, so "change nothing" is a real choice, not the
only option. `qwen2.5:14b`, temperature 0, `max_delta 6`.

**Result.**
```
per-arm false-suggestion rate (n=200)              of which harmful
  market          0/200                             0
  single-no-pkt   0/200                             0
  single-pkt      0/200                             0
  debate-pkt      0/200                             0
PRIMARY McNemar debate-pkt vs single-pkt (H1 debate fires LESS):
  b=0 c=0   one-sided p=1.0000   -> tie at the floor
```

**Findings.**

1. **Perfect specificity.** Every arm retained the market on all 200 real no-exception scenes.
   Despite ~77 legal alternatives per scene, both `single-pkt` and `debate-pkt` chose "change
   nothing" every time — zero false suggestions, zero harmful overrides.

2. **Validity anchored by Exp 89, not assumed.** The identical arms fired **5/17** on Exp 89's
   authored placebo/confirm controls and **27–43/81** on its primary — so they *can and do* fire.
   The ~70-min run for 200 scenes rules out a silent no-op (a broken arm returns in seconds). So
   0/200 is a genuine "the LLM chose restraint," not a wiring artifact.

3. **The primary is a NULL, and that is the correct outcome.** Debate cannot beat `single-pkt` on
   false suggestions when both sit at the floor. Debate's edge lives in text-dependent exceptions
   (Exp 89); on ordinary jobs there is nothing to reason about, so it neither helps nor harms —
   consistent with Exp 87's SLA-neutral result, now shown with a cleaner per-decision metric.
   (This retires the original "debate beats floor and single on the no-text trace" hypothesis: it
   ties them, at the floor.)

4. **Sensitivity/specificity pair.** Exp 89 = recall on real exceptions (43/81, p=0.0007); Exp 90
   = precision on non-exceptions (0/200 false). The machinery catches authored exceptions AND does
   no harm on real jobs it was never authored for — the direct answer to "your text cases are
   self-authored and few."

5. **Honest caveat — informativeness.** 0/200 is clean enough that "the real scenes are not
   adversarial" is a live alternative to "the machinery is restrained." The Exp 89 control contrast
   (**5/17 fired on authored tension vs 0/200 on real jobs**) is what makes the restraint reading
   credible — and simultaneously argues that hand-authored placebo scenes **overstate** the
   meddling risk. A harder-scene variant (real jobs perturbed toward tension) would stress-test
   restraint further; not built.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6
```
Built subagent-driven (sampler + driver, TDD, `pins/test_exp90_scenes.py` +
`pins/test_exp90_analysis.py`); results at `pins/results_exp90_qwen2514b.json`.

## Experiment 91 — HARD-SPECIFICITY: restraint survives efficiency bait, still 0/200 (2026-07-24)

Pre-reg: `docs/superpowers/specs/2026-07-24-exp91-hard-specificity-design.md`;
plan: `docs/superpowers/plans/2026-07-24-exp91-hard-specificity.md`.

Direct upgrade of Exp 90, answering its own caveat ("0/200 is clean enough that 'the real
scenes are not adversarial' is a live alternative to 'the machinery is restrained'"). Same
retain-market invariant (inelastic jobs, no text, neutral supply — every override is a false
suggestion by construction), same 4 arms, same one-sided McNemar primary. The **only** change is
the sampler bait: an **imbalance gate** (`spread_min=4` — reject scenes unless base spread ≥ 4,
forcing an ~8-GPU job beside 1–2-GPU jobs) and a **tight-slack** range (`slack=[0.05,0.30]` vs
Exp 90's `[0.15,0.60]`) so headroom is scarce and the scene reads as contested. Both are
default-off params on `sample_scenes`; with them unset Exp 90 reproduces byte-identically (guarded
by `test_defaults_reproduce_exp90` + the untouched `test_exp90_scenes.py`).

**Scenes (n=200, seed 0).** jobs/scene mean 4.08; free_gpus mean 21.7; **headroom mean 3.23**
(vs Exp 90's 5.74 — bait bites); **spread mean 6.24**; **menu_size mean 80.3, min 28** — every
scene offers ≥28 legal non-retain grant/transfer actions, so "change nothing" is a real choice
being declined, not the only option.

**Result.**
```
per-arm false-suggestion rate (n=200)          harmful     easy Exp90
  market          0/200                          0           0/200
  single-no-pkt   0/200                          0           0/200
  single-pkt      0/200                          0           0/200
  debate-pkt      0/200                          0           0/200
PRIMARY McNemar debate-pkt vs single-pkt (H1 debate fires LESS):
  b(debate-only)=0  c(single-only)=0   one-sided p=1.0000  two-sided 1.0000  -> tie at the floor
sanity: market & single-no-pkt = 0/200 (floor arms correct; min_menu=28, harness not a no-op)
```

**Findings.**

1. **Restraint is robust under bait — the Exp 90 caveat is answered.** With lopsided bases,
   scarce headroom, and ≥28 tempting actions per scene, all four arms retained the market on all
   200 real jobs. The 0/200 is no longer explainable by "unchallenging scenes": the scenes are
   demonstrably adversarial (menu mean 80.3) and the arms decline every time. This is a *stronger*
   zero than Exp 90's.

2. **The primary is again a NULL, and again the correct outcome.** `debate-pkt ≡ single-pkt` at
   the floor (b=c=0). Per §8 decision rule this is the `≈0/≈0` branch: the packet does not meddle
   even under bait, so the debate rebuttal has nothing to talk down. Consistent with Exp 87/90.

3. **Why the bait doesn't bite: inelasticity, not scene difficulty.** A 14b model reasoning over
   jobs that each requested exactly their base, with no free text, will not hand out GPUs nobody
   asked for — *visual* imbalance (a big job beside small ones) is not a signal it acts on. The
   bait made silence tempting to a greedy heuristic, not to a model that reads the (inelastic)
   declarations. So the harder test confirms restraint but still does not separate the arms;
   separation would require scenes where acting is genuinely warranted (a mixed precision+recall
   suite — explicitly out of scope here, it stops being the specificity mirror of Exp 89).

4. **Specificity pair, now stress-tested.** Exp 89 = recall on authored exceptions (43/81,
   p=0.0007); Exp 90 = precision on ordinary jobs (0/200); Exp 91 = precision on jobs **perturbed
   toward tension** (0/200, adversarial). The machinery catches authored exceptions AND does no
   harm on real jobs — including real jobs arranged to tempt it — using zero authored text.

**Honest caveat.** 0/200 under bait is a robustness win for the *specificity* claim, but it also
means this axis is now saturated for 14b: no bait short of an actual warranted action will move
these arms off the floor. Further separation is a recall/mixed-suite question, not a
harder-specificity one.

**Reproduce.**
```bash
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6 \
  --spread-min 4 --slack-lo 0.05 --slack-hi 0.30 \
  --easy pins/results_exp90_qwen2514b.json \
  --out pins/results_exp91_hard_qwen2514b.json
```
Built subagent-driven (sampler bait + driver flags, TDD, `pins/test_exp91_scenes.py`); results at
`pins/results_exp91_hard_qwen2514b.json`, log `pins/exp91_hard_14b.log`.

---

## Experiment 92 — THE BOUNDARY TEST: the winning escalation is a literal NO-OP in-sim (2026-07-28)

**Date:** 2026-07-28

**Question.** §4.4's boundary claim rested on Exp 84, where an in-sim debate cost −12.0\* prodSLA.
But that arm is a *different mechanism* from the one that wins the hard-case suite (see the
two-debates caveat on Exp 84): `referee.py` re-decides every job with no text gate, whereas
`correction_signed.debate_signed` only corrects jobs that carry a note. So Exp 84 conflated two
explanations — "no text to act on" and "a more invasive architecture". This experiment separates
them by running the **winning** architecture in-sim, unchanged.

**Method.** New arm `corrected` (`pins/market.py::make_policy_corrected`, `--corrected`):
validated auction every tick, and on the **identical** contextual trigger as `gated` (the trigger
was extracted to `_make_trigger()` so the two arms fire on exactly the same conditions) the
escalation runs `gather_signed → debate_signed → referee_signed → apply_signed` on top of the
market's allocation. `supply_note` is mapped to `""` **honestly** — the replay world has no text
channel (`Job` has no note field; no `ctx` key carries one), and synthesising one from the numbers
would manufacture the very channel this experiment exists to show is absent. `--llm` stays ON, so
a null result is caused by absent text and not by `gather_signed`'s `if not use_llm` early return.

qwen2.5:14b, caps=predicted, pool 8, **n=32 paired seeds**, v2020 replay.

**Findings.**

| pool | policy | SLA | prodSLA | util | useful | regret | slowdown | wait | fb |
|---|---|---|---|---|---|---|---|---|---|
| 8 | no-llm (floor) | 53.9% | 59.1% | 80% | 75% | 10% | 8.73 | 22.7 | 0% |
| 8 | market | 51.8%\* | 55.3% | 81% | 76% | 8% | 8.62 | 22.6 | 0% |
| 8 | **corrected** | **51.8%\*** | **55.3%** | **81%** | **76%** | **8%** | **8.62** | **22.6** | **0%** |

```
market       vs floor:  dSLA -2.1 ± 1.7*  dprodSLA -3.8 ± 3.1*  dutil +1.3 ± 0.5*  duseful +0.7 ± 0.4*  dregret -1.3 ± 0.8*
corrected    vs floor:  dSLA -2.1 ± 1.7*  dprodSLA -3.8 ± 3.1*  dutil +1.3 ± 0.5*  duseful +0.7 ± 0.4*  dregret -1.3 ± 0.8*
Holm (family of 30): market/util p=0.001, corrected/util p=0.001, market/regret p=0.003,
                     corrected/regret p=0.003, market/useful p=0.016, corrected/useful p=0.016
corrected: 843 escalations, 0 LLM calls, 0 proposals, 0 ticks changed, 0 rejected
```

- **`corrected` is identical to `market` on every metric, digit for digit, including the vs-floor
  deltas, their confidence intervals, and every surviving Holm p-value.** This is not a
  statistical tie — it is the same allocation, tick for tick. The `market` row also reproduces
  §4.3 of the paper (Exp 87: market 51.8% / 55.3% / 81%) on the same seeds.
- **The gate genuinely ran.** 843 escalations fired across the 32 seeds (matching the Exp-87 trigger
  rate); each one entered the correction pipeline, found no job carrying text, and returned.
- **Zero LLM calls.** Both the per-job reviewer loop and the supply call are text-gated, so with no
  text anywhere the escalation costs nothing at all — not even tokens.
- Reproduced at n=8 (193 escalations) and n=2 (48 escalations): same zeros throughout.

**What this buys.** The boundary claim no longer depends on Exp 84's −12.0\*. The correct statement
is now: *the mechanism that wins on text exceptions (43/81, p=0.0007, Exp 89) does exactly nothing
where there is no text — zero calls, zero changes, byte-identical allocation.* Exp 84's penalty is
re-read as the cost of the **more invasive** in-sim architecture replacing the auction, not as
evidence about text. §4.4 should lead with this and demote Exp 84 to an architecture ablation.

**Note on n.** Run at the project standard n=32 (4m36s). The effect is exact rather than
statistical — seeds buy robustness against seed-specific luck, not significance — but the paired
design matches every other in-sim result, so the arm drops straight into the §4.3 table.

**Reproduce.**
```
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b --caps predicted \
    --pools 8 --seeds 32 --market --corrected
```

## Experiment 93 — THE UNTESTED STRUCTURES: debate's win is the SECOND PASS, not the arguments (2026-07-28)

**Pre-registration:** `docs/superpowers/specs/2026-07-28-exp93-structures-on-packet-design.md`
(written **mid-run** at 7/98 cases, before any result table — disclosed there, §7).

**Question.** Open item 2. Exp 65–67 tested self-consistency, an argumentation ablation and a
critic arm, and found everything null — but all three ran the day *before* the decision packet
(`b77d3fa`), which took infeasible rulings from 11–16/31 to 0/31. Untested, not refuted. Two arms
were new (`selfcons` was **not** re-run: `single-pkt-boN` already *is* majority-vote
self-consistency, with an n=81 result in Exp 89):

- **`debate-noarg-pkt`** — full gather+debate, then `evidence` is stripped from proposals before
  `build_packet`. The referee sees WHO proposed WHAT but not WHY. Isolates argument *content* from
  the second pass. Identical call count to `debate-pkt` — the one clean contrast here.
- **`critic-pkt`** — a **reconstruction** of Exp 67's critic (the original was never committed).
  One objection-only reviewer per text-bearing job; objections reach the referee via
  `reviewer_proposals`. ~40% of debate's calls.

qwen2.5:14b, suite r34, POOLED primary n=81, CONTROLS n=17, STRICT scoring, exact McNemar,
one-sided per hypothesis, **Holm over the two pre-registered tests**.

**Findings.**

| arm | STRICT | calls |
|---|---|---|
| market | 0/81 | 0 |
| single-pkt | 27/81 | 81 |
| **debate-pkt** | **43/81** | 569 |
| **debate-noarg-pkt** | **43/81** | 569 |
| critic-pkt | 34/81 | 244 |

```
H1  debate > noarg   b=4  c=4   D=+0   1-sided p=0.6367   Holm vs 0.0500 -> ns
H2  critic > single  b=10 c=3   D=+7   1-sided p=0.0461   Holm vs 0.0250 -> ns
H1 equivalence (TOST, margin +/-3): D=+0, 90% CI [-4.91, +4.91] -> FAIL (m=8 too small)
EXPLORATORY, budget-confounded:  debate > critic  b=15 c=6  1-sided p=0.0392
harness check:                   debate > single  b=18 c=2  1-sided p=0.0002 (reproduces Exp 89)
CONTROLS n=17: market 16/17, single 12/17, debate 12/17, noarg 11/17, critic 12/17
```

- **H1 rejected, and by the strongest available form of a null: an exact tie.** Stripping the
  arguments out of the packet changes the score by **zero cases** (43 = 43, discordant 4-4).
  Debate's Exp 89 win is the **second pass**, not argument content — the pre-registered
  simplify-the-architecture branch (§6). The r4-only blind stratum agrees in direction and is also
  ns (29 vs 27, p=0.34).
- **Read the tie as evidence of a null, not as proven equivalence.** TOST at the project's ±3
  margin FAILS: only 8 discordant pairs, so the 90% CI is [−4.91, +4.91]. The point estimate is
  exactly 0 and the test is ns, but the data cannot *exclude* a ±3-case effect. Stated this way in
  the paper or not at all.
- **H2 does not survive multiplicity.** Critic 34/81 vs single 27/81 is p=0.0461 alone — and
  **0.0461 > 0.05/2**, so under the pre-registered Holm correction it is **not significant**. The
  honest statement is a suggestive, unconfirmed cost result: 34 vs 27 at 3× the calls, direction
  consistent, not established. A budget-matched critic arm is the follow-up, and it is now a
  cheaper question than it looked at 18:50.
- **Debate still beats critic** (15-6) but at 2.3× the calls, so per §5 this may not be claimed as
  a structure comparison in either direction.
- **Specificity cost reproduces**: market 16/17 vs every LLM arm 11–12/17 (claim 9).
- STRICT and bare-handled tables are identical — **zero infeasible rulings in every arm**, the
  packet holding as in Exp 82.

**What this changes.** Claim 5 stands untouched (this experiment can only add arms). What it
removes is a *supporting* reading of it: the debate transcript is not what the referee is using,
which weakens the interpretability framing in §5 for this mechanism specifically — the transcript
remains readable, but its content is not load-bearing for the decision. The architectural payoff is
that `evidence` can be dropped from the packet at no measured cost.

**Reproduce.**
```
PINS_RESULTS=pins/results_exp93_qwen2514b.json \
  .venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b --suite r34 \
    --arms market,single-pkt,debate-pkt,debate-noarg-pkt,critic-pkt
.venv/bin/python -m pins.exp93_analyse pins/results_exp93_qwen2514b.json
```
`pins/exp89_analyse.py` does **not** work on this run — it tests against `single-pkt-boN`, which is
not an arm here. `pins/exp93_analyse.py` carries the pre-registered axes.

## Experiment 94 — THE AGENT-AUTHORED TEXT CHANNEL: generated text is inert in-sim (2026-07-29)

**Pre-registration:** `docs/superpowers/specs/2026-07-28-exp94-agent-authored-text-channel-design.md`,
written before any code (commit `e0a2b0b`); arm + harness test committed before the run (`01910ea`).
**Predicted outcome: EQUIVALENCE (TOST ±2.0 SLA pts).** Confirmed, and more strongly than stated.

**Question.** Exp 92 showed the winning escalation is a literal no-op in-sim because the v2020
replay carries no operator text, so §4.4 rests on an authored suite and §5 concedes the limitation.
Can the text channel be **generated rather than fabricated**? When state changes materially between
t−1 and t, the demand and supply agents report in plain text *why their situation changed*, and the
referee rules on the packet as it already does. This is not code synthesising operator notes from
public numbers — §4.4 declines that, and it would measure a translation, not a channel.

**The gap it exploits.** The market has one-step *allocation* memory (`C_resize` prices change from
current holdings) and no *causal* memory: a margin lost to a prod arrival and one lost to a downward
usage revision are the same number to it.

**Method.** New arm `make_policy_authored` (`pins/market.py`, `--authored narrated,attributed`),
implemented as an `authored=` parameter on `make_policy_corrected` so the Exp 92 arm stays
byte-identical. Notes land in the same `demand_claims[].note` / `supply_claim.note` fields the
authored suite uses, so the mechanism validated in Exp 82–89 is untouched — only the *source* of the
text changes. Three arms, paired within seed, pool 8, `caps=predicted`, qwen2.5:14b, v2020 replay,
n=32.

| arm | note content | authoring budget |
|---|---|---|
| `market` | none | 0 |
| `narrated` (**placebo**) | states **what** changed; forbidden by prompt from stating why | matched |
| `attributed` | states **why** it changed | matched |

**The placebo is the design.** Both notes are authored by the same model on the same trigger and the
same per-job authoring condition (held margin moved, deadline bucket moved, or arrived this tick;
supply always speaks on a fired tick). Without it, a win could not be separated from *"the LLM was
handed cross-tick history the market lacks"* — a finding about `BID_W`, not about text. What the
agents may see is pinned in code: previous allocation, previous free GPUs, previous deadline bucket,
own previous request. Nothing else.

**Findings.**

| pool | policy | SLA | prodSLA | util | useful | regret | slowdown | wait | calls/seed | tokens/seed |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | no-llm (floor) | 53.9% | 59.1% | 80% | 75% | 10% | 8.73 | 22.7 | 0 | 0 |
| 8 | market | 51.8%\* | 55.3% | 81% | 76% | 8% | 8.62 | 22.6 | 0 | 0 |
| 8 | **narrated** | **51.8%\*** | **55.3%** | 81% | 76% | 8% | 8.62 | 22.6 | 51.9 | 11,679 |
| 8 | **attributed** | **51.8%\*** | **55.3%** | 81% | 76% | 8% | 8.62 | 22.6 | 69.8 | 17,774 |

```
H1  attributed - narrated, SLA:  +0.0000, 0/32 seeds differ   -> EQ±2 passes DEGENERATELY (sd=0)
H2a attributed - market,   SLA:  +0.0000, 0/32 seeds differ
H2b narrated   - market,   SLA:  +0.0000, 0/32 seeds differ
secondary family (4 metrics x 3 contrasts, Holm): every adjusted p = 1.000; min raw p = 0.285
narrated:   835 escalations, 1360 notes, 1662 calls,   9 proposals, 2 ticks changed,   1 rejected
attributed: 842 escalations, 1385 notes, 2232 calls, 156 proposals, 4 ticks changed, 108 rejected
```

- **Decision rule branch 1: the channel is inert in-sim.** `sla`, `prod_sla` and `finished` are
  identical *seed for seed at full float precision* across market/narrated/attributed. This is
  stronger than the pre-registered equivalence: not "within ±2 points" but **exact identity of every
  SLA outcome on every seed**.
- **The channel is not inert at the allocation level — only at the outcome level.** Rounding in the
  run log hid this. `util` differs from `market` on seed 13 (both arms); `u_useful` and `regret`
  differ on seeds 13/22/27 for `attributed`, seed 13 for `narrated`. The text moved the allocation
  on 4 of ~842 escalated ticks and flipped no job's outcome anywhere.
- **The manipulation landed.** Read off the persisted cache: causal markers in 0/371 narrated demand
  notes vs **353/378 (93.4%)** attributed; 34/632 vs 437/635 (68.8%) on the supply side. On identical
  state payloads the two modes produced the same sentence **0/368** and **0/628** times. The placebo
  held its prohibition essentially perfectly.
- **Branch 3 ("the gain is history, not text") is excluded conclusively**, not by failure-to-reject.
  Branch 3 requires *both arms > market\**; both arms equal market exactly. The agents were handed
  precisely the cross-tick state `BID_W`/`C_resize` lack, and it bought nothing. **§4.1 does not
  change.**
- **The cost of the causal channel is interventionism the guard absorbs.** Attributed's reviewers
  raised **156 proposals against narrated's 9** — a 17× increase — and ~96% of its proposal-bearing
  ticks produced a judgement `apply_signed` rejected outright (108 rejected). Causal text makes the
  referee far more willing to act; the feasibility guard catches nearly all of it. Which of the four
  rejection rules fired is **not recorded** (`market.py` counts, never logs, the violation strings).
- The `dSLA −2.1 ± 1.7*` vs floor is inherited identically from `market` — a pre-existing market
  property, not a cost of the channel. Not branch 4.

**Budget divergence — the design-level match held; the rest is downstream of the treatment.** Notes
per escalation matched to **1.0%** (1.629 vs 1.645), which is what §3 promises. Total calls diverged
34% and tokens 52%, from three verified causes: `llm_calls` counts *cache misses* and attributed's
notes are 2.6× longer and more varied (§7's own "cache dilution" prediction), `referee_signed` fired
on ~112 ticks vs ~3, and the 7-escalation gap is itself an outcome — once 2/4 ticks changed the
allocation, `_make_trigger`'s carried state diverged for the rest of those seeds. **This cannot
manufacture the null: the extra budget went to the treatment arm** (+17.8 ± 3.0 calls/seed on 31/32
seeds), which still produced an SLA vector identical to the cheaper arm and to the zero-token market.

**Threat that was NOT met.** §7 pinned *prompt* symmetry, and that held (same role sentence, same
one-sentence instruction, same prohibition clause, same JSON contract). *Output* symmetry was never
controlled and did not hold: demand notes average 10.9 words narrated vs 28.2 attributed. So H1 is
"causal-and-longer vs factual-and-shorter", not a pure causality contrast. Immaterial to a
zero-variance null; **would have to be controlled before any positive result could be read as
causality per se.**

**What this buys.** The honest limitation in §5 gets sharper rather than smaller. Exp 92 showed the
mechanism does nothing when there is *no* text; Exp 94 shows it does nothing when text is *generated
from the simulator's own state* — because, as §7 predicted, in a simulator the causes of a change
are largely recoverable from the numbers already logged. The remaining case for the text channel
therefore rests entirely on text carrying facts the state does **not** contain (operator
instructions, contractual limits, suspensions), which is exactly what the authored suite tests and
what no public GPU trace ships. The generated-channel route is closed; the authoring bottleneck
stands.

**Caveats.** Degenerate TOST (sd = 0) — report as exact identity, not as a powered null; the
`p = 1.000` values are the harness's `"degenerate"` sentinel, not computed tests. Single pool, single
model, same model authors and rules. Secondary-family Holm composition was the analyst's choice, not
pinned by the pre-reg (non-determining: min raw p = 0.285). Transcripts for the two arms were not
written to the `decisions` block; note texts survive only in the cumulative `llm_agent_cache.json`.

**Reproduce.**
```
.venv/bin/python -m pins.test_exp94_authoring        # budget match + cache-tag disjointness
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b --caps predicted \
    --pools 8 --seeds 32 --market --authored narrated,attributed
```
Both modes belong in ONE tier (`+authored-narrated-attributed`) so the placebo contrast is paired
within seed. Trap: `correction._ask` keys the cache on `(tag, user, model)` and **not** on the system
prompt — the tags are `note-{mode}-{dem,sup}` for that reason, and the harness test asserts they stay
disjoint while the payloads stay identical.

## Experiment 95 — THE BUDGET-MATCHED CRITIC: the objection channel survives its control (2026-07-28)

**Pre-registration:** `docs/superpowers/specs/2026-07-28-exp95-budget-matched-critic-design.md`,
committed at `ea2de32` **before the arm existed**; arm and analyser at `8876f99`, **before the run**.

**Question.** Exp 93 left critic at 34/81 vs `single-pkt` 27/81, p=0.0461 — which **failed Holm** and
was budget-confounded by construction (244 calls vs 81). Exp 93 §5 named this follow-up in advance.
Exp 88/89 settled the budget question at **debate's** 7×; critic's 3× is a different point on the
curve, and critic's whole interest is that it might be the cheap structure.

**Method.** New arm **`single-pkt-boNc`**: the existing best-of-N control with `k = n_text_jobs + 1`
— critic's per-case call count exactly, from the same predicate `_critic_signed` uses. Matched
**per case**, not in total, and the matching is **verified rather than assumed**
(`exp95_analyse.check_budget`). One pre-registered test, so no multiplicity correction; that
asymmetry with Exp 93 was recorded in the spec before the run. qwen2.5:14b, suite r34, STRICT.

**Findings.**

| arm | STRICT | calls |
|---|---|---|
| market | 0/81 | 0 |
| single-pkt | 27/81 | 81 |
| **single-pkt-boNc** | **26/81** | **244** |
| **critic-pkt** | **34/81** | **244** |

```
BUDGET MATCH: matched exactly on every one of the 98 cases; 292 calls each
PRIMARY  critic > boNc          b=10 c=2  D=+8  1-sided p=0.0193   -> H1 SUPPORTED
         equivalence: 90% CI [+1.49, +11.27], margin +/-3 -> FAIL (m=12)
SECONDARY boNc > single         b=0  c=1  D=-1  p=1.0000
         critic > single        b=10 c=3  D=+7  p=0.0461  (reproduces Exp 93 exactly)
BLIND r4 critic > boNc          b=5  c=2  D=+3  p=0.2266  (ns, direction agrees)
CONTROLS n=17: market 16/17, single 12/17, boNc 12/17, critic 12/17
harness check: critic 34/81 MATCHES Exp 93, single 27/81 MATCHES, b=10 c=3 MATCHES
```

- **H1 supported.** Critic beats the sampler at identical spend, 34 vs 26, p=0.0193. Exp 93's
  Holm-failing lead promotes to a result — and against a *stronger* comparator than the one that
  failed, since a budget-matched control is the comparison that was actually in question.
- **The secondary is the cleanest line in the experiment.** `boNc` spends 244 calls to score
  **26/81 — one case WORSE than plain `single-pkt` at 81 calls**. Resampling at critic's budget buys
  nothing. This is the **third independent replication** of "budget alone buys nothing" (Exp 88 at
  debate's 7×, Exp 89 at n=81, Exp 95 at critic's 3×), and it is what makes the critic number mean
  something rather than being a spend artefact.
- **The blind stratum does not confirm it.** r4-only is 20 vs 17, D=+3, **p=0.227**. Direction
  agrees and nothing contradicts, but the pooled win is carried disproportionately by r3 (which
  contributes b=5 c=0). This is **weaker confirmatory support than Exp 89 had**, and the claim
  should be stated at that strength.
- **TOST fails as pre-declared** — 12 discordant pairs give a 90% CI of [+1.49, +11.27]. A real
  effect, badly bounded. The lower bound excluding 0 is the useful part.
- **Critic is the cheaper structure, not the better one.** Debate remains 43/81 at 569 calls, and
  Exp 93's debate-vs-critic was 15-6. The ordering on this suite is
  **debate (43, 7×) > critic (34, 3×) > single (27, 1×) ≈ boNc (26, 3×)**.
- **Specificity cost is flat across every LLM arm** (12/17 vs market's 16/17) — critic buys its
  recall without buying extra false positives, and without reducing them either.
- Determinism confirmed: critic at temperature 0 reproduced Exp 93 digit for digit, including the
  discordant pattern b=10 c=3.

**What this changes.** A structure other than debate now survives a budget control on this suite —
the first one. Read with Exp 93's H1 tie (stripping arguments from debate costs zero cases), the
picture is: **extra passes help, extra samples do not, and argument content does not.** What the
objection channel and the debate second pass share is that a *separate call looks at the allocation
with a different instruction*, which is a cheaper thing to claim than "argumentation".

**Open.** Critic vs debate is still budget-confounded in both directions (Exp 93 §5). The clean
question left is a **debate arm cut down to critic's budget**, or equivalently a critic scaled up to
debate's — that is what would separate "more passes" from "which structure".

**Reproduce.**
```
PINS_RESULTS=pins/results_exp95_qwen2514b.json \
  .venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b --suite r34 \
    --arms market,single-pkt,single-pkt-boNc,critic-pkt
.venv/bin/python -m pins.exp95_analyse pins/results_exp95_qwen2514b.json
```
**Trap.** The harness's default output path is `results_{exp88|exp89}_{model}_t{temp}.json`, which
collides with committed Exp 88/89 artefacts — a `--no-llm` smoke test silently overwrote
`results_exp89_qwen2.514b_t0.8.json` during this experiment (restored from git). Always set
`PINS_RESULTS`.

## Experiment 96 — THE TIER/LAXITY DE-CONFOUND: the confound was real, and the two laws disagree (2026-07-29/30)

**Pre-registration:** `docs/superpowers/specs/2026-07-29-exp96-tier-laxity-deconfound-design.md`,
committed at `0ba20ee` **before any code existed**; the `--decorrelate` flag and the `tight_sla`
metric at `a4acb11`, **before the run**. The analyser (`225d256`) was written **after** the amdahl
runs finished — disclosed; it implements spec §5–6 verbatim, with no test added or dropped after
seeing data.

**Question.** Claim 2 says the reasoning layer's reproducible contribution is *production-tier
protection*. `trace_replay.py:211-214` drew tier and deadline slack from **one** uniform: `prod` ⟺
`urgency ≥ 1.667` ⟺ `slack ∈ [1.15, 1.42]`. So no experiment to date could separate "the prod tier
was protected" from "the tightest deadlines were served". This separates them.

**Method.** `--decorrelate` redraws the tier label independently at the same marginal
(`P(prod) = 1/3`) from **its own RNG stream**, so the two worlds share byte-identical jobs and differ
only in which ones are labelled prod — the diff-in-diff is exact, not statistical. New metric
`tight_sla` = violation rate over the tightest-laxity tercile, `(deadline − arrival) / work`. Exp
73's configuration exactly: `--referee --market --composed --llm --model qwen2.5:14b --caps
predicted --pools 8 --seeds 32`, both laws, n=32 paired seeds.

**H3 — the confound, measured rather than asserted (amdahl).** Regenerating both worlds' windows
through `make_trace_workload`:

| world | mean laxity, prod | mean laxity, besteffort | point-biserial r | prod/window |
|---|---|---|---|---|
| correlated | 1.198 | 1.770 | **−0.778** | 5.53 (3–10) |
| decorrelated | 1.564 | 1.576 | **−0.017** | 5.91 (3–12) |

Jobs are **byte-identical across worlds** (arrival/work/deadline), so the pairing the DiD needs
holds. At the floor, `tight_sla` tracks `prod_sla` seed-wise at r = +0.889 correlated and +0.442
decorrelated. The confound was real and the manipulation removed it.

**H1 — protection survives de-confounding (amdahl, 8 GPUs, violation rates, lower is better).**

| arm | dprodSLA correlated | dprodSLA decorrelated | DiD (decorr − corr) |
|---|---|---|---|
| referee | −7.6 ± 5.0\* | −5.5 ± 3.9\* | +2.1 ± 5.7 (p=0.530) |
| negotiated | −6.6 ± 4.6\* | −4.7 ± 3.5\* | +1.9 ± 4.9 (p=0.378) |
| market | −3.8 ± 3.1\* | −3.3 ± 2.8\* | +0.5 ± 3.3 (p=0.575) |
| **composed** (primary) | −5.6 ± 4.1\* | **−3.4 ± 3.2\*** | **+2.2 ± 4.3 (p=0.235)** |

Read alone, amdahl is **branch 1**: every arm still protects the prod tier when the label no longer
implies a tight deadline, and no diff-in-diff is distinguishable from zero. The point estimates all
drift the predicted way (~1–2 pts smaller), so this is "no measurable shrinkage", not "provably
none" — the DiD CIs are ±3 to ±6 pts wide. **The sat law does not agree, and it is the law where
the correlated effect was strongest.**

**H1 — the sat law: the strongest cell does not survive de-confounding.**

| arm | dprodSLA correlated | dprodSLA decorrelated | DiD (decorr − corr) |
|---|---|---|---|
| referee | −4.6 ± 3.4\* | −7.8 ± 6.7\* | −3.2 ± 7.0 (p=0.500) |
| negotiated | −7.9 ± 4.4\* | −3.3 ± 4.5 | +4.6 ± 6.2 (p=0.196) |
| market | −0.7 ± 2.1 | −0.6 ± 1.6 | +0.1 ± 2.8 (p=1.000) |
| **composed** (primary) | **−8.0 ± 4.5\*** (Holm 0.017) | **−2.7 ± 4.1 (ns)** (Holm 0.584) | **+5.3 ± 5.8 (p=0.053)** |

Under sat, `composed` is the one cell in this whole experiment whose protection **survived Holm**
in the correlated world (adj p=0.017, and `trace_replay`'s own 24-test family agreed at p=0.041).
De-correlating the label removes it: −2.7 ± 4.1, Holm adj p=0.584.

**Verdict: between branch 2 and branch 3, and n=32 cannot separate them.** Stating "significant in
the correlated world, null in the decorrelated one" as if that were a finding would be the
difference-of-significance fallacy; the DiD is the correct test and it lands at **+5.3 ± 5.8
(p=0.053)** — the right sign and size for branch 2 (protection shrinks) but not significant, in the
law that shows it, while amdahl's DiD sits at +2.2 ± 4.3. So the experiment establishes that the
confound was real and removable, and **fails to resolve how much of claim 2 it was carrying**.
Claim 2 is not refuted and is no longer clean: its strongest cell does not survive the de-confound.

**Power, concretely.** The DiD standard deviations are ~12 pts (amdahl) and ~16 pts (sat), so a
±2.5-pt DiD CI needs roughly **n≈95 (amdahl) and n≈175 (sat)** paired seeds. At ~45 min per 32
seeds for the referee arm that is an overnight run per world per law — the cheapest available way
to close this, and the one thing that would settle claim 2's wording.

**The pre-registration did not say how to combine the two laws.** Spec §4 asked for both and §6
wrote the branches as if one verdict would come back. It did not. Reporting both, with neither
promoted, is the disclosure; no law was dropped and no combination rule was invented after the
fact.

**H2 — nothing protects the deadline, in either law.**

amdahl:

| arm | d tight_sla correlated | d tight_sla decorrelated |
|---|---|---|
| referee | −5.6 ± 3.3\* | **+2.5 ± 4.0** (worse, ns) |
| negotiated | −5.6 ± 3.8\* | −1.3 ± 4.1 |
| market | −3.1 ± 2.7\* | −3.1 ± 2.7\* |
| composed | −5.0 ± 3.7\* | −2.5 ± 4.0 |

sat:

| arm | d tight_sla correlated | d tight_sla decorrelated |
|---|---|---|
| referee | −3.8 ± 3.9 | **+6.9 ± 5.1\*** (significantly **worse**) |
| negotiated | −6.9 ± 4.3\* | +0.0 ± 2.6 |
| market | +0.0 ± 1.8 | +0.0 ± 1.8 |
| composed | −7.5 ± 4.4\* | +0.0 ± 3.2 |

In the correlated world every arm "protects the tight tercile" — but that stratum *is* the prod
stratum there (H3), so it says nothing. Once laxity is independent, **no reasoning arm protects it
in either law**, every deterministic arm goes exactly flat under sat, and the referee's sign flips —
to significantly worse under sat (+6.9 ± 5.1\*), i.e. giving the tightest-deadline jobs to a
reasoning layer that cannot see laxity actively costs them. Stronger, and true in **both** laws: the
market's tight-tercile effect is **identical seed-for-seed across the two worlds** (vector equality,
not equal means), i.e. it does not run through the tier label at all, while every LLM arm's is
label-dependent. Branch 4 therefore holds
for the reasoning arms: nothing they do serves laxity, which is what motivates least-laxity grant
ordering (`two_sided_sim.py:454` still orders by frozen bid value) as the next lever.

**Multiplicity, stated plainly.** Under amdahl, Holm across the 8-test prodSLA family leaves only
`correlated/referee` (adj p=0.026) and puts the primary arm at adj p=0.093 in **both** worlds;
`trace_replay`'s own 24-test family kills every amdahl prodSLA star in both worlds. Under sat the
correlated world keeps `composed` (0.017) and `negotiated` (0.017) and the decorrelated world keeps
nothing. So amdahl's stars are nominal throughout — which is why its DiD null is weak evidence —
while sat's are the real ones, and they are the ones the de-confound removes.

**What this changes.** Claim 2 goes from *solid* to **qualified**: prod-tier protection is real in
the correlated world (that has not changed), but the workload that produced it made tier and
deadline tightness the same variable, and in the law where the effect was strongest it does not
survive their separation. The DiD cannot yet say whether the loss is real, so the claim should be
stated with both worlds' numbers, not restated as a clean tier result. Claims 1, 3, 4, 5 and 9 are
untouched, as pre-registered. The new negative — **laxity is unserved by every reasoning arm, and
under sat the referee actively harms the tightest tercile** — is a fresh, pre-registered result
rather than a caveat, and it is the same shape as claim 9's specificity cost: reasoning applied to a
stratum it cannot perceive is worse than not reasoning.

**Caveats.** Single pool (8 GPU), single model (qwen2.5:14b), the two laws disagreeing and n=32 too
small to adjudicate between them (see the power note). Tier still drives
grant precedence structurally (`two_sided_sim.py:456-457`) — deliberately, so this de-confounds the
*deadline*, not the *precedence*. `urgency` still drives the bid, so a decorrelated prod job also
loses its implicit bid advantage: two things change for it, not one (spec §7 pinned this in
advance). Prod count per window is binomial in both worlds (3–12 of 16), so `prod_sla` denominators
are small and noisy.

**Reproduce.**
```
PINS_RESULTS=pins/results_exp96_amdahl.json .venv/bin/python -m pins.trace_replay \
  --referee --market --composed --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32
PINS_RESULTS=pins/results_exp96_amdahl.json .venv/bin/python -m pins.trace_replay \
  --referee --market --composed --llm --model qwen2.5:14b --caps predicted --pools 8 --seeds 32 \
  --decorrelate
.venv/bin/python -m pins.exp96_analyse pins/results_exp96_amdahl.json
```
Add `--law sat` to both runs with `PINS_RESULTS=pins/results_exp96_sat.json` for the second law, then
`.venv/bin/python -m pins.exp96_analyse pins/results_exp96_sat.json`.

Both worlds must live in ONE results file (the analyser pairs the `+decorr` tier against its base
tier); `--law sat` needs its own file because the tier name does **not** encode the law — running
sat into the amdahl file would silently overwrite the amdahl rows under the same tier key, which is
the Exp 59 unpaired-comparison trap in a new costume.

## Experiment 97 — THE PRACTICAL OPERATING POINT: the deadline recipe was infeasible, and the rebase (2026-07-30)

**Type: calibration + rebase, not a hypothesis test** (Exp 62 precedent). No pre-registration; the
choice of operating point is an input to the headline experiments, not a claim, and it is recorded
here in full so the choice can be argued with.

**The problem.** `trace_replay.py:215-216` sets a deadline against the job's **solo** runtime —
`slack ∈ [1.15, 2.4] × work` — while the sim runs at 76–89% utilisation, where queue delay routinely
exceeds a job's own runtime. Missing half the deadlines is therefore forced by the recipe before any
policy runs: the amdahl floor violates **53.9%** of deadlines and the sat floor **39.5%**. This was
diagnosed inside the Exp 63 build note on 2026-07-22 (*"slack 1.55× a job's own work (p10 = 1.00×) is
unmeetable in a shared queue… ~8 points of dynamic range on a ~55-point floor"*) and never acted on.
A second, smaller defect sat under it: at horizon 300 about **2.5% of jobs never finish** and count
as violations by censoring alone — a constant floor no policy can move.

**Calibration sweep (floor arm, pool 8, n=8 seeds).**

| n_jobs | slack | horizon | SLA | prodSLA | util | done |
|---|---|---|---|---|---|---|
| 16 | 1× | 300 | 45.3% | 52.4% | 76% | 15.6/16 |
| 16 | 2× | 300 | 21.1% | 21.6% | 76% | 15.6/16 |
| 16 | 4× | 300 | 16.4% | 14.0% | 76% | 15.6/16 |
| 16 | 8× | 300 | 12.5% | 8.8% | 76% | 15.6/16 |
| 16 | 16× | 300 | 9.4% | 8.8% | 76% | 15.6/16 |
| 10 | 4× | 300 | 11.2% | 2.5% | 67% | 10/10 |
| 16 | 4× | **400** | 15.6% | 14.0% | 76% | **16/16** |
| 16 | 8× | **400** | 11.7% | 8.8% | 76% | **16/16** |
| 16 | **10×** | **400** | **9.4%** | **8.8%** | **76%** | **16/16** |
| 16 | 12× | 400 | 9.4% | 8.8% | 76% | 16/16 |
| 20 | 8× | 400 | 20.0% | 17.2% | 88% | 20/20 |

Three readings. **(a)** slack 1× is a cliff, not a setting — one step to 2× takes violations 45% →
21%, and the headline config sat on the worst point of the curve. **(b)** horizon 400 drains the
censored jobs (16/16 done) and 600 is identical to 400, so 400 is sufficient. **(c)** past slack 8–10×
the floor's prod violations **stop falling at 8.8%**: those jobs are not late, they are **starved**,
and no amount of deadline slack reaches them — which is exactly the failure a mechanism can fix
(`negotiated` at slack 10× takes prod to 2.1%\* in the n=8 probe).

**Chosen operating point.** `--pools 8 --n-jobs 16 --horizon 400 --slack-mult 10 --caps predicted`:
utilisation 76–78%, **every job finishes**, floor prod attainment ~87–90%, and the queue still binds
(wait 19–23 ticks). It keeps the contention that makes scheduling matter instead of buying a pretty
SLA by unloading the cluster (n=10 reaches 2.5% prod violations but only at 67% util).

**Infrastructure** (`47e0007`, all no-ops at the defaults): `--horizon T`; the **arrival span pinned**
to 180 ticks so the horizon adds drain without thinning arrivals; the horizon added to the tier name
so a rebase cannot land on the 300-tick rows. Default runs verified byte-identical.

**Stage 1 — the zero-token arms rebased (n=32, both laws).**

| law | policy | SLA | prodSLA | tight | util | useful | regret | wait |
|---|---|---|---|---|---|---|---|---|
| amdahl | no-llm | 12.9% | 12.9% | 11.3% | 78% | 74% | 10% | 23.0 |
| amdahl | negotiated (rule) | 10.5%\* | 7.9%\* | 8.1% | 77% | 73% | 12% | 21.2 |
| amdahl | market | 12.9% | 12.9% | 11.3% | 80% | 75% | 9% | 23.0 |
| sat | no-llm | 10.9% | 10.4% | 9.4% | 77% | 71% | 10% | 19.5 |
| sat | negotiated (rule) | 9.2%\* | 6.1%\* | 5.6% | 75% | 70% | 12% | 16.7 |
| sat | market | 10.7% | 9.9% | 8.8% | 79% | 72% | 9% | 19.4 |

```
amdahl  negotiated vs floor:  dSLA -2.3 ± 2.1*  dprodSLA -5.0 ± 3.6*  dutil -1.0 ± 0.8*  dregret +1.5 ± 0.7*
amdahl  market     vs floor:  dSLA +0.0 ± 0.0   dprodSLA +0.0 ± 0.0   dutil +1.6 ± 0.8*  duseful +1.1 ± 0.7*  dregret -1.5 ± 0.9*
sat     negotiated vs floor:  dSLA -1.8 ± 1.7*  dprodSLA -4.3 ± 3.7*  dutil -1.5 ± 1.2*  dregret +1.9 ± 0.8*
sat     market     vs floor:  dSLA -0.2 ± 0.4   dprodSLA -0.4 ± 0.9   dutil +2.6 ± 1.1*  duseful +1.2 ± 1.1*  dregret -0.8 ± 0.5*
```

- **The market's deadline effect is exactly zero under amdahl** (`+0.0 ± 0.0`, not a small ns
  estimate) and ns under sat, while its efficiency gains hold at every metric. Claim 1 survives the
  rebase and gets **sharper**: the market buys throughput and moves no job across a deadline.
- **The margin arms gain a real SLA effect that they did not have before** — overall −2.3\*/−1.8\*
  where the old operating point gave ns, and prod −5.0\*/−4.3\* — for a −1.0/−1.5 utilisation cost.
  Mechanically this follows from (c): with generous deadlines the only violators are starved jobs,
  and margin/reserve is what un-starves them. Under sat the prodSLA stars do **not** survive Holm
  (only util/useful/regret do); under amdahl they are reported nominal pending the same check.
- Efficiency and protection therefore look **complementary at this operating point** — the market
  and the reserve move disjoint sets of jobs — where claim 3 read them as one dial. That claim is
  now explicitly open and is the first thing the LLM arms will test.

**Stage 2 (running).** `--referee --market --composed --llm --model qwen2.5:14b`, both laws, n=32 —
the Exp 72/73 frontier at the practical point. Nothing is cached at a new operating point, so the
referee arm runs cold.

**What this invalidates.** Every in-sim **SLA / prodSLA** number in Exp 63, 70–73, 84, 86, 87, 92, 94
and 96 was measured at the infeasible point and must be re-read as such. The **efficiency** results
(util, useful util, regret) never depended on the deadline recipe and are unaffected. The
**hard-case suite** results (Exp 79–83, 88, 89, 93, 95) are not in-sim at all and are untouched.

**The standard operating point is now the DEFAULT** (2026-07-30, after the ladder below). Running
`trace_replay` with no scale flags gives **pool 32 quanta = 8 GPUs, 96 jobs (auto at 3× pool),
horizon 400, slack ×10** — floor 80% utilisation, 11.7% violations, 96/96 finished. Pools were
always in quarter-GPU quanta and the old `4,6,8` default was **1–2 physical GPUs**; the progress
line now prints `32q/8gpu` instead of the misleading `8gpu`. Pre-Exp-97 tiers reproduce exactly with
`--pools 8 --n-jobs 16 --horizon 300 --slack-mult 1` and keep their old names (verified byte-identical:
45.3/52.4/52.5/76/73/16.6/15.6 at n=8, tier `rule+pred`); every other run is tagged
`+h400+slack10+n3x` so old and new can never merge.

**The scale ladder that chose it** (floor arm, 4 seeds, slack ×10, horizon 400):

| pool | GPUs | jobs | ratio | util | SLA | prodSLA | wait | done |
|---|---|---|---|---|---|---|---|---|
| 8 | 2 | 16 | ×2 | 78% | 12.9% | 12.9% | 23.0 | 16/16 |
| 32 | 8 | 64 | ×2 | 64% | 4.3% | 2.1% | 9.4 | 64/64 |
| 32 | 8 | **96** | **×3** | **80%** | **11.7%** | **2.1%** | **42.0** | **96/96** |
| 32 | 8 | 128 | ×4 | 91% | 18.9% | 3.6% | 63.7 | 121.8/128 |
| 64 | 16 | 128 | ×2 | 63% | 2.5% | 0.5% | 7.3 | 128/128 |
| 120 | 30 | 240 | ×2 | 62% | 1.1% | 0.3% | 4.8 | 240/240 |
| 120 | 30 | 360 | ×3 | 71% | 10.1% | 0.0% | 34.1 | 359.8/360 |
| 120 | 30 | 480 | ×4 | 86% | 20.5% | 1.4% | 65.2 | 469.2/480 |
| 120 | 30 | 600 | ×5 | 93% | 36.5% | 2.2% | 110.7 | 488.2/600 |

- **Load must scale with the pool or contention evaporates** — at a fixed ×2 ratio, utilisation
  falls 78% → 64% → 62% as the cluster grows and violations collapse to 1%, which is statistical
  smoothing, not a scheduler doing well. **×3 is the recipe**: the last ratio at which every job
  still finishes at every scale.
- **Compute is not the constraint for the deterministic arms**: 4 seeds × 5 policies costs 10.9 s
  at 8 GPUs and 15.1 s at 30 GPUs.
- **Prod protection has less and less headroom as the cluster grows** — floor prod violations are
  12.9% at 2 GPUs, 2.1% at 8, and 0.0% at 30. Claim 2 is about exactly this quantity, so the scale
  at which it is measured is part of the claim. Recorded now, before the arms are re-measured.
- **The per-job referee is the arm that does not scale**: its token cost is ∝ jobs × ticks (Exp 64
  died at 89 jobs), while `composed` is one call per tick **regardless of job count** and
  market/negotiated are zero. At 8 GPUs the standard point has 96 jobs, i.e. 6× the pool-8 prompt.
- No batch scheduler is reachable from this node (`sbatch`/`qsub`/`pjsub`/`bsub` all absent), so
  long runs must be sharded with `--seed-start` to survive the reaper.

**Stage 2 was killed** mid-run at the old pool-8 point rather than finishing a measurement of a
superseded configuration; the frontier re-runs at the standard point instead.

**Tick shortened 120 s → 30 s (2026-07-30), as a pure resolution change.** A job's median lifetime
is now ~36 decision points instead of ~9. Three workload constants were denominated in *ticks*, so a
naive change would have quartered the arrival window, the simulated span and the longest
representable job all at once; they are now in **seconds** (`ARRIVAL_SPAN_S` 6 h, `HORIZON_S` 13.3 h,
`WORK_CLAMP_S` 2 min–2 h) and `--horizon 0` derives the tick count. Four *thresholds* were also
wall-clock intentions written in ticks — `STARVE_TICKS` (60 min), `TTF_HORIZON` (4 min), `DYN_AFTER`
(6 min) and referee's `STARVE_WAIT_TICKS` (20 min) — and `two_sided_sim.set_tick()` restates them,
a no-op at 120 s.

| | tick 120 s | tick 30 s |
|---|---|---|
| floor SLA | 11.7% | 12.0% |
| floor prodSLA | 2.1% | 2.8% |
| util | 80% | 80% |
| slowdown | 11.62 | 11.10 |
| wait | 42.0 ticks = **84 min** | 162.4 ticks = **81 min** |
| finished | 96/96 | 96/96 |

The same physical workload, sliced finer — which is the check that the refactor is a resolution
change and not a new world. Verified byte-identical at the reference tick, and pre-Exp-97 tiers
still reproduce exactly (`--tick 120 --pools 8 --n-jobs 16 --horizon 300 --slack-mult 1` →
45.3/52.4/52.5/76/73/16.6/15.6, tier `rule+pred`). `test_mechanism` 5/5.

**Cost, stated before anything is measured on it.** 4× the ticks means 4× the policy invocations per
seed; the deterministic arms go from ~11 s to ~23 s per 4 seeds, which is nothing. For the LLM arms
the bill is set by *distinct scenes*, not raw ticks — adjacent 30 s ticks repeat scene keys more
often, so the cache absorbs an unknown fraction of the 4×. That fraction is unmeasured. Combined
with the 8-GPU standard scale (96 jobs, 6× the pool-8 prompt), the per-job referee is the arm most
likely to be priced out; `composed` (one call per tick, job-count-independent) and the 0-token arms
are not.

## Experiment 98 — REAL LABELS: tier comes from the trace, and prod protection has no headroom left (2026-07-30)

**Type: instrumentation + calibration.** Motivated by Exp 96 (tier and deadline tightness were one
synthetic variable, r = −0.78) and by the observation that the floor was clearing work too easily
after Exp 97 loosened the deadlines.

**Method — Google's label approach, on the GPU trace.** Google cluster-data 2011 was evaluated as a
replacement and integrated (`data/build_google2011_replay.py`, trace `google2011`, 73,272 jobs over
105.8 h, real priority 0–11 and scheduling class 0–3). It is **not** adopted as the primary trace: it
is CPU/memory only, and the project is GPU scheduling. What it supplied was the *method* — take the
labels from the data, and take tier and tightness from **different** fields:

| | Google 2011 | v2020 with `--real-tiers` |
|---|---|---|
| tier | real: `priority ≥ 9` (7.8%) | real: instance carries a registered workload tag (**13.8%** of replay jobs) |
| tightness | real: `sched_class` | **synthetic, own rng stream** — no real field exists |
| corr(tier, tightness) | −0.113 | **−0.000** (was **−0.778**) |

v2020's own role field (`task_name`) was tested as the tightness signal and **rejected as degenerate**:
691,792 jobs in one class against 23,412 and 57. v2020 also records **no queueing delay** —
`pai_job_table.start_time` ≡ `pai_task_table.start_time`, p99 wait = 0 s — so "deadline = what the
real scheduler delivered" is available in Google's trace and not in this one. Both facts are limits
of the data, recorded rather than papered over.

Labels are a **side-car** (`data/build_v2020_labels.py` → `job_labels.csv`, keyed by job name):
`replay_jobs.csv` is not rebuilt, so every committed v2020 window stays byte-identical.

**Finding (pool 32 = 8 GPUs, 96 jobs, tick 30 s, slack ×10, n=4).**

| world | SLA | prodSLA | tight | util | wait |
|---|---|---|---|---|---|
| synthetic tier | 12.0% | 2.8% | 2.3% | 80% | 162.4 |
| **real tier** | 10.9%\* | **0.0%\*** | 11.7% | 79% | 161.3 |

**The floor already achieves 100% production attainment.** With a real tier the sim's structural
prod-precedence is sufficient on its own, and the tight-laxity stratum separates cleanly from the
prod stratum (11.7% vs 0.0%) — the separation `tight_sla` was added to detect.

**What this does to claim 2.** Prod-tier protection is the reasoning layer's one reproducible
contribution (Exp 70/71/73). Three results now converge on the same place: Exp 96 (the effect was
entangled with a synthetic confound and its strongest cell did not survive de-confounding), the
Exp 97 scale ladder (floor prod violations 12.9% at 2 GPUs → 2.1% at 8 → 0.0% at 30), and Exp 98
(0.0% at the floor with real labels). **On a realistic GPU cluster with trace-derived tiers there is
no prod-protection problem for the LLM to solve.** Claim 2 should be read as an artefact of the
synthetic tier recipe until shown otherwise on a workload where the floor actually misses prod
deadlines.

**Not yet done.** The LLM arms have not been re-measured in this world; the deterministic rows above
are n=4 and need n=32. Whether any arm can beat a 0.0% floor is not a question worth much compute —
the useful next question is which metric *does* have headroom here (overall SLA 10.9%, the tight
tercile 11.7%, and utilisation).

## Experiment 99 — STRATIFIED TIGHTNESS CLASSES + DYNAMIC URGENCY: the laxity lever pays hugely (2026-07-30)

**Two design fixes, both flagged, both no-ops when off** (invariance verified: the real-tier world
and the pre-Exp-97 world both reproduce byte-identically; `test_mechanism` 5/5).

**1. `--slack-classes` (tier suffix `+strat3`).** SLA tightness becomes a **stratified class** —
equal counts of tight/medium/loose (ρ = 1.20/1.60/2.05) per window, shuffled on its own rng stream —
instead of a continuous per-job draw. Mean slack is preserved (1.62 vs 1.59) so the operating point
does not move, but a window's tightness mix stops being a lottery, which is the same
binomial-variance problem Exp 96 flagged for prod counts.

**2. `--dyn-urgency` (tier suffix `+dynurg`).** The bid's urgency is recomputed **every tick** from
normalised deadline slack, `lax = (deadline − t − remaining) / (deadline − arrival)`, mapped onto the
same [0.6, 2.2] band the static draw used — so bid magnitudes stay comparable and only the ORDERING
changes. Committed as an arm, not a swap, because Exp 9–12 found committed bids beat per-round
value-max; this is the test of that boundary, not an assumption about it.

**Finding (pool 32 = 8 GPUs, 96 jobs, tick 30 s, real tiers, stratified classes, n=32 paired).**

| arm | SLA | prodSLA | tight | util | useful | slowdown | wait | done |
|---|---|---|---|---|---|---|---|---|
| frozen bid (floor) | 23.8% | 2.4% | 26.7% | 83% | 79% | 17.82 | 223.3 | 94.5/96 |
| **dynamic urgency (floor)** | **7.4%** | **0.8%** | **7.9%** | 81% | 77% | **5.35** | **121.2** | 95.0/96 |

```
frozen MINUS dynamic, paired by seed, n=32 (negative = frozen better):
  dSLA +16.4 ± 3.1*   dprodSLA +1.7 ± 1.9   dutil +1.7 ± 1.9   duseful +1.6 ± 1.8
  dregret -0.4 ± 0.8 (TOST EQ±2)   dslow +12.5 ± 2.9*
```

**Ordering grants by current deadline slack cuts violations from 23.8% to 7.4% (−16.4 ± 3.1\*) and
slowdown from 17.8 to 5.3 (−12.5 ± 2.9\*), for −1.7 utilisation and no regret change (equivalent at
±2).** This is exactly the **least-laxity grant ordering** that Exp 96's claims-table row 16 named as
the untried lever — `two_sided_sim.py` ordered grants by frozen bid value and nothing keyed on
laxity. It is a **0-token deterministic change** and it is the largest single SLA effect measured in
this project.

**What it does to the argument.** Every LLM arm is flat against the floor in BOTH worlds
(|dSLA| ≤ 0.1, all ns), so the reasoning layer adds nothing here either way. The headroom that Exp 98
left — overall SLA and the tight tercile — is now mostly consumed by a deterministic ordering rule:
the floor misses 7.4% instead of 23.8%. Combined with Exp 96/97/98, the pattern is consistent and
uncomfortable: **each time a piece of the world is made more realistic or a deterministic lever is
pulled, the space the LLM could occupy shrinks.**

**Caveats.** Exp 9–12's committed-bid result is not refuted: this is deadline-aware *priority*, not
per-round value re-auctioning, and churn/thrash metrics were not examined here. n=32, one pool, one
law, floor arm only for the headline contrast; the LLM arms were run but are inert. The stratified
classes and the dynamic bid were introduced together, so the +16.4 is the pair's joint effect
measured against the stratified frozen baseline — the classes alone move SLA by ~0.3 pts (n=4).

### Exp 99b — the ρ ladder: the dynamic-urgency win is invariant to the deadline calibration

ρ (`--slack-mult`) is the one number in the deadline recipe with no empirical referent — it was
chosen so the floor lands at a plausible violation rate. A single calibrated value makes any SLA
*level* a statement about that choice, so the contrast is reported across the regime instead.

| ρ | dSLA | dprodSLA | dslow | dutil |
|---|---|---|---|---|
| 2 | +3.8 ± 3.1\* | +7.2 ± 4.0\* | +9.7 ± 2.5\* | +1.3 ± 1.9 |
| 4 | +14.6 ± 3.4\* | +5.1 ± 3.0\* | +12.1 ± 2.8\* | +1.5 ± 2.0 |
| 10 | +16.4 ± 3.1\* | +1.7 ± 1.9 | +12.5 ± 2.9\* | +1.7 ± 1.9 |
| 16 | +14.4 ± 2.9\* | +0.5 ± 1.5 | +12.5 ± 2.9\* | +1.6 ± 1.9 |

(frozen MINUS dynamic, n=32 paired per cell; positive = dynamic better)

- **Dynamic urgency wins at every ρ**, so the result is an invariance claim, not an artefact of the
  operating point. The SLA *magnitude* traces the predicted hump — small at ρ=2 (everything misses,
  policies compress), peaking near ρ=10, easing at 16 — which is exactly why a single ρ must never
  be quoted as a level.
- **`dslow` is ~ρ-independent (+9.7 to +12.5\*)**: slowdown carries no deadline, so it is the same
  result stated without any invented quantity. Lead with it.
- **prod protection headroom shrinks as deadlines loosen** (+7.2\* at ρ=2 → +0.5 ns at ρ=16),
  consistent with Exp 98's 0.0% floor: the tighter the world, the more there is to protect.

### Exp 99c — whole-GPU quantum at the floor: Exp 48's penalty reproduces on the new world

Same physical cluster (8 GPUs), same 96 jobs, `--caps real`, real tiers, stratified classes, n=32
paired (window sampling is quantum-independent, so the tiers stay seed-paired).

| quantum | SLA | prodSLA | tight | util | slowdown | wait | done |
|---|---|---|---|---|---|---|---|
| quarter-GPU (32 q) | 19.9% | 5.2% | 22.9% | 85% | 14.20 | 171.8 | 95.7/96 |
| **whole-GPU (8 GPUs)** | **28.2%** | 4.5% | **31.8%** | 91% | **21.59** | **288.4** | **93.8/96** |

```
quarter MINUS whole (negative = quarter better), n=32:
  dSLA -8.3 ± 1.9*   dtight -9.0 ± 2.4*   dslow -7.4 ± 1.8*   dwait -116.5 ± 25.4*
  dutil -5.5 ± 1.5*  dfinished +2.0 ± 1.4*   dprodSLA +0.6 ± 1.0 (ns)
```

Forcing whole-card allocation costs **8.3 SLA points and 116 ticks of queueing** at the floor and
pushes 2 more jobs per window past the horizon — an independent replication of Exp 48 (~10 pts) on a
different operating point, different tick, and real tiers. Utilisation *rises* 5.5 pts, which is the
trap: a job pinned to a full card it does not need shows as occupancy. `u_useful` tracks `util`
exactly here because with `caps real` the rounded-up cap becomes the job's base, so the waste is
invisible to the useful-utilisation accounting — the whole-GPU rows must not be read as more
efficient.

### Exp 99d — the market on both quanta: it helps more where the quantum is coarse, and still cannot recover it

Same config as 99c (8 GPUs, 96 jobs, `--caps real`, real tiers, stratified classes, n=32), with the
0-token bid/ask market added to both.

| quantum | arm | SLA | prodSLA | tight | util | regret | slowdown | wait | done |
|---|---|---|---|---|---|---|---|---|---|
| quarter | no-llm | 19.9% | 5.2% | 22.9% | 85% | 2% | 14.20 | 171.8 | 95.7/96 |
| quarter | **market** | **19.7%\*** | 5.0% | 22.9% | 86% | **0%** | 13.97 | 169.5 | 95.7/96 |
| whole | no-llm | 28.2% | 4.5% | 31.8% | 91% | 3% | 21.59 | 288.4 | 93.8/96 |
| whole | **market** | **27.3%\*** | 5.3% | 30.7% | 93% | **0%** | 21.04 | 283.5 | 94.1/96 |

```
market vs floor, paired, n=32:
  quarter:  dSLA -0.2 ± 0.3    dutil +0.9 ± 0.5*  duseful +0.9 ± 0.5*  dregret -1.6 ± 0.5*  dslow -0.2 ± 0.2*
  whole:    dSLA -0.9 ± 0.4*   dutil +1.9 ± 0.9*  duseful +1.9 ± 0.9*  dregret -2.1 ± 0.6*  dslow -0.5 ± 0.3*
```

- **The market's SLA effect is significant only under the coarse quantum** (−0.9 ± 0.4\* whole vs
  −0.2 ± 0.3 ns quarter), and its efficiency gains roughly double (dutil +1.9\* vs +0.9\*, dregret
  −2.1\* vs −1.6\*). A coarser allocation unit leaves more for a clearing rule to fix. This is the
  first in-sim cell where the market moves SLA at all — every previous one was exactly 0.0 (Exp 97).
- **It does not come close to recovering the quantum penalty**, which is the result that matters:

```
floor@quarter MINUS market@whole, paired, n=32  (the recovery test)
  dSLA -7.4 ± 1.9*   dtight -7.8 ± 2.5*   dslow -6.8 ± 1.8*   dwait -111.7 ± 25.4*
```

  The market on whole cards is still **7.4 SLA points and 112 ticks of queueing worse than doing
  nothing on quarter cards**. It buys back 0.9 of the 8.3-point penalty. Exp 48 found the same for
  the negotiation-era mechanism ("the world costs ~10 SLA pts that negotiation can't recover"); it
  now holds for the market too, on a different operating point with real tiers.
- Every reasoning-shaped arm (`isolated`/`negotiated`/`single-llm`, rule mode) is *worse* than the
  floor on SLA in both quanta (+0.3 to +0.4, ns) and pays regret (+0.2 to +0.9\*).

**The standing lesson.** The allocation *unit* dominates every policy effect measured here: 8.3
points from the quantum against 0.9 from the best mechanism and 16.4 from the ordering rule (99).
Two of the three largest in-sim effects in this project are properties of the world, not of the
scheduler — which is the case for reporting them in that order.

---

## Experiment 100 — SYMMETRIC vs OPPOSED DEBATE: unresolved, and NOT equivalent (2026-08-04)

**Date:** 2026-08-04. **Spec:** `docs/superpowers/specs/2026-07-31-exp100-symmetric-vs-opposed-debate-design.md`,
pre-registered 2026-07-31 before any code, **amended §9 mid-run** (provenance disclosed there: 6 of
98 case lines had streamed, no aggregate/McNemar/TOST computed).

```
PINS_RESULTS=pins/results_exp100.json .venv/bin/python -m pins.exp88_budget_control \
    --suite r34 --model qwen2.5:14b --arms debate-pkt,debate-sym-pkt,critic-pkt,single-pkt
```

**Harness tripwire (spec §7) PASSES.** `debate-pkt` = **43/81**, byte-reproducing Exp 89. The
prompt-set refactor did not change behaviour, so the comparison is valid.

### H1 — the question: is the demand/supply asymmetry load-bearing?

```
debate-pkt      43/81   (opposed)      debate-sym-pkt  41/81   (symmetric)

b=4  c=2  b+c=6   net +2 cases
McNemar exact 2-sided p = 0.6875      one-sided (opposed>symmetric) p = 0.3438
TOST ±5  PRE-REGISTERED  : D=+2, 90% CI [-2.74, +5.25]  -> FAIL
TOST ±3  post-hoc (§9.2) : D=+2, 90% CI [-2.74, +5.25]  -> FAIL
```

**No decision branch of §6 fires.** Rule 1 ("symmetric ≡ opposed → the asymmetry is scaffolding")
required *equivalence*, and TOST fails at the pre-registered ±5 as well as at ±3. Rule 2 required a
significant difference, and McNemar is null. The result is **underpowered and unresolved**: only
**6 discordant pairs** in 81 cases, so the instrument cannot separate the hypotheses.

`CLAUDE.md`'s founding assumption — *"with symmetric objectives the discussion is theater"* — is
therefore **neither vindicated nor measured false.** It stays open. The pre-registered sceptical
prediction (symmetric ties opposed) is **not** confirmed; a tie in totals is not equivalence.

**This is exactly the failure mode §9.3 exists to prevent.** Reported as bare totals — *43 vs 41,
p=0.69* — this reads as equivalence and would have licensed rewriting §4.4 away from "opposed
advocates". The confidence interval says the data cannot support that.

### H2 — placement on the ladder (POOLED n=81, STRICT)

| comparison | b | c | b+c | 2-sided p |
|---|---|---|---|---|
| debate-pkt vs single-pkt | 18 | 3 | 21 | **0.0015** |
| debate-sym-pkt vs single-pkt | 16 | 3 | 19 | **0.0044** |
| debate-pkt vs critic-pkt | 14 | 6 | 20 | 0.1153 ns |
| debate-sym-pkt vs critic-pkt | 12 | 6 | 18 | 0.2379 ns |
| critic-pkt vs single-pkt | 10 | 3 | 13 | 0.0923 ns |

**The robust finding: both debate arms beat the single call decisively, whether or not the
reviewers are opposed** (43 and 41 vs 28). The clean confirmatory r4 stratum (n=50) agrees in
direction and significance — debate 30/50, sym 28/50, critic 21/50, single 18/50; debate vs single
p=0.0018, sym vs single p=0.0129. So *"a second pass with cross-talk beats one call"* is what this
run establishes; *"the reviewers must want different things"* is what it fails to resolve.

### H3 — controls (n=17): no specificity confound

debate 13/17, symmetric 11/17, critic 11/17, single 12/17. Neither debate arm fires more than the
other (b=2, c=0, p=0.50), so §6's rule 4 does not apply and H1 is not confounded by specificity.

### Harness notes

- **Arm drift, disclosed rather than absorbed.** `critic-pkt` scored 35/81 and `single-pkt` 28/81,
  against Exp 89/95's 34 and 27 — each **+1**. Only `debate-pkt` was the registered tripwire and it
  reproduced exactly, so nothing is voided, but the two non-tripwire arms are not byte-stable and
  any future comparison against 34/27 must say so.
- **`exp89_analyse` crashes on this file** with `KeyError: 'single-pkt-boN'` — that arm is Exp 89's
  budget-matched comparator and was not in this run's `--arms`. The numbers above come from the
  same helpers (`exp88_analyse.discordant/counts/mcnemar_exact_two_sided/tost`,
  `exp89_analyse.mcnemar_one_sided`) driven directly over `blob["results"]`. Either add
  `single-pkt-boN` to the arm list on a re-run, or give the analyser a skip-missing-arm guard.

### What would resolve it

Power, not more arms. At b+c=6 the discordant rate is ~7%; separating opposed from symmetric at
±3 cases needs several times the case count. Per the external review this remains bounded even if
resolved: it would speak to *this authored suite and this architecture*, and would still not
separate second-pass from role diversity, cross-talk from independent reconsideration, or repeated
calls to one model from genuinely distinct agents holding private information.
