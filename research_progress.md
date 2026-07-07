# Research Progress — PINS experiment log

> **Pruned 2026-07-06:** stepping-stone experiments are compressed to their lessons and
> superseded findings are marked in place; the full original log is in git history
> (`git log -- research_progress.md`, through commit `4a74508`).

## State of the claims (2026-07-06)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | LLMs cannot calibrate absolute resource numbers; LLM-derives-structure + code-computes predicts peak GPU memory to ~2% (~40x better than the params heuristic), robust across precision, architectures, long context | Exp 1-7 | solid |
| 2 | A small attention forecaster beats persistence on the dynamic channels; a quantile head + conformal calibration adds a usable per-job uncertainty signal at no accuracy cost | Exp 8, 16A | solid |
| 3 | Per-round value-max auctions lose SLA to stable serialisation; the bid-once committed-auction matches greedy on raw SLA and ~halves prod-tier SLA; an LLM can set & justify the priority (interpretable, matches deterministic) | Exp 9-12 | solid |
| 4 | The committed mechanism is gameable via inflated self-reports; a flat budget does NOT fix it — incentive-compatible clearing (payments) is unbuilt | Exp 13 | **open problem** |
| 5 | The supply agent's headroom-reservation lever pays only against rigid incumbents at moderate contention; malleability-aware reservation recovers the utilisation cost | Exp 14-15 | solid (regime-gated) |
| 6 | Uncertainty-sized margins are insurance whose value grows with tail severity; blanket margins backfire; the LLM hedge (7b+) beats the deterministic margin once given a spike-risk signal | Exp 16-17 | solid |
| 7 | The ILP ties the auction on a 1-D pool (~150x cost, not worth it) and earns its keep exactly where count-only clearing structurally fails: node placement (ploss to 0, util +7-12 pts) and making aggressive LLM margins SAFE | Exp 18, 25 | solid |
| 8 | The LLM does not earn its cost as a numeric predictor (runtime: retrieval wins; DAG demand: GBT/one-line rules win); its measured value is judgement + justification in the agent layer | Exp 19-21 | solid (negative for the LLM) |
| 9 | The two-sided split beats the single-LLM-both-objectives baseline: the concession ladder is the brake a lone agent lacks (single-llm over-commits at every scale) | Exp 22, 24, 27-29 | solid |
| 10 | On real trace replay, negotiation buys significant prod-SLA protection (-4..-8 pts, 95% CI) at no measurable overall-SLA or slowdown cost; "beats the floor on overall SLA" was 8-seed noise | Exp 28 -> **Exp 29** | solid (n=32) |
| 11 | The bounded protocol substitutes for scale as SUFFICIENCY: negotiated rule ~ 3b ~ 7b ~ 14b, all statistically indistinguishable (the earlier "3b beats 14b" was noise); the un-braked single-llm still needs scale and still loses at every size tried | Exp 27-28 -> **Exp 29 (+7b addendum)** | solid (n=32) |
| 12 | With Stage-1 predicted demand in the loop, prediction error costs every policy, yet prod-SLA protection survives it at every pool (rule AND 3b tiers); the slack-regime cushion vs the floor is significant at the rule tier, direction-consistent at 3b | Exp 30 | solid (rule + 3b) |
| 13 | Naive per-state LLM reflection does not converge (period-2 limit cycle) — the deterministic ladder, not reflection, is what makes a small model safe | Exp 26 | negative |

**Environment:** single A100-PCIE-40GB · PyTorch 2.6.0+cu124 (`.venv-forecast` for torch) ·
qwen2.5 3b/7b/14b via Ollama `localhost:11434` · simulators pure-Python in `.venv`.

---

# Stage-1 STATIC — peak-GPU-memory prediction (closed-loop on real CNNs)

**Date:** 2026-06-06..12 · `pins/eval/predict_cnn.py`, `predict_arch.py` · all ground truth
**measured** on the A100 (`torch.cuda.max_memory_allocated()`), never estimated.

## Experiments 1-3 — the eliminations: every LLM-emitted NUMBER is the weak link (compressed)

Three designs, each removing one more number from the LLM, each still failing the
beats-the-params-heuristic gate — but each isolating WHERE the failure lives:

- **Exp 1, raw LLM number** (`{peak_mem_gb}` from metadata): every model size over-predicts
  7x-170x (3b mem MAE 283 GB, 7b 77, 14b 24 — vs heuristic 1.7 GB); scale helps monotonically but
  never closes the order-of-magnitude gap, and 7b even inverts the ranking (rho = -1). LLMs cannot
  calibrate an absolute memory magnitude; **bigger is not the fix**.
- **Exp 2, extract-then-compute** (LLM emits structured facts; a deterministic formula computes):
  error drops ~70x (MAE 3.9 GB) and becomes model-size-robust — the facts are correct — but the one
  remaining guessed number (`activation_mb_per_sample`, ~18-80x too low) still loses to the
  heuristic. Plugging the TRUE activation into the same formula matches measurement: the formula
  is exact, the guessed number is the leak.
- **Exp 3, free-text layer-walk reasoning (14b)**: the per-layer SHAPE derivation is 100% correct
  for both probe nets; every error is bookkeeping (inconsistent inclusion rule, waffling optimizer
  multiplier; flips the ranking, rho = -1). The LLM knows the *method* but cannot apply the
  summation consistently — so move the arithmetic into code.


## Experiment 4 — Deterministic: LLM-shapes → code-sum (SUCCESS)

**Method.** The culmination. The LLM's only job is to emit per-layer **shapes** (verified
reliable in Exp 3). Deterministic code does everything else:
- `param_term = 4 · P · bytes` (weights + grad + Adam moments — exact),
- `activation_raw = batch · bytes · Σ(conv + pool feature-map elements)` via
  `feature_map_elements()` (replays the architecture),
- `peak ≈ param_term + a · activation_raw + b`, where `a` = activation-retention factor
  (BN/ReLU buffers, autograd saves) and `b` = fixed cudnn/workspace overhead.

Evaluated honestly: **6 varied CNNs** (width × depth × resolution × batch) with `(a, b)`
**leave-one-out calibrated** — each prediction uses constants fit on the *other 5* configs.
(`--deterministic`)

**Result (leave-one-out, fp32).**

| Config | Params | **Measured** | Deterministic | Heuristic | Mean |
|---|---|---|---|---|---|
| w32-b3-64px-bs128 | 0.29M | 0.61 GB | 0.59 GB | 4.00 GB | 3.0 GB |
| w64-b3-64px-bs256 | 1.15M | 2.40 GB | 2.34 GB | 4.00 GB | 3.0 GB |
| w64-b4-96px-bs128 | 4.69M | 2.87 GB | 2.89 GB | 4.00 GB | 3.0 GB |
| w128-b4-96px-bs128 | 18.75M | 5.92 GB | 5.88 GB | 4.10 GB | 3.0 GB |
| w96-b3-128px-bs64 | 2.58M | 3.54 GB | 3.54 GB | 4.00 GB | 3.0 GB |
| w64-b5-128px-bs64 | 18.86M | 2.82 GB | 2.90 GB | 4.10 GB | 3.0 GB |

| Predictor | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|
| **Deterministic (LOOCV)** | **0.04 GB** | **1.8%** | **100%** | **0.94** |
| Heuristic | 1.6 GB | 124% | 67% | 0.66 |
| Mean | 1.1 GB | 83% | 67% | 0.60 |

**Beats-heuristic gate: PASS (0.04 vs 1.6 GB MAE — ~40× better).**

**Why it SUCCEEDED.** It assigns each subtask to the component that's good at it: the LLM
supplies architecture/shapes (which even 14b does perfectly), and deterministic code does the
summation + arithmetic (which the LLM does unreliably). The activation-retention ambiguity
that broke Exp 3 is absorbed into a single calibrated factor `a`, and fixed framework
overhead into `b`.

**Money shot — why activation-awareness matters.** Two nets with **~identical parameter
counts** but **2× different real memory**:
- w128-b4 → 18.75M params → **5.92 GB**
- w64-b5 → 18.86M params → **2.82 GB**

The params heuristic gives both ~4.1 GB (blind). The deterministic model nails **5.88 vs
2.90 GB**, because the deep net with small spatial dims holds far fewer activations despite
more parameters.

---

## Experiments 5-7 — robustness: precision, architectures, long context (compressed)

- **Exp 5 — fp16/bf16: SURVIVES, and the margin GROWS.** Ground truth re-measured and (a,b)
  re-calibrated per precision: deterministic MAE 0.06 GB, 100% within-1.5x, rho 0.94. Mixed
  precision roughly halves real memory, which the precision-blind params heuristic cannot see
  (its within-1.5x collapses 50% -> 17%). Caveat: for param-heavy models split the byte widths
  (params 4 B, activations 2 B).
- **Exp 6 — ResNet + tiny Transformer via architecture-agnostic forward hooks: generalises, with
  one precise boundary.** One global (a,b) fits CNN (MAE 0.13 GB) and ResNet (0.37 GB) with no
  special-casing; transformers break at long context because module-output hooks miss the internal
  seq^2 attention score matrix (seq 1024: predicted 1.99 vs measured 3.52 GB — and the closed-form
  matrix size matches the ~1.5 GB gap almost exactly).
- **Exp 7 (2026-06-12) — add the analytic attention term** `layers*nhead*seq^2` **in deterministic
  code** (not the LLM): the long-context gap closes and one global (a,b) fits all three families.

| Predictor (global LOOCV, 10 jobs, 3 families, fp32) | mem MAE | MAPE | within 1.5x | rho |
|---|---|---|---|---|
| **Deterministic (Exp 7)** | **0.28 GB** | **17.6%** | **100%** | **0.96** |
| Deterministic (Exp 6, no attention term) | 0.49 GB | 38% | 60% | 0.89 |
| Params heuristic | 1.9 GB | 223% | 40% | **-0.30** |

Per-family MAE: cnn 0.16 · resnet 0.34 · transformer 0.30 GB (was 0.71). The heuristic's rho is
*negative* across architectures — params anti-correlate with memory here; activation-awareness is
not a luxury. Residual: fp32 only — gate the attention term on the attention backend before
trusting it under flash kernels. **The fix for an LLM blind spot was more deterministic code, not
more LLM.**


## The arc in one picture

Each step that moved a **number** out of the LLM and into code cut error ~an order of magnitude:

```
Exp 1  raw LLM number          MAE 283 → 24 GB    useless (LLM can't calibrate magnitude)
Exp 2  hybrid, guessed facts   MAE 3.9 GB         facts right, activation guessed too low
Exp 3  hybrid, 14b reasoning   MAE 2.6 GB         shapes perfect, counting inconsistent (ρ −1)
Exp 4  deterministic shapes    MAE 0.04 GB        LLM→shapes, code→sum  ✅ PASS
```

**Headline conclusion.** This is a clean empirical proof of PINS's governing principle —
**the LLM reasons (derives architecture/shapes); deterministic code decides (the arithmetic).**
The answer to "would a bigger/better LLM help?" is, with data: **no** — even 14b already
produced perfect shapes; *removing the LLM from the arithmetic* was the fix, not scaling it.

---

**Status.** All Stage-1-static follow-ups closed by Exp 5-7 (precision, architectures, attention
term). The remaining integration item — feeding this estimator into the live predictor
(`pins/predictor.py`) — was overtaken by the GPU-track wiring (Exp 27+), which routes *predicted
requested GPU* into the negotiation instead. Artifacts: `pins/eval/results_cnn*.json`,
`results_arch.json`; reproduce via `pins.eval.predict_cnn --deterministic` / `pins.eval.predict_arch`.

---


# Stage-1 DYNAMIC — trajectory forecasting on MIT Supercloud

## Experiment 8 — Residual attention forecaster vs the persistence gate (PASS)

**Date:** 2026-06-18 · 100 joint CPU+GPU Supercloud jobs on a common 10 s grid; lookback 30 ->
forecast 30 steps x 4 channels; a small Transformer encoder predicts the **residual from
persistence** (flat channels degrade gracefully to persistence).
`pins/forecast/{dataset,baselines,model}.py`, runs in `.venv-forecast`.

| Forecaster | nMAE_mean | gpu_util MAE | cpu_util MAE |
|---|---|---|---|
| persistence | 0.068 | 6.26 | 42.5 |
| moving_avg(k=6) | 0.063 | 5.71 | 38.9 |
| **attn (ours)** | **0.058** | **4.96** | **35.0** |

**Gate PASS.** The win concentrates in the dynamic channels (-21%/-18% vs persistence) — exactly
where a forecaster earns its keep; the residual anchor protects the flat memory channels (it is
marginally worse there). Single 70/30 split, one seed. Extended with the quantile head +
conformal calibration in Exp 16A.


# Stage-2 NEGOTIATION — which allocation mechanism rations GPUs best?

The headline negotiation experiment (thesis refocus 2026-06-17): on one shared job stream,
does the PINS sealed-bid auction beat value-blind scheduler baselines on **SLA-violation rate**
at high utilisation? Harness: `pins/negotiation_sim.py` (pure Python, runs in `.venv`, no
LLM/MCP). It reuses the real decider `pins/mechanism.py:clear` and the predictor's
`marginal_values`/`PHASE_PROFILES`; baselines are wrapped with the same signature.

## Experiment 9 — Value-max auction vs greedy/equal/static (NEGATIVE; the diagnosis that shaped Stage-2)

**Date:** 2026-06-18 · `pins/negotiation_sim.py` (16 jobs, urgency scales both bids and deadline
tightness; 8 seeds; welfare always scored on static base bids). En route, a real `mechanism.py`
fix: the anti-thrashing gate wrongly charged rescale cost for filling IDLE GPUs (a cold start sat
at 0% utilisation); now only preemptions pay. All `test_mechanism.py` tests stay green.

**Result (SLA-violation %, 8-seed mean):** greedy-FIFO beats every auction at every contended
pool — pool 8: greedy **46.9** vs PINS-auction 74.2, equal-share 76.6, static-sticky 100. The
auction does win **welfare** (its own objective); deadline-scaled bidding made SLA *worse*
(preemption churn).

**The lesson that drives Exp 11:** value-max with diminishing-returns curves SPREADS GPUs thin —
everyone runs slow, everyone finishes late; SLA rewards CONCENTRATION plus a STABLE order, which
greedy provides by accident. The mechanism *objective* and *run-stability*, not value-awareness,
are the SLA levers.

## Experiment 10 — LLM agents bid strategically: hinge-safe and interpretable, but SLA unchanged (compressed)

**Date:** 2026-06-18 · `pins/llm_agent.py`: the LLM (qwen2.5:3b) picks a categorical stance +
focus-GPU count per discretised state (32 cached states, out of the hot loop, rule fallback);
deterministic code maps it onto the calibrated bid curve. It bids coherently, wins welfare/goodput
(pool-8 14550 vs 12637) and ships an auditable one-line justification per decision — but **no bid
design (static, deadline-scaled, or LLM-strategic) beats greedy on SLA** (pool 8: 78.1 vs 46.9),
because greedy's edge is allocator *stability*, not bid quality. Corroborates Exp 9 and sets up
the stability mechanism of Exp 11.


## Experiment 11 — The stability lever: committed-auction beats greedy on prod-tier SLA

**Date:** 2026-06-18

**Question.** Exp 9-10 showed every per-round marginal auction (static / deadline / LLM) loses SLA
to greedy-FIFO. Greedy's edge is *stability* (run-to-completion), not intelligence. Can a stability
mechanism close the gap? Three probes (all in `pins/negotiation_sim.py`, 8-seed means):

**11a — Incumbency bonus (`make_stable_auction(beta)`).** GPUs a job holds get `+beta` in the
clearing sort, so a challenger must outbid `incumbent + beta` to preempt. **Result: barely moves.**
Even beta=40, pool-8 SLA 73% vs greedy 47%. *Thrashing was not the problem.*

**11b — Value-block serialisation.** Serialise like greedy (full GPU block per job, run to
completion) but order by bid value instead of arrival. **Result: WORSE than greedy** (pool-8 SLA
79% vs 47%). Diagnosis: greedy's order is by job-id = **constant across rounds**, so the same job
stays at the front and finishes; ordering by a *changing* value flips the front every round and
re-thrashes. *The lever is serialise + STABLE order.*

**11c — Committed-auction (`make_committed_auction`) — THE WINNER.** Bid-once: each job's priority
is frozen by its first bid (urgency-scaled) on arrival; the orchestrator then serialises — highest
priority first, full block, run to completion. Stable like FCFS, value-aware like an auction (urgent
/ prod jobs, urgency 1.667-2.2, sort strictly above best-effort, so prod is served first).

| pool | metric | PINS-auct-DL | **greedy-FIFO** | **committed-auction** |
|---|---|---|---|---|
| 4 | SLA / prodSLA | 98 / 100 | 88 / 100 | 98 / 100 |
| 6 | SLA / prodSLA | 87 / 83 | 70 / 70 | **72 / 48** |
| 8 | SLA / prodSLA | 74 / 75 | 47 / 54 | **53 / 23** |
| 12 | SLA / prodSLA | 37 / 33 | 22 / 26 | **22 / 15** |

**Result: committed-auction roughly MATCHES greedy on raw SLA and roughly HALVES prod-tier SLA**
(pool 8: 23% vs 54%; pool 12: 15% vs 26%). It deliberately spends best-effort deadlines to protect
production deadlines — exactly the value-weighted behaviour the thesis wants, and the per-round
auctions never delivered. (Single-seed sweep is even stronger: committed wins BOTH metrics at pools
6 & 8.)

**Why it works / the through-line of Stage-2.** Deadline-meeting needs (a) **concentration** — full
capacity to one job so it finishes, not GPUs spread thin so all run slow — and (b) a **stable
order**. The per-round marginal auction violates both: diminishing bids spread GPUs across jobs, and
re-pricing every round flips the order. **The fix was to stop re-auctioning: bid once, commit,
serialise.** The "negotiation" collapses to a one-shot priority declaration — which is also the
natural seam for the LLM: set & justify that priority (the interpretable, AI-agent version) is the
next step. Honest caveat: the win lives in the value-weighted (prod-tier) metric; on the flat count
committed only matches greedy, because protecting prod *costs* best-effort deadlines by design.

**Reproduce.**
```bash
cd MCP
(cd pins && ../.venv/bin/python test_mechanism.py)   # decider untouched, still green
.venv/bin/python -m pins.negotiation_sim             # committed-auction now in the default sweep
```

## Experiment 12 — LLM sets & justifies the committed priority (interpretable winner)

**Date:** 2026-06-18

**Method.** Put the LLM back in, on the Exp-11 winner. The committed-auction serialises by a frozen
per-job priority; here an LLM (qwen2.5:3b) SETS that priority once, on arrival, from the job's
intrinsic profile `(tier, deadline tightness, size)` — as an **ordinal class** `critical|high|
normal|low`, never a number. Code maps class→weight and does all serialisation; the LLM touches only
the ORDER (+ a one-sentence justification). New `pins/llm_agent.llm_priority` (cached per profile,
≤8 states); `make_llm_committed` in `negotiation_sim.py` returns the (bid_builder, allocator) pair
sharing the frozen map. Hinge-safe and out of the hot loop, same as Exp 10.

**Result (8-seed mean; SLA / prodSLA, lower = better).**

| pool | greedy-FIFO | committed (deterministic) | **llm-committed** |
|---|---|---|---|
| 6 | 69.5 / 70.2 | 71.9 / 47.5 | 71.9 / **51.7** |
| 8 | 46.9 / 53.8 | 53.1 / 23.0 | **49.2** / **25.1** |
| 12 | 21.9 / 26.3 | 21.9 / 14.5 | 21.9 / **9.9** |

Sample LLM priorities (auditable): *prod+tight+large → **critical*** ("Large, high-impact prod job
with a tight deadline is most at risk"); *besteffort+loose+small → **normal*** ("Best-effort jobs
with loose deadlines are assigned normal priority"). Artifact: `pins/results_llm_negotiation.json`
(`committed_priorities`).

**Honest read.**
1. **The LLM-priority committed auction keeps the win** — ≈2× lower prod-tier SLA than greedy-FIFO
   at every contended pool (pool 8: 25% vs 54%), matching raw SLA. The interpretable, AI-agent
   version of the Exp-11 mechanism delivers the same headline result.
2. **The LLM MATCHES, not beats, the deterministic priority.** Within seed noise it is slightly
   worse at pool 6/8 (its nuance — elevating *tight-deadline best-effort* to `high` — dilutes
   strict tier-first ordering) and better at pool 12. So the LLM's value here is **not** a better
   number; it is (a) **interpretability** — every serialisation order ships with a justification,
   the edge vs a black-box RL/greedy — and (b) reasoning over **richer/unstructured context** that
   a hand rule would have to be engineered for case by case.
3. **Hinge held throughout.** The LLM emitted only an ordinal class; code owned every magnitude and
   the serialisation. Consistent with Stage-1's lesson and kept out of the hot loop (cached).

**Stage-2 arc:** per-round auction loses SLA (Exp 9, spreads+thrashes) → LLM per-round bidding is
interpretable but still loses SLA (Exp 10) → **committed-auction (bid-once, serialise) beats greedy
~2× on prod-tier SLA** (Exp 11) → **LLM sets & justifies that priority, preserving the win with
auditable decisions** (Exp 12). Defensible thesis position: *match the schedulers on outcomes,
uniquely explain every decision.*

**Reproduce.**
```bash
cd MCP
.venv/bin/python -m pins.llm_agent                                  # priorities + justifications
.venv/bin/python -m pins.negotiation_sim --llm --model qwen2.5:3b   # adds llm-committed row
```

## Experiment 13 — Incentives: the committed-auction is gameable; a flat budget does NOT fix it

**Date:** 2026-06-18

**Question.** Exp 11-12 assumed agents report their priority HONESTLY (priority = true urgency/tier,
or the LLM reading the true profile). But the committed-auction trusts that self-report. What if jobs
LIE? New in `negotiation_sim.py`: `declare_*` fns (truthful / inflate) + `make_declared_committed`
(serialise by DECLARED class; metric still scores TRUE prod jobs).

**Part A — the vulnerability (no incentive layer). 8-seed, SLA / prodSLA.**

| pool | greedy (ref) | committed TRUTHFUL | committed BE-lie | committed ALL-lie |
|---|---|---|---|---|
| 6 | 69.5 / 70.2 | 74.2 / **51.7** | 69.5 / 70.2 | 69.5 / 70.2 |
| 8 | 46.9 / 53.8 | 50.8 / **25.1** | 46.9 / **53.8** | 46.9 / 53.8 |
| 12 | 21.9 / 26.3 | 22.7 / **9.9** | 21.9 / 26.3 | 21.9 / 26.3 |

**When best-effort jobs lie ('critical'), prod-tier SLA collapses from 25% back to 54% — EXACTLY
greedy.** With everyone at the top class, priorities tie and the mechanism degenerates to
serve-by-arrival. The entire Exp-11/12 advantage rested on **trusted self-reports.** Mechanism is
not manipulation-resistant.

**Part B — a flat per-claim budget does NOT fix it (NEGATIVE, compressed).** Charging a per-tick
class cost from an equal budget does not restore the honest optimum (pool-8 prodSLA stays 52-64%
vs the truthful 25%) — and the control kills the idea: TRUTHFUL play with a small budget craters
to 88%, because honest prod jobs declare the same class and pay the same cost. A flat cost on the
class tracks job residence, not truthfulness: a liar and an honest job declaring the same class
are indistinguishable by ANY cost on the class itself. Separating them requires eliciting true
value with PAYMENTS (uniform-price / VCG) — which exposes the **core Stage-2 tension**: the
SLA-winning mechanism (committed priority classes, no payments) and the incentive-compatible one
(payments) are different designs. Unifying them (e.g. exogenous per-user budgets a la fair-share,
which needs multi-job agents) is the identified open problem.

**Reproduce.** `declare_*` fns + `make_declared_committed(declare_fn, budget=...)` in
`pins/negotiation_sim.py`; 8 seeds, pools {6,8,12}.


## Experiment 14 — the SUPPLY agent (headroom reservation): regime-gated win; model size matters (compressed)

**Date:** 2026-06-18/19 · `pins/supply_sim.py`; lever = reserve R GPUs from best-effort so a
late-arriving prod job lands on idle capacity. **Malleable regime: reservation is redundant** —
the committed auction already serialises prod first and preemption is free, so prodSLA is
untouched by any R while SLA/util only worsen. **Rigid-incumbent regime (running jobs
non-preemptable): the WIN** — at moderate contention (pool 12) reserve-adaptive is a Pareto move
(prodSLA 26.8 -> **18.8** at equal SLA, -2 pts util); a trade at pool 8; harmful when tight.
Adaptive (release when no prod incoming) beats static.

**LLM supply agent (categorical reserve level, 9 cached states) — the first decision where model
size has stakes:**

| state | 3b | 7b | 14b | oracle |
|---|---|---|---|---|
| scarce/few | **light X** | none ok | none ok | none |
| scarce/many | **heavy X** | **heavy X** | none ok | none |
| moderate/none | **light X** | none ok | none ok | none |

**3b systematically over-reserves** — its own justification misreads "scarce" as low contention —
and it hurts SLA at tight pools (pool 4: 100 vs 14b's 96.1); **14b matches the hand-built
oracle**. The justification trace makes the failure visible and auditable (the edge vs RL).

## Experiment 15 — MIXED malleable+rigid: malleability-AWARE reservation recovers the util cost (compressed)

**Date:** 2026-06-19 · `simulate_mixed` (phi = malleable fraction) reproduces `simulate_rigid` at
phi=0 and the Exp-14 redundancy at phi=1 — endpoint gate PASS, so it faithfully contains both
regimes. An AWARE agent reserves idle headroom only against the rigid fraction and reclaims
malleable incumbents on demand: **aware == blind on prodSLA at every phi (keeps the full QoS win)
while recovering the idle-utilisation cost, and the recovery grows with phi** (pool-8 util gap
over blind: 0/4/5/7/8 pts across phi=0..1) — aware Pareto-dominates blind for all phi>0. The
supply agent's prodSLA edge itself is phi-graded — largest when mostly rigid (66 vs 73 at phi=0,
pool 8), vanishing at phi=1. **The supply agent's value scales with the rigid fraction.** Reclaim
is modelled free; a non-zero reclaim penalty is the open stressor.


## Experiment 16 — Uncertainty as a first-class signal: quantile forecasting → uncertainty-sized safety margin

**Date:** 2026-06-19

**Why.** Exp 8 forecasts a single trajectory (a POINT estimate); the `research_plan.md` prediction
co-contribution needs *uncertainty* — "forecast demand WITH an explicit uncertainty estimate, which
sizes the safety margin the demand agent bids for." This experiment builds that end-to-end: a
quantile forecaster (Part A) and the first wiring of its uncertainty into the Stage-2 demand agent
(Part B), running the plan's required **no-uncertainty ablation**.

### Part A — Quantile-regression forecaster (`pins/forecast/model_quantile.py`)

**Method.** Same small Transformer encoder as Exp 8, but the head emits P10/P50/P90 per channel
trained with **pinball (quantile) loss**. P50 stays a residual-from-persistence (the Exp-8 anchor);
the interval edges are **softplus half-widths** around P50, so `P10 ≤ P50 ≤ P90` by construction
(no quantile crossing). 60 epochs on the A100. The train set is split **fit / cal** (≈50/20 jobs) so
the raw intervals can be **conformalised** (split-conformal / CQR, Romano et al. 2019): the
calibration set's conformity scores `max(P10−y, y−P90)` give a per-channel finite-sample width
adjustment so test coverage → the 0.80 nominal (distribution-free). Test = the held-out 30 jobs.
Runs in `.venv-forecast`.

**Result.**

| metric | persistence | quantile P50 |
|---|---|---|
| nMAE_mean (accuracy gate) | 0.072 | **0.066** |
| gpu_util MAE | 6.26 | **5.40** |
| cpu_util MAE | 42.5 | **35.8** |

| coverage (target 0.80) | gpu_util | gpu_mem_gb | cpu_util | mem_gb | aggregate |
|---|---|---|---|---|---|
| raw | 0.78 | 0.52 | 0.64 | 0.76 | 0.67 |
| **conformalised** | 0.76 | **0.77** | **0.75** | 0.73 | **0.75** |

**Beats-baseline gate (P50): PASS** (0.066 vs 0.072) — **adding uncertainty did not cost accuracy**;
the P50 win is still concentrated in the dynamic channels (gpu_util, cpu_util), exactly as Exp 8.
**Conformal calibration lifts aggregate coverage 0.67 → 0.75** (toward nominal 0.80), fixing the two
under-covered channels (gpu_mem 0.52→0.77, cpu_util 0.64→0.75) by widening only where needed — the
conformal add is negative for the *over*-covered gpu_util (it shrinks it). Still a touch under 0.80
(small calibration set → noisy per-channel quantile). Calibrated per-job uncertainty (norm width over
the GPU-demand channels) spans **min 0.01 / median 0.16 / max 0.90** — a real, well-spread signal,
written to `pins/forecast/results_quantile.json` as the Stage-2 bridge.

### Part B — Uncertainty sizes the demand agent's safety margin (`pins/uncertainty_sim.py`)

**Method.** Connect Stage-1 → Stage-2 for the first time (both were stubbed apart: `predictor.py`
was a phase-curve stub). `marginal_values(phase, urgency, uncertainty)` now appends
`round(uncertainty·scale)` **safety-margin GPUs** to the bid curve (backward compatible:
`uncertainty=0` reproduces the old curve; `test_mechanism.py` still 5/5). The Stage-2 mechanism that
makes a margin *matter*: a job's TRUE train work can **spike** above the forecast by an amount
bounded by its uncertainty (the tail the point forecast was blind to); to finish before its deadline
the job must run faster, i.e. use margin GPUs (`rate = min(alloc, C0+margin)/C0`). Three bid policies
on the SAME stochastic workload, cleared by the committed auction (Exp-11/12 winner), 16 seeds, real
uncertainty distribution from Part A:
- **no-margin** — bid C0 only (the point-forecast demand agent, Exp-8 era);
- **fixed-margin** — bid C0+1 for EVERY job (a blanket headroom);
- **uncertainty-sized** — bid C0 + round(u·scale), margin where the spike risk actually is.

**Result (16-seed mean, pool 12 = moderate contention; calibrated uncertainty). The value of a
margin depends on how heavy the demand TAIL is, so we sweep `spike_max`. Lower = better.**

| spike_max | metric | no-margin | fixed-margin | uncertainty-sized |
|---|---|---|---|---|
| 0.6 | SLA / prodSLA | 23.8 / **11.9** | 27.7 / 13.2 | 23.8 / **11.9** |
| 1.0 | SLA / prodSLA | **26.2** / **13.0** | 30.5 / 14.2 | 26.6 / **13.0** |
| 1.5 | SLA / prodSLA | 34.0 / 23.7 | 34.4 / 18.3 | **32.8 / 17.4** |
| 2.0 | SLA / prodSLA | 40.2 / 28.1 | 37.9 / 20.6 | **35.9 / 18.4** |

(At pools 6/8 the pool is saturated — 96–98% util — so no margin can be granted and all three
policies coincide; the lever needs spare capacity, like every lever in this project.)

**Findings.**
1. **Uncertainty-sizing is insurance: its value GROWS with tail severity.** When spikes are mild
   (0.6–1.0) demand is well-behaved, a margin barely matters, and uncertainty-sized ties the
   point-forecast (no-margin). As the tail grows heavy (1.5–2.0) — exactly the regime a forecaster's
   *uncertainty* is meant to flag — uncertainty-sized pulls clearly ahead on BOTH metrics, and the
   gap widens: at spike 2.0 it cuts **prod-SLA 28.1 → 18.4** (~35% relative) and SLA 40.2 → 35.9 vs
   the point forecast. This is the plan's "uncertainty sizes the margin; the auction rations it",
   demonstrated.
2. **A fixed/blanket margin is the wrong answer — sizing by the per-job quantile width is the point.**
   fixed-margin is consistently mediocre: it *over-subscribes* (util ~85 vs ~82) when spikes are mild
   — actually worse than no margin (e.g. 27.7 vs 23.8 at spike 0.6) — and *under-protects* the
   high-uncertainty jobs when spikes are heavy (prod-SLA 20.6 vs 18.4 at spike 2.0). The value is the
   *signal*, not the headroom.
3. **Calibration mattered.** With the raw (over-confident) intervals the per-job uncertainties were
   small (median 0.09) and the effect was marginal; the conformalised signal (median 0.16) gives the
   margin enough resolution to target the heavy-tail jobs — the Part-A and Part-B improvements are
   linked.

**Caveats.** The stochastic-demand mechanic (spike bounded by a job's uncertainty) is a modelling
choice operationalising "the point forecast was blind to the tail"; the synthetic workload is not the
real Supercloud trace (the bridge passes only the uncertainty *distribution*, not per-trace demand).
Next: feed uncertainty to the **LLM demand agent's justification** (it already sets priority — Exp 12
— uncertainty is a natural extra input).

**Reproduce.**
```bash
cd Research
.venv-forecast/bin/python -m pins.forecast.model_quantile   # Part A: train + coverage + artifact
.venv/bin/python -m pins.uncertainty_sim --seeds 16          # Part B: no/fixed/sized margin ablation
```
`pins/forecast/model_quantile.py` (pinball, softplus widths, `results_quantile.json`);
`pins/predictor.marginal_values(uncertainty=…)`; `pins/uncertainty_sim.py`
(`simulate_stochastic`, `--spike`, `--scale`, `--fixed-u`).

## Experiment 17 — the LLM demand agent decides the hedge from uncertainty; "fix the decision, not the model" (compressed)

**Date:** 2026-06-19 · `llm_agent.llm_margin`: from `(uncertainty, deadline, contention, tier)`
the LLM emits a categorical hedge {none, some, heavy} + a justification; `predictor.
marginal_values` owns every GPU count (<=36 cached states, rule fallback — hinge intact). Mild
tails (spike 0.6, pool 12): **3b over-hedges and HURTS** (SLA 26.2 vs deterministic 23.8) — its
justification misreads "high contention" as "spare capacity", the same comprehension failure as
Exp 14C; 7b matches the deterministic policy (23.8), 14b edges it (23.4).

**Heavy tails exposed a mis-specified decision — and a SIGNAL, not a bigger model, fixed it.** At
spike 1.5 the contention-gate (correct at mild spikes) suppresses margin exactly where spiking
jobs need it: 14b fell to 33.6/22.8 vs the blanket deterministic 32.8/17.4. Adding a `spike_risk`
context dimension (upper-tail severity, overriding the gate) flipped it:

| spike 1.5, pool 12, 16 seeds | SLA | prodSLA |
|---|---|---|
| uncertainty-sized (deterministic) | 32.8 | 17.4 |
| llm-margin **7b / 14b** | **32.0** | 17.4 |
| llm-margin 3b | 34.4 | 17.4 |

Mild-tail regression stays clean (14b 23.8/11.9). **With the spike-risk signal, 7b+ beats the
deterministic margin in BOTH regimes with an auditable justification per hedge** — the
demand-side echo of the Stage-1 arc: the cure for an LLM blind spot is the missing
signal/structure, not scale. (3b improved but still over-hedges — the weak model stays weak.)


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

## Experiment 19 — runtime prediction on the real MIT Supercloud: retrieval beats the LLM (NEGATIVE for the LLM)

The real trace killed the other targets first (GPU memory pinned by TensorFlow whole-GPU
reservation; `tres_req` a flat template; per-job util already 92% median — the real inefficiency
is queueing, so the Stage-1 target is wall-clock runtime + uncertainty). On 3,414 completed DNN
jobs: per-model **retrieval quantiles** win decisively (MAE 172 min, within-2x 57%, rho +0.50,
coverage 0.79 vs the 0.80 nominal) against the best LLM (14b: MAE 236, within-2x 29%, rho +0.11,
coverage **0.16** — badly over-confident), and retrieval also wins at every held-out model family
OOD. **The LLM is out of the Stage-1 runtime path**; its measured value stays in the
negotiation/justification layer. `pins/eval/predict_runtime.py`.

## Experiment 20 — Alibaba v2018: DAG topology predicts task demand (PASS, with a caveat) (2026-06-25)

`extract_dag.py` parses task-name-encoded DAGs from the full 4.2M-job trace: **48.3% of jobs
carry real precedence edges** (Supercloud: 0%) — the make-or-break prevalence holds. Prediction
gate (`predict_dag.py`, GBT quantiles on `plan_mem`, split by job): with the co-requested
`plan_cpu` available topology adds only -1.2% MAE (requests are tier-templated), but **without it
topology substitutes for the co-request** — MAE 0.1175 -> 0.0377 (-68%), rho +0.48 -> +0.86,
intervals stay calibrated. Caveats: the dominant feature is upstream `parent_mem` (partly job
co-membership, not pure graph structure); the target is *requested* memory (v2018 has no usage
file).

## Experiment 21 — DAG -> ACTUAL GPU, executed on the A100: a one-line rule wins (2026-06-25)

No public trace pairs DAGs with measured GPU, so labels were minted by running rank-matched CNN
stand-ins on real v2018 topologies (real durations govern overlap). **Peak-concurrent GPU memory
is additive**: naive sum fails (117-154% MAPE) but the heaviest-(depth-)level rule predicts it to
**~3% MAPE, ~100% within-1.5x** on synthetic AND real DAGs. Utilisation is a dead learned target
(every CNN saturates the A100 solo; co-run util pegs at 100%); co-location slowdown is large and
super-linear (2.4x/3.5x at k=2/3) but **k-determined, not mix-dependent** (std <= 0.09) — a
`slowdown ~ k` rule suffices. The one open door for a relational model: HETEROGENEOUS bottlenecks
(compute vs bandwidth) — picked up by Exp 23. Deterministic rules keep beating learned models
(the Exp 9/18 pattern). `pins/eval/dag_gpu_bench.py`, `dag_gpu_trace_bench.py`,
`gpu_coexec_probe.py`.

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
reserve (Exp 14-15) over one free pool; 4 policies on identical workloads: no-llm floor /
isolated / **negotiated** (bounded concession protocol, `pins/negotiation_protocol.py`) /
**single-llm** (one LLM, both objectives — the must-have baseline, `llm_joint`).

**Findings that SURVIVED:** (1) **single-llm over-commits both levers** with no brake — the worst
overall-SLA agent under contention (pool 6 @3b: util 87%, only 14.5/16 done); the negotiation's
concession ladder is exactly the missing brake. (2) negotiated uniquely beats the floor at slack
(pool 12: 31.2 vs 33.6) by restraint. **ARTIFACTS, corrected later:** the 96/74/49% negotiation
fallback and the resulting "negotiated is byte-identical across 3b/14b" were caused by
non-negotiable base demand polluting the margin table (fixed in Exp 24) on top of unrealistic
flat-8 synthetic caps (fixed at the source in Exp 27) — with both fixed, fallback is 0%
everywhere and the model tiers genuinely separate.

## Experiment 23 — LLM affinity hint + ILP placement: a knife-edge gated on model size (compressed)

**Date:** 2026-06-26 · the Exp-21 open door, executed. The LLM only CLASSIFIES each task's
bottleneck (compute vs bandwidth) from its op-profile text; code builds the affinity matrix; the
ILP places within hard capacity/co-location constraints. At heterogeneity 0.5 (blind spread 6.46,
oracle 6.87 throughput):

| classifier | acc | throughput | verdict |
|---|---|---|---|
| keyword rule | 6/8 | 6.46 | ties spread |
| qwen2.5:3b | 6/8 | **6.04** | **WORSE than no hint** (confident mislabels mis-separate pairs) |
| qwen2.5:14b | **8/8** | **6.87** | **= oracle** |

Homogeneous workloads: all placers tie (no signal, per Exp 21). **The guarantee held regardless:**
every wrong 3b hint cost throughput, none produced an infeasible placement. An LLM in the
placement loop is justified only at >=14b with a fallback. `pins/affinity.py`,
`pins/placement_affinity_sim.py`.


## Experiment 24 — Contested-slice negotiation: the fallback was an ARTIFACT; the two-sided split now BEATS the single-LLM

**Date:** 2026-06-29

**Why.** Exp 22's `negotiated` policy fell back to the heuristic 49–96% of the time, which made it
(a) byte-identical across 3b/14b (the LLM's choices never survived) and (b) "a wash" vs the
single-LLM baseline. Exp 22's own next-step (a) blamed *"`negotiate()` is handed the WHOLE pool so
aggregate demand exceeds it."* This experiment implements that fix — and finds the diagnosis was
only half right.

**The fix, in two steps.**
1. **The literal "pass the contested slice not the whole pool" is a NO-OP.** Subtracting base demand
   from both sides of the fit test is algebraic identity: `base+margins ≤ free−reserve`
   ⟺ `margins ≤ (free−base)−reserve`. The fallback condition reduces to *base demand > pool* — which
   under contention is **genuine oversubscription, not an artifact of the number passed**. So the
   high fallback was NOT premature in the way the note assumed; verified empirically (rule sweep
   unchanged, fb still 96/74/53%).
2. **The real bug: the demand table mixed non-negotiable base into the margin negotiation.** It
   handed `negotiate()` *every active job's full base* `cap0`, including jobs that won't run this
   tick. But a speed-up **margin is only usable by a job already RUNNING its base** — a waiting job
   needs base first, which is the auction's job, not the margin negotiation's. Fixed
   (`two_sided_sim.simulate`): the margin table is the **running train jobs only** (`held ≥ cap0 > 0`,
   `forecast_cap=0`), contesting the **genuinely-free GPUs** (`total − Σheld`) against the supply
   reserve. Base demand never enters `want`, so the negotiation cannot false-fallback on base
   oversubscription. (`negotiate()`'s `want`/`avail` were also made explicit — margins vs slice;
   `aggregate_joint_ctx` guards the empty-table tick. `test_mechanism.py` 5/5, decider untouched.)

**Result — `negotiated` across agents (8-seed mean; SLA / prodSLA, lower = better; fb now 0% everywhere).**

| pool | rule | qwen2.5:3b | qwen2.5:14b |
|---|---|---|---|
| 6  | 89.8 / 95.4 | 91.4 / 95.4 | 89.8 / 95.4 |
| 8  | 71.1 / 71.3 | 73.4 / 71.3 | 71.1 / 71.3 |
| 12 | 34.4 / 33.1 | 35.2 / 38.0 | 35.2 / **33.4** |

**Result — head-to-head at 14b (the load-bearing comparison; util%, done/16 shown).**

| pool | policy | SLA | prodSLA | util | done |
|---|---|---|---|---|---|
| 6  | no-llm (floor) | **88.3** | **93.3** | 98 | 15.6 |
| 6  | **negotiated** | 89.8 | 95.4 | **96** | 15.6 |
| 6  | single-llm | 91.4 | 95.4 | 90 | **15.0** |
| 8  | no-llm | 72.7 | 77.0 | 95 | 16.0 |
| 8  | **negotiated** | **71.1** | **71.3** | 94 | 16.0 |
| 8  | single-llm | 74.2 | 73.4 | 90 | 16.0 |
| 12 | no-llm | **33.6** | 39.0 | 80 | 16.0 |
| 12 | **negotiated** | 35.2 | **33.4** | 82 | 16.0 |
| 12 | single-llm | 35.2 | 37.7 | 79 | 16.0 |

**Findings.**
1. **The fallback was an artifact of the WRONG decision boundary, not the pool size.** Once base
   demand is removed from the negotiation, fallback drops **96/74/49% → 0%** at every pool: the
   negotiation always converges, because margins over the free pool can always be conceded to fit.
   The two-sided protocol now actually *runs* instead of degenerating to the heuristic.
2. **Model size now MATTERS for the negotiation** (it could not in Exp 22 — fallback washed it out).
   The LLM's per-job hedge choices survive, so 14b separates from 3b: pool-12 prodSLA **33.4 (14b)
   vs 38.0 (3b)**; 14b matches the deterministic rule (the oracle) while 3b over-hedges and self-harms
   — the same "judgement with stakes" pattern as Exp 14C/17.
3. **The headline FLIPPED: `negotiated@14b` now beats or ties `single-llm@14b` at EVERY pool**
   (was "a wash" in Exp 22). single-llm consistently **over-commits** — util 90/90/79 and only
   15.0/16 done at pool 6 — because a lone agent has no brake; the negotiation's **concession ladder
   is exactly that brake** (util 96/94/82, 15.6 done). And the per-job margin granularity protects
   prod better than single-llm's one uniform hedge (pool-12 prodSLA 33.4 vs 37.7). This is the first
   measurement where the two-sided split wins the load-bearing comparison on the NUMBERS, not just on
   interpretability — corroborating and strengthening Exp 22 finding #1.
4. **vs the floor:** `negotiated@14b` wins prod-tier SLA at pool 8 (71.3 vs 77.0) and pool 12 (33.4
   vs 39.0), trading a little overall SLA at pool 12 (35.2 vs 33.6). Its character shifted from
   Exp 22's "win overall-SLA by restraint" to "protect prod deadlines by granting margins to running
   at-risk jobs" — the value-weighted behaviour the thesis wants (Exp 11/12 lineage).

**Honest read / caveats.** The win is concentrated in **prod-tier SLA at moderate contention**
(pools 8/12, where free GPUs exist); at pool 6 (util 96–98%) the slice is near-empty and all
policies converge, as every lever in this project does under saturation. The contested-slice
restriction also changed `single-llm`/`isolated` slightly (all policies now see the running-jobs
table) — the comparison stays apples-to-apples. Reserve still competes with margin for the free
pool, but reclaim/preemption costs are not modelled here (Exp 14A regime); single 8-seed synthetic
workload; the committed/reserve grant still stands in for the full `mechanism.clear` + ILP placement
(follow-up (b)).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.negotiation_protocol            # smoke: contested-slice agreement + no-slice fallback
.venv/bin/python -m pins.two_sided_sim                   # rule-fallback 4-policy sweep (fb now 0%)
.venv/bin/python -m pins.two_sided_sim --llm --model qwen2.5:3b
.venv/bin/python -m pins.two_sided_sim --llm --model qwen2.5:14b
```
`pins/two_sided_sim.py` (`simulate`: running-jobs margin table + `free_now`); `pins/
negotiation_protocol.py` (`negotiate` margins-vs-slice, `aggregate_joint_ctx` empty guard);
artifact `pins/results_two_sided.json`.


## Experiment 25 — The full locked pipeline end-to-end: the ILP guarantee makes the LLM's over-demand SAFE

**Date:** 2026-06-29

**Why.** Exp 22/24 resolved the demand-margin ⇄ supply-reserve negotiation but then allocated with a
**bespoke prod-first grant** — a stand-in for the decider. The locked architecture
(`research_progress.md` 2026-06-25; `Research/CLAUDE.md` §3) is *LLMs reason/bid → committed-auction
decides (who/how-many) → ILP places (where) / guarantees*. Follow-up (b): replace the stand-in with
the **real deciders** so the merged sim exercises the pipeline end-to-end.

**Method.** New `pins/pipeline_sim.py` composing the validated harness (imported, unmodified): the
Exp-22/24 world (rigid incumbents + stochastic train-work spikes; a margin GPU buys spike-absorbing
speed) lifted onto a **node cluster** so placement bites. Per tick: **RATION** = the committed-auction
(bid-once frozen-urgency priority + run-to-completion `_serialise` from `negotiation_sim.py` — the
Exp-11 SLA winner, *not* the per-round `mechanism.clear`, which Exp 9/18A showed spreads+thrashes)
over `(total − reserve)`; **PLACE** = `place_sticky` (count-only, fragments) vs
`ilp.allocate_placement` (plans node + migrates to consolidate, feasible by construction). Negotiation
= the Exp-24 contested slice (running train jobs contest the free GPUs for a margin vs the reserve).
A clean **2×2: negotiation {off,on} × placement {sticky,ILP}** — `floor`, `floor+ILP`, `nego+sticky`,
`pipeline`. Contended regime (32 jobs, arrivals compressed ≤48 so whole-node train jobs fragment the
cluster, horizon 400, 4 seeds, nodes {2,3,4,6}×8). `ploss` = mean GPUs/round won but unplaceable.

**A bug found & fixed en route.** `bid_with_margin` first read `job.phase()`, but the sim tracks each
job's phase in a local dict and never mutates the `Job` — so every bid was stuck at the *initial*
phase (preprocess, length 1), starving every train job to 1 GPU (100% SLA at any cluster size). Fixed
to pass the current phase explicitly. (`test_mechanism.py` still 5/5; no validated module touched.)

**Result (agents=qwen2.5:14b; SLA / prodSLA / util / ploss; lower SLA/prodSLA/ploss = better).**

| cluster | floor | floor+ILP | nego+sticky | pipeline |
|---|---|---|---|---|
| 2×8=16 | 82.8 / 45.3 / 96 / 0.17 | **82.0 / 42.2** / 97 / **0.08** | 82.8 / 45.3 / 93 / 0.60 | **82.0 / 42.2** / 94 / 0.53 |
| 3×8=24 | 71.9 / 23.5 / 92 / 0.62 | **70.3 / 21.0** / 94 / **0.20** | 73.4 / 29.7 / 89 / 1.35 | **70.3 / 21.0** / 91 / 0.97 |
| 4×8=32 | 58.6 / 6.6 / 88 / 0.93 | **57.0 / 6.6** / 90 / **0.60** | 58.6 / 6.6 / 85 / 2.16 | **57.0 / 6.6** / 87 / 1.63 |
| 6×8=48 | 32.0 / 5.4 / 79 / 1.59 | **28.9 / 2.3** / 80 / **1.08** | 34.4 / 7.5 / 78 / 3.04 | 30.5 / **2.3** / 80 / 2.21 |

(`floor`/`floor+ILP` are negotiation-free → identical at rule and 14b; the negotiation arms are 14b.
Rule-fallback negotiation arms are milder, e.g. 6×8 `nego+sticky` ploss 2.09 / SLA 31.2 vs 14b's 3.04 / 34.4.)

**Findings.**
1. **The ILP placement layer delivers Exp-18's win INSIDE the full pipeline.** `floor+ILP` vs `floor`
   roughly **halves the fragmentation loss** (ploss 1.59→1.08, 0.93→0.60, 0.62→0.20, 0.17→0.08) and
   lifts SLA/prodSLA/util at every cluster (6×8: SLA 32.0→28.9, prodSLA 5.4→2.3). The placement value
   survives being fed by the committed-auction + negotiation rather than a raw bid stream.
2. **The committed-auction + contested-slice negotiation carry over cleanly:** fallback **0%**
   everywhere (the Exp-24 property holds), all 32 jobs finish, util stays 79–97%.
3. **A bigger LLM bids MORE aggressive margins — which BACKFIRE under count-only placement.** 14b
   grants more headroom than the rule (ploss up across the board), and under `sticky` placement those
   bigger blocks **fragment the cluster and make `nego+sticky` WORSE than the no-negotiation floor**
   (6×8: SLA 32.0→34.4, prodSLA 5.4→7.5; 3×8: prodSLA 23.5→29.7). Demanding more GPUs is actively
   harmful when there is no way to consolidate them.
4. **The ILP guarantee RESCUES the aggressive negotiation — the headline.** `pipeline` (nego+ILP)
   absorbs the fragmentation the margins caused and pulls back to the `floor+ILP` frontier
   (6×8: SLA 34.4→30.5, prodSLA 7.5→2.3; ploss 3.04→2.21). So the ILP placement layer is precisely
   **what makes the LLM's over-demand safe** — the design hinge *"the LLM reasons/bids; deterministic
   code decides and GUARANTEES"* demonstrated end-to-end: the guarantee layer protects the system from
   the proposer's excess, exactly the LLMSched complementarity (`Research/CLAUDE.md` §2).

**Honest read / caveats.** In this heavily-contended regime the negotiation does **not add SLA over
the floor** — margins need slack (Exp 24), and under saturation the pipeline's net effect is the ILP's
placement gain plus the safety it lends the negotiation, not a new negotiation SLA win. The committed
counts are fed to `allocate_placement` as flat priority-weighted demands (it places/migrates, does not
re-ration); the committed-auction deliberately stands in for per-round `mechanism.clear` (shown worse
Exp 9/18A); reserve is folded as a scalar pool holdback; spike_max 0.6 (mild), 4 seeds, co-location +
sticky one point on the placement-rigidity axis; CBC per round is the cost (~50 s for the rule sweep,
within LLMSched's budget framing). The pipeline now matches the locked architecture end-to-end — the
remaining gap to "production" is the incentive layer (c) and richer placement (d).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.pipeline_sim                                   # rule-fallback 2×2 sweep
.venv/bin/python -m pins.pipeline_sim --llm --model qwen2.5:14b         # LLM negotiation arms
(cd pins && ../.venv/bin/python test_mechanism.py)                      # deciders untouched → 5/5
```
`pins/pipeline_sim.py` (`simulate`: committed `_serialise` ration + `place_sticky`/`allocate_placement`;
`bid_with_margin` takes the phase explicitly); reuses `negotiation_sim`, `ilp`, `placement`,
`negotiation_protocol`, `two_sided_sim.job_facts`, `uncertainty_sim`, `bridge`.


## Experiment 26 — reflective margin agent: NEGATIVE — reflection thrashes into a limit cycle (compressed)

**Date:** 2026-06-30 · `pins/reflective_sim.py`. Can a weak model (3b) reflect its way to the
14b/oracle margin policy from per-state outcome feedback? **No — it enters an exact period-2
limit cycle** (metrics of cycles 0=2=4 vs 1=3=5; 40-41 policy changes every cycle, 40/41 of them
reverting to a previously-held hedge), bouncing at SLA 15.6-16.4% vs the oracle/14b's 14.8% and
never converging. The failure is itself interpretable: 25 revisions cut the hedge to `none` in
states that had spiked >=50% of the time — the model reasons from realised deadline misses, not
spike rate, so each "fix" re-creates the problem it fixed. Un-stabilised naive reflection does
NOT substitute for a bigger model (damping/hysteresis + a spike-rate signal are the identified
fixes); the deterministic concession ladder is what actually makes the small model safe (shown in
Exp 27/29).


## Experiment 27 — REAL Stage-1 predicted GPU caps wired into the two-sided sim: the fallback problem vanishes, negotiation buys prod-SLA at slack cost

**Date:** 2026-07-01/02

**Why.** Every two-sided run so far (Exp 22/24/25) gave the train phase a FLAT synthetic cap of 8
GPUs per job — the one number in the loop that was never a Stage-1 output. The Stage-1 GPU track
(`pins/eval/predict_gpu.py`) now produces real per-job predicted requests, so this closes the last
stubbed input: the demand agent negotiates over the job's ACTUAL forecast demand.

**Method.** `load_gpu_distribution()` + `assign_gpu()` in `pins/uncertainty_sim.py` draw each job's
train-phase cap (`forecast_cap`, per-seed deterministic) from the 2000 predicted P50 `plan_gpu%`
values in `pins/eval/results_gpu.json`. v2020 is a GPU-SHARING trace — plan_gpu is fractional
(25 = ¼ A100) and the predicted P50s cluster at 25/50/100%, so rounding to whole GPUs collapses
~all jobs to 1; caps are instead expressed in the trace's natural QUARTER-GPU quantum
(`max(1, round(pct/25))` → a genuine ~1/2/4 spread, mean ~2.35). `two_sided_sim.py` uses
`forecast_cap` as the non-negotiable base; pools rebalanced to {3,4,6} because the old {6,8,12}
left the pool near-idle once flat-8 caps were replaced by realistic ones. Same 4 policies,
16 jobs, horizon 300, 8 seeds, spike_max 0.6, scale 3.

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
| 4 | single-llm | 35.9 | 24.3 | 76 | — |
| 6 | no-llm | **7.0** | 14.4 | 58 | — |
| 6 | isolated | 9.4 | **10.1** | 58 | — |
| 6 | negotiated | **7.0** | 12.6 | 58 | 0% |
| 6 | single-llm | 10.2 | 12.6 | 58 | — |

All policies finish 16/16 at every pool.

**Result (LLM tiers, same sweep; SLA / prodSLA / util; slowdown for the over-commit story).**

| | | **qwen2.5:3b** | | | | **qwen2.5:14b** | | |
|---|---|---|---|---|---|---|---|---|
| pool | policy | SLA | prodSLA | util | slow | SLA | prodSLA | util |
| 3 | isolated | 64.1 | 50.7 | 81 | 3.00 | 64.1 | 52.9 | 84 |
| 3 | negotiated | **59.4** | **48.3** | 83 | 2.74 | 63.3 | 50.7 | 86 |
| 3 | single-llm | 65.6 | 50.7 | 75 | 3.20 | 65.6 | **48.9** | 75 |
| 4 | isolated | 40.6 | 28.2 | 75 | 1.75 | 35.9 | 26.4 | 77 |
| 4 | negotiated | **34.4** | **26.4** | 77 | 1.65 | 35.9 | **26.1** | 78 |
| 4 | single-llm | 43.0 | 28.2 | 74 | 1.79 | 39.8 | 26.4 | 74 |
| 6 | isolated | 11.7 | 14.4 | 62 | 1.10 | 7.8 | **10.1** | 60 |
| 6 | negotiated | **7.0** | 12.6 | 62 | 1.04 | **7.0** | 12.6 | 61 |
| 6 | single-llm | 12.5 | **10.1** | 61 | 1.14 | **7.0** | **10.1** | 60 |

(no-llm floor identical to the rule table: 57.0/62.3, 30.5/34.9, 7.0/14.4. fb = 0% everywhere,
all tiers, all pools.)

**Findings.**
1. **The Exp-22 fallback pathology is GONE: fb = 0% everywhere** (was 96/74/49%). With realistic
   quarter-GPU-quantum demand, aggregate forecast demand no longer swamps `free_gpus`, so the
   negotiation actually negotiates instead of collapsing to the heuristic. The Exp-22 "negotiate
   over the contested slice" fix and this change attack the same root cause — demand/pool ratio —
   from opposite ends; real caps fix it at the source.
2. **A clean SLA ⇄ prodSLA trade emerges at every pool.** The floor (no-llm) wins overall SLA;
   every agent policy buys prod-SLA with best-effort slowdown — most sharply `negotiated` at
   pool 3: prodSLA 62.3 → 47.1 (−15.2 pts) for +3.9 overall. With flat-8 caps this trade was
   masked by the fallback; with real caps it is the headline behaviour.
3. **Negotiated dominates single-llm at the extremes** (pool 3 both metrics; pool 6 both), and at
   pool 6 it is the only agent policy to tie the floor's overall SLA while still improving prod —
   the "restraint" property (Exp 22 finding 3) survives the switch to real demand.
4. **With fb = 0% the model finally matters — `negotiated` is no longer model-invariant.** Exp-22
   finding 2 ("byte-identical 3b vs 14b") was an artifact of the 96/74/49% fallback; with real
   caps the agents' positions actually reach the ladder, and the tiers now differ (pool-3 SLA:
   60.9 rule / 59.4 3b / 63.3 14b).
5. **The bounded protocol is a guardrail that SUBSTITUTES FOR SCALE.** At 3b, `negotiated` is the
   best agent policy on overall SLA at every pool (59.4/34.4/7.0 vs isolated 64.1/40.6/11.7 and
   single-llm 65.6/43.0/12.5) — and negotiated@3b matches or beats negotiated@14b (pools 3–4). *(Exp 29, 32 seeds on trace replay: this inversion is a statistical TIE — read as sufficiency, not superiority.)*
   The lone agent shows the mirror image: `single-llm@14b` at pool 6 ties the floor's SLA and
   takes best prodSLA (7.0/10.1) where 3b self-harms (12.5) — model size rescues the un-braked
   agent (Exp-22 finding 4), but the brake makes the bigger model unnecessary. This is the
   "mechanism substitutes for a bigger model" claim Exp 26 could NOT get from reflection,
   delivered instead by the deterministic concession ladder.
6. **`single-llm` still over-commits** — lowest util (75/74%) and worst slowdown (3.14–3.20) at
   contention in BOTH LLM tiers; Exp-22 finding 1 survives real caps intact.

**Honest read / caveats.** All three tiers measured (rule / 3b / 14b, tables above). En route this
experiment exposed and fixed a cache-clobbering bug: `sweep()` started from an EMPTY cache dict and
`save_cache()` overwrote the disk file, so every run destroyed the previous model's cached
decisions (this is how the original 2026-07-02-morning 14b trace was lost). `save_cache` now merges
with disk, and `sweep` warm-starts from `load_cache()` + checkpoints per pool — re-runs are
Ollama-free. Remaining caveats: the P50 request is a *point* choice — the [P10,P90] width in
`per_job_gpu` is still unused for margin sizing (margins remain `HEDGE_GPUS` levels); pools were
hand-picked to restore contention (a demand/capacity-ratio-matched comparison vs the flat-8 world
would be cleaner); and the negotiated 3b-over-14b inversion (finding 5) is from a single 8-seed
synthetic workload — run a seed sweep before leaning on it in the thesis.

**Conclusion.** Once the negotiation runs over real Stage-1 predicted demand instead of synthetic
caps, the fallback pathology disappears — and the deterministic protocol turns out to SUBSTITUTE
FOR MODEL SCALE. Three claims the data supports: (1) realistic demand fixed the *mechanism*, not
the model — Exp 22's 96/74/49% fallback was an artifact of flat-8 caps swamping the pool; at real
quarter-GPU quanta it is 0% everywhere and the negotiation finally negotiates. (2) What negotiation
buys is prod-SLA protection priced in best-effort slowdown — the floor always wins raw overall SLA;
the agents deliberately, auditably reprioritise (pool 3: prodSLA 62.3 → ~47-48). (3) The brake
matters more than the brain: negotiated@3b matches or beats negotiated@14b, while the un-braked
single-llm self-harms at 3b and needs 14b to recover — the concession ladder is a guardrail that
makes a SMALL model safe, the "mechanism substitutes for a bigger model" result Exp 26 failed to
get from reflection. Thesis slot: Exp 25 showed the ILP makes aggressive LLM bids *safe*; Exp 27
shows the protocol makes a *cheap* LLM *sufficient* — the guarantees don't just protect against
the LLM's mistakes, they let you run a much smaller LLM at all. (Seed-sweep the 3b≥14b inversion
before it becomes a headline claim — **done in Exp 29: it is a statistical tie; the sufficiency claim stands**.)

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_gpu               # regenerates results_gpu.json (per_job_gpu)
.venv/bin/python -m pins.two_sided_sim                  # rule tier (table above)
.venv/bin/python -m pins.two_sided_sim --llm --model qwen2.5:3b   # LLM tiers (cache-warm = fast)
.venv/bin/python -m pins.two_sided_sim --llm --model qwen2.5:14b
```
`load_gpu_distribution` / `assign_gpu` in `pins/uncertainty_sim.py`; consumed at
`two_sided_sim.py` sweep (`cap_map`); artifact `pins/results_two_sided.json`.

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

**Why.** Exp 28's own caveat: caps were the trace's **real** requests, so the "prediction feeds
negotiation" story was demonstrated with ground truth on the demand side. This experiment finally
separates the two: the agents request/negotiate over the **Stage-1 GBT prediction** for each
replayed job, while the job's true demand governs what it can actually do with the grant —
under-prediction starves a job (rate<1 even fully granted), over-prediction hogs GPUs it cannot
convert into progress. Prediction ERROR, not just prediction, now hits outcomes.

**Method.** `predict_gpu.py` now emits per-JOB predicted quanta keyed by `job_name`
(test split only — 254,476 jobs; aggregated `Σ p50·inst_num/25`, the same recipe as
`replay_jobs.csv` aggregates the truth) → `pins/eval/pred_job_gpu.csv`. `two_sided_sim.simulate`
gains an optional `true_cap_map`: `cap_map` (what agents see/request) and the progress-rate
denominator (truth) decouple; `None` reproduces Exp 28/29 byte-for-byte. `trace_replay --caps
predicted` restricts windows to prediction-covered jobs and requests the P50 prediction;
`--caps oracle` requests the truth **on the same windows** — the matched control, so
pred − oracle, paired by seed, *is* the cost of prediction error per policy. Rule tier, 32 seeds,
pools {4,6,8}. fb = 0% everywhere, both modes.

**Result (paired by seed, n=32; * = 95% CI excludes 0).**

Cost of prediction error (pred − oracle, same windows; + = worse):

| pool | policy | ΔSLA (pts) | ΔprodSLA (pts) |
|---|---|---|---|
| 4 | no-llm floor | +5.3 ±3.2* | +5.2 ±5.1* |
| 4 | negotiated | +4.1 ±2.6* | +4.1 ±5.2 |
| 6 | no-llm floor | +6.6 ±3.0* | +8.2 ±5.6* |
| 6 | negotiated | +5.7 ±3.2* | +10.1 ±6.5* |
| 8 | no-llm floor | **+8.8 ±3.8*** | **+9.7 ±5.4*** |
| 8 | negotiated | **+6.1 ±3.3*** | +5.0 ±6.2 |

Diff-of-diffs (does the policy absorb prediction error better than the floor? − = yes):

| pool | negotiated | single-llm |
|---|---|---|
| 4 | ΔΔSLA −1.2 ±3.0 · ΔΔprodSLA −1.1 ±5.6 | −2.0 ±3.4 · −2.1 ±5.0 |
| 6 | ΔΔSLA −1.0 ±2.4 · ΔΔprodSLA +1.8 ±5.5 | −0.6 ±2.4 · +3.5 ±5.5 |
| 8 | **ΔΔSLA −2.7 ±2.1*** · **ΔΔprodSLA −4.7 ±4.3*** | **−2.3 ±2.3*** · **−4.7 ±3.7*** |

And within the predicted-caps world, negotiated − floor at pool 8: SLA −1.0 ±2.1 ns, prodSLA
**−6.2 ±4.3***, slowdown −0.7 ±0.7 (marginally *negative* — the agents no longer even pay latency).

**Findings.**
1. **Prediction error is expensive for everyone** — +4 to +10 SLA/prodSLA points across policies
   and pools (GBT plan_gpu prediction: MAE ~27 plan_gpu%, ρ 0.56 — a realistic, imperfect
   predictor, not a straw man).
2. **At slack (pool 8) the negotiation measurably CUSHIONS prediction error**: it absorbs
   2.7 SLA pts and 4.7 prodSLA pts of the floor's error cost (diff-of-diffs significant). The
   mechanism's margins + reserve act as insurance against demand misestimation — the Exp-16
   "uncertainty is insurance" story reappearing at the system level, unprompted.
3. **The Exp-29 headline survives prediction error**: with predicted caps, negotiated still buys
   significant prod-SLA protection (−6.2 pts) at no significant overall-SLA or slowdown cost.
   The end-to-end claim — *Stage-1 prediction (with its real errors) feeds Stage-2 negotiation and
   the value proposition holds* — is now measured, not assumed.
4. Regime unchanged: at saturation (pool 4) nothing cushions anything; every lever in this project
   needs slack.

**Honest read / caveats.** Rule-tier agents only (LLM tiers on predicted caps not yet run —
Exp 29 suggests they would track the rule closely); the P50 is still a *point* request (the
[P10,P90] width in `pred_job_gpu.csv` sits unused — sizing the hedge from it is the natural next
step and now has job-level plumbing); windows restricted to predict_gpu's test jobs (25% split;
~550 arrivals per 10-h window, plenty); oracle mode requests the truth but both modes clip at
CAP_CLIP=8 and the pool. The prediction is of the *requested* GPU (plan_gpu), not measured usage —
same honest framing as the Stage-1 GPU track itself.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_gpu                        # regenerates pred_job_gpu.csv (~2 min)
.venv/bin/python -m pins.trace_replay --seeds 32 --caps oracle   # matched-window control
.venv/bin/python -m pins.trace_replay --seeds 32 --caps predicted
.venv/bin/python -m pins.trace_replay --stats                    # includes rule+pred vs rule+oracle
```
`true_cap_map` in `two_sided_sim.simulate`; `--caps` + `load_predicted_quanta` in
`pins/trace_replay.py`; tiers `rule+pred` / `rule+oracle` in `pins/results_trace_replay.json`.

**Addendum (2026-07-06): the 3b LLM tier measured (`qwen2.5:3b+pred` / `+oracle`, 32 seeds).**
1. **The headline strengthens with LLM agents in the loop:** under predicted caps,
   `negotiated@3b` buys significant prod-SLA protection at EVERY pool (−8.9 ±5.8* / −7.6 ±4.6* /
   −8.7 ±4.4* at pools 4/6/8) at no significant overall-SLA or slowdown cost — prod protection
   survives prediction error with the actual LLM agents, not just the rule.
2. **Prediction error still costs the 3b agents** (+3.7/+6.4/+7.0 SLA pts*, pred − oracle) —
   same order as the rule tier.
3. **The pool-8 cushion points the right way at 3b but loses significance** (diff-of-diffs
   −1.8 ±2.7 SLA / −2.4 ±4.1 prodSLA vs the rule tier's −2.7*/−4.7*): the LLM agents' extra
   decision variance widens the CIs. The "negotiation cushions prediction error" claim should be
   stated as rule-tier-significant, direction-consistent at 3b.

## Experiment 31 — UNCERTAINTY SIZES THE REQUEST: agents ask for a quantile of the predicted interval

**Date:** 2026-07-07

**Why.** Exp 30's own caveat: the P50 was a *point* request while `pred_job_gpu.csv` carries a
per-job [P10, P90] interval that sat unused. The request quantile is a newsvendor choice —
under-request starves the job (rate < 1 even fully granted), over-request hoards GPUs it cannot
convert into progress — and the interval lets a job *choose where to sit* on that trade-off,
which a point predictor cannot. This is the thesis sentence ("uncertainty sizes the margin")
finally tested with REAL Stage-1 intervals at the system level. Note the hedge is automatically
width-sized: after rounding+clipping, P90 raises the request on ~40% of jobs (mean +0.81 quanta)
and P10 lowers it on ~70% (mean −1.37) — jobs with wide intervals hedge more.

**Method.** One knob: `trace_replay --caps predicted --quantile p10|p50|p90` picks which column
of `pred_job_gpu.csv` the agents request. Coverage (the key set) is identical for every
quantile, so windows — and therefore seeds — stay perfectly paired across all pred tiers AND
the Exp-30 oracle tier. Rule tier, 32 seeds, pools {4,6,8}; tiers `rule+pred-p10` /
`rule+pred-p90` land beside Exp 30's `rule+pred` (P50) and `rule+oracle`.

**Result (paired by seed, n=32; * = 95% CI excludes 0).**

Quantile vs the P50 request, negotiated policy (+ = quantile worse):

| pool | p10 − p50 | p90 − p50 |
|---|---|---|
| 4 | ΔSLA **+5.3 ±4.3*** · Δutil −10.5* | ΔSLA −1.6 ±3.5 · ΔprodSLA **−6.7 ±6.3*** |
| 6 | ΔSLA **+10.7 ±4.6*** · Δutil −14.1* | ΔSLA +0.0 ±3.8 · ΔprodSLA −5.4 ±6.0 |
| 8 | ΔSLA **+17.2 ±6.4*** · ΔprodSLA **+14.5 ±8.9*** | ΔSLA −1.6 ±4.0 · Δslow **+3.1 ±1.6*** |

(No-llm floor and single-llm show the same pattern; p10 is +5..+17 SLA pts worse for every
policy at every pool.)

Does the hedge buy back the prediction-error cost? p90-pred − oracle (same windows):

| pool | policy | ΔSLA | ΔprodSLA |
|---|---|---|---|
| 4 | negotiated | +2.5 ±3.7 | −2.6 ±6.3 |
| 6 | negotiated | +5.7 ±3.6* | +4.7 ±7.4 |
| 8 | negotiated | +4.5 ±3.4* | +1.7 ±5.6 |

vs Exp 30's P50-pred − oracle prodSLA cost of +4 to +10 pts*: **on prodSLA, the P90 hedge is
statistically indistinguishable from the oracle at every pool and every policy** — hedging
recovers essentially the whole prediction-error cost for the protected tier. Overall SLA still
pays (+4.5..+6.6* at pools 6/8) and best-effort slowdown worsens (+2.3..+3.0* at slack).

Negotiated − floor *within* each request world (does negotiation still earn its keep?):

| world | pool 4 | pool 8 |
|---|---|---|
| p10 | ΔprodSLA **−3.9 ±2.9*** | ΔprodSLA **−3.8 ±3.1*** · Δslow −0.5* |
| p50 (Exp 30) | ΔprodSLA −3.1 ±4.1 | ΔprodSLA **−6.2 ±4.3*** |
| p90 | ΔSLA **−2.0 ±1.8*** · Δslow **−2.2 ±1.9*** | ΔprodSLA −1.2 ±4.0 |

**Findings.**
1. **The quantile choice is violently asymmetric.** Requesting P10 is catastrophic (+5..+17 SLA
   pts, worst at slack where there was room it refused to ask for); requesting P90 is roughly
   free on overall SLA and *buys* 2–8 prodSLA pts. With CAP_CLIP and pool clipping, the system
   sits on the flat side of the newsvendor curve: over-asking is cheap, under-asking is not.
2. **The P90 hedge buys back the prod tier entirely**: p90-pred ≈ oracle on prodSLA everywhere
   (Exp 30's +4..+10* prodSLA error cost → n.s.). The price is best-effort slowdown (+3 ticks
   at slack) and hoarded GPUs (util +4..+10* is *held*, not productive). "Uncertainty sizes the
   margin" is now a measured system-level claim, not a slogan.
3. **The hedge and the negotiation are (partial) substitutes for prod protection.** In the P90
   world the negotiated reserve's pool-8 prodSLA win disappears (−1.2 n.s. vs P50's −6.2*) —
   once every request already carries its insurance, the reserve has little left to protect.
   Conversely in the P10 world negotiation protects prod significantly at EVERY pool: the
   negotiated margins (extra GPUs above base) literally repair under-request, pushing grants
   back toward true demand. Negotiation is insurance against *mis-sized* requests, whichever
   direction they miss.
4. **At saturation + P90, negotiation shows its first significant overall-SLA win at pool 4**
   (−2.0* SLA, −2.2* slowdown vs floor): when everyone over-asks, contention is partly
   artificial, and the mechanism that rations margins and reserves headroom wastes less of it.

**Honest read / caveats.** Rule tier only (Exp 29/30 suggest LLM tiers track it); the quantile
is applied uniformly to all jobs — a per-job newsvendor rule (hedge only prod / only wide
intervals) is the refinement, and finding 3 says it should target the jobs the reserve can't
cover; util counts held GPUs, so the P90 util gain is partly hoarding by construction;
CAP_CLIP=8 + pool clipping compress the hedge for big jobs (the asymmetry may soften with a
higher clip). The GBT intervals are plain quantile regression (NOT conformal-calibrated — that
was Exp 17's runtime track; porting its calibration here is cheap if coverage proves off), and
the P90 is still a *requested-plan_gpu* quantile, same framing as Exp 30.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.trace_replay --seeds 32 --caps predicted --quantile p90
.venv/bin/python -m pins.trace_replay --seeds 32 --caps predicted --quantile p10
.venv/bin/python -m pins.trace_replay --stats   # cross-tier pairs incl. rule+pred-p10/-p90
```
`--quantile` + quantile-aware `load_predicted_quanta` in `pins/trace_replay.py`; tiers
`rule+pred-p10` / `rule+pred-p90` in `pins/results_trace_replay.json`.

**Addendum (2026-07-07, follow-up l): the per-job newsvendor rule `prod-p90` — hedge only prod.**
`--quantile prod-p90` (prod jobs request P90, best-effort request P50; ~1/3 of jobs hedge, tier
`rule+pred-prod-p90`, same seed-paired windows). 32 seeds, rule tier. Both hypotheses confirmed:

1. **Recovery retained, price collapsed.** prod-p90 ≈ oracle on prodSLA (negotiated: −0.1/+4.2/
   −0.3, all ns) — same recovery as uniform P90 — while vs uniform P90 the price drops
   significantly: slowdown −0.9/−2.0* (pools 6/8), util −3..−7* (less hoarding). Net vs the P50
   baseline: prodSLA −3.1..−7.7 (significant in 5/9 cells), SLA ns everywhere, and the residual
   price is ~+1 tick slowdown* and only +1..+3 util pts (uniform P90 paid +3* and +4..+10*).
   Hedging the third of jobs that matter captures essentially all the value of hedging everyone.
2. **Hedge and reserve become complements again.** Within the prod-p90 world, negotiated − floor
   on prodSLA regains significance at pool 8 (−3.8 ±3.4*; was −1.2 ns under uniform P90) — with
   best-effort no longer carrying insurance, the negotiated reserve adds protection ON TOP of
   the targeted hedge instead of duplicating it. Some substitution remains (p50 world was −6.2*).

Thesis-ready statement: *size the request hedge from the prediction interval, but only for the
tier you protect; let the negotiation insure the rest.* Caveats as Exp 31 (rule tier only,
intervals not conformal-calibrated); one asymmetry unexplained: single-llm pool 6 prod-p90 −
oracle prodSLA +6.1* (the lone cell where the targeted hedge misses oracle — single-llm's
over-commitment interacts with the un-hedged best-effort majority there).

```bash
.venv/bin/python -m pins.trace_replay --seeds 32 --caps predicted --quantile prod-p90
```
