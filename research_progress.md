# Research Progress — PINS experiment log

> **Pruned 2026-07-06:** stepping-stone experiments are compressed to their lessons and
> superseded findings are marked in place; the full original log is in git history
> (`git log -- research_progress.md`, through commit `4a74508`).

## State of the claims (2026-07-06)

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

## Experiment 32 — THE INCENTIVE LAYER: per-user budgets + contested-tick claim pricing

**Date:** 2026-07-07

**Why.** Open problem (c), the claims-table hole since Exp 13: the committed auction trusts
declared priority classes; best-effort jobs lying 'critical' collapse prod protection to the
greedy floor, and a flat per-job budget cannot fix it (any cost on the class hits a liar and an
honest declarer identically, and it taxed honest jobs even uncontested). Exp 13's post-mortem
named the fix — exogenous per-USER budgets, fair-share style, which needs multi-job agents.
Built here, with the measurement upgraded to the question IC actually asks: not "what happens
to system SLA" but **is lying a profitable best response for a single deviating user?**

**Mechanism (`pins/incentive_sim.py`, pure code — no LLM in the loop).** 16 jobs → 4 users
(seeded round-robin). A user's jobs share ONE purse. Each tick, a SERVED job pays its own
declared class cost (`PRIO_CLASS_COST`: critical 4 … low 0) **only when someone else waits**
(contested ticks) — nobody pays in uncontested regimes (Exp 13's flat-budget failure), waiting
costs nothing (you pay for what the claim GOT you). A user at purse ≤ 0 has every job demoted
below 'low' (live, so running jobs lose priority mid-flight). Optional scrip income `B+r`
(start B, earn r/tick, cap B). Deviation = the Exp-13 damaging one: the user's best-effort
jobs claim 'critical', its prod jobs stay honest.

**Two designs tried and REJECTED before it worked** (recorded because both are the "obvious"
choices): (i) *uniform second-price* (price = highest waiting class, charged to all served)
SOCIALISES the lie — probe showed the honest heavy user draining to 16 while the deviator kept
92; (ii) Exp 13's flat per-job budget, already dead. Pay-your-own-claim from a SHARED purse is
what internalises the externality: the probe shows lying multiplies the deviator's own spend
2–40× (honest user-0 spend 0/52/4/48 across seeds → lying 28/124/156/160).

**Result (32 seeds, pools {6,8,12}, paired by seed; deviator gain = user-0's own violation
rate, lie − honest; + = lying hurts the liar).**

| tariff | d(all) pool 6/8/12 | d(own prod) | honest prodSLA cost vs none |
|---|---|---|---|
| none | **−24.2* / −22.7* / −14.1*** (lying pays) | +10/+9/0 ns | — (46.3/21.6/9.1%) |
| 120 lump | +3.9 / −2.3 / −3.1 all ns | **+15.1* / +15.6*** / +7.3 | +9.1 / +5.8 / +3.1 |
| 90+0.3 | +6.2 / +3.9 / −2.3 ns | +15.1* / +18.2* / +6.2 | +10.1 / +7.5 / +4.0 |
| 60+0.5 | +5.5 / +8.6 / +4.7 ns | +3 / +18.2* / +8 | +24.9 / +13.3 / +6.6 |
| 30+0.8 | **+8.6* / +14.1*** / +6.2 (lying punished) | 0 / +7 / +5 | +38.2 / +46.0 / +16.2 |

Victim prod harm (deviation's externality on other users' prod jobs): +6.2*/+9.6* at pools 6/8
unpriced → ns at every priced tariff.

**Findings.**
1. **The vulnerability, restated as best response:** without an incentive layer, lying is
   individually profitable at every pool (−14..−24 pts* net for the deviator, −21..−39* for its
   best-effort jobs) and dumps +6..+10* violation pts on other users' prod jobs.
2. **The layer works, through the designed channel:** at 120-lump the net deviation gain is
   statistically zero at every pool, because the lie now significantly damages the deviator's
   OWN prod jobs (+15*) — the purse the lying best-effort jobs drain is the purse the user's
   critical jobs need. Externality internalised; victim harm goes ns.
3. **Incentive compatibility has a price, and it's a monotone frontier:** the honest world pays
   rationing (+3..+9 prodSLA pts at the neutralising tariff, worse at tighter ones, absurd at
   30+0.8). Anonymous budgets cannot deter for free — the spend-level probe shows honest-heavy
   users (spend up to 196) OVERLAP liars (124–160): spend tracks contested residence, not
   truthfulness (Exp 13's identification limit reappearing at user level). Deterrence operates
   on the deviator's own counterfactual delta, which is exactly what best-response IC needs —
   identification is not required, but neither is it available, so honest heavy users share the
   rationing.
4. **What pricing can NOT do: rescue the all-liars world.** System prodSLA collapse under
   universal lying (+23..+36*) persists at every non-degenerate tariff — when every declaration
   is 'critical' there is no information left for any anonymous mechanism to recover. The
   incentive layer's claim is deterrence (honesty is a Nash-consistent best response), not
   robustness to a coordinated all-liar population.

**Honest read / caveats.** One deviation strategy (BE→critical; class-mix deviations like
"everything high" unprobed); 4 users × 4 jobs → deviator rates are coarse (quantised at 1/4),
CIs honest but wide; frozen-class committed auction from Exp 11 world (synthetic workload, not
trace replay); the budget is exogenous and equal per user — sizing it (or income r) IS the
operator's fairness policy, and the frontier table is exactly that policy's menu; no LLM tier —
the natural interpretability follow-up is an LLM user-agent that must *decide whether to lie*
given the tariff (does it discover honesty? can it explain why?).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.incentive_sim            # 32 seeds, pools {6,8,12}, tariff ladder
.venv/bin/python -m pins.incentive_sim --seeds 8 --budgets none,120 --pools 8   # quick
```
`make_user_budget_committed` / `assign_users` / `declare_for` in `pins/incentive_sim.py`;
per-seed data in `pins/results_incentive.json`; `simulate(return_jobs=True)` added to
`negotiation_sim.py` (default unchanged).

## Experiment 33 — THE LLM USER AGENT vs THE TARIFF: does a self-interested LLM discover honesty?

**Date:** 2026-07-07

**Why.** Exp 32 proved deterrence against *scripted* deviations. But the thesis's agents are
LLMs — so the behavioural question is whether an LLM user, told only to maximise its own jobs'
deadlines (never told to lie, never told to be honest), (a) exploits the trusting mechanism
unprompted, and (b) is flipped to honesty by the tariff — with the transcript showing *why*.
This is the interpretability claim meeting the incentive layer.

**Method.** New `llm_declare` in `llm_agent.py`: the agent plays the USER (not the admission
controller of `llm_priority`) — system prompt states the game (declarations trusted, higher
classes served first) and, in the priced variant, the exact Exp-32 tariff (class prices,
shared purse, portfolio-wide demotion on insolvency). Ctx buckets: tier × deadline × size ×
tariff (≤16 states, cached; code owns all magnitudes as always). `incentive_sim --llm`: user 0
declared by the agent, other 3 users truthful, budget none vs 120-lump, 32 seeds, pools
{6,8,12}. Reference points: user-0's truthful world and the scripted all-critical-BE liar,
same seeds. The rule fallback IS the rational agent (exploit when free / truthful when priced)
and validates the harness exactly (unpriced: −25 vs truthful ≡ liar; priced: ≡ truthful).

**Result (user-0's own violation rate, paired diffs ±95% CI; − = agent better).**

| | vs truthful, unpriced | vs liar, unpriced | vs truthful, priced |
|---|---|---|---|
| qwen2.5:3b | **−11.7* / −12.5* / −14.1*** | +12.5* / +10.2* / 0 | +2.3 / −4.7 / −2.3 (ns) |
| qwen2.5:14b | **−4.7* / −5.5* / −7.0*** | +19.5* / +17.2* / +7.0* | **+0.0 / +0.0 / +0.0** |

Declarations (the decision table): unpriced, 3b lifts every best-effort job (loose-large →
'high', tight-small → 'high'); 14b lifts only the tight ones. Priced, both shift down
(3b: prod-tight-small critical→high, BE-loose-small→low; 14b: BE-loose→low, prod stays
critical) — and 14b's outcomes become *exactly* the truthful world's at every pool.

**Findings.**
1. **The vulnerability is emergent, not hypothetical:** given only "maximise your own jobs,"
   both models exploit the unpriced mechanism significantly (3b −12..−14*, 14b −5..−7* vs
   honest play). Nobody has to teach an LLM user to over-claim.
2. **Scale is INVERSELY related to ruthlessness:** both leave gains on the table vs the
   scripted liar, 14b much more (+17..+20* behind the liar). Its unpriced justifications use
   normative language ("minimizing impact on others") — prior alignment, not payoff
   maximisation. The smaller model games harder.
3. **The Exp-32 tariff flips both to honesty-equivalence:** priced, the agent−truthful gap is
   ns at every pool for 3b and identically zero for 14b — deterrence holds against LLM agents,
   not just scripted deviations. 14b's priced justification cites the mechanism's logic
   explicitly ("low importance, minimizing cost impact on other critical jobs") — the tariff
   is *legible* to the agent, which is what makes the deterrence work through reasoning rather
   than trial and error.
4. **Decisions respond to incentives more reliably than explanations describe them:** 3b's
   justifications confabulate (claims "tight deadlines" for a loose-deadline state while
   declaring it 'high'); one 14b transcript degenerated into Chinese mid-sentence (class still
   valid). The class shifts track the tariff; the prose only tracks it at 14b — consistent
   with the Exp 12/23 scale pattern for justification quality.

**Honest read / caveats.** One-shot declaration per state (no in-context learning from
outcomes — the agent never SEES its purse drain; flipping via the prompt's rule description
alone is the strong version of the result, but an adaptive/multi-round variant is untested and
Exp 26 warns naive reflection may not converge); 16-state bucketing hides within-state
heterogeneity; temperature 0, one sample per state (no variance over drafts); the scripted
liar comparison at priced pools is ns everywhere (tariff neutralises the liar too — consistent
with Exp 32).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.incentive_sim --llm --model qwen2.5:3b    # 32 seeds, pools {6,8,12}
.venv/bin/python -m pins.incentive_sim --llm --model qwen2.5:14b
.venv/bin/python -m pins.incentive_sim --llm --no-llm              # rational-rule reference
```
`llm_declare`/`SYSTEM_DECLARE_*` in `pins/llm_agent.py`; `make_llm_declare0`/`llm_sweep` in
`pins/incentive_sim.py`; per-seed data + decision tables in
`pins/results_incentive_llm_qwen2.5_{3b,14b}.json`.


## Experiment 34 — Exp 1-7 REDONE on real architectures: gate still passes, but the 40x headline was a synthetic-benchmark artifact (2026-07-07)

**Question.** The whole Stage-1-static result (claims-table row 1) rested on 6 hand-written
synthetic CNNs + a toy ResNet/TinyLM. Does it survive on REAL models? Model list externally
justified: exactly the DNN families in the MIT Supercloud labelled trace
(`data/labelled_jobids_full.csv`) — resnet50/101/152, vgg11/16/19, inception3,
bert-base/distilbert (HF, eager attention — verified `_attn_implementation == "eager"` so the
seq^2 score matrix IS materialised, matching the Exp-7 analytic term), U-Net family
U{3,4,5}-{32,64,128}. GNNs skipped (PyG). 13 jobs, truth measured on the A100
(fp32, Adam, 6 steps), batch/res/seq controlled. `pins/eval/predict_real.py`, reusing the
Exp 6/7 machinery (leaf-module hook activation proxy + analytic attention term + global
LOOCV (a,b)).

**Result (fp32, n=13).**

| Predictor | mem MAE | MAPE | within 1.5x | rho |
|---|---|---|---|---|
| **Deterministic (LOOCV global)** | **1.30 GB** | **18.4%** | **84.6%** | **0.80** |
| Params heuristic | 1.66 GB | 24.0% | 69.2% | 0.40 |
| Mean (no prediction) | 1.87 GB | 34.0% | 69.2% | −0.10 |
| Raw LLM qwen2.5:3b | 572 GB | 9902% | 0% | 0.25 |
| Raw LLM qwen2.5:7b | 43 GB | 854% | 0% | −0.09 |
| Raw LLM qwen2.5:14b | 22 GB | 479% | 7.7% | −0.14 |

Beats-heuristic gate: **PASS** (1.30 vs 1.66 GB) — but the margin is 1.3x, not 40x.

**What replicated exactly — the Exp-1 negative.** Raw-LLM miscalibration is fully confirmed
on FAMOUS models: knowing the name "ResNet-101" does not rescue calibration (3b said
1792 GB for resnet101@bs32; truth 4.7). Same monotonic scale ladder as Exp 1
(572→43→22 GB MAE vs 283→77→24 on synthetic), same never-closing order-of-magnitude gap,
and ranking is destroyed at 7b/14b (rho ≤ 0). Claim-1's first clause is now trace-grounded.

**What did NOT replicate — the ~2%/40x headline.** On real architectures the deterministic
estimator degrades to MAPE 18.4%, and the heuristic is far less bad than on the synthetic set
(the real-model truths cluster at 3–12 GB, close to the heuristic's ~4 GB floor). The honest
claim after Exp 34: the deterministic route still wins every metric and is the only predictor
with usable ranking (rho 0.80 vs 0.40), but the 40x figure was an artifact of a synthetic
family designed to spread activation memory. Two localized blind spots explain most of the
residual, and both are the Exp-6 lesson recurring (structure the proxy can't see):
1. **inception3** (det 4.2 vs truth 6.7 GB): torchvision Inception uses FUNCTIONAL pooling
   (`F.max_pool2d` in `forward`) — invisible to leaf-MODULE hooks, so activations are
   undercounted. Fix is mechanical (also hook `torch.nn.functional` or count via autograd
   graph), not conceptual.
2. **bert-s512** (det 7.0 vs truth 12.0 GB): eager HF attention materialises more than the
   score matrix (post-softmax probs saved for backward, dropout masks, 4×hidden FF
   intermediates); one global `a` fit mostly on convnets under-credits transformer
   internals. A per-family `a` or an explicit probs term would close it — at the cost of
   more calibration data.
Per-family MAE: unet 0.47 · vgg 0.91 · resnet 1.47 · transformer 1.94 · inception 2.50 GB.

**Claims-table impact.** Row 1 weakened from "~2% (~40x)" to: raw-LLM incapability CONFIRMED
on real models; deterministic route beats heuristic/mean/LLM on all metrics on real
architectures (MAPE 18%, rho 0.80) with known, mechanically fixable blind spots. Updated in
the table.

**Honest read / caveats.** n=13, single precision (fp32), one batch-size choice per model
(truth range 2.9–12 GB is narrower than production jobs — a wider batch/seq sweep would
stretch the spread and likely favour the activation-aware route); U-Net "Supercloud" configs
are our reconstruction of the U{depth}-{filters} labels, not the original code; GNN families
skipped entirely.

**Reproduce.**
```bash
cd Research
.venv-forecast/bin/python -m pins.eval.predict_real            # needs the A100 + Ollama
.venv-forecast/bin/python -m pins.eval.predict_real --skip-llm # deterministic arms only
```
Per-job data in `pins/eval/results_real.json`.

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

**Why.** Exp 36's headline rested on the OPTIMISTIC truth definition its own honest-read
flagged: "true need = mean utilization" lets a job averaging 0.2% util be right-sized to one
quantum even if it keeps 10 GB resident the whole time. The promised one-flag counter-world:
true need = **peak GPU-memory residency**, in quarter-GPU quanta of the job's own card. If
the right-sizing win was an artifact of the optimistic truth, it should collapse here.

**Method.** `predict_gpu --target gpu_mem` now exports `pred_job_mem.csv` (same 209,336 test
jobs as the usage csv — identical key set, so mem-world windows sample the same job
population): per-instance quanta = `mem_GB / (card_GB/4)`, card memory from `gpu_type`
(V100M32 = 32 GB, T4/P100/V100/MISC = 16 GB), × inst_num, summed per job. Underlying GBT
byte-identical to Exp 35 (re-ran: MAE 1.60 GB, rho 0.628, gate PASS +8.5% — reproduced
exactly). `trace_replay --truth mem` reuses the whole Exp-36 path (`load_usage_quanta` with
a different csv); tiers land as `*@mem`, usage/plan worlds untouched. Same design: 32 seeds,
pools {4,6,8}, arms = request {declaration, Stage-1 P50, oracle}, tiers {rule, qwen2.5:3b}
(fb=0% everywhere).

**The wedge, remeasured under memory.** Raw mem-need is ~10× heavier than util-need (median
0.166 vs 0.015 quanta) — but after the floor-at-1 + CAP_CLIP the worlds nearly coincide:
**78.8% of jobs need ≤1 quantum under memory vs 82.7% under utilization** (declaration
median ≈ 4). Peak residency, not just average burn, is far below the ask for most jobs —
the over-provisioning premise is NOT an artifact of the optimistic truth.

**Result (paired by seed, n=32; * = 95% CI excludes 0).**

Declared − predicted (value of right-sizing; + = declaration worse), negotiated policy:

| pool | rule | qwen2.5:3b |
|---|---|---|
| 4 | ΔSLA **+10.9 ±5.9*** · Δutil +17.5* · Δslow +4.1* | ΔSLA **+15.2 ±5.9*** · Δutil +12.0* · Δslow +4.7* |
| 6 | ΔSLA **+6.6 ±5.4*** · Δutil +24.4* · Δslow +1.2* | ΔSLA **+11.5 ±5.3*** · Δutil +22.4* · Δslow +2.0* |
| 8 | ΔSLA +3.1 ±4.5 ns · Δutil +25.6* · Δslow +1.0* | ΔSLA **+7.2 ±4.4*** · Δutil +24.7* · Δslow +1.3* |

(Exp-36 usage-world reference, negotiated: rule +13.1*/+8.2*/+5.3*, 3b +17.0*/+14.1*/+9.2*.)

Predicted − oracle (cost of GBT error), negotiated: rule +3.5/+7.4/+7.2 SLA*
(prodSLA +8.6/+9.4/+12.2*); 3b +1.8ns/+4.5*/+5.1* (prodSLA +2.1ns/+9.9*/+9.7*).

Negotiated − floor WITHIN the predicted-request world:

| pool | rule | qwen2.5:3b |
|---|---|---|
| 4 | SLA +0.4 ns · prodSLA −3.6* | SLA **−3.7 ±3.6*** · prodSLA **−8.2*** · util +7.1* · slow −0.1 |
| 6 | SLA −0.4 ns · prodSLA −2.4 ns | SLA **−6.2*** · prodSLA **−4.1*** · util +9.0* · slow −0.4* |
| 8 | SLA −2.1* · prodSLA −3.1* | SLA **−5.9*** · prodSLA **−5.7*** · util +8.0* · slow −0.3* |

**Findings.**
1. **Right-sizing survives the pessimistic truth.** Declared−predicted stays positive and
   significant at every pool for 3b (+7.2..+15.2 SLA*) and at pools 4/6 for the rule
   (+6.6..+10.9*); the win attenuates ~2 pts vs the usage world and loses significance
   only at rule/pool-8. Exp 36's headline was not an artifact of the optimistic truth —
   reality, bracketed between the two worlds, keeps the win.
2. **Exp-36 finding 3 (negotiation + prediction are complements) replicates at every
   pool:** under predicted requests, negotiated@3b beats its floor on all four headline
   metrics — SLA −3.7*/−6.2*/−5.9*, prodSLA −8.2*/−4.1*/−5.7*, productive utilization
   +7..+9*, slowdown ≤0. The rule tier's advantage again mostly evaporates (SLA ns at 4/6) —
   the state-dependent LLM margins, not the protocol alone, carry the predicted-request world.
3. **Memory misprediction is the costliest error so far for the protected tier:**
   pred−oracle prodSLA +8.6..+12.2* at the rule tier (usage world: +3.6ns..+9.0*).
   Under-predicting residency starves a prod job outright rather than merely slowing it —
   better mem predictors (or a prod-p90 hedge on the mem interval, wired but unrun) have
   the largest headroom here.
4. **Declaration-as-accidental-hedge at slack:** decl−pred prodSLA turns negative-ns at
   pool 8 (rule −6.5, 3b −1.3) — with GPUs to spare, the inflated ask shields prod jobs from
   under-prediction, exactly the hedge/reserve substitution of Exp 31/36 seen from the other
   side. Right-sizing is a contention-regime play; at slack it trades prod insurance for
   freed capacity.

**Bugfix (2026-07-08, later the same day).** Numbers above are the CORRECTED 32-seed values
(window-reroll bug, see the note before Exp 38; original run had 5/32 seeds zeroed on
decl−pred and plan-world-contaminated on pred−oracle). Two claims moved: the complement
finding upgraded from "pools 6/8" to ALL pools (pool-4 SLA −3.3ns → −3.7*), and pred−oracle
shrank (was +5.1..+10.2* at 3b, now +1.8ns..+5.1*) while staying costliest-for-prod.

**Honest read / caveats.** Card memory for MISC (67% of GPU tasks) is assumed 16 GB — the
trace does not say; max-over-workers × inst_num over-counts heterogeneous gangs (the recipe
mirrors the usage world's mean×inst_num, biased the opposite, pessimistic direction);
residency truth still floors at 1 quantum and clips at 8; peak residency ≠ exclusive need
either (PAI ran GPU sharing — the true requirement lies between mean-util and peak-mem, so
the two worlds bracket it); mem-world tiers pair only with each other (gpu_mem test-job
windows); 14b unrun; quantile knob still unexplored outside the plan world.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_gpu --target gpu_mem       # + pred_job_mem.csv export
.venv/bin/python -m pins.trace_replay --seeds 32 --truth mem --caps real       # declaration
.venv/bin/python -m pins.trace_replay --seeds 32 --truth mem --caps predicted  # Stage-1 P50
.venv/bin/python -m pins.trace_replay --seeds 32 --truth mem --caps oracle     # matched control
# + the same three with --llm --model qwen2.5:3b
```
`GPU_MEM_GB` + mem export in `pins/eval/predict_gpu.py`; `MEM_CSV` + `--truth mem` in
`pins/trace_replay.py`; tiers `{rule,qwen2.5:3b}+{decl,pred,oracle}@mem` in
`pins/results_trace_replay.json`.

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

**Why.** The demand agent's stance driver — the behind/ontrack/ahead deadline bucket — has
been computed from TRUE remaining work since Exp 22 (`two_sided_sim` fed
`bridge.deadline_bucket(remaining(j), …)` the simulator's own state): an oracle time signal
inside every negotiation result to date. Meanwhile Exp 35's runtime GBT (the weakest
retargeted predictor: within-2x only 41%) was never wired into Stage-2. This experiment
swaps the belief: what is the oracle worth, and does the noisy prediction keep it?

**Method.** `predict_gpu --target runtime` exports `pred_job_runtime.csv` (178,815 test
jobs; job runtime = MAX over its tasks — PAI gangs launch together, replay dur is
max end − min start; Exp-35 metrics reproduced exactly). `trace_replay --time
predicted|blind|oracle` + `belief_work` in `simulate()`: the deadline bucket (and the
demand-table rank derived from it) now reads believed remaining = belief − progress-so-far,
where belief is the Stage-1 P50 in ticks ("predicted"), nothing (bucket forced "ontrack",
"blind"), or true work on the same windows ("oracle" = matched control; absent flag = old
path, byte-identical). Dynamics, SLA scoring, and deadlines themselves stay on truth. Plan
world, caps real — the time axis isolated from the request axis. Windows restricted to
runtime-covered jobs (151,883/606k; zero rerolls), 32 seeds, pools {4,6,8}, rule + 3b
(fb=0%). In sim units the belief is genuinely noisy: 52% within 2x of true work.

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
|  | 6 | ΔSLA −2.1 ±2.3 · ΔpSLA −1.8 | ΔSLA −0.6 ±1.6 · ΔpSLA +1.8 |
|  | 8 | ΔSLA −2.1 ±2.3 · ΔpSLA **−2.5 ±2.5*** | ΔSLA −0.2 ±0.7 · ΔpSLA +1.0 |

Negotiated − floor prodSLA within each time world: rule −1.9ns/−6.5*/−3.8ns (blind),
+0.5/−4.8/−1.3 ns (predicted), +0.2/−4.8/−1.9 ns (oracle);
**3b −8.5*/−7.5*/−7.6* (blind), −8.2*/−9.3*/−8.7* (predicted), −7.9*/−10.2*/−7.2* (oracle)**.

**Findings.**
1. **The oracle time signal was worth almost nothing — de-oracling is free.** At 3b every
   blind/predicted/oracle contrast is within ±2.6 pts and ns; at the rule tier the oracle
   is worth at most ~2 ns pts. The retroactive worry about Exp 22–37 (all carried the
   oracle deadline signal) is resolved: nothing load-bearing leaned on it.
2. **Predicted ≈ oracle at both tiers** (|ΔSLA| ≤ 0.8, one marginal +0.8* at rule pool 4):
   the coarse behind/ontrack/ahead bucket fully absorbs a predictor that is within 2x only
   half the time. This is the project's signal-interface pattern at its cleanest — the LLM
   consumes BUCKETS, and buckets forgive exactly the noise the runtime GBT has. A weak
   predictor is sufficient because the interface is coarse by design.
3. **The surprise: at the rule tier the time signal is mildly COUNTERPRODUCTIVE** — blind
   beats predicted (SLA −2.1* at pool 4, prodSLA −2.5* at pool 8) and directionally beats
   oracle everywhere. The fixed ladder over-reacts to "ahead" (hedging down releases GPUs
   that then feed best-effort competitors of its own prod jobs); forcing a moderate
   "ontrack" stance is better than reacting. The 3b LLM does NOT inherit this pathology
   (all ≈ 0) — its stance depends on more than the deadline axis.
4. **The negotiation's value is time-signal-independent:** negotiated@3b − floor prodSLA
   is −7..−10* in ALL three time worlds. The brake + judgment carry the result; the
   deadline belief — oracle, predicted, or absent — barely moves it.

**Honest read / caveats.** Belief enters ONLY the deadline bucket: the frozen bid priority
(`Job.bid`) still uses nominal need, and the supply agent still gets no time-to-free signal
from incumbents' predicted completions — that (predicted releases sizing the reserve) is
the genuinely new lever this opens, untested. P50 only (interval/newsvendor hedge in time
unexplored); job runtime = max task runtime is a lower bound of wall clock; the time worlds
pair only with each other (runtime-covered windows); deadlines/urgency remain synthetic
constructs of the world; rule-tier negativity is small and mostly ns — a curiosity to
re-test, not a claim.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_gpu --target runtime       # + pred_job_runtime.csv
.venv/bin/python -m pins.trace_replay --seeds 32 --time blind
.venv/bin/python -m pins.trace_replay --seeds 32 --time predicted
.venv/bin/python -m pins.trace_replay --seeds 32 --time oracle
# + the same three with --llm --model qwen2.5:3b
```
Runtime export in `pins/eval/predict_gpu.py`; `--time` + `load_runtime_pred` +
`belief_work` plumbing in `pins/trace_replay.py` and `pins/two_sided_sim.py`; tiers
`{rule,qwen2.5:3b}+time-{blind,predicted,oracle}` in `pins/results_trace_replay.json`.

## Experiment 39 — SUPPLY-SIDE TIME-TO-FREE: the last plausible slot for runtime prediction

**Date:** 2026-07-08

**Why.** Exp 38 closed the demand-side slot (deadline belief: oracle worth nothing). The
genuinely NEW thing runtime prediction could enable is on the supply side: the reserve
agent decides how much idle headroom to hold knowing only {contention, incoming-prod} —
it cannot see that a running job is about to release its GPUs. Imminent releases should
SUBSTITUTE for idle reserve ("a 2-GPU job frees up in ~2 ticks — don't hold GPUs idle,
the arrivals can ride the release"). Is that information worth anything?

**Method.** `reserve_ctx` gains an optional `release` bucket (none/some/wave = 0 / 1–2 /
≥3 held GPUs of running jobs with believed remaining work ≤ TTF_HORIZON=2 ticks);
`_rule_reserve` steps the reserve down one level on "some", zeroes it on "wave";
SYSTEM_RESERVE states the substitution principle; state keys extend only when the signal
is present (signal-off path byte-identical, verified). `trace_replay --ttf
predicted|oracle` (Stage-1 runtime P50 vs true realised remaining; remaining WORK proxies
remaining time). CONTROL = the Exp-38 `time-oracle` tiers — same runtime-covered windows,
no supply signal, so all arms stay seed-paired for free. 32 seeds, pools {4,6,8},
rule + 3b, fb=0%.

**Result (negotiated policy vs control, paired, n=32).**

| contrast | rule | qwen2.5:3b |
|---|---|---|
| ttf-oracle − control | ΔSLA −0.6/−1.0/−0.8 ns · ΔpSLA ns · Δutil +0.2..0.3* | **exactly 0 on every metric, every seed** |
| ttf-predicted − control | ΔSLA **−2.9*/−2.1*/−1.4ns** · ΔpSLA ns · Δutil +0.6..1.0* | **exactly 0** |
| ttf-predicted − ttf-oracle | ΔSLA −2.3*/−1.2ns/−0.6ns | exactly 0 |

**Findings.**
1. **A PERFECT time-to-free signal is worth ~nothing** (rule: −0.6..−1.0 SLA ns). The
   reserve lever it modulates is small by construction (0–2 GPUs, already gated off at
   scarce contention per Exp 14) — there is not enough conservatism to save. Combined
   with Exp 38: runtime prediction has now been tested in BOTH plausible Stage-2 slots
   (demand deadline belief, supply time-to-free) and both are near-zero-value. The
   negotiation is robust to time information — full stop. Runtime prediction's remaining
   role is the EASY-backfilling baseline (open item k), where it is the core fuel.
2. **The noisy prediction beats the oracle at the rule tier** (pred−oracle −2.3* at
   pool 4; pred−control −2.9*/−2.1*): the GBT's P50 under-predicts long right-skewed
   jobs, so "release imminent" fires EARLY and cuts the reserve more aggressively —
   accidental de-conservatism, not information (Exp 38 already showed this world punishes
   supply-side caution). An honest negative: the "value of the signal" here is really the
   value of holding less reserve, which a constant policy could capture without any
   predictor.
3. **3b is signal-blind on this axis: byte-identical decisions in ALL 27 release states**
   (hence exactly-zero diffs — the sims replayed identically). The justifications show it
   is not even parsing the field: a rel:wave state is justified with "there are no
   imminent GPU releases". Third rung of the legibility ladder — Exp 23 (3b bottleneck
   hints worse than none), Exp 33 (3b confabulates state), now a secondary supply axis
   invisible at 3b. The tariff (Exp 33) was legible where this is not: incentives pointed
   at the agent's OWN objective get parsed; contextual second-order signals do not.
4. Prod protection is untouched everywhere (ΔpSLA ns even with the reserve cut) — the
   margins, not the reserve, carry tier protection in this world, consistent with the
   Exp 31/36 hedge-reserve substitution.

**Honest read / caveats.** TTF_HORIZON=2 and the absolute none/some/wave thresholds are
untuned; reserve amounts are only 0/1/2 GPUs so the lever's ceiling is inherently low —
a redesign where the reserve is REPLACED by `max(0, need − upcoming)` (rather than
stepped down) could give the signal a bigger surface; remaining work proxies remaining
time (rate≈1 assumed); 14b unrun (the legibility question — does scale read the release
field? — is open and is the interesting follow-up); 3b ttf tiers are redundant with their
control by construction (kept in the json for the record).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.trace_replay --seeds 32 --ttf predicted   # + --ttf oracle
# + the same with --llm --model qwen2.5:3b; control = the time-oracle tiers
```
`release_bucket`/`reserve_ctx` in `pins/bridge.py`; prompt/rule/state-key in
`pins/llm_agent.py`; `TTF_HORIZON` + `ttf_work` in `pins/two_sided_sim.py`; `--ttf` in
`pins/trace_replay.py`; tiers `{rule,qwen2.5:3b}+ttf-{predicted,oracle}`.

## Experiment 40 — MODEL-FAMILY ABLATION: llama3:8b vs qwen2.5:{3b,14b} on paired windows

> **Upgraded to n=32 in Exp 43:** findings 1-3 all survive (3b's floor win extends to
> prodSLA at every pool; llama3 penalty shrinks to +7.0* but holds); the finding-4
> single-llm oddity was n=8 noise (+0.4 ns at n=32).

**Date:** 2026-07-09

**Why.** Every LLM tier so far is qwen2.5. Two confounds remained: (a) the 3b≥14b
"sufficiency" comparisons (Exp 27/29) were run at different times on different sampled
windows; (b) nothing says the negotiation's value survives a change of model FAMILY.
`--time predicted` (the de-oracled Exp-38 world) is the honest default going forward, so
the ablation ran there.

**Method.** `trace_replay --llm --time predicted` at llama3:8b and qwen2.5:14b, plus a
3b RERUN so all three tiers share the exact same windows (floor identical by seed:
60.9/45.3/41.4 mean SLA at pools 4/6/8). Old Exp-38 3b tier backed up to
`results_trace_replay.pre-3b-rerun.bak.json` (its floor was 67.4/53.5/47.3 — different
windows, which is exactly why the rerun was needed). 8 seeds, fb=0% everywhere.

**Result (negotiated policy, paired vs own floor, n=8; pools 4/6/8).**

| model | dSLA | dprodSLA |
|---|---|---|
| qwen2.5:3b | +5.5ns / −0.8ns / **−3.9*** | −0.4 / −8.5 / **−12.2*** |
| qwen2.5:14b | +5.5ns / +2.4ns / −1.6ns | −0.4 / +2.5 / −3.8 (all ns) |
| llama3:8b | **+14.1*** / +3.1ns / +0.8ns | −1.1 / +0.4 / −1.7 (all ns) |

**Findings.**
1. **3b is the only tier that significantly beats the floor anywhere** (pool 8: SLA −3.9*,
   prodSLA −12.2*). The 3b≥14b sufficiency claim now holds ON PAIRED WINDOWS in the
   predicted-time world — scale buys nothing here.
2. **llama3:8b actively hurts at contention** (negotiated +14.1* vs floor at pool 4;
   isolated worse still, +16.4*). Model FAMILY (instruction-following / format compliance
   in the protocol) matters more than parameter count: llama3:8b sits between the qwens
   in size and below both in outcome.
3. **The protocol cushions but cannot rescue a weak model**: negotiated < isolated for
   llama3 at every pool (e.g. 75.0% vs 77.3% at pool 4), consistent with the
   protocol-substitutes-for-scale arc — but the cushion stops short of the floor.
4. Oddity for the log: llama3 single-llm is best-in-pool at 8 (39.8%*), the reverse of
   the qwen ordering (Exp 24). n=8; unexplained, likely noise.

**Caveats.** n=8 (pool-4 CIs are wide); one non-qwen family only; llama3's decisions came
from fresh Ollama calls (cache was cold) — cost not compared.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --llm --model llama3:8b    --time predicted
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b  --time predicted
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:3b   --time predicted
```
Tiers `{qwen2.5:3b,qwen2.5:14b,llama3:8b}+time-predicted` in `results_trace_replay.json`.

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

**Why.** The 3b-vs-14b story rested on two legs of different strength: (a) Exp 29's
"statistical tie" is only *failure to find a difference* — an underpowered study would say
the same, so it cannot be claimed as equivalence; (b) Exp 40's "3b is the only tier to beat
the floor" in the predicted-time world was n=8. Before "the bounded protocol makes a small
model enough" becomes a thesis headline, close both: run TOST equivalence on the existing
32-seed data, and upgrade the predicted-time comparison to 32 paired windows.
**Margin pre-registered before the rerun: ±3 SLA pts** (below the smallest effect of
interest, the −4..−8 pt prodSLA protection); a stricter ±2 verdict reported alongside.

**Method.** (1) `t90()` + a `TOST:` line added to `--stats`/`--compare` in
`trace_replay.py`: the paired diff is *equivalent within ±m* iff its 90% CI lies inside
(−m,+m) (two one-sided tests at α=0.05); verdicts cross-checked against scipy. Existing
95%-CI output unchanged (Exp 29 numbers reproduce digit-for-digit). (2) 32-seed sweeps of
rule / qwen2.5:3b / qwen2.5:14b / llama3:8b at `--time predicted` on paired windows
(floor identical by seed: 67.4/53.5/47.3 SLA at pools 4/6/8); old n=8 tiers backed up to
`results_trace_replay.pre-exp43.bak.json`. fb=0% everywhere; `test_mechanism` green.

**Result A — TOST on the base world (Exp 29/addendum data, negotiated, n=32).**
Overall SLA: 3b≡14b within ±3 at pools 6/8 (pool 6 even ±2: 90%CI[−1.7,+0.1]); 3b≡7b
within ±3 at all pools; 14b≡rule within ±3 on BOTH metrics at all pools. Pool-4 3b-vs-14b
stays a real 14b edge (−2.9 ±2.8*, not equivalent) — the slack-regime exception stands.
prodSLA between LLM sizes: CIs ±3.2–4.7, too wide to certify ±3-equivalence at n=32 —
there the claim remains "no detectable difference", not "equivalent".

**Result B — predicted-time world, n=32 paired windows (negotiated vs own floor,
dSLA / dprodSLA; * = 95% CI excludes 0).**

| pool | rule | qwen2.5:3b | qwen2.5:14b | llama3:8b |
|---|---|---|---|---|
| 4 | **+3.3*** / +0.5 | +2.1 / **−8.2*** | +2.3 / −1.6 | **+7.0*** / **−5.6*** |
| 6 | +2.0 / −4.8 | −0.2 / **−9.3*** | +0.2 / −3.6 | +2.1 / −5.3 |
| 8 | +0.4 / −1.3 | **−3.1*** / **−8.7*** | −1.4 / −2.3 | −1.4 / **−5.2*** |

Head-to-head (paired): 3b−14b prodSLA **−6.6 ±6.3*** / −5.7 ±7.0 / **−6.4 ±5.8***
(3b better, same direction everywhere); SLA equivalent within ±3 at pools 4/6, 3b-lean at
8 (90%CI[−3.5,−0.0]). 14b−rule: −1..−2 pts, mostly ns — 14b still *concurs with the
ladder*. 3b−rule: pool-8 SLA **−3.5*** and prodSLA **−7.4***, pool-4 prodSLA **−8.8***.
llama3:8b−3b: SLA **+4.9*/+2.3*/+1.8*** — worse at every pool.

**Findings.**
1. **The sufficiency claim is upgraded from "tie" to "statistical equivalence"** (TOST,
   ±3 pts, pre-registered): on overall SLA at contention, 3b ≡ 7b ≡ 14b ≡ rule in the
   base world. The pool-4 14b edge survives as the known exception — the claim is never
   "smaller is better" there.
2. **In the honest predicted-time world, small is genuinely BETTER on prod protection.**
   Exp 40's n=8 hint survives n=32 and sharpens: negotiated@3b is the only arm beating
   the floor (pool-8 BOTH metrics; prodSLA −8..−9* at ALL pools), and beats 14b directly
   on prodSLA (2/3 pools*, same direction at the third). Mechanism, consistent with
   Exp 29 finding 5 + Exp 38: scale buys *conformity to the deterministic ladder*
   (14b−rule ≈ 0), the ladder's reactivity is mildly harmful under a noisy time signal
   (rule pool-4 +3.3* vs floor), and 3b's deviation from it is where the value lives.
3. **Family beats scale, now at n=32 and every pool**: llama3:8b (between the qwens in
   size) is significantly worse than 3b at all pools and significantly worse than the
   floor at pool 4 (+7.0*) — though the protocol still cushions it (negotiated +7.0* vs
   isolated +8.4*) and even llama buys some prod protection (−5.6*/−5.2* at 4/8).
   Exp 40's finding-4 oddity (llama single-llm best at pool 8) did NOT survive
   (+0.4 ±2.9 ns) — it was n=8 noise, as suspected.
4. Un-braked single-llm@14b pool 4 posts the biggest prodSLA number (−13.6*) but pays
   where the negotiated arm doesn't: util 84% vs floor 93%, done 13.0 vs 13.6 — the
   brake, not scale, is still what makes the protection cheap.

**Honest read / caveats.** One trace, one contention recipe, one prompt set; the 3b>14b
prodSLA stars are 2-of-3 pools with no multiple-comparison correction (the same-direction-
everywhere pattern is the safe part, per Exp 29 practice); prodSLA equivalence between
sizes is NOT certifiable at ±3 (CIs too wide at n=32 — would need ~4× the seeds). The
"3b beats 14b under prediction error" result is world-dependent: in the oracle-time base
world it is equivalence with a slack-regime 14b edge. Deferred robustness axes: size
ladder below 3b (where does sufficiency break?), a second non-qwen family, a second trace.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --stats        # TOST lines on base-world tiers
.venv/bin/python -m pins.trace_replay --seeds 32 --time predicted
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model qwen2.5:3b  --time predicted
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model qwen2.5:14b --time predicted
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model llama3:8b   --time predicted
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:3b+time-predicted/negotiated,qwen2.5:14b+time-predicted/negotiated"
```
Tiers `{rule,qwen2.5:3b,qwen2.5:14b,llama3:8b}+time-predicted` now n=32 in
`results_trace_replay.json`; TOST helpers `t90`/`tost_line` in `pins/trace_replay.py`.

## Experiment 44 — SECOND FAMILY: gemma2:9b joins the paired predicted-time windows (Exp-43 caveat, family leg)

**Date:** 2026-07-10

**Why.** Exp 43's "family matters more than scale" rested on ONE non-qwen family; with
n=1 a reviewer can read llama3:8b as "one broken model", or the whole negotiation value
as qwen-specific. gemma2:9b (same mid-size band) disambiguates. The sub-3b ladder
(qwen2.5:0.5b/1.5b, already pulled) is DEFERRED to a later session by user decision.

**Method.** Same recipe as Exp 43: `trace_replay --seeds 32 --llm --model gemma2:9b
--time predicted` on the identical paired windows (floor 67.4/53.5/47.3 by seed);
714 fresh decisions (fully cold cache, ~28 min); fb=0%.

**Result (negotiated, paired, n=32; dSLA / dprodSLA).**
vs own floor: pool 4 +3.1 ±3.3 / **−10.0 ±7.0***; pool 6 +2.1 ±2.4 / **−9.7 ±7.4***;
pool 8 −1.2 ±2.5 / **−7.2 ±6.5***.
gemma−3b: SLA **+2.3*/+2.0*** at pools 6/8 (3b better, ~2 pts), pool 4 EQ±3; prodSLA ns.
gemma−llama3: pool 4 SLA **−3.9*** (gemma better), pools 6/8 EQ±2; prodSLA −2..−4 ns.
gemma−14b: prodSLA **−8.4*** at pool 4 (gemma better; −6.1/−4.9 same direction, pool-6
90%CI excludes 0), SLA +2.0* at pool 6, else ns/EQ.

**Findings.**
1. **The negotiation's value is NOT qwen-specific.** gemma2:9b buys the prod-tier
   protection at every pool (−7..−10*, the 3b pattern, vs 14b's ns everywhere) and never
   significantly hurts overall SLA vs the floor. llama3:8b is now the OUTLIER: gemma
   beats it outright at pool 4 (−3.9*) where llama pays +7.0* vs floor.
2. **"Family/instruction-following matters" sharpens to a threshold, not a ranking**:
   two families clear the bar (qwen, gemma), one does not (llama3) — parameter count
   predicts none of this (8b < 9b < the working 3b).
3. **qwen2.5:3b keeps a small real edge over gemma** (+2 pts SLA at contention*) and
   stays the only model beating the floor on overall SLA (pool 8). The headline model
   remains 3b; gemma is the existence proof that the mechanism travels across families.
4. Exp-43 conditional resolved: gemma WORKS → the informative extra point is
   **gemma2:2b** (cross-family "small suffices"), not 27b (nothing to rescue). Deferred
   with the ladder.

**Caveats.** Still one trace/recipe/prompt set; all models at ollama default q4 quant;
prodSLA cross-model CIs remain wide (±3.5-7); n=2 working families, n=1 failing.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model gemma2:9b --time predicted
.venv/bin/python -m pins.trace_replay --compare "gemma2:9b+time-predicted/negotiated,qwen2.5:3b+time-predicted/negotiated"
```
Tier `gemma2:9b+time-predicted` (n=32) in `results_trace_replay.json`.

### Exp 44 addendum — gemma2:27b: the scale leg inside the second family (2026-07-10)

**Why.** Finding 4 above deferred 27b as "nothing to rescue" — but running it anyway
closes the scale axis WITHIN gemma: if 27b beat 9b, "family is a threshold" would need a
scale asterisk. (A base-world 27b run at n=8 existed first, tier `gemma2:27b`; it showed
the same vs-floor prodSLA pattern at −5.7..−7.7* but is not window-paired with Exp 43/44 —
superseded by this run.)

**Method.** Identical recipe: `trace_replay --seeds 32 --llm --model gemma2:27b --time
predicted`, same paired windows; fb=0%.

**Result (negotiated, paired, n=32; dSLA / dprodSLA).**
vs own floor: pool 4 +3.5 ±3.6 / **−8.1 ±6.6***; pool 6 +1.6 ±2.5 / **−6.9 ±6.6***;
pool 8 −2.0 ±2.7 / **−7.0 ±5.9***.
27b−9b: prodSLA **+2.8 ±2.7*** at pool 6 (27b WORSE), everything else ns (SLA within ±1).
27b−3b: SLA **+1.8*/+1.2*** at pools 6/8 (3b better), prodSLA ns everywhere.

**Findings.**
1. **3× scale inside the working family buys nothing**: 27b clears the threshold exactly
   like 9b (prod protection −7..−8* at every pool, SLA never significantly hurt) but never
   beats 9b — and is significantly worse once (pool-6 prodSLA). The threshold picture
   survives its strongest in-family scale test.
2. **qwen2.5:3b ≥ 27b**: the 2-pt SLA edge at contention that 3b held over 9b holds
   verbatim over 27b (+1.8*/+1.2* at pools 6/8). The headline model is unchanged.
3. Combined with Exp 40/43/44: scale is now null across THREE spans (qwen 3b→14b, gemma
   9b→27b, and cross-family 8b/9b vs 3b). What predicts success is family/instruction-
   following, and it is binary.

**Caveats.** Same as Exp 44 (one trace/recipe/prompt set, q4 quant); the pool-6 27b<9b
star is a single significant cell out of six — read as "no gain", not "scale hurts".

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model gemma2:27b --time predicted
.venv/bin/python -m pins.trace_replay --compare "gemma2:27b+time-predicted/negotiated,gemma2:9b+time-predicted/negotiated"
```
Tier `gemma2:27b+time-predicted` (n=32) in `results_trace_replay.json`.

## Experiment 45 — DYNAMIC CAP: telemetry-corrected allocation base (rule tier) (2026-07-13)

**Why.** Every world so far fixes a job's allocation base at admission (the request or a
Stage-1 prediction). Real elastic GPU jobs emit telemetry, so a GBT could re-estimate true
need *while the job runs* and let the system right-size continuously — the user's declared
request stays fixed; only the system's belief moves. Before wiring a real telemetry
predictor, this ablation bounds the value: after a job has RUN 3 ticks, its train-phase
base switches from the Stage-1 predicted request to (a) the true need (`--dyncap oracle`,
the telemetry upper bound) or (b) truth ±25% uniform (`--dyncap noisy`, a GBT-like read).
Exp-30 world (`--caps predicted`, plan truth); negotiation facts stay on the admission
request, so the margin layer is untouched — a clean cap-only lever.

**Method.** `dyn_cap_map`/`dyn_after` in `two_sided_sim.simulate` (falling cap rides the
existing voluntary-shrink path; rising cap makes the job a wanter again; `dyn_cap_map=None`
is byte-identical pre-Exp-45). `--dyncap {oracle,noisy}` in `trace_replay`; rule tier,
n=32 paired windows, pools 4/6/8.

**Result (paired, n=32; dSLA / dprodSLA, negative = dynamic better).**
dyn-oracle − static pred (negotiated): **−2.3* / −3.1* / −5.7*** SLA at pools 4/6/8
(prodSLA ns). Same compare on the FLOOR arm: **−3.1* / −4.3* / −7.2*** (pool-8 prodSLA
−5.6* too). dyn-noisy − static pred: −1.4 / −1.6 / **−3.5*** SLA. dyn-oracle − static
ORACLE-at-admission: dyn worse or equal (+1.8/+2.5/+0.4 SLA; pool-6 prodSLA +5.8*).

**Findings.**
1. **Dynamic right-sizing is a real, growing-with-slack SLA lever**: correcting the cap
   after 3 observed ticks recovers most of the prediction-error cost, biggest at pool 8
   (−5.7* negotiated, −7.2* floor) where wrong caps waste the most elastic room.
2. **It is mechanism-independent** — it helps the floor at least as much as the negotiated
   tier, so it's an ORTHOGONAL lever (a Stage-1.5 telemetry loop), not a negotiation
   interaction. Claims about the negotiation should not lean on it.
3. **A ±25% telemetry read keeps roughly half the win** (significant only at slack);
   telemetry quality matters more at contention.
4. **Admission-time oracle ≥ dynamic oracle**: the 3-tick correction delay has a real cost
   (pool-6 prodSLA +5.8*) — knowing need up front beats learning it, so dynamic caps
   COMPLEMENT better Stage-1 predictions rather than replacing them.

**Caveats.** Rule tier only (no LLM margins yet); truth is per-phase constant so this
measures error-correction, not within-phase demand drift (the sim has none); 3-tick delay
and ±25% noise are single design points; one trace.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --caps predicted --dyncap oracle --seeds 32
.venv/bin/python -m pins.trace_replay --caps predicted --dyncap noisy  --seeds 32
.venv/bin/python -m pins.trace_replay --compare "rule+pred+dyn-oracle/negotiated,rule+pred/negotiated"
```
Tiers `rule+pred+dyn-{oracle,noisy}` (n=32) in `results_trace_replay.json`.

### Exp 45 addendum — 3b margins × dynamic cap: both levers keep full value (2026-07-13)

**Why.** Exp 45 was rule-tier only; the proposed final stack is fixed request + GBT dynamic
cap + LLM-negotiated margin, so the open question was interaction (Exp-31-style hedge/cap
substitution was plausible). Same recipe at the headline model:
`--caps predicted --dyncap oracle --seeds 32 --llm --model qwen2.5:3b`.

**Result (paired, n=32; dSLA / dprodSLA).**
3b − rule, both dyn-oracle: prodSLA **−4.0* / −2.8* / −4.0*** at pools 4/6/8 (SLA −2.1* at
pool 8) — margins still pay after right-sizing. dyn − static, both 3b (negotiated): SLA
−2.1 / **−4.5* / −6.6*** — the cap lever survives real margins, same growing-with-slack
shape as the rule tier (−2.3*/−3.1*/−5.7*). Within the 3b dyn world, negotiated vs floor:
prodSLA **−7.1*** (pool 8), isolated −9.2*, single-llm −8.4*.

**Findings.**
1. **No interaction — the levers are COMPLEMENTS**: dynamic caps fix *how much* a job
   holds (SLA, all jobs), LLM margins fix *who is protected* (prodSLA); each keeps its
   full Exp-45/Exp-29 value in the other's presence. The proposed stack (fixed request +
   telemetry cap + negotiated margin) is validated end-to-end in sim.
2. Exp-45 finding 2 upgraded: "orthogonal lever" now holds under the headline model, not
   just the rule fallback.

**Noisy leg (same day).** `--dyncap noisy --llm qwen2.5:3b` n=32: noisy − static (3b,
negotiated) SLA −1.6 / **−3.9* / −5.3*** — nearly the full oracle win (−2.1/−4.5*/−6.6*),
and noisy ≡ oracle on SLA by TOST ±3 at ALL pools (prodSLA +2..+2.6 ns). 3b − rule in the
noisy world: prodSLA **−3.7* / −6.2*** (pools 6/8), SLA −2.9* (pool 8). Within-tier pool 8,
negotiated vs floor: SLA **−3.3*** AND prodSLA **−10.4*** — both metrics significant, the
strongest single cell of the dyn worlds.
3. **The margin layer cushions telemetry noise**: at the rule tier ±25% noise cost the cap
   lever its significance at pools 4/6; under 3b margins the noisy arm is TOST-equivalent
   to the oracle arm on SLA everywhere. A realistic GBT read + LLM hedge ≈ perfect
   telemetry — the complement works in the direction that matters for deployment.

**Caveats.** Exp 45's caveats (3-tick delay and ±25% noise single design points; per-phase
constant truth; one trace).

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --caps predicted --dyncap oracle --seeds 32 --llm --model qwen2.5:3b
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:3b+pred+dyn-oracle/negotiated,qwen2.5:3b+pred/negotiated"
```
Tier `qwen2.5:3b+pred+dyn-oracle` (n=32) in `results_trace_replay.json`.

## Experiment 46 — SUB-3B LADDER + gemma2:2b: where "small suffices" stops (2026-07-13)

**Why.** Exp 43/44 established sufficiency at 3b and a family threshold (qwen/gemma pass,
llama3 fails) but left the deferred queue: how far DOWN does sufficiency extend, and does
the second family pass at small scale too? qwen2.5:{1.5b,0.5b} + gemma2:2b, all
`--seeds 32 --time predicted` (the honest-default world), same paired windows.

**Result (paired, n=32; dSLA / dprodSLA, negotiated arm).**
vs own floor (pool 8): 1.5b **−2.7* / −5.6***; 0.5b +0.6 / −5.6 (ns), its single-llm arm
HURTS (+6.6* SLA); gemma2:2b **−2.9* / −8.3***.
vs 3b head-to-head: 1.5b SLA TOST-EQ±3 at ALL pools (±2 at 4/8) but pays prodSLA (+4.4*
pool 6); 0.5b SLA WORSE at all pools (**+2.5*/+3.5*/+3.7***); gemma2:2b EQ±3/±2 at pools
6/8, worse at slack (+3.3* pool 4).
vs deterministic rule ladder: 1.5b better (SLA −3.1*/−3.1* pools 6/8, prodSLA −5.9*/−4.3*
pools 4/8); 0.5b SLA ns everywhere (EQ at 6/8) — no SLA value over the rule — though it
still buys prod protection at slack (−12.1* pool 4); gemma2:2b better at contention
(pool 8 −3.3*/−7.0*).

**Findings.**
1. **The sufficiency floor sits between 0.5b and 1.5b**: 1.5b is SLA-equivalent to 3b,
   beats its own floor on both metrics at pool 8, and beats the deterministic ladder —
   the full 3b pattern. 0.5b breaks it: worse than 3b on SLA everywhere, no SLA value
   over the rule, and its single-llm arm actively damages the floor (+6.6*) — the first
   qwen tier to hurt.
2. **Cross-family "small suffices" confirmed at 2b**: gemma2:2b shows the 3b pattern at
   contention (floor-beating both metrics at pool 8, SLA-equivalent to 3b at pools 6/8).
   The family threshold of Exp 44 holds at the small end — it is not a big-gemma artifact.
3. **3b stays the headline**: it keeps a prodSLA edge over 1.5b (+4.4* pool 6) and an SLA
   edge over gemma2:2b at slack (+3.3* pool 4). The claim sharpens to: *sufficiency
   extends down to ~1.5-2b in both working families; below that the LLM stops paying for
   itself* — instruction-following quality, not scale, remains the predictor, but it has
   a size floor.
4. Exp-43/44 deferred queue CLOSED (ladder + gemma2:2b done); second trace is Exp 47.

**Caveats.** One trace/recipe/prompt set, q4 quant; 0.5b's prod-tier protection at slack
(−12.1*) says even the failing size extracts SOME signal — "fails" means "no longer beats
the deterministic ladder on SLA", not "emits garbage".

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --seeds 32 --llm --model qwen2.5:1.5b --time predicted
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:1.5b+time-predicted/negotiated,qwen2.5:3b+time-predicted/negotiated"
```
Tiers `qwen2.5:{1.5b,0.5b}+time-predicted`, `gemma2:2b+time-predicted` (n=32).

## Experiment 47 — SECOND TRACE (MIT Supercloud): the mechanism's applicability BOUNDARY (2026-07-13)

**Why.** Every replay number so far is one trace (Alibaba v2020). Exp 47 replays the MIT
Supercloud slurm log (`--trace supercloud`, commit 125e720): university HPC batch, whole
V100s, jobs ~14x longer (tick 900 s keeps median work ≈ 9 ticks — same sim regime,
different world). Base world only (request == truth, oracle time); Stage-1 CSVs are
v2020-keyed. Tiers `rule+supercloud`, `qwen2.5:3b+supercloud`, n=32, pools 4/6/8.

**Result (paired, n=32).** NOT a replication — a boundary. At pool 4 (1 whole-GPU slot)
every 3b arm LOSES SLA to its own floor (negotiated **+7.2 ±3.7***; floor runs 100% util);
pools 6/8 are a wash (negotiated +1.4/+2.1 ns; prodSLA −3.2/−2.4 ns, CIs ±7). 3b vs rule:
worse at pool 4 (+3.7*), TOST-EQ±3 at 6/8. Rule tier vs floor: ns everywhere.
**Slot-count control** (pools 16/24/32 = same jobs, 4-8 whole GPUs; json backed up —
`--pools` is not in the tier tag): ALL deltas collapse to ~0 (|dSLA| ≤ 0.6 ±1.1, ns) —
with enough whole-GPU slots and thinned load, the mechanism is INERT, not harmful.

**Findings.**
1. **The v2020 headline does not transfer to whole-GPU HPC batch**: hedging costs SLA when
   a margin/reserve eats a whole slot on a 1-2-slot pool, and does nothing when jobs fit.
   The mechanism's value requires demand FINER than the allocation quantum — v2020's
   quarter-GPU quanta gave margins sub-slot room; Supercloud's whole GPUs do not.
2. This turns the scope sentence added to `research_plan.md` (elastic fractional-GPU ML
   workloads; rigid whole-allocation batch out of scope) from a disclaimer into a
   **measured boundary** — the honest thesis form: "here is where it works, here is where
   it measurably stops".
3. prodSLA stays directionally protective on Supercloud (−2..−5, ns) — the priority
   serialisation survives; it's the margin/reserve HEDGING that has no room.

**Caveats.** Contended many-slot Supercloud regime untested (needs an n_jobs knob — at
pools 16-32 the thinned 16-job load undershoots contention); deadlines/urgency/tiers stay
synthetic (the make_workload recipe) on this trace too; base world only.

**Reproduce.**
```bash
.venv/bin/python data/build_supercloud_replay.py
.venv/bin/python -m pins.trace_replay --trace supercloud --seeds 32
.venv/bin/python -m pins.trace_replay --trace supercloud --seeds 32 --llm --model qwen2.5:3b
.venv/bin/python -m pins.trace_replay --compare "qwen2.5:3b+supercloud/negotiated,rule+supercloud/negotiated"
```
Tiers `rule+supercloud`, `qwen2.5:3b+supercloud` (n=32) in `results_trace_replay.json`.

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

**Why.** Before wiring a referee into the sim, measure the primitive: given statements and
a free pool, does the LLM's direct allocation respect the budget and the tier rules?
`pins/referee.py` (statements → referee → evaluator, scene-cached, rule-referee fallback)
+ `pins/referee_eval.py` (toy 3-job scenes + real v2020 scenes, 6 skewed jobs, pool factors
from surplus to shortfall).

**Result.**
- **Toy scenes:** 3b 0/3 feasible (budget-blind), 14b 2/3, 27b 2/3; a SELF-CHECK prompt
  line gets 14b to 3/3 but conservative. Every model computes `total_awarded` CORRECTLY,
  then fails to act on the ≤ comparison — **constraint enforcement, not arithmetic, is the
  failure**.
- **Real v2020 scenes:** feasibility collapses to **0% at exact/shortfall pools for ALL
  chat models** (even with self-check); prodcov 1.0 + overcommit 5–8 GPUs ⇒ **chat LLMs
  won't say no under scarcity** — they serve everyone and blow the budget.
- **deepseek-r1:32b flips it: 100% feasible at ALL pool factors incl. shortfall.** Not
  parroting the rule referee: 10/24 scenes differ while feasible (egalitarian partial
  coverage vs the rule's all-or-nothing; different victim choices, rationale stated).
  Needed `num_predict` 4096 (the thinking channel ate the 300 budget); ~1–3 min/call.

**Findings.** Feasibility under scarcity is a *reasoning-model* property, not a scale
property (27b chat fails where 32b-reasoning passes). The referee thesis is alive but only
above the reasoning threshold.

**Reproduce.**
```bash
.venv/bin/python -m pins.referee_eval --models rule,qwen2.5:3b,qwen2.5:14b,gemma2:27b
.venv/bin/python -m pins.referee_eval --models deepseek-r1:32b
```
Results in `pins/results_referee_eval.json`.

## Experiment 50 — REFEREE IN-SIM: trace replay with the LLM as the allocator (2026-07-15/16)

**Why.** Exp 49 measured scenes in isolation; the claim needs outcomes. `trace_replay
--referee` swaps the policy rows for no-llm / **referee** / negotiated (statements pinned
at 3b so referee-model ablations hold submissions fixed), honest fallback semantics
(infeasible tick → floor, counted). Also: the 2026-07-15 run died mid-deepseek (login-node
reaper); the tier save now **merges pools per tier** on resume instead of clobbering
finished ones (`trace_replay.py`), and reruns went one pool per background task.

**Result (v2020 base world, n=8, paired vs floor; SLA deltas in pp, lower better).**

| arm | pool 4 | pool 6 | pool 8 | fb |
|---|---|---|---|---|
| referee@3b        | −1.6 / −11.9 | +2.3 / +5.2 | 0.0 / −10.4 | 45–58% |
| referee@r1:32b    | +0.8 / −1.8  | **+0.0 ±3.0 / +0.0 ±0.0** | +0.8 ±7.1 / **−10.4 ±16.1** | **0%** |
| negotiated@r1:32b | +2.3 / −1.8  | +1.6 / +2.1 | +0.0 ±6.8 / −5.2 ±9.0 | 0% |

All deltas ns at n=8. r1:32b pool 6 is an **exact tie with the floor**: 6/8 seeds
outcome-identical; the two divergent seeds move exactly one job each (±6.25pp) and cancel.
At pool 6 the referee **held the floor while the negotiated arm slipped**. **Pool 8
(2026-07-16) is the first nominal outcome win:** referee prodSLA 50.1% vs floor 60.6%
(−10.4, ±16.1 ns) — double the negotiated arm's improvement (−5.2) on the same windows,
SLA flat, fb 0%. Slack pool + margins-only demand = exactly the venue the flexibility
argument predicts; ±16.1 at n=8 is the reason the seed sweep is next, aimed at pool 8.

**Transcript case study** (`pins/transcripts_seed23_pool6.txt`, regenerable from the LLM
cache via `pins/replay_transcripts.py`): the referee **won seed 3** by spending margins on
prod jobs (incl. a stated *partial* grant: asked 2, gave 1 + reserve) and **lost seed 2**
by hedging ahead-of-schedule besteffort jobs so the pool was empty when the prod job
arrived. Same supply request gets opposite rulings by cluster state ("no evidence of
incoming load" at an empty pool vs granted mid-window) — situational judgment is real but
**unaimed** under scarcity.

**Findings.**
1. **Exp 49's feasibility transfers perfectly to the sim** (r1:32b fb 0%) — failures are
   now judgment failures, not constraint failures.
2. **Sufficiency, not superiority (so far):** the reasoning referee replaces the guarantee
   layer without loss and adds auditable rationales; it does not yet beat the rule pipeline
   on averages. The 3b referee overcommits half its ticks and survives only via fallback.
3. The seed-2 failure mode is **promptable** (margin-priority under `incoming_prod=many` +
   tight pool), targetable without touching the seed-3 win.

**Caveats.** n=8 (CI ±3–7pp SLA, ±16pp prodSLA at pool 8); single trace, base world;
statements fixed at 3b; wall-clock ~1–3 min per uncached referee call at 32b (each r1:32b
pool ≈ 85–100 min on the login node, run one pool per background task).

**Next.** Pool 8 → n=32 on the best pool; **conditional hard-tick analysis** (split ticks
by "the rigid rule decided badly here" — the flexibility claim predicts the win lives
there); margin-priority prompt; report fallback_rate alongside outcomes always.

**Reproduce.**
```bash
.venv/bin/python -m pins.trace_replay --referee --llm --model qwen2.5:3b --seeds 8 --pools 4,6,8
.venv/bin/python -m pins.trace_replay --referee --llm --model deepseek-r1:32b --seeds 8 --pools 4   # then 6, 8 (one pool per task)
PYTHONPATH=. .venv/bin/python pins/replay_transcripts.py   # seed 2/3 transcripts from cache
```
Tiers `rule+referee`, `qwen2.5:3b+referee`, `deepseek-r1:32b+referee` in
`results_trace_replay.json`.
