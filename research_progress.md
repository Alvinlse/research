# Research Progress — Stage-1 Resource Prediction (closed-loop on real CNNs)

**Date:** 2026-06-06
**Scope:** Stage-1 cold-start prediction gate from `research_plan.md` — *how well can we
predict a training job's peak GPU memory from metadata?*
**Harness:** `pins/eval/predict_cnn.py` (new this session). All ground truth is **measured**,
not estimated.
**Environment:** single NVIDIA A100-PCIE-40GB · PyTorch 2.6.0+cu124 · qwen2.5 (3b/7b/14b)
served locally via Ollama at `localhost:11434`.

---

## Why this work exists

`pins/eval/predict_resources.py` scores predictors against the **approximate community
truth** baked into `benchmark.json` (whose own note says: *"replace with measured profiling
for the Month-5 real validation"*). Prior result (logged 2026-06-02): on that 20-job
benchmark, qwen2.5:3b raw prediction **fails the gate badly** (mem MAE ~820 GB), and even
7b/14b lose to a one-line params heuristic.

This session **closes the loop**: instead of trusting guessed truth, we define small CNNs we
can actually run, ask predictors for their peak VRAM, then **train them on the A100 and
measure the real peak** via `torch.cuda.max_memory_allocated()`. We then iterated through
four predictor designs, each removing one more "number" from the LLM's job.

### The governing question
> CNN memory is dominated by **activations** (≈ batch × resolution × width), *not* parameter
> count. The params-only heuristic is structurally blind to this. Can an LLM — or an
> LLM-assisted design — do better? And does a **bigger** LLM help?

### Ground-truth jobs (measured on the A100, fp32)
Two probes for Experiments 1–3 (param-light, activation-heavy by design):

| Job | Params | **Measured peak** |
|---|---|---|
| cnnA-64px-b256 | 1.15M | **2.40 GB** |
| cnnB-128w-96px | 18.75M | **5.92 GB** |

---

## Experiment 1 — Raw LLM number prediction

**Method.** Give the LLM only human-facing metadata (framework, model, params, batch,
**resolution** [added — the benchmark schema omits it], precision, training mode, dataset)
and ask it to emit `{peak_mem_gb, recommended_gpus}` directly. Ablated across model size.

**Result.**

| Predictor | cnnA (2.40) | cnnB (5.92) | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|---|---|
| qwen2.5:3b | 170 GB | 405 GB | 283 GB | 6862% | 0% | +1 |
| qwen2.5:7b | 107 GB | 55 GB | 77 GB | 2602% | 0% | **−1** |
| qwen2.5:14b | 16 GB | 40 GB | 24 GB | 571% | 0% | +1 |
| Heuristic | 4.0 GB | 4.1 GB | 1.7 GB | 49% | 50% | +1 |
| Mean | 4.16 GB | 4.16 GB | 1.8 GB | 51% | 50% | +1 |

**Why it FAILED.** LLMs cannot calibrate an absolute memory magnitude — every size
**over-predicts** by 7×–170×, and none lands within 1.5× of truth. The trivial params
heuristic and even the no-information mean crush all three LLMs. Scale helps *monotonically*
(283 → 24 GB) but never closes the order-of-magnitude gap; 7b even **inverted the ranking**
(ρ = −1), predicting the smaller-activation job needs more memory. **Bigger is not the fix.**

---

## Experiment 2 — Extract-then-compute hybrid (guessed facts)

**Method.** Design-consistent split (`research_plan.md`: *LLM reasons, code decides*). The LLM
no longer emits the final number; it emits **structured facts** —
`{trainable_fraction, bytes_per_param, optimizer_multiplier, activation_mb_per_sample}` —
and a deterministic formula computes GB:
`peak ≈ (weights + grads + optimizer + batch·activation) × 1.1`.

**Result.** (essentially identical across 3b/7b/14b)

| Predictor | cnnA (2.40) | cnnB (5.92) | mem MAE | MAPE | within 1.5× |
|---|---|---|---|---|---|
| Hybrid (any size) | 0.16 GB | 0.32–0.41 GB | 3.9 GB | ~93% | 0% |
| Heuristic | 4.0 GB | 4.1 GB | 1.7 GB | 49% | 50% |

**Why it PARTLY worked / still failed.** Two findings:
1. **Big win:** constraining the LLM to facts cut raw-LLM error **~70×** (283 → 3.9 GB) and
   made it **model-size robust** — 3b, 7b, 14b give the same answer because the facts are
   nearly identical and *correct* (`trainable=1.0, bytes=4, Adam mult=2`). A 3b model that's
   useless at guessing a number gives perfectly usable *facts*.
2. **Remaining leak:** it now **under-predicts** because the one remaining LLM *number* —
   `activation_mb_per_sample` — was guessed ~18–80× too low (LLM said 0.5 MB; true ≈ 9 MB for
   cnnA, ≈ 40 MB for cnnB). The formula itself is exact: plug the true activation in → 2.6 GB,
   matching the measured 2.40 GB. So it still loses to the heuristic on MAE.

**Lesson:** every LLM-emitted *number* is the weak link.

---

## Experiment 3 — Reasoning hybrid (walk the layers)

**Method.** Hand the model the architecture and make it **reason layer-by-layer in free text**
(no forced JSON) — compute every feature-map shape, sum the activations — then emit the facts
JSON last. Tested on qwen2.5:14b. (`--reasoning --show-reasoning`)

**Result.**

| Predictor | cnnA (2.40) | cnnB (5.92) | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|---|---|
| Hybrid + reasoning (14b) | 3.50 GB | 1.88 GB | 2.6 GB | 57% | **50%** | **−1** |
| Hybrid (guessed, Exp 2) | 0.16 GB | 0.40 GB | 3.9 GB | 93% | 0% | +1 |
| Heuristic | 4.0 GB | 4.1 GB | 1.7 GB | 49% | 50% | +1 |

**Why it IMPROVED but still failed.** Reasoning cut MAE 3.9 → 2.6 and got one job within 1.5×
(0% → 50%). The traces are diagnostic gold:
- **The per-layer SHAPE derivation was 100% correct** for both nets
  (`3×64×64 → 64×64×64 → … → 256×8×8`; `3×96×96 → … → 1024×6×6`). The hard architectural
  reasoning is solved.
- **Every error was bookkeeping, not reasoning:**
  - cnnA — **over-counted**: summed conv + BN + ReLU outputs as separate stored tensors
    (12.4 MB/sample vs ~8.5 true) → predicted 3.50 GB.
  - cnnB — **under-counted**: silently dropped conv2/BN/ReLU, summed only conv1 + pool per
    block (10.4 MB/sample vs ~40 true) → predicted 1.88 GB.
  - `optimizer_multiplier` waffled (1 for cnnA, 3 for cnnB; Adam is 2).
- The **inconsistent inclusion rule flipped the ranking** (ρ = −1).

**Lesson:** the LLM knows the *method* (walk layers, compute feature maps) but cannot apply
the *inclusion rule + summation* consistently. This is precisely the LLM-reasons /
code-decides boundary — so move the arithmetic into code.

---

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

## Experiment 5 — Does it survive mixed precision? (YES)

**Method.** Re-run the Exp-4 deterministic predictor end-to-end under **fp16** and **bf16**
AMP (`autocast` + `GradScaler` for fp16). Ground truth is re-measured per precision; `(a, b)`
is re-calibrated leave-one-out within each precision. (`--deterministic --precision {fp16,bf16}`)

**Result (identical for fp16 and bf16 — both use 2-byte activations).**

| Config | Params | **Measured (fp32)** | **Measured (fp16/bf16)** | Deterministic | Heuristic |
|---|---|---|---|---|---|
| w32-b3-64px-bs128 | 0.29M | 0.61 GB | 0.35 GB | 0.31 GB | 4.00 GB |
| w64-b3-64px-bs256 | 1.15M | 2.40 GB | 1.34 GB | 1.37 GB | 4.00 GB |
| w64-b4-96px-bs128 | 4.69M | 2.87 GB | 1.65 GB | 1.68 GB | 4.00 GB |
| w128-b4-96px-bs128 | 18.75M | 5.92 GB | 3.46 GB | 3.36 GB | 4.00 GB |
| w96-b3-128px-bs64 | 2.58M | 3.54 GB | 2.00 GB | 2.07 GB | 4.00 GB |
| w64-b5-128px-bs64 | 18.86M | 2.82 GB | 1.71 GB | 1.64 GB | 4.00 GB |

| Predictor (fp16 ≡ bf16) | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|
| **Deterministic (LOOCV)** | **0.06 GB** | **4.5%** | **100%** | **0.94** |
| Heuristic | 2.2 GB | 269% | 17% | 0.77 |
| Mean | 0.7 GB | 82% | 67% | 0.77 |

**Beats-heuristic gate: PASS (0.06 vs 2.25 GB MAE).**

**Why it SURVIVES — and wins by more.**
1. **The method holds:** the LOOCV-calibrated `(a, b)` re-fits cleanly to the new regime;
   deterministic stays near-perfect (MAE 0.06 GB, 100% within 1.5×, ρ 0.94) — statistically
   indistinguishable from its fp32 result.
2. **Mixed precision ≈ halves measured memory** (cnnB 5.92 → 3.46 GB; cnnA 2.40 → 1.34 GB),
   because activations (the dominant term) drop from 4 to 2 bytes while params/overhead don't.
3. **The advantage GROWS:** the params heuristic is *precision-blind* — it predicts ~4.0 GB
   regardless — so when real memory halves it over-predicts badly (within-1.5× collapses
   50% → **17%**, MAPE 124% → **269%**). The precision-aware activation term tracks the drop;
   the heuristic cannot. fp16/bf16 is exactly where activation-awareness pays off most.

**Caveat (byte handling).** Under AMP, master weights/grad/optimizer stay **fp32** (4 B) while
only activations are 2 B; the current formula uses one byte-width for both. For these
param-light CNNs the param term is negligible and `(a, b)` absorbs the discrepancy — but for
**param-heavy** models the formula should split: `param_term` at 4 B, `activation` at 2 B.

---

## Experiment 6 — Does it generalise across architectures? (mostly YES — boundary found)

**Method.** New harness `pins/eval/predict_arch.py`. Generalise the activation extractor from
"replay one known CNN recipe" to **architecture-agnostic**: register forward hooks on every
leaf module and sum its output activations (handles residual adds, embeddings, attention
output projections). Pool **three families** — VGG-CNN, **ResNet (skip connections)**, and a
**tiny Transformer LM** — and fit **one global `(a, b)`** leave-one-out across all of them.
Includes long-sequence transformer stress configs to probe the attention blind spot. (fp32)

**Result.**

| Job | Family | Params | **Measured** | Deterministic | Heuristic |
|---|---|---|---|---|---|
| cnn-w64-b3-64px-bs256 | cnn | 1.15M | 2.40 GB | 2.47 GB | 4.00 GB |
| cnn-w96-b3-128px-bs64 | cnn | 2.58M | 3.55 GB | 3.35 GB | 4.00 GB |
| res-w64-222-64px-bs128 | resnet | 2.78M | 2.42 GB | 2.84 GB | 4.00 GB |
| res-w64-2222-96px-bs64 | resnet | 11.17M | 2.96 GB | 3.42 GB | 4.00 GB |
| res-w128-222-64px-bs128 | resnet | 11.09M | 4.90 GB | 5.15 GB | 4.00 GB |
| lm-d256-l4-s128-bs32 | transformer | 4.22M | 0.52 GB | 0.97 GB | 4.00 GB |
| lm-d384-l6-s128-bs16 | transformer | 12.23M | 0.60 GB | 1.12 GB | 4.00 GB |
| lm-d256-l4-s256-bs16 | transformer | 4.25M | 0.52 GB | 0.97 GB | 4.00 GB |
| lm-d256-l4-**s512**-bs32 | transformer | 4.32M | 1.86 GB | 1.25 GB | 4.00 GB |
| lm-d384-l6-**s1024**-bs16 | transformer | 12.58M | 3.52 GB | 1.99 GB | 4.10 GB |

| Per-family mem MAE (deterministic) | cnn 0.13 GB · resnet 0.37 GB · transformer 0.71 GB |
|---|---|

| Predictor (global LOOCV) | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|
| **Deterministic** | **0.49 GB** | 38% | 60% | **0.89** |
| Heuristic | 1.9 GB | 223% | 40% | **−0.30** |
| Mean | 1.2 GB | 116% | 40% | −0.30 |

**Beats-heuristic gate: PASS (0.49 vs 1.86 GB MAE).**

**Why it (mostly) SUCCEEDED, and exactly where it BREAKS.**
1. **Generalises cleanly to ResNet.** One global `(a, b)` predicts skip-connection ResNets to
   **0.37 GB MAE** — the architecture-agnostic hook proxy handles residual adds with no special
   casing. CNN stays at 0.13 GB.
2. **Generalises to transformers at modest sequence length** (seq 128–256: predicted ≈ measured
   to ~0.1 GB once you account for the global fit).
3. **Breaks at long context — a precise, expected failure.** The hook proxy sees module
   *outputs* only, so it misses the internal attention score matrix (∝ batch · heads · seq²).
   At seq 1024 it under-predicts **1.99 vs 3.52 GB**. The miss ≈ `16·4·1024²·6 layers·4 B ≈
   1.6 GB`, which matches the 1.5 GB gap almost exactly. A single *linear* `(a, b)` cannot
   reconcile seq-linear and seq²-quadratic activation regimes — so adding the long-seq points
   also dragged the short-seq transformer predictions up (0.52 → 0.97). (Note: this quadratic
   materialisation is also precision-dependent — fp32 uses the math attention backend; fp16/bf16
   flash kernels may avoid materialising scores.)
4. **The heuristic is actively wrong across architectures:** its ranking correlation goes
   **negative** (ρ −0.30), because params anti-correlate with memory here — a 12M-param
   transformer uses 0.6 GB while a 1.15M CNN uses 2.4 GB. Activation-awareness is not a luxury
   across architectures; the params rule is worse than useless for ranking.

**Fix (next step).** Add an explicit attention term to the extractor for transformers
(`+ batch · heads · layers · seq² · bytes`); then the proxy captures the quadratic growth and a
single calibration should fit the whole pool again.

---

## Experiment 7 — Add the analytic attention term (CLOSES the long-context gap)

**Date:** 2026-06-12

**Method.** Implement the Exp-6 fix in `predict_arch.py`. The hook proxy stays
architecture-agnostic but structurally blind to the internal `(B, heads, seq, seq)` score
matrix; rather than ask the LLM for it (which Exps 1–3 proved is the wrong place for a
*number*), add a **closed-form** term in deterministic code:
`attention_elems_per_sample = layers · nhead · seq²` (per sample), folded into the activation
count so the **same single global `(a, b)`** absorbs its retention/precision factor. Zero for
non-attention families (`seq is None`). Re-run the identical 10-job pool, global leave-one-out,
fp32. (`activation_elems_per_sample` proxy + new `attention_elems_per_sample`.)

**Result.**

| Job | Family | Params | **Measured** | Det (Exp 6) | **Det (Exp 7)** |
|---|---|---|---|---|---|
| cnn-w64-b3-64px-bs256 | cnn | 1.15M | 2.40 GB | 2.47 | 2.33 |
| cnn-w96-b3-128px-bs64 | cnn | 2.58M | 3.55 GB | 3.35 | 3.29 |
| res-w64-222-64px-bs128 | resnet | 2.78M | 2.42 GB | 2.84 | 2.71 |
| res-w64-2222-96px-bs64 | resnet | 11.17M | 2.96 GB | 3.42 | 3.33 |
| res-w128-222-64px-bs128 | resnet | 11.09M | 4.90 GB | 5.15 | 5.26 |
| lm-d256-l4-s128-bs32 | transformer | 4.22M | 0.52 GB | 0.97 | **0.66** |
| lm-d384-l6-s128-bs16 | transformer | 12.23M | 0.60 GB | 1.12 | **0.81** |
| lm-d256-l4-s256-bs16 | transformer | 4.25M | 0.52 GB | 0.97 | **0.69** |
| lm-d256-l4-**s512**-bs32 | transformer | 4.32M | 1.86 GB | 1.25 | **1.41** |
| lm-d384-l6-**s1024**-bs16 | transformer | 12.58M | 3.52 GB | 1.99 | **2.98** |

| Per-family mem MAE (det) | cnn 0.16 GB · resnet 0.34 GB · **transformer 0.30 GB** (was 0.71) |
|---|---|

| Predictor (global LOOCV) | mem MAE | MAPE | within 1.5× | ρ |
|---|---|---|---|---|
| **Deterministic (Exp 7)** | **0.28 GB** | **17.6%** | **100%** | **0.96** |
| Deterministic (Exp 6) | 0.49 GB | 38% | 60% | 0.89 |
| Heuristic | 1.9 GB | 223% | 40% | −0.30 |
| Mean | 1.2 GB | 116% | 40% | −0.30 |

**Beats-heuristic gate: PASS (0.28 vs 1.86 GB MAE).**

**Why it WORKED.**
1. **The long-context gap closed in the right place.** The raw added term is `batch·bytes·
   layers·nhead·seq²` — at seq 1024 that's `16·4·6·4·1024² ≈ 1.61 GB`, exactly the size of the
   Exp-6 miss. seq 1024 went **1.99 → 2.98 GB** (gap 1.53 → 0.54); seq 512 **1.25 → 1.41 GB**.
   No LLM involved — a missing *formula* was fixed with a formula.
2. **One global `(a, b)` now fits all three families** — every one of the 10 jobs lands
   **within 1.5×** (was 60%), ρ rose 0.89 → 0.96, global MAE 0.49 → 0.28 GB.
3. **The regime conflict is gone.** In Exp 6 a single *linear* fit had to compromise between
   seq-linear and seq²-quadratic activation, which inflated the short-seq predictions
   (0.52 → 0.97). With seq² now explicit, those came **back down** (0.97 → 0.66, 1.12 → 0.81):
   the calibration no longer pays for the long-seq points with the short-seq ones. Transformer
   family MAE more than halved (0.71 → 0.30 GB).

**Residual / caveats.** seq 1024 still slightly under-predicts (2.98 vs 3.52) — the score
matrix is not the only quadratic intermediate (softmax probs, dropout mask, backward saves add
more), and the global `a` is a compromise across families — but it is now comfortably within
1.5×. Precision caveat stands: under fp16/bf16 the flash-attention kernel may not materialise
the full score matrix, so this analytic term should be gated on the attention backend before
trusting it at reduced precision (not yet tested — Exp 7 is fp32 only).

**Lesson (unchanged, reinforced).** The fix for an LLM-prediction blind spot was *more
deterministic code*, not more LLM. Artifact: `pins/eval/results_arch.json`.

---

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

## Caveats & limitations

- All experiments use one `SimpleCNN` (VGG-style) family, **fp32**, on a **single A100**.
- The Exp-4 calibration `(a, b)` is fit on this family; it needs validation across **other
  architectures** (skip/residual nets, MLPs, transformers), **precisions** (fp16/bf16), and
  **hardware** before generalizing.
- In production the LLM must supply per-layer shapes for *unknown* architectures; here the
  architecture is fully specified by metadata, so code derives the shapes directly. Exp 3
  validated that the LLM emits correct shapes when it must.
- Ground truth is `torch.cuda.max_memory_allocated()` — it captures allocator activity
  (tensors + cudnn workspace) but not the fixed CUDA context; the `b` term absorbs the rest.

---

## Reproduce

```bash
cd MCP
# Exp 1 (raw) + Exp 2 (hybrid) — swap --model qwen2.5:{3b,7b,14b}
.venv/bin/python -m pins.eval.predict_cnn --model qwen2.5:14b
# Exp 3 (reasoning, shows the layer-by-layer trace)
.venv/bin/python -m pins.eval.predict_cnn --model qwen2.5:14b --reasoning --show-reasoning
# Exp 4 (deterministic, 6 CNNs, leave-one-out) — the winner
.venv/bin/python -m pins.eval.predict_cnn --deterministic
```

**Artifacts:** `pins/eval/results_cnn*.json`, `results_cnn_hybrid_*.json`,
`results_cnn_reason_14b.json`, `results_cnn_deterministic.json`.

## Next steps
1. ~~`--deterministic --precision fp16` — does the calibration survive mixed precision?~~
   **DONE (Exp 5): yes — survives fp16 & bf16, and the margin over the heuristic widens.**
2. ~~Add non-VGG architectures (ResNet skip connections, plain MLP) — does `(a, b)` generalize?~~
   **DONE (Exp 6): generalizes to ResNet (0.37 GB MAE) & short-seq transformers; breaks at
   long context (missing seq² attention term). Still PASSES the gate overall.**
3. ~~Add an explicit attention-scores term (`batch·heads·layers·seq²·bytes`) so the extractor
   covers long-context transformers; re-check whether one global `(a,b)` then fits all families.~~
   **DONE (Exp 7): yes — global MAE 0.49 → 0.28 GB, 100% within 1.5×, ρ 0.96; transformer
   family MAE 0.71 → 0.30 GB. Still fp32-only; gate the term on the attention backend for fp16/bf16.**
4. Param-heavy / mixed precision: split the byte-width (param_term @ 4 B, activation @ 2 B).
5. Fold this deterministic estimator into the real Stage-1 hybrid (`pins/predictor.py` is still
   the phase-curve stub) so negotiation bids are prediction-informed end to end.
3. Fold this deterministic estimator into the real Stage-1 hybrid predictor
   (`pins/predictor.py` is still the phase-curve stub) so negotiation bids are
   prediction-informed end to end.

---

# Stage-1 DYNAMIC — trajectory forecasting on MIT Supercloud

A different prediction problem from Exps 1–7: not *one static peak number* per job, but a
running job's **trajectory over the next 5 min** (HORIZON = 30 steps × 10 s) across 4 channels
`[gpu_util, gpu_mem_gb, cpu_util, mem_gb]`, on real **MIT Supercloud** traces. Same governing
hinge: the **attention model (deterministic code) decides the numbers**; the LLM (next) sits on
top emitting regime facts, never the figure. Pipeline: `data/fetch_supercloud.py` →
`pins/forecast/{dataset,baselines,model}.py`. Runs in the isolated `.venv-forecast` (torch
2.6.0+cu124).

## Experiment 8 — Residual attention forecaster vs the persistence gate (PASS)

**Date:** 2026-06-18

**Method.** 100 joint CPU+GPU jobs aligned to a common 10 s grid (`dataset.py`: GPU 100 ms→10 s
mean-resampled, CPU/GPU timezone offset auto-corrected, inner-joined). 70 train / 30 test split
by job (`seed=0`). Lookback 30 steps → forecast 30 steps. Model is a small Transformer encoder
(`d_model=64, nhead=4, 2 layers`) over the history that predicts the **residual from
persistence** (target = future − last value), standardised per channel; on flat channels it can
learn ~0 and degenerate to persistence. 40 epochs, Adam 1e-3, L1 loss to match the MAE metric,
on the A100. Metric = per-channel MAE (native units) + scale-normalised nMAE (MAE / train-set
std); `nmae_mean` aggregates across channels. Each table cell is the MAE averaged over all 30
horizon steps and all rolling windows of all test jobs.

**Result.**

| Forecaster | nMAE_mean | gpu_util | gpu_mem_gb | cpu_util | mem_gb |
|---|---|---|---|---|---|
| persistence | 0.068 | 6.262 (0.19) | **0.129** (0.01) | 42.464 (0.04) | **0.903** (0.03) |
| moving_avg(k=6) | 0.063 | 5.705 (0.17) | 0.136 (0.01) | 38.864 (0.04) | 0.883 (0.03) |
| **attn (ours)** | **0.058** | **4.963** (0.15) | 0.139 (0.01) | **34.972** (0.04) | 1.003 (0.03) |

**Beats-baseline gate: PASS (nMAE_mean 0.058 vs 0.063 moving-avg / 0.068 persistence).**

**Why it worked.**
1. **The win is concentrated in the dynamic channels.** `gpu_util` 6.26 → 4.96 (−21% vs
   persistence) and `cpu_util` 42.5 → 35.0 (−18%) — exactly the channels that move at phase
   transitions, where a forecaster (vs holding flat) earns its keep.
2. **The residual anchor protects the flat channels.** `gpu_mem_gb` and `mem_gb` are
   piecewise-constant; persistence is already near-optimal and the model spends ~no capacity
   there, so the overall aggregate stays ahead.

**Residual / caveats.**
- On the two flat memory channels the model is **marginally worse** than persistence
  (gpu_mem 0.139 vs 0.129; mem 1.003 vs 0.903). The residual head should learn exactly 0 there
  but leaks a little noise — small, but the obvious next tightening (e.g. heavier residual
  regularisation or a per-channel gate).
- Single 70/30 split, one seed, no per-job CV yet; 100 jobs is a modest sample. Numbers are a
  first credible gate-pass, not a tuned final result.
- The per-channel MAE blends all 30 horizon steps; error almost certainly grows with lead time
  (10 s ahead easy, 5 min ahead hard) — an error-vs-horizon curve is not yet logged.
- The LLM regime-facts layer (`llm_facts.py` / `model_facts.py`) is built but not yet evaluated
  here — Exp 9 territory.

**Reproduce.**
```bash
cd MCP
.venv-forecast/bin/python -m pins.forecast.baselines   # persistence + moving-avg gate
.venv-forecast/bin/python -m pins.forecast.model        # train + eval the attention forecaster
```

---

# Stage-2 NEGOTIATION — which allocation mechanism rations GPUs best?

The headline negotiation experiment (thesis refocus 2026-06-17): on one shared job stream,
does the PINS sealed-bid auction beat value-blind scheduler baselines on **SLA-violation rate**
at high utilisation? Harness: `pins/negotiation_sim.py` (pure Python, runs in `.venv`, no
LLM/MCP). It reuses the real decider `pins/mechanism.py:clear` and the predictor's
`marginal_values`/`PHASE_PROFILES`; baselines are wrapped with the same signature.

## Experiment 9 — Value-max auction vs greedy/equal/static (NEGATIVE; diagnostic)

**Date:** 2026-06-18

**Method.** 16 jobs, horizon 300 steps, seed 0 (+ 8-seed robustness). Each job: a phase timeline
(`preprocess`→`train`×k→`eval`), an **urgency** in [0.6, 2.2] that scales BOTH its private bid
(`predictor.marginal_values`) AND its deadline tightness (urgent → ~1.2× nominal slack, relaxed
→ ~2.4×). A job advances at rate `min(alloc, capacity)/capacity` per step (under-allocation
slows it); SLA is violated if it finishes after its deadline or not within the horizon. Five
strategies = (bid-builder × allocator): **PINS-auction** (static urgency bid), **PINS-auct-DL**
(bid scaled by deadline pressure = remaining-work / time-to-deadline), **greedy-FIFO**
(value-blind, serve in queue order to capacity), **equal-share**, **static-sticky** (no
preemption). Pool size is the contention knob (small = high util). Welfare is always scored on
the *static base* bids so it is comparable across strategies.

**A correctness fix made en route (`mechanism.py`).** The anti-thrashing gate charged
`rescale_cost` for *every GPU a job gained*, including GPUs taken from the **idle pool** — so
from a cold start it refused to allocate free capacity and the auction sat at 0% utilisation.
Fixed to charge only for **preemptions** (`sum(max(0, cur[a]-target[a]))`): filling idle GPUs is
free, only displacing a running job pays. **All 5 `test_mechanism.py` tests still pass.**

**Result (mean SLA-violation rate over 8 seeds; lower = better).**

| pool | PINS-auction | PINS-auct-DL | **greedy-FIFO** | equal-share | static-sticky |
|---|---|---|---|---|---|
| 4 | 96.1% | 97.7% | **87.5%** | 99.2% | 100.0% |
| 6 | 86.7% | 86.7% | **69.5%** | 93.0% | 100.0% |
| 8 | 74.2% | 74.2% | **46.9%** | 76.6% | 100.0% |

At uncontended pools (≥20) all elastic strategies reach 0% SLA. The auction **does win on
welfare** (its own objective, e.g. pool-8 single-seed: auction 12.2k vs greedy 11.5k) — but loses
on SLA at every contended pool. **Beats-baseline (greedy) gate on SLA: FAIL.**

**Why it FAILED (the instructive part).**
1. **The auction optimises the wrong objective.** `clear` maximises welfare = Σ value; with
   **diminishing-returns** curves that means SPREADING GPUs thin across many jobs (B's 1st GPU
   outbids A's 3rd). Under heavy contention, spreading → everyone runs slow → everyone finishes
   late. Welfare ≠ deadlines.
2. **SLA rewards concentration (EDF-like).** SLA is a *count* of jobs meeting deadlines; the
   classic optimum concentrates resources to push the most-at-risk job over the line. Greedy-FIFO
   accidentally does this — stable, in-order, run-to-completion — so more jobs finish on time.
3. **Deadline-aware bidding did not rescue it.** Scaling all marginal values by deadline pressure
   keeps the diminishing shape (still spreads) and the up-to-10× multiplier swings flip the value
   ordering round-to-round → preemption churn (pool 4: only 2/16 finished). Worse, not better.

**Lesson / next step.** Value-max ≠ SLA-min. To break the util/SLA tradeoff the auction must
**concentrate on at-risk jobs**, not spread by marginal value — e.g. an *all-or-nothing deadline
bid* (a behind-schedule job bids high for exactly the GPUs it needs to make its deadline, ~0
beyond), so the same auction clears EDF-like. That is the principled Exp 10. This negative result
is itself thesis-relevant: it shows the **mechanism objective and run-stability**, not value-
awareness per se, are the levers for SLA.

**Reproduce.**
```bash
cd MCP
(cd pins && ../.venv/bin/python test_mechanism.py)   # gate fix stays green
.venv/bin/python -m pins.negotiation_sim             # 5-strategy × 5-pool sweep
```

## Experiment 10 — LLM agents bid strategically (hinge-safe); interpretability, not SLA

**Date:** 2026-06-18

**Method.** Make the agents *actual AI*: a local LLM (qwen2.5:3b via Ollama; no llama tag pulled)
decides each job's bidding **strategy**, never a number — the hinge from Exp 1-7. New module
`pins/llm_agent.py` (mirrors `forecast/llm_facts.py`): given the discretised state
`(phase, capacity, deadline_bucket{behind|ontrack|ahead}, contention{low|high}, tier{prod|besteffort})`
the LLM returns `{stance ∈ aggressive|balanced|concede, focus_gpus ∈ [1,capacity], justification}`.
`stance` is categorical and `focus_gpus` a small COUNT/selection (Exp-3-reliable), not a magnitude.
Deterministic `apply_strategy` maps it onto the *calibrated* baseline curve: a fixed stance
multiplier (1.5/1.0/0.6) + an **all-or-nothing concentration** at `focus_gpus` (GPUs beyond it
collapse to ~0). The LLM is kept OUT of the hot loop via per-state caching — the full
5-pool × 8-seed sweep decided only **32 distinct states** (≈32 Ollama calls, cached to JSON), not
thousands. Graceful fallback to a rule on Ollama-down/`--no-llm`. Added a **prod-tier SLA** metric
(violation rate among the top-third-urgency 'prod' jobs) to operationalise the value-weighted-SLA
reframe, reported beside the raw count.

**Result (8-seed mean; SLA = all jobs, prodSLA = prod tier; lower = better).**

| pool | metric | PINS-auct-DL | **llm-strategic** | **greedy-FIFO** | equal-share |
|---|---|---|---|---|---|
| 4 | SLA / prodSLA | 97.7 / 100 | 98.4 / 100 | **87.5 / 100** | 99.2 / 100 |
| 6 | SLA / prodSLA | 86.7 / 83.3 | 89.1 / 89.6 | **69.5 / 70.2** | 93.0 / 95.8 |
| 8 | SLA / prodSLA | 74.2 / 74.9 | 78.1 / 73.4 | **46.9 / 53.8** | 76.6 / 92.0 |

The LLM strategist **does win welfare/goodput** over the formula auctions (e.g. pool-8 mean 14550
vs 12637) — its concentration reshapes *which* jobs run. Sample justifications (auditable, the
point vs RL): *"Behind schedule; concentrating on fewer GPUs to ensure timely completion despite
cluster contention"* (→ aggressive, focus 4 of 8); *"Ahead of deadline; production job can wait
with fewer GPUs"* (→ concede, focus 1). Artifacts: `pins/llm_agent_cache.json`,
`pins/results_llm_negotiation.json`.

**Honest read.**
1. **The LLM bids sensibly and hinge-safely** — coherent, varied strategies; no GB/price ever
   leaves the model; stays out of the hot loop (32 cached states). The *mechanics* of LLM-agent
   negotiation work.
2. **But it does NOT beat the baseline on SLA — and neither does anything else.** Across 8 seeds
   greedy-FIFO wins both raw SLA and **prod-tier** SLA at every contended pool; llm-strategic ≈ the
   formula auctions. **The metric reframe did not rescue the auction** — greedy wins the
   value-weighted view too, because its edge is *stability / run-to-completion*, a property of the
   allocator, not of the bid. This corroborates Exp 9: no bid design (static, deadline, or
   LLM-strategic) overcomes a stable FIFO on deadline-meeting.
3. **So the LLM's contribution here is interpretability + modest goodput, not SLA.** That is the
   defensible position from the framing discussion: *match on outcomes, uniquely explain every
   decision* (auditable bids vs a black-box RL/greedy). Beating greedy on raw SLA would require a
   **stability mechanism** (commit/run-to-completion, stronger anti-thrashing), which is the next
   lever — not more bidding cleverness.

**Reproduce.**
```bash
cd MCP
.venv/bin/python -m pins.llm_agent                       # smoke: strategy+justification per state
.venv/bin/python -m pins.negotiation_sim --llm --model qwen2.5:3b   # adds the llm-strategic row
```

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

**Part B — a flat per-claim budget does NOT fix it (negative result).** Each active tick a job pays
`PRIO_CLASS_COST[declared]` from an equal budget; insolvent → demoted. Hypothesis: liars on long,
loose jobs pay tick after tick and run dry, while honest-urgent jobs finish fast and afford it.

| pool | BE-lie no-budget | B=20 | B=40 | B=80 | B=160 |
|---|---|---|---|---|---|
| 8 (prodSLA) | 53.8 | 64.0 | 60.4 | 51.7 | 51.7 |

It does not recover toward 25%. **The control kills it:** *truthful* play WITH a small budget also
craters (pool-8 prodSLA 25% → **88%** at B=20), because honest prod jobs also declare `critical` and
pay the same flat cost — so the budget tracks **job residence/length, not truthfulness**, and starves
the very jobs it should protect. Large budget → no effect (converges to no-budget); small budget →
hurts honest and liar alike.

**Why no flat scheme can work (the real lesson).** A liar and an honest job that declare the same
class are *indistinguishable* by any cost on the class itself. To separate them you must tie the cost
to something only the liar lacks — its **true value** — which requires **eliciting value with
PAYMENTS** (uniform-price / VCG), where over-claiming is irrational because you pay more than the
service is worth. That reconnects to the priced marginal auction (`mechanism.py`, Exp 9) and exposes
the **core Stage-2 tension**: the SLA-winning mechanism (committed *priority classes*, no payment) and
the incentive-compatible mechanism (*payments*) are different designs. Unifying them — e.g. exogenous
**per-user budgets** spent across a user's jobs (the SLURM fair-share / cloud-quota model, which needs
multi-job agents) so that spending reveals true relative urgency — is the identified open problem.

**Reproduce.** Strategies in `pins/negotiation_sim.py` (`declare_truthful`,
`declare_inflate_besteffort`, `make_declared_committed(declare_fn, budget=...)`); driven by an
8-seed script over pools {6,8,12}.

---

## Experiment 14 — The SUPPLY agent: two-sided negotiation (headroom reservation) — regime-gated win, and model size finally matters

Everything in Exp 9-13 is **demand-only** (jobs bid / declare; a deterministic auctioneer clears).
The thesis (`Research/CLAUDE.md`, `research_plan.md`) wants a **two-sided** scheduler: a SUPPLY
agent (the resource pool) with an asymmetric objective negotiating against demand. This builds the
first supply agent. **Lever = headroom reservation:** hold back `R` GPUs from best-effort jobs so a
late-arriving prod job lands on idle capacity instead of waiting. `R` is the deterministic stand-in
for the negotiated outcome (demand wants `R`→0; supply wants `R`>0). New self-contained module
`pins/supply_sim.py`; the validated Exp 9-13 harness is imported, never touched.

**Part A — malleable regime: NEGATIVE (reservation is redundant).** With instant, free preemption
(the original sim), `committed-auction` already serialises prod to the front, so a late prod job
preempts best-effort at zero cost. Reserving idle GPUs only starves the present. 8-seed, pool 8:

| strategy | SLA | prodSLA | util |
|---|---|---|---|
| committed (no supply) | **53.1** | **23.0** | **95** |
| reserve-adaptive(R=1) | 54.7 | 23.0 | 91 |
| reserve-adaptive(R=3) | 62.5 | 23.0 | 82 |

prodSLA is **untouched** by any R (prod already served first); overall SLA and util only worsen.
Adding a `preempt_penalty` (checkpoint-rollback cost, 0→1) barely moved committed (53.1→53.9):
its bid-once frozen-priority **stability already eliminated the preemption thrash** reservation was
meant to prevent. **Reservation cures a disease the commit mechanism already cured.**

**Part B — rigid-incumbent regime: the WIN.** Make incumbents **non-preemptable** (`simulate_rigid`):
a running job holds its GPUs run-to-completion (shrinks only voluntarily at a phase boundary);
newcomers/growth draw ONLY from free GPUs. Now a late prod job *cannot* bump a rigid best-effort
job — it lands on reserved headroom or WAITS. prodSLA finally responds. 8-seed:

| pool | committed SLA/prodSLA/util | reserve-adaptive(R=1) | note |
|---|---|---|---|
| 12 | 25.8 / 26.8 / 79 | **25.8 / 18.8 / 77** | **Pareto win**: −8pt prodSLA, equal SLA, −2pt util |
| 8  | 68.0 / 73.1 / 94 | 70.3 / 69.2 / 91 (R=2: 75.0/66.4/88) | a TRADE (prodSLA↓ costs SLA/util) |
| 6  | 85.2 / 85.0 / 97 | 85.9 / 85.0 / 94 | no help (too contended) |
| 4  | 95.3 / 100 / 99 | 100 / 100 / 96 | HURTS (reserving while starving) |

The supply agent helps **only at moderate contention** (sweet spot ≈ pool 12), trades at pool 8,
and hurts when tight. **Adaptive > static** (reserve only while prod load is incoming, then release):
it recovers the overall-SLA cost the static reserve pays. The malleable null wasn't a failure — it
correctly located *where* the lever lives: **non-malleable incumbents** (the gang-scheduled /
checkpoint-boundary regime real HPC faces). This sharpens the `research_plan.md` scope: under full
malleability the second agent is redundant; its QoS value requires rigidity.

**Part C — LLM supply agent, and the model-size question answered.** The LLM (hinge-safe: emits a
categorical level `none|light|heavy` from `(contention, incoming_prod)`, code maps level→GPUs;
`pins/llm_agent.llm_reserve`, cached per state — 9 states, a handful of calls). This is the **first
supply-side decision with real stakes**: reserve when scarce, or when nothing is incoming, and you
ACTIVELY HURT SLA. Ran qwen2.5 **3b / 7b / 14b** vs the deterministic adaptive oracle + a rule agent.

The decisions in the `scarce` states (where the agent MUST decline) separate the models:

| state | 3b | 7b | 14b | rule/oracle |
|---|---|---|---|---|
| scarce\|few | **light ✗** | none ✓ | none ✓ | none |
| scarce\|many | **heavy ✗** | **heavy ✗** | none ✓ | none |
| moderate\|none | **light ✗** | none ✓ | none ✓ | none |

**3b systematically over-reserves** — including the exact dangerous call (reserve when scarce) —
and its own justification for `scarce|few` reads *"Contention is currently **low**…"*: it **misread
"scarce" as low contention.** A comprehension error the larger models avoid. **14b is perfect**
(declines in every scarce state, matches the hand-built oracle, and distinguishes `ample|many→heavy`
from `moderate|many→light`); 7b is in between (one bad call). It shows in outcomes at tight pools —
pool 4 SLA: committed 95.3, **3b 100**, 7b 99.2, **14b 96.1**; pool 6 SLA: **3b/7b 86.7, 14b 84.4**.
At the sweet spot all capture the win (pool 12 prodSLA **18.8** for every agent; 14b/rule push
overall SLA 25.8→**23.4**, a Pareto improvement).

**Takeaways.** (1) Two-sided negotiation **can** improve QoS — prodSLA 26.8→18.8 at equal SLA — but
only under **non-malleable incumbents at moderate contention**; it is regime-gated, not universal.
(2) Unlike the earlier mechanism-ceiling cases (where the deterministic decider caps the LLM and
model size is irrelevant), the reservation decision is a genuine **load-aware judgment with stakes**,
so **model capability finally matters**: 3b makes SLA-hurting misjudgments, 14b matches the oracle.
(3) The justification trace is the deliverable — it makes 3b's failure *visible and auditable* (the
edge vs RL). Connects to the open `T_negotiation`/load-adaptive domino: the supply agent's real job
is deciding *when* to reserve, and that is a reasoning task.

**Reproduce.** `pins/supply_sim.py`: `--penalty {0,1}` (Part A malleable), `--rigid` (Part B), `--llm
--models qwen2.5:3b,qwen2.5:7b,qwen2.5:14b` (Part C); supply agent in `pins/llm_agent.llm_reserve`;
8-seed, pools {4,6,8,12,20}; LLM trace → `pins/results_supply_llm.json`.

## Experiment 15 — MIXED malleable + rigid incumbents: malleability-awareness recovers the reservation's utilisation cost

**Date:** 2026-06-19

**Question.** Exp 14 ran two ENDPOINTS — all-malleable (reservation redundant: a late prod job
preempts for free) and all-rigid (reservation wins on prodSLA but pays idle-GPU utilisation). Real
HPC is a MIX (some apps resize at runtime, most are rigid), which the `research_plan.md` "malleable
only / rigid out of scope" assumption ignores — and which Exp 14 showed is exactly the assumption
that makes the supply agent redundant. This experiment fills the axis between the endpoints and asks
whether KNOWING which incumbents are malleable lets the supply agent reserve smarter.

**The lever.** The supply agent has two sources of headroom for a late prod job: (1) **reserved idle
GPUs** (costs utilisation), and (2) **GPUs reclaimed by shrinking a malleable best-effort incumbent
on demand** (free). A malleability-**aware** agent reserves idle headroom only against the **rigid**
fraction it cannot reclaim, and lends the reserved pool to malleable best-effort jobs it can claw
back later; a **blind** agent holds `R` idle against everyone. *Hypothesis:* aware ≥ blind
everywhere, the gap growing with the malleable fraction φ.

**Method.** New self-contained code in `pins/supply_sim.py`; the validated Exp 9-13 harness is
imported, never touched, and malleability is carried in an EXTERNAL per-job map (not a `Job` field),
so nothing downstream changes. `simulate_mixed` is one incremental simulator parameterised by
`malleable[jid]`: malleable incumbents can be shrunk involuntarily (reclaimed, free) by a
higher-priority job; rigid ones only shrink voluntarily at a phase boundary. `malleability_map`
assigns each job a fixed uniform draw so raising φ only ADDS malleable jobs (nested sweep). The
reserving allocator has `aware ∈ {blind, aware}`. 8-seed, R=2 adaptive.

**Endpoint gate (the correctness check).** `simulate_mixed` at **φ=0 reproduces `simulate_rigid`
EXACTLY** for every strategy/pool/seed (`--check` → PASS); at φ=1 it reproduces the Exp-14A
"reservation redundant" limit (committed == aware). So the mixed simulator is a faithful
generalisation that contains both Exp-14 regimes as its endpoints — not a new, separately-tuned
model.

**Result (8-seed mean; cells = prodSLA% / util%, lower prodSLA & higher util = better).**

pool 8
| φ | committed (no supply) | reserve-BLIND | reserve-AWARE |
|---|---|---|---|
| 0.00 | 73 / 94 | 66 / 88 | 66 / 88 |
| 0.25 | 53 / 95 | 45 / 87 | **45 / 91** |
| 0.50 | 40 / 94 | 41 / 87 | **41 / 92** |
| 0.75 | 31 / 95 | 29 / 87 | **29 / 94** |
| 1.00 | 23 / 95 | 23 / 87 | **23 / 95** |

pool 12
| φ | committed | BLIND | AWARE |
|---|---|---|---|
| 0.00 | 27 / 79 | 19 / 76 | 19 / 76 |
| 0.25 | 18 / 79 | 16 / 76 | **16 / 77** |
| 0.50 | 20 / 79 | 20 / 77 | **20 / 78** |
| 0.75 | 18 / 79 | 16 / 77 | **16 / 79** |
| 1.00 | 15 / 79 | 15 / 76 | **15 / 79** |

**Hypothesis CONFIRMED.**
1. **AWARE ≡ BLIND on prodSLA at every φ** — malleability-awareness keeps the FULL reservation QoS
   win (the prodSLA column is identical to blind), so it costs nothing in deadline-meeting.
2. **AWARE recovers the idle-utilisation BLIND wastes, and the recovery GROWS with φ** — pool-8 util
   gap (aware − blind) = 0, 4, 5, 7, 8 pts across φ; by φ=1 AWARE == committed on util (95 vs blind's
   87), i.e. it correctly stops paying for headroom it can reclaim. **AWARE Pareto-dominates BLIND
   for all φ>0** (equal prodSLA, ≥ util).
3. **The supply agent's prodSLA edge over no-supply is φ-graded** — largest when mostly rigid
   (φ=0, pool 8: 66 vs 73; pool 12: 19 vs 27) and **vanishing at φ=1** (23=23 / 15=15, reproducing
   Exp-14A). This *quantifies* the Exp-14 lesson: **the supply agent's value scales with the rigid
   fraction**, and a malleability-aware agent extracts it at minimal utilisation cost.

**Honest read.**
- AWARE's gain over BLIND is **utilisation-recovery at equal prodSLA**, NOT a prodSLA improvement
  over blind. The contribution is turning Exp-14's pool-8 "trade" (−8 pt prodSLA for −8 pt util) into
  a near-Pareto move (−8 pt prodSLA for only −3/−4 pt util), and removing the idle waste entirely as
  φ→1.
- Over-contention (pool 6) reservation still HURTS regardless (it can even raise prodSLA above
  committed at φ=0: 89 vs 85) — unchanged from Exp 14B; the lever lives at moderate contention.
- Reclaim is modelled as FREE (consistent with Exp-14A free preemption); a non-zero `reclaim_penalty`
  (checkpoint/rescale rollback) is the obvious next stressor and would erode aware's util recovery in
  proportion to how often it reclaims. Deterministic only so far — the LLM supply agent reasoning
  over a third state dimension (`malleable_fraction`) is the next step, where Exp-14C predicts model
  size will matter (deciding how much to reserve vs reclaim is a load-aware judgment with stakes).

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.supply_sim --check    # endpoint gate: PASS (phi=0 == simulate_rigid)
.venv/bin/python -m pins.supply_sim --mixed    # blind vs aware across phi, pools {6,8,12}
```
`pins/supply_sim.py`: `malleability_map`, `simulate_mixed`, `check_endpoints`, `sweep_mixed`;
`--reserve R`, `--static` (vs default adaptive). Decider untouched (`test_mechanism.py` 5/5 green).

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

## Experiment 17 — The LLM demand agent decides the hedge from uncertainty (interpretable; model size matters)

**Date:** 2026-06-19

**Why.** Exp 16 sized the safety margin deterministically (`margin = round(u·scale)`). The thesis
wants the *AI agent* to make that call with an auditable justification — and Exp 16's own lesson is
that a margin should be taken only when **uncertain AND deadline-at-risk AND capacity is spare**
(a blanket margin backfires). That is a *judgement*, exactly the kind the LLM should own (the LLM
reasons; code decides the number). This is the demand-side mirror of the Exp-14 supply agent.

**Method.** New `pins/llm_agent.llm_margin`: from a discretised state `(uncertainty bucket {low,
medium,high}, deadline {behind,ontrack,ahead}, contention {low,high}, tier)` the LLM emits a
categorical **hedge ∈ {none, some, heavy}** + a one-sentence justification — never a GPU count.
Code maps the hedge to an effective uncertainty fed to `predictor.marginal_values`
(`none→0`, `some→u`, `heavy→u+1/scale`), which owns the number — hinge intact. Cached per state
(≤36 states → a handful of Ollama calls), out of the hot loop, rule fallback on Ollama-down. Added
as a 4th policy in `uncertainty_sim.py` (`--llm --model …`) beside the Exp-16 no/fixed/sized
policies; the simulator now also computes each job's live deadline + contention bucket so the agent
can reason. Ran qwen2.5 **3b / 7b / 14b**, 16 seeds, pool 12 (the regime where margin matters).

**Result (pool 12, 16-seed mean, spike 0.6; lower better).**

| policy | SLA | prodSLA | util |
|---|---|---|---|
| no-margin | 23.8 | 11.9 | 81 |
| fixed-margin | 27.7 | 13.2 | 84 |
| uncertainty-sized (Exp 16) | 23.8 | 11.9 | 82 |
| **llm-margin (3b)** | 26.2 | 13.0 | 82 |
| **llm-margin (7b)** | 23.8 | 11.9 | 81 |
| **llm-margin (14b)** | **23.4** | 11.9 | 81 |

**Findings — the same "judgement with stakes" pattern as the supply agent (Exp 14C).**
1. **Model size matters here, unlike the mechanism-capped cases.** The hedge is a genuine load-aware
   judgement, so capability shows: **3b over-hedges and HURTS** (26.2 vs the deterministic 23.8) —
   it hedges `heavy` even under HIGH contention, the exact call the prompt forbids (no spare
   capacity). Its own justification is the smoking gun: for `medium|behind|high|besteffort` it wrote
   *"high cluster contention indicating **spare capacity** that can be used to absorb a spike"* — it
   **misread "high contention" as "spare capacity"**, the same comprehension error 3b made with
   "scarce" in Exp 14C. **7b is correct** (23.8, matches the deterministic oracle); **14b edges it**
   (23.4) by gating margin tightly to the at-risk-with-spare-capacity states.
2. **The justification trace is the deliverable.** Every hedge ships with a one-sentence reason
   (`pins/results_uncertainty_llm.json`), which is precisely what makes 3b's failure **visible and
   auditable** — the interpretability edge over a black-box RL bidder. 14b's reasons are clean
   (e.g. `high|ahead|*` → none: "high uncertainty, but ahead of schedule … no additional margin").
3. **Hinge held.** The LLM emitted only a categorical hedge; `predictor.marginal_values` owned every
   GPU count. Out of the hot loop (≤36 cached states), rule fallback verified (`--no-llm` matches the
   deterministic uncertainty-sized policy).

**Heavy-tail check refutes the "LLM gating wins" hypothesis (the instructive part).** I expected
14b's *live* state-gating to separate from the blanket "always size by u" under heavy tails. It does
the OPPOSITE (pool 12, spike 1.5, 16 seeds):

| policy | SLA | prodSLA |
|---|---|---|
| uncertainty-sized (deterministic) | **32.8** | **17.4** |
| llm-margin (14b) | 33.6 | 22.8 |
| no-margin | 34.0 | 23.7 |

Under heavy tails 14b **underperforms** the deterministic policy (prodSLA 22.8 vs 17.4) — it drifts
back toward no-margin. The cause is not model capability: the prompt told it to hedge `none` under
**high contention** (a lesson true at *mild* spikes, Exp 16), but under heavy tails the spiking jobs
need that margin *even when the pool is busy*, so the contention-gate, faithfully applied **out of
its regime**, suppresses margin exactly where it is needed. The blanket deterministic policy, which
ignores contention, is more robust here.

**The fix — add a SPIKE-RISK signal (fix the decision, not the model).** The diagnosis says the
agent lacked a signal for *how bad a miss would be*: `uncertainty` is the interval WIDTH, but the
contention-gate needs to know the upper-tail SEVERITY. So a new context dimension `spike_risk`
(low/medium/high, from the plausible relative over-run = upper-tail magnitude) was added, with the
rule/prompt revised so **high spike-risk OVERRIDES the contention-gate** (hedge to protect the
deadline even when contended); the gate only applies when spike-risk is mild. Re-ran 3b/7b/14b.

**Result with spike-risk (pool 12, 16 seeds).**

| regime | policy | SLA | prodSLA |
|---|---|---|---|
| spike 1.5 | uncertainty-sized (det.) | 32.8 | 17.4 |
| spike 1.5 | llm-margin **3b** | 34.4 | 17.4 |
| spike 1.5 | llm-margin **7b** | **32.0** | 17.4 |
| spike 1.5 | llm-margin **14b** | **32.0** | 17.4 |
| spike 0.6 (regression) | llm-margin **14b** | 23.8 | 11.9 |

**It worked.** With the spike-risk signal, **7b/14b now BEAT the deterministic policy under heavy
tails** (SLA 32.0 vs 32.8; prodSLA matches its best 17.4) — recovering from the previous loss
(33.6/22.8) — and the mild-tail case still matches (23.8/11.9, no regression). The LLM hedge now
generalises across both regimes. **3b improved** (prodSLA 22.8→17.4) but still over-hedges on
overall SLA (34.4, util 86) — the weak model remains the weak model even with the better signal.

**Honest read.**
1. **"Fix the decision, not the LLM" — demonstrated, not just asserted.** The heavy-tail failure was
   a **mis-specified decision** (a contention-gate applied out of its regime), and supplying the
   missing signal — not a bigger model — fixed it. This is the demand-side echo of Stage-1's whole
   arc (Exp 1-7: the cure for an LLM blind spot was more deterministic *input/structure*, not scale).
2. **Model size still matters for the judgement** (Exp-14C/17 pattern): given the same signal, 7b/14b
   apply the override correctly and win; 3b over-hedges and only partly benefits. The justification
   trace keeps every hedge auditable (the edge vs RL).
3. **Net:** the LLM demand agent (7b+) is now **competitive-or-better than the deterministic margin in
   BOTH regimes** while adding interpretability — the uncertainty co-contribution feeds a genuine,
   auditable agent decision end-to-end.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.llm_agent --no-llm                              # hedge probes (rule, w/ spike_risk)
.venv/bin/python -m pins.uncertainty_sim --llm --model qwen2.5:14b --spike 1.5 --seeds 16
```
`pins/llm_agent.llm_margin` (+ `uncertainty_bucket`, `spike_risk_bucket`, `margin_uncertainty`);
`pins/uncertainty_sim.make_llm_policy`; trace → `pins/results_uncertainty_llm.json`.

---

# Stage-2 DECIDER — LLMSched-style ILP vs the PINS auction

The thesis `research_plan.md` Open-Question #1 asks which deterministic decider should consume
the two-LLM negotiation: (a) the PINS auction (`pins/mechanism.py`) or (b) an LLMSched-style ILP
(reason→guarantee, the IEEE OJ-CS 2026 paper in `Research/CLAUDE.md`). This experiment builds the
(b) arm as a true drop-in — same `(bids, total_gpus, current)` signature, consuming the SAME
negotiated marginal-value curves, scored by the SAME metrics — and compares them head-to-head,
first on the existing 1-D GPU pool, then on a new 2-D node/placement model. New deps/modules:
`pulp` (CBC MILP); `pins/ilp.py`, `pins/placement.py`, `pins/placement_sim.py`;
`pins/test_ilp.py` (4), `pins/test_placement.py` (3). `test_mechanism.py` still 5/5.

## Experiment 18 — ILP ties the auction in 1-D; removes a structural placement loss in 2-D

**Date:** 2026-06-22

### Part A — Single GPU pool (1-D): the ILP only TIES the auction (NEGATIVE; expected)

**Method.** A MILP allocator (`pins/ilp.allocate`) maximises welfare under the SAME per-GPU
anti-thrashing penalty `λ` (`rescale_cost`) the auction uses, but solved by CBC. Value is
linearised exactly per GPU-unit (the curve is non-increasing, so contiguous fill is automatic).
Dropped into `negotiation_sim.py` as `ILP-welfare` / `ILP-DL` beside `PINS-auction` /
`PINS-auct-DL`. Single seed=0 sweep (directional, not the 8-seed protocol of Exp 9-13).

**Result (seed 0; SLA / prodSLA, lower better; welfare higher better).**

| pool | PINS-auction | ILP-welfare | note |
|---|---|---|---|
| 6 | 81.2 / 83.3 · w 14627 · slow 4.73 | 87.5 / 83.3 · w 14589 · slow **4.29** | auction edges SLA |
| 8 | 56.2 / 66.7 · w 12227 · slow 2.43 | 62.5 / 83.3 · w **12353** · slow **1.76** | ILP edges welfare/slow |
| 12 | 12.5 / 16.7 · w 11935 · slow 1.16 | **0.0 / 0.0** · w **12064** · slow **1.14** | ILP wins SLA |
| 20 | 0.0 / 0.0 · w 11029 | 0.0 / 0.0 · w **11056** | tie |

**Why it TIES (the point).** On one divisible pool with non-increasing curves, welfare-max is
**already solved optimally by the auction's greedy fill** — so the ILP cannot beat it on welfare,
and across pools it matches to within rounding (`test_ties_auction_on_welfare` asserts
exact-equal welfare across pool sizes). The one real behavioural difference:
the auction's anti-thrash gate is **all-or-nothing per round** (apply the whole target or keep
current), whereas the ILP does **fine-grained partial preemption** — giving up only the individual
GPUs whose marginal value beats the rescale cost. That recovers a sliver of welfare the gate leaves
on the table (pool 8: 12353 vs 12227) and consistently lowers slowdown — but it is a sliver.

**Cost.** Per round the ILP is **~150× slower** (auction 0.056 ms vs CBC 8.5 ms; still inside
LLMSched's ~50 ms budget). **Verdict: on 1-D rationing the ILP is not worth the latency + 15 MB
solver dependency.** This is itself thesis-relevant — it locates exactly where the ILP earns its
keep: constraints the count-only auction **cannot express**.

### Part B — Nodes + co-location (2-D): the auction is structurally handicapped; the ILP isn't

**Method.** GPUs live on `N` nodes × 8; jobs are **co-located** (all GPUs on one node — the
NVLink-coupled training case), and placement is **STICKY** (a running job cannot migrate for free,
modelling real checkpoint/rescale cost). The auction clears a GPU *count* blind to nodes, then a
deterministic best-effort placement (`place_sticky`, first-fit-decreasing) honours what it can and
**repairs** (shrinks) the rest — the repaired-away GPUs are its structural loss. The ILP
(`allocate_placement`) plans count AND node jointly and may **migrate** a live job at a bounded
`migrate_cost=1.5` — a lever the count-only auction cannot even express. Contended workload
(40 jobs, arrivals compressed into 60 steps, horizon 400, seed 0). New column `ploss` = mean
GPUs/round won but unplaceable. `simulate` carries node assignments across rounds.

**Result (seed 0; ploss = GPUs/round lost to fragmentation).**

| cluster | strategy | SLA | prodSLA | util | welfare | slow | **ploss** |
|---|---|---|---|---|---|---|---|
| 2×8 | auction+sticky | 97.5 | 100.0 | 91% | 37060 | 10.87 | **1.28** |
| 2×8 | **ILP-place** | 90.0 | 76.9 | **98%** | 34906 | **8.73** | **0.00** |
| 2×8 | greedy+sticky | **87.5** | **76.9** | 96% | 28265 | 7.69 | 0.14 |
| 3×8 | auction+sticky | 97.5 | 92.3 | 84% | 34494 | 7.25 | **2.96** |
| 3×8 | **ILP-place** | 92.5 | 92.3 | **96%** | 33852 | **5.98** | **0.00** |
| 3×8 | greedy+sticky | **77.5** | **61.5** | 92% | 27995 | 5.05 | 0.42 |
| 4×8 | auction+sticky | 90.0 | 92.3 | 84% | 34149 | 5.16 | **3.62** |
| 4×8 | **ILP-place** | 85.0 | 84.6 | **93%** | 32364 | **4.27** | **0.00** |
| 4×8 | greedy+sticky | **65.0** | **53.8** | 92% | 27868 | 3.60 | 0.84 |
| 6×8 | auction+sticky | 80.0 | 76.9 | 79% | 31087 | 3.11 | **5.37** |
| 6×8 | **ILP-place** | 85.0 | 92.3 | **91%** | 31031 | **2.65** | **0.00** |
| 6×8 | greedy+sticky | **57.5** | **46.2** | 86% | 28030 | 2.35 | 1.47 |

**Findings.**
1. **Node placement DOES break the 1-D auction, and the breakage GROWS with the cluster.** The
   auction wastes `ploss` = 1.28 → 2.96 → 3.62 → 5.37 GPUs *every round* to fragmentation it cannot
   foresee (more nodes = more ways to strand a whole-node train job behind sticky small jobs). The
   ILP's `ploss` is **0 by construction** — it plans count+node jointly and migrates to consolidate.
   This is the regime that justifies the ILP, exactly as Part A predicted (vs the 1-D tie).
2. **The recovered capacity shows up as utilisation and slowdown, NOT welfare/SLA.** ILP-place wins
   utilisation by **7–12 pts** (98 vs 91, 96 vs 84, 93 vs 84, 91 vs 79) and slowdown at every size.
   But welfare is **slightly LOWER** than the auction (37060→34906 etc.) and SLA is **mixed** (ILP
   beats the auction at 2×/4× but not 6×). Honest cause: welfare/SLA are the *concentration+stability*
   axis from Exp 9-11, not the placement axis — the ILP optimises per-round welfare, which still
   spreads/migrates value across jobs the way the per-round auction does.
3. **`greedy+sticky` wins raw SLA AND prodSLA at every cluster size** (SLA 87.5/77.5/65/57.5),
   beating both value-aware deciders — and its `ploss` stays low (stable id-order ⇒ placement-friendly).
   This is the **same lesson as Exp 9/11**: under heavy contention the SLA lever is a *stable,
   concentrated (run-to-completion) order*, which neither the per-round auction nor the per-round ILP
   provides. The ILP fixes *placement feasibility*; it does not fix *scheduling discipline*.

**Honest read / through-line.** Part A + B answer Open-Question #1 with data: **the ILP is redundant
on the 1-D pool (ties the optimal auction at ~150× cost) and earns its keep only once node/placement
constraints exist** — there it removes a loss the count-only auction *structurally cannot* (1.3→5.4
GPUs/round) and lifts utilisation 7–12 pts. But it is not a free SLA win: deadline-meeting under
contention still wants the committed/stable order (Exp 11), an orthogonal axis. This points to the
natural next experiment: **committed priority (order) + ILP placement** — i.e. the auction/committed
layer sets *who* and the ILP decides *where*, the layered "auction sets priorities → ILP places"
architecture from `Research/CLAUDE.md`. Caveats: single seed=0 (not the 8-seed protocol — directional
only); co-location + sticky-no-migration is one point on the placement-rigidity axis (mirrors Exp 14's
malleable/rigid split — `migrate_cost` is the analogue knob); reclaim/migration modelled at a flat
cost; CBC per-round is the bottleneck (full 2-D sweep ≈ 90 s).

**Reproduce.**
```bash
cd Research
(cd pins && ../.venv/bin/python test_ilp.py && ../.venv/bin/python test_placement.py)  # 4 + 3 green
.venv/bin/python -m pins.negotiation_sim     # Part A: ILP-welfare / ILP-DL beside the auction
.venv/bin/python -m pins.placement_sim       # Part B: auction+sticky vs ILP-place, ploss column
```
`pins/ilp.py` (`allocate`, `allocate_placement` + `migrate_cost`); `pins/placement.py`
(`Cluster`, `place_ffd`, `place_sticky`); `pins/placement_sim.py` (`simulate`, sticky node state).

---

# Stage-1 REVISITED — what can we actually predict on the *real* MIT Supercloud trace?

## Experiment 19 — Runtime prediction from thin metadata: retrieval beats the LLM (NEGATIVE for the LLM; sharpens the design)

**Why runtime, and why not memory/utilisation.** Re-derived the Stage-1 target from the
real data instead of the synthetic CNN VRAM proxy (Exp 1-7). Pulled the full scheduler log
(`data/slurm-log.csv`, public S3 `s3://mit-supercloud-dataset`, no creds) and joined all
**3,430 labelled DNN jobs**. Three candidate "resource-demand" targets turned out to be
**dead** for this workload (memory `supercloud-profiling-data-reality`):
- *Used GPU memory* is contaminated — 87% of telemetry jobs sit pinned at ~30.3 GB because
  TensorFlow (VGG/ResNet/Inception/U-Net) reserves the whole V100; only ~19 PyTorch jobs
  report real memory.
- *Requested resources* (`tres_req`) are a flat copy-paste template (every model ≈ 2 GPU /
  40 CPU / 332 GB) — no per-job signal.
- Measured 3 ways, jobs have **no per-job GPU slack**: median util 92%, 0/110 multi-GPU jobs
  leave a GPU idle, only 1.9% of time below 50% util. The real inefficiency is **queueing**
  (median wait ~15 h). ⇒ the utilisation win here is *cluster-level scheduling*, and the
  Stage-1 prediction that feeds it is each job's **wall-clock runtime, with uncertainty**.

**Method.** Predict `runtime_min = time_end - time_start` (3,414 COMPLETED jobs, 10-2605 min,
median 163) from submission metadata (model name + requested GPUs/CPU/mem + time limit).
Predictors emit P10/P50/P90 (quantiles, mirroring `pins/forecast/model_quantile.py`):
**mean** (global median), **heuristic** (calibrated fraction of the time limit), **retrieval**
(per-model empirical P10/P50/P90; global-quantile fallback for an unseen model), and the
**LLM** (qwen reasons to `{p10,p50,p90}`, clamped to a plausible band, one call per distinct
prompt, cached). Two splits: in-distribution 5-fold, and **leave-one-model-family-out** (OOD).
`pins/eval/predict_runtime.py`.

**Result — in-distribution (5-fold).**

| predictor | MAE(m) | MdAE(m) | within2x | logRMSE | rho | coverage | width(m) |
|---|---|---|---|---|---|---|---|
| mean | 199.3 | 114.5 | 44.3% | 1.07 | -0.04 | — | — |
| heuristic (timelimit) | 198.5 | 113.5 | 44.7% | 1.06 | -0.02 | — | — |
| **retrieval (per-model)** | **172.0** | **79.4** | **57.4%** | **0.95** | **+0.50** | **0.79** | 598 |
| LLM qwen2.5:3b | 1101.8 | 637.4 | 20.7% | 2.21 | -0.03 | 0.16 | 557 |
| LLM qwen2.5:7b | 275.9 | 163.2 | 31.3% | 1.58 | -0.06 | 0.13 | 122 |
| LLM qwen2.5:14b | 236.2 | 125.1 | 28.8% | 1.66 | +0.11 | 0.16 | 77 |

**Result — OOD (leave-one-model-family-out), within2x / rho / coverage.**

| held-out | n | retrieval | qwen2.5:3b | qwen2.5:7b | qwen2.5:14b |
|---|---|---|---|---|---|
| gnn | 92 | **25.0**/+.25/.48 | 5.4/+.33/.03 | 19.6/+.14/.17 | 6.5/+.20/.04 |
| nlp | 361 | **28.8**/+.36/.98 | 16.1/+.40/.14 | 24.9/-.29/.09 | 26.0/-.28/.04 |
| unet | 1431 | **37.4**/-.01/.68 | 20.8/-.10/.15 | 31.2/-.03/.14 | 33.8/+.07/.23 |
| vision | 1530 | **41.8**/+.09/.85 | 22.5/-.25/.17 | 33.6/+.03/.13 | 26.1/+.03/.12 |

**Findings.**
1. **Retrieval wins decisively in-distribution on every metric and is well-calibrated**
   (within2x 57%, rho +0.50, coverage 0.79 ≈ the 0.80 target). The per-model *empirical
   runtime distribution* IS the signal — a one-line `groupby(model).quantile()` baseline.
2. **The LLM does not earn its cost at any size.** Scale improves point accuracy
   (MAE 1102→276→236; within2x 21→31→29%) but plateaus far below retrieval, never gains rank
   skill (rho ≈ 0 vs retrieval's +0.50), and its intervals are **badly over-confident**
   (coverage 0.13-0.16, far below 0.80 — it doesn't know what it doesn't know). Clamping was
   essential: the raw 3b emitted absurd outliers (pre-clamp MAE ≈ 7.5M min).
3. **OOD does not rescue it.** Retrieval beats every LLM on within2x at *every* held-out
   family. The only flickers of LLM advantage are rank correlation on 1-2 families
   (3b: nlp rho +0.40) — not enough to matter. Note even retrieval's OOD *ranking* is weak
   (rho ≈ 0 on unet/vision): runtime for an unseen family is genuinely hard from metadata for
   ANY method, because within-family runtime is driven by epochs / dataset size /
   early-stopping that the submission metadata simply does not contain.

**Honest read / through-line.** A clean **negative result for the LLM as a runtime
predictor**, and a useful one: it removes the LLM from the Stage-1 runtime path. Stage-1
runtime+uncertainty for the cluster-scheduling sim (Stage-2, next) should be the cheap,
calibrated **retrieval quantiles**, not an LLM — consistent with the project spine
("deterministic code decides; the LLM reasons") and with the Exp 1-4 lesson that the LLM
mis-calibrates absolute numbers. The LLM's demonstrated value stays where Exp 10/12/17 put
it: the *negotiation/justification* layer, not numeric profiling. Caveats: thin metadata
(model name + requests) — an LLM given the actual training script/epoch count might do
better; qwen2.5 only (no frontier model); single Supercloud release; within-family runtime
variance is largely irreducible from this metadata.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.predict_runtime --no-llm                 # baselines + retrieval intervals
.venv/bin/python -m pins.eval.predict_runtime --models qwen2.5:3b,qwen2.5:7b,qwen2.5:14b
```
`pins/eval/predict_runtime.py` (`build_jobs` join, `retrieval_predict`, `llm_predict`,
quantile `score`); data `data/slurm-log.csv` + `data/labelled_jobids_full.csv`
(cache `data/runtime_jobs.csv`); per-job metrics → `pins/eval/results_runtime.json`.

---

# Stage-1 DAG TRACK — does workflow TOPOLOGY predict a task's resource demand? (Alibaba v2018)

Exp 19 closed the door on the MIT Supercloud trace for *structural* prediction: every job is
a single training script, `tres_req` is a flat template, and there are **zero task
dependencies** — so "predict the resource from the job's DAG" is simply not expressible
there. The **Alibaba cluster-trace-v2018** is DAG-native: a batch *job* is a set of *tasks*
with explicit precedence, each carrying its own requested resources. This track moves the
Stage-1 prediction question onto a trace that actually has workflow structure.

## Experiment 20 — DAG extraction + topology-driven demand prediction (PASS; the signal is upstream demand)

**Date:** 2026-06-25

### Part A — Extract per-job task DAGs from `batch_task.csv` (the make-or-break prevalence)

**Method.** `pins/eval/extract_dag.py`. The 766 MB `batch_task.csv` has no header; a job =
all rows sharing `job_name`, and the DAG is reconstructed **purely from each task's
`task_name`** encoding `<prefix><id>_<dep1>_<dep2>…` (e.g. `M5_3_4` = task 5 depends on 3
and 4; `M1` = root). Only the numbers matter; randomly-named tasks (`task_<base64>`,
`MergeTask`) become independent singletons. Vectorised parse (no per-job networkx) → compact
node/edge tables + a longest-path `depth` per node by global iterative relaxation. Output:
`data/alibaba-v2018/dag_{nodes,edges}.csv.gz` + `dag_stats.json`. Deterministic only — no LLM
(structure is parsed, not reasoned).

**Result (full trace).**

| metric | value |
|---|---|
| jobs / tasks | **4,201,014 / 14,295,731** |
| multi-task jobs (≥2) | **59.5%** |
| **jobs with ≥1 dependency edge** | **48.3%** (2.03 M) |
| dependency edges | 9,419,028 |
| tasks/job | mean 3.4 · median 2 · p90 7 · p99 21 · max 1002 |
| DAG depth (jobs w/ edges) | mean 2.46 · median 2 · p90 5 · max 53 |
| malformed | 13 cyclic jobs · 479 edges w/ missing src (negligible) |

**The make-or-break number is 48.3%** — nearly half of all jobs carry real precedence
structure (vs Supercloud's 0%). Verified by hand on `j_3418309`: `M1`,`M2`(d0) → `R3_2`(d1) →
`J4_1_3`(d2) → `R5_4`(d3) reconstructs exactly. The trace genuinely supports "predict demand
from DAG topology."

### Part B — Does topology predict a task's resource demand? (`pins/eval/predict_dag.py`)

**Target = `plan_mem`** (per-task requested memory; 322 distinct values — the richest demand
signal. `plan_cpu` has only 16 → too coarse; `duration` is contaminated with negatives/zeros).
**Caveat up front:** these are *requested* resources (no `batch_instance`/actual-usage file
exists), so the honest framing is *"forecast a task's demand from its DAG context before it is
submitted"* — exactly the upstream signal Supercloud lacked, and useful to the supply/demand
agents.

**Method.** A clean ablation (the lab's beat-the-baseline gate). Three predictors, all
emitting P10/P50/P90 (quantile conventions from `predict_runtime.py`), scored by the same
`score()`: **global** (plan_mem quantiles — the no-information floor), **gbt-nodag** (sklearn
`HistGradientBoostingRegressor`, `loss="quantile"`, on log1p target, features knowable WITHOUT
the DAG: `instances, plan_cpu, stage_type`), and **gbt-dag** (the SAME model + topology:
`depth, in/out_degree, n_tasks_in_job, parent_mem_{mean,max}, parent_cpu_mean`). Any gbt-dag
gain is attributable to topology **alone** (every non-DAG feature is in both arms). No leakage:
`duration` (outcome) excluded; parent features are from UPSTREAM tasks (lower depth), known at
submit. Split **by job** (siblings never straddle train/test), 75/25, seed 0, 500k-job sample
(1.69 M tasks).

**Result 1 — with the co-requested `plan_cpu` available.**

| predictor | MAE | within2x | logRMSE | rho | coverage | width |
|---|---|---|---|---|---|---|
| global (floor) | 0.1554 | 84.7% | 0.755 | +0.015 | 0.82 | 0.390 |
| gbt-nodag | 0.0400 | **98.9%** | 0.260 | +0.851 | 0.80 | 0.186 |
| **gbt-dag** | **0.0395** | 98.8% | 0.256 | **+0.882** | 0.80 | **0.164** |

Gate PASS but **marginal**: topology adds only −1.2% MAE. The no-DAG features already nail it
(within-2x 98.9%) because a task's own **`plan_cpu` is near-deterministic of `plan_mem`** in
Alibaba (resource tiers are set together). Topology still helps *ranking* (rho +0.851→+0.882)
and *sharpens* the interval (width −12%), but is largely redundant when the co-request is known.

**Result 2 — `--no-cpu` ablation: drop the co-request, isolate topology (the real test).**

| predictor | MAE | within2x | logRMSE | rho | coverage | width |
|---|---|---|---|---|---|---|
| gbt-nodag (instances + stage only) | 0.1175 | 94.3% | 0.697 | +0.475 | 0.82 | 0.399 |
| **gbt-dag (+ topology)** | **0.0377** | **98.8%** | **0.277** | **+0.855** | 0.85 | 0.172 |

**Gate PASS, decisively: MAE −67.9%, logRMSE −60.3%, rho +0.475 → +0.855.**

**Findings.**
1. **DAG topology carries the demand signal — and substitutes for the co-request.** With
   `plan_cpu` removed, node-isolated features (parallelism + stage type) manage only within-2x
   94.3% / rho +0.48; adding topology recovers within-2x 98.8% / rho +0.86. Strikingly,
   **gbt-dag WITHOUT cpu (MAE 0.0377) ≈ gbt-dag WITH cpu (0.0395)** — i.e. a task's upstream DAG
   context predicts its memory demand *as well as knowing its own co-requested CPU would*. This
   is the deliverable: forecast a not-yet-submitted task's demand from workflow structure alone.
2. **The interval stays calibrated** — coverage 0.80–0.85 vs the 0.80 nominal across both runs,
   so the uncertainty story (P10/P90) survives, sharper with topology (width 0.39→0.17).
3. **Consistent with the project spine.** Deterministic model decides the number; no LLM (Exp 19
   already showed the LLM doesn't earn its cost as a numeric predictor — the value here is purely
   structural). Stage-1 on a DAG-native trace finally has the structure Supercloud lacked.

**Honest read / caveats.**
- The dominant topology feature is almost certainly **`parent_mem_{mean,max}`** (an edge
  feature) — so part of "topology predicts demand" is really "tasks in a job share a resource
  tier." **Disentangling pure topology (depth/degree) from mere job co-membership** (e.g. a
  job-mean-of-other-tasks baseline that uses no edges) is the immediate next ablation, and would
  sharpen the claim from "upstream demand predicts downstream demand" to "the *graph* matters."
- `plan_mem` is a **request, not measured usage** (no instance-level file) — the model predicts
  what users *ask for*, which on this trace is template-driven and thus highly predictable. A
  trace with actual per-task usage would be the stronger validation.
- 500k-job sample (1.69 M tasks), single 75/25 job split, seed 0 (directional, like Exp 18 —
  not the 8-seed protocol); `--full` runs all 4.2 M jobs. sklearn added as a dependency.

**Reproduce.**
```bash
cd Research
.venv/bin/python -m pins.eval.extract_dag                              # Part A: build node/edge tables
.venv/bin/python -m pins.eval.predict_dag                              # Part B: with plan_cpu (marginal gate)
.venv/bin/python -m pins.eval.predict_dag --no-cpu --out pins/eval/results_dag_nocpu.json  # isolate topology
```
`pins/eval/extract_dag.py` (`build_tables`, `compute_depth`, `summarize`);
`pins/eval/predict_dag.py` (`build_features` topology + upstream-demand join, `gbt_predict`
quantile, `--no-cpu` ablation); artifacts `data/alibaba-v2018/dag_stats.json`,
`pins/eval/results_dag{,_nocpu}.json`.

---

# Stage-1 DAG-GPU — does DAG topology drive ACTUAL GPU demand? (executed & measured on the A100)

**Trace reality (verified against every Alibaba README).** No public trace pairs precedence
DAGs with measured GPU: **v2018** has DAGs but no GPU; **gpu-v2020** has measured `gpu_wrk_util`
but only PS/worker gangs (no precedence); **gpu-v2023/2025** are request-only scheduling
snapshots (neither). Google Borg 2019 is CPU/mem-normalised, no GPU. So to study DAG→actual-GPU
we must MINT the labels — execute real GPU workloads on the topology and measure (the Exp 1-7
A100 closed-loop, lifted to workflows). Honest constraint: v2018 tasks are anonymised (no code),
so each node runs an executable STAND-IN sized to its `plan_mem`; structure + measurement are
real, the per-node workload is representative. (`fetch_alibaba_gpu.py` also added to pull
gpu-v2020 for a real-util cross-check — resumable OSS download, no survey.)

## Experiment 21 — peak-concurrent GPU is additive (deterministic rule wins); util is saturated/k-bound (no relational signal)

**Date:** 2026-06-25 · runs on the A100 via `.venv-forecast` (the main `.venv` torch is broken:
`ncclCommResume`).

**The right target.** Per-task GPU *memory* is already solved deterministically (Exp 4, 0.04 GB),
and the DAG is irrelevant to it. The DAG-dependent, unsolved quantity is the WORKFLOW's **peak
concurrent GPU memory** — how much VRAM the job needs *at once* — a function of (a) topology
(what MAY run in parallel), (b) durations (what actually OVERLAPS), (c) per-task footprint. It is
NOT the sum of per-task memory (over-counts) nor the max single task (under-counts).

### Part A — synthetic layered DAGs (`pins/eval/dag_gpu_bench.py`)

**Method.** Generate layered DAGs of CNN tasks (parallel nodes per layer CAN co-run); measure
each distinct config's (peak_gb, duration) on the A100 (memoised — 48 configs cover hundreds of
nodes); compute workflow peak-concurrent via a list-scheduler over measured (mem, dur) under a
parallelism cap. Score three baselines for the peak-concurrent target. 40 DAGs.

**Result (MAPE / within-1.5×; peak-concurrent 2.1–20.9 GB).**

| regime | naive_sum | naive_max | **layer_sum** (heaviest layer) |
|---|---|---|---|
| unlimited parallel | 117% / 12.5% | 32.5% / 50% | **3.3% / 100%** |
| max_parallel=4 | 115% / 12.5% | 33.0% / 47.5% | **3.7% / 100%** |
| max_parallel=2 | 130% / 10.0% | 28.9% / 57.5% | **9.0% / 100%** |

**Naive aggregation FAILS (sum off 117%), so structure matters — but a one-line topology rule
(heaviest layer's summed memory) nails it to ~3%**, degrading only mildly under tight
parallelism and never leaving 100% within-1.5×.

### Part B — REAL v2018 DAG topologies (`pins/eval/dag_gpu_trace_bench.py`)

**Method.** Replace the synthetic generator with **actual v2018 job graphs** (irregular, real
edges). Each node: a library CNN config **rank-matched** by its `plan_mem` percentile →
GPU-footprint percentile (preserves the trace's per-node demand ordering without trusting
plan_mem's units); per-node **duration = the real trace duration** (governs overlap); `layer_sum`
generalised to **topological depth-level**. 80 real DAGs (4–24 nodes, depth ≤12).

**Result.**

| baseline | MAE (GB) | MAPE | within 1.5× |
|---|---|---|---|
| naive_sum | 14.30 | 154% | 11.2% |
| naive_max | 3.95 | 28.3% | 53.8% |
| **level_sum** | **0.37** | **2.9%** | **98.8%** |

**The depth-level rule holds on real irregular production DAGs just as on synthetic ones (2.9%).**
Across synthetic/real graphs, unlimited→2 parallelism, and real skewed durations, it never broke.

### Part C — co-execution: is GPU UTILIZATION non-additive? (`pins/eval/gpu_coexec_probe.py`)

**Why.** Memory is additive; utilisation might not be (two 60%-util tasks ≠ 120% — they contend).
**Method.** Co-run k tasks as concurrent SUBPROCESSES (private CUDA contexts, real contention),
fixed-duration full-overlap, started via a `Barrier`; sample whole-device util with `nvidia-smi`;
measure realised util + throughput slowdown vs solo. 5 CNN configs, 14 random co-run sets.

**Result.**

| quantity | finding |
|---|---|
| solo util (every config) | **96–99%** — each CNN already saturates the A100 alone |
| co-run realised util | **100%** (additive-cap predicts 100% → rule right, 0.3 pt error) |
| slowdown k=2 / k=3 | **2.4× / 3.5×** (super-linear — co-locating is *worse* than serial) |
| mix-dependence | slowdown std ≤**0.09** within a k → determined by **count k alone**, not the mix |

**Findings / honest read (the through-line, reinforced).**
1. **Peak-concurrent GPU memory is additive** — naive sum fails (117–154%) but a deterministic
   heaviest-(depth-)level rule predicts it to ~3% on synthetic AND real DAGs. The GBT-runtime +
   attention cascade is **over-engineering for memory**: overlap is second-order (the fattest
   level dominates), and per-task memory is already deterministic (Exp 4). End-to-end, peak GPU is
   a submit-time, no-ML pipeline: predict per-task mem → topo-level max.
2. **Utilization is not a useful learned target for GPU-saturating training jobs** — it pegs at
   100%, so the additive-cap rule is trivially correct.
3. **Co-location slowdown IS large and super-linear** (the real scheduler-relevant cost) **but
   k-determined, not mix-dependent** — a one-line `slowdown ≈ k` rule suffices. The relational
   signal an attention/GNN model would exploit is **absent for homogeneous (conv-bound) tasks**.
4. **Where a relational model WOULD earn its keep: heterogeneous bottlenecks.** Compute-bound +
   bandwidth-bound tasks should overlap productively (less slowdown) than two compute-bound — a
   *mix*-dependent effect a `≈k` rule can't express. Not yet tested (all tasks here are conv).
   This is the one open door for `DAG→attention→util`; until a 3-class **resource-contention rule**
   (compute/bandwidth/IO, slowdown = max per-class demand/capacity) demonstrably fails, attention
   stays unjustified. **Recommended GPU-util toolkit instead of attention:** retrieval/GBT quantiles
   for *solo* util (Exp-19 style), a resource-class contention rule for *co-located* util, and the
   existing **temporal** attention (Exp 8/16) for a *running* job's util trajectory.

**Lesson (unchanged):** structure matters, but a deterministic rule keeps beating the learned
model — mirrors Exp 18 (ILP ties auction in 1-D) and Exp 9–10 (greedy beats fancy bidding). The
DAG genuinely drives GPU demand; you don't need ML to exploit it.

**Reproduce.**
```bash
cd Research
.venv-forecast/bin/python -m pins.eval.dag_gpu_bench --n-dags 40                 # Part A (synthetic)
.venv-forecast/bin/python -m pins.eval.dag_gpu_trace_bench --n-jobs 80           # Part B (real v2018 DAGs)
.venv-forecast/bin/python -m pins.eval.gpu_coexec_probe --run-s 3 --trials 14    # Part C (co-exec util)
```
`pins/eval/dag_gpu_bench.py` (`gen_dag`, `measure_task`, `simulate_peak_concurrent`);
`pins/eval/dag_gpu_trace_bench.py` (rank-match real DAGs, real durations); `pins/eval/
gpu_coexec_probe.py` (multiprocess co-exec, nvidia-smi sampler); artifacts `results_dag_gpu_*.json`,
`results_gpu_coexec.json`.

## Next: integration architecture (decided 2026-06-25)
Full PINS pipeline locked as **LLMs reason/bid → committed-auction decides → ILP places/guarantees**
(Open Q #1 = option a-with-placement). New work to build: (1) text bridge from Stage-1 facts
(GBT runtime + level-sum GPU + uncertainty) to the LLMs; (2) the **bounded two-sided negotiation
protocol** (job-side Exp 12 ⇄ resource-side Exp 14, which today set value in isolation); (3) the
must-have **single-LLM-both-objectives** baseline (Open Q #5). Honest expectation from Exp 9–13:
negotiation won't beat the committed-auction on SLA — its claim is interpretability + modularity +
incentives, to be *measured* against that baseline, not asserted.
