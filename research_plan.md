# Research Plan: PINS — Prediction-Informed Negotiated Scheduling for Elastic HPC

*A 6-month plan for building and evaluating an LLM-prediction + agent-negotiation system for dynamic GPU allocation on a supercomputer.*

> **Status note (2026-06-06).** Stage-1 prediction has an active, measured sub-project — a
> **task-classified, RAG-augmented resource predictor**. Its empirical findings to date are in
> §5 and the running log `research_progress.md`; its concrete plan is §6. Stage-2 negotiation
> (the headline contribution) and the 6-month structure are unchanged.

---

## 1. The Idea in Plain Language

### The problem
A training job doesn't need the same number of GPUs the whole time. During data prep it barely uses the GPU; during heavy training it wants a lot; during evaluation, somewhere in between. If you give it a **fixed** number of GPUs the whole run, you waste GPUs when it's idle and starve it when it's busy.

**Goal:** give each job the GPUs it needs *at each moment*, so the whole supercomputer gets more work done.

> Note on "performance": this improves **efficiency, throughput, and utilization** — *not* the model's accuracy. The trained model's quality stays the same; we just waste fewer GPU-hours and finish sooner.

### The method (like a marketplace)
1. **Predict** what each job will need over time.
2. **Negotiate** who gets the GPUs, and adjust on the fly.

**One-line takeaway:**
> An LLM guesses each new job's GPU needs, agents negotiate who gets GPUs at each moment via a fast auction, and the system shifts GPUs around live — so the supercomputer wastes less and finishes more, without hurting model quality.

---

## 2. The Method (PINS), Component by Component

```
   Job submission (script, config, code, history)
            │
   ┌────────▼─────────┐   Stage 1: HYBRID PREDICTOR
   │  Numeric forecaster (warm jobs, has history) ──┐
   │  Classified RAG profiler (cold start)        ──┤→ predicted demand
   │   classify task → retrieve class facts-schema  │   profile per phase
   │   → LLM extracts facts → code computes curve    │
   └──────────────────┘                             │
            │  (per-job time-varying demand curve)   │
   ┌────────▼─────────┐   Stage 2: NEGOTIATION (the star)
   │ Job-agents bid ──► fast auction/bargaining ──► allocation deltas
   │   (LLM forms utility at submit/phase change;     │
   │    mechanism clears in ms — LLM NOT in hot loop) │
   └──────────────────┘                               │
            │  (who gains/releases which GPUs)         │
   ┌────────▼─────────┐   ELASTIC ACTUATION
   │ TorchElastic rescale + batch-size/LR co-adapt    │
   │ (rescale-cost-aware, anti-thrashing, SLURM-layer)│
   └──────────────────┘
```

### Stage 1 — Hybrid Predictor (LLM only where it has a moat)
- **Warm jobs (have history):** cheap, fast numeric forecaster (TCN / small transformer, or a time-series foundation model like Chronos) and/or **structured k-NN over past measured runs**. Accurate, no GPU tax.
- **Cold-start jobs (no history):** a **task-classified, RAG-augmented profiler** (§6). The job is classified into a workload class; RAG retrieves that class's *facts schema + domain knowledge*; the LLM reads the submission and emits **structured facts** for that class (e.g. for NN training: per-layer shapes, precision, optimizer); **deterministic code computes the numbers** and the per-phase demand curve.
- **Key rule:** the LLM runs **only** at job submission, plus once per detected phase transition — never per scheduling tick.
- **Output:** a per-job *time-varying demand curve* (peak memory → feasibility; marginal-value over GPUs → the bid curve) with phase boundaries.

> **Empirically grounded design rule (see §5).** We measured that a zero-shot LLM *number* is
> unusable (over-predicts CNN peak memory 7×–170×, even at 14B), but the LLM emits *structured
> facts/shapes* reliably. So the LLM **reasons/extracts; deterministic code computes.** This is
> not a stylistic choice — each time we moved a *number* from the LLM into code, error dropped
> ~an order of magnitude.

### Stage 2 — Prediction-Informed Negotiation (the contribution)
- **Agents:** one job-agent per running malleable job; a clearing mechanism (auctioneer) per GPU pool.
- **Valuation:** each agent's bid = *expected goodput gain* from getting one more (or releasing one) GPU in its current/predicted phase, computed from the Stage-1 curve.
- **Mechanism:** an **event-triggered sealed-bid auction** (or alternating-offer bargaining) that clears in milliseconds and has principled properties (efficiency, no starvation via a fairness term).
- **Triggers:** negotiation fires on **events** — job arrival/exit or a predicted **phase transition** — not every tick. Kills overhead and thrashing.
- **Anti-thrashing:** minimum-hold time + hysteresis; only rescale if predicted goodput gain exceeds the rescale cost.

> **Design hinge:** LLMs *reason and explain*; a fast mechanism *decides*. This resolves the latency, non-determinism, and "no-guarantees" objections in one stroke. The existing MCP agent prototype (`pins/`) is the natural substrate for the agent layer.

### Elastic Actuation (HPC-aware)
- Rescale via **TorchElastic** (checkpoint → re-shard → resume).
- **Co-adapt batch size + learning rate** on rescale (optimize *goodput* = throughput × statistical efficiency, à la Pollux) so model quality is preserved.
- **Rescale-cost model:** every reallocation has a measured cost; only act when predicted benefit beats it.
- **SLURM coexistence:** run PINS *as a layer above* the batch scheduler (manage a pool SLURM hands you), so we don't fight the site scheduler.

---

## 3. Scope (state explicitly)

- **Workloads:** the **malleable** subset of HPC — DL training (TorchElastic/Horovod Elastic), HPO sweeps, workflow pipelines. **Not** rigid monolithic MPI jobs.
- **Stage-1 predictor scope (v1):** **NN-training jobs only** (CNN / ResNet / Transformer). The design is class-pluggable (§6) so other workload classes (CFD, MD, …) can be added later without touching the classifier core or the negotiation layer — but they are explicitly **out of scope for v1** and only sketched.
- **Resource:** GPUs (extendable to GPU+memory bundles later).
- **Objective:** goodput + utilization + fairness + job completion time — explicitly **at equal final model quality**.

---

## 4. Why This Is a Real Contribution (not a two-word swap)

1. **Cold-start prediction the original paper can't do** — attention-on-history is blind to a brand-new job; our profiler reads its script. Sharpened: **task-classified RAG extraction** (§6, H2) makes this robust across heterogeneous submission formats, and our measurements show *extraction-then-compute* beats both raw-LLM and naive heuristics. This is the concrete capability gap.
2. **Decentralized, prediction-informed negotiation** vs. a centralized allocator — scales, surfaces private per-job valuations, handles conflicting deadlines.
3. **HPC-specific realism** — non-preemption-aware, rescale-cost-aware, batch-size-coupling-aware. Most elastic-DL papers ignore at least one.

---

## 5. Stage-1 Findings So Far (measured, NN training)

All on a single **A100-PCIE-40GB**, ground truth = `torch.cuda.max_memory_allocated()`. Harnesses:
`pins/eval/predict_cnn.py`, `pins/eval/predict_arch.py`. Full detail + tables in
`research_progress.md`. The arc — each step moved one *number* out of the LLM into code:

| Method | What the LLM did | Peak-mem MAE | Verdict |
|---|---|---|---|
| Raw LLM number (3b→14b) | emit GB directly | 283 → 24 GB | **fail** — over-predicts 7–170×, no size fixes it; 7b inverts ranking |
| Hybrid, guessed facts | emit `{trainable_frac, bytes, optim_mult, activation_MB}` | 3.9 GB | partial — facts right & size-robust, but guessed activation too low |
| Hybrid, 14b reasoning | walk layers in text | 2.6 GB | partial — shapes 100% correct, but inconsistent counting flips ranking |
| **Deterministic: LLM-shapes → code-sum** | emit per-layer **shapes** | **0.04 GB** | **PASS** — 100% within 1.5×, beats heuristic ~40× |

**Robustness checks.**
- **Mixed precision:** survives fp16/bf16 (MAE 0.06 GB, 100% within 1.5×); margin over the
  precision-blind heuristic *widens* (memory ~halves, heuristic doesn't track it).
- **Across architectures (one global calibration):** generalizes to **ResNet skip-connections**
  (per-family MAE 0.37 GB) and short-context transformers; overall still PASS (MAE 0.49 GB).
  The params-heuristic's ranking goes **negative** across architectures (params anti-correlate
  with memory) — activation-awareness is essential, not a luxury.
- **Known boundary:** the architecture-agnostic activation extractor (forward-hook on module
  *outputs*) misses a transformer's internal **attention scores** (∝ `batch·heads·seq²`), so it
  under-predicts long-context training (seq 1024: 1.99 vs 3.52 GB). Fix scheduled in §6 P1.

**Takeaways that drive §6:** (a) keep the LLM out of the arithmetic; (b) the LLM's reliable
contribution is **structured extraction** (facts/shapes), which is exactly what RAG can make
*class-adaptive*; (c) a small set of deterministic *resource primitives* (feature maps,
attention scores, optimizer states) is the irreducible domain knowledge to encode by hand.

---

## 6. Stage-1+ : Task-Classified, RAG-Augmented Resource Prediction

**Premise.** Resource drivers are class-specific. Building on §5 (extract-then-compute works for
NN training), generalize it with a front-end that makes *extraction* class-adaptive:
**classify the job → RAG-retrieve that class's facts-schema + domain knowledge → LLM extracts
structured facts → deterministic per-class model computes the resource curve.** v1 covers
**NN-training only**; the design is a class-pluggable registry so future classes drop in.

### Hypotheses
- **H1 (routing).** A classifier assigns the workload class from the submission with high
  accuracy and a safe `unknown` fallback.
- **H2 (extraction — HEADLINE).** **Class-conditioned RAG extraction** yields more complete and
  accurate structured facts than a single generic prompt, especially across heterogeneous
  submission formats. *This is the primary novel claim.*
- **H3 (prediction).** Class-specific extract-then-compute beats the mean and params-heuristic on
  peak-memory MAE with within-1.5× ≥ 80% (extends the §5 single-family result).
- **H4 (warm jobs, stretch).** Structured k-NN retrieval over past *measured* runs further
  reduces error vs formula-only.

### System design
```
submission → [Classifier] → class (+confidence)
          → [RAG: retrieve class facts-SCHEMA + domain knowledge (+ warm neighbors)]
          → [LLM: fill structured facts JSON]                ← "reason / extract"
          → [Deterministic per-class resource model + calibration]   ← "decide / compute"
          → peak-mem + marginal-value curve + retrieved-evidence justification
                                   ↓
                    existing job_agent / auctioneer (Stage 2, untouched)
```

### Components & decisions
- **Class registry.** Each class is a plug-in: `{facts schema, retrieval corpus, resource
  formula, calibration constants}`. Adding a workload = a registry entry, not core surgery. v1
  ships the **NN-training** class (sub-types CNN / ResNet / Transformer).
- **Classifier.** v0 rule-based (framework, file types, imports) → v1 LLM only if rules plateau.
  Confidence threshold; `unknown` → conservative estimate or a short **profiling probe** (reuse
  `measure_peak_gb`). Decisions logged/auditable.
- **RAG (H2 focus).** Per-class corpus = the *extraction template* + "how to read this kind of
  submission" + scaling knowledge. Retrieval grounds the LLM in *what to extract* for the class
  (e.g. NN → per-layer shapes, precision, optimizer). Local embeddings via Ollama; offline;
  **must degrade gracefully** (fall back to a generic schema / heuristic if RAG/LLM is down).
  *Scope guard:* this is text-RAG for **schema/knowledge grounding**; **warm-job history uses
  structured k-NN** (a DB query on numeric metadata), not a vector store.
- **Resource model.** NN formula already built (`predict_arch.py`); P1 adds the attention term
  (`+ batch·heads·layers·seq²·bytes`) to close the long-context gap. Calibration via LOOCV /
  regression on measured runs (built).
- **Invariant.** RAG/LLM *reason*; code *computes the number*. Retrieved evidence = the
  human-readable justification (valuable for a scheduler users must trust). RAG must **never**
  produce the final GB/curve directly — that is the §5 Experiment-1 failure with extra steps.

### Evaluation
- **Data.** Extend `benchmark.json` with `class` labels + a measured corpus (we already have
  CNN/ResNet/Transformer truth from §5; grow it). Hold out for LOOCV.
- **Metrics.** Classifier accuracy + confusion matrix; **field-level extraction accuracy** vs
  ground-truth facts (the H2 metric); end-to-end mem MAE / MAPE / within-1.5× / Spearman.
- **Ablations.** generic vs class-conditioned extraction (H2); formula-only vs +retrieval (H4);
  LLM size; classifier on/off.

### Phases & gates
| Phase | Deliverable | 🚦 Gate |
|---|---|---|
| **P0 ✅** | Single-class extract-then-compute validated; arch generalization + boundary (§5) | met (MAE 0.04 GB; beats heuristic) |
| **P1** | NN class registry + close long-seq attention term | one global `(a,b)` fits all NN incl. seq 1024 |
| **P2** | Classifier v0 + labeled eval harness | **A:** ≥90% routing acc on held-out + safe fallback |
| **P3** | Class-conditioned RAG extraction + ablation | **B (headline, H2):** beats generic extraction on field accuracy |
| **P4** | End-to-end predictor (NN sub-types) | **C (H3):** beats heuristic+mean, within-1.5× ≥ 80% |
| **P5** | Warm-job k-NN retrieval | **D (H4):** retrieval adds measurable accuracy, else drop it |
| **P6** | Emit value curves into `pins/predictor.py`; wire to negotiation demo | prediction-informed bids end to end |

### Risks (predictor-specific)
- **Misclassification compounds** → confidence threshold + `unknown`→profiling probe.
- **RAG over-engineering** → every retrieval step must pass its ablation gate or be cut;
  structured k-NN for warm jobs, text-RAG only where it provably beats the formula.
- **Principle drift** → never let RAG/LLM emit the final number.
- **Non-NN data scarcity** on a single A100 → v1 stays NN-only; other classes are design-only.

---

## 7. Six-Month Plan

**Strategy:** simulation-first (cheap, fast, thousands of experiments) + a small real-hardware validation near the end. Decision gates let you fail fast.

| Month | Phase | Main deliverable |
|---|---|---|
| 1 | Foundation & premise check | Evidence demand varies + locked problem statement |
| 2 | Simulator & baselines | Working sim with baseline numbers |
| 3 | Stage 1 — Prediction | **Task-classified RAG predictor (NN), §6 P1–P4** |
| 4 | Stage 2 — Negotiation | Full PINS running end-to-end in sim |
| 5 | Evaluation & real test | Complete results + ablations + A100 validation |
| 6 | Write-up & buffer | Submitted paper / thesis chapter |

**Realistic target output:** a workshop paper or thesis chapter. A top-tier conference paper solo in 6 months is ambitious but possible if Months 1–4 go smoothly.

### Month 1 — Foundation & premise check
**Goal:** prove the idea is worth building before building it.
- Profile real workloads — log GPU memory/utilization over time through a real training run (preprocess → train → eval). *Does demand actually go up and down?*
- Read & position key papers: original attention paper, Pollux, Gandiva, TorchElastic, one time-series-foundation-model paper, one "LLM for time series" paper, and IR/RAG basics for H2. Write 2 pages on differences.
- Lock the 3 scope decisions: competitive vs cooperative agents; GPUs only vs bundles; simulation-only vs real validation.

**Deliverable:** profiling plots + 2-page related-work + one-paragraph locked problem statement.
**🚦 Gate:** if demand is basically flat → pivot to a **multi-job** framing (many jobs sharing a cluster) instead of single-job phases. Decide now, not in Month 4.

### Month 2 — Simulator & baselines
**Goal:** something to measure against.
- Build a **trace-replay simulator**: jobs arrive over time, each with a demand curve, GPUs get allocated.
- Get public traces: Parallel Workloads Archive, Microsoft Philly, or Alibaba GPU traces.
- Implement baselines: static allocation, SLURM-style backfill, simplified Pollux.
- Define metrics in code: goodput, GPU utilization, job completion time, fairness.

**Deliverable:** simulator that prints baseline numbers to beat.
**🚦 Gate:** baselines run and produce sensible numbers.

### Month 3 — Stage 1: Prediction (task-classified RAG predictor, NN)
**Goal:** predict each NN job's needs from its submission, accurately and cold-start.
- Execute §6 **P1–P4**: NN class registry + attention term (P1); classifier v0 (P2);
  class-conditioned RAG extraction + ablation (P3, the H2 headline); end-to-end vs baselines (P4).
- Warm-job numeric forecaster / k-NN as available (P5 if time).
- Measure prediction accuracy vs. the attention baseline — especially the cold-start win.

**Deliverable:** the predictor + accuracy/extraction tables + ablations (extends `research_progress.md`).
**🚦 Gate:** Gate C (§6) — predictor beats "no prediction" and the params heuristic for new jobs.
*(Already provisionally met for the single-class case in §5; Month 3 generalizes it.)*

### Month 4 — Stage 2: Negotiation (the core)
**Goal:** agents negotiate, system reallocates.
- Implement the auction mechanism: agents bid on predicted goodput gain; mechanism clears fast.
- Reuse the MCP agent prototype (`pins/`) for the agent layer.
- Wire together: prediction → negotiation → allocation, event-triggered (phase change / job arrival).
- Add anti-thrashing (min-hold time) and a rescale-cost gate.

**Deliverable:** full PINS running end-to-end in simulation, first results vs. baselines.
**🚦 Gate (most important):** PINS beats at least one strong baseline on goodput or utilization. Go/no-go for the whole paper.

### Month 5 — Evaluation & real validation
**Goal:** turn results into evidence.
- Full experiments vs. all baselines across several traces.
- Ablations: classified-RAG extraction on/off (H2); LLM cold-start on/off; negotiation vs. central allocator; event-triggered vs. periodic; rescale-cost on/off.
- Small real test on the A100: 2–4 jobs with TorchElastic; confirm it rescales and that model accuracy is unchanged.

**Deliverable:** complete results tables + ablations + one real-hardware demo.

### Month 6 — Write-up & buffer
**Goal:** ship it.
- Write the paper (intro, method, eval), polish figures, clean code for reproducibility.
- ~1.5 weeks of buffer for overruns from earlier months.
- Pick the venue and submit / hand in the thesis chapter.

**Deliverable:** submitted paper or completed thesis chapter.

---

## 8. Evaluation Plan

- **Baselines (must beat, not just coexist):** original attention paper's method, SLURM backfill, Pollux, static allocation; for Stage-1: mean (no-prediction), params-heuristic, raw-LLM, generic (un-classified) extraction.
- **Metrics:** goodput, GPU utilization, average/95p job completion time, fairness (e.g., finish-time fairness), **plus a model-quality check** (final accuracy ≈ baseline). For Stage-1: peak-mem MAE/MAPE/within-1.5×/Spearman, classifier accuracy, field-level extraction accuracy.
- **Ablations:** classified-RAG extraction on/off; LLM cold-start on/off; negotiation vs. centralized allocator (same predictor); event-triggered vs. periodic; with/without rescale-cost awareness.
- **Setup:** simulation first (trace replay), then small real validation on the A100.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM prediction wrong → bad allocation | Hybrid: numeric forecaster dominates once history exists; LLM only cold-starts; **LLM never emits the number** (§5) |
| LLM/negotiation latency | Event-triggered, coarse granularity; mechanism (not LLM) in the hot loop |
| Rescale thrashing | Hysteresis + min-hold + rescale-cost gate |
| Batch-size changes hurt accuracy | Goodput + LR co-adaptation; report final accuracy |
| Task misclassification → wrong predictor | Confidence threshold + `unknown`→profiling probe (§6) |
| RAG over-engineering / weak retrieval | Ablation-gated; structured k-NN for warm jobs; text-RAG only if it beats the formula |
| "Delta too small" | Lead with cold-start capability gap + decentralization + the measured extract-then-compute result |
| Workload demand doesn't actually vary | Month-1 premise check; pivot to multi-job framing if flat |

---

## 10. Staying on Track

- **Weekly:** one measurable result (a plot, a number, a working component).
- **Monthly:** hit the gate before moving on — don't carry a broken stage forward.
- **If behind:** cut the real A100 validation and the bundle/combinatorial extension first. The simulation results + ablations are the publishable core.

---

## 11. Open Decisions

**Resolved (2026-06-06):**
- **Stage-1 v1 scope:** NN-training only (class-pluggable for later). *(§3, §6)*
- **Stage-1 headline contribution:** task-classified **RAG extraction** (H2). *(§6)*

**Still open:**
1. **Competitive or cooperative agents?** Auction (mechanism design) vs distributed optimization. Changes Stage 2's math.
2. **Simulation-only thesis, or real-cluster validation?** Decides TorchElastic plumbing investment.
3. **GPUs only, or bundles (GPU + memory + interconnect)?** Bundles → combinatorial auction (more novel, more work).

---

## 12. Immediate Next Step

Stage-1 has measured momentum (§5). The concrete next actions are §6 **P1** (NN class registry +
close the long-context attention term so one calibration fits all NN sub-types) and **P2**
(classifier v0 + labeled eval harness), leading into the **H2 headline** ablation (P3). In
parallel, keep the Month-1 **premise check** (does demand vary over a real run?) honest — it
gates the whole negotiation story.

---

## Key References to Read

- Original paper: *Attention-based workload prediction and dynamic resource allocation for heterogeneous computing environments.*
- **Pollux** (OSDI '21) — goodput-based elastic DL scheduling; co-adapts allocation + batch size/LR.
- **Gandiva**, **Optimus**, **AntMan**, **Tiresias** — elastic GPU schedulers for DL clusters.
- **TorchElastic** / **Horovod Elastic** — the mechanism for rescaling a training job mid-run.
- Time-series foundation models: **Chronos**, **TimesFM**, **Moirai**, **Lag-Llama**, **TimeGPT**.
- Retrieval / RAG (for §6 H2): the RAG paper (Lewis et al.), dense retrieval (DPR), plus
  structured-feature k-NN baselines for the warm-job path.
- Market-based scheduling background: Mirage (combinatorial auction), Nimrod/G, Tycoon.
- Trace sources: Parallel Workloads Archive, Microsoft Philly trace, Alibaba GPU trace.
