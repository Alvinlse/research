# Progress Report
**AI-Agent Negotiation for Dynamic Resource Allocation in HPC**

Lay Kim Seng — 2026 / 06 / 19
Cyberscience Center, Tohoku University

> *Rev. 2026-06-19:* updated to the **two-sided (demand/supply) negotiation** design developed
> in [`CLAUDE.md`](./CLAUDE.md) after reading *"LLM-Driven Adaptive Cloud Resource Scheduling"*
> (IEEE OJ-CS 2026). The single-orchestrator framing is superseded by a **demand-LLM ⇄
> supply-LLM** negotiation cleared by an auction and guaranteed by an ILP.

---

## Research Topic

# Breaking the Utilization–Service-Quality Tradeoff in HPC via AI-Agent Negotiation

> **Focus.** The contribution is the **two-sided AI-agent negotiation layer**: a **demand
> agent** (job queue) and a **supply agent** (resource pool) with *deliberately asymmetric*
> objectives reason over cluster state and exchange valued, justified offers. A deterministic
> **auction clears** the negotiation and an **ILP guarantees** feasibility against live state.
> Uncertainty-aware prediction is a **supporting co-contribution** — it sizes each bid's
> safety margin, but the negotiation is what allocates.

---

## Background & Problem

### The orchestration gap
Modern AI / scientific workloads are bursty and heterogeneous, and need a more intelligent,
runtime resource-management layer than static schedulers provide.

### The core tension (unchanged)
With current methods, pushing resource utilization higher also pushes the **SLA-violation
rate** up. Utilization and service quality trade off against each other.

> **Why it's hard:** production HPC schedulers still do not exploit **malleability** (resizing
> jobs at runtime), even after 20+ years of research — supporting it touches the entire HPC
> software stack (Tarraf et al., IEEE TPDS 2024). Closing the utilization–SLA gap requires a
> smarter allocation *policy*, not just more capacity.

### The key insight — contention over safety margin
At high utilization the system **cannot grant every job its safety margin** — aggregate demand
exceeds capacity. The decision that actually determines service quality is therefore *"whose
safety margin gets cut?"* Classical schedulers ration that margin blindly. **This research
rations it by negotiation between the side that wants the margin (demand) and the side that
must protect headroom and fairness (supply).**

### The reason-then-guarantee spine (from the reference paper)
The LLMsched paper pairs an **LLM that proposes** schedules with a **lightweight ILP that
repairs** them to satisfy hard constraints — *the LLM is smart but unreliable; the ILP is
reliable but not smart.* We keep that spine but **split the single LLM proposer into a
two-sided negotiation**, because a single LLM holding both objectives collapses the
urgency-vs-utilization-vs-fairness tension *invisibly*; two agents make it **explicit and
auditable**. That transcript is the interpretability edge over RL.

---

## Research Goal

**1. Allocate by two-sided negotiation (the contribution)**
A demand agent and a supply agent turn their private state into reasoned offers; a
deterministic mechanism (auction) clears the negotiation and an ILP reconciles it to live
state — rationing scarce resources to the jobs that value them most *without* letting either
LLM make the final, unguaranteed decision.

**2. Predict workload (co-contribution)**
Forecast each task's resource demand over the next T timesteps *with an explicit uncertainty
estimate*, which sizes the safety margin the demand agent bids for.

**Target outcome (core claim):** raise resource utilization while keeping the SLA-violation
rate **competitive-or-better than every baseline** — classical, learning-based, and the
single-LLM proposer — *plus* an auditable justification for every allocation that RL cannot
offer.

**Governing design rule:** *the LLMs reason / explain; deterministic code decides.* The
deciders are the **auction** (clears the negotiation) and the **ILP** (guarantees
feasibility) — never the free-form LLM chat.

---

## Why Two-Sided Negotiation? — positioning vs. related work

### 1. LLM-Driven / Learning-Based Scheduling, incl. the reference paper (RL, LLMsched)
RL and LLM-driven schedulers learn an allocation policy and can optimize utilization/SLA. The
reference paper adds an ILP guarantee to a **single centralized LLM proposer**.

**Gap we address.** *RL schedulers are black boxes — an operator cannot see why a job was
shrunk — and generalize poorly to unseen workloads. The single-LLM proposer hides the
demand-vs-supply tradeoff inside one prompt: you cannot inspect, swap, or audit the supply
side's policy independently of the demand side.*

**Our edge.**
- **Interpretability** — every allocation ships with both agents' offers and natural-language
  justifications, so the urgency-vs-utilization tradeoff is explicit and auditable.
- **Modularity** — swap the resource-side policy (greener / fairer operator) by editing *one
  agent's objective prompt*, with no retraining.
- **Incentive structure** — a well-designed clearing rule (uniform-price / payments) can make
  honest bidding optimal; ad-hoc LLM chat cannot.

### 2. Malleability in Modern HPC Systems (Tarraf et al., 2024)
Survey of runtime job resizing across the HPC stack; shows malleability raises throughput &
utilization but is unused in production.

**Gap we address.** *Surveys mechanisms, but offers no prediction-driven policy for deciding
when and how much to resize.*

---

## Proposed System Overview

The negotiation never decides alone. Two LLMs **reason**; a structured mechanism and an ILP
**decide and guarantee** — keeping the project's hinge intact and avoiding "three deciders."

```
Demand-LLM  ─┐
 (job queue) │
             ├─ negotiate (reasoned offers + justifications) ─► auction ─► ILP
Supply-LLM  ─┘   bounded rounds, time-boxed, best-effort       (clears)   (feasibility /
 (resource pool)                                                           repair vs. live state)
```

**Deliberate asymmetry (the make-or-break design choice).** Two agents negotiating is only
meaningful if they want / know different things; symmetric objectives make the discussion
theater.

| | **Demand agent** (job queue) | **Supply agent** (resource pool) |
|---|---|---|
| Objective | minimize JCT, hit deadlines, get jobs placed | maximize utilization, protect fairness, keep headroom for future arrivals |
| Private info | each job's urgency, true value, deadline flexibility | real capacity, predicted load, preemption cost |
| Pressure it applies | "concentrate resources on my deadline job **now**" | "reserve headroom / cap monopolization for jobs still to arrive" |

### System shape (selective, amortized, fallback-protected)
- **Fast path (~97%):** cheap heuristic / cached strategy, milliseconds. Most decisions never
  invoke an LLM.
- **Escalation gate (two-dimensional):** escalate a queued job to negotiation only when
  `waited_long_enough AND valuable_enough` — *waiting* makes the LLM cost **affordable** (it
  is hidden behind wait time already incurred and signals the cheap path failed *this* job);
  *value / urgency / deadline* makes it **worthwhile** (long waits skew toward low-priority
  jobs, so waiting alone is not enough).
- **Two-LLM negotiation:** time-boxed, **best-effort**, emits a proposed plan + a
  justification transcript. Bounded rounds of offers — *not* open chat — so termination is
  provable.
- **Mechanism (auction):** clears the negotiated offers deterministically.
- **ILP:** reconciles the (slightly stale) plan to current cluster state and guarantees hard
  constraints (capacity, precedence, affinity).
- **Fallback:** on timeout or non-convergence, drop to the cheap heuristic — negotiation can
  **only help or be neutral, never stall** the job.

> **Thesis sentence:** *the LLMs reason about value under uncertainty; the auction decides the
> allocation; the ILP guarantees feasibility.* Each layer does a job no other layer can.

---

## Methodology — Workload Prediction (co-contribution)

**Input:** task code + runtime parameters + historical run traces.
**Output:** predicted resource demand over the next T timesteps, as a distribution.

1. **LLM metadata extraction.** A local LLM (qwen2.5 / LLaMA) reads source code to extract
   semantic features static analysis misses. Run offline / once per job — out of the
   latency-critical loop.
2. **Temporal–spatial attention.** Temporal = workload over time; spatial = across nodes &
   co-located jobs that contend for shared resources. Multi-head attention forecasts load.
3. **Quantile regression.** Outputs prediction intervals (P10 / P50 / P90), not a point.
   **This interval sizes the demand agent's safety margin** — the bridge from prediction to
   negotiation. *Uncertainty sizes the margin; the auction rations it.*
   **Built & measured (Exp 16):** a pinball-loss quantile head on the Exp-8 forecaster (softplus
   widths → no quantile crossing) keeps the P50 accuracy gate (nMAE 0.066 vs 0.072) and is
   **conformally calibrated** (split-conformal/CQR: aggregate coverage 0.67 → 0.75 toward nominal
   0.80). The per-job width feeds `predictor.marginal_values(uncertainty=…)`; the demand-side
   ablation shows the uncertainty-sized margin is **insurance whose value grows with tail severity**
   — it ties the point forecast on mild demand but cuts prod-SLA ~35% under heavy tails, while a
   fixed/blanket margin backfires. The co-contribution is now real end-to-end, not stubbed.
   **The LLM demand agent now makes the hedge call (Exp 17):** from `(uncertainty, deadline,
   contention, tier)` it emits a categorical hedge (none/some/heavy) + justification; code owns the
   GPU count. As with the supply agent (Exp 14C), this is a judgement with stakes, so **model size
   matters** — 3b over-hedges and hurts SLA (auditably misreading "high contention" as spare
   capacity), 14b matches/edges the oracle at mild tails. A heavy-tail failure (14b initially lost,
   because a contention-gate from the mild regime suppressed needed margin) was fixed by **adding a
   spike-risk signal** — not a bigger model: with it, 7b/14b now **beat** the deterministic margin in
   BOTH regimes (a clean "fix the decision, not the LLM" demonstration, echoing Stage-1).

---

## Methodology — Resource Allocation (the contribution)

### Demand agent (job side)
Reasons over each job's predicted workload, deadline pressure, and priority tier to form a bid
= **(resource amount incl. an uncertainty-sized safety margin, a value reflecting how badly a
shrink would hurt its SLA, and a short justification)**. Higher predicted uncertainty → larger
requested margin; tighter deadline / higher penalty → higher bid value. Per the project's
hard-won lesson, the LLM emits **categorical, justified strategy** (stance / priority class /
focus), never a calibrated magnitude — deterministic code turns that into the bid curve.

### Supply agent (resource side)
Reasons over real capacity, predicted future load, and fairness state to **push back**:
reserve headroom for anticipated high-priority arrivals, cap any single job's monopolization,
or release surplus. Its objective prompt is the **modularity knob** — swap it for a greener or
fairer operator policy without retraining.

### Negotiation protocol
**Bounded rounds of offers** (not open chat), with provable termination. Each round both agents
exchange reasoned offers / counter-offers; on convergence (or the round budget) the result is
handed to the mechanism. The LLM stays **out of the hot loop**: queried once per discretised
state and **cached**, so a long multi-job sweep costs only tens of model calls and degrades
gracefully when the model is unavailable (rule-based fallback).

### Clearing & guarantee
- **Auction (decider).** A pure, unit-testable sealed-bid mechanism clears the negotiated
  offers. *Empirical note (prototype, Exp 9–13): a per-round value-max auction spreads GPUs
  and loses SLA; a **committed / serialized run-to-completion** clearing (bid-once, freeze
  priority, concentrate) wins ~2× on prod-tier SLA. The supply agent must therefore work
  **through** this commit spine, not re-introduce per-round thrashing.*
- **ILP (guarantee).** Repairs the negotiated plan to satisfy hard constraints and reconciles
  it to current state, since the cluster moves while the LLMs negotiate. *Negotiation proposes
  on a slightly-stale snapshot; the ILP reconciles to reality.*

### Two design traps the methodology must avoid
1. **Three deciders.** With demand-LLM + supply-LLM + ILP, if all three "decide," nothing is
   guaranteed. LLMs output **valuations + justifications + offers**, never the final free-form
   plan; the auction clears, the ILP guarantees.
2. **Incentive gaming.** A demand agent can exaggerate urgency to grab resources. Prototype
   Exp 13 showed priority classes are gameable and a flat per-claim budget does **not** fix it
   — over-claiming must be made self-defeating via **payments** (uniform-price / VCG) or
   **per-user budgets** spent across a user's jobs. This is both a risk and a differentiator.

---

## Key Assumptions & Scope

- **Mixed malleable + rigid workloads.** Jobs may resize at runtime via a malleability framework
  (dynamic MPI / DMR-DROM), but most production jobs are rigid (gang-scheduled / checkpoint-boundary).
  The malleable fraction φ is a first-class axis, not an assumption: **the supply agent's value scales
  with the rigid fraction** — under full malleability (φ=1) a late high-priority job preempts for free
  and the second agent is redundant (Exp 14A); its QoS contribution lives in the rigid fraction
  (Exp 14B). A **malleability-aware** supply agent reserves idle headroom only against rigid
  incumbents and reclaims malleable ones on demand, capturing the rigid-fraction prodSLA win at
  near-zero utilisation cost (Exp 15). *(Earlier revs scoped rigid jobs out of v1; Exp 14-15 inverted
  that — rigidity is where the contribution lives.)*
- **Negotiation is best-effort, never on the critical path.** Time-boxed with a mandatory
  heuristic fallback; the ILP absorbs staleness. *Property: negotiation can only help or be
  neutral, never block a job.*
- **LLM stays out of the millisecond loop.** Per-job metadata extraction and per-state
  negotiation strategy are offline / cached; no LLM inference sits inside the allocation loop.
- **Simulation first.** Evaluate in a scheduler simulator (Batsim / SimGrid) before any
  real-cluster trial — safer for a master's timeline.
- **Scoped contribution.** The novel core is the **two-sided negotiation policy**. The
  predictor/uncertainty layer supports it (it sizes the bids); the ILP is adopted from the
  reference paper as the guarantee layer.

---

## Evaluation Plan

| | |
|---|---|
| **Workloads** | Public traces (Google / Alibaba cluster, MIT Supercloud) + Cyberscience Center job logs, if accessible. |
| **Baselines** | SLURM default · FCFS + EASY backfilling · one learning-based scheduler (DRL) · **single-LLM-with-both-objectives + ILP** (the must-have baseline — proves the negotiation earns its extra cost). |
| **Metrics** | **SLA-violation rate (primary, incl. value-weighted / prod-tier)** · utilization · mean wait time · makespan · fairness · **negotiation cost (latency / tokens)** the layer must earn back. |
| **Environment** | Scheduler simulator (Batsim / SimGrid); selective real-cluster runs (single A100-PCIE-40GB) later. |

> **Success criterion:** SLA-violation rate competitive-or-better than every baseline at
> equal-or-higher utilization, *plus* interpretable allocation decisions, *plus* measured
> evidence that two agents beat the single-LLM proposer enough to justify their cost.

### Required ablations (defend the claim)
- **No-ILP:** remove the guarantee layer. *(Reference-paper ablation: SLA violations should
  jump — proves the ILP is load-bearing.)*
- **No-negotiation (single LLM):** collapse both objectives into one proposer. *(Proves the
  two-sided split earns its cost in interpretability / modularity / SLA.)*
- **No-supply-agent:** demand-only bidding into the auction (the current prototype). *(Isolates
  what the supply side actually adds — the open question this plan must answer.)*
- **No-uncertainty:** fix the safety margin instead of sizing it from quantiles. *(Proves
  uncertainty is a real co-contribution.)* **Done (Exp 16):** the uncertainty-sized margin is
  **insurance whose value grows with tail severity** — it ties the point forecast on mild demand but
  cuts prod-SLA ~35% under heavy tails, while a fixed/blanket margin *hurts* (over-subscribes). So the
  per-job quantile width is the load-bearing signal, not headroom per se. Regime-gated (needs spare
  capacity; vanishes at saturation).
- **Honesty / incentive test:** let agents misreport urgency; show the clearing rule
  (payments / budgets) keeps truthful bidding optimal.

---

## Current Status (prototype: PINS Stage-1 Exp 1–8,16–17 · Stage-2 Exp 9–15)

The deterministic spine and the **demand-side** LLM agent are built and measured in
`pins/` (pure-Python simulator, no network in the hot loop):

- The auctioneer (`pins/mechanism.py`) is pure and unit-tested.
- **Per-round value-max auction loses SLA** to greedy-FIFO (it spreads GPUs); the
  **committed / serialized** clearing **wins ~2× on prod-tier SLA**.
- An LLM that **sets and justifies** the committed priority (categorical, never a number)
  **matches** the deterministic version and adds the auditable transcript.
- Priority reports are **gameable**; a flat budget does not fix it → payments / per-user
  budgets are the open incentive problem.

**The supply agent is now built and measured (Exp 14-15).** A resource-side agent with the
asymmetric objective (headroom reservation) negotiates against the demand side through the commit
spine — turning demand-only bidding into the two-sided negotiation this plan proposes.

- **Regime-gated win (Exp 14).** Under full malleability the second agent is redundant (a late prod
  job preempts for free); its QoS contribution requires **rigid incumbents**, where it lifts prodSLA
  ~27→19% at moderate contention. An LLM sets & justifies the reservation level (categorical), and
  here — unlike the demand side — **model size matters**: qwen2.5:3b over-reserves dangerously,
  14b matches the deterministic oracle.
- **Malleability-aware reservation (Exp 15).** On mixed malleable+rigid workloads, an agent that
  reserves idle headroom only against the *rigid* fraction and reclaims malleable jobs on demand
  keeps the full prodSLA win **and recovers the utilisation cost** blind reservation pays — the
  recovery growing with the malleable fraction φ. Quantifies *"the supply agent's value scales with
  the rigid fraction."*

**The prediction co-contribution is now real end-to-end (Exp 16-17).** The dynamic forecaster
(Exp 8) gained a **quantile head** (P10/P50/P90, pinball loss) that is **conformally calibrated**
(coverage 0.67→0.75) without losing P50 accuracy. Its per-job uncertainty sizes the demand agent's
**safety margin** (`predictor.marginal_values`): the ablation shows an uncertainty-sized margin is
insurance whose value grows with the demand tail, while a fixed/blanket margin backfires. The
**LLM demand agent** then makes the hedge call from `(uncertainty, spike-risk, deadline, contention,
tier)` — categorical, code owns the GPU count; with the spike-risk signal, 7b/14b **beat** the
deterministic margin in both mild and heavy-tail regimes, with an auditable justification per hedge.
A demand-side echo of Stage-1's lesson: a heavy-tail failure was fixed by adding the missing *signal*
(spike-risk), not a bigger model — and model size still matters for the judgement (3b over-hedges).

**Still unbuilt:** the bounded multi-round *protocol* collapsed empirically to a one-shot
declaration (Exp 11 showed re-auctioning thrashes), incentive-compatible clearing (Exp 13: gameable,
payments/per-user budgets open), and the ILP guarantee layer (Phase 3c).

---

## Open Questions (next dominoes)

1. **Where is the allocation actually made?** (a) negotiation *produces* the allocation and the
   ILP only feasibility-checks (more novel), vs (b) the ILP does the heavy optimization and
   negotiation just sets its objective / priorities (safer, closer to the paper).
2. ~~**The supply agent's lever:**~~ **Answered (Exp 14-15): headroom reservation**, and it pays
   off only against **rigid** incumbents at moderate contention; a malleability-aware variant
   recovers its utilisation cost (Exp 15). Open sub-question: a non-zero reclaim penalty
   (checkpoint/rescale rollback) — how much does it erode the aware win?
3. **Setting `T_negotiation`:** fixed budget vs. adaptive to cluster load (tighter under load).
4. **Gate threshold:** learned from traces vs. a fixed multiple of average wait.
5. **Incentive-compatible clearing:** how many rounds, what is exchanged (bids? prices? NL
   justifications?), and how is honesty enforced by the clearing rule.

---

## Timeline & Milestones

| Phase | Name | Description | Status |
|-------|------|-------------|--------|
| **Phase 1** | Foundation | Load a trace; reproduce a baseline scheduler's utilization / SLA as a yardstick. | in progress |
| **Phase 2** | Predictor | Local-LLM metadata + ST-attention + quantile regression; validate P10/P50/P90 intervals (uncertainty sizes the margin). | **quantile + conformal calibration + margin + LLM-hedge done (Exp 16-17)**; local-LLM metadata extraction remains |
| **Phase 3a** | Mechanism + demand agent | Sealed-bid / committed auction + demand-side LLM bidding & priority. | **done (Exp 9–13)** |
| **Phase 3b** | Supply agent + protocol | Asymmetric supply LLM (headroom reservation); regime-gated to rigid incumbents; malleability-aware reservation. | **done (Exp 14–15)** · protocol → one-shot (Exp 11) |
| **Phase 3c** | ILP guarantee | Lightweight ILP repair / reconcile-to-live-state layer. | planned |
| **Phase 4** | Integrate & evaluate | End-to-end in simulator; run all ablations incl. single-LLM baseline; compare vs baselines; thesis write-up. | planned |

*Master's entrance exam preparation runs in parallel with Phase 1.*

---

## Next Week Plan

1. Study for the master's entrance exam.
2. **Supply agent — next steps after Exp 14-15:** (a) add a non-zero `reclaim_penalty` to
   `simulate_mixed` and measure how far it erodes the malleability-aware util recovery; (b) put the
   **LLM supply agent on the mixed regime** with a third state dimension (`malleable_fraction`) —
   Exp-14C predicts model size will matter for the reserve-vs-reclaim judgment.
3. **Must-have baseline:** build the **single-LLM-with-both-objectives + ILP** comparator so the
   two-sided split can be shown to earn its cost (still unbuilt).
4. **Prediction:** quantile outputs + conformal calibration + uncertainty→margin bridge + the
   **LLM demand agent that hedges from uncertainty (with a spike-risk signal)** are now built and
   working end-to-end (Exp 16-17): 7b/14b beat the deterministic margin in both mild and heavy-tail
   regimes, with auditable per-decision justifications. Remaining: (a) the still-missing local-LLM
   metadata extraction for cold-start prediction; (b) optionally derive spike-risk directly from the
   forecaster's P90/P50 ratio (currently a per-job tail proxy) for a fully end-to-end signal.
