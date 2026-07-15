# Progress Report
**AI-Agent Negotiation for Dynamic Resource Allocation in HPC**

Lay Kim Seng — 2026 / 06 / 19
Cyberscience Center, Tohoku University

> *Rev. 2026-06-19:* updated to the **two-sided (demand/supply) negotiation** design developed
> in [`CLAUDE.md`](./CLAUDE.md) after reading *"LLM-Driven Adaptive Cloud Resource Scheduling"*
> (IEEE OJ-CS 2026). The single-orchestrator framing is superseded by a **demand-LLM ⇄
> supply-LLM** negotiation cleared by an auction and guaranteed by an ILP.

> *Rev. 2026-07-16:* **pivot — reason-then-referee (see the new section below).** The research
> goal is now to **beat the rule/ILP-guaranteed pipeline with a referee LLM that decides
> allocations directly**. The governing rule "LLMs reason, code decides" is deliberately
> inverted: no mathematical rule set covers every situation, so the allocator itself must be a
> flexible, situation-reasoning agent. Deterministic code is demoted to **evaluator** (it
> reports violations, never repairs). Everything below the pivot section documents the
> negotiate→auction→ILP pipeline, which remains fully valid as the **baseline arm** the
> referee must beat.

---

## Research Topic

# Breaking the Utilization–Service-Quality Tradeoff in HPC via AI-Agent Negotiation

> **Focus (updated 2026-07-16).** The contribution is the **referee-LLM allocator**: a
> demand agent and a supply agent submit reasoned, justified statements, and a **referee LLM
> applies the cluster's rules and judgment to output the allocation directly**. No ILP sits
> behind the referee — deterministic code only *evaluates* (reports violations; an infeasible
> tick falls to the floor and is charged to the referee). The previous contribution — the
> two-sided negotiation cleared by a deterministic **auction** and guaranteed by an **ILP** —
> is retained in full as the **baseline arm** the referee must beat. Uncertainty-aware
> prediction remains a **supporting co-contribution**: it sizes the margins the agents ask
> for, in both arms.

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

**1. Allocate by referee LLM (the contribution — 2026-07-15 pivot)**
A demand agent and a supply agent submit reasoned statements; a **referee LLM decides the
allocation directly**, with deterministic code as evaluator only (violations → floor
fallback, charged to the referee). Rationale: no fixed rule set covers every situation.

**1b. The baseline arm (former contribution): two-sided negotiation**
A demand agent and a supply agent turn their private state into reasoned offers; a
deterministic mechanism (auction) clears the negotiation and an ILP reconciles it to live
state — rationing scarce resources to the jobs that value them most *without* letting either
LLM make the final, unguaranteed decision. Retained in full as the arm the referee must beat.

**2. Predict workload (co-contribution)**
Forecast each task's resource demand over the next T timesteps *with an explicit uncertainty
estimate*, which sizes the safety margin the demand agent bids for.

**Target outcome (core claim):** raise resource utilization while keeping the SLA-violation
rate **competitive-or-better than every baseline** — classical, learning-based, and the
single-LLM proposer — *plus* an auditable justification for every allocation that RL cannot
offer.

**Governing design rule** *(superseded 2026-07-15 — kept for the baseline arm)*: *the LLMs
reason / explain; deterministic code decides.* The deciders are the **auction** (clears the
negotiation) and the **ILP** (guarantees feasibility) — never the free-form LLM chat.
The referee pivot inverts this rule for the new headline arm; see the pivot section.

---

## Pivot (2026-07-15) — Reason-then-Referee: the LLM decides

**Goal.** Beat the rule-based / ILP-guaranteed allocator with a **referee LLM agent** that
outputs the allocation directly. **Rationale:** no mathematical equation or fixed rule set
deals with every situation; the allocator must reason flexibly about the situation in front
of it. The old pipeline's rigidity is now the thing under test, not the guarantee.

**New flow.** Demand agent and supply agent each reason privately and submit statements
(base need + requested margin + justification; requested reserve + justification) → a
**third referee LLM** applies the supercomputer's rules and judgment → **outputs the
allocation directly**. Deterministic code (`check_allocation`) only **evaluates**: an
infeasible referee tick falls back to the floor and is charged to `fallback_rate` — code
never repairs a decision.

**Win condition.** Feasibility is table stakes (the rule arm is 100% feasible by
construction); the win must show in **outcomes** (SLA / prodSLA / slowdown) where
flexibility pays — ideally *conditionally*, on the ticks where the rigid rule decides badly.

**Evidence so far** (`pins/referee.py`, `referee_eval.py` = Exp 49, `trace_replay
--referee` = Exp 50; branch `referee_allocator`):

- **Constraint enforcement, not arithmetic, is the chat-model failure.** All chat models
  compute `total_awarded` correctly then ignore the ≤ comparison; on real v2020 scenes
  feasibility collapses to 0% under scarcity — chat LLMs **won't say no** (they serve
  everyone and blow the budget). A self-check prompt fixes toy scenes only.
- **Reasoning models fix it: deepseek-r1:32b is 100% feasible** at every pool factor
  including shortfall, and it is not parroting the rule referee — 10/24 scenes differ while
  feasible (egalitarian partial coverage vs all-or-nothing, stated rationales).
- **In-sim (Exp 50, v2020 replay, pools 4/6/8, n=8):** r1:32b referee has **0% fallback**
  and **ties** the floor and the negotiated arm (all deltas ns). At pool 6 it held the floor
  exactly while the negotiated arm slipped (+1.6/+2.1 vs floor). 3b referee overcommits
  45–58% of ticks yet still ties — the fallback floor protects it.
- **Transcript case study** (`pins/transcripts_seed23_pool6.txt`, replayable via
  `pins/replay_transcripts.py`): the two outcome-flipping windows are fully auditable —
  the referee **won** seed 3 by spending margins on prod jobs (incl. a stated *partial*
  grant), and **lost** seed 2 by hedging ahead-of-schedule besteffort jobs so the pool was
  empty when the prod job arrived. Same supply request gets opposite rulings depending on
  cluster state ("no evidence of incoming load" at an empty pool vs granted mid-window) —
  genuinely situational judgment, currently **unaimed** under scarcity.

**Current read:** *sufficiency, not superiority* — a reasoning referee replaces the
guarantee layer without loss and adds auditable rationales; it does not yet beat the rule
pipeline on averages.

**Next steps.**
1. Finish the r1:32b sweep (pool 8 running) → then **n=32 seeds** on the best pool so
   ±7pp CI bands cannot hide a real effect.
2. **Conditional (hard-tick) analysis:** split ticks by "the rigid rule decided badly here"
   and measure the referee's delta on that subset — the flexibility claim predicts the win
   lives there, diluted away in uniform averages.
3. **Aim the flexibility:** a margin-priority rule in `SYSTEM_REFEREE` (don't hedge
   besteffort when `incoming_prod=many` and the pool is tight) targets the seed-2 failure
   mode without touching the seed-3 win.
4. Fallback-semantics honesty in the write-up: report `fallback_rate` alongside outcomes.
5. If superiority does not materialize: the defensible headline is *"a reasoning LLM
   replaces the guarantee layer without loss + auditable rationales"* (LLMsched needed an
   ILP to make its LLM safe; the referee doesn't).

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

## Methodology — Resource Allocation (the baseline arm; former contribution)

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
  **Headline job model (limitation):** the trace-replay world (`two_sided_sim`, Exp 28+) models jobs
  as *moldable and grow-only* — never involuntarily preempted, but able to start below their full
  request and ramp up, returning margin GPUs voluntarily. This matches **elastic GPU/ML workloads**
  (TorchElastic-style rescaling; the containerized ML tasks of the Alibaba trace it replays), which
  is the scoped target. It does **not** cover totally rigid fixed-communicator MPI jobs, whose
  all-or-nothing discipline is represented by the EASY baseline (Exp 41) and deliberately not improved.
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

## Current Status (prototype: Stage-1 Exp 1–8,16–17,19–21 · Stage-2 Exp 9–15,18 · integration Exp 22–28)

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

**The integration phase is underway (Exp 22–28).** The pieces above are now wired into one
locked pipeline (negotiate → committed-auction → ILP placement) and measured end-to-end:

- **ILP guarantee layer built and measured (Exp 18, 23, 25).** The LLMSched-style ILP ties the
  auction in 1-D and removes a structural placement loss in 2-D (Exp 18); LLM affinity hints +
  ILP node placement are a knife-edge gated on classification accuracy (Exp 23); and in the full
  pipeline the ILP guarantee makes the LLM's aggressive over-demand *safe* (Exp 25).
- **Bounded two-sided protocol built (Exp 22, 24).** A margin ⇄ reserve concession ladder with
  rule fallback; the Exp-22 fallback pathology (96/74/49%) was an artifact of flat synthetic
  caps — with real Stage-1 demand it is 0% everywhere (Exp 27). Contested-slice negotiation
  flipped the headline: negotiated beats the single-LLM baseline (Exp 24).
- **The must-have single-LLM-with-both-objectives baseline exists** and is measured in every
  two-sided sweep (Exp 22–28); it consistently over-commits (lowest util, worst slowdown).
- **Real Stage-1 predictions + real trace jobs (Exp 27–28).** With predicted GPU caps
  (quarter-GPU quanta) and then full Alibaba v2020 trace replay (real arrivals/durations/demand,
  jointly), the headline survives: agents buy prod-SLA; at pool 8 `negotiated` beats the no-LLM
  floor on BOTH metrics; and **the protocol substitutes for model scale** — negotiated@3b ≥
  negotiated@14b while the un-braked single-LLM needs 14b and still loses (needs a seed sweep
  before it becomes a thesis headline; 8 seeds so far).
- A reflective margin agent was tried and is a clean negative (Exp 26: limit-cycle thrashing) —
  the deterministic concession ladder, not reflection, is what makes the small model safe.

**Still unbuilt:** incentive-compatible clearing (Exp 13: priority reports are gameable, a flat
budget does not fix it; payments / uniform-price clearing open) and the classical-scheduler
baselines from the evaluation plan — now scoped and run (Exp 41 EASY, Exp 42 tabular-Q; full DRL out, see decision 5).

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
| **Phase 1** | Foundation | Load a trace; reproduce a baseline scheduler's utilization / SLA as a yardstick. | **done for Alibaba v2020 (Exp 28 trace replay: real arrivals/durations/demand vs no-LLM floor)**; EASY + tabular-Q yardsticks done (Exp 41/42) |
| **Phase 2** | Predictor | Local-LLM metadata + ST-attention + quantile regression; validate P10/P50/P90 intervals (uncertainty sizes the margin). | **quantile + conformal calibration + margin + LLM-hedge done (Exp 16-17)**; local-LLM metadata extraction remains |
| **Phase 3a** | Mechanism + demand agent | Sealed-bid / committed auction + demand-side LLM bidding & priority. | **done (Exp 9–13)** |
| **Phase 3b** | Supply agent + protocol | Asymmetric supply LLM (headroom reservation); regime-gated to rigid incumbents; malleability-aware reservation. | **done (Exp 14–15)** · protocol → one-shot (Exp 11) |
| **Phase 3c** | ILP guarantee | Lightweight ILP repair / reconcile-to-live-state layer. | **done (Exp 18, 23, 25)** — ties auction in 1-D, fixes 2-D placement, makes LLM over-demand safe |
| **Phase 4** | Integrate & evaluate | End-to-end in simulator; run all ablations incl. single-LLM baseline; compare vs baselines; thesis write-up. | **in progress (Exp 22–28)** — locked pipeline + single-LLM baseline measured; real caps (Exp 27) + trace replay (Exp 28); remaining: seed-swept statistics, prediction error in loop, incentives, write-up |
| **Phase 5** | Referee allocator (pivot) | Referee LLM decides allocations directly; code demoted to evaluator; old pipeline = baseline arm. | **in progress (Exp 49–50)** — r1:32b 100% feasible, 0% fallback, ties floor/negotiated at n=8; remaining: pool 8, n=32 sweep, hard-tick conditional analysis, margin-priority prompt |

*Master's entrance exam preparation runs in parallel with Phase 1.*

---

## Next Week Plan (post-Exp-28, updated 2026-07-06)

1. Study for the master's entrance exam.
2. **Seed-sweep the headline (Exp 29):** rerun the Exp-28 trace replay at 32 seeds with per-seed
   logging and paired 95% CIs — the "negotiated beats the floor on both metrics at pool 8" and
   "negotiated@3b ≥ @14b" claims currently rest on 8 seeds with no error bars. Report the
   prod-SLA-gain vs best-effort-slowdown trade-off from the same data (the price of protection).
3. **Prediction error into the loop (Exp 30):** key `predict_gpu.py` outputs by `job_name` and
   replay with *predicted* caps instead of the trace's real requests — the last step for a true
   end-to-end prediction→negotiation claim (Exp 28's own caveat).
4. **Incentives (open since Exp 13):** design a payments/uniform-price clearing rule so truthful
   priority declaration is optimal; the flat budget is proven not to work.
5. **Scope decision (RESOLVED, Exp 41/42):** EASY-backfilling IN (built on the Stage-1 runtime
   prediction — its one live Stage-2 role after Exp 38/39; negotiated@3b beats it, even with
   oracle runtimes, Exp 41). Learning-based baseline IN as tabular Q-learning over the LLM's
   own discretised state/action interface (loses to rule and negotiation at every pool, Exp 42).
   Full DeepRM/Decima-style DRL OUT — defense: it owns a different (whole-allocator) action
   space, so it cannot isolate the decision-rule question; and Exp 42 shows the binding
   constraint is return-variance at feasible sample sizes (~10^3 episodes), which
   policy-gradient training on the same episode budget would share, while published DRL
   schedulers train on 10^5-10^6 episodes — a compute budget outside this thesis.
6. **Prediction leftovers:** local-LLM metadata extraction for cold-start; derive spike-risk from
   the forecaster's P90/P50 ratio for a fully end-to-end signal.
