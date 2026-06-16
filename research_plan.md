# Progress Report
**Uncertainty-Aware Dynamic Resource Allocation for HPC**

Lay Kim Seng — 2026 / 06 / 16
Cyberscience Center, Tohoku University

---

## Research Topic

# Uncertainty-Aware Dynamic Resource Allocation in HPC Systems via Market-Based Multi-Agent Negotiation

> **Framing refined.** Agents do not talk peer-to-peer — they submit valued, justified requests (bids) to a central orchestrator that clears them. That mechanism is an **orchestrator-mediated auction/market**, which is a recognised form of agent negotiation. Naming it precisely removes the title-vs-method mismatch.

---

## Background & Problem

### The orchestration gap
Modern AI / scientific workloads are bursty and heterogeneous, and need a more intelligent, runtime resource-management layer than static schedulers provide.

### The core tension
With current methods, pushing resource utilization higher also pushes the SLA-violation rate up. Utilization and reliability trade off against each other.

> **Why it's hard:** production HPC schedulers still do not exploit **malleability** (resizing jobs at runtime), even after 20+ years of research — supporting it touches the entire HPC software stack (Tarraf et al., IEEE TPDS 2024). Closing the utilization–SLA gap requires both better prediction and a smarter allocation policy.

---

## Research Goal

**1. Predict workload**
Forecast each task's resource demand over the next T timesteps, with an explicit uncertainty estimate.

**2. Allocate by negotiation**
Turn predictions + uncertainty into bids; an orchestrator clears them into runtime allocations.

**Target outcome:** a resource-management system that raises resource utilization while holding the SLA-violation rate at or below today's schedulers.

**Core contribution (thesis focus):** the uncertainty-aware allocation policy. Prediction and the agent layer are supporting components.

---

## Related Work

### 1. LLM-Driven Adaptive Cloud Resource Scheduling
LLMs generate scheduling decisions with reasoning, validated/optimized against learned objectives — bringing interpretability to scheduling.

**Gap we address.** *cloud-centric, no uncertainty modelling, and no runtime malleability for tightly-coupled HPC jobs.*

### 2. Malleability in Modern HPC Systems (Tarraf et al., 2024)
Survey of runtime job resizing across the HPC stack; shows malleability raises throughput & utilization but is unused in production.

**Gap we address.** *surveys mechanisms, but offers no learned, prediction-driven policy for deciding when and how much to resize.*

---

## Proposed System Overview

```
Inputs              Workload Predictor      Agent Bids              Orchestrator
──────────────      ──────────────────      ──────────────────      ──────────────────
Source code +   →   LLM features →      →   Request + value +   →   Market clearing +
runtime params +    ST-Attention →          uncertainty +           fairness → malleable
historical traces   P10 / P50 / P90         justification           allocation
```

> **Key idea:** prediction uncertainty flows all the way through — agents bid for a safety margin (e.g. P90) when confidence is low, and the orchestrator prices that into the allocation.

---

## Methodology — Workload Prediction

**Input:** task code + runtime parameters + historical run traces
**Output:** predicted resource demand over the next T timesteps, as a distribution.

### 1. LLM Metadata Extraction
A local LLM (e.g. LLaMA) reads source code to extract semantic features static analysis misses. Run offline / once per job — kept out of the latency-critical loop.

### 2. Temporal–Spatial Attention
Temporal = workload over time; spatial = across nodes & co-located jobs that contend for shared resources. Multi-head attention forecasts future load.

### 3. Quantile Regression
Outputs prediction intervals (P10 / P50 / P90), not a single point — this uncertainty is the input the allocation layer actually negotiates over.

---

## Methodology — Resource Allocation

### Orchestrator (resource manager)
Extends / shrinks resources at runtime. Receives all bids and solves a constrained allocation: maximize aggregate predicted value subject to capacity, with a fairness floor per agent.

### AI Agent (per task)
Submits a bid = (resource amount from a chosen quantile, a value/utility, and a short justification). Higher predicted uncertainty → bids for more safety margin.

- Agents never talk peer-to-peer — they only bid to the orchestrator, which keeps the protocol simple and auditable.
- No agent is guaranteed its full request; when it yields resources it is compensated (priority / credit) so cooperation is incentivized.
- MCP is the transport that connects agents to the orchestrator — the allocation policy is the research contribution, not MCP itself.

---

## Key Assumptions & Scope

### Malleable workloads
Jobs can resize at runtime via a malleability framework (e.g. dynamic MPI / DMR-DROM). Rigid jobs are out of scope for v1.

### Simulation first
Evaluate in a scheduler simulator (Batsim / SimGrid) before any real-cluster trial — far safer for a master's timeline.

### LLM stays offline
The LLM only does per-job metadata extraction; no LLM inference sits inside the millisecond-level allocation loop.

### Scoped contribution
The novel core is the uncertainty-aware allocation policy. Predictor & agent layer are built to support it, not to be SOTA themselves.

---

## Evaluation Plan

| | |
|---|---|
| **Workloads** | Public traces (Google / Alibaba cluster) + real job logs from the Cyberscience Center, if accessible. |
| **Baselines** | SLURM default · FCFS + EASY backfilling · one learning-based scheduler (e.g. DRL). |
| **Metrics** | Resource utilization · SLA-violation rate · mean wait time · makespan · fairness. |
| **Environment** | Scheduler simulator (Batsim / SimGrid); selective real-cluster runs later. |

> **Success criterion:** achieve higher utilization at equal-or-lower SLA-violation rate than every baseline.

---

## Timeline & Milestones

| Phase | Name | Description |
|-------|------|-------------|
| **Phase 1** | Foundation | Load a trace; reproduce a baseline scheduler's utilization / SLA as a yardstick. |
| **Phase 2** | Predictor | Local-LLM metadata + ST-attention + quantile regression; validate intervals. |
| **Phase 3** | Allocation | Design & implement the uncertainty-aware bidding / clearing policy. |
| **Phase 4** | Integrate & evaluate | End-to-end in simulator; compare vs baselines; thesis write-up. |

*Master's entrance exam preparation runs in parallel with Phase 1.*

---

## Next Week Plan

1. Study for the master's entrance exam.
2. Run a local model (LLaMA) to extract metadata from sample source code — first prototype of the predictor input.
3. **New:** load one cluster trace and reproduce a baseline scheduler's utilization / SLA numbers, so there is a measuring stick before building anything.
