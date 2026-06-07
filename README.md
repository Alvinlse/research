# PINS — Prediction-Informed Negotiated Scheduling for Elastic HPC

Bachelor's research project (Tohoku University).

PINS combines two ideas to cut wasted GPU-hours on shared HPC clusters:

1. **LLM-based resource prediction (Stage 1)** — an LLM predicts each job's
   time-varying GPU/VRAM demand. The governing rule is *"the LLM reasons and
   explains; deterministic code decides"* — the model emits per-layer shapes and
   structured facts, and plain code does the final arithmetic. This consistently
   beats letting the LLM emit the final number (error drops ~an order of
   magnitude every time a figure moves from the LLM into code).

2. **Agent-negotiated allocation (Stage 2)** — job-agents negotiate GPU
   allocation through a fast sealed-bid uniform-price auction, and jobs are
   rescaled live. The auctioneer is pure, deterministic code; the LLM only
   produces optional human-readable justifications.

## Documents

| File | What it is |
|---|---|
| [`research_plan.md`](research_plan.md) | The full 6-month research plan: stages, gates, what counts as a contribution, and the task-classified RAG predictor sub-plan. |
| [`research_progress.md`](research_progress.md) | Running experiment log — every Stage-1 result with measured numbers. |

## Status

Active research. Stage-1 prediction results are measured on a single
A100-PCIE-40GB; the negotiation layer runs without a GPU.
