# PINS — Prediction-Informed Negotiated Scheduling for Elastic HPC

Bachelor's research project (Tohoku University).

PINS combines two ideas to cut wasted GPU-hours on shared HPC clusters:

1. **Resource prediction (Stage 1)** — a quantile gradient-boosted-tree model
   predicts what a job's submission script does *not* already say: its runtime,
   its actual GPU utilization, and its actual peak GPU memory. The governing
   rule is *"declared fields are features, not targets"* — the user's requested
   `plan_gpu` is an input, never a label, because the scheduler can just read it
   off the script. Trained and evaluated on the Alibaba v2020 GPU trace. The
   P10–P90 interval it emits is what sizes the margin in Stage 2.

2. **Agent-negotiated allocation (Stage 2)** — demand- and supply-side LLM
   agents negotiate GPU allocation, a sealed-bid uniform-price auction clears
   it, and jobs are rescaled live. This is where the LLM lives: it *reasons and
   explains*, while the auctioneer — pure, deterministic code — *decides*. The
   LLM is never in the clearing hot loop.

## Documents

| File | What it is |
|---|---|
| [`research_plan.md`](research_plan.md) | The full 6-month research plan: stages, gates, what counts as a contribution, and the task-classified RAG predictor sub-plan. |
| [`research_progress.md`](research_progress.md) | Running experiment log — every Stage-1 result with measured numbers. |

## Status

Active research. Stage-1 prediction is evaluated offline on the Alibaba v2020
GPU trace; the negotiation layer runs in simulation. Neither needs a GPU.
