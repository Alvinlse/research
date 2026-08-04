# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **PINS — Prediction-Informed Negotiated Scheduling for Elastic HPC**: an
LLM predicts each job's time-varying GPU demand, agents negotiate allocation via a fast
auction, and jobs are rescaled live to cut wasted GPU-hours. The full 6-month plan and the
scientific framing live in `research_plan.md` — **read it first**; it defines the stages,
the gates, and what counts as a contribution (incl. §5 measured Stage-1 findings and §6, the
task-classified RAG predictor sub-plan). `research_progress.md` is the running experiment log
(every Stage-1 result, with numbers). `pins/README.md` documents the negotiation layer.

This is a `uv` project (Python 3.10). This `Research/` directory (formerly `MCP/`) is the
project root AND its own git repository — run all commands and make all commits from here.

## Commands

```bash
# Use the project venv directly (uv run also works):
.venv/bin/python -m pins.test_mechanism            # unit-test the auctioneer (no network/LLM, instant)

# Stage-1: predict GPU/runtime/util/mem on the Alibaba v2020 trace vs baselines
.venv/bin/python -m pins.eval.predict_gpu --target runtime   # --target {plan_gpu,runtime,gpu_util,gpu_mem}
#   declared fields are FEATURES, not targets

# Stage-2 TRACE REPLAY: the current architecture (validated auction + debate-on-trigger)
.venv/bin/python -m pins.trace_replay                # replay v2020 jobs against a GPU pool, no LLM
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:3b --pools 32 --seeds 8
.venv/bin/python -m pins.two_sided_sim               # two-sided demand/supply market sweep

# Stage-2 SIMULATION (older synthetic world, still the mechanism sweep): pure Python, no GPU, instant
.venv/bin/python -m pins.negotiation_sim             # auction vs greedy/equal/static/committed sweep
.venv/bin/python -m pins.negotiation_sim --llm --model qwen2.5:3b   # + LLM-strategist & LLM-priority rows
.venv/bin/python -m pins.llm_agent                   # smoke: LLM bid-strategy + committed-priority per state
.venv/bin/python -m pins.llm_agent --no-llm          # same via the deterministic rule fallback
.venv/bin/python -m pins.referee                     # smoke: referee LLM on a scene
```

**Tests are a plain script, not pytest.** `pins/test_mechanism.py` auto-discovers and runs every
`test_*` function in `__main__`. To run a single test: `.venv/bin/python -c "from pins.test_mechanism import test_efficiency_total_value as t; t()"`.

## Environment assumptions (hard dependencies of the demos)

- **Ollama** running at `http://localhost:11434`; `qwen2.5` in `3b` (default), `7b`, `14b` are pulled.
- Nothing binds a network port any more — the MCP server/SSE transport was deleted with `1534db2`.
- Stage-1 is evaluated offline on the **Alibaba v2020 GPU trace** and Stage-2 in simulation, so
  **no part of the current code needs a GPU**. (The A100-PCIE-40GB target and the closed-loop
  `predict_cnn.py`/`predict_arch.py` belonged to the retired LLM-prediction track.)
- The LLM path degrades gracefully: every `--llm` arm falls back to its deterministic rule if
  Ollama is down.
- **torch/CUDA gotcha (READ before installing anything):** `pyproject.toml` pins `torch>=2.12.0`,
  whose default PyPI wheel is built for **CUDA 13.0 — too new for this node's 12.7 driver**
  (it errors with `undefined symbol: ncclCommResume` / `cuda.is_available()` False). The working
  install is `uv pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124`.
  **Any plain `uv pip install <pkg>` (even `pandas`) re-resolves the pin and silently upgrades
  torch back to the broken 2.12.0+cu13** — install other packages with `--no-deps`, or reinstall
  the cu124 torch right after. Always verify with `python -c "import torch; print(torch.cuda.is_available())"`.

## Architecture

Two independent layers live here. **`pins/` is the real work**; the top-level `chat_*.py` files
are an earlier throwaway prototype.

### The governing design principle (applies to both stages)

> **The LLM reasons/explains; deterministic code decides.** The LLM is never in the hot loop.

This is not a slogan — it's enforced structurally, and it is now **measured** (see
`research_progress.md`). In negotiation, the auctioneer that clears allocations is pure Python
(`mechanism.py`); the LLM only produces an optional justification. In prediction, the closed-loop
experiments showed a zero-shot LLM *number* over-predicts CNN peak VRAM 7×–170× (even at 14B),
while a predictor that has the LLM emit **per-layer shapes** and lets code do the arithmetic hits
~0.04 GB MAE and beats the params heuristic ~40×. **Empirical rule:** every time a *number* moved
from the LLM into code, error dropped ~an order of magnitude. When extending Stage-1, keep the LLM
on extraction/shapes and never let it emit the final figure.

### `pins/` — Stage-2 negotiation

The live MCP wiring (`negotiation_server.py` + `job_agent.py` + `run_demo.sh`) was **deleted** in
`1534db2` ("trim to paper-relevant code and results"). Everything below is pure in-process Python —
no MCP, no network. The research results all come from the simulation/replay stack.

- `mechanism.py` — **pure** sealed-bid uniform-price auctioneer + anti-thrashing rescale gate.
  No LLM, no network; this is the "decider" and the only thing the unit tests cover.
- `predictor.py` — Stage-1 **stub**: maps a job phase → non-increasing marginal-value curve.
  Encodes the project's premise (demand varies by phase). Superseded by `trace_replay.py` for results.

The clearing produces allocation *deltas*. Wiring those deltas to real rescaling (TorchElastic/SLURM)
is the actuation layer and is not implemented.

**Anti-thrashing gotcha (fixed Exp 9, keep it this way):** `mechanism.clear`'s gate charges
`rescale_cost` only for **preemptions** (`sum(max(0, cur-target))`), never for filling *idle* GPUs —
otherwise a cold start refuses free capacity and the auction sits at 0% utilisation.

### `pins/negotiation_sim.py` + `pins/llm_agent.py` — Stage-2 SIMULATION (where the negotiation results live)

The MCP server/agent above is the *live* wiring; the **research results come from this pure-Python
simulator** (no MCP, no network, no GPU — runs in `.venv`, fast & seeded). It streams jobs (phase
timelines, urgency, deadline, prod/best-effort **tier**) against a fixed GPU pool and compares
allocation strategies on **SLA-violation rate** (raw *and* prod-tier), utilisation, welfare. A
"strategy" = a **bid-builder** (`static_bid`/`deadline_bid`/LLM) paired with an **allocator factory**.
The full experiment log with numbers is `research_progress.md` (now through **Exp 13**); the arc:

- **Per-round marginal auction loses SLA to greedy-FIFO** (Exp 9–10): diminishing-value bids *spread*
  GPUs thin (everyone runs slow) and re-pricing each round *thrashes*. Even an LLM strategist
  (`make_llm_bidder`, Exp 10) doesn't beat greedy — its value is interpretability + goodput, not SLA.
- **`make_committed_auction` is the winner** (Exp 11): **bid-once → freeze priority → serialise**
  (full GPU block per job, run to completion). Concentration + a *stable* order ≈ halves prod-tier SLA
  vs greedy. Value-block (dynamic order) and an incumbency bonus both fail — the lever is a *stable*
  order, not stability of who-holds-what.
- **`make_llm_committed`** (Exp 12): the LLM sets & **justifies** each job's committed priority as an
  ordinal class (critical/high/normal/low) via `llm_agent.llm_priority`; code maps class→weight and
  serialises. Preserves the win, adds auditable decisions, matches the deterministic priority.
- **Incentives are unsolved** (Exp 13): priority is a *trusted self-report* — `make_declared_committed`
  shows best-effort jobs lying ('critical') collapse prod protection back to greedy, and a flat
  budget does **not** fix it (it punishes long honest jobs as much as liars). True fix needs
  value-elicitation with **payments** (per-user budgets / VCG) — the open problem.

**The LLM hinge applies here too:** in `llm_agent.py` the LLM emits
only **categorical/ordinal** choices (a bidding `stance`+`focus_gpus`, or a priority *class*) plus a
justification, **never a number**; deterministic code owns every magnitude. It is kept out of the hot
loop by **caching one decision per discretised state** (`llm_agent_cache.json`), so a full sweep costs
~tens of Ollama calls, not thousands. Both LLM paths degrade gracefully to a rule on `--no-llm`/Ollama-down.

### `pins/eval/` — Stage-1 prediction evaluation

**Only `predict_gpu.py` survives** (the LLM-VRAM harnesses below were deleted with `1534db2`).
It predicts a task's resources from **submission metadata** on the Alibaba v2020 GPU trace with
quantile GBTs, scored against a no-information mean predictor (the research gate). Results land
in `results_gpu*.json`; `pins/bridge.py` converts the predictions into the qualitative buckets
Stage-2 consumes.

`--target` picks the quantity (retarget of 2026-07-08):

| target | what it is |
|---|---|
| `plan_gpu` | the original track, byte-identical — keeps Exp 30/31 reproducible |
| `runtime` | `end_time − start_time` (s), Terminated tasks only |
| `gpu_util` | ACTUAL GPU utilization % (`gpu_wrk_util`, mean over workers) |
| `gpu_mem` | ACTUAL peak GPU memory GB (`max_gpu_wrk_mem`, max over workers) |

**The governing caveat:** `plan_gpu` is a **user-declared** field — the scheduler already has it at
decision time, so predicting it is imputation, not demand prediction. For every other target
`plan_gpu` joins the *features*. Rule: **declared fields are features, not targets.** The
`gpu_util`/`gpu_mem` targets need `pai_sensor_table`
(`data/fetch_alibaba_gpu.py --tables pai_sensor_table`).

<details>
<summary>Retired: the LLM-VRAM prediction track (files deleted, finding retained)</summary>

`predict_resources.py`, `predict_cnn.py`, `predict_arch.py` + `benchmark.json` are gone. What they
established and why it still constrains the design:

- `benchmark.json` + `predict_resources.py` — prompted an LLM (metadata only) for
  `{peak_mem_gb, recommended_gpus}` and scores it against two baselines it must beat: a
  no-information **mean** predictor (the research gate) and a **params×bytes heuristic**.
- `predict_cnn.py` — **closed-loop on a VGG-style CNN family.** Defines `SimpleCNN`, predicts its
  peak VRAM four ways, then trains it to get measured truth. Four predictor modes share the file:
  raw-LLM number (default), `query_hybrid` (LLM emits structured facts → formula), `--reasoning`
  (LLM walks layers in free text → `_extract_last_json`), and `--deterministic` (the winner:
  `feature_map_elements` sums per-layer shapes, `(a,b)` overhead is **leave-one-out calibrated**
  over `DET_CONFIGS`). `--precision {fp32,fp16,bf16}` toggles AMP.
- `predict_arch.py` — **does the deterministic recipe generalize?** Pools VGG-CNN + `SmallResNet`
  (skip connections) + `TinyLM` (transformer) and fits **one global `(a,b)`** leave-one-out. Key
  generalization: `activation_elems_per_sample` is architecture-agnostic (forward hooks summing
  leaf-module outputs) instead of replaying one recipe. The hooks see module *outputs* only, so
  they miss a transformer's internal `batch·heads·seq²` attention scores; `attention_elems_per_sample`
  adds that term analytically from metadata (`layers·nhead·seq²`, deterministic — **not** an LLM
  number) so one global `(a,b)` fits CNN+ResNet+Transformer with every job within 1.5× (Exp 7).
  **Remaining limit:** fp32-only — under fp16/bf16 a flash-attention kernel may not materialise the
  score matrix, so the term should be gated on the attention backend. `research_progress.md` has the
  full result tables.

**The finding that outlived the code:** every time a *number* moved from the LLM into code, error
dropped ~an order of magnitude — but the headline 40× win was **synthetic-only**; on real models it
was MAPE 18% / ρ 0.80. This is why Stage-1 is now GBTs on a real trace.

</details>

<details>
<summary>Retired: `pins/forecast/` + `data/` — Stage-1 DYNAMIC prediction (time-series)</summary>

`pins/forecast/` was deleted with `1534db2`; the `data/` fetchers remain. Kept for the design points.

Where `eval/` predicts **one static peak number** per job, this layer forecasts a *running*
job's **trajectory** — GPU/CPU/memory over the next **5 min (HORIZON=30 steps × 10 s)** — on
real **MIT Supercloud** traces. Same governing hinge: an **attention model (deterministic
code) decides the numbers**; the LLM (being added next) sits on top emitting structured regime
facts, never the number. The pipeline is three stacked modules, each runnable standalone:

- `data/fetch_supercloud.py` — pulls a **joint CPU+GPU** job sample from the public, anonymous
  S3 bucket (`mit-supercloud-dataset`, no creds — plain HTTPS REST, no `aws` CLI). The `cpu/`
  and `gpu/` folders partition jobids *differently* and not every job has both, so it indexes
  both sides and intersects. Sample lands in `data/supercloud-sample/` (+ `joint_jobids.txt`).
- `pins/forecast/dataset.py` — the aligner. **CPU is 10 s, GPU is ~100 ms**, and the two streams
  are logged in **different time zones** (GPU `timestamp` trails CPU `EpochTime` by a whole-hour
  offset — auto-detected per job by max bin-overlap, robust to DST). Resamples GPU→10 s, inner-joins
  on the overlapping wall-clock window, emits one aligned frame per job:
  `CHANNELS = [gpu_util, gpu_mem_gb, cpu_util, mem_gb]`.
- `pins/forecast/baselines.py` — **persistence** + **moving-avg** + the shared `evaluate()` harness
  (a forecaster is any `f(history)->(HORIZON,C)`; metric = per-channel MAE + scale-normalised nMAE).
  Persistence is a *strong* gate here (telemetry is piecewise-flat); the real errors are at phase
  transitions.
- `pins/forecast/model.py` — the deterministic decider: a small Transformer encoder over the
  history. **Critical design point:** it predicts the **residual from persistence** (the *change*,
  not the level), so on flat channels it degenerates to persistence (can't lose) and spends
  capacity only where the signal moves. This is what makes it beat the gate overall (and decisively
  on the dynamic `gpu_util`/`cpu_util` channels). A naive absolute-value model loses on the flat
  memory channels — don't revert to it.

</details>

## Conventions

- Code is written to be *read by a researcher*: heavy docstrings tie each module back to specific
  lines in `research_plan.md`. Match that style — explain the "why" and the design link, not just the "what".
- Keep the pure decider (`mechanism.py`) free of MCP/network/LLM imports so it stays unit-testable
  and provable in isolation. Verify it (`test_mechanism`) before wiring anything above it.

## Operational manual — solved problems (add yours here, especially time-consuming ones)

Treat this as the ops half of the "manual" idea (the referee's half is
`pins/referee_manual.md`): when a problem costs real time to solve, distill the fix
here so no future session re-derives it.

- **The standard operating point (Exp 97, 2026-07-30) is now the DEFAULT** of
  `trace_replay.py`: pool **32 quanta = 8 GPUs**, jobs **auto at 3× pool** (`--n-jobs 0`),
  `--tick 30`, `--horizon 0` (= 1600 ticks = the same 13.3 h), `--slack-mult 10`. Floor: 80%
  utilisation, 12.0% deadline violations, every job finishes. The **tick is a resolution knob
  only** — arrival span (6 h), simulated span (13.3 h) and the work clamp (2 min..2 h) are
  denominated in SECONDS, and `two_sided_sim.set_tick()` restates the wall-clock thresholds
  (`STARVE_TICKS`, `TTF_HORIZON`, `DYN_AFTER`, referee's `STARVE_WAIT_TICKS`) in the new units.
  Shortening the tick costs per-tick work proportionally: 4x the policy invocations. Pools are in **quarter-GPU quanta** (4q = 1 GPU) — the old `4,6,8`
  default was 1–2 physical GPUs, and its deadline recipe (slack vs a job's *solo* runtime at
  76% util) floored violations at 40–54% before any policy ran. Pre-Exp-97 tiers are
  reproducible with `--tick 120 --pools 8 --n-jobs 16 --horizon 300 --slack-mult 1` and keep
  their old tier names; anything else gets `+t30+h1600+slack10+n3x` so the two can never merge.
- **Login-node reaper** kills background shells after ~12–15 CPU-minutes. Run one
  sweep/pool per background task; don't chain them in one shell. (Cost: a dead
  mid-deepseek run, 2026-07-15.)
- **Parallel Ollama sweeps**: `pins/run_parallel_sweep.sh` (WORKERS ollama slots, seed
  waves, peek tables between waves). VRAM budget: weights ~20GB + WORKERS x num_ctx of
  KV — 4 x 8192 fits the 40GB A100; don't raise both.
- **`save_cache` is flocked + pid-unique tmp** (llm_agent.py) because parallel workers
  finishing together used to crash on a shared `.tmp` rename and could silently drop
  each other's keys — each lost key costs 1–3 GPU-minutes to recompute. Don't
  "simplify" it back.
- **Reasoning models (deepseek-r1) need `num_predict` 4096**: the thinking channel eats
  a 300-token budget before any JSON is emitted. (Cost: a day of empty referee replies,
  Exp 49.)
- **Never edit modules a running sweep imports** (`trace_replay.py`, `llm_agent.py`,
  `referee.py`): each wave launches fresh processes, so mid-sweep edits mix code
  versions across seeds and poison cache comparability.
