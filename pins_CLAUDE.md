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

This is a `uv` project (Python 3.10). The git repo root is the user's home directory; this
`MCP/` directory is the actual project root — run all commands from here.

## Commands

```bash
# Use the project venv directly (uv run also works):
.venv/bin/python -m pins.test_mechanism            # unit-test the auctioneer (no network/LLM, instant)
bash pins/run_demo.sh                               # full negotiation: 1 server + 3 agents, 4 GPUs, no LLM
bash pins/run_demo.sh --llm                          # same, with LLM justifications (needs Ollama)
.venv/bin/python -m pins.eval.predict_resources      # Stage-1 eval vs baselines on benchmark.json (approx truth)
.venv/bin/python -m pins.eval.predict_resources --model qwen2.5:7b   # swap model for a size ablation

# Closed-loop Stage-1: predict, then TRAIN the model on the A100 and MEASURE real peak VRAM (needs torch+GPU):
.venv/bin/python -m pins.eval.predict_cnn                       # raw-LLM + hybrid(facts) vs heuristic/mean, measured
.venv/bin/python -m pins.eval.predict_cnn --reasoning --show-reasoning   # LLM walks layers; prints the trace
.venv/bin/python -m pins.eval.predict_cnn --deterministic               # LLM-shapes->code-sum, LOOCV (the winner)
.venv/bin/python -m pins.eval.predict_cnn --deterministic --precision fp16   # mixed-precision check
.venv/bin/python -m pins.eval.predict_arch                      # same recipe across CNN/ResNet/Transformer, 1 global calib

# Stage-1 DYNAMIC: forecast a running job's GPU/CPU/mem 5 min ahead on MIT Supercloud traces:
python data/fetch_supercloud.py --n-jobs 100 --min-gpu-mb 1 --max-gpu-mb 8  # pull a joint CPU+GPU sample (S3, no creds)
.venv/bin/python -m pins.forecast.dataset            # align cpu(10s)+gpu(100ms)->10s grid; sanity-print
.venv/bin/python -m pins.forecast.baselines          # persistence / moving-avg gate (per-channel + nMAE)
.venv/bin/python -m pins.forecast.model              # train+eval the residual attention forecaster vs the gate

# Stage-2 SIMULATION: which allocation MECHANISM rations GPUs best? (pure Python, no GPU, instant)
.venv/bin/python -m pins.negotiation_sim             # auction vs greedy/equal/static/committed sweep
.venv/bin/python -m pins.negotiation_sim --llm --model qwen2.5:3b   # + LLM-strategist & LLM-priority rows
.venv/bin/python -m pins.llm_agent                   # smoke: LLM bid-strategy + committed-priority per state
.venv/bin/python -m pins.llm_agent --no-llm          # same via the deterministic rule fallback
```

Run the negotiation system by hand (server first, then one agent process per job):

```bash
.venv/bin/python -m pins.negotiation_server --agents 3 --gpus 4 --transport sse
.venv/bin/python -m pins.job_agent --id jobA --urgency 1.3 --timeline preprocess,train,train,eval --no-llm
```

**Tests are a plain script, not pytest.** `pins/test_mechanism.py` auto-discovers and runs every
`test_*` function in `__main__`. To run a single test: `.venv/bin/python -c "from pins.test_mechanism import test_efficiency_total_value as t; t()"`.

## Environment assumptions (hard dependencies of the demos)

- **Ollama** running at `http://localhost:11434`; `qwen2.5` in `3b` (default), `7b`, `14b` are pulled.
- The negotiation server binds **SSE on `http://localhost:8000/sse`** (hardcoded in `job_agent.py` / `run_demo.sh`).
- Real-hardware validation + the closed-loop `eval/predict_cnn.py`/`predict_arch.py` target a single **A100-PCIE-40GB**.
- The LLM path degrades gracefully: the negotiation demo and `predict_resources` fall back if
  Ollama is down. **Only `predict_cnn.py`/`predict_arch.py` hard-require torch+CUDA** (they
  measure real VRAM); the rest of the project runs with no GPU.
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

### `pins/` — Stage-2 negotiation (MCP-based)

MCP is repurposed as an **agent-to-agent** substrate: it is natively LLM-client ↔ server, so all
job-agents are independent MCP *clients* that rendezvous through one shared, stateful MCP *server*.
A networked transport (SSE) is therefore mandatory — agents must see shared allocation state.

- `mechanism.py` — **pure** sealed-bid uniform-price auctioneer + anti-thrashing rescale gate.
  No MCP, no LLM, no network; this is the "decider" and the only thing the unit tests cover.
- `predictor.py` — Stage-1 **stub**: maps a job phase → non-increasing marginal-value curve.
  Encodes the project's premise (demand varies by phase). Meant to be replaced by the real hybrid predictor.
- `negotiation_server.py` — the shared MCP server: owns the GPU allocation table (state), exposes
  `market://state` (resource) + `register`/`submit_bid`/`round_result`/`get_allocation` (tools), and
  runs the auctioneer. Uses a **round/barrier protocol**: a round clears only when all expected
  agents have bid. **Gotcha: `--agents` MUST equal the number of agent processes**, or rounds never clear.
- `job_agent.py` — one MCP client per job: reads the market, asks `predictor.py` for a bid curve,
  optionally asks the LLM for a one-line justification, submits a *structured* bid, reads its allocation.

The clearing produces allocation *deltas*. Wiring those deltas to real rescaling (TorchElastic/SLURM)
is the actuation layer that lives **below** MCP and is not yet implemented.

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

**The LLM hinge applies here too:** `llm_agent.py` mirrors `forecast/llm_facts.py` — the LLM emits
only **categorical/ordinal** choices (a bidding `stance`+`focus_gpus`, or a priority *class*) plus a
justification, **never a number**; deterministic code owns every magnitude. It is kept out of the hot
loop by **caching one decision per discretised state** (`llm_agent_cache.json`), so a full sweep costs
~tens of Ollama calls, not thousands. Both LLM paths degrade gracefully to a rule on `--no-llm`/Ollama-down.

### `pins/eval/` — Stage-1 prediction evaluation

Two generations of harness. The first scores against *approximate* truth; the second **measures**
truth by actually training on the A100 (`torch.cuda.max_memory_allocated`). All write `results*.json`.

- `benchmark.json` + `predict_resources.py` — prompts an LLM (metadata only) for
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

### `pins/forecast/` + `data/` — Stage-1 DYNAMIC prediction (time-series, the active direction)

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

### Top-level `chat_*.py` — legacy prototype

`chat_server.py` (a message-board MCP server) + `chat_agent_a.py`/`chat_agent_b.py` (two Ollama
agents free-texting through it). Superseded by `pins/`; kept for reference. `main.py` is empty.

## Conventions

- Code is written to be *read by a researcher*: heavy docstrings tie each module back to specific
  lines in `research_plan.md`. Match that style — explain the "why" and the design link, not just the "what".
- Keep the pure decider (`mechanism.py`) free of MCP/network/LLM imports so it stays unit-testable
  and provable in isolation. Verify it (`test_mechanism`) before wiring anything above it.
