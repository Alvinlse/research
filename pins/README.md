# PINS — Stage-2 negotiation layer

Stage-2 of the PINS plan (`../research_plan.md`): prediction-informed, agent-negotiated GPU
allocation. The LLM **reasons/explains**; a deterministic auctioneer **decides** — the LLM is
never in the clearing hot loop.

> **The MCP prototype is gone.** `negotiation_server.py`, `job_agent.py` and `run_demo.sh` (a
> networked SSE server with one MCP client per job) were deleted in `1534db2`, "trim to
> paper-relevant code and results". Everything here is now pure in-process Python — no MCP, no
> network, no GPU. The research results always came from the simulators, not the live wiring.

## Architecture (Exp 84–87, what the paper is built on)

A validated auction runs **every tick**; the LLM is escalated **on trigger** and may only
*correct* what the market already decided. Nobody generates an allocation from scratch.

```
per tick:  jobs ──► market.clear_market ──► A_bid ──► placement/ilp validate ──► allocation
                                              │
                        gate fires (~15.7%)   ▼
                    packet.py (budget + action menu) ──► debate ──► correction_signed
                                                                        │
                                                    accept / shrink only ┘
```

| File | Role |
|---|---|
| `mechanism.py` | pure sealed-bid uniform-price auctioneer + anti-thrashing gate — the decider, and the only thing the unit tests cover |
| `market.py` | the explicit GPU market: real supply ask + clearing condition (elevated plan §6) |
| `ilp.py` | LLMSched-style ILP allocator — drop-in alternative decider to the auction |
| `placement.py` | node placement + repair; the auction clears a GPU *count*, this makes it physically placeable |
| `packet.py` | the referee packet: structured decision document with a code-generated action menu and the budget (fixes the "cannot fund" failure) |
| `correction.py` | bid-first correction — agents propose a delta to the market's allocation |
| `correction_signed.py` | signed (up **and** down) corrections; the arm that actually wins |
| `referee.py` | referee-LLM allocator (2026-07-15 pivot) — supersedes the bilateral ladder |
| `negotiation_protocol.py` | the older bounded two-sided concession ladder; retained as a baseline arm |
| `trace_replay.py` | replays real Alibaba v2020 windows — arrivals, durations, GPU demand jointly from the trace |
| `two_sided_sim.py` | merged two-sided world: demand margin + supply reserve on the SAME free pool |
| `llm_agent.py` | LLM bid-strategy / priority class, cached per discretised state |
| `bridge.py` | Stage-1 → Stage-2: turns calibrated numbers into the qualitative buckets the agents reason over |
| `predictor.py` | legacy phase → marginal-value-curve stub; superseded by `trace_replay.py` |
| `test_mechanism.py` | deterministic unit tests for the decider |

`hardcases*.py` + `referee_eval.py` are the pre-registered exception-scene suites; the
`exp*_*.py` files are per-experiment harnesses and analysers, each named for its log entry in
`../research_progress.md`.

## Run

```bash
# from the Research/ project root
.venv/bin/python -m pins.test_mechanism      # verify the auctioneer (instant, no LLM)
.venv/bin/python -m pins.trace_replay        # replay v2020 jobs against a GPU pool, no LLM
.venv/bin/python -m pins.trace_replay --llm --model qwen2.5:3b --pools 32 --seeds 8
.venv/bin/python -m pins.two_sided_sim       # two-sided demand/supply sweep
.venv/bin/python -m pins.negotiation_sim     # mechanism sweep in the synthetic world
```

Every `--llm` arm falls back to a deterministic rule if Ollama (`localhost:11434`) is down.

## What this is / isn't

- **Is:** the evaluated Stage-2 substrate — structured bids, a provable clearing rule, gated
  LLM escalation, anti-thrashing, and the replay harness the results come from.
- **Isn't:** connected to real actuation. Clearing produces allocation *deltas*; wiring them to
  TorchElastic/SLURM (`research_plan.md:62`) is not implemented.

## Where Stage-1 enters

Not through `predictor.py`. `trace_replay.py` feeds the negotiation the quantile-GBT predictions
written to `eval/pred_job_{runtime,usage,mem}.csv` by `pins.eval.predict_gpu`. The P10–P90
interval is what sizes the agents' request margin.
