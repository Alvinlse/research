# PINS — negotiation layer (prototype)

Stage-2 of the PINS plan (`../research_plan.md`): prediction-informed, agent-negotiated
GPU allocation, built on MCP. The LLM **reasons/explains**; a deterministic
auctioneer on the server **decides** — the LLM is never in the clearing hot loop.

## Architecture (maps to the Step 1–9 design)

```
job-agent (MCP client + occasional LLM)  ─┐
job-agent (MCP client + occasional LLM)  ─┼─ SSE ─►  negotiation_server (MCP server)
job-agent (MCP client + occasional LLM)  ─┘            ├─ state: GPU allocation table
                                                       ├─ resource: market://state
                                                       ├─ tools: register/submit_bid/...
                                                       └─ auctioneer (mechanism.py, no LLM)
```

| File | Role | Step |
|---|---|---|
| `mechanism.py` | pure sealed-bid uniform-price auctioneer + anti-thrashing gate | 3, 8 |
| `predictor.py` | demo-only stub: phase → marginal-value curve (the real Stage-1 reaches Stage-2 via `trace_replay.py`) | 5 |
| `negotiation_server.py` | networked stateful MCP server: state + mechanism + round/barrier protocol | 2,3,6,7 |
| `job_agent.py` | MCP client wrapping Ollama; reads market, bids, reads allocation | 4 |
| `test_mechanism.py` | deterministic unit tests for the decider (no MCP/network) | — |
| `run_demo.sh` | 1 server + 3 agents contending for 4 GPUs, end-to-end | 8 |

## Run

```bash
# from the Research/ project root
.venv/bin/python -m pins.test_mechanism      # verify the auctioneer (instant)
bash pins/run_demo.sh                         # full negotiation, no LLM (fast, deterministic)
bash pins/run_demo.sh --llm                   # same, with LLM justifications (needs Ollama)
```

Or by hand:

```bash
.venv/bin/python -m pins.negotiation_server --agents 3 --gpus 4 --transport sse
.venv/bin/python -m pins.job_agent --id jobA --urgency 1.3 --timeline preprocess,train,train,eval --no-llm
# ...one job_agent process per job; --agents must equal the number of agents.
```

## What this prototype is / isn't

- **Is:** a faithful, runnable substrate for Stage-2 negotiation — structured bids,
  a provable clearing rule, event/round triggering, anti-thrashing, concurrency-safe state.
- **Isn't (yet):** connected to real actuation. `round_result` deltas would feed
  TorchElastic/SLURM (below MCP, `research_plan.md:62`). `predictor.py` is a stub
  used only by the demo.

## Where the real Stage-1 enters

Not through `predictor.py`. The evaluated pipeline is `trace_replay.py`, which replays
real Alibaba v2020 jobs and feeds the negotiation the quantile-GBT predictions written
to `eval/pred_job_{runtime,usage,mem}.csv` by `pins.eval.predict_gpu`. The P10–P90
interval is what sizes the agents' request margin.

## Next steps

- Decide round-based auction vs. peer-to-peer bargaining (Open Decision #1).
- Wire `deltas` to a (simulated first) TorchElastic rescale; measure goodput vs. baselines.
