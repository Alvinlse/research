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
| `policy_debate.py` | post-core experimental arm: weighted Demand/Supply openings, machine-readable red lines, one bounded rebuttal, candidate-constrained ratification, strict fallback, and an audit package |
| `policy_debate_v2.py` | exploratory cross-window revision: disjoint advocate roles, online episode state, role-enforced objections, and mandatory component-constrained Referee |
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
.venv/bin/python -m pins.elastisim_bench run --world <world> --arm policy_debate --family mkt \
  --fallback-ordering auction_deadline --fallback-sizing adaptive
.venv/bin/python -m pins.elastisim_bench run --world <world> --arm policy_debate_v2 --family mkt \
  --fallback-ordering auction_deadline --fallback-sizing adaptive
```

Every `--llm` arm falls back to a deterministic rule if Ollama (`localhost:11434`) is down.
The unfrozen `policy_debate` follow-up uses two calls when the openings agree, four when rebuttals
converge, and five when a Referee must resolve two surviving candidates. Demand and Supply receive
disclosed incentive vectors with $L_1$ distance 1.3 and code-evaluated red lines. Consensus is
ratified directly; a Referee cannot invent or splice a third policy. The arm logs incentives,
red-line state, openings, validated rebuttals, candidates, concessions, dissent, rejected outputs,
the executed policy, fallback use, and residual risks in `<tag>_policy_log.json`. It has no result
claim yet and must receive a fresh pre-registration and evaluation split before measurement.

`policy_debate_v2` is a separately named exploratory successor; it does not alter or resume the
frozen v1 arm. Demand may defend deadline or production-priority ordering, while Supply may defend
fairness or accumulated-wait ordering. Both also propose a bounded sizing position. If both
openings validate, a Referee always selects from their component-wise cross-product; if either
opening is invalid, code executes the configured deterministic floor. During a non-trivial backlog,
material production pressure latches a Demand ordering episode; otherwise a batch-dominated
majority-user burst latches a Supply fairness episode. An episode keeps that role's ordering
binding until the backlog drains, while the Referee still chooses sizing. This is exactly three LLM
calls for a valid epoch and two calls for a failed opening, with no rebuttal round. Its added state
is online-observable (deadline-budget distribution, user concentration, production share, recent
queue trend, and episode memory), not true duration or realised evaluation outcomes. The real-window
checks in `debate_cross_window_review.md` are post-hoc diagnostics, not a validation claim; v2.2 needs
a new pre-registration before comparative measurement. The completed 24-window v1 audit further
shows that its majority-user guard is too broad to promote: v2.2 is retained as a rejected
development prototype, while the recommended next rule treats concentration as an objection and
requires observed deterioration before any binding Supply escalation.

## What this is / isn't

- **Is:** the evaluated Stage-2 substrate — structured bids, a provable clearing rule, gated
  LLM escalation, anti-thrashing, and the replay harness the results come from.
- **Isn't:** connected to real actuation. Clearing produces allocation *deltas*; wiring them to
  TorchElastic/SLURM (`research_plan.md:62`) is not implemented.

## Where Stage-1 enters

Not through `predictor.py`. `trace_replay.py` feeds the negotiation the quantile-GBT predictions
written to `eval/pred_job_{runtime,usage,mem}.csv` by `pins.eval.predict_gpu`. The P10–P90
interval is what sizes the agents' request margin.
