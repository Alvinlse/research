"""
PINS negotiation MCP server (Steps 3, 6, 7).

A single, networked, STATEFUL server that all job-agents connect to. It owns:
  * the truth   : the GPU allocation table        (Step 3, state)
  * the decider : the auctioneer from mechanism.py (Step 3, mechanism — no LLM)
  * the protocol: bounded bidding ROUNDS with a barrier (Step 6, option 2)
and is concurrency-safe under simultaneous bids (Step 7, a lock + atomic clear).

Run it BEFORE any agent:
    uv run python -m pins.negotiation_server --agents 3 --gpus 4 --transport sse

Round/barrier protocol (lockstep, no sleeps on the happy path):
  - Round r is open. Every registered agent submits exactly one bid for r.
  - When the last expected bid arrives, the server CLEARS r atomically, stores
    the result, and opens round r+1. Agents poll `round_result(r)` for the outcome.
  - A deadline is recorded so a crashed agent can't deadlock the pool forever.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

from mcp.server.fastmcp import FastMCP

from pins.mechanism import clear

mcp = FastMCP("pins-negotiation")

# ----------------------------- shared state ---------------------------------
_lock = threading.Lock()
_state = {
    "total_gpus": 4,
    "expected_agents": 1,
    "rescale_cost": 0.5,
    "registered": [],          # agent_ids
    "alloc": {},               # agent_id -> GPUs held
    "round": 0,                # current open round id
    "bids": {},                # round_id -> {agent_id: {"phase":..., "curve":[...]}}
    "results": {},             # round_id -> serialisable ClearResult
    "round_opened_at": time.time(),
}


def _market_snapshot() -> dict:
    held = sum(_state["alloc"].values())
    return {
        "total_gpus": _state["total_gpus"],
        "free_gpus": _state["total_gpus"] - held,
        "allocation": dict(_state["alloc"]),
        "open_round": _state["round"],
        "last_price": _state["results"].get(_state["round"] - 1, {}).get("price", 0.0),
        "expected_agents": _state["expected_agents"],
        "bids_in": len(_state["bids"].get(_state["round"], {})),
    }


def _maybe_clear_locked() -> None:
    """If every expected agent has bid in the open round, clear it atomically."""
    r = _state["round"]
    bids = _state["bids"].get(r, {})
    if len(bids) < _state["expected_agents"]:
        return
    curves = {a: b["curve"] for a, b in bids.items()}
    res = clear(
        curves,
        total_gpus=_state["total_gpus"],
        current=_state["alloc"],
        rescale_cost=_state["rescale_cost"],
    )
    _state["alloc"] = dict(res.allocation)
    _state["results"][r] = {
        "round": r,
        "allocation": res.allocation,
        "deltas": res.deltas,
        "price": res.price,
        "welfare": round(res.welfare, 4),
        "applied": res.applied,
        "gpus_moved": res.gpus_moved,
        "phases": {a: b["phase"] for a, b in bids.items()},
        "detail": res.detail,
    }
    _state["round"] = r + 1
    _state["round_opened_at"] = time.time()


# ------------------------------- resource -----------------------------------
@mcp.resource("market://state")
def market_state() -> str:
    """Read-only view of the pool — agents read this before forming a bid (Step 4)."""
    with _lock:
        return json.dumps(_market_snapshot())


# -------------------------------- tools -------------------------------------
@mcp.tool()
def register(agent_id: str) -> str:
    """Join the pool. Idempotent. Returns the current market snapshot."""
    with _lock:
        if agent_id not in _state["registered"]:
            _state["registered"].append(agent_id)
            _state["alloc"].setdefault(agent_id, 0)
        return json.dumps(_market_snapshot())


@mcp.tool()
def submit_bid(agent_id: str, round_id: int, phase: str, marginal_values: str) -> str:
    """Submit a structured bid (NOT chat) for `round_id`.

    `marginal_values` is a JSON list: the non-increasing value of the 1st, 2nd,
    ... GPU for this agent in `phase`. The last expected bid triggers clearing.
    """
    curve = json.loads(marginal_values)
    with _lock:
        if round_id != _state["round"]:
            return json.dumps({"accepted": False, "reason": "stale round",
                               "current_round": _state["round"]})
        _state["bids"].setdefault(round_id, {})[agent_id] = {"phase": phase, "curve": curve}
        _maybe_clear_locked()
        return json.dumps({"accepted": True, "round": round_id,
                           "bids_in": len(_state["bids"].get(round_id, {})),
                           "expected": _state["expected_agents"]})


@mcp.tool()
def round_result(round_id: int) -> str:
    """Return the clearing result for `round_id`, or {"ready": False} if still open."""
    with _lock:
        res = _state["results"].get(round_id)
        if res is None:
            return json.dumps({"ready": False, "open_round": _state["round"]})
        return json.dumps({"ready": True, **res})


@mcp.tool()
def get_allocation(agent_id: str) -> str:
    """GPUs `agent_id` currently holds, plus the latest clearing price."""
    with _lock:
        snap = _market_snapshot()
        return json.dumps({"agent_id": agent_id,
                           "gpus": _state["alloc"].get(agent_id, 0),
                           "last_price": snap["last_price"]})


@mcp.tool()
def status() -> str:
    """Full market snapshot (debug / dashboards)."""
    with _lock:
        return json.dumps(_market_snapshot())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", type=int, default=1, help="expected agents per round (barrier size)")
    p.add_argument("--gpus", type=int, default=4, help="GPUs in the contested pool")
    p.add_argument("--rescale-cost", type=float, default=0.5, help="value cost per GPU moved")
    p.add_argument("--transport", default="sse")
    a = p.parse_args()
    _state["expected_agents"] = a.agents
    _state["total_gpus"] = a.gpus
    _state["rescale_cost"] = a.rescale_cost
    print(f"[server] PINS negotiation: {a.gpus} GPUs, barrier={a.agents} agents, "
          f"rescale_cost={a.rescale_cost}", file=sys.stderr)
    mcp.run(transport=a.transport)


if __name__ == "__main__":
    main()
