"""
PINS job-agent — an MCP CLIENT wrapping an (occasional) LLM (Step 4 / Step 8).

Each malleable HPC job is one agent process. Per round it:
  1. reads market://state                              (resource)
  2. advances to its current phase, asks the Stage-1 predictor for a bid curve
  3. (optional) asks the LLM for a ONE-LINE justification  <- the only NL left
  4. submit_bid(...)                                   (tool)
  5. polls round_result(r) for the cleared outcome     (tool)
  6. get_allocation(me) -> would hand deltas to actuation (TorchElastic), below MCP

The LLM is at the EDGE and runs at most once per round (a phase event), never in
the clearing hot loop — research_plan.md:50,60. With --no-llm it is skipped
entirely so the whole system runs with zero GPU/Ollama dependency.

Usage (after the server is up):
    uv run python -m pins.job_agent --id jobA --timeline preprocess,train,train,eval
"""
from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.sse import sse_client

from pins.predictor import marginal_values

SERVER_URL = "http://localhost:8000/sse"


async def _tool(session: ClientSession, name: str, args: dict) -> dict:
    res = await session.call_tool(name, args)
    text = res.content[0].text if res.content else "{}"
    return json.loads(text)


async def _read_market(session: ClientSession) -> dict:
    res = await session.read_resource("market://state")
    return json.loads(res.contents[0].text)


def _justify(agent_id: str, phase: str, curve: list[float], market: dict, model: str) -> str:
    """Optional LLM 'explain' role. Non-fatal if Ollama is unavailable."""
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434")
        prompt = (
            f"You are {agent_id}, an HPC job currently in its '{phase}' phase. "
            f"Your marginal GPU values are {curve}. The pool has {market['free_gpus']} "
            f"of {market['total_gpus']} GPUs free. In ONE short sentence, justify your bid."
        )
        out = client.chat(model=model, messages=[{"role": "user", "content": prompt}],
                          options={"num_predict": 40})
        return out.message.content.strip().replace("\n", " ")
    except Exception as e:  # ollama down / model missing -> skip gracefully
        return f"(no llm: {type(e).__name__})"


async def run(agent_id: str, timeline: list[str], urgency: float,
              use_llm: bool, model: str) -> None:
    async with sse_client(SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await _tool(session, "register", {"agent_id": agent_id})
            print(f"[{agent_id}] registered; timeline={timeline}")

            for step, phase in enumerate(timeline):
                market = await _read_market(session)
                r = market["open_round"]
                curve = marginal_values(phase, urgency=urgency)

                note = _justify(agent_id, phase, curve, market, model) if use_llm else ""
                tag = f"  // {note}" if note else ""
                print(f"[{agent_id}] round {r}: phase={phase} bid={curve}{tag}")

                ack = await _tool(session, "submit_bid", {
                    "agent_id": agent_id, "round_id": r,
                    "phase": phase, "marginal_values": json.dumps(curve),
                })
                if not ack.get("accepted"):
                    print(f"[{agent_id}] bid rejected: {ack}")

                # Barrier: wait for the round to clear.
                for _ in range(60):
                    res = await _tool(session, "round_result", {"round_id": r})
                    if res.get("ready"):
                        break
                    await asyncio.sleep(0.25)
                else:
                    print(f"[{agent_id}] round {r} never cleared; stopping")
                    return

                mine = res["allocation"].get(agent_id, 0)
                moved = "applied" if res["applied"] else "held (anti-thrash)"
                print(f"[{agent_id}] round {r} cleared -> I hold {mine} GPUs "
                      f"(price={res['price']}, {moved}, full={res['allocation']})")

            print(f"[{agent_id}] done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--timeline", default="preprocess,train,train,eval",
                   help="comma-separated phases, one per round")
    p.add_argument("--urgency", type=float, default=1.0, help="private priority multiplier")
    p.add_argument("--no-llm", action="store_true", help="skip the LLM justification step")
    p.add_argument("--model", default="qwen2.5:3b")
    a = p.parse_args()
    asyncio.run(run(a.id, a.timeline.split(","), a.urgency, not a.no_llm, a.model))


if __name__ == "__main__":
    main()
