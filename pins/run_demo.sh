#!/usr/bin/env bash
# End-to-end PINS negotiation demo: 1 server + 3 job-agents contending for 4 GPUs.
# Usage:  bash pins/run_demo.sh [--llm]
# Run from the MCP/ project root.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LLM_FLAG="--no-llm"
[[ "${1:-}" == "--llm" ]] && LLM_FLAG=""

echo "[demo] starting negotiation server (4 GPUs, 3-agent barrier)…"
$PY -m pins.negotiation_server --agents 3 --gpus 4 --rescale-cost 0.5 --transport sse \
    >/tmp/pins_server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 3  # let SSE bind to :8000

# Three jobs with different phase timelines and priorities (private valuations).
#  jobA: long training run (high urgency)
#  jobB: starts heavy, then winds down to eval
#  jobC: light preprocess-heavy pipeline
$PY -m pins.job_agent --id jobA --urgency 1.3 --timeline preprocess,train,train,eval   $LLM_FLAG &
$PY -m pins.job_agent --id jobB --urgency 1.0 --timeline train,train,eval,idle         $LLM_FLAG &
$PY -m pins.job_agent --id jobC --urgency 0.8 --timeline preprocess,preprocess,train,train $LLM_FLAG &
wait $(jobs -p | grep -v $SERVER_PID) 2>/dev/null || true

echo
echo "[demo] final market state:"
$PY - <<'PY'
import asyncio, json
from mcp import ClientSession
from mcp.client.sse import sse_client
async def main():
    async with sse_client("http://localhost:8000/sse") as (r,w):
        async with ClientSession(r,w) as s:
            await s.initialize()
            res = await s.call_tool("status", {})
            print(json.dumps(json.loads(res.content[0].text), indent=2))
asyncio.run(main())
PY
echo "[demo] server log -> /tmp/pins_server.log"
