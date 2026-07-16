#!/bin/bash
# Parallel Ollama sweep with chained peeks — 2-4x faster wall-clock on one A100,
# printing an intermediate result table every WORKERS seeds.
#
# How: restart ollama with WORKERS parallel slots, then warm the LLM cache in
# WAVES — WORKERS single-seed runs in parallel (results to throwaway files; only
# the shared llm_agent_cache.json matters, merged per seed). After each wave the
# banked seeds are contiguous, so a peek replay of seeds 0..n is pure cache hits:
# it prints the paired table (dSLA/dprodSLA/dutil/dslow, CI shrinking as n grows)
# in seconds without touching the GPU. A final canonical run writes the real
# results file — bit-identical to a serial sweep, no merge step.
#
# VRAM: weights ~20GB + WORKERS x NUM_CTX of KV. 4 x 8192 on the 40GB A100
# matches today's single 32k slot; don't raise both.
#
# Usage:  MODEL=deepseek-r1:32b SEEDS=32 POOLS=8 WORKERS=4 bash pins/run_parallel_sweep.sh
#   ARGS         extra trace_replay args (default "--referee --llm")
#   SKIP_OLLAMA  =1 to keep the current ollama server (testing / already parallel)
set -euo pipefail
MODEL=${MODEL:-deepseek-r1:32b}
SEEDS=${SEEDS:-32}
WORKERS=${WORKERS:-4}
POOLS=${POOLS:-8}
ARGS=${ARGS:---referee --llm}
NUM_CTX=${NUM_CTX:-8192}
cd "$(dirname "$0")/.."

if [ -z "${SKIP_OLLAMA:-}" ]; then
    if pgrep -f "python.*pins\.trace_replay" >/dev/null; then
        echo "a trace_replay run is active — refusing to restart ollama under it" >&2
        exit 1
    fi
    pkill -x ollama 2>/dev/null && sleep 3 || true
    OLLAMA_NUM_PARALLEL=$WORKERS nohup ollama serve >/dev/null 2>&1 &
    until curl -sf localhost:11434/api/version >/dev/null; do sleep 1; done
    echo "ollama up with $WORKERS parallel slots"
fi

run() {  # run <extra args...> — one trace_replay invocation with the sweep's params
    PINS_NUM_CTX=$NUM_CTX .venv/bin/python -u -m pins.trace_replay \
        $ARGS --model "$MODEL" --pools "$POOLS" "$@"
}

for ((base = 0; base < SEEDS; base += WORKERS)); do
    pids=()
    for ((s = base; s < base + WORKERS && s < SEEDS; s++)); do
        PINS_RESULTS=$(mktemp -u) run --seed-start "$s" --seeds $((s + 1)) \
            > "/tmp/pins_warmer_seed$s.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    n=$(( base + WORKERS < SEEDS ? base + WORKERS : SEEDS ))
    if ((n < SEEDS)); then           # final wave's table comes from the canonical run
        echo "=== peek @ n=$n (cache replay) ==="
        PINS_RESULTS=$(mktemp -u) run --seeds "$n" | tail -10
    fi
done

echo "=== canonical run @ n=$SEEDS -> real results file ==="
run --seeds "$SEEDS"
