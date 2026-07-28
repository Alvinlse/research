#!/bin/bash
# Exp 55 (NOT YET RUN) — does the cross-talk round earn its cost?
#
# Rows: no-llm / referee / debate / negotiated, paired on the same seeds.
# Both referee and debate take the SAME advocates (3b, the default), r1:32b rules both, so the
# referee-vs-debate delta isolates the round itself — not the advocate model. The old
# hardcoded split (3b advocates for referee, --model advocates for debate) confounded the two.
#
# Advocates left at the 3b default ON PURPOSE (decided 2026-07-19): Exp 50 already computed
# `deepseek-r1:32b+referee` at 3b advocates, pool 8, n=32, same spike/scale/clip — so the
# referee AND negotiated rows replay from the LLM cache and only `debate` costs new inference.
# Passing `--advocates qwen2.5:14b` instead makes all three rows cold: measured 966s/seed,
# ~8h per row, ~20h total. Caveat to carry into the write-up: 3b is a weak arguer (Exp 33,
# Exp 51), so a null concession result is "3b had nothing to say" as much as "the round is
# decoration". Re-run with --advocates qwen2.5:14b if the 3b pilot shows signal.
#
# Read before running: ollama MUST be at OLLAMA_NUM_PARALLEL=1 or r1:32b spills to CPU (~6x).
#   ~/.local/bin/ollama ps    # want "100% GPU"
#   pkill -x ollama; OLLAMA_NUM_PARALLEL=1 setsid nohup ~/.local/bin/ollama serve >~/.ollama/serve.log 2>&1 &
# Do NOT use run_parallel_sweep.sh here — it restarts ollama with NUM_PARALLEL=$WORKERS.
# One sweep per background task (login-node reaper kills shells at ~12-15 CPU-min).
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-32}
PINS_NUM_CTX=${PINS_NUM_CTX:-8192} .venv/bin/python -u -m pins.trace_replay \
    --referee --debate --llm \
    --model deepseek-r1:32b \
    --pools 8 --seeds "$SEEDS"
# Tier written: deepseek-r1:32b+referee — the SAME tier Exp 50 wrote, now gaining a `debate`
# row. Back up results_trace_replay.json first; the referee/negotiated rows are rewritten
# (identically, from cache) rather than appended.
# Smoke first (seconds, no GPU): drop --llm, it's opt-in. This script does not forward "$@",
# so passing flags to it is silently ignored — run the module directly:
#   PINS_RESULTS=$(mktemp -u) .venv/bin/python -m pins.trace_replay \
#     --referee --debate --model deepseek-r1:32b --pools 8 --seeds 1
# Metric to check beyond SLA: concession rate — referee.py records the opening ask in
# `_r0_gpus`. Nobody moves => the round is decoration; everyone caves => sycophancy.
