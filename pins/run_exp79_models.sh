#!/usr/bin/env bash
# Exp 79 — run the round-3 2x2 (40 cases x 4 arms) across the model ladder.
# RESUMABLE: a model whose result file already exists is skipped, so the Tohoku login node's
# background-CPU reaper can kill this mid-ladder and a plain re-run continues where it stopped.
#
#   bash pins/run_exp79_models.sh                  # tier 1 (fast, <=9 GB)
#   bash pins/run_exp79_models.sh slow             # tier 2 (27b/32b/35b, hours)
set -uo pipefail
cd "$(dirname "$0")/.."

FAST=(qwen2.5:1.5b gemma2:2b qwen2.5:3b qwen2.5:7b llama3:8b gemma2:9b)
SLOW=(gemma2:27b deepseek-r1:32b qwen3.5:35b)
MODELS=("${FAST[@]}"); [ "${1:-}" = slow ] && MODELS=("${SLOW[@]}")

for m in "${MODELS[@]}"; do
  tag=${m//[:.]/}
  out=pins/results_hardcases_r3_2x2_${tag}.json
  if [ -s "$out" ]; then echo "== skip $m (have $out)"; continue; fi
  echo "== $m -> $out"
  # --no-think: r1 is the only hybrid reasoner here and ollama 400s on the rest; referee.py
  # gates the API flag by model, so this is uniform across the ladder.
  PINS_RESULTS="$out" .venv/bin/python -u -m pins.hardcase_eval \
      --suite r3 --arms single,single-noarg,referee,referee-noarg \
      --model "$m" --no-think > "pins/exp79_${tag}.log" 2>&1
  echo "   exit=$? $(grep -c . "pins/exp79_${tag}.log") log lines"
done
echo "== done; analyse with: .venv/bin/python -m pins.exp79_analyse"
