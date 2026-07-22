#!/usr/bin/env bash
# Exp 81 on the two reasoners, at FULL STRENGTH (thinking on, 4096-token budget).
# Resumable: skips a model whose result file exists.
set -uo pipefail
cd "$(dirname "$0")"
for m in deepseek-r1:32b qwen3.5:35b; do
  tag=${m//[:.]/}
  out=pins/results_exp81_${tag}.json
  [ -s "$out" ] && { echo "== skip $m"; continue; }
  echo "== $m -> $out  ($(date +%H:%M))"
  PINS_RESULTS="$out" .venv/bin/python -u -m pins.exp81_signed_2x2 \
      --suite r3 --model "$m" > "pins/exp81_${tag}.log" 2>&1
  echo "   exit=$? $(grep -c '^P3-' "pins/exp81_${tag}.log") cases  ($(date +%H:%M))"
done
echo "== exp81 reasoners done"
