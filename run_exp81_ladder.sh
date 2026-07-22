#!/usr/bin/env bash
# Exp 81 across the ladder. 7b first (explicitly asked for), then the models Exp 79
# disqualified for feasibility arithmetic -- the signed contract removes that barrier, so they
# can finally express judgement. RESUMABLE: a model with a result file is skipped.
set -uo pipefail
cd "$(dirname "$0")"

for m in qwen2.5:7b qwen2.5:3b gemma2:2b llama3:8b gemma2:9b qwen2.5:1.5b; do
  tag=${m//[:.]/}
  out=pins/results_exp81_${tag}.json
  if [ -s "$out" ]; then echo "== skip $m"; continue; fi
  echo "== $m -> $out"
  PINS_RESULTS="$out" .venv/bin/python -u -m pins.exp81_signed_2x2 \
      --suite r3 --model "$m" > "pins/exp81_${tag}.log" 2>&1
  echo "   exit=$? $(grep -c '^P3-' "pins/exp81_${tag}.log") cases"
done
echo "== exp81 ladder done"
