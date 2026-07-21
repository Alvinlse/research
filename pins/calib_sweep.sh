#!/bin/bash
# Step 1 of the calibration plan: where does the FLOOR's SLA land as we vary cluster
# granularity (pool size, jobs scaled to hold offered load ~0.7) and deadline slack?
# Rule floor only -- no Ollama, no LLM. The point is to find an operating point where a
# scheduler has room to matter, instead of the saturated corner every prior Exp measured.
set -u
for slack in 1 2 4; do
  for pool in 8 16 32 64; do
    n=$(( pool * 28 / 10 ))            # n_jobs ~ 2.8 * pool holds load ~0.7
    echo "### slack=${slack}x pool=${pool} n_jobs=${n}"
    .venv/bin/python -m pins.trace_replay --caps predicted --pools "$pool" \
      --n-jobs "$n" --seeds 32 --slack-mult "$slack" 2>&1 \
      | tr '\r' '\n' | grep -E "^ +$pool +(no-llm|negotiated) "
  done
done
