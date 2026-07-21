#!/bin/bash
# Targeted arm of the calibration: the slack axis at cheap pools, since Exp 62's first block
# showed pool size moves the floor only 68->65->62 while utilisation stays pinned at 87-89%.
set -u
for slack in 2 4 8; do
  for pool in 8 16; do
    n=$(( pool * 28 / 10 ))
    echo "### slack=${slack}x pool=${pool} n_jobs=${n}"
    .venv/bin/python -m pins.trace_replay --caps predicted --pools "$pool" \
      --n-jobs "$n" --seeds 32 --slack-mult "$slack" 2>&1 \
      | tr '\r' '\n' | grep -E "^ +$pool +(no-llm|negotiated) "
  done
done
