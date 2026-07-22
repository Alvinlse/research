#!/usr/bin/env bash
# Elevated plan §17 — the sensitivity grid, rule tier (no Ollama, minutes not hours).
#
# One cell per dimension value, each written to its own results file so nothing clobbers
# anything. The LLM tiers are NOT swept here: at ~40 min/cell they need a pre-registered
# subset, not a grid. Use this to find which dimensions move the outcome at all, then spend
# GPU time only on those.
#
# Usage:  bash pins/run_sensitivity.sh [SEEDS]        (default 32)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=${1:-32}
OUT=pins/sensitivity
mkdir -p $OUT

run() {   # run <cell-name> <extra flags...>
  local name=$1; shift
  echo "=== $name ==="
  PINS_RESULTS=$OUT/$name.json $PY -m pins.trace_replay --seeds "$S" "$@" \
    | tee $OUT/$name.log | grep -E "^ +[0-9]+  |vs floor|Holm"
}

# offered load rho: jobs per window against a fixed pool (the plan's 0.3 -> 1.3 sweep)
for N in 6 10 16 24 32; do run "rho_n$N" --pools 8 --n-jobs "$N"; done

# prediction-error regime: what the scheduler believes about demand
for C in oracle predicted real; do run "caps_$C" --pools 8 --caps "$C"; done

# SLA tightness lambda (deadline slack multiplier)
for L in 0.8 1.0 1.5 2.0; do run "lambda_$L" --pools 8 --slack-mult "$L"; done

# resize overhead: flat cost, per-GPU cost, and cooldown
run resize_none --pools 8
run resize_low  --pools 8 --realloc-cost 0.05
run resize_high --pools 8 --realloc-cost 0.15 --resize-c1 0.05
for K in 1 2 5; do run "cooldown_$K" --pools 8 --cooldown "$K"; done

# starvation protection
for P in 0 1 4; do run "phi_$P" --pools 4 --phi "$P"; done

# resource granularity + progress law (the §3.3 robustness pair)
run quantum_quarter --pools 8
run quantum_whole   --pools 8 --quantum whole
run law_sat         --pools 8 --law sat

echo "grid done -> $OUT/"
