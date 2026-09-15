#!/bin/bash
# Bounded worker for one explicitly assigned reduced-oracle window.
set -u

REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/core_referee_oracle_reduced
PY=$REPO/.venv/bin/python
WORKER=${1:?usage: sweep_core_oracle_window_tick.sh WORKER SPLIT WINDOW}
SPLIT=${2:?}
WINDOW=${3:?}
LOG=$OUT/window.$WORKER.log

mkdir -p "$OUT"
exec 9>"$OUT/.window_lock.$WORKER"
flock -n 9 || exit 0
cd "$REPO" || exit 1

DONE=$OUT/done.$SPLIT.nm-mkt.$WORKER
[ -f "$DONE" ] && exit 0
echo "=== tick $(date -Is) worker=$WORKER split=$SPLIT window=$WINDOW" >> "$LOG"
"$PY" -m pins.core_oracle_labels --out "$OUT" --splits "$SPLIT" \
    --window-names "$WINDOW" --epochs 2 --shard 0 --shards 1 --worker-id "$WORKER" \
    --max-rollouts 8 >> "$LOG" 2>&1
