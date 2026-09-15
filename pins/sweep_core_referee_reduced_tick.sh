#!/bin/bash
# One resumable shard tick for the reduced current-space oracle dataset.
set -u

REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/core_referee_oracle_reduced
PY=$REPO/.venv/bin/python
SHARD=${1:?usage: sweep_core_referee_reduced_tick.sh SHARD}
SHARDS=4
CHUNK=8
LOG=$OUT/tick.$SHARD.log

mkdir -p "$OUT"
[ -f "$OUT/.pause.$SHARD" ] && exit 0
exec 9>"$OUT/.lock.$SHARD"
flock -n 9 || exit 0
cd "$REPO" || exit 1

echo "=== tick $(date -Is) shard=$SHARD" >> "$LOG"

if [ ! -f "$OUT/done.train.$SHARD" ]; then
    "$PY" -m pins.core_oracle_labels --out "$OUT" --splits train --limit 12 --epochs 2 \
        --shard "$SHARD" --shards "$SHARDS" --max-rollouts "$CHUNK" >> "$LOG" 2>&1
    [ -f "$OUT/done.train.nm-mkt.$SHARD" ] && touch "$OUT/done.train.$SHARD"
fi

if [ -f "$OUT/done.train.$SHARD" ] && [ ! -f "$OUT/done.validation.$SHARD" ]; then
    "$PY" -m pins.core_oracle_labels --out "$OUT" --splits validation --limit 6 --epochs 2 \
        --shard "$SHARD" --shards "$SHARDS" --max-rollouts "$CHUNK" >> "$LOG" 2>&1
    [ -f "$OUT/done.validation.nm-mkt.$SHARD" ] && touch "$OUT/done.validation.$SHARD"
fi

if [ -f "$OUT/done.train.$SHARD" ] && [ -f "$OUT/done.validation.$SHARD" ]; then
    touch "$OUT/worker_done.$SHARD"
fi
