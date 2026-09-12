#!/bin/bash
# One bounded chunk of the core 2x2 (plan phases 8 and 9). Driven by cron, NOT a long-running shell:
# the login node reaps a process at ~12-15 CPU-min and a background shell dies with the ssh session,
# while the LLM arms need ~16 h of wall time in total. Each tick runs a few cells, `pins.run_frozen`
# appends every finished run to rows.jsonl immediately and skips completed keys, so a lost tick costs
# at most the one run in flight.
#
# Cell order is deliberate: the deterministic baselines finish first, so a partial sweep is still a
# usable result, and the expensive multi-agent arm is last.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/frozen_2x2
PY=$REPO/.venv/bin/python
LOG=$OUT/tick.log
CELLS="fcfs:as_requested,least_laxity:as_requested,least_laxity:adaptive,market:as_requested,market:adaptive,least_laxity:by_size,market:by_size,policy_select:as_requested:nm,policy_select:as_requested:mkt,policy_negotiate:as_requested:nm,policy_negotiate:as_requested:mkt"
TOTAL=605        # 11 cells x 55 windows
mkdir -p "$OUT"

exec 9>"$OUT/.lock"
flock -n 9 || exit 0          # a tick arriving while the previous one works exits, never queues

cd "$REPO" || exit 1
[ -f "$OUT/.done" ] && exit 0

echo "=== tick $(date -Is)" >> "$LOG"
"$PY" -m pins.run_frozen --cells "$CELLS" --out "$OUT" --max-seconds 480 >> "$LOG" 2>&1

n=$(wc -l < "$OUT/rows.jsonl" 2>/dev/null || echo 0)
echo "rows $n / $TOTAL" >> "$LOG"
if [ "$n" -ge "$TOTAL" ]; then
    touch "$OUT/.done"
    crontab -l | grep -v sweep_2x2_tick | crontab -
    echo "=== ALL CELLS COMPLETE $(date -Is)" >> "$LOG"
fi
