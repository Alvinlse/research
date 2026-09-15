#!/bin/bash
cd /import/gp-home.ciero/kimseng/Research || exit 1
OUT=runs/pilot_debate_v243_two
TOTAL=2
N=0
[ ! -f "$OUT/rows.jsonl" ] || N=$(wc -l < "$OUT/rows.jsonl")
printf 'completed: %d / %d windows\n' "$N" "$TOTAL"
echo "cron:      $(crontab -l 2>/dev/null | grep -c debate_v243_pilot_tick) entries"
RUNNING=0
if [ -f "$OUT/.lock" ]; then
  flock -n "$OUT/.lock" true || RUNNING=1
fi
echo "running:   $RUNNING tick(s)"
ERRORS=0
[ ! -f "$OUT/tick.log" ] || ERRORS=$(grep -c Traceback "$OUT/tick.log" || true)
echo "errors:    $ERRORS"
CUR=$(find runs/core_2x2_worlds -path '*/out/pilot_v243_s17_sim.log' -mmin -3 -type f \
  2>/dev/null | head -1)
if [ -n "$CUR" ]; then
  WINDOW=$(echo "$CUR" | cut -d/ -f3)
  PROGRESS=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '[0-9]*/[0-9]* jobs processed' | head -1)
  ETA=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '<[0-9hms:]*]' | tr -d '<]' | head -1)
  echo "current:   $WINDOW ${PROGRESS:-starting} eta ${ETA:-?}"
fi
echo "--- completed"
tail -3 "$OUT/tick.log" 2>/dev/null | grep -E 'left\)|complete|published' || true
