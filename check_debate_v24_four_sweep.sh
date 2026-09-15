#!/bin/bash
cd /import/gp-home.ciero/kimseng/Research || exit 1
OUT=runs/sweep_debate_v24_next4
TOTAL=4
bar() {
  local done=$1 total=${2:-1} width=${3:-34} filled i out=""
  [ "$total" -gt 0 ] || total=1
  filled=$((done * width / total))
  for ((i=0; i<width; i++)); do
    [ "$i" -lt "$filled" ] && out="${out}#" || out="${out}."
  done
  printf '[%s] %3d%%' "$out" "$((done * 100 / total))"
}
N=0
[ ! -f "$OUT/rows.jsonl" ] || N=$(wc -l < "$OUT/rows.jsonl")
echo "overall:   $(bar "$N" "$TOTAL")  $N / $TOTAL windows"
if [ "$N" -gt 0 ]; then
  .venv/bin/python - "$OUT/rows.jsonl" "$TOTAL" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()
        if line.strip()]
remaining = max(0, int(sys.argv[2]) - len(rows))
mean_s = sum(row['wall_s'] for row in rows) / len(rows)
seconds = round(mean_s * remaining)
hours, seconds = divmod(seconds, 3600)
minutes, seconds = divmod(seconds, 60)
eta = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"
print(f"mean/run:  {round(mean_s / 60, 1)}m")
print(f"est left:  {eta}")
PY
else
  echo "mean/run:  waiting for the first completed window"
  echo "est left:  available after the first completed window"
fi
echo "cron:      $(crontab -l 2>/dev/null | grep -c debate_v24_four_sweep_tick) entries"
echo "running:   $(ps -ef | grep -F -- 'debate_v24_four_sweep_tick' | grep -v grep | wc -l) tick(s)"
ERRORS=0
[ ! -f "$OUT/tick.log" ] || ERRORS=$(grep -c Traceback "$OUT/tick.log" || true)
echo "errors:    $ERRORS"
CUR=$(find runs/core_2x2_worlds -path '*/out/sw_v24_s17_sim.log' -mmin -3 -type f \
  2>/dev/null | head -1)
if [ -n "$CUR" ]; then
  WINDOW=$(echo "$CUR" | cut -d/ -f3)
  PROGRESS=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '[0-9]*/[0-9]* jobs processed' | head -1)
  ETA=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '<[0-9hms:]*]' | tr -d '<]' | head -1)
  DONE=${PROGRESS%%/*}
  REST=${PROGRESS#*/}; REST=${REST%% *}
  if [ -n "$PROGRESS" ]; then
    echo "current:   $(bar "${DONE:-0}" "${REST:-1}")  $WINDOW $PROGRESS eta ${ETA:-?}"
  else
    echo "current:   $WINDOW starting, eta ?"
  fi
fi
echo "--- completed"
tail -4 "$OUT/tick.log" 2>/dev/null | grep -E 'left\)|complete'
