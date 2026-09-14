#!/bin/bash
# Status of the bounded-debate training sweep (pins/debate_sweep_prereg.md).
cd /import/gp-home.ciero/kimseng/Research
OUT=runs/sweep_train_debate
bar() {  # bar <done> <total> [width]
  local d=$1 t=${2:-1} w=${3:-34} f i out=""
  [ "$t" -gt 0 ] || t=1
  f=$(( d * w / t ))
  for ((i=0;i<w;i++)); do [ $i -lt $f ] && out="$out#" || out="$out."; done
  printf '[%s] %3d%%' "$out" $(( d * 100 / t ))
}
N=$(wc -l < $OUT/rows.jsonl 2>/dev/null || echo 0)
P1=$(grep -c '"config": "\(single\|debate\)"' $OUT/rows.jsonl 2>/dev/null); P1=${P1:-0}
echo "phase 1:   $(bar $P1 48)  $P1 / 48 runs"
echo "cron:      $(crontab -l 2>/dev/null | grep -c debate_sweep_tick) entries"
echo "running:   $(ps -ef | grep -F -- 'debate_sweep_tick' | grep -v grep | wc -l) tick(s)"
ERR=$(grep -c Traceback $OUT/tick.log 2>/dev/null); echo "errors:    ${ERR:-0}"
# which arm is mid-run, and how far along
CUR=$(ls -t runs/core_2x2_worlds/*/out/sw_*_sim.log 2>/dev/null | head -1)
if [ -n "$CUR" ] && [ -n "$(find "$CUR" -newermt '-3 minutes' 2>/dev/null)" ]; then
  W=$(echo "$CUR" | cut -d/ -f3); A=$(basename "$CUR" | sed 's/sw_\(.*\)_s17_sim.log/\1/')
  PROG=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '[0-9]*/[0-9]* jobs processed' | head -1)
  ETA=$(tr '\r' '\n' < "$CUR" | tail -1 | grep -o '<[0-9hms:]*]' | tr -d '<]' | head -1)
  D=${PROG%%/*}; R=${PROG#*/}; R=${R%% *}
  echo "now:       $(bar ${D:-0} ${R:-1})  $W $A  ${PROG:-starting}  eta ${ETA:-?}"
fi
echo "--- last 3 runs:"
tail -3 $OUT/tick.log 2>/dev/null | grep -E 'left\)$|complete'
echo "--- paired results so far (debate minus single; negative favours debate):"
.venv/bin/python - <<'PY'
import json, collections
from pathlib import Path
p = Path('runs/sweep_train_debate/rows.jsonl')
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
by = collections.defaultdict(dict)
for r in rows:
    by[r['window']][r['config']] = r
hdr = f"{'window':9s} {'load':>5s} {'single':>7s} {'debate':>7s} {'bo3':>7s} {'diff':>7s}"
print(hdr); print('-' * len(hdr))
diffs = []
for w, d in sorted(by.items(), key=lambda x: -x[1][list(x[1])[0]]['offered_load']):
    s = d.get('single', {}).get('deadline_viol_pct')
    b = d.get('debate', {}).get('deadline_viol_pct')
    o = d.get('bo3', {}).get('deadline_viol_pct')
    load = d[list(d)[0]]['offered_load']
    diff = f"{b - s:+7.1f}" if (s is not None and b is not None) else f"{'':>7s}"
    if s is not None and b is not None: diffs.append(b - s)
    f = lambda v: f"{v:7.1f}" if v is not None else f"{'-':>7s}"
    print(f"{w:9s} {load:5.2f} {f(s)} {f(b)} {f(o)} {diff}")
if diffs:
    win = sum(1 for x in diffs if x < 0); tie = sum(1 for x in diffs if x == 0)
    print(f"\npaired n={len(diffs)}  mean {sum(diffs)/len(diffs):+.2f} pts  "
          f"debate better {win}, tie {tie}, worse {len(diffs)-win-tie}")
    print("NOTE: descriptive only. The pre-registered tests run once all 24 windows are complete.")
PY
