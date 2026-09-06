#!/bin/bash
# Status of the overnight policy-label sweep.
cd /import/gp-home.ciero/kimseng/Research
N=$(cat runs/labels_full/rollouts.*.jsonl 2>/dev/null | wc -l)
echo "rollouts:  $N / 9900  ($((N*100/9900))%)"
echo "workers:   $(ps -ef | grep -F -- '-m pins.policy_labels' | grep -v grep | grep -vc check_sweep) / 4"
echo "shards done: $(ls runs/labels_full/done.* 2>/dev/null | wc -l) / 4"
echo "cron:      $(crontab -l 2>/dev/null | grep -c sweep_labels_tick) entries"
for f in runs/labels_full/tick.*.log; do
  n=$(grep -c Traceback "$f" 2>/dev/null)
  [ "${n:-0}" -gt 0 ] && echo "!! $(basename $f): $n tracebacks -- LAST: $(grep -A0 'Error' $f | tail -1)"
done
[ -f runs/labels_full/finalise.log ] && { echo "--- finalise:"; cat runs/labels_full/finalise.log; }
[ -f runs/labels_full/states.json ] && echo "states.json: $(python3 -c "import json;print(len(json.load(open('runs/labels_full/states.json'))))" 2>/dev/null) states"
echo "--- git: $(git log --oneline -1)"
