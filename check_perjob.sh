#!/bin/bash
# Status of the overnight policy-label sweep.
cd /import/gp-home.ciero/kimseng/Research
N=$(cat runs/labels_perjob/rollouts.*.jsonl 2>/dev/null | wc -l)
echo "rollouts:  $N / 8835  ($((N*100/8835))%)"
echo "workers:   $(ps -ef | grep -F -- '-m pins.policy_labels' | grep -v grep | grep -vc check_sweep) / 4"
echo "shards done: $(ls runs/labels_perjob/done.* 2>/dev/null | wc -l) / 4"
echo "cron:      $(crontab -l 2>/dev/null | grep -c sweep_perjob_tick) entries"
for f in runs/labels_perjob/tick.*.log; do
  n=$(grep -c Traceback "$f" 2>/dev/null)
  [ "${n:-0}" -gt 0 ] && echo "!! $(basename $f): $n tracebacks -- LAST: $(grep -A0 'Error' $f | tail -1)"
done
[ -f runs/labels_perjob/finalise.log ] && { echo "--- finalise:"; cat runs/labels_perjob/finalise.log; }
[ -f runs/labels_perjob/states.json ] && echo "states.json: $(python3 -c "import json;print(len(json.load(open('runs/labels_perjob/states.json'))))" 2>/dev/null) states"
echo "--- git: $(git log --oneline -1)"
echo "perjob rows: $(cat runs/labels_perjob/perjob.*.jsonl 2>/dev/null | wc -l)  (should equal rollouts)"
