#!/bin/bash
# One cron tick for the demand-anchor diagnostic. Cron, not a background shell: the login node reaps
# a long-running shell at ~12-15 CPU-min and this single run takes ~45 min of simulator time.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/pilot_anchor
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
[ -s "$OUT/rows.jsonl" ] && { crontab -l | grep -v anchor_tick | crontab -; exit 0; }
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python \
  /tmp/claude-200154/-import-gp-home-ciero-kimseng/ecfde4e0-d954-47e9-b76c-a50f699856b9/scratchpad/anchor.py \
  >> "$OUT/tick.log" 2>&1
