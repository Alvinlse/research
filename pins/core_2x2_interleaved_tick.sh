#!/bin/bash
# Run one complete paired-test cell per cron tick.  The file lock prevents overlapping model calls;
# the Python driver is append-only and resumes at the first missing window/cell.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/core_2x2_interleaved_test
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1

if "$REPO/.venv/bin/python" -m pins.run_core_2x2_interleaved --status | grep -q '"complete": true'; then
    touch "$OUT/.done"
    crontab -l | grep -v core_2x2_interleaved_tick | crontab -
    exit 0
fi

PYTHONPATH=$REPO "$REPO/.venv/bin/python" -m pins.run_core_2x2_interleaved --max-runs 1 \
    >> "$OUT/tick.log" 2>&1

if "$REPO/.venv/bin/python" -m pins.run_core_2x2_interleaved --status | grep -q '"complete": true'; then
    touch "$OUT/.done"
    crontab -l | grep -v core_2x2_interleaved_tick | crontab -
fi
