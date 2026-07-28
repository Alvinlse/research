#!/bin/bash
# Wait for both 35b sweeps to exit, then merge seeds 8-31 into the live thinking tier
# and write the report. Mostly sleeps -> negligible CPU -> survives the login-node reaper.
cd /import/gp-home.ciero/kimseng/Research
NOTHINK_PID=2581302
THINK_PID=2584685
deadline=$(( $(date +%s) + 21600 ))   # 6h hard cap
while kill -0 $NOTHINK_PID 2>/dev/null || kill -0 $THINK_PID 2>/dev/null; do
  [ "$(date +%s)" -gt "$deadline" ] && { echo "TIMEOUT waiting for runs" > pins/babysit_35b.status; break; }
  sleep 120
done
echo "both runs exited at $(date)" > pins/babysit_35b.status
PINS_NUM_CTX=8192 .venv/bin/python -m pins.merge_report_35b >> pins/babysit_35b.status 2>&1
echo "DONE $(date) -> pins/exp49_35b_report.md" >> pins/babysit_35b.status
