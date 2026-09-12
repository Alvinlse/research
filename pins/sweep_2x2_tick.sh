#!/bin/bash
# One bounded chunk of the verified core 2x2. Driven by cron, NOT a long-running shell:
# the login node reaps a process at ~12-15 CPU-min and a background shell dies with the ssh session,
# while the LLM arms need ~16 h of wall time in total. Each tick runs a few cells, `pins.run_frozen`
# appends every finished run to rows.jsonl immediately and skips completed keys, so a lost tick costs
# at most the one run in flight.
#
# Cell order is deliberate: the deterministic baselines finish first, so a partial sweep is still a
# usable result, and the expensive multi-agent arm is last.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/core_2x2
PY=$REPO/.venv/bin/python
LOG=$OUT/tick.log

# A scheduled experiment must be armed explicitly after its manifest, implementation and analysis
# pre-registration have passed `pins.verify_2x2`.  This also keeps cron from appending results while
# those inputs are being edited.
mkdir -p "$OUT"
[ -f "$OUT/.armed" ] || exit 0
TOTAL=1056       # 24 eval windows x (32 fixed policies + 4 factorial cells x 3 seeds)

exec 9>"$OUT/.lock"
flock -n 9 || exit 0          # a tick arriving while the previous one works exits, never queues

cd "$REPO" || exit 1
[ -f "$OUT/.done" ] && exit 0

echo "=== tick $(date -Is)" >> "$LOG"
"$PY" -m pins.run_core_2x2 --out "$OUT" --max-seconds 480 >> "$LOG" 2>&1

n=$(wc -l < "$OUT/rows.jsonl" 2>/dev/null || echo 0)
echo "rows $n / $TOTAL" >> "$LOG"
if [ "$n" -ge "$TOTAL" ]; then
    touch "$OUT/.done"
    echo "=== ALL CELLS COMPLETE $(date -Is)" >> "$LOG"
fi
