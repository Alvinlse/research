#!/bin/bash
# Status of the core 2x2 sweep.
cd /import/gp-home.ciero/kimseng/Research
OUT=runs/frozen_2x2
n=$(wc -l < $OUT/rows.jsonl 2>/dev/null || echo 0)
echo "rows:   $n / 605  ($((n*100/605))%)"
echo "cron:   $(crontab -l 2>/dev/null | grep -c sweep_2x2_tick) entries   done-marker: $([ -f $OUT/.done ] && echo yes || echo no)"
echo "errors: $(grep -c Traceback $OUT/tick.log 2>/dev/null || echo 0)"
.venv/bin/python -m pins.run_frozen --out $OUT --report 2>/dev/null | tail -15
