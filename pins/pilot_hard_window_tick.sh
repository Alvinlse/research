#!/bin/bash
# One configuration per tick on a contended training window, cheap enough to run the whole chain.
# Cron, not a background shell: the reaper kills a long shell, cron survives. Resumable: each tick
# runs the first configuration that has no row yet, and the last tick removes its own cron entry.
# Usage: pilot_hard_window_tick.sh <window>
set -u
WIN=${1:?window required}
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/pilot_$WIN
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1
WIN=$WIN PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json, os, time
from pathlib import Path
from pins.elastisim_bench import run
ROOT=Path('/import/gp-home.ciero/kimseng/Research')
WIN=os.environ['WIN']
OUT=ROOT/f'runs/pilot_{WIN}/rows.jsonl'
man=json.loads((ROOT/'pins/core_2x2_manifest.json').read_text())
cfg,sel,inf=man['execution'],man['selector'],man['inference']
CONFIGS=[  # name, arm, sizer
    ('floor',  'auction_deadline', 'adaptive'),      # 0-token validation-selected market floor
    ('single', 'policy_select',    'as_requested'),
    ('debate', 'policy_debate',    'as_requested'),
]
done={json.loads(l)['config'] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
todo=[c for c in CONFIGS if c[0] not in done]
if not todo:
    import subprocess
    subprocess.run(f"crontab -l | grep -v 'pilot_hard_window_tick.sh {WIN}' | crontab -", shell=True)
    raise SystemExit('chain complete')
name,arm,sizer=todo[0]
t0=time.time()
r=run(ROOT/f'runs/core_2x2_worlds/{WIN}',arm,inf['model'],interval=cfg['simulation_interval_s'],
      tag=f'hw_{name}_s17',quiet=True,sizer=sizer,family='mkt',
      sel_every=sel['selection_interval_s'],temperature=inf['temperature'],llm_seed=17,
      num_predict=inf['num_predict'],rcon_threshold_s=sel['rcon_threshold_s'],
      fallback_ordering='auction_deadline',fallback_sizing='adaptive')
row={'config':name,'window':WIN,'arm':arm,'sizer':sizer,
     'deadline_viol_pct':r['deadline_viol_pct'],'mean_wait_s':r['mean_wait_s'],
     'p90_wait_s':r['p90_wait_s'],'sla2_viol_pct':r['sla2_viol_pct'],
     'sla5_viol_pct':r['sla5_viol_pct'],'useful_util_win':r['useful_util_win'],
     'mean_bsd':r['mean_bsd'],'llm_calls':r['llm_calls'],'sel_invalid':r['sel_invalid'],
     'fallbacks':r['fallbacks'],'sel_counts':r.get('sel_counts'),
     'debate_epochs':r.get('debate_epochs'),
     'debate_opening_agreements':r.get('debate_opening_agreements'),
     'debate_referee_calls':r.get('debate_referee_calls'),
     'debate_rejected_outputs':r.get('debate_rejected_outputs'),
     'wall_s':round(time.time()-t0)}
with OUT.open('a') as f: f.write(json.dumps(row)+'\n')
print(name,'->',row['deadline_viol_pct'],'% wait',row['mean_wait_s'],'calls',row['llm_calls'],
      row['wall_s'],'s',flush=True)
PY
