#!/bin/bash
# One configuration per tick on d58h19 (138 jobs, offered load 1.35), cheap enough to run the whole
# chain. Cron, not a background shell: the reaper kills a long shell, cron survives. Resumable: each
# tick runs the first configuration that has no row yet, and the last tick removes its own cron entry.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/pilot_d58h19
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json, time
from pathlib import Path
from pins.elastisim_bench import run
ROOT=Path('/import/gp-home.ciero/kimseng/Research')
OUT=ROOT/'runs/pilot_d58h19/rows.jsonl'
man=json.loads((ROOT/'pins/core_2x2_manifest.json').read_text())
cfg,sel,inf=man['execution'],man['selector'],man['inference']
CONFIGS=[  # name, arm, demand_v2, packet_v2
    ('single',        'policy_select',    False, False),
    ('roles_frozen',  'policy_negotiate', False, False),
    ('roles_demandv2','policy_negotiate', True,  False),
    ('roles_packetv2','policy_negotiate', True,  True),
]
done={json.loads(l)['config'] for l in OUT.read_text().splitlines() if l.strip()} if OUT.exists() else set()
todo=[c for c in CONFIGS if c[0] not in done]
if not todo:
    import subprocess
    subprocess.run("crontab -l | grep -v pilot_d58h19_tick | crontab -", shell=True)
    raise SystemExit('chain complete')
name,arm,dv2,pv2=todo[0]
t0=time.time()
r=run(ROOT/'runs/core_2x2_worlds/d58h19',arm,inf['model'],interval=cfg['simulation_interval_s'],
      tag=f'p58_{name}',quiet=True,sizer='as_requested',family='nm',
      sel_every=sel['selection_interval_s'],temperature=inf['temperature'],llm_seed=17,
      num_predict=inf['num_predict'],rcon_threshold_s=sel['rcon_threshold_s'],
      packet_v2=pv2,demand_v2=dv2)
row={'config':name,'arm':arm,'demand_v2':dv2,'packet_v2':pv2,
     'deadline_viol_pct':r['deadline_viol_pct'],'mean_wait_s':r['mean_wait_s'],
     'llm_calls':r['llm_calls'],'sel_invalid':r['sel_invalid'],'sel_counts':r.get('sel_counts'),
     'tok_prompt':r.get('tok_prompt'),'wall_s':round(time.time()-t0)}
with OUT.open('a') as f: f.write(json.dumps(row)+'\n')
print(name,'->',row['deadline_viol_pct'],'%',row['sel_counts'],row['wall_s'],'s',flush=True)
PY
