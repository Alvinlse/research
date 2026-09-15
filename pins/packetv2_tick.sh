#!/bin/bash
# Cron tick for the packet-v2 diagnostic. Waits for the anchor-fix run to finish first, so the two
# interventions are measured in sequence on the same window rather than simultaneously.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/pilot_packetv2
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
[ -s "$REPO/runs/pilot_anchor/rows.jsonl" ] || exit 0      # step 2 not finished yet
[ -s "$OUT/rows.jsonl" ] && { crontab -l | grep -v packetv2_tick | crontab -; exit 0; }
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json, time
from pathlib import Path
from pins.elastisim_bench import run
ROOT=Path('/import/gp-home.ciero/kimseng/Research')
man=json.loads((ROOT/'pins/core_2x2_manifest.json').read_text())
cfg,sel,inf=man['execution'],man['selector'],man['inference']
t0=time.time()
r=run(ROOT/'runs/core_2x2_worlds/d48h23','policy_negotiate',inf['model'],
      interval=cfg['simulation_interval_s'],tag='pv2_negotiate',quiet=True,sizer='as_requested',
      family='nm',sel_every=sel['selection_interval_s'],temperature=inf['temperature'],
      llm_seed=17,num_predict=inf['num_predict'],rcon_threshold_s=sel['rcon_threshold_s'],
      packet_v2=True)
row={'window':'d48h23','arm':'policy_negotiate_packetv2',
     'deadline_viol_pct':r['deadline_viol_pct'],'mean_wait_s':r['mean_wait_s'],
     'llm_calls':r['llm_calls'],'sel_invalid':r['sel_invalid'],'sel_counts':r.get('sel_counts'),
     'tok_prompt':r.get('tok_prompt'),'wall_s':round(time.time()-t0)}
(ROOT/'runs/pilot_packetv2/rows.jsonl').open('a').write(json.dumps(row)+'\n')
print('done', row)
PY
