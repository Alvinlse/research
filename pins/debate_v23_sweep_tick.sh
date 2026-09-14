#!/bin/bash
# One v2.3 real-window run per tick. Rules/analysis are frozen in debate_v23_sweep_prereg.md.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/sweep_train_debate_v23
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json, time
from pathlib import Path
from pins.elastisim_bench import run

ROOT = Path('/import/gp-home.ciero/kimseng/Research')
OUT = ROOT / 'runs/sweep_train_debate_v23/rows.jsonl'
manifest = json.loads((ROOT / 'pins/core_2x2_manifest.json').read_text())
cfg, selector, inference = (
    manifest['execution'], manifest['selector'], manifest['inference'])
windows = sorted(
    (w for w in manifest['windows'] if w['split'] == 'train'),
    key=lambda w: -w['offered_load'])
rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()] \
    if OUT.exists() else []
done = {row['window'] for row in rows}
todo = [window for window in windows if window['window'] not in done]
if not todo:
    import subprocess
    subprocess.run("crontab -l | grep -v debate_v23_sweep_tick | crontab -", shell=True)
    raise SystemExit('v2.3 chain complete')

window = todo[0]
name = window['window']
started = time.time()
result = run(
    ROOT / f'runs/core_2x2_worlds/{name}', 'policy_debate_v2_3', inference['model'],
    interval=cfg['simulation_interval_s'], tag='sw_v23_s17', quiet=True,
    sizer='as_requested', family='mkt', sel_every=selector['selection_interval_s'],
    temperature=inference['temperature'], llm_seed=17,
    num_predict=inference['num_predict'], rcon_threshold_s=selector['rcon_threshold_s'],
    fallback_ordering='auction_deadline', fallback_sizing='adaptive')
row = {
    'window': name, 'config': 'v2.3', 'arm': 'policy_debate_v2_3',
    'offered_load': window['offered_load'], 'n_jobs': window['n_jobs'], 'split': 'train',
    **{key: result[key] for key in (
        'deadline_viol_pct', 'mean_wait_s', 'p90_wait_s', 'sla2_viol_pct',
        'sla5_viol_pct', 'useful_util_win', 'mean_bsd', 'llm_calls', 'llm_errors',
        'sel_invalid', 'fallbacks', 'sel_counts', 'debate_epochs', 'debate_referee_calls',
        'debate_rejected_outputs', 'debate_v23_epochs', 'debate_v23_parallel_opening_rounds',
        'debate_v23_supply_escalation_epochs', 'debate_v23_supply_escalation_triggers',
        'debate_v23_demand_escalation_epochs', 'debate_v23_demand_escalation_triggers')},
    'wall_s': round(time.time() - started),
}
with OUT.open('a') as handle:
    handle.write(json.dumps(row) + '\n')
print(
    f"{name} v2.3 -> {row['deadline_viol_pct']}% wait {row['mean_wait_s']} "
    f"calls {row['llm_calls']} wall {row['wall_s']}s ({len(todo) - 1} left)", flush=True)
PY
