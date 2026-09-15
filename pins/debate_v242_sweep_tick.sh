#!/bin/bash
# One v2.4.2 real training window per tick; Demand/Supply calls are parallel inside the arm.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/sweep_train_debate_v242
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json
import subprocess
import time
from pathlib import Path

from pins.elastisim_bench import run

ROOT = Path('/import/gp-home.ciero/kimseng/Research')
OUT = ROOT / 'runs/sweep_train_debate_v242/rows.jsonl'
manifest = json.loads((ROOT / 'pins/core_2x2_manifest.json').read_text())
cfg, selector, inference = (
    manifest['execution'], manifest['selector'], manifest['inference'])
windows = sorted(
    (window for window in manifest['windows'] if window['split'] == 'train'),
    key=lambda window: -window['offered_load'])
rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()] \
    if OUT.exists() else []
done = {row['window'] for row in rows}
todo = [window for window in windows if window['window'] not in done]
if not todo:
    current = subprocess.run(
        ['crontab', '-l'], text=True, capture_output=True, check=False).stdout
    kept = [line for line in current.splitlines() if 'debate_v242_sweep_tick' not in line]
    subprocess.run(
        ['crontab', '-'], input=('\n'.join(kept) + ('\n' if kept else '')),
        text=True, check=True)
    raise SystemExit('v2.4.2 24-window chain complete; awaiting analysis')

window = todo[0]
name = window['window']
started = time.time()
result = run(
    ROOT / f'runs/core_2x2_worlds/{name}', 'policy_debate_v2_4', inference['model'],
    interval=cfg['simulation_interval_s'], tag='sw_v242_s17', quiet=True,
    sizer='as_requested', family='mkt', sel_every=selector['selection_interval_s'],
    temperature=inference['temperature'], llm_seed=17,
    num_predict=inference['num_predict'], rcon_threshold_s=selector['rcon_threshold_s'],
    fallback_ordering='auction_deadline', fallback_sizing='adaptive')
row = {
    'window': name, 'config': 'v2.4.2', 'arm': 'policy_debate_v2_4',
    'protocol_version': '2.4.2', 'offered_load': window['offered_load'],
    'n_jobs': window['n_jobs'], 'split': window['split'],
    **{key: result[key] for key in (
        'deadline_viol_pct', 'mean_wait_s', 'p90_wait_s', 'sla2_viol_pct',
        'sla5_viol_pct', 'useful_util_win', 'mean_bsd', 'llm_calls', 'llm_errors',
        'sel_invalid', 'fallbacks', 'sel_counts', 'debate_epochs',
        'debate_referee_calls', 'debate_rejected_outputs', 'debate_v24_epochs',
        'debate_v24_sizing_reviews', 'debate_v24_sizing_transitions',
        'debate_v24_trials', 'debate_v24_trial_probations',
        'debate_v24_trial_accepts', 'debate_v24_trial_rollbacks',
        'debate_v24_invalid_holds')},
    'wall_s': round(time.time() - started),
}
with OUT.open('a') as handle:
    handle.write(json.dumps(row) + '\n')
completed = rows + [row]
mean_wall_s = sum(item['wall_s'] for item in completed) / len(completed)
remaining_s = round(mean_wall_s * (len(todo) - 1))
hours, remaining_s = divmod(remaining_s, 3600)
minutes, remaining_s = divmod(remaining_s, 60)
eta = f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {remaining_s:02d}s'
print(
    f"{name} v2.4.2 -> {row['deadline_viol_pct']}% wait {row['mean_wait_s']} "
    f"util {row['useful_util_win']} calls {row['llm_calls']} "
    f"trials {row['debate_v24_trials']} probation "
    f"{row['debate_v24_trial_probations']} accept {row['debate_v24_trial_accepts']} "
    f"rollback {row['debate_v24_trial_rollbacks']} wall {row['wall_s']}s "
    f"({len(todo) - 1} left, est {eta})", flush=True)
PY
