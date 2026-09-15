#!/bin/bash
# One v2.4.3 real training window per tick; Demand/Supply calls are parallel inside the arm.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/sweep_train_debate_v243
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
from pins.policy_debate_v2_4 import PROTOCOL_VERSION

ROOT = Path('/import/gp-home.ciero/kimseng/Research')
OUT = ROOT / 'runs/sweep_train_debate_v243/rows.jsonl'
PILOT = ROOT / 'runs/pilot_debate_v243_two/rows.jsonl'
PILOT_WINDOWS = {'d111h22', 'd223h11'}
if PROTOCOL_VERSION != '2.4.3':
    raise SystemExit(f'protocol drift: expected 2.4.3, found {PROTOCOL_VERSION}')
manifest = json.loads((ROOT / 'pins/core_2x2_manifest.json').read_text())
cfg, selector, inference = (
    manifest['execution'], manifest['selector'], manifest['inference'])
windows = sorted(
    (window for window in manifest['windows'] if window['split'] == 'train'),
    key=lambda window: -window['offered_load'])
rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()] \
    if OUT.exists() else []
done = {row['window'] for row in rows}
if PILOT.exists():
    for line in PILOT.read_text().splitlines():
        if not line.strip():
            continue
        pilot_row = json.loads(line)
        if pilot_row['window'] in PILOT_WINDOWS and pilot_row['window'] not in done:
            imported = {**pilot_row, 'source_tag': 'pilot_v243_s17'}
            with OUT.open('a') as handle:
                handle.write(json.dumps(imported) + '\n')
            rows.append(imported)
            done.add(imported['window'])
missing_pilot = PILOT_WINDOWS - done
if missing_pilot:
    raise SystemExit(
        f"single-GPU queue: awaiting pilot rows {sorted(missing_pilot)}")
todo = [window for window in windows
        if window['window'] not in done and window['window'] not in PILOT_WINDOWS]
if not todo:
    subprocess.run(
        [str(ROOT / '.venv/bin/python'),
         str(ROOT / 'pins/analyze_debate_v243_sweep.py')],
        cwd=ROOT, check=True)
    raise SystemExit('v2.4.3 24-window chain complete and analyzed')

window = todo[0]
name = window['window']
started = time.time()
result = run(
    ROOT / f'runs/core_2x2_worlds/{name}', 'policy_debate_v2_4', inference['model'],
    interval=cfg['simulation_interval_s'], tag='sw_v243_s17', quiet=True,
    sizer='as_requested', family='mkt', sel_every=selector['selection_interval_s'],
    temperature=inference['temperature'], llm_seed=17,
    num_predict=inference['num_predict'], rcon_threshold_s=selector['rcon_threshold_s'],
    fallback_ordering='auction_deadline', fallback_sizing='adaptive')
row = {
    'window': name, 'config': 'v2.4.3', 'arm': 'policy_debate_v2_4',
    'protocol_version': '2.4.3', 'offered_load': window['offered_load'],
    'n_jobs': window['n_jobs'], 'split': window['split'],
    'source_tag': 'sw_v243_s17',
    **{key: result[key] for key in (
        'deadline_viol_pct', 'mean_wait_s', 'p90_wait_s', 'sla2_viol_pct',
        'sla5_viol_pct', 'useful_util_win', 'mean_bsd', 'llm_calls', 'llm_errors',
        'sel_invalid', 'fallbacks', 'sel_counts', 'debate_epochs',
        'debate_referee_calls', 'debate_rejected_outputs', 'debate_v24_epochs',
        'debate_v24_sizing_reviews', 'debate_v24_sizing_transitions',
        'debate_v24_trials', 'debate_v24_trial_probations',
        'debate_v24_trial_accepts', 'debate_v24_trial_rollbacks',
        'debate_v24_invalid_holds', 'debate_v24_transition_blocked_requests',
        'debate_v24_transition_backoff_holds')},
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
    f"{name} v2.4.3 -> {row['deadline_viol_pct']}% wait {row['mean_wait_s']} "
    f"util {row['useful_util_win']} calls {row['llm_calls']} "
    f"trials {row['debate_v24_trials']} blocked "
    f"{row['debate_v24_transition_blocked_requests']} wall {row['wall_s']}s "
    f"({len(todo) - 1} left, est {eta})", flush=True)
PY
