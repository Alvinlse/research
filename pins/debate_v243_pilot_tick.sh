#!/bin/bash
# One v2.4.3 focused pilot per tick; completed rows are resumable checkpoints.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/pilot_debate_v243_two
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
OUT = ROOT / 'runs/pilot_debate_v243_two/rows.jsonl'
WINDOWS = ('d111h22', 'd223h11')
if PROTOCOL_VERSION != '2.4.3':
    raise SystemExit(f'protocol drift: expected 2.4.3, found {PROTOCOL_VERSION}')
manifest = json.loads((ROOT / 'pins/core_2x2_manifest.json').read_text())
cfg, selector, inference = (
    manifest['execution'], manifest['selector'], manifest['inference'])
by_name = {window['window']: window for window in manifest['windows']}
rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()] \
    if OUT.exists() else []
done = {row['window'] for row in rows}
todo = [name for name in WINDOWS if name not in done]
if not todo:
    subprocess.run(
        [str(ROOT / '.venv/bin/python'),
         str(ROOT / 'pins/analyze_debate_v243_pilot.py')],
        cwd=ROOT, check=True)
    raise SystemExit('v2.4.3 pilot chain complete and analyzed')

name = todo[0]
window = by_name[name]
started = time.time()
result = run(
    ROOT / f'runs/core_2x2_worlds/{name}', 'policy_debate_v2_4', inference['model'],
    interval=cfg['simulation_interval_s'], tag='pilot_v243_s17', quiet=True,
    sizer='as_requested', family='mkt', sel_every=selector['selection_interval_s'],
    temperature=inference['temperature'], llm_seed=17,
    num_predict=inference['num_predict'], rcon_threshold_s=selector['rcon_threshold_s'],
    fallback_ordering='auction_deadline', fallback_sizing='adaptive')
row = {
    'window': name, 'config': 'v2.4.3', 'arm': 'policy_debate_v2_4',
    'protocol_version': '2.4.3', 'offered_load': window['offered_load'],
    'n_jobs': window['n_jobs'], 'split': window['split'],
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
print(
    f"{name} v2.4.3 -> {row['deadline_viol_pct']}% wait {row['mean_wait_s']} "
    f"util {row['useful_util_win']} trials {row['debate_v24_trials']} "
    f"accept {row['debate_v24_trial_accepts']} rollback "
    f"{row['debate_v24_trial_rollbacks']} blocked "
    f"{row['debate_v24_transition_blocked_requests']} wall {row['wall_s']}s "
    f"({len(todo) - 1} left)", flush=True)
PY
