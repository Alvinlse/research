#!/bin/bash
# One (window, arm) configuration per tick across the 24 training windows, per pins/debate_sweep_prereg.md.
# Cron, not a background shell: the reaper kills a long shell, cron survives. Resumable and append-only;
# arms are ordered within a window so an interruption leaves complete pairs. The last tick removes its
# own cron entry.
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/sweep_train_debate
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 0
cd "$REPO" || exit 1
PYTHONPATH=$REPO $REPO/.venv/bin/python - >> "$OUT/tick.log" 2>&1 <<'PY'
import json, time
from pathlib import Path
from pins.elastisim_bench import run
ROOT = Path('/import/gp-home.ciero/kimseng/Research')
OUT = ROOT / 'runs/sweep_train_debate/rows.jsonl'
man = json.loads((ROOT / 'pins/core_2x2_manifest.json').read_text())
cfg, sel, inf = man['execution'], man['selector'], man['inference']
wins = [w for w in man['windows'] if w['split'] == 'train']
wins.sort(key=lambda w: -w['offered_load'])          # hardest first: the informative end
# Amendment 1: after both pre-registered arms finish on all 24 windows, the budget-matched control
# runs on the 8 windows with the largest debate gain. Descriptive; not an arm of the comparison.
ARMS = [('single', 'policy_select'), ('debate', 'policy_debate')]
rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
done = {(r['window'], r['config']) for r in rows}
# Phase 1: both pre-registered arms on all 24 windows.
todo = [(w, name, arm) for w in wins for name, arm in ARMS if (w['window'], name) not in done]
if not todo:
    # Phase 2 (Amendment 1): the budget-matched control on the 8 windows where debate gained most
    # over the single selector -- where a structure-or-spend question exists at all. Selected on
    # phase-1 outcomes and disclosed as such in pins/debate_sweep_prereg.md.
    viol = {(r['window'], r['config']): r['deadline_viol_pct'] for r in rows}
    gap = sorted(((viol[(w['window'], 'debate')] - viol[(w['window'], 'single')], w) for w in wins),
                 key=lambda x: x[0])
    control = [w for _, w in gap[:8]]
    todo = [(w, 'bo3', 'policy_bo3') for w in control if (w['window'], 'bo3') not in done]
if not todo:
    import subprocess
    subprocess.run("crontab -l | grep -v debate_sweep_tick | crontab -", shell=True)
    raise SystemExit('chain complete')
w, name, arm = todo[0]
win = w['window']
t0 = time.time()
r = run(ROOT / f'runs/core_2x2_worlds/{win}', arm, inf['model'],
        interval=cfg['simulation_interval_s'], tag=f'sw_{name}_s17', quiet=True,
        sizer='as_requested', family='mkt', sel_every=sel['selection_interval_s'],
        temperature=inf['temperature'], llm_seed=17, num_predict=inf['num_predict'],
        rcon_threshold_s=sel['rcon_threshold_s'],
        fallback_ordering='auction_deadline', fallback_sizing='adaptive')
row = {'window': win, 'config': name, 'arm': arm, 'offered_load': w['offered_load'],
       'n_jobs': w['n_jobs'], 'split': 'train',
       'deadline_viol_pct': r['deadline_viol_pct'], 'mean_wait_s': r['mean_wait_s'],
       'p90_wait_s': r['p90_wait_s'], 'sla2_viol_pct': r['sla2_viol_pct'],
       'sla5_viol_pct': r['sla5_viol_pct'], 'useful_util_win': r['useful_util_win'],
       'mean_bsd': r['mean_bsd'], 'llm_calls': r['llm_calls'], 'llm_errors': r['llm_errors'],
       'sel_invalid': r['sel_invalid'], 'fallbacks': r['fallbacks'], 'sel_counts': r.get('sel_counts'),
       'debate_epochs': r.get('debate_epochs'),
       'debate_opening_agreements': r.get('debate_opening_agreements'),
       'debate_referee_calls': r.get('debate_referee_calls'),
       'debate_rejected_outputs': r.get('debate_rejected_outputs'),
       'wall_s': round(time.time() - t0)}
with OUT.open('a') as f:
    f.write(json.dumps(row) + '\n')
left = len(todo) - 1
print(f"{win} {name} -> {row['deadline_viol_pct']}% wait {row['mean_wait_s']} "
      f"calls {row['llm_calls']} {row['wall_s']}s ({left} left)", flush=True)
PY
