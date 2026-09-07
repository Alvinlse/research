#!/bin/bash
# One bounded chunk of the PER-JOB outcome sweep: same rollouts as the first sweep, but every
# job's own outcome is kept per rollout (perjob.<shard>.jsonl) instead of being overwritten. Driven by cron, NOT by a long-running shell.
#
# The login node reaps a process at ~12-15 CPU-min, so a multi-hour sweep cannot be one process
# and cannot be a background shell either -- both die, and a background shell dies with the ssh
# session as well. Cron survives all three: each tick does a bounded amount of work, appends every
# finished rollout immediately, and the next tick resumes from what is on disk. Losing a tick to
# the reaper costs at most the single rollout that was in flight.
#
#   usage: sweep_labels_tick.sh <shard>
set -u
REPO=/import/gp-home.ciero/kimseng/Research
OUT=$REPO/runs/labels_perjob
PY=$REPO/.venv/bin/python
SHARD=${1:-0}
SHARDS=4
LOG=$OUT/tick.$SHARD.log
mkdir -p "$OUT"

# Never let two ticks of the same shard overlap; -n means a tick that arrives while the previous
# one is still working exits immediately rather than queueing up behind it.
exec 9>"$OUT/.lock.$SHARD"
flock -n 9 || exit 0

cd "$REPO" || exit 1

if [ ! -f "$OUT/done.$SHARD" ]; then
    echo "=== tick $(date -Is) shard $SHARD" >> "$LOG"
    # 150 rollouts x ~3.7 CPU-s ~= 9 CPU-min, comfortably inside the reaper's window.
    "$PY" -m pins.policy_labels --worlds "$REPO/runs/holdout" --out "$OUT" \
        --epochs 12 --shard "$SHARD" --shards "$SHARDS" --max-rollouts 150 --perjob >> "$LOG" 2>&1
fi

# Finalisation is shard 0's job, and only once every shard has walked its whole window list.
[ "$SHARD" != "0" ] && exit 0
for s in 0 1 2 3; do [ -f "$OUT/done.$s" ] || exit 0; done
[ -f "$OUT/.finalised" ] && exit 0
FIN=$OUT/finalise.log
PY310=$REPO/.venv/bin/python
export PYTHONPATH=$REPO
{
echo "=== finalise $(date -Is)"
"$PY310" -m pins.policy_labels --worlds "$REPO/runs/holdout" --out "$OUT" \
    --epochs 12 --shards "$SHARDS" --max-rollouts 1 --collate || { echo "collate failed"; exit 0; }
gzip -kf "$OUT"/perjob.*.jsonl "$OUT"/*_packets.jsonl 2>/dev/null
git add pins/policy_labels.py pins/sweep_perjob_tick.sh
git add -f "$OUT"/states.json "$OUT"/rollouts.*.jsonl "$OUT"/perjob.*.jsonl.gz 2>/dev/null
git diff --cached --quiet && { echo "nothing staged; not committing"; exit 0; }
git commit -q -F - <<'MSG'
per-job outcome sweep: every job's own result under every policy, for all 589 states

Re-run of the 8,835-rollout counterfactual sweep with one addition: after each
rollout the simulator's per-job statistics are captured before the next
rollout's identical tag overwrites them. The first sweep discarded these and
kept only window summaries, which is exactly the information a per-job policy
assigner needs and a global selector cannot use.

Window-level rewards reproduce the first sweep (deterministic simulator).
Packets are the enriched 28-field ones; epoch timestamps match dataset_v3.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019aA1e1dJTRw9vxqMyXiXu8
MSG
pushed=0
for attempt in 1 2 3; do
    git pull --rebase --quiet origin elastisim 2>&1
    if git push origin elastisim 2>&1; then echo "PUSHED OK (attempt $attempt)"; pushed=1; break; fi
    sleep 30
done
if [ "$pushed" = "1" ]; then
    touch "$OUT/.finalised"
    crontab -l | grep -v sweep_perjob_tick | crontab -
    echo "=== ALL DONE $(date -Is)"
else
    echo "=== PUSH FAILED, next tick retries $(date -Is)"
fi
} >> "$FIN" 2>&1
