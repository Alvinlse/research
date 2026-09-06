#!/bin/bash
# One bounded chunk of the policy-label sweep. Driven by cron, NOT by a long-running shell.
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
OUT=$REPO/runs/labels_full
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
        --epochs 12 --shard "$SHARD" --shards "$SHARDS" --max-rollouts 150 >> "$LOG" 2>&1
fi

# Finalisation is shard 0's job, and only once every shard has walked its whole window list.
[ "$SHARD" != "0" ] && exit 0
for s in 0 1 2 3; do [ -f "$OUT/done.$s" ] || exit 0; done
[ -f "$OUT/.finalised" ] && exit 0
touch "$OUT/.finalised"

{
  echo "=== FINALISING $(date -Is)"
  "$PY" -m pins.policy_labels --worlds "$REPO/runs/holdout" --out "$OUT" \
      --epochs 12 --shards "$SHARDS" --max-rollouts 1 --collate
  gzip -kf "$OUT"/*_packets.jsonl 2>/dev/null
  # runs/ is gitignored (.gitignore:48) and these artifacts are the deliverable, so force-add the
  # specific files rather than loosening the ignore rule for the whole tree.
  git add pins/elastisim_bench.py pins/policy_labels.py pins/sweep_labels_tick.sh
  git add -f "$OUT"/states.json "$OUT"/rollouts.*.jsonl "$OUT"/*_packets.jsonl.gz
  git commit -q -F - <<'MSG'
policy labels: full counterfactual sweep over the 55 held-out windows

Teacher labels for the fine-tuned policy-selector referee. At each sampled
epoch every (ordering, sizing) pair in POLICY_MENU takes over from a fixed
baseline and runs to the end of the window; the reward vector over all 15
candidates is the label and its argmax the supervised target.

Two calibrations are baked in and were measured, not assumed:
- epochs are restricted to the 5-60% band of each window, because the reward
  gap decays hard with switch time (17.27 at t=198 down to 0.41 at t=99617),
  so late switches label noise;
- reward weights put the service metrics in front. A first pass gave mean
  bounded slowdown a 51% share of total cost and resize churn 16%, which let
  a never-resize policy win states where it waited 53% longer.

Rollouts store raw metrics, so re-scoring under another operating point is a
re-collate with no simulator time.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019aA1e1dJTRw9vxqMyXiXu8
MSG
  # Retry, and rebase in case the branch moved while the sweep was running. Cron is only torn
  # down on a CONFIRMED push -- otherwise the next tick re-enters here and tries again, because a
  # finished sweep that never reached GitHub is the one failure that would waste the whole night.
  pushed=0
  for attempt in 1 2 3; do
      git pull --rebase --quiet origin elastisim 2>&1
      if git push origin elastisim 2>&1; then echo "PUSHED OK (attempt $attempt)"; pushed=1; break; fi
      sleep 30
  done
  if [ "$pushed" = "1" ]; then
      crontab -l | grep -v sweep_labels_tick | crontab -   # stop ticking once it is done
      echo "=== DONE $(date -Is)"
  else
      rm -f "$OUT/.finalised"    # let the next tick retry the whole finalisation
      echo "=== PUSH FAILED, will retry next tick $(date -Is)"
  fi
} >> "$OUT/finalise.log" 2>&1
