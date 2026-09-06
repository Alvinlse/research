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

# Staged, each guarded by its own marker. The reaper will kill this tick partway through -- the
# next one then resumes at the stage that did not finish instead of redoing the pipeline. Training
# in particular resumes from its own HF checkpoint, so being killed costs at most `save_steps`.
FIN=$OUT/finalise.log
PY310=$REPO/.venv/bin/python
PYGPU=$REPO/.venv-forecast/bin/python
export PYTHONPATH=$REPO
{
echo "=== finalise tick $(date -Is)"

if [ ! -f "$OUT/.st_collate" ]; then
    "$PY310" -m pins.policy_labels --worlds "$REPO/runs/holdout" --out "$OUT" \
        --epochs 12 --shards "$SHARDS" --max-rollouts 1 --collate && touch "$OUT/.st_collate"
fi

if [ -f "$OUT/.st_collate" ] && [ ! -f "$OUT/.st_dataset" ]; then
    "$PY310" -m pins.policy_dataset --labels "$OUT" --out "$REPO/runs/dataset" \
        && touch "$OUT/.st_dataset"
fi

# Only mark training done on its own success line: a reaped run exits non-zero mid-way and must
# be resumed, not recorded as finished.
if [ -f "$OUT/.st_dataset" ] && [ ! -f "$OUT/.st_train" ]; then
    "$PYGPU" -m pins.finetune_referee --data "$REPO/runs/dataset" \
        --out "$REPO/runs/referee_lora" --epochs 3 | tee -a "$FIN" | grep -q "TRAINING COMPLETE" \
        && touch "$OUT/.st_train"
fi

if [ -f "$OUT/.st_train" ] && [ ! -f "$OUT/.st_eval" ]; then
    "$PYGPU" -m pins.policy_eval --data "$REPO/runs/dataset" \
        --lora "$REPO/runs/referee_lora/final" --split test \
        --out "$REPO/runs/eval_report.json" && touch "$OUT/.st_eval"
fi

[ -f "$OUT/.st_eval" ] || { echo "stages incomplete, next tick continues"; exit 0; }

gzip -kf "$OUT"/*_packets.jsonl 2>/dev/null
git add pins/*.py pins/sweep_labels_tick.sh check_sweep.sh
# runs/ is gitignored (.gitignore:48); force-add just the deliverables. The LoRA adapter is ~120MB
# of safetensors, so only its config and the report travel -- the adapter itself stays local.
git add -f "$OUT"/states.json "$OUT"/rollouts.*.jsonl "$OUT"/*_packets.jsonl.gz \
    "$REPO"/runs/dataset/*.jsonl "$REPO"/runs/dataset/meta.json "$REPO"/runs/eval_report.json \
    "$REPO"/runs/referee_lora/final/adapter_config.json 2>/dev/null
# Nothing staged means a glob matched no file -- commit would fail and the push would then
# publish unrelated state. Bail so the next tick retries instead.
git diff --cached --quiet && { echo "nothing staged; not committing"; exit 0; }
git commit -q -F - <<'MSG'
policy-selector referee: full label sweep, LoRA fine-tune and regret evaluation

9,900 counterfactual rollouts over 55 held-out windows, the chat dataset built
from them, and a LoRA-tuned Qwen2.5-3B scored by regret against baselines.

Two defects in the partial labels shaped the design and are fixed here:

- resize_conservative+greedy takes the largest legal size and never releases
  it, deadlocking the cluster; it is the worst of the 15 actions in ~91% of
  states. Regret over the full menu therefore mostly measures avoiding one
  broken action, so regret_safe (catastrophic arms removed) is the headline and
  catastrophe_rate is reported on its own.
- The real choice is a near-tie: the best-vs-second margin is under 0.05 in
  ~62% of states. Raw argmax targets would be contradictory between near
  identical packets, so targets are canonicalised within an epsilon band and
  training examples are weighted by margin.

Baselines are in the same table because a constant predictor is strong here.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019aA1e1dJTRw9vxqMyXiXu8
MSG

pushed=0
for attempt in 1 2 3; do
    git pull --rebase --quiet origin elastisim 2>&1
    if git push origin elastisim 2>&1; then echo "PUSHED OK (attempt $attempt)"; pushed=1; break; fi
    sleep 30
done
if [ "$pushed" = "1" ]; then
    crontab -l | grep -v sweep_labels_tick | crontab -
    echo "=== ALL DONE $(date -Is)"
else
    echo "=== PUSH FAILED, next tick retries $(date -Is)"
fi
} >> "$FIN" 2>&1
