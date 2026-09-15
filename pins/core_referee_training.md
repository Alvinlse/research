# Core Referee LoRA follow-up

This is a separate follow-up to the incomplete `core-2x2-v1` run. It does not overwrite or mix
with `runs/core_2x2/rows.jsonl`.

## Scientific boundary

- Demand and Supply agents remain frozen. Only the final Referee is LoRA-tuned.
- The online packet contains observable scheduler state only.
- Oracle rollouts use future outcomes only to create offline labels.
- Labels optimize the registered primary outcome, deadline-violation percentage.
- Complete workload windows remain within one split.
- Generate and tune on `train` and `validation`; do not generate `test` labels until the model,
  prompt, action space, and training hyperparameters are frozen.

## 1. Generate oracle labels

Each state evaluates 16 actions in each family. Four evenly spaced states are retained per window.
The command is resumable and appends each completed rollout immediately.

```bash
.venv/bin/python -m pins.core_oracle_labels \
  --out runs/core_referee_oracle_v1 \
  --splits train validation --epochs 4 --shard 0 --shards 4 --max-rollouts 32
```

Run the same command for shards `1`, `2`, and `3`, and repeat bounded invocations until every shard
writes its split/family-specific `done.*.<shard>` marker. Then collate:

```bash
.venv/bin/python -m pins.core_oracle_labels \
  --out runs/core_referee_oracle_v1 --collate
```

The resulting dataset is under `runs/core_referee_oracle_v1/dataset`. Its `meta.json` records split
counts, windows, action counts, the deadline objective, and the leakage audit.

## 2. LoRA-tune one Referee

Qwen3-8B:

```bash
.venv-forecast/bin/python -m pins.finetune_core_referee \
  --data runs/core_referee_oracle_v1/dataset --model qwen3-8b \
  --out runs/core_referee_lora_qwen3_8b --epochs 3
```

Gemma 3 4B:

```bash
.venv-forecast/bin/python -m pins.finetune_core_referee \
  --data runs/core_referee_oracle_v1/dataset --model gemma3-4b \
  --out runs/core_referee_lora_gemma3_4b --epochs 3
```

The Hugging Face base checkpoint must be available before training. Gemma may require accepting its
model license and authenticating with Hugging Face. The Ollama quantization is for inference and is
not used as the training checkpoint.

## 3. Validate before touching test

```bash
.venv-forecast/bin/python -m pins.eval_core_referee \
  --data runs/core_referee_oracle_v1/dataset --split validation --model qwen3-8b \
  --adapter runs/core_referee_lora_qwen3_8b/final \
  --out runs/core_referee_eval_qwen3_validation.json
```

The tuned model must beat both the zero-shot model and the best fixed-on-train policy in mean oracle
regret, without increasing invalid JSON. Otherwise fine-tuning is considered unsuccessful.

## 4. Final held-out evaluation

After freezing the winner and its training settings, generate oracle rows for `test`, collate again,
and run `pins.eval_core_referee --split test`. Report the zero-shot, tuned, majority-by-family, and
best-fixed-on-train baselines together.

## Fast gates

```bash
.venv-forecast/bin/python -m pins.test_core_training
.venv/bin/python -m py_compile pins/core_oracle_labels.py \
  pins/finetune_core_referee.py pins/eval_core_referee.py
```

A two-step LoRA smoke run has already verified that gradients reach non-zero LoRA B matrices and
that the adapter can be saved. Full training retains five warm-up steps; only a tiny smoke run uses
`--warmup-steps 0`.
