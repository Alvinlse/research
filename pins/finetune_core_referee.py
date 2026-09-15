"""LoRA-tune a policy Referee on the current core-2x2 oracle dataset.

Demand and Supply roles remain frozen.  Only the final policy Referee is trained, with loss on the
assistant JSON span.  The base checkpoint is configurable so Qwen3 and Gemma can use the identical
data, split, objective, and optimisation settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


BASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma3-4b": "google/gemma-3-4b-it",
}
MAX_LENGTH = 2048


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render the strict-JSON path without spending the response budget on Qwen3 thinking."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class RefereeRows(Dataset):
    def __init__(self, path: Path, tokenizer, limit: int = 0):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if limit:
            self.rows = self.rows[:limit]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        messages = row["messages"]
        prompt = render_prompt(self.tokenizer, messages[:-1])
        answer = messages[-1]["content"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, truncation=True,
                                    max_length=MAX_LENGTH)["input_ids"]
        full_ids = self.tokenizer(prompt + answer, truncation=True,
                                  max_length=MAX_LENGTH)["input_ids"]
        labels = [-100] * min(len(prompt_ids), len(full_ids))
        labels += full_ids[len(labels):]
        spread = max(float(row.get("spread", 0.0)), 1e-9)
        relative_margin = float(row.get("margin", 0.0)) / spread
        weight = 0.25 + 0.75 * min(1.0, relative_margin / 0.05)
        return {"input_ids": full_ids, "labels": labels, "weight": weight}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(row["input_ids"]) for row in batch)
    output = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in batch:
        padding = width - len(row["input_ids"])
        output["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        output["labels"].append(row["labels"] + [-100] * padding)
        output["attention_mask"].append([1] * len(row["input_ids"]) + [0] * padding)
    tensors = {name: torch.tensor(values) for name, values in output.items()}
    tensors["weight"] = torch.tensor([row["weight"] for row in batch], dtype=torch.float)
    return tensors


class MarginWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1]
        labels = inputs["labels"][:, 1:]
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="none").view(labels.shape)
        mask = (labels != -100).float()
        sequence_loss = (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = (sequence_loss * weights.to(sequence_loss.device)).mean()
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("runs/core_referee_oracle_v1/dataset"))
    parser.add_argument("--model", choices=tuple(BASES), default="qwen3-8b")
    parser.add_argument("--base", help="override the Hugging Face checkpoint")
    parser.add_argument("--out", type=Path, default=Path("runs/core_referee_lora"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="use only N rows from each split for a training smoke test")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="set to 0 only for a very short learning-path smoke test")
    args = parser.parse_args()

    base = args.base or BASES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    validation_path = args.data / "validation.jsonl"
    if not validation_path.exists() and (args.data / "val.jsonl").exists():
        validation_path = args.data / "val.jsonl"  # compatibility for a pipeline smoke test
    train = RefereeRows(args.data / "train.jsonl", tokenizer, args.limit)
    validation = RefereeRows(validation_path, tokenizer, args.limit)
    if not train or not validation:
        raise SystemExit("training and validation datasets must both be non-empty")

    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.enable_input_require_grads()
    projection_names = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
    # Gemma 3 is a multimodal checkpoint.  Train only its language tower; adapting matching
    # projections in the frozen vision tower would spend memory on modules this text task never
    # invokes.  Qwen3 is text-only, so the conventional suffix list is sufficient.
    target_modules = (rf".*language_model.*\.({projection_names})$"
                      if args.model == "gemma3-4b" else
                      ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=target_modules))
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=args.warmup_steps, logging_steps=5,
        eval_strategy="steps", eval_steps=20, save_steps=20, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    trainer = MarginWeightedTrainer(
        model=model, args=train_args, train_dataset=train, eval_dataset=validation,
        data_collator=lambda rows: collate(rows, tokenizer.pad_token_id))
    checkpoints = sorted(args.out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=bool(checkpoints))

    adapter_mass = sum(float(parameter.detach().abs().sum())
                       for name, parameter in model.named_parameters() if "lora_B" in name)
    if adapter_mass == 0.0:
        raise SystemExit("REFUSING TO SAVE: LoRA B matrices are still zero")
    final = args.out / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (final / "training_meta.json").write_text(json.dumps({
        "base": base, "model_key": args.model, "train_rows": len(train),
        "validation_rows": len(validation), "epochs": args.epochs,
        "adapter_mass": adapter_mass,
    }, indent=2) + "\n")
    print(f"TRAINING COMPLETE: {final}", flush=True)


if __name__ == "__main__":
    main()
