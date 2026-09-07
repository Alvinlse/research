"""LoRA SFT of the policy-selector referee: packet -> {ordering, sizing}.

Deliberately plain `transformers.Trainer` + `peft` rather than a higher-level SFT wrapper: this
runs unattended overnight on a node that kills a process every ~12-15 CPU-min, so it needs
checkpoint/resume to be exactly the well-understood HF one, with no wrapper API in between.

Loss is taken on the assistant span only -- the packet is ~1.6k characters of cluster state that
the model must condition on, not reproduce.

Examples are weighted by `margin` (best minus second-best reward). ~62% of states are near-ties
where the label is close to arbitrary, so training them at full strength teaches noise; they are
kept, because they carry real information about what NOT to pick, but they carry less weight.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

BASE = "Qwen/Qwen2.5-3B-Instruct"
MAXLEN = 2048


class Packets(Dataset):
    def __init__(self, path: Path, tok, min_weight: float = 0.25):
        self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.tok, self.min_weight = tok, min_weight

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        m = r["messages"]
        prompt = self.tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        full = prompt + m[-1]["content"] + self.tok.eos_token
        ids = self.tok(full, truncation=True, max_length=MAXLEN)["input_ids"]
        n_prompt = len(self.tok(prompt, truncation=True, max_length=MAXLEN)["input_ids"])
        labels = [-100] * min(n_prompt, len(ids)) + ids[min(n_prompt, len(ids)):]
        # A near-tie is a weak teaching signal, not a wrong one: squash rather than drop.
        # RELATIVE margin, not absolute. Raw margin scales with the window's reward spread, so
        # loaded windows -- where every policy is within ~2% of the others and nothing can be won --
        # were getting the LARGEST weights, while idle windows, which hold ~30% of the available
        # gain, were being trained at the floor. Dividing by the state's own spread makes the weight
        # mean "how decisive is this choice", independent of how hard the window is.
        rv = list(r.get("rewards", {}).values())
        spread = (max(rv) - min(rv)) if rv else 0.0
        rel = r.get("margin", 0) / spread if spread > 1e-9 else 0.0
        w = self.min_weight + (1 - self.min_weight) * min(1.0, rel / 0.05)
        return {"input_ids": ids, "labels": labels, "weight": w}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
    d = {k: torch.tensor(v) for k, v in out.items()}
    d["weight"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float)
    return d


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        w = inputs.pop("weight")
        out = model(**inputs)
        logits, labels = out.logits[:, :-1], inputs["labels"][:, 1:]
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="none").view(labels.shape)
        mask = (labels != -100).float()
        per_seq = (ll * mask).sum(1) / mask.sum(1).clamp(min=1)
        # Scale each example's loss by its margin WITHOUT dividing by that example's own weight.
        # The previous form, (per_seq*w).sum()/w.sum(), cancels exactly at batch size 1 -- which is
        # the configured batch size -- so the margin weighting was algebraically a no-op and near
        # ties were trained at full strength anyway.
        loss = (per_seq * w.to(per_seq.device)).mean()
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/dataset"))
    ap.add_argument("--out", type=Path, default=Path("runs/referee_lora"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="smoke run on N rows")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    train = Packets(a.data / "train.jsonl", tok)
    val = Packets(a.data / "val.jsonl", tok)
    if a.limit:
        train.rows = train.rows[:a.limit]
    print(f"training rows: {len(train)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(a.out), num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        # warmup_steps, not warmup_ratio: on a short run a ratio rounds to zero warmup, and with
        # a cosine schedule a 1-step run decays the LR to ~0 before the only optimiser step -- the
        # LoRA B matrices then stay at their zero init and the adapter is a silent no-op.
        learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=5,
        logging_steps=5, eval_strategy="steps", eval_steps=20,
        save_steps=20, save_total_limit=3,                    # frequent: the reaper is the reason
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=True, report_to=[], remove_unused_columns=False,
        gradient_checkpointing=True)
    tr = WeightedTrainer(model=model, args=args, train_dataset=train,
                         eval_dataset=val,
                         data_collator=lambda b: collate(b, tok.pad_token_id))
    ck = list(a.out.glob("checkpoint-*"))
    resume = bool(ck)
    print(f"resume_from_checkpoint={resume} ({len(ck)} checkpoints present)", flush=True)
    tr.train(resume_from_checkpoint=resume)
    mass = sum(float(p_.abs().sum()) for n, p_ in model.named_parameters() if "lora_B" in n)
    print(f"lora_B mass after training: {mass:.6f}", flush=True)
    if mass == 0.0:
        raise SystemExit("REFUSING TO SAVE: lora_B is still zero, so nothing was learned "
                         "and the adapter is a no-op. Check the LR schedule and step count.")
    model.save_pretrained(str(a.out / "final"))
    tok.save_pretrained(str(a.out / "final"))
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
