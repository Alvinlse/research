"""Score a policy-selector referee by REGRET, against the baselines that make regret meaningful.

Three regret numbers are reported, not one, because a single number here is misleading:

  regret_full   -- over all 15 actions. Mostly measures whether the model avoids ONE obviously
                   broken action (resize_conservative+greedy deadlocks the cluster and is the
                   worst choice in ~91% of states), so a useless model scores well on it.
  regret_safe   -- the HEADLINE. Same states, catastrophic arms removed from the menu, so it
                   measures the choice that is actually in question.
  catastrophe_rate -- how often the model picks a catastrophic arm at all, reported separately
                   rather than folded into a mean.

The baselines are not optional. ~62% of states are near-ties and one action wins a large
plurality, so a constant predictor is strong; if the tuned model cannot beat `majority`, it has
learned nothing, and that is the result.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics
from pathlib import Path

from pins.elastisim_bench import POLICY_MENU

ACTIONS = [f"{o}+{z}" for o in POLICY_MENU["ordering"] for z in POLICY_MENU["sizing"]]


def regret(rows: list[dict], pick, menu: list[str]) -> dict:
    """Mean/percentile shortfall of `pick` against the best action available in `menu`."""
    out, bad, invalid = [], 0, 0
    for r in rows:
        sc = {a: v for a, v in r["rewards"].items() if a in menu}
        a = pick(r)
        if a not in sc:                      # off-menu, or a catastrophic arm under regret_safe
            invalid += 1
            a = min(sc, key=sc.get)          # scored as the worst legal option, never skipped
            bad += 1
        out.append(max(sc.values()) - sc[a])
    out.sort()
    n = len(out)
    return {"mean": round(statistics.mean(out), 4), "p50": round(out[n // 2], 4),
            "p90": round(out[int(0.9 * (n - 1))], 4), "max": round(out[-1], 4),
            "eps1": round(sum(x <= 0.01 for x in out) / n, 3),
            "eps5": round(sum(x <= 0.05 for x in out) / n, 3),
            "off_menu": invalid}


def load_model(path: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pins.finetune_referee import BASE
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    if path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(path))
    model.eval()
    return model, tok


def generate(rows, model, tok, tag: str) -> list[str]:
    import torch
    picks = []
    for i, r in enumerate(rows):
        m = r["messages"][:-1]
        p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=60, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        txt = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        mm = re.search(r'"ordering"\s*:\s*"([^"]+)".*?"sizing"\s*:\s*"([^"]+)"', txt, re.S)
        picks.append(f"{mm.group(1)}+{mm.group(2)}" if mm else "UNPARSEABLE")
        if (i + 1) % 20 == 0:
            print(f"  [{tag}] {i+1}/{len(rows)}", flush=True)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/dataset"))
    ap.add_argument("--lora", type=Path, default=Path("runs/referee_lora/final"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("runs/eval_report.json"))
    ap.add_argument("--skip-llm", action="store_true", help="baselines only")
    a = ap.parse_args()

    rows = [json.loads(l) for l in (a.data / f"{a.split}.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((a.data / "meta.json").read_text())
    cat = meta["catastrophic_arms"]
    safe = [x for x in ACTIONS if x not in cat]
    train = [json.loads(l) for l in (a.data / "train.jsonl").read_text().splitlines() if l.strip()]
    majority = collections.Counter(r["target"] for r in train).most_common(1)[0][0]
    print(f"{len(rows)} {a.split} states | catastrophic={cat} | majority baseline={majority}")

    # Draw the random baselines ONCE per state. `regret()` and the catastrophe count each call the
    # strategy, so a live rng.choice() returned a different action to each metric -- which is how
    # random_safe_menu came out with regret_safe > regret_full, an impossibility for a fixed pick.
    rng = random.Random(0)
    fixed_rand = {(r["window"], r["t"]): rng.choice(ACTIONS) for r in rows}
    fixed_rand_safe = {(r["window"], r["t"]): rng.choice(safe) for r in rows}
    best_fixed = max(ACTIONS, key=lambda x: sum(t["rewards"].get(x, -9e9) for t in train))
    strategies = {
        "majority_constant": lambda r: majority,
        "random_menu": lambda r: fixed_rand[(r["window"], r["t"])],
        "random_safe_menu": lambda r: fixed_rand_safe[(r["window"], r["t"])],
        # The best SINGLE fixed action chosen on TRAIN -- the honest "best fixed policy" baseline,
        # since choosing it on test would be peeking.
        "best_fixed_on_train": lambda r, b=best_fixed: b,
    }
    report = {"n": len(rows), "split": a.split, "catastrophic_arms": cat,
              "majority": majority, "best_fixed_policy": best_fixed,
              "test_windows": sorted({r["window"] for r in rows}), "arms": {}}

    if not a.skip_llm:
        for name, lora in (("zero_shot", None), ("finetuned", a.lora if a.lora.exists() else None)):
            if name == "finetuned" and lora is None:
                print("!! no LoRA adapter found; skipping finetuned arm"); continue
            model, tok = load_model(lora)
            picks = generate(rows, model, tok, name)
            strategies[name] = lambda r, p=dict(zip(range(len(rows)), picks)), \
                idx={id(x): i for i, x in enumerate(rows)}: p[idx[id(r)]]
            report["arms"].setdefault(name, {})["parse_fail"] = picks.count("UNPARSEABLE")
            report["arms"][name]["picks"] = dict(collections.Counter(picks).most_common(8))
            # Preserve the window pairing. Aggregate pick counts hid that the
            # first apparent model gap was carried by one test window.
            report["arms"][name]["decisions"] = [
                {"window": r["window"], "t": r["t"], "policy": p}
                for r, p in zip(rows, picks)]
            del model
            import torch, gc; gc.collect(); torch.cuda.empty_cache()

    for name, fn in strategies.items():
        d = report["arms"].setdefault(name, {})
        d["regret_full"] = regret(rows, fn, ACTIONS)
        d["regret_safe"] = regret(rows, fn, safe)
        d["catastrophe_rate"] = round(sum(fn(r) in cat for r in rows) / len(rows), 3)
    # The ceiling, by construction: the rollout selector picks the argmax of the stored vector.
    report["arms"]["rollout_selector_ceiling"] = {
        "regret_full": regret(rows, lambda r: max(r["rewards"], key=r["rewards"].get), ACTIONS)}

    a.out.write_text(json.dumps(report, indent=1))
    print(f"\n{'arm':<26} {'regret_safe':>12} {'regret_full':>12} {'eps5_safe':>10} {'catastrophe':>12}")
    for k, v in report["arms"].items():
        if "regret_safe" in v:
            print(f"{k:<26} {v['regret_safe']['mean']:>12.4f} {v['regret_full']['mean']:>12.4f} "
                  f"{v['regret_safe']['eps5']:>10.3f} {v['catastrophe_rate']:>12.3f}")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
