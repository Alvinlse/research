"""Exp 79 — model-family generality of the perspective x text interaction.

PRE-REGISTERED 2026-07-22, written and committed BEFORE any model beyond qwen2.5:14b was run.
The 14b result is already known and is a flat null (PRIMARY n=31: text effect +11 for BOTH
structures, interaction +0, zero discordant pairs). This analysis exists because one model
cannot kill a mechanism: the split may plausibly matter for models that 14b's competence hides.

  H_int (unchanged): (referee_text - referee_notext) > (single_text - single_notext)

  PER MODEL: exact McNemar on the per-case difference-in-differences, restricted to PRIMARY.
  POOLED:    stratified McNemar across models — sum the discordant counts b and c over strata
             and test the pooled pair exactly. This is the matched-pair CMH; it is valid
             because each case contributes one pair per model and models are the strata.
  DIRECTION: one-sided, alpha = 0.05. Controls are reported separately and NEVER pooled in.

  MECHANISTIC PREDICTION, declared in advance so the shape is falsifiable: if the split works by
  SCAFFOLDING the use of free text, the interaction should be largest in the MIDDLE of the
  capability ladder — models strong enough to follow the structure but too weak to exploit the
  text unaided — and vanish at both ends. A flat zero across the whole ladder means the
  mechanism does not exist at any scale, and the multi-agent line is closed on evidence rather
  than on one model.

  SECONDARY, and given the 14b outcome arguably the more valuable half: the TEXT MAIN EFFECT per
  model. Text moved 14b from 2/31 to 13/31 against ILP/rule at 0/31. Whether that generalises
  across families is the generality test for the claim that actually survived.

Note on a bias that does NOT apply here: the suite was sharpened until the rigid arms failed,
which inflates any LLM-vs-ILP comparison. It is neutral for the text ablation and for the
interaction, since every arm faces the identical sharpened cases.

  .venv/bin/python -m pins.exp79_analyse
"""
from __future__ import annotations

import glob
import json
import os
import re
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("single", "single-noarg", "referee", "referee-noarg")


def exact_p(b: int, c: int, one_sided: bool = True) -> float:
    """Exact binomial McNemar on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    # one-sided in the direction of H_int (referee-favoured = b)
    p = sum(comb(n, i) for i in range(c + 1)) / 2 ** n
    return min(1.0, p if one_sided else 2 * min(p, 1 - p + comb(n, c) / 2 ** n))


def load() -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results_hardcases_r3_2x2_*.json"))):
        d = json.load(open(path))
        if set(ARMS) <= set(d.get("arms", [])):
            out[d["model"]] = d["results"]
    return out


def counts(res: dict, ids: list[str]) -> dict:
    h = {a: sum(1 for c in ids if res[c]["arms"][a]["handled"]) for a in ARMS}
    dd = [(res[c]["arms"]["referee"]["handled"] - res[c]["arms"]["referee-noarg"]["handled"])
          - (res[c]["arms"]["single"]["handled"] - res[c]["arms"]["single-noarg"]["handled"])
          for c in ids]
    return {"h": h, "b": sum(1 for x in dd if x > 0), "c": sum(1 for x in dd if x < 0),
            "text_single": h["single"] - h["single-noarg"],
            "text_referee": h["referee"] - h["referee-noarg"]}


def main() -> None:
    from pins.hardcases_r3 import CONTROLS, PRIMARY

    runs = load()
    if not runs:
        print("no 2x2 result files yet (pins/results_hardcases_r3_2x2_*.json)")
        return

    # ladder order: roughly ascending capability, for the inverted-U prediction
    order = ["qwen2.5:0.5b", "qwen2.5:1.5b", "gemma2:2b", "qwen2.5:3b", "qwen2.5:7b",
             "llama3:8b", "gemma2:9b", "qwen2.5:14b", "gemma2:27b", "deepseek-r1:32b",
             "qwen3.5:35b"]
    models = sorted(runs, key=lambda m: (order.index(m) if m in order else 99, m))

    for label, ids in (("PRIMARY (pre-registered)", PRIMARY), ("CONTROLS", CONTROLS)):
        print(f"\n=== {label}  n={len(ids)} ===")
        print(f"{'model':18s}{'single':>16s}{'referee':>16s}"
              f"{'text S':>8s}{'text R':>8s}{'inter':>7s}{'b':>4s}{'c':>4s}{'p':>8s}")
        B = C = 0
        for m in models:
            k = counts(runs[m], ids)
            if label.startswith("PRIMARY"):
                B += k["b"]; C += k["c"]
            print(f"{m:18s}{k['h']['single']:>8d}/{k['h']['single-noarg']:<7d}"
                  f"{k['h']['referee']:>8d}/{k['h']['referee-noarg']:<7d}"
                  f"{k['text_single']:>+8d}{k['text_referee']:>+8d}"
                  f"{k['text_referee'] - k['text_single']:>+7d}"
                  f"{k['b']:>4d}{k['c']:>4d}{exact_p(k['b'], k['c']):>8.3f}")
        if label.startswith("PRIMARY"):
            print(f"\n  POOLED stratified McNemar over {len(models)} model(s): "
                  f"b={B} c={C}  one-sided exact p={exact_p(B, C):.4f}")
            print("  (b = referee-favoured discordant pairs, c = single-favoured)")

    print("\n=== rescue view: among cases the SINGLE LLM fails with text, does referee save it? ===")
    print(f"{'model':18s}{'single fails':>13s}{'rescued':>9s}{'broken':>8s}{'net':>6s}{'p':>8s}")
    for m in models:
        res = runs[m]
        fails = [c for c in PRIMARY if not res[c]["arms"]["single"]["handled"]]
        b = sum(1 for c in fails if res[c]["arms"]["referee"]["handled"])
        cc = sum(1 for c in PRIMARY if res[c]["arms"]["single"]["handled"]
                 and not res[c]["arms"]["referee"]["handled"])
        print(f"{m:18s}{len(fails):>13d}{b:>9d}{cc:>8d}{b - cc:>+6d}{exact_p(b, cc):>8.3f}")


if __name__ == "__main__":
    main()
