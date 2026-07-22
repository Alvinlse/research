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

AMENDMENT, 2026-07-22, after qwen2.5:1.5b (the FIRST ladder model) and before any other model
was run. Disclosed rather than silently applied.

  qwen2.5:1.5b allocated 146-156% of the pool and "passed" 12/31 primary cases by awarding
  capacity that does not exist: it over-awarded on 25 of 31. Bare `handled` therefore rewards
  infeasibility, and on a weak model it measures brute force rather than judgement. Under
  STRICT scoring (handled AND feasible) its 12 becomes 2, while qwen2.5:14b moves 13 -> 12.

  So STRICT is the headline metric for the ladder and bare `handled` is reported beside it.
  This is not a new invention chosen to fit a result: over-award is named as a secondary axis
  in the pins/hardcases_r3.py pre-registration, feasibility is rule 1 of the referee prompt,
  and Exp 54 already reported round 2 as "4/12, or 3/12 scoring over-award as auto-fail".
  It is applied uniformly to every arm and model, it is strictly more conservative, and it
  cannot select the interaction's direction — b and c are reported under both scorings.

COMPETENCE BAR, locked 2026-07-22 22:44 JST — after qwen2.5:1.5b and gemma2:2b, while
qwen2.5:3b was at 33/40 with NO result file written and none of its numbers seen.

  Why: both sub-3b models allocate ~150% of the pool and fail rule 1 (feasibility) on ~80% of
  cases. Their bare `handled` counts come from awarding GPUs that do not exist. An arm that
  breaks feasibility has not produced an allocation, it has produced a number, so a predicate
  it happens to satisfy is not evidence of judgement — and an interaction measured among such
  arms cannot support a claim about scheduling.

  Rule: a model enters the CONFIRMATORY pool iff its infeasibility rate over PRIMARY, counted
  across all four arms (over-awarded arm-case pairs / (4 x 31)), is BELOW 25%.

  The criterion is deliberately FEASIBILITY, not score. Feasibility is rule-1 compliance — a
  format/competence property, orthogonal to the difference-in-differences under test. Barring
  on `handled` would risk selecting on the outcome; barring on over-award cannot, because b and
  c are differences of differences that the rate does not enter.

  Excluded models are still printed in full, and the pooled statistic is reported BOTH ways
  (confirmatory-only and all-models) so the exclusion can never hide a result.

  Current rates: qwen2.5:1.5b 70%, gemma2:2b 81%, qwen2.5:14b 1.6%. The bar separates these
  cleanly and was not tuned to any threshold in between.

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


def ok(res: dict, cid: str, arm: str, strict: bool) -> int:
    a = res[cid]["arms"][arm]
    return int(a["handled"] and not (strict and a["overcommitted"]))


BAR = 0.25          # locked 2026-07-22 22:44 JST, before qwen2.5:3b's results existed


def infeasibility(res: dict, ids: list[str]) -> float:
    """Share of arm-case pairs whose allocation broke rule 1. The competence bar."""
    bad = sum(res[c]["arms"][a]["overcommitted"] for c in ids for a in ARMS)
    return bad / (len(ARMS) * len(ids))


def counts(res: dict, ids: list[str], strict: bool = True) -> dict:
    h = {a: sum(ok(res, c, a, strict) for c in ids) for a in ARMS}
    dd = [(ok(res, c, "referee", strict) - ok(res, c, "referee-noarg", strict))
          - (ok(res, c, "single", strict) - ok(res, c, "single-noarg", strict))
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

    for strict in (True, False):
        tag = ("STRICT: handled AND feasible (headline)" if strict
               else "BARE handled: over-award NOT penalised (reported for continuity)")
        print(f"\n########## {tag} ##########")
        for label, ids in (("PRIMARY (pre-registered)", PRIMARY), ("CONTROLS", CONTROLS)):
            print(f"\n=== {label}  n={len(ids)} ===")
            print(f"{'model':18s}{'single':>16s}{'referee':>16s}"
                  f"{'text S':>8s}{'text R':>8s}{'inter':>7s}{'b':>4s}{'c':>4s}{'p':>8s}"
                  f"{'infeas':>8s}")
            B = C = Bc = Cc = 0
            for m in models:
                k = counts(runs[m], ids, strict)
                rate = infeasibility(runs[m], PRIMARY)
                elig = rate < BAR
                if label.startswith("PRIMARY"):
                    B += k["b"]; C += k["c"]
                    if elig:
                        Bc += k["b"]; Cc += k["c"]
                print(f"{m:18s}{k['h']['single']:>8d}/{k['h']['single-noarg']:<7d}"
                      f"{k['h']['referee']:>8d}/{k['h']['referee-noarg']:<7d}"
                      f"{k['text_single']:>+8d}{k['text_referee']:>+8d}"
                      f"{k['text_referee'] - k['text_single']:>+7d}"
                      f"{k['b']:>4d}{k['c']:>4d}{exact_p(k['b'], k['c']):>8.3f}"
                      f"{rate:>7.0%}{'' if elig else ' *below bar'}")
            if label.startswith("PRIMARY"):
                elig = [m for m in models if infeasibility(runs[m], PRIMARY) < BAR]
                print(f"\n  CONFIRMATORY pool (infeasibility < {BAR:.0%}), "
                      f"{len(elig)}/{len(models)} model(s) {elig}:")
                print(f"    b={Bc} c={Cc}  one-sided exact p={exact_p(Bc, Cc):.4f}")
                print(f"  all models incl. below-bar ({len(models)}): "
                      f"b={B} c={C}  p={exact_p(B, C):.4f}")
                print("  (b = referee-favoured discordant pairs, c = single-favoured)")

    print("\n=== rescue view (STRICT): among PRIMARY cases the single LLM fails, "
          "does the referee save it? ===")
    print(f"{'model':18s}{'single fails':>13s}{'rescued':>9s}{'broken':>8s}{'net':>6s}{'p':>8s}"
          f"{'  feasibility of single':>24s}")
    for m in models:
        res = runs[m]
        fails = [c for c in PRIMARY if not ok(res, c, "single", True)]
        b = sum(1 for c in fails if ok(res, c, "referee", True))
        cc = sum(1 for c in PRIMARY if ok(res, c, "single", True)
                 and not ok(res, c, "referee", True))
        over = sum(1 for c in PRIMARY if res[c]["arms"]["single"]["overcommitted"])
        print(f"{m:18s}{len(fails):>13d}{b:>9d}{cc:>8d}{b - cc:>+6d}{exact_p(b, cc):>8.3f}"
              f"{f'  over-awarded {over}/{len(PRIMARY)}':>24s}")


if __name__ == "__main__":
    main()
