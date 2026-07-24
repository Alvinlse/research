"""Exp 89 analysis — PRE-REGISTERED 2026-07-24, committed BEFORE the run.
Spec: docs/superpowers/specs/2026-07-24-exp89-round3-extension-design.md

PRIMARY TEST. One-sided exact McNemar, debate-pkt vs single-pkt-boN (the budget-matched single
LLM), on the handled indicator, STRICT scoring. H1: debate handles MORE than boN — the rebuttal
round buys something more samples at the same budget cannot.

Two strata printed from the one run (spec §2):
  POOLED PRIMARY (round-3 31 + round-4 50 = 81) — the headline; powered to ~0.83 at the Exp 88
    point estimate.
  NEW-BATCH PRIMARY (round-4 50 only) — the clean confirmatory number, authored blind to Exp 88
    per-case outcomes; reported as SUPPORTING, not decisive.

Decision rule (spec §8):
  pooled p<0.05 AND new-batch direction agrees   -> structure beats matched budget; confirmed.
  pooled p>=0.05 but direction holds             -> inconclusive, consistent with structure.
  debate ~= boN both directions                  -> structure is sampling; multi-agent line closes.
  controls show a text effect in any arm         -> discount the primary as text-volume.

SECONDARY: two-sided McNemar; debate vs the original single-pkt (the Exp 83 comparison, must
still reproduce b=6 c=1 on the round-3 31 to confirm the harness); bare handled beside STRICT;
controls, never pooled into the primary.

  .venv/bin/python -m pins.exp89_analyse pins/results_exp89_qwen2514b_t0.8.json
"""
from __future__ import annotations

import json
import sys
from math import comb

# reuse the exact tests written and reviewed for Exp 88
from pins.exp88_analyse import counts, discordant, mcnemar_exact_two_sided


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X >= b | X ~ Binom(b+c, 0.5)) — the exact one-sided test for H1: debate > boN."""
    m = b + c
    if m == 0:
        return 1.0
    return sum(comb(m, i) for i in range(b, m + 1)) / 2 ** m


def _pair(res, sub, x, y, strict, label):
    b, c = discordant(res, sub, x, y, strict)
    p1 = mcnemar_one_sided(b, c)
    p2 = mcnemar_exact_two_sided(b, c)
    print(f"    {label:24s} b={b} c={c}  D={b - c:+d}  "
          f"McNemar 1-sided p={p1:.4f}  (2-sided {p2:.4f})")
    return b, c, p1


def _stratum(res, name, ids, arms, strict):
    sub = [c for c in res if c in set(ids)]
    print(f"\n  {name}  n={len(sub)}")
    for arm in arms:
        nc = sum(res[c]["arms"][arm]["meta"]["n_calls"] for c in sub)
        print(f"    {arm:16s} {counts(res, sub, arm, strict):>3d}/{len(sub):<3d} calls {nc:>5d}")
    print("    PRIMARY TEST (H1: debate-pkt > single-pkt-boN):")
    b, c, p1 = _pair(res, sub, "debate-pkt", "single-pkt-boN", strict, "debate vs boN")
    verdict = ("debate > boN, significant — STRUCTURE beats matched budget" if p1 < 0.05 else
               "not significant at this n — direction " +
               ("favors debate (inconclusive, consistent with structure)" if b > c else
                "does not favor debate"))
    print(f"      -> {verdict}")
    print("    SECONDARY:")
    _pair(res, sub, "debate-pkt", "single-pkt", strict, "debate vs single")
    _pair(res, sub, "single-pkt-boN", "single-pkt", strict, "boN vs single")


def per_category(res, ids, arms, strict):
    """EXPLORATORY, post-hoc: where does the debate edge live? n per category is 8-21, so this
    is direction-only — never a significance claim. Not part of the pre-registered test."""
    prim = [c for c in res if c in set(ids)]
    cats: list[str] = []
    for c in prim:
        if res[c]["category"] not in cats:
            cats.append(res[c]["category"])
    print("\n=== EXPLORATORY — per-category (STRICT, POOLED primary) ===")
    print("  post-hoc slice, small n per cell (underpowered) — read as DIRECTION, not significance")
    print(f"  {'category':14s}{'n':>4s}{'market':>8s}{'single':>8s}{'boN':>8s}{'debate':>8s}"
          f"{'dbt-v-boN':>13s}{'dbt-v-sgl':>12s}")
    for cat in cats:
        sub = [c for c in prim if res[c]["category"] == cat]
        cells = "".join(f"{counts(res, sub, a, strict):>5d}/{len(sub):<2d}" for a in arms)
        b1, c1 = discordant(res, sub, "debate-pkt", "single-pkt-boN", strict)
        b2, c2 = discordant(res, sub, "debate-pkt", "single-pkt", strict)
        print(f"  {cat:14s}{len(sub):>4d}{cells}{f'  b={b1} c={c1}':>13s}{f'  b={b2} c={c2}':>12s}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pins/results_exp89_qwen2514b_t0.8.json"
    with open(path) as f:
        blob = json.load(f)
    res = blob["results"]
    arms = blob["arms"]

    from pins.hardcases_r3 import CONTROLS as CTRL3, PRIMARY as PRIM3
    from pins.hardcases_r4 import CONTROLS_R4, PRIMARY_R4
    POOLED = PRIM3 + PRIMARY_R4
    NEWBATCH = PRIMARY_R4
    CONTROLS = CTRL3 + CONTROLS_R4

    prim = [c for c in res if c in set(POOLED)]
    ctrl = [c for c in res if c in set(CONTROLS)]
    print(f"model={blob['model']}  boN temperature={blob.get('temperature_boN')}  "
          f"suite={blob.get('suite')}  POOLED primary n={len(prim)}  CONTROLS n={len(ctrl)}")

    for strict in (True, False):
        print(f"\n=== {'STRICT (handled AND feasible)' if strict else 'bare handled'} ===")
        _stratum(res, "POOLED PRIMARY (r3 31 + r4 50)", POOLED, arms, strict)
        _stratum(res, "NEW-BATCH PRIMARY (r4 50, clean confirmatory)", NEWBATCH, arms, strict)

    per_category(res, POOLED, arms, strict=True)

    print(f"\n=== CONTROLS n={len(ctrl)} (boundary: text effect must be ~0; never pooled) ===")
    for arm in arms:
        print(f"  {arm:16s} handled {counts(res, ctrl, arm, False):>2d}/{len(ctrl)}   "
              f"strict {counts(res, ctrl, arm, True):>2d}/{len(ctrl)}")

    print("\n=== harness reproduction check vs Exp 83 (STRICT, round-3 31 only) ===")
    r3prim = [c for c in res if c in set(PRIM3)]
    b3, c3 = discordant(res, r3prim, "debate-pkt", "single-pkt", True)
    print(f"  debate vs single on r3: b={b3} c={c3}   Exp 83 reported b=6 c=1")
    print("  " + ("MATCHES Exp 83" if (b3, c3) == (6, 1) else
                  "DOES NOT MATCH Exp 83 — report this, do not silently accept it"))


if __name__ == "__main__":
    main()
