"""Exp 95 analysis — PRE-REGISTERED 2026-07-28, committed BEFORE the run.
Spec: docs/superpowers/specs/2026-07-28-exp95-budget-matched-critic-design.md

PRIMARY TEST. One-sided exact McNemar, `critic-pkt` vs `single-pkt-boNc` (best-of-N drawing
k = n_text_jobs + 1 samples, critic's per-case call count exactly), STRICT scoring, POOLED n=81.
H1: critic handles MORE than the budget-matched sampler — the objection channel buys something
resampling at the same spend does not.

This is the ONLY pre-registered test, so no multiplicity correction applies (spec §3). That is a
consequence of asking one question, not a relaxation of the standard Exp 93's H2 died to.

BUDGET MATCHING IS VERIFIED, NOT ASSUMED. The two arms' per-case `n_calls` must be equal for every
case; any mismatch is printed as a hard failure and invalidates the comparison.

SECONDARY (spec §3): TOST at +/-3 cases for the null read; boNc vs single-pkt (pure sampling at
critic's budget); critic vs single-pkt, which must reproduce Exp 93's b=10 c=3; bare handled;
r4-only blind stratum; CONTROLS n=17 never pooled.

Declared in advance (spec §5): an inconclusive McNemar and a FAILED TOST are both expected
possibilities at m~10-20 discordant pairs. Neither may afterwards be read as support.

  .venv/bin/python -m pins.exp95_analyse pins/results_exp95_qwen2514b.json
"""
from __future__ import annotations

import json
import sys

from pins.exp88_analyse import counts, discordant, mcnemar_exact_two_sided, tost
from pins.exp89_analyse import mcnemar_one_sided

MARGIN = 3.0
PRIMARY_PAIR = ("critic-pkt", "single-pkt-boNc")


def _row(res, sub, x, y, strict, label):
    b, c = discordant(res, sub, x, y, strict)
    p1, p2 = mcnemar_one_sided(b, c), mcnemar_exact_two_sided(b, c)
    print(f"    {label:34s} b={b} c={c}  D={b - c:+d}  1-sided p={p1:.4f}  (2-sided {p2:.4f})")
    return b, c, p1


def _stratum(res, name, ids, arms, strict):
    sub = [c for c in res if c in set(ids)]
    print(f"\n  {name}  n={len(sub)}")
    for arm in arms:
        nc = sum(res[c]["arms"][arm]["meta"]["n_calls"] for c in sub)
        print(f"    {arm:18s} {counts(res, sub, arm, strict):>3d}/{len(sub):<3d} calls {nc:>5d}")

    print(f"    PRIMARY TEST (H1: {PRIMARY_PAIR[0]} > {PRIMARY_PAIR[1]}), one-sided, alpha 0.05:")
    b, c, p1 = _row(res, sub, *PRIMARY_PAIR, strict, "critic > boNc")
    if p1 < 0.05:
        verdict = "critic > matched budget — the objection channel is STRUCTURE"
    elif b > c:
        verdict = "not significant; direction favours critic — inconclusive, do not spin it"
    else:
        verdict = "not significant; direction does NOT favour critic"
    print(f"      -> {verdict}")
    d, lo, hi, ok = tost(b, c, len(sub), MARGIN)
    print(f"      equivalence: D={d:+d} cases  90% CI [{lo:+.2f}, {hi:+.2f}]  margin +/-{MARGIN:.0f}"
          f" -> {'PASS — critic is sampling, not objection' if ok else 'FAIL (CI too wide at this m)'}")

    print("    SECONDARY:")
    _row(res, sub, "single-pkt-boNc", "single-pkt", strict, "boNc > single (pure sampling)")
    _row(res, sub, "critic-pkt", "single-pkt", strict, "critic > single [Exp 93 check]")


def check_budget(res, arms) -> None:
    """Per-case call-count equality between critic and its control. Matched PER CASE, not in total
    (Exp 88 precedent) — a total-only match could hide compensating per-case errors."""
    if not all(a in arms for a in PRIMARY_PAIR):
        return
    bad = [(c, res[c]["arms"][PRIMARY_PAIR[0]]["meta"]["n_calls"],
            res[c]["arms"][PRIMARY_PAIR[1]]["meta"]["n_calls"])
           for c in res
           if res[c]["arms"][PRIMARY_PAIR[0]]["meta"]["n_calls"]
           != res[c]["arms"][PRIMARY_PAIR[1]]["meta"]["n_calls"]]
    print(f"\n=== BUDGET MATCH CHECK (per case, all {len(res)}) ===")
    if not bad:
        tot = sum(res[c]["arms"][PRIMARY_PAIR[0]]["meta"]["n_calls"] for c in res)
        print(f"  MATCHED exactly on every case; {tot} calls each")
    else:
        print(f"  *** MISMATCH on {len(bad)} case(s) — THE PRIMARY COMPARISON IS INVALID ***")
        for c, x, y in bad[:10]:
            print(f"    {c}: critic={x} boNc={y}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pins/results_exp95_qwen2514b.json"
    with open(path) as f:
        blob = json.load(f)
    res, arms = blob["results"], blob["arms"]

    from pins.hardcases_r3 import CONTROLS as CTRL3, PRIMARY as PRIM3
    from pins.hardcases_r4 import CONTROLS_R4, PRIMARY_R4
    POOLED, CONTROLS = PRIM3 + PRIMARY_R4, CTRL3 + CONTROLS_R4

    prim = [c for c in res if c in set(POOLED)]
    ctrl = [c for c in res if c in set(CONTROLS)]
    print(f"model={blob['model']}  suite={blob.get('suite')}  "
          f"boNc temperature={blob.get('temperature_boN')}  "
          f"POOLED primary n={len(prim)}  CONTROLS n={len(ctrl)}")

    check_budget(res, arms)

    for strict in (True, False):
        print(f"\n=== {'STRICT (handled AND feasible)' if strict else 'bare handled'} ===")
        _stratum(res, "POOLED PRIMARY (r3 31 + r4 50)", POOLED, arms, strict)
        _stratum(res, "NEW-BATCH (r4 50, blind)", PRIMARY_R4, arms, strict)

    print(f"\n=== CONTROLS n={len(ctrl)} (specificity cost; never pooled) ===")
    for arm in arms:
        print(f"  {arm:18s} handled {counts(res, ctrl, arm, False):>2d}/{len(ctrl)}   "
              f"strict {counts(res, ctrl, arm, True):>2d}/{len(ctrl)}")

    print("\n=== harness reproduction check vs Exp 93 (STRICT, POOLED) ===")
    for arm, want in (("critic-pkt", 34), ("single-pkt", 27)):
        if arm not in arms:
            continue
        got = counts(res, prim, arm, True)
        print(f"  {arm:12s} {got}/81   Exp 93 reported {want}/81   "
              + ("MATCHES" if got == want else "DIFFERS — report this, do not silently accept it"))
    b, c = discordant(res, prim, "critic-pkt", "single-pkt", True)
    print(f"  critic vs single: b={b} c={c}   Exp 93 reported b=10 c=3   "
          + ("MATCHES" if (b, c) == (10, 3) else "DIFFERS — report this"))


if __name__ == "__main__":
    main()
