"""Exp 93 analysis — the two untested structures on the strong packet.
Spec: docs/superpowers/specs/2026-07-28-exp93-untested-structures-design.md

Written AFTER the run (the run's own arms are fixed and its table already printed); the analysis
axes are NOT new — they are inherited verbatim from the Exp 93 pre-registration §4, which inherits
them from Exp 89. Nothing here is chosen after seeing a p-value.

  scoring  : STRICT (handled AND feasible)
  test     : exact McNemar on discordant pairs
  sidedness: ONE-SIDED for H1 and H2, in the §3 directions
  alpha    : 0.05, Holm across the two pre-registered tests
  strata   : POOLED n=81 headline; r4-only n=50 as the blind batch; CONTROLS n=17 never pooled

H1 (argument content): debate-noarg-pkt < debate-pkt.
  Rejecting it means debate's win is the second pass, not the arguments.
H2 (critic):           critic-pkt > single-pkt.

Exp 89's analyser cannot be reused: it tests against `single-pkt-boN`, which is not an arm here
(pre-reg §2 — boN already *is* self-consistency and has its own n=81 result).

EXPLORATORY, labelled as such per pre-reg §4: critic vs debate (confounded by budget, §5 forbids
claiming it as a structure comparison in either direction) and debate vs single (harness check).

  .venv/bin/python -m pins.exp93_analyse pins/results_exp93_qwen2514b.json
"""
from __future__ import annotations

import json
import sys

from pins.exp88_analyse import counts, discordant, mcnemar_exact_two_sided, tost
from pins.exp89_analyse import mcnemar_one_sided

MARGIN = 3.0  # cases; project precedent (Exp 55, Exp 88)


def _row(res, sub, x, y, strict, label):
    """One-sided p for H: x > y, plus the two-sided p, always both."""
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

    print("    PRE-REGISTERED TESTS (one-sided, Holm over the two):")
    b1, c1, p_h1 = _row(res, sub, "debate-pkt", "debate-noarg-pkt", strict,
                        "H1  debate > noarg (arg content)")
    _, _, p_h2 = _row(res, sub, "critic-pkt", "single-pkt", strict,
                      "H2  critic > single")
    # Holm: sort ascending, compare against alpha/(k-i)
    order = sorted([("H1", p_h1), ("H2", p_h2)], key=lambda t: t[1])
    k = len(order)
    for i, (name_, p) in enumerate(order):
        thr = 0.05 / (k - i)
        print(f"      Holm {name_}: p={p:.4f} vs {thr:.4f} -> "
              f"{'SIGNIFICANT' if p < thr else 'not significant'}")

    # H1's decision branch needs an equivalence read, not just a null (pre-reg §6).
    d, lo, hi, ok = tost(b1, c1, len(sub), MARGIN)
    print(f"      H1 equivalence: D={d:+d} cases  90% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"margin +/-{MARGIN:.0f} -> {'PASS' if ok else 'FAIL (CI too wide at this m)'}")

    print("    EXPLORATORY (not claimable as structure comparisons):")
    _row(res, sub, "debate-pkt", "critic-pkt", strict, "debate > critic  [BUDGET-CONFOUNDED]")
    _row(res, sub, "debate-pkt", "single-pkt", strict, "debate > single  [harness check]")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pins/results_exp93_qwen2514b.json"
    with open(path) as f:
        blob = json.load(f)
    res, arms = blob["results"], blob["arms"]

    from pins.hardcases_r3 import CONTROLS as CTRL3, PRIMARY as PRIM3
    from pins.hardcases_r4 import CONTROLS_R4, PRIMARY_R4
    POOLED, CONTROLS = PRIM3 + PRIMARY_R4, CTRL3 + CONTROLS_R4

    prim = [c for c in res if c in set(POOLED)]
    ctrl = [c for c in res if c in set(CONTROLS)]
    print(f"model={blob['model']}  suite={blob.get('suite')}  "
          f"POOLED primary n={len(prim)}  CONTROLS n={len(ctrl)}")

    for strict in (True, False):
        print(f"\n=== {'STRICT (handled AND feasible)' if strict else 'bare handled'} ===")
        _stratum(res, "POOLED PRIMARY (r3 31 + r4 50)", POOLED, arms, strict)
        _stratum(res, "NEW-BATCH (r4 50, blind)", PRIMARY_R4, arms, strict)

    print(f"\n=== CONTROLS n={len(ctrl)} (specificity cost; never pooled) ===")
    for arm in arms:
        print(f"  {arm:18s} handled {counts(res, ctrl, arm, False):>2d}/{len(ctrl)}   "
              f"strict {counts(res, ctrl, arm, True):>2d}/{len(ctrl)}")

    print("\n=== harness reproduction check vs Exp 89 (STRICT, POOLED) ===")
    for arm, want in (("debate-pkt", 43), ("single-pkt", 27)):
        got = counts(res, prim, arm, True)
        print(f"  {arm:12s} {got}/81   Exp 89 reported {want}/81   "
              + ("MATCHES" if got == want else "DIFFERS — report this, do not silently accept it"))


if __name__ == "__main__":
    main()
