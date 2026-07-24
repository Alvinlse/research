"""Exp 88 analysis — PRE-REGISTERED 2026-07-24, committed BEFORE the run.

PRIMARY TEST. TOST equivalence between `debate-pkt` and `single-pkt-boN` (the budget-matched
single LLM), on the round-3 PRIMARY 31, STRICT scoring.

  estimand : D = (# cases debate handles) - (# cases boN handles), in CASES out of 31.
             On paired binary data D = b - c, the discordant counts.
  margin   : +/- 3 cases (declared in pins/exp88_budget_control.py before the run).
             Project precedent: Exp 55 used +/-3 for the debate-vs-referee increment.
  alpha    : 0.05, and TOST at alpha is exactly "the 90% CI for D lies inside (-3, +3)".
  method   : exact conditional. Condition on m = b + c; then b ~ Binomial(m, p) and
             D = 2b - m. A Clopper-Pearson 90% interval for p maps to an exact 90%
             interval for D. Stdlib only (math.comb + bisection), matching the house style
             of pins/exp79_analyse.py.

  READING IT.
    equivalence PASS -> debate's Exp 83 win is explained by inference budget. Debate is
      sampling, not argument. Combined with Exp 79 (parallel review p=0.50) the multi-agent
      structure line is closed on evidence.
    equivalence FAIL + McNemar significant -> structure buys something samples do not, and
      the Exp 83 result survives its most serious objection.
    equivalence FAIL + McNemar ns -> underpowered at n=31; report as inconclusive and do
      NOT read the point estimate as a result. This outcome was declared possible in advance.

SECONDARY, reported always, never substituted for the primary:
  - exact McNemar (two-sided) on the same pair;
  - the same two tests for debate vs the ORIGINAL single-pkt, which is the Exp 83 comparison
    and must reproduce b=6 c=1 to confirm the harness;
  - bare `handled` beside STRICT;
  - the 9 CONTROLS, separately, never pooled.

  .venv/bin/python -m pins.exp88_analyse pins/results_exp88_qwen2514b_t0.8.json
"""
from __future__ import annotations

import json
import sys
from math import comb


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f, target: float, increasing: bool) -> float:
    """Root of f(p) = target on [0,1] for a monotone f. `increasing` names f's direction --
    the two Clopper-Pearson bounds solve one increasing and one decreasing equation, and
    getting that backwards silently returns the opposite endpoint."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(b: int, m: int, alpha: float) -> tuple[float, float]:
    """Exact one-sided-alpha bounds; (lo, hi) is a 1-2*alpha interval for p."""
    # P(X >= b | p) increases in p;  P(X <= b | p) decreases in p.
    lo = 0.0 if b == 0 else _solve(lambda p: 1 - binom_cdf(b - 1, m, p), alpha, True)
    hi = 1.0 if b == m else _solve(lambda p: binom_cdf(b, m, p), alpha, False)
    return lo, hi


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def tost(b: int, c: int, n: int, margin: float, alpha: float = 0.05):
    """Exact conditional TOST. Returns (D, lo, hi, pass) with the CI in CASES."""
    m = b + c
    if m == 0:
        return 0, 0.0, 0.0, margin > 0
    p_lo, p_hi = clopper_pearson(b, m, alpha)
    d = b - c
    lo, hi = 2 * m * p_lo - m, 2 * m * p_hi - m
    return d, lo, hi, (lo > -margin and hi < margin)


def counts(res: dict, sub: list, arm: str, strict: bool) -> int:
    return sum(1 for c in sub if res[c]["arms"][arm]["handled"]
               and not (strict and res[c]["arms"][arm]["meta"]["rejected"]))


def discordant(res: dict, sub: list, x: str, y: str, strict: bool) -> tuple[int, int]:
    def ok(c, arm):
        a = res[c]["arms"][arm]
        return a["handled"] and not (strict and a["meta"]["rejected"])
    b = sum(1 for c in sub if ok(c, x) and not ok(c, y))
    cc = sum(1 for c in sub if ok(c, y) and not ok(c, x))
    return b, cc


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pins/results_exp88_qwen2514b_t0.8.json"
    with open(path) as f:
        blob = json.load(f)
    res = blob["results"]
    from pins.hardcases_r3 import CONTROLS, PRIMARY
    prim = [c for c in res if c in set(PRIMARY)]
    ctrl = [c for c in res if c in set(CONTROLS)]
    MARGIN = 3.0

    print(f"model={blob['model']}  boN temperature={blob.get('temperature_boN')}  "
          f"PRIMARY n={len(prim)}  CONTROLS n={len(ctrl)}\n")

    for strict in (True, False):
        tag = "STRICT (handled AND feasible)" if strict else "bare handled"
        print(f"=== {tag} — PRIMARY n={len(prim)} ===")
        for arm in blob["arms"]:
            nc = sum(res[c]["arms"][arm]["meta"]["n_calls"] for c in prim)
            print(f"  {arm:15s} {counts(res, prim, arm, strict):>3d}/{len(prim)}   "
                  f"calls {nc:>4d}")

        print(f"\n  PRIMARY TEST — TOST equivalence, margin +/-{MARGIN:.0f} cases, alpha 0.05")
        b, c = discordant(res, prim, "debate-pkt", "single-pkt-boN", strict)
        d, lo, hi, ok = tost(b, c, len(prim), MARGIN)
        p2 = mcnemar_exact_two_sided(b, c)
        print(f"    debate-pkt vs single-pkt-boN : b={b} c={c}  D={d:+d} cases  "
              f"90% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"    equivalence: {'PASS — budget explains debate' if ok else 'FAIL'}"
              f"   (exact McNemar two-sided p={p2:.4f})")
        if not ok:
            verdict = ("structure survives the budget control" if p2 < 0.05 else
                       "INCONCLUSIVE at n=31 — neither equivalent nor significant; "
                       "do NOT read D as a result")
            print(f"    -> {verdict}")

        print("\n  SECONDARY")
        for x, y in (("debate-pkt", "single-pkt"), ("single-pkt-boN", "single-pkt")):
            b2, c2 = discordant(res, prim, x, y, strict)
            print(f"    {x} vs {y}: b={b2} c={c2}  "
                  f"exact McNemar two-sided p={mcnemar_exact_two_sided(b2, c2):.4f}")
        print()

    print(f"=== CONTROLS n={len(ctrl)} (reported separately, never pooled) ===")
    for arm in blob["arms"]:
        print(f"  {arm:15s} handled {counts(res, ctrl, arm, False):>2d}/{len(ctrl)}   "
              f"strict {counts(res, ctrl, arm, True):>2d}/{len(ctrl)}")

    print("\n=== harness reproduction check vs Exp 83 (STRICT, PRIMARY) ===")
    b3, c3 = discordant(res, prim, "debate-pkt", "single-pkt", True)
    print(f"  debate vs single: b={b3} c={c3}   Exp 83 reported b=6 c=1")
    print("  " + ("MATCHES Exp 83" if (b3, c3) == (6, 1) else
                  "DOES NOT MATCH Exp 83 — report this, do not silently accept it"))


if __name__ == "__main__":
    main()
