"""Exp 96 analysis — the tier/laxity de-confound.
Spec: docs/superpowers/specs/2026-07-29-exp96-tier-laxity-deconfound-design.md

The correlated and decorrelated worlds are two tiers in ONE results file, sharing seeds, so
every contrast here is paired within seed and the diff-in-diff is exact (the tier draw runs on
its own RNG stream, so the jobs themselves are byte-identical between worlds).

  H1 (primary)   dprodSLA(arm - floor) in the decorrelated world, and the diff-in-diff
                 [arm - floor]_decorr - [arm - floor]_correlated.
  H2 (decisive)  d tight_sla(arm - floor), every arm, both worlds.
  H3 (check)     tier/laxity correlation (regenerated from the workload maker, not asserted),
                 and tight_sla ~ prod_sla at the floor in the correlated world only.

Two-sided throughout, house rule for significance (95% CI excludes 0, paired by seed), Holm
over each vs-floor family exactly as trace_replay reports its own.

  .venv/bin/python -m pins.exp96_analyse pins/results_exp96_amdahl.json
"""
from __future__ import annotations

import json
import sys

from pins.trace_replay import _paired_p, holm, paired_ci

PRIMARY_ARM = "composed"      # spec §5 H1; the reserve-scalar arm claim 2 is about
FLOOR = "no-llm"


def _fmt(m: float, h: float, unit: float = 100.0) -> str:
    return f"{m*unit:+6.1f} +/-{h*unit:4.1f}{'*' if h < abs(m) else ' '}"


def _diffs(rows: list[dict], floor: list[dict], metric: str) -> list[float]:
    return [a[metric] - b[metric] for a, b in zip(rows, floor)]


def _family(label: str, per_world: dict, metric: str) -> dict:
    """One vs-floor family (arm x world) for `metric`: print each, Holm-correct together."""
    print(f"\n  {label}")
    raw, shown, how_ = {}, {}, {}
    for world, pool in per_world.items():
        floor = pool[FLOOR]
        for arm, rows in pool.items():
            if arm == FLOOR:
                continue
            d = _diffs(rows, floor, metric)
            m, h = paired_ci(d)
            shown[(world, arm)] = (m, h)
            p, how = _paired_p(d)
            raw[(world, arm)] = p
            how_[(world, arm)] = how
    for (world, arm), (m, h) in shown.items():
        # the star is the house rule (95% CI); the p is printed next to it because a coarse-t CI
        # and a Wilcoxon p can disagree, and the disagreement must be visible, not resolved here
        print(f"    {world:12s} {arm:<12s} {_fmt(m, h)}   "
              f"p={raw[(world, arm)]:.3f}[{how_[(world, arm)]}]")
    adj = holm(raw)
    surv = sorted((k for k, v in adj.items() if v < 0.05), key=lambda k: adj[k])
    print(f"    Holm (family of {len(raw)}): "
          + (", ".join(f"{w}/{a} p={adj[(w, a)]:.3f}" for w, a in surv) if surv
             else "NOTHING survives correction"))
    return {k: (shown[k], adj[k]) for k in shown}


def manipulation_check(n_seeds: int, n_jobs: int = 16, horizon: int = 300) -> None:
    """H3. Regenerate both worlds' windows and measure the confound instead of asserting it.

    Also verifies the pre-registration's pairing claim directly: with the tier draw on its own
    stream the two worlds must produce byte-identical arrivals, work and deadlines.
    """
    from pins.trace_replay import (TICK_S, TRACES, load_predicted_quanta, load_trace,
                                   make_trace_workload)
    trace = load_trace(TRACES["v2020"][0])
    pred = load_predicted_quanta()
    print("\n=== H3 manipulation check (regenerated windows) ===")
    ident = {}
    for world, dec in (("correlated", False), ("decorr", True)):
        lax_prod, lax_be, prod_n = [], [], []
        ident[world] = []
        for s in range(n_seeds):
            jobs, *_ = make_trace_workload(trace, n_jobs, s, horizon, pred, tick=TICK_S,
                                           decorrelate=dec)
            ident[world].append([(j.arrival, tuple(j.need), j.deadline) for j in jobs])
            prod_n.append(sum(j.tier == "prod" for j in jobs))
            for j in jobs:
                # the same laxity two_sided_sim.tight_third ranks on
                lax = (j.deadline - j.arrival) / max(sum(j.need), 1e-9)
                (lax_prod if j.tier == "prod" else lax_be).append(lax)
        r = _point_biserial(lax_prod, lax_be)
        print(f"  {world:11s} laxity | prod {sum(lax_prod)/len(lax_prod):.3f}   "
              f"besteffort {sum(lax_be)/len(lax_be):.3f}   point-biserial r={r:+.3f}   "
              f"prod/window {sum(prod_n)/len(prod_n):.2f} (min {min(prod_n)}, max {max(prod_n)})")
    print("  jobs byte-identical across worlds: "
          + ("YES (arrival/work/deadline)" if ident["correlated"] == ident["decorr"]
             else "*** NO — the worlds are NOT paired, H1's diff-in-diff is invalid ***"))


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x) ** 0.5
    syy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sxx * syy) if sxx and syy else 0.0


def _point_biserial(a: list[float], b: list[float]) -> float:
    """Correlation between the prod label and laxity: two-group r, sign = prod is laxer."""
    n1, n2 = len(a), len(b)
    m1, m2 = sum(a) / n1, sum(b) / n2
    allv = a + b
    mu = sum(allv) / (n1 + n2)
    sd = (sum((x - mu) ** 2 for x in allv) / (n1 + n2)) ** 0.5
    return (m1 - m2) / sd * (n1 * n2 / (n1 + n2) ** 2) ** 0.5


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "pins/results_exp96_amdahl.json"
    pool = sys.argv[2] if len(sys.argv) > 2 else "8"
    with open(path) as f:
        tiers = json.load(f)["tiers"]

    base = [t for t in tiers if not t.endswith("+decorr")]
    assert len(base) == 1 and base[0] + "+decorr" in tiers, \
        f"need one correlated tier and its +decorr partner; got {list(tiers)}"
    per_world = {"correlated": tiers[base[0]]["per_seed"][pool],
                 "decorr": tiers[base[0] + "+decorr"]["per_seed"][pool]}
    n = len(per_world["correlated"][FLOOR])
    assert all(len(r) == n for p in per_world.values() for r in p.values()), \
        "seed counts differ between arms/worlds — the pairing is broken"
    print(f"{path}  pool={pool}gpu  tier={base[0]}  n={n} paired seeds")

    print("\n=== floor means (violation rates; lower is better) ===")
    for world, p in per_world.items():
        for arm, rows in p.items():
            g = lambda k: sum(r[k] for r in rows) / n            # noqa: E731
            print(f"  {world:11s} {arm:<12s} SLA {g('sla'):6.1%}  prodSLA {g('prod_sla'):6.1%}  "
                  f"tight {g('tight_sla'):6.1%}")

    h1 = _family("=== H1: dprodSLA vs floor, per world ===", per_world, "prod_sla")
    h2 = _family("=== H2: d tight_sla vs floor, per world ===", per_world, "tight_sla")

    print("\n=== H1 diff-in-diff: [arm - floor]_decorr - [arm - floor]_correlated ===")
    did = {}
    for arm in per_world["decorr"]:
        if arm == FLOOR:
            continue
        d = [x - y for x, y in zip(_diffs(per_world["decorr"][arm], per_world["decorr"][FLOOR],
                                          "prod_sla"),
                                   _diffs(per_world["correlated"][arm],
                                          per_world["correlated"][FLOOR], "prod_sla"))]
        m, h = paired_ci(d)
        did[arm] = (m, h, _paired_p(d)[0])
        print(f"  {arm:<12s} DiD {_fmt(m, h)}  p={did[arm][2]:.3f}   "
              f"(positive = protection SHRANK when tier stopped implying a tight deadline)")

    manipulation_check(n)
    for world in per_world:
        rows = per_world[world][FLOOR]
        m, h = paired_ci([r["tight_sla"] - r["prod_sla"] for r in rows])
        # seed-wise correlation is the real "do these two strata measure the same thing" test;
        # the mean gap can sit near zero for two strata that move independently
        rho = _pearson([r["tight_sla"] for r in rows], [r["prod_sla"] for r in rows])
        print(f"  {world:11s} floor tight_sla - prod_sla {_fmt(m, h)}   seed-wise r={rho:+.3f}")

    print("\n=== DECISION RULE (spec §6) ===")
    (mc, _), _ = h1[("correlated", PRIMARY_ARM)]
    (md, hd), pd_ = h1[("decorr", PRIMARY_ARM)]
    mdid, hdid, _ = did[PRIMARY_ARM]
    if hd >= abs(md):
        br = (f"3 — VANISHES: {PRIMARY_ARM} prodSLA is null decorrelated ({_fmt(md, hd).strip()}). "
              "Claim 2 was tight-deadline protection mislabelled as tier protection; paper §4 "
              "needs rewriting.")
    elif hdid < abs(mdid) and mdid > 0:
        br = (f"2 — SHRINKS but survives: {mc*100:+.1f} -> {md*100:+.1f} pts, DiD "
              f"{mdid*100:+.1f} pts (significant). Claim 2 must be restated as partly "
              "laxity-driven, both worlds reported side by side.")
    else:
        br = (f"1 — UNDIMINISHED: DiD {_fmt(mdid, hdid).strip()} is not distinguishable from 0 "
              f"and decorrelated protection stands ({_fmt(md, hd).strip()}). Claim 2 strengthens: "
              "tier is what is protected, the correlation was incidental.")
    print(f"  H1 -> branch {br}")
    # branch 4 is judged in the DECORRELATED world only: correlated tight_sla is ~the prod
    # stratum by construction (H3), so protection there says nothing about laxity
    prot = [a for (w, a), ((m, h), _) in h2.items() if w == "decorr" and h < abs(m) and m < 0]
    b4 = ("4 — tight_sla flat/unprotected in every arm; nothing in the system serves laxity, "
          "which motivates least-laxity grant ordering next." if not prot
          else "NOT 4 (decorrelated world) — laxity IS protected by: " + ", ".join(prot))
    print(f"  H2 -> branch {b4}")
    (_, pc_) = h1[("correlated", PRIMARY_ARM)]
    print(f"  (H1 primary arm = {PRIMARY_ARM}. Holm-adjusted within the 8-test prodSLA family: "
          f"correlated p={pc_:.3f}, decorr p={pd_:.3f} — the star is the house-rule CI, and if "
          "protection fails correction it fails in BOTH worlds, which is what the DiD reads.")
    print("   Claims 1, 4, 5, 9 unaffected by any branch, per spec §6.)")


if __name__ == "__main__":
    main()
