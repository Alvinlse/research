"""Exp 63 — the admission lever. House-format tables + the pre-registered H1-H4 contrasts.

Reads PINS_RESULTS (default pins/results_exp63.json), which holds two tiers that differ only by
`--admit`. SLA/prodSLA/tight are deadline MISS rates: LOWER IS BETTER, and `*` marks the best cell
in a column, exactly as trace_replay prints them.

Run:  .venv/bin/python -m pins.exp63_analyse
"""
from __future__ import annotations

import json
import os
import statistics as s

PATH = os.environ.get("PINS_RESULTS", os.path.join(os.path.dirname(__file__), "results_exp63.json"))
COLS = [("SLA", "sla", "pct1"), ("prodSLA", "prod_sla", "pct1"), ("tight", "tight_sla", "pct1"),
        ("util", "util", "pct0"), ("useful", "u_useful", "pct0"), ("regret", "regret", "pct0"),
        ("slowdown", "slowdown", "f2"), ("wait", "wait", "f1"), ("fb", "fallback_rate", "pct0"),
        ("done", "finished", "f1")]
# the guard family (spec section 5): an SLA gain bought by not starting jobs is not a win
GUARD = [("finished", "finished"), ("starved", "starved"), ("wait_max", "wait_max"),
         ("wait_full", "wait_full")]


def fmt(v, kind):
    return {"pct1": f"{v:.1%}", "pct0": f"{v:.0%}", "f2": f"{v:.2f}", "f1": f"{v:.1f}"}[kind]


def mean(rows, key):
    return s.mean(r[key] for r in rows)


def paired(a, b, key):
    """a - b, paired by seed. Returns (mean, halfwidth, significant)."""
    d = [x[key] - y[key] for x, y in zip(a, b)]
    m, hw = s.mean(d), 1.96 * s.stdev(d) / len(d) ** 0.5
    return m, hw, abs(m) > hw


def table(pool, pols):
    head = f"{'pool':>4}  {'policy':<14}" + "".join(f"{c[0]:>9}" for c in COLS)
    print(head)
    print("-" * len(head))
    best = {k: min(mean(r, k) for r in pols.values()) for _, k, _ in COLS[:3]}
    for name, rows in pols.items():
        line = f"{pool:>4}  {name:<14}"
        for label, key, kind in COLS:
            if key not in rows[0]:
                line += f"{'-':>9}"
                continue
            v = mean(rows, key)
            star = "*" if key in best and abs(v - best[key]) < 1e-12 else " "
            line += f"{fmt(v, kind):>8}{star}"
        print(line)


def contrast(title, a, b, keys):
    parts = []
    for label, key in keys:
        if key not in a[0]:
            continue
        m, hw, sig = paired(a, b, key)
        scale = 100 if key in ("sla", "prod_sla", "tight_sla", "util", "regret") else 1
        parts.append(f"d{label} {m*scale:+.1f} +- {hw*scale:.1f}{'*' if sig else ''}")
    print(f"{title:<34} " + "  ".join(parts))


def main():
    tiers = json.load(open(PATH))["tiers"]
    base = {k: v for k, v in tiers.items() if "+admit" not in k}
    admit = {k: v for k, v in tiers.items() if "+admit" in k}
    print(f"SLA/prodSLA/tight are MISS rates - lower is better, '*' = best in column\n")

    for label, group in (("BASELINE", base), ("+ADMIT", admit)):
        for name, t in group.items():
            print(f"=== {label}: {name}  (n={t['n_seeds']}) ===")
            for pool, pols in t["per_seed"].items():
                table(pool, pols)
                print()
                for arm in pols:
                    if arm != "no-llm":
                        contrast(f"  {arm} vs floor:", pols[arm], pols["no-llm"],
                                 [("SLA", "sla"), ("prodSLA", "prod_sla"), ("util", "util")])
                print()

    if not (base and admit):
        print("!! only one tier present - H1-H4 need both; re-run the missing arm")
        return

    # pair the tiers that differ ONLY by the flag: dropping '+admit' must give the baseline key.
    # Guards against the stale pre-rebase baseline tier also sitting in this file.
    akey = next(iter(admit))
    bkey = akey.replace("+admit", "")
    if bkey not in base:
        print(f"!! no baseline tier '{bkey}' to pair with '{akey}'; "
              f"present: {sorted(base)} - re-run the baseline under current defaults")
        return
    b, a = base[bkey]["per_seed"], admit[akey]["per_seed"]
    for pool in sorted(set(b) & set(a)):
        bp, ap = b[pool], a[pool]
        if bp["no-llm"] != ap["no-llm"]:
            print(f"!! pool {pool}: floor rows DIFFER - the tiers are not the same world, "
                  f"H1-H4 are invalid (spec section 3 tripwire)")
            continue
        print(f"=== pool {pool}: pre-registered contrasts (floor byte-identity OK) ===")
        contrast("H1 referee+admit - referee:", ap["referee"], bp["referee"],
                 [("SLA", "sla"), ("prodSLA", "prod_sla")])
        contrast("H2 negotiated+admit - negotiated:", ap["negotiated"], bp["negotiated"],
                 [("SLA", "sla"), ("prodSLA", "prod_sla")])
        contrast("H3 referee+admit - negotiated+admit:", ap["referee"], ap["negotiated"],
                 [("SLA", "sla"), ("prodSLA", "prod_sla")])
        print("H4 manipulation check (did the lever engage?) + guard family:")
        contrast("   referee   +admit - base:", ap["referee"], bp["referee"], GUARD)
        contrast("   negotiated+admit - base:", ap["negotiated"], bp["negotiated"], GUARD)


if __name__ == "__main__":
    main()
