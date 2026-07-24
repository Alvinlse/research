"""Run the hard-case suite: LLM referee vs ILP vs the rule floor, scored per case.

  .venv/bin/python -m pins.hardcase_eval --model qwen2.5:14b --no-think

Scoring is per case, never averaged across categories — the thesis is about the tail, so a mean
would hide exactly the effect under test (see pins/hardcases.py). Three axes:

  resolved  — produced any allocation at all (an ILP INFEASIBLE does not)
  feasible  — passes the existing evaluator (referee.check_allocation): no over-award, no
              hallucinated job. Cheating by awarding capacity that does not exist fails here.
  handled   — satisfies the case's PRE-REGISTERED predicate: did it do the defensible thing?

`handled` is the headline. Justifications are captured verbatim for the qualitative table, and
`cited` reports whether the referee's own words reference the fact that should have driven it —
the same faithfulness axis as the Exp 50 citation study.

The ILP arm's objective is fixed here, written once, and is NOT tuned per case: that would be
the strawman the suite exists to avoid. It is the ranking a competent modeller writes in
advance — tier first, then deadline pressure, bases above margins, reserve last.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from pins.hardcases import CASES, CATEGORIES, HardCase
from pins.ilp import allocate
from pins.referee import _rule_referee, check_allocation, referee_decide

HERE = os.path.dirname(os.path.abspath(__file__))
# honour PINS_RESULTS like h2_eval/trace_replay: a --no-llm smoke run must not be able to
# overwrite a completed LLM run's per-case detail (this file is rewritten on every run)
OUT = os.environ.get("PINS_RESULTS", os.path.join(HERE, "results_hardcases.json"))

# PRE-REGISTERED static objective for the ILP arm. Value per GPU, highest first.
TIER_VALUE = {("prod", "behind"): 4.0, ("prod", "ontrack"): 3.5, ("prod", "ahead"): 3.0,
              ("besteffort", "behind"): 2.0, ("besteffort", "ontrack"): 1.5,
              ("besteffort", "ahead"): 1.0}
MARGIN_DISCOUNT = 0.6      # a hedge GPU is worth less than a base GPU
RESERVE_VALUE = 1.2        # holding capacity for unarrived prod, between the best-effort ranks


def ilp_arm(case: HardCase) -> dict:
    """One LLM-free MILP over the same submissions, using the fixed objective above."""
    bids: dict[str, list[float]] = {}
    for s in case.stmts:
        if s["side"] != "demand":
            continue
        v = TIER_VALUE.get((s["tier"], s["deadline"]), 1.0)
        base = max(0, int(s["base_gpus"]))            # a negative declaration buys nothing
        margin = max(0, int(s["requested_margin_gpus"]))
        # duplicate ids: later submission extends the same bid vector, it does not double-count
        bids.setdefault(s["job_id"], [])
        bids[s["job_id"]] += [v] * base + [v * MARGIN_DISCOUNT] * margin
    reserve_req = sum(max(0, int(s.get("requested_reserve_gpus", 0)))
                      for s in case.stmts if s["side"] == "supply")
    if reserve_req:
        bids["_reserve"] = [RESERVE_VALUE] * reserve_req
    bids = {k: v for k, v in bids.items() if v}
    if not bids:
        return {"alloc": {}, "reserve": 0, "justification": "no bids", "resolved": True}
    try:
        r = allocate(bids, case.free_gpus)
        alloc = {k: v for k, v in r.allocation.items() if k != "_reserve"}
        return {"alloc": alloc, "reserve": r.allocation.get("_reserve", 0),
                "justification": "MILP: maximise static tier/deadline value under capacity",
                "resolved": True}
    except Exception as e:                       # infeasible / solver failure = did not resolve
        return {"alloc": {}, "reserve": 0, "resolved": False,
                "justification": f"{type(e).__name__}: {e}"}


def score(case: HardCase, alloc: dict, reserve: int, why: str, resolved: bool) -> dict:
    demand = case.demand()
    violations = check_allocation(alloc, reserve, demand, case.free_gpus)
    # rule 1 (over-award / hallucinated job) is a HARD error — capacity that does not exist.
    # Rules 2-4 encode the normal-case policy (base before margin, prod first, envy-freeness);
    # in a triage scene, breaking one is frequently the correct act, so they are reported as
    # soft flags and must not count against an arm here.
    hard = [v for v in violations if v.startswith("infeasible:") or v.startswith("hallucinated")]
    handled = False
    if resolved:
        try:
            handled = bool(case.predicate(alloc, reserve))
        except Exception:
            handled = False
    cited = (all(c.lower() in why.lower() for c in case.must_cite)
             if case.must_cite else None)
    return {"resolved": resolved, "overcommitted": bool(hard), "handled": handled,
            "cited": cited, "alloc": alloc, "reserve": reserve,
            "hard_violations": hard, "policy_flags": [v for v in violations if v not in hard],
            "justification": why}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="rule referee only (smoke test)")
    ap.add_argument("--suite", default="r12", choices=["r12", "r3", "r4", "r34", "all"],
                    help="r12 = rounds 1-2 (Exp 54/66), r3 = round-3, r4 = round-4 (Exp 89), "
                         "r34 = pooled r3+r4")
    ap.add_argument("--arms", default="referee",
                    help="comma list from single,single-noarg,referee,referee-noarg. "
                         "The 2x2 perspective x text design is all four.")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke runs)")
    a = ap.parse_args()

    llm_arms = [x for x in a.arms.split(",") if x]
    bad = set(llm_arms) - {"single", "single-noarg", "referee", "referee-noarg"}
    if bad:
        ap.error(f"unknown arm(s): {sorted(bad)}")

    cases, categories = CASES, CATEGORIES
    if a.suite != "r12":
        from pins.hardcases_r3 import CASES_R3, CATEGORIES_R3
        categories = CATEGORIES_R3
        if a.suite == "r3":
            cases = CASES_R3
        elif a.suite in ("r4", "r34"):
            from pins.hardcases_r4 import CASES_R4
            cases = CASES_R4 if a.suite == "r4" else CASES_R3 + CASES_R4
        else:  # all
            cases = CASES + CASES_R3
    if a.limit:
        cases = cases[:a.limit]

    results: dict[str, dict] = {}
    for case in cases:
        arms: dict[str, dict] = {}

        r = ilp_arm(case)
        arms["ilp"] = score(case, r["alloc"], r["reserve"], r["justification"], r["resolved"])

        f = _rule_referee(case.stmts, case.free_gpus)
        arms["rule"] = score(case, f["alloc"], f["reserve"], f.get("justification", ""), True)

        if not a.no_llm:
            for name in llm_arms:
                # fresh cache per arm AND per case: the scene key does not hash the
                # justification text, so a shared cache would collide two cases that differ
                # only in their exception, and `noarg` blanks that text to the empty string —
                # which would make every no-text case in a suite share one entry.
                out = referee_decide(case.demand(), {}, case.free_gpus, use_llm=True,
                                     model=a.model, cache={}, think=not a.no_think,
                                     stmts=case.stmts,
                                     perspective=not name.startswith("single"),
                                     no_argue=name.endswith("-noarg"))
                arms[name] = score(case, out.alloc, out.reserve, out.justification, True)

        results[case.id] = {"category": case.category, "expect": case.expect, "arms": arms}
        marks = " ".join(f"{k}={'H' if v['handled'] else ('f' if v['resolved'] else 'X')}"
                         for k, v in arms.items())
        print(f"{case.id:12s} {case.category:14s} {marks}")

    json.dump({"model": a.model, "suite": a.suite, "arms": llm_arms, "results": results},
              open(OUT, "w"), indent=1)

    arm_names = list(next(iter(results.values()))["arms"])
    print(f"\n{'category':16s}" + "".join(f"{n:>12s}" for n in arm_names))
    for cat in categories:
        rows = [r for r in results.values() if r["category"] == cat]
        line = f"{cat:16s}"
        for n in arm_names:
            k = sum(1 for r in rows if r["arms"][n]["handled"])
            line += f"{k:>7d}/{len(rows):<4d}"
        print(line)
    print(f"{'TOTAL':16s}" + "".join(
        f"{sum(1 for r in results.values() if r['arms'][n]['handled']):>7d}/{len(results):<4d}"
        for n in arm_names))

    for n in arm_names:
        over = sum(1 for r in results.values() if r["arms"][n]["overcommitted"])
        unres = sum(1 for r in results.values() if not r["arms"][n]["resolved"])
        soft = sum(1 for r in results.values() if r["arms"][n]["policy_flags"])
        print(f"  {n:10s} over-awarded {over:2d}   unresolved {unres:2d}   policy-flagged {soft:2d}")
    for n in llm_arms:
        cites = [r["arms"][n]["cited"] for r in results.values()
                 if r["arms"][n]["cited"] is not None]
        if cites:
            print(f"  {n:14s} cited the driving fact in {sum(cites)}/{len(cites)} "
                  f"cases that name one")

    # The pre-registered PRIMARY test (pins/hardcases_r3.py): does the text buy the referee
    # more than it buys a single LLM? Reported as the 2x2 plus the per-case discordant counts
    # McNemar needs. Printed only when the full design was run; never averaged with controls.
    if {"single", "single-noarg", "referee", "referee-noarg"} <= set(llm_arms):
        from pins.hardcases_r3 import CONTROLS, PRIMARY
        for label, ids in (("PRIMARY (pre-registered)", PRIMARY),
                           ("CONTROLS (text effect must be ~0)", CONTROLS),
                           ("all cases run", list(results))):
            rows = [r for cid, r in results.items() if cid in set(ids)]
            if not rows:
                continue
            h = {n: sum(1 for r in rows if r["arms"][n]["handled"]) for n in llm_arms}
            eff_s = h["single"] - h["single-noarg"]
            eff_r = h["referee"] - h["referee-noarg"]
            # difference-in-differences, per case: +1 = text helped the referee only
            b = sum(1 for r in rows
                    if (r["arms"]["referee"]["handled"] - r["arms"]["referee-noarg"]["handled"])
                    > (r["arms"]["single"]["handled"] - r["arms"]["single-noarg"]["handled"]))
            c = sum(1 for r in rows
                    if (r["arms"]["referee"]["handled"] - r["arms"]["referee-noarg"]["handled"])
                    < (r["arms"]["single"]["handled"] - r["arms"]["single-noarg"]["handled"]))
            print(f"\n{label}  n={len(rows)}")
            print(f"  single   {h['single']:2d} with text vs {h['single-noarg']:2d} without"
                  f"   -> text effect {eff_s:+d}")
            print(f"  referee  {h['referee']:2d} with text vs {h['referee-noarg']:2d} without"
                  f"   -> text effect {eff_r:+d}")
            print(f"  interaction {eff_r - eff_s:+d}   discordant pairs "
                  f"referee-favoured={b} single-favoured={c}")
    print(f"\nfull transcripts -> {OUT}")


if __name__ == "__main__":
    main()
