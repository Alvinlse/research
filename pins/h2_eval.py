"""H2: does restricting the referee to CORRECTING a market allocation beat letting it GENERATE one?

  .venv/bin/python -m pins.h2_eval --model qwen2.5:14b

Three arms on the pre-registered hard-case suite, scored with the suite's own predicates:

  market     — bases guaranteed, the §6 market clears the MARGIN. No LLM. The anchor.
  corrected  — market anchor + the Exp 77 correction layer: reviewers read the free text (only
               where text exists), the referee says who deserves how much, `fund()` moves
               margin (never a base), non-repairing validator, invalid -> the anchor stands.
  scratch    — the Exp 50-73 arm: the referee GENERATES the whole allocation from the same
               submissions. The control H2 is stated against.

H2 predicts `corrected` has fewer harmful changes and fewer invalid decisions than `scratch`.
H1's exception half predicts `corrected` handles more cases than `market` alone — that gap is
the only thing the LLM is being paid for here, so it is reported per category, never averaged
(the suite exists because a mean hides the tail).

The anchor respects the project's allocation model: each job's MINIMUM is granted before any
bidding, and only the margin above it is ever contested or moved.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

from pins.correction import (apply_correction, gather_corrections, referee_delta,
                             validate_delta)
from pins.hardcase_eval import score
from pins.hardcases import CASES
from pins.market import ask_curve, bid_curve
from pins.referee import referee_decide

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_h2.json")
DORDER = {"behind": 0, "ontrack": 1, "ahead": 2}


def build_anchor(case) -> tuple[dict, dict, list, dict]:
    """(floors, allocation, ascending value ranking, env) — bases first, then the market.

    Bases are granted in a fixed, deterministic order (prod before best-effort, then closest to
    deadline) because a scene whose bases do not fit is exactly the INFEASIBLE category: the
    anchor must ration them somehow, and it does so without any LLM. The margin above the
    granted bases is then cleared by the §6 bid/ask rule."""
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    order = sorted(dem, key=lambda s: (s.get("tier") != "prod",
                                       DORDER.get(s.get("deadline"), 1), s.get("job_id")))
    floors, alloc, left = {}, {}, case.free_gpus
    for s in order:                                   # minimums, granted before any bidding
        b = max(0, int(s.get("base_gpus", 0)))
        g = min(b, max(0, left))
        floors[s["job_id"]] = g
        alloc[s["job_id"]] = g
        left -= g
    env = {"total_gpus": case.free_gpus, "incoming_prod": sum(
               1 for s in case.stmts if s.get("side") == "supply"
               and int(s.get("requested_reserve_gpus", 0) or 0) > 0),
           "n_waiting": 0, "n_active": max(len(dem), 1), "law": "amdahl", "alpha": 0.0}
    # margin market over what the bases left
    units = []
    for s in dem:
        jid = s["job_id"]
        facts = {"base": max(1, int(s.get("base_gpus", 0)) or 1),
                 "usable": max(0, int(s.get("requested_margin_gpus", 0))),
                 "waited": 0, "held": alloc.get(jid, 0)}
        for k, v in enumerate(bid_curve(facts, s, env), start=1):
            units.append((v, jid, k))
    units.sort(key=lambda u: (-u[0], u[1], u[2]))
    asks = ask_curve(max(0, left), env)
    sold = 0
    for q, (v, jid, _k) in enumerate(units[:max(0, left)]):
        if v < asks[q]:
            break
        alloc[jid] += 1
        sold += 1
    # who pays for a correction: the market's own ascending value order over AWARDED margin
    ranking = [(v, jid) for v, jid, _k in reversed(units[:sold])]
    return floors, alloc, ranking, env


def corrected_arm(case, model: str, use_llm: bool) -> tuple[dict, int, str, dict]:
    floors, alloc, ranking, _env = build_anchor(case)
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    jobs = [{"jid": s["job_id"], "tier": s.get("tier"), "deadline": s.get("deadline"),
             "requested": int(s.get("base_gpus", 0)) + int(s.get("requested_margin_gpus", 0)),
             "note": s.get("justification", "")} for s in dem]
    sup = next((s.get("justification", "") for s in case.stmts
                if s.get("side") == "supply"), "")
    cache: dict = {}          # per case: the scene key does not hash the exception text
    props = gather_corrections(jobs, sup, case.free_gpus, alloc, use_llm=use_llm,
                               model=model, cache=cache)
    props["ranking"], props["floors"] = ranking, floors
    d = referee_delta(alloc, props, case.free_gpus, use_llm=use_llm, model=model, cache=cache)
    caps = {j["jid"]: j["requested"] for j in jobs}
    # only a CORRECTION can be rejected; an empty delta is not a decision to validate (in the
    # INFEASIBLE cases the anchor itself cannot fit every base, and that is the anchor's
    # property, not the correction layer's)
    bad = (validate_delta(alloc, d["delta"], case.free_gpus, caps=caps, floors=floors)
           if d["delta"] else [])
    final = alloc if bad else apply_correction(alloc, d["delta"])
    meta = {"delta": d["delta"], "l1": sum(abs(v) for v in d["delta"].values()),
            "rejected": bool(bad), "violations": bad,
            "proposals": len(props["demand"]) + (1 if props["supply"] else 0)}
    return final, 0, d.get("justification", ""), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    a = ap.parse_args()
    use_llm = not a.no_llm

    results, agg = {}, collections.defaultdict(lambda: collections.Counter())
    for i, case in enumerate(CASES, 1):
        arms = {}
        _f, anchor, _r, _e = build_anchor(case)
        arms["market"] = score(case, anchor, 0, "market: bases then §6 margin clearing", True)

        final, res, why, meta = corrected_arm(case, a.model, use_llm)
        arms["corrected"] = score(case, final, res, why, True) | {"meta": meta}

        if use_llm:
            out = referee_decide(case.demand(), {}, case.free_gpus, use_llm=True,
                                 model=a.model, cache={}, think=not a.no_think,
                                 stmts=case.stmts)
            arms["scratch"] = score(case, out.alloc, out.reserve, out.justification, True)

        for name, r in arms.items():
            agg[name][case.category + "/handled"] += bool(r["handled"])
            agg[name][case.category + "/n"] += 1
            agg[name]["handled"] += bool(r["handled"])
            agg[name]["overcommitted"] += bool(r["overcommitted"])
            agg[name]["n"] += 1
        agg["corrected"]["l1"] += arms["corrected"]["meta"]["l1"]
        agg["corrected"]["rejected"] += arms["corrected"]["meta"]["rejected"]
        agg["corrected"]["fired"] += bool(arms["corrected"]["meta"]["delta"])
        results[case.id] = {"category": case.category, "arms": arms}
        print(f"\r  {i}/{len(CASES)} {case.id:<12}", end="", flush=True)
    print()

    cats = sorted({c.category for c in CASES})
    names = [n for n in ("market", "corrected", "scratch") if n in agg]
    print(f"\n{'arm':<11}" + "".join(f"{c[:9]:>11}" for c in cats) + f"{'TOTAL':>9}{'over':>7}")
    for n in names:
        row = "".join(f"{agg[n][c+'/handled']:>6}/{agg[n][c+'/n']:<4}" for c in cats)
        print(f"{n:<11}{row}{agg[n]['handled']:>4}/{agg[n]['n']:<4}{agg[n]['overcommitted']:>7}")
    print(f"\ncorrected: fired on {agg['corrected']['fired']}/{agg['corrected']['n']} cases, "
          f"{agg['corrected']['rejected']} corrections rejected by the validator, "
          f"total ||dA||_1 = {agg['corrected']['l1']}")
    with open(OUT, "w") as f:
        json.dump({"model": a.model, "results": results}, f, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
