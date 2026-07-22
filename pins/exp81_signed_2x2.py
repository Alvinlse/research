"""Exp 81 — referee vs single LLM in the SIGNED correction interface.

The question Exp 79 and Exp 80 both failed to answer, for two different reasons:

  Exp 79  the LLM emitted the allocation itself, so every model below 14b failed on feasibility
          arithmetic rather than judgement (63-81% infeasible). Ladder verdict on the 3 models
          that cleared the competence bar: interaction 0, pooled b=3 c=2, p=0.50.
  Exp 80  the arithmetic-free correction interface removed that confound but could only ADD
          GPUs, and 22 of 31 primary cases require restraint. Both arms fired on 1/40 and the
          one delta was rejected. Inconclusive: the channel could not carry the answer.

Exp 81 keeps the arithmetic-free contract and widens it to signed changes plus an explicit
hold_free (pins/correction_signed.py). Now BOTH halves of the action space are expressible, and
neither arm can violate feasibility. If the split still shows nothing here, the multi-agent line
is closed on evidence across three interfaces rather than on one model in one interface.

ARMS — identical anchor, identical contract, identical funding/validation. The only difference
is who produces the decision:

  market        deterministic anchor, no LLM. Also the text-blind arm BY CONSTRUCTION: with no
                notes, no reviewer fires and no correction is possible.
  single-sgn    ONE call over the raw notes.
  referee-sgn   per-job demand reviewer (signed) + supply reviewer (hold_free), then a referee
                ruling on their proposals.

REPORTED, with "useful" and "valid" kept apart, and rescued paired with broke so a wash cannot
be read as a win:  handled / fired / invalid / rescued / broke / net, plus the head-to-head.

  .venv/bin/python -m pins.exp81_signed_2x2 --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import os

from pins.correction_signed import (SIGNED_BUDGET, apply_signed, disruption, gather_signed,
                                    referee_signed, single_signed)
from pins.h2_eval import build_anchor
from pins.hardcase_eval import score

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["market", "single-sgn", "referee-sgn"]


def run_case(case, model: str, use_llm: bool, arm: str, budget: int):
    floors, alloc, ranking, _env = build_anchor(case)
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    jobs = [{"jid": s["job_id"], "tier": s.get("tier"), "deadline": s.get("deadline"),
             "requested": int(s.get("base_gpus", 0)) + int(s.get("requested_margin_gpus", 0)),
             "note": s.get("justification", "")} for s in dem]
    sup = next((s.get("justification", "") for s in case.stmts if s.get("side") == "supply"), "")
    cache: dict = {}          # per case: the scene key does not hash the exception text

    if arm == "market":
        return alloc, "market: bases then §6 margin clearing", {
            "changes": {}, "hold_free": 0, "moved": 0, "rejected": False, "fired": False}

    if arm == "referee-sgn":
        props = gather_signed(jobs, sup, case.free_gpus, alloc, use_llm=use_llm,
                              model=model, cache=cache)
        d = referee_signed(alloc, props, case.free_gpus, jobs, use_llm=use_llm,
                           model=model, cache=cache)
        nprops = len(props["demand"]) + (1 if props["supply"] else 0)
    else:
        d = single_signed(alloc, jobs, sup, case.free_gpus, use_llm=use_llm,
                          model=model, cache=cache)
        nprops = None

    fired = bool(d.get("changes")) or bool(d.get("hold_free"))
    final, viol = apply_signed(alloc, d, case.free_gpus, ranking=ranking, floors=floors,
                               budget=budget)
    meta = {"changes": d.get("changes", {}), "hold_free": d.get("hold_free", 0),
            "moved": disruption(alloc, final),
            "rejected": bool(viol), "violations": viol, "fired": fired, "proposals": nprops}
    return final, d.get("justification", ""), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--suite", default="r3", choices=["r3", "r12", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget", type=int, default=SIGNED_BUDGET)
    a = ap.parse_args()

    from pins.hardcases import CASES
    from pins.hardcases_r3 import CASES_R3, CONTROLS, PRIMARY
    cases = {"r3": CASES_R3, "r12": CASES, "all": CASES + CASES_R3}[a.suite]
    if a.limit:
        cases = cases[:a.limit]

    out_path = os.environ.get(
        "PINS_RESULTS", os.path.join(HERE, f"results_exp81_{a.model.replace(':', '')}.json"))
    results: dict[str, dict] = {}
    for case in cases:
        arms = {}
        for arm in ARMS:
            final, why, meta = run_case(case, a.model, not a.no_llm and arm != "market",
                                        arm, a.budget)
            arms[arm] = score(case, final, 0, why, True) | {"meta": meta}
        results[case.id] = {"category": case.category, "arms": arms}
        print(f"{case.id:14s} {case.category:12s} " + " ".join(
            f"{k}={'H' if v['handled'] else '.'}"
            f"{'!' if v['meta']['rejected'] else ('*' if v['meta']['fired'] else '')}"
            for k, v in arms.items()))

    json.dump({"model": a.model, "suite": a.suite, "arms": ARMS, "budget": a.budget,
               "results": results}, open(out_path, "w"), indent=1)

    ids = [c.id for c in cases]
    for label, sub in (("PRIMARY", [c for c in ids if c in set(PRIMARY)]),
                       ("CONTROLS", [c for c in ids if c in set(CONTROLS)]),
                       ("ALL", ids)):
        if not sub:
            continue
        mk = {c: results[c]["arms"]["market"]["handled"] for c in sub}
        print(f"\n=== {label}  n={len(sub)} ===")
        print(f"  {'arm':14s}{'handled':>9s}{'fired':>7s}{'invalid':>9s}"
              f"{'rescued':>9s}{'broke':>7s}{'net':>6s}")
        for arm in ARMS:
            h = sum(1 for c in sub if results[c]["arms"][arm]["handled"])
            f = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["fired"])
            r = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["rejected"])
            resc = sum(1 for c in sub if not mk[c] and results[c]["arms"][arm]["handled"])
            brk = sum(1 for c in sub if mk[c] and not results[c]["arms"][arm]["handled"])
            print(f"  {arm:14s}{h:>6d}/{len(sub):<3d}{f:>7d}{r:>9d}"
                  f"{resc:>9d}{brk:>7d}{resc - brk:>+6d}")
        b = sum(1 for c in sub if results[c]["arms"]["referee-sgn"]["handled"]
                and not results[c]["arms"]["single-sgn"]["handled"])
        cc = sum(1 for c in sub if results[c]["arms"]["single-sgn"]["handled"]
                 and not results[c]["arms"]["referee-sgn"]["handled"])
        print(f"  head-to-head referee vs single: referee-only {b}, single-only {cc}")

    print(f"\nbudget={a.budget}   full detail -> {out_path}")


if __name__ == "__main__":
    main()
