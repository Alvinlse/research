"""Exp 82 — referee vs single LLM, both reading the same structured decision packet.

Lineage. Exp 79: the LLM emitted allocations, so everything below 14b failed on feasibility
arithmetic; interaction 0 in every model that could allocate (pooled p=0.50). Exp 80: the
arithmetic-free contract could only ADD, and 22/31 primary cases need restraint, so both arms
fired on 1/40. Exp 81: signed contract, firing 1/40 -> 31/40, and the exception layer finally
worked (market 0/31 -> single 10/31 at 14b, zero harm) — but the referee lost, 5/31 to 10/31,
with 52-62% of its decisions rejected as "cannot fund" because no reviewer sees the budget.

Exp 82 answers the user's revision: the referee's context was too thin. The packet
(pins/packet.py) gives it state, market baseline, the raw claims, hard constraints, history and
a CODE-GENERATED menu of legal actions, so "cannot fund" is unreachable by construction.

The comparison is deliberately stacked AGAINST the hypothesis being tested:

  - both arms receive the IDENTICAL packet, raw notes included;
  - the referee additionally receives `reviewer_proposals`, making its information a strict
    SUPERSET of the single LLM's. In Exp 81 the referee saw only the reviewers' compressed
    summaries and never the note itself, which confounded "does the split help?" with "does the
    split lose information?". That confound is now removed in the referee's favour.

If a strictly better-informed referee still does not beat one call, the multi-agent line is
closed on evidence rather than on an interface artifact.

CONFIDENCE IS COLLECTED, NOT ENFORCED. The packet states 0.70 as guidance; the harness reports
whether confidence >= 0.70 actually predicts a correct override. Gating on a self-reported
number would let an uncalibrated scalar decide the experiment.

  .venv/bin/python -m pins.exp82_packet_2x2 --model qwen2.5:14b [--max-delta 6]
"""
from __future__ import annotations

import argparse
import json
import os

from pins.correction_signed import (_ask, apply_signed, debate_signed, disruption,
                                    gather_signed)
from pins.h2_eval import build_anchor
from pins.hardcase_eval import score
from pins.packet import (SYSTEM_PACKET_REFEREE, SYSTEM_PACKET_SINGLE, build_packet,
                         decision_from_ids)

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["market", "single-pkt", "referee-pkt"]
DEBATE_ARM = "debate-pkt"


def _decide(system: str, packet: dict, model: str, cache: dict, tag: str) -> dict:
    obj = _ask(system, json.dumps(packet, indent=1), model, None, cache, tag) or {}
    d = decision_from_ids(packet, obj.get("action_ids"))
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    d["confidence"] = max(0.0, min(1.0, conf))
    d["justification"] = str(obj.get("justification", ""))[:300]
    return d


def load_history(which: str) -> list:
    """The Exp 51 self-authored precedent manual, as packet.history.

    DOMAIN MISMATCH, stated because it decides what a null means: every WHEN clause keys on
    `incoming_prod`, a variable the hard-case world does not have, and every rule is about
    sizing a reserve for an incoming prod wave. None concerns suspensions, caps, corrupt
    declarations or dependencies -- the things the round-3 cases are about. So a null result
    here is evidence about THIS manual in THIS domain, not about precedent in general, and a
    NEGATIVE result would mean irrelevant precedent actively distracts.
    """
    if which != "manual":
        return []
    with open(os.path.join(HERE, "manual_learned.json")) as f:
        return json.load(f)


def run_case(case, model: str, use_llm: bool, arm: str, max_delta: int, history=None,
             no_text: bool = False):
    floors, alloc, ranking, _env = build_anchor(case)
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    # `no_text` is the pre-registered ablation (pins/hardcases_r3.py): identical numeric packet,
    # every note blanked. Blanking here covers the reviewers AND the packet in one place, since
    # both read `jobs`/`sup`.
    jobs = [{"jid": s["job_id"], "tier": s.get("tier"), "deadline": s.get("deadline"),
             "base": int(s.get("base_gpus", 0)),
             "margin": int(s.get("requested_margin_gpus", 0)),
             "requested": int(s.get("base_gpus", 0)) + int(s.get("requested_margin_gpus", 0)),
             "note": "" if no_text else s.get("justification", "")} for s in dem]
    sup = "" if no_text else next(
        (s.get("justification", "") for s in case.stmts if s.get("side") == "supply"), "")
    cache: dict = {}

    if arm == "market":
        return alloc, "market: bases then §6 margin clearing", {
            "changes": {}, "hold_free": 0, "moved": 0, "rejected": False, "fired": False,
            "confidence": None, "n_actions": 0, "unknown_ids": []}

    props = None
    if arm in ("referee-pkt", "debate-pkt"):
        p = gather_signed(jobs, sup, case.free_gpus, alloc, use_llm=use_llm,
                          model=model, cache=cache)
        if arm == "debate-pkt":
            # one rebuttal round: each reviewer reads the others and may revise its own corner
            p = debate_signed(jobs, sup, case.free_gpus, alloc, p, use_llm=use_llm,
                              model=model, cache=cache)
        props = {k: v for k, v in p.items() if k in ("demand", "supply", "opening", "debated")}

    packet = build_packet(jobs, sup, case.free_gpus, alloc, ranking, floors,
                          max_delta=max_delta, reviewer_proposals=props, history=history)
    if not use_llm:
        d = {"changes": {}, "hold_free": 0, "picked": [], "unknown_ids": [],
             "confidence": 0.0, "justification": "rule: market stands"}
    else:
        d = _decide(SYSTEM_PACKET_SINGLE if arm == "single-pkt" else SYSTEM_PACKET_REFEREE,
                    packet, model, cache, tag=arm)

    fired = bool(d["changes"]) or bool(d["hold_free"])
    final, viol = apply_signed(alloc, d, case.free_gpus, ranking=ranking, floors=floors,
                               budget=max_delta)
    meta = {"changes": d["changes"], "hold_free": d["hold_free"],
            "moved": disruption(alloc, final), "rejected": bool(viol), "violations": viol,
            "fired": fired, "confidence": d["confidence"], "n_actions": len(d.get("picked", [])),
            "unknown_ids": d.get("unknown_ids", []), "menu": len(packet["candidate_actions"])}
    return final, d["justification"], meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--suite", default="r3", choices=["r3", "r12", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=6,
                    help="GPUs movable per decision; 1 reproduces the proposed tight policy")
    ap.add_argument("--debate", action="store_true",
                    help="add the debate-pkt arm: one rebuttal round before the referee rules")
    ap.add_argument("--history", default="none", choices=["none", "manual"],
                    help="manual = load pins/manual_learned.json into packet.history")
    ap.add_argument("--no-text", action="store_true",
                    help="pre-registered ablation: blank every job/supply note, same numbers. "
                         "Round-3 PRIMARY answers live in the note, so this is the floor arm")
    a = ap.parse_args()

    from pins.hardcases import CASES
    from pins.hardcases_r3 import CASES_R3, CONTROLS, PRIMARY
    cases = {"r3": CASES_R3, "r12": CASES, "all": CASES + CASES_R3}[a.suite]
    if a.limit:
        cases = cases[:a.limit]

    hist = load_history(a.history)
    arms_run = ARMS + ([DEBATE_ARM] if a.debate else [])

    out_path = os.environ.get(
        "PINS_RESULTS",
        os.path.join(HERE, f"results_exp82_{a.model.replace(':', '')}_d{a.max_delta}.json"))
    results: dict[str, dict] = {}
    for case in cases:
        arms = {}
        for arm in arms_run:
            final, why, meta = run_case(case, a.model, not a.no_llm and arm != "market",
                                        arm, a.max_delta, history=hist, no_text=a.no_text)
            arms[arm] = score(case, final, 0, why, True) | {"meta": meta}
        results[case.id] = {"category": case.category, "arms": arms}
        print(f"{case.id:14s} {case.category:12s} " + " ".join(
            f"{k}={'H' if v['handled'] else '.'}"
            f"{'!' if v['meta']['rejected'] else ('*' if v['meta']['fired'] else '')}"
            for k, v in arms.items()))

    json.dump({"model": a.model, "suite": a.suite, "arms": arms_run, "max_delta": a.max_delta,
               "history": a.history, "n_precedents": len(hist), "no_text": a.no_text,
               "results": results}, open(out_path, "w"), indent=1)

    ids = [c.id for c in cases]
    for label, sub in (("PRIMARY", [c for c in ids if c in set(PRIMARY)]),
                       ("CONTROLS", [c for c in ids if c in set(CONTROLS)]),
                       ("ALL", ids)):
        if not sub:
            continue
        mk = {c: results[c]["arms"]["market"]["handled"] for c in sub}
        print(f"\n=== {label}  n={len(sub)} ===")
        print(f"  {'arm':13s}{'handled':>9s}{'fired':>7s}{'invalid':>9s}"
              f"{'rescued':>9s}{'broke':>7s}{'net':>6s}{'badid':>7s}")
        for arm in arms_run:
            h = sum(1 for c in sub if results[c]["arms"][arm]["handled"])
            f = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["fired"])
            r = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["rejected"])
            resc = sum(1 for c in sub if not mk[c] and results[c]["arms"][arm]["handled"])
            brk = sum(1 for c in sub if mk[c] and not results[c]["arms"][arm]["handled"])
            bad = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["unknown_ids"])
            print(f"  {arm:13s}{h:>6d}/{len(sub):<3d}{f:>7d}{r:>9d}"
                  f"{resc:>9d}{brk:>7d}{resc - brk:>+6d}{bad:>7d}")
        def h2h(x, y):
            b = sum(1 for c in sub if results[c]["arms"][x]["handled"]
                    and not results[c]["arms"][y]["handled"])
            cc = sum(1 for c in sub if results[c]["arms"][y]["handled"]
                     and not results[c]["arms"][x]["handled"])
            print(f"  head-to-head {x} vs {y}: {x}-only {b}, {y}-only {cc}")
        h2h("referee-pkt", "single-pkt")
        if DEBATE_ARM in arms_run:
            h2h(DEBATE_ARM, "referee-pkt")      # the increment debate adds over parallel review
            h2h(DEBATE_ARM, "single-pkt")

    # is the self-reported confidence worth gating on? (collected, never enforced)
    print("\n=== confidence calibration (PRIMARY): does conf >= 0.70 predict a correct override? ===")
    prim = [c for c in ids if c in set(PRIMARY)]
    for arm in arms_run[1:]:
        rows = [(results[c]["arms"][arm]["meta"]["confidence"],
                 results[c]["arms"][arm]["handled"]) for c in prim
                if results[c]["arms"][arm]["meta"]["fired"]]
        hi = [ok for cf, ok in rows if cf >= 0.70]
        lo = [ok for cf, ok in rows if cf < 0.70]
        f = lambda xs: f"{sum(xs)}/{len(xs)}" if xs else "n/a"
        print(f"  {arm:13s} fired {len(rows):2d}   conf>=0.70 correct {f(hi):>7s}   "
              f"conf<0.70 correct {f(lo):>7s}")

    print(f"\nmax_delta={a.max_delta}  history={a.history}({len(hist)})   full detail -> {out_path}")


if __name__ == "__main__":
    main()
