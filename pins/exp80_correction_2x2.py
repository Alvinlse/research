"""Exp 80 — perspective test in the ARITHMETIC-FREE (correction) interface.

WHY. Exp 79 ran the perspective x text 2x2 with the LLM emitting the allocation directly, and
every model below 14b failed it for the wrong reason: 1.5b/2b/3b allocated 130-150% of the pool
and broke feasibility on 63-81% of cases. Splitting the primary cases by whether awarding every
job its full ask already satisfies the predicate showed their apparent competence was an
artifact — 9/9 on the grant-satisfiable cases (free for an over-allocator), 3-4/22 on the cases
that require restraint, against 14b's 10/22. So Exp 79's ladder cannot answer whether structure
helps: below 14b it measures arithmetic, not judgement.

That failure is the project's own design rule being violated. "The LLM reasons; deterministic
code decides" — yet the referee arm asks the LLM to emit the numbers. Exp 77 already fixed this
one level up (pins/correction.py): the market allocates, agents may only PROPOSE a correction,
the referee says only WHICH job deserves HOW MANY extra on the evidence, `fund()` moves the
GPUs, and an invalid delta is rejected so the market's allocation stands. Feasibility becomes
impossible to violate rather than something a 3b model must compute.

THE QUESTION, in the user's words: can the referee generate a useful / valid suggestion, and
does it beat a single LLM doing the same job?

ARMS (all share the same anchor, the same output contract and the same funding/validation code;
the ONLY difference is whether the proposal is produced by a split or by one call):

  market        deterministic anchor, no LLM. Also the text-blind arm: with notes blanked no
                reviewer fires and no correction is possible, so text-blind == market BY
                CONSTRUCTION. It is asserted, not spent on LLM calls.
  single-corr   ONE call. Sees the anchor, every job's note and the supply note, and emits the
                same {"accept": {...}} delta. No reviewer/referee split.
  referee-corr  the Exp 77 pipeline: per-job demand reviewer + supply reviewer extract evidence,
                then the referee rules on their proposals.

MEASURES — "useful" and "valid" are separate axes and are reported separately:

  fired      did the arm propose any change at all?  (Exp 77 on rounds 1-2: 2/54, the finding
             that left H1's exception half unsupported)
  rejected   did the proposed delta fail validate_delta? (invalid => market stands)
  handled    does the FINAL allocation satisfy the case's pre-registered predicate?
  rescued    market failed the case, the arm's correction fixed it        <- usefulness
  broke      market handled the case, the arm's correction destroyed it   <- harm

`rescued` and `broke` are the honest pair: an arm that rescues 5 and breaks 5 has done nothing,
and reporting only rescues would repeat the selection error we already caught once.

  .venv/bin/python -m pins.exp80_correction_2x2 --model qwen2.5:14b --no-think
"""
from __future__ import annotations

import argparse
import json
import os

from pins.correction import (HOST, _ask, apply_correction,
                            fund, gather_corrections, referee_delta, validate_delta)
from pins.h2_eval import build_anchor
from pins.hardcase_eval import score

HERE = os.path.dirname(os.path.abspath(__file__))

# One LLM, same contract as SYSTEM_REFEREE_DELTA, same four rules verbatim — the only edit is
# the framing: it reads the notes itself instead of ruling on two reviewers' proposals.
SYSTEM_SINGLE_DELTA = (
    "You are the SCHEDULER. A market has produced an allocation from the jobs' numerical bids. "
    "You are given every job's free-text note. You do NOT produce an allocation. You decide "
    "only whether any job deserves more than the market gave it, and how much.\n"
    "Rules, in order:\n"
    "1. THE MARKET IS THE DEFAULT. Override it only where a note carries a fact the bid "
    "could not contain. No evidence, no change.\n"
    "2. EVERY CHANGE COSTS. A job whose allocation moves pays a reconfiguration penalty, so a "
    "change must be worth more than the disruption. Prefer the smallest correction that "
    "addresses the evidence.\n"
    "3. YOU DO NOT DO ARITHMETIC. Say only WHICH job deserves HOW MANY extra GPUs on the "
    "evidence. You never name a donor and never balance the books: the market funds every "
    "accepted correction by revoking the GPUs it valued least, and rejects it outright if it "
    "cannot. Feasibility is not your problem and must not shape your judgement.\n"
    "4. NO DOUBLE-COUNTING. Urgency, tier and deadline are already priced into the bid. "
    "Reject any proposal whose only evidence is one of those.\n"
    "Respond with ONLY this JSON object:\n"
    '{"accept": {"<job_id>": <extra GPUs justified by the evidence>}, '
    '"justification": "<one sentence naming the evidence you accepted or why you accepted '
    'none>"}\n'
    "An empty accept object is the correct answer for an ordinary scene."
)


def single_delta(alloc: dict[str, int], jobs: list[dict], supply_note: str, free: int,
                 model: str, cache: dict, host: str = HOST) -> dict:
    """One call: notes in, {"accept": ...} out. Mirrors referee_delta's contract exactly."""
    notes = [f"  {j['jid']} (tier={j['tier']}, deadline={j['deadline']}, "
             f"requested={j['requested']}): {j.get('note') or '(no note)'}" for j in jobs]
    user = "\n".join([f"free_gpus: {free}",
                      f"market_allocation: {dict(sorted(alloc.items()))}",
                      f"unsold_in_pool: {free - sum(alloc.values())}",
                      "job notes:", *notes,
                      f"supply note: {supply_note or '(none)'}"])
    obj = _ask(SYSTEM_SINGLE_DELTA, user, model, host, cache, tag="single-delta")
    if not obj or not isinstance(obj.get("accept"), dict):
        return {"delta": {}, "justification": "no parse", "_source": "fallback"}
    accept = {}
    for jid, n in obj["accept"].items():
        try:
            v = int(round(float(n)))
        except Exception:
            continue
        if v > 0:
            accept[str(jid)] = v
    return {"accept": accept, "justification": str(obj.get("justification", ""))[:300],
            "_source": f"llm:{model}"}


def run_case(case, model: str, use_llm: bool, arm: str) -> tuple[dict, str, dict]:
    """Anchor -> proposal (split or single) -> fund -> validate. Identical after the proposal."""
    floors, alloc, ranking, _env = build_anchor(case)
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    jobs = [{"jid": s["job_id"], "tier": s.get("tier"), "deadline": s.get("deadline"),
             "requested": int(s.get("base_gpus", 0)) + int(s.get("requested_margin_gpus", 0)),
             "note": s.get("justification", "")} for s in dem]
    sup = next((s.get("justification", "") for s in case.stmts if s.get("side") == "supply"), "")
    cache: dict = {}          # per case: the scene key does not hash the exception text

    if arm == "market":
        return alloc, "market: bases then §6 margin clearing", {"delta": {}, "l1": 0,
                                                                "rejected": False, "fired": False}
    if arm == "referee-corr":
        props = gather_corrections(jobs, sup, case.free_gpus, alloc, use_llm=use_llm,
                                   model=model, cache=cache)
        props["ranking"], props["floors"] = ranking, floors
        d = referee_delta(alloc, props, case.free_gpus, use_llm=use_llm, model=model, cache=cache)
        delta = d["delta"]
    else:                                                    # single-corr
        s = single_delta(alloc, jobs, sup, case.free_gpus, model, cache) if use_llm else \
            {"accept": {}, "justification": "rule: market stands"}
        d = s
        # funded through the SAME code path referee_delta uses, with the same ranking/floors,
        # so the two arms differ only in who produced `accept` (the budget is enforced later,
        # in validate_delta, for both arms alike)
        delta = fund(alloc, s.get("accept", {}), case.free_gpus,
                     ranking=ranking, floors=floors) if s.get("accept") else {}

    caps = {j["jid"]: j["requested"] for j in jobs}
    bad = (validate_delta(alloc, delta, case.free_gpus, caps=caps, floors=floors)
           if delta else [])
    final = alloc if bad else apply_correction(alloc, delta)
    meta = {"delta": delta, "l1": sum(abs(v) for v in delta.values()),
            "rejected": bool(bad), "violations": bad, "fired": bool(delta)}
    return final, d.get("justification", ""), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-think", action="store_true")   # accepted for symmetry; correction._ask
    ap.add_argument("--suite", default="r3", choices=["r3", "r12", "all"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from pins.hardcases import CASES
    from pins.hardcases_r3 import CASES_R3, CONTROLS, PRIMARY
    cases = {"r3": CASES_R3, "r12": CASES, "all": CASES + CASES_R3}[a.suite]
    if a.limit:
        cases = cases[:a.limit]

    ARMS = ["market", "single-corr", "referee-corr"]
    out_path = os.environ.get("PINS_RESULTS",
                              os.path.join(HERE, f"results_exp80_{a.model.replace(':', '')}.json"))
    results: dict[str, dict] = {}
    for case in cases:
        arms = {}
        for arm in ARMS:
            final, why, meta = run_case(case, a.model, not a.no_llm and arm != "market", arm)
            arms[arm] = score(case, final, 0, why, True) | {"meta": meta}
        results[case.id] = {"category": case.category, "arms": arms}
        print(f"{case.id:14s} {case.category:12s} " + " ".join(
            f"{k}={'H' if v['handled'] else '.'}"
            f"{'!' if v['meta']['rejected'] else ('*' if v['meta']['fired'] else '')}"
            for k, v in arms.items()))

    json.dump({"model": a.model, "suite": a.suite, "arms": ARMS, "results": results},
              open(out_path, "w"), indent=1)

    ids = [c.id for c in cases]
    prim = [c for c in ids if c in set(PRIMARY)]
    ctrl = [c for c in ids if c in set(CONTROLS)]
    for label, sub in (("PRIMARY", prim), ("CONTROLS", ctrl), ("ALL", ids)):
        if not sub:
            continue
        print(f"\n=== {label}  n={len(sub)} ===")
        mk = {c: results[c]["arms"]["market"]["handled"] for c in sub}
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
        # head-to-head: the question is referee vs single, not either vs the market
        b = sum(1 for c in sub if results[c]["arms"]["referee-corr"]["handled"]
                and not results[c]["arms"]["single-corr"]["handled"])
        cc = sum(1 for c in sub if results[c]["arms"]["single-corr"]["handled"]
                 and not results[c]["arms"]["referee-corr"]["handled"])
        print(f"  head-to-head referee vs single: referee-only {b}, single-only {cc}")

    print(f"\nfull detail -> {out_path}")


if __name__ == "__main__":
    main()
