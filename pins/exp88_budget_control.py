"""Exp 88 — is debate's hard-case win STRUCTURE, or just a bigger inference budget?

PRE-REGISTERED 2026-07-24, written and committed BEFORE the run. Nothing below was chosen
after seeing a number.

WHY THIS EXPERIMENT EXISTS. Exp 83 is the only surviving pro-LLM result in the project:
on the round-3 PRIMARY 31, market 0/31 -> single-pkt 9/31 -> referee-pkt 11/31 ->
debate-pkt 14/31, monotone, zero harm on controls. But debate spends 2N+3 calls per case
against single's 1 (N = demand jobs; N=1..3 here, so 5-9 calls/case, 217 vs 31 over the
suite -- a 7x budget gap). Exp 70/71 already showed what happens when this project's
pro-LLM results meet a budget control: the referee TIED a single-LLM control running on
45% of its tokens and LOST to a 0-token auction, and the mean-outcome case collapsed. The
elevated plan's Sec.13 equal-inference-budget protocol has been open since Exp 68
(research_progress.md:3554) and has never been applied to the tail claim.

  H_0 (the one this experiment is built to be able to CONFIRM): debate-pkt is EQUIVALENT to
  a single LLM given the same number of calls. If so, debate is sampling, not argument, and
  the multi-agent line is closed on evidence -- parallel review is already dead (Exp 79,
  p=0.50) and the split is interface-bound (Exp 81 vs 82).

  H_1: debate-pkt beats the budget-matched single arm, i.e. the STRUCTURE of a rebuttal
  round buys something more samples cannot.

THE CONTROL ARM: `single-pkt-boN`. Identical packet, identical system prompt, identical
menu as `single-pkt`. It draws K = 2N+3 samples -- matched to debate PER CASE, not merely
in total -- and majority-votes the selected action_ids.

  SAMPLING. correction._ask pins temperature 0, so K draws of one prompt are K identical
  answers and self-consistency is unbuildable without changing it. This arm therefore runs
  at temperature 0.8 (`--temp`), and per-sample cache tags keep the draws distinct.
  DECLARED CONFOUND: the control differs from debate on temperature as well as on call
  count. This is intrinsic to best-of-N and is not removable -- deterministic sampling has
  no N. It is disclosed here, before the run, rather than discovered afterwards. It biases
  in the CONTROL's favour if anything (more diverse proposals), which is the conservative
  direction for a test whose interesting outcome is equivalence.

  AGGREGATION. Majority vote over the frozenset of chosen action_ids. Ties are broken
  toward FEWER actions (packet rule 2: "prefer the smallest correction that addresses the
  evidence"), then lexicographically on the id tuple for determinism. The tie-break follows
  the referee prompt's own stated rule rather than being invented to fit an outcome, and it
  is applied identically whatever the result. No extra LLM call is used to aggregate --
  a judge call would re-introduce the structure under test.

SCORING. STRICT (handled AND feasible) is the headline, bare `handled` reported beside it,
exactly as amended in pins/exp79_analyse.py. PRIMARY n=31 only; the 9 controls are reported
separately and NEVER pooled in.

TEST. Pre-registered in pins/exp88_analyse.py: TOST equivalence on the paired per-case
outcomes, margin +/-3 cases, alpha 0.05, via exact conditional (Clopper-Pearson) bounds on
the discordant pairs. Equivalence is the RIGHT frame because the informative result here is
the null; a one-sided McNemar in debate's favour would be the generous direction for the
very claim under suspicion. Project precedent for TOST: Exp 55 (debate-vs-referee increment
EQUIVALENT +/-3), Exp 43/44 (3b == 14b).

DEBATE IS RE-RUN, NOT REPLAYED. debate-pkt is deterministic at temperature 0, so
results_exp83_qwen2514b_debate.json would have been a valid replay. It is re-run anyway so
the harness is shown to reproduce 14/31 end-to-end before anything is compared against it --
the same guard Exp 86 applied to the validator. A mismatch against Exp 83 is itself a
finding and must be reported, not silently accepted.

AMENDMENT, 2026-07-24, made BEFORE the run after a power calculation, disclosed rather than
silently applied. THIS IS A SCREENING RUN, NOT A CONFIRMATORY ONE.

  The +/-3-case margin cannot be met at n=31 by any method. Exact conditional TOST can only
  return PASS when total discordant pairs m <= 3; the unconditional (Wald) version is worse
  still -- a perfect 2v2 tie gives a 90% CI of [-3.29, +3.29], already outside the margin.
  A balanced 5v5 tie yields a CI halfwidth of 16.8pp against a 9.7pp margin. Clearing it
  needs ~217 pairs (7 models x 31), and pooling models is itself unsafe here because Exp 82
  showed the packet is capability-gated (worth -6 at 7b, both arms driven to 0/31): weak
  models would contribute pairs where NEITHER arm handles anything, manufacturing fake
  equivalence out of a floor effect.

  So the formal equivalence test is expected to return INCONCLUSIVE, and that outcome is
  declared in advance so it cannot later be reported as support for either hypothesis. What
  this run is actually for is the POINT ESTIMATE, which is decisive in practice at the
  effect sizes in play:

    boN reaches ~13-14/31  -> debate's Exp 83 win is inference budget, not argument.
    boN stays at ~9-10/31  -> structure survives its most serious objection, and a powered
                              confirmatory run (more CASES, not more models) is then worth
                              the authoring cost.
    anything between       -> genuinely undecided; report as such and do not spin it.

  The TOST remains pre-registered and is still reported; it is simply not expected to be the
  informative half. No number below was chosen after seeing a result.

  .venv/bin/python -m pins.exp88_budget_control --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from pins.correction_signed import (_ask, _job_lines, apply_signed, debate_signed,
                                    disruption, gather_signed)
from pins.h2_eval import build_anchor
from pins.hardcase_eval import score
from pins.packet import SYSTEM_PACKET_SINGLE, build_packet, decision_from_ids

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["market", "single-pkt", "single-pkt-boN", "debate-pkt"]

# Exp 93 structures: the three arms Exp 65-67 tested on the WEAK pre-packet interface and that
# were therefore untested, not refuted. `selfcons` is NOT here -- `single-pkt-boN` already is
# self-consistency (it samples k times at temperature and takes the MODAL action set, see _vote).
STRUCTURE_ARMS = ["debate-noarg-pkt", "critic-pkt"]

SYSTEM_CRITIC = (
    "You are a CRITIC reviewing one job's share of a GPU allocation the market has already made.\n"
    "State only what is WRONG. Do NOT propose a number, a delta, or a new allocation -- objections\n"
    "only. If the allocation is defensible for this job, say so and raise nothing.\n"
    'Reply as JSON: {"problems": ["...", "..."], "ok": true|false}')


def _strip_evidence(props: dict) -> dict:
    """The Exp 66 --no-argue ablation, on the packet: keep WHO proposed WHAT, drop the WHY.

    Isolates whether debate's win is the CONTENT of the arguments or merely the second pass.
    """
    out = dict(props)
    out["demand"] = [{k: v for k, v in d.items() if k != "evidence"}
                     for d in (props.get("demand") or [])]
    if props.get("supply"):
        out["supply"] = {k: v for k, v in props["supply"].items() if k != "evidence"}
    return out


def _critic_signed(jobs, supply_note, free, alloc, model, cache, use_llm=True):
    """Exp 67's critic, rebuilt on the packet: one objection-only reviewer per text-bearing job.

    The original implementation was never committed (only its result JSON survives), so this is a
    faithful reconstruction from those transcripts, which carry a `problems` list and no deltas.
    """
    objections, calls = [], 0
    if not use_llm:
        return {"objections": objections}, calls
    for j in jobs:
        if not (j.get("note") or "").strip():
            continue
        user = "\n".join([f"free_gpus: {free}", f"market_gave_this_job: {alloc.get(j['jid'], 0)}",
                           f"market_allocation: {dict(sorted(alloc.items()))}",
                           "this job:", *_job_lines([j])])
        o = _ask(SYSTEM_CRITIC, user, model, None, cache, tag="critic") or {}
        calls += 1
        probs = [str(x)[:200] for x in (o.get("problems") or [])][:4]
        if probs and not o.get("ok"):
            objections.append({"jid": j["jid"], "problems": probs})
    return {"objections": objections}, calls


def _jobs_of(case, no_text: bool = False):
    """The exp82 packet inputs, rebuilt identically so arms stay paired on the same scene."""
    dem = [s for s in case.stmts if s.get("side") == "demand"]
    jobs = [{"jid": s["job_id"], "tier": s.get("tier"), "deadline": s.get("deadline"),
             "base": int(s.get("base_gpus", 0)),
             "margin": int(s.get("requested_margin_gpus", 0)),
             "requested": int(s.get("base_gpus", 0)) + int(s.get("requested_margin_gpus", 0)),
             "note": "" if no_text else s.get("justification", "")} for s in dem]
    sup = "" if no_text else next(
        (s.get("justification", "") for s in case.stmts if s.get("side") == "supply"), "")
    return jobs, sup


def _decide_once(packet, model, cache, tag, temperature):
    obj = _ask(SYSTEM_PACKET_SINGLE, json.dumps(packet, indent=1), model, None, cache, tag,
               temperature=temperature) or {}
    ids = obj.get("action_ids") or []
    ids = tuple(sorted(str(i) for i in ids))
    return ids, str(obj.get("justification", ""))[:300]


def _vote(samples):
    """Modal action_id set. Ties -> fewer actions (packet rule 2), then lexicographic."""
    counts = Counter(ids for ids, _ in samples)
    best = min(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0]
    why = next(j for ids, j in samples if ids == best)
    return best, why, dict(sorted(((",".join(k) or "(none)", v) for k, v in counts.items())))


def run_case(case, model, use_llm, arm, max_delta, temperature, no_text=False):
    floors, alloc, ranking, _env = build_anchor(case)
    jobs, sup = _jobs_of(case, no_text)
    cache: dict = {}
    n_jobs = len(jobs)
    k = 2 * n_jobs + 3                     # debate's per-case call count, matched exactly

    if arm == "market":
        return alloc, "market: bases then §6 margin clearing", {
            "changes": {}, "hold_free": 0, "moved": 0, "rejected": False, "fired": False,
            "n_calls": 0, "k": 0, "votes": {}}

    props = None
    critic_calls = 0
    if arm in ("debate-pkt", "debate-noarg-pkt"):
        p = gather_signed(jobs, sup, case.free_gpus, alloc, use_llm=use_llm,
                          model=model, cache=cache)
        p = debate_signed(jobs, sup, case.free_gpus, alloc, p, use_llm=use_llm,
                          model=model, cache=cache)
        props = {kk: v for kk, v in p.items() if kk in ("demand", "supply", "opening", "debated")}
        if arm == "debate-noarg-pkt":
            props = _strip_evidence(props)
    elif arm == "critic-pkt":
        props, critic_calls = _critic_signed(jobs, sup, case.free_gpus, alloc, model, cache,
                                             use_llm=use_llm)

    packet = build_packet(jobs, sup, case.free_gpus, alloc, ranking, floors,
                          max_delta=max_delta, reviewer_proposals=props, history=None)

    votes: dict = {}
    if not use_llm:
        d, why, n_calls = {"changes": {}, "hold_free": 0, "picked": [], "unknown_ids": []}, \
            "rule: market stands", 0
    elif arm == "single-pkt-boN":
        samples = [_decide_once(packet, model, cache, f"sgl-pkt-bon-s{i}", temperature)
                   for i in range(k)]
        ids, why, votes = _vote(samples)
        d = decision_from_ids(packet, list(ids))
        n_calls = k
    else:                                   # single-pkt (1 call) and debate-pkt (referee call)
        from pins.packet import SYSTEM_PACKET_REFEREE
        sysmsg = SYSTEM_PACKET_SINGLE if arm == "single-pkt" else SYSTEM_PACKET_REFEREE
        obj = _ask(sysmsg, json.dumps(packet, indent=1), model, None, cache, arm) or {}
        d = decision_from_ids(packet, obj.get("action_ids"))
        why = str(obj.get("justification", ""))[:300]
        n_calls = (2 * n_jobs + 3) if arm in ("debate-pkt", "debate-noarg-pkt") else (
            critic_calls + 1 if arm == "critic-pkt" else 1)

    fired = bool(d["changes"]) or bool(d["hold_free"])
    final, viol = apply_signed(alloc, d, case.free_gpus, ranking=ranking, floors=floors,
                               budget=max_delta)
    meta = {"changes": d["changes"], "hold_free": d["hold_free"],
            "moved": disruption(alloc, final), "rejected": bool(viol), "fired": fired,
            "n_calls": n_calls, "k": k if arm == "single-pkt-boN" else 0, "votes": votes,
            "unknown_ids": d.get("unknown_ids", []), "n_jobs": n_jobs}
    return final, why, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.8,
                    help="temperature for the boN control ONLY; every other arm stays at 0")
    ap.add_argument("--suite", default="r3", choices=["r3", "r34"],
                    help="r3 = Exp 88 (round-3 31/9). r34 = Exp 89 pooled round-3 + round-4 "
                         "(81 primary / 17 control).")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm list. Default reproduces Exp 88/89 exactly. "
                         f"Exp 93 structures: {','.join(STRUCTURE_ARMS)} "
                         "(selfcons is NOT one -- single-pkt-boN already is self-consistency).")
    a = ap.parse_args()
    arm_list = [s.strip() for s in a.arms.split(",") if s.strip()] or ARMS
    bad = [x for x in arm_list if x not in ARMS + STRUCTURE_ARMS]
    assert not bad, f"unknown arm(s): {bad}"

    from pins.hardcases_r3 import CASES_R3, CONTROLS as CTRL3, PRIMARY as PRIM3
    if a.suite == "r34":
        from pins.hardcases_r4 import CASES_R4, CONTROLS_R4, PRIMARY_R4
        all_cases, PRIMARY, CONTROLS = CASES_R3 + CASES_R4, PRIM3 + PRIMARY_R4, CTRL3 + CONTROLS_R4
    else:
        all_cases, PRIMARY, CONTROLS = CASES_R3, PRIM3, CTRL3
    cases = all_cases[:a.limit] if a.limit else all_cases
    exp = "exp89" if a.suite == "r34" else "exp88"

    out_path = os.environ.get(
        "PINS_RESULTS",
        os.path.join(HERE, f"results_{exp}_{a.model.replace(':', '')}_t{a.temp}.json"))
    results: dict = {}
    for case in cases:
        arms = {}
        for arm in arm_list:
            temp = a.temp if arm == "single-pkt-boN" else 0
            final, why, meta = run_case(case, a.model, not a.no_llm and arm != "market",
                                        arm, a.max_delta, temp)
            arms[arm] = score(case, final, 0, why, True) | {"meta": meta}
        results[case.id] = {"category": case.category, "arms": arms}
        print(f"{case.id:14s} {case.category:12s} " + " ".join(
            f"{k}={'H' if v['handled'] else '.'}"
            f"{'!' if v['meta']['rejected'] else ('*' if v['meta']['fired'] else '')}"
            for k, v in arms.items()), flush=True)

    json.dump({"model": a.model, "suite": a.suite, "arms": arm_list, "max_delta": a.max_delta,
               "temperature_boN": a.temp, "results": results}, open(out_path, "w"), indent=1)

    ids = [c.id for c in cases]
    for label, sub in (("PRIMARY", [c for c in ids if c in set(PRIMARY)]),
                       ("CONTROLS", [c for c in ids if c in set(CONTROLS)])):
        if not sub:
            continue
        print(f"\n=== {label}  n={len(sub)} ===")
        print(f"  {'arm':15s}{'strict':>9s}{'handled':>9s}{'fired':>7s}{'invalid':>9s}{'calls':>7s}")
        for arm in arm_list:
            st = sum(1 for c in sub if results[c]["arms"][arm]["handled"]
                     and not results[c]["arms"][arm]["meta"]["rejected"])
            h = sum(1 for c in sub if results[c]["arms"][arm]["handled"])
            f = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["fired"])
            r = sum(1 for c in sub if results[c]["arms"][arm]["meta"]["rejected"])
            nc = sum(results[c]["arms"][arm]["meta"]["n_calls"] for c in sub)
            print(f"  {arm:15s}{st:>6d}/{len(sub):<3d}{h:>6d}/{len(sub):<3d}"
                  f"{f:>7d}{r:>9d}{nc:>7d}")
        pairs = [(x, y) for x, y in (("debate-pkt", "single-pkt-boN"),
                                     ("debate-pkt", "single-pkt"),
                                     ("single-pkt-boN", "single-pkt"),
                                     ("debate-noarg-pkt", "debate-pkt"),
                                     ("debate-noarg-pkt", "single-pkt"),
                                     ("critic-pkt", "debate-pkt"),
                                     ("critic-pkt", "single-pkt"))
                 if x in arm_list and y in arm_list]
        for x, y in pairs:
            b = sum(1 for c in sub if results[c]["arms"][x]["handled"]
                    and not results[c]["arms"][y]["handled"])
            cc = sum(1 for c in sub if results[c]["arms"][y]["handled"]
                     and not results[c]["arms"][x]["handled"])
            print(f"  head-to-head {x} vs {y}: b={b} c={cc}")

    analyser = "exp89_analyse" if a.suite == "r34" else "exp88_analyse"
    print(f"\nfull detail -> {out_path}\nnow run: .venv/bin/python -m pins.{analyser} {out_path}")


if __name__ == "__main__":
    main()
