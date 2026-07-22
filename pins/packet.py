"""The referee PACKET — a structured decision document, with a code-generated action menu.

User's revision (2026-07-23): the referee's context was too thin to judge well. The measurement
agrees. In Exp 81 the referee's dominant failure was "cannot fund" — 52% of its decisions
rejected at 7b, 62% at 14b — because each reviewer sees exactly one job and NOBODY sees the
budget. It was proposing changes the market could not pay for.

The packet fixes that by construction rather than by prompting:

  candidate_actions is ENUMERATED IN CODE and every entry is verified legal before the model
  ever sees it — feasible, fundable, within budget, floors respected. The model's job stops
  being generation and becomes SELECTION from a menu. "Cannot fund" becomes unreachable.

This is the project's design rule tightened, not relaxed: the LLM reasons over a menu, code
guarantees every item on the menu is executable.

TWO DELIBERATE DEPARTURES FROM THE SCHEMA AS PROPOSED, both flagged before building:

1. `max_gpu_delta: 1` is a flag defaulting to 6, not a constant of 1. 22 of the 31 primary
   round-3 cases require RESTRAINT, and the actions they need are "zero the suspended 4-GPU
   job" or "hold 4 free for the grid cap". A delta cap of 1 makes most of them unreachable, so
   a cap of 1 would produce a null BY CONSTRUCTION — the same class of artifact as Exp 80's
   add-only contract. Run `--max-delta 1` deliberately to test the tight-budget policy; do not
   let it be the silent default.

2. `minimum_confidence_to_override` is COLLECTED, not enforced. Self-reported confidence from
   an LLM is generally uncalibrated, and this project has already measured justifications that
   do not track actions. Gating on an invented number would silently decide the experiment. So
   the model reports confidence, the packet states the threshold as guidance, and the harness
   measures whether confidence >= 0.70 actually predicts a correct override. If it turns out
   calibrated, the gate becomes evidence-based; if not, that is a finding about faithfulness.

INFORMATION IS HELD CONSTANT ACROSS ARMS. Both the single LLM and the referee receive the
identical packet, including the raw job notes in `demand_claims`. The referee additionally
receives `reviewer_proposals`. That makes the referee's information a strict SUPERSET of the
single LLM's — deliberately, because Exp 81's referee saw only the reviewers' compressed
summaries and never the note itself, which confounded "does the split help?" with "does the
split lose information?". If a strictly-better-informed referee still does not win, that is
decisive in a way Exp 81 was not.
"""
from __future__ import annotations

from pins.correction_signed import apply_signed

DEFAULT_RULES = {
    "default_action": "retain_market",
    "max_gpu_delta": 6,
    "minimum_confidence_to_override": 0.70,
    "role_label_alone_is_not_evidence": True,
}


def _legal(alloc, decision, free, ranking, floors, budget):
    final, viol = apply_signed(alloc, decision, free, ranking=ranking, floors=floors,
                               budget=budget)
    return (None if viol else final)


def candidate_actions(alloc: dict[str, int], free: int, ranking, floors,
                      max_delta: int) -> list[dict]:
    """Every action the code can actually execute, verified one by one. Menu, not suggestions."""
    out = [{"id": 0, "type": "retain_market", "description": "change nothing",
            "resulting_allocation": dict(sorted(alloc.items()))}]
    nid = 1
    for jid in sorted(alloc):
        for k in range(1, max_delta + 1):
            for typ, delta in (("grant", k), ("revoke", -k)):
                if typ == "revoke" and alloc.get(jid, 0) < k:
                    continue
                final = _legal(alloc, {"changes": {jid: delta}, "hold_free": 0},
                               free, ranking, floors, max_delta)
                if final is None or final == alloc:
                    continue
                out.append({"id": nid, "type": typ, "job": jid, "gpus": k,
                            "description": (f"give {jid} {k} more GPU(s)" if typ == "grant"
                                            else f"take {k} GPU(s) away from {jid}"),
                            "resulting_allocation": dict(sorted(final.items()))})
                nid += 1
    # TRANSFERS are not a convenience: a bare grant is usually unfundable, because funding it
    # would have to come out of another job's floor and code refuses that. Only an EXPLICIT
    # negative judgement may breach a floor (correction_signed departure 1), so "take 4 from the
    # suspended job, give them to the one that can use them" exists only as a paired action.
    # Without these the menu cannot reach the answer for most restraint cases, and the referee
    # would fail them for want of a listable option.
    for donor in sorted(alloc):
        for benef in sorted(alloc):
            if donor == benef:
                continue
            for k in range(1, max_delta + 1):
                if alloc.get(donor, 0) < k:
                    continue
                final = _legal(alloc, {"changes": {donor: -k, benef: k}, "hold_free": 0},
                               free, ranking, floors, max_delta)
                if final is None or final == alloc:
                    continue
                out.append({"id": nid, "type": "transfer", "from": donor, "to": benef, "gpus": k,
                            "description": f"move {k} GPU(s) from {donor} to {benef}",
                            "resulting_allocation": dict(sorted(final.items()))})
                nid += 1
    for k in range(1, max_delta + 1):
        final = _legal(alloc, {"changes": {}, "hold_free": k}, free, ranking, floors, max_delta)
        if final is None or final == alloc:
            continue
        out.append({"id": nid, "type": "hold_free", "gpus": k,
                    "description": f"leave {k} GPU(s) unallocated",
                    "resulting_allocation": dict(sorted(final.items()))})
        nid += 1

    # collapse duplicates: several action shapes reach the same allocation, and a menu that
    # lists one outcome three ways invites an arbitrary choice between identical options
    seen, dedup = set(), []
    for a in out:
        k = tuple(sorted(a["resulting_allocation"].items()))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(dict(a, id=len(dedup)))
    return dedup


def build_packet(jobs: list[dict], supply_note: str, free: int, alloc: dict[str, int],
                 ranking, floors, *, max_delta: int = 6, history: list | None = None,
                 reviewer_proposals: dict | None = None) -> dict:
    rules = dict(DEFAULT_RULES, max_gpu_delta=max_delta)
    pkt = {
        "state_summary": {
            "free_gpus": free,
            "allocated_by_market": sum(alloc.values()),
            "unsold_in_pool": free - sum(alloc.values()),
            "job_count": len(jobs),
        },
        "market_baseline": {
            "allocation": dict(sorted(alloc.items())),
            "valued_least_first": [j for _v, j in (ranking or [])],
            "note": "the market funds any grant by revoking from the jobs it valued least",
        },
        "demand_claims": [
            {"job_id": j["jid"], "tier": j["tier"], "deadline": j["deadline"],
             "requested": j["requested"], "market_gave": alloc.get(j["jid"], 0),
             "note": j.get("note") or ""} for j in jobs
        ],
        "supply_claim": {"note": supply_note or ""},
        "hard_constraints": {
            "pool_capacity": free,
            "guaranteed_minimum_per_job": dict(sorted((floors or {}).items())),
            "allocations_cannot_be_negative": True,
            "max_gpus_moved_per_decision": max_delta,
        },
        "history": history or [],
        "candidate_actions": candidate_actions(alloc, free, ranking, floors, max_delta),
        "decision_rules": rules,
    }
    if reviewer_proposals is not None:
        pkt["reviewer_proposals"] = reviewer_proposals
    return pkt


_CONTRACT = (
    "You are given a decision packet. `candidate_actions` lists every action that is legal, "
    "fundable and within budget — the list is generated by code and nothing outside it can be "
    "executed. Choose the actions to take, by id.\n"
    "Rules, in order:\n"
    "1. THE MARKET IS THE DEFAULT. `retain_market` (id 0) is the correct answer unless a note "
    "carries a fact the numerical bid could not contain — a standing operator instruction, a "
    "physical or contractual limit, a dependency, a suspension, a starvation history, work that "
    "will be discarded, or a declaration the note itself says is wrong.\n"
    "2. A ROLE, TIER OR DEADLINE IS NOT EVIDENCE. Those are already priced into the bid. A note "
    "that only restates them justifies nothing.\n"
    "3. EVERY CHANGE COSTS. Prefer the smallest set of actions that addresses the evidence.\n"
    "4. YOU DO NOT DO ARITHMETIC. Every listed action already fits the pool; feasibility is not "
    "your problem. Choosing several actions at once may not fit, and will then be rejected — so "
    "keep the set small.\n"
    "Respond with ONLY this JSON object:\n"
    '{"action_ids": [<ids from candidate_actions, [0] to change nothing>], '
    '"confidence": <0.0 to 1.0, how sure you are the evidence justifies overriding the market>, '
    '"justification": "<one sentence quoting the evidence you acted on, or why none qualified>"}'
)

SYSTEM_PACKET_SINGLE = (
    "You are the SCHEDULER of a GPU pool. A market has already allocated it from the jobs' "
    "numerical bids.\n" + _CONTRACT)

SYSTEM_PACKET_REFEREE = (
    "You are the REFEREE of a GPU pool. A market has already allocated it from the jobs' "
    "numerical bids, and two reviewers have inspected it — their proposals are in "
    "`reviewer_proposals`. You decide what is actually justified.\n" + _CONTRACT)


def decision_from_ids(packet: dict, action_ids) -> dict:
    """Turn selected ids into a signed decision. Unknown ids are dropped, not guessed at."""
    by_id = {a["id"]: a for a in packet["candidate_actions"]}
    changes: dict[str, int] = {}
    hold = 0
    picked, unknown = [], []
    for i in action_ids or []:
        try:
            i = int(i)
        except (TypeError, ValueError):
            continue
        a = by_id.get(i)
        if a is None:
            unknown.append(i)
            continue
        picked.append(i)
        if a["type"] == "grant":
            changes[a["job"]] = changes.get(a["job"], 0) + a["gpus"]
        elif a["type"] == "revoke":
            changes[a["job"]] = changes.get(a["job"], 0) - a["gpus"]
        elif a["type"] == "transfer":
            changes[a["from"]] = changes.get(a["from"], 0) - a["gpus"]
            changes[a["to"]] = changes.get(a["to"], 0) + a["gpus"]
        elif a["type"] == "hold_free":
            hold = max(hold, a["gpus"])
    return {"changes": {j: v for j, v in changes.items() if v}, "hold_free": hold,
            "picked": picked, "unknown_ids": unknown}
