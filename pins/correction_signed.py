"""SIGNED corrections — the half of the action space pins/correction.py cannot express.

Exp 80 finding that motivates this file. On the 40 round-3 cases, both the referee split and the
single LLM fired on exactly ONE case, and its delta was rejected. It was not a wiring bug: the
models read the notes and reasoned correctly, producing justifications like "the operator note
restricts allocations to bases only", "the failure of the chiller provides valid evidence to
reduce", "the account owning job r00 is suspended, justifying withhold". Every one of those is a
DOWNWARD correction, and the Exp 77 contract is `{"accept": {jid: extra_gpus}}` with positive
integers only — the demand reviewer is literally asked "does this job need MORE GPU than its bid
suggests?". 22 of the 31 primary cases require restraint, so the channel could not carry the
answer for ~70% of the suite. The LLM said the right thing and the interface discarded it.

So this module keeps everything that worked and widens only the contract:

    changes    {jid: SIGNED int}   negative = this job should get less, or nothing
    hold_free  int                 leave this many GPUs unallocated (caps, quiet windows,
                                   reservations for work that has not been submitted)

The design rule is untouched: the LLM still names WHO deserves WHAT and never does arithmetic.
Code still funds every grant, still decides who pays, still guarantees feasibility, and an
invalid decision still leaves the market's allocation standing.

Four deliberate departures from correction.py, each of which could be wrong and so is stated:

1. A NEGATIVE ENTRY MAY BREACH A JOB'S FLOOR. In correction.py a job's base is sacred and only
   its margin is contested. That rule makes the suspended-account and killed-by-maintenance
   cases unanswerable, because the correct act is to give a job with a valid base exactly zero.
   A floor is protection against the MECHANISM taking your base to fund someone else; it is not
   protection against a judgement that your job should not run. So code still never revokes a
   base to FUND a grant, but an explicit negative judgement about a named job may.

2. POSITIVE GRANTS ARE NOT CAPPED BY THE JOB'S REQUEST. correction.py caps a grant at the job's
   own declared base+margin, which in Exp 80 rejected the single correction anyone proposed:
   CORRUPT-05 declares 0 because its monitoring crashed, so `r00: 3 exceeds its usable ceiling
   0` — the cap trusted exactly the number the case is about. The change budget already bounds
   how far any decision can move, so the request-derived cap is redundant protection that
   specifically breaks corrupt-declaration cases.

3. BUDGET DEFAULTS TO 6, NOT 4. A restraint action costs the full size of the job being
   withheld, so zeroing one 4-GPU job plus holding 2 free already exceeds the Exp 77 budget of
   4. Reported alongside results, and adjustable, because it is a knob and not a finding.

4. DISRUPTION IS MEASURED AS max(GAINED, LOST), NOT ||dA||_1. L1 double-counts a transfer:
   moving 4 GPUs from one job to another scores 8, so the Exp 77 budget of 4 would reject every
   transfer larger than two GPUs — and a transfer is precisely what a restraint case needs
   ("take r00's four, give them to r01"). max(gained, lost) is the number of GPUs that actually
   changed hands: 4 for that transfer, 4 for holding four idle, 3 for a grant funded from
   unsold capacity. Caught by a unit test on the suspended-account shape before any run.
"""
from __future__ import annotations

import json

from pins.correction import HOST, _ask as _ask_raw
from pins.llm_agent import DEFAULT_MODEL
from pins.referee import _HYBRID


def _ask(system, user, model, host, cache, tag):
    """Give hybrid reasoners room to think AND still emit the JSON.

    correction._ask caps output at 200 tokens, which is ample for a non-reasoner emitting a
    small object but is entirely consumed by deepseek-r1/qwen3's thinking channel -- the model
    would return nothing parseable and be scored as 'proposed no change'. Exp 79 dodged this by
    forcing --no-think; here the reasoners are tested at FULL STRENGTH instead, with the budget
    raised to match referee.py's 4096."""
    hybrid = _HYBRID(model)
    return _ask_raw(system, user, model, host, cache, tag,
                    num_predict=4096 if hybrid else 200,
                    think=True if hybrid else None)

SIGNED_BUDGET = 6

_RULES = (
    "Rules, in order:\n"
    "1. THE MARKET IS THE DEFAULT. Override it only where a note carries a fact the bid could "
    "not contain — a standing operator instruction, a physical or contractual limit, a "
    "dependency, a starvation history, a job that should not run at all. No evidence, no "
    "change.\n"
    "2. EVERY CHANGE COSTS. A job whose allocation moves pays a reconfiguration penalty, so "
    "prefer the smallest correction that addresses the evidence.\n"
    "3. YOU DO NOT DO ARITHMETIC. Say only WHICH job should get MORE or LESS, and how much, and "
    "how many GPUs (if any) must be left unallocated. Never name a donor and never balance the "
    "books: code funds every grant by revoking the units the market valued least, and rejects "
    "the whole decision if it cannot. Feasibility is not your problem and must not shape your "
    "judgement.\n"
    "4. NO DOUBLE-COUNTING. Urgency, tier and deadline are already priced into the bid. Reject "
    "any proposal whose only evidence is one of those.\n"
    "Respond with ONLY this JSON object:\n"
    '{"changes": {"<job_id>": <signed integer, negative to take GPUs away>}, '
    '"hold_free": <integer GPUs to leave unallocated, 0 if none>, '
    '"justification": "<one sentence naming the evidence you acted on, or why you acted on '
    'none>"}\n'
    "Empty changes and hold_free 0 is the correct answer for an ordinary scene."
)

SYSTEM_REFEREE_SIGNED = (
    "You are the REFEREE. A market has produced an allocation from the jobs' numerical bids. "
    "Two reviewers have proposed CHANGES to it, each with evidence. You do NOT produce an "
    "allocation. You decide only how much of the proposed change is justified.\n" + _RULES)

SYSTEM_SINGLE_SIGNED = (
    "You are the SCHEDULER. A market has produced an allocation from the jobs' numerical bids. "
    "You are given every job's free-text note. You do NOT produce an allocation. You decide "
    "only which jobs deserve more or less than the market gave them.\n" + _RULES)

SYSTEM_DEMAND_SIGNED = (
    "You are the DEMAND-side reviewer for one job in a GPU scheduler. A market has already "
    "allocated the pool using each job's numerical bid. Your ONLY question is:\n"
    "  should this job get MORE than the market gave it, or LESS (possibly nothing)?\n"
    "Answer non-zero only when the job's free-text note carries a fact the numbers cannot "
    "express — starvation history, an external or contractual deadline, a standing operator "
    "instruction, a dependency, a suspension, work that will be discarded or killed, a "
    "declaration the note itself says is wrong. Ordinary urgency, a tight deadline or a large "
    "request are ALREADY in the bid: they are not evidence.\n"
    "Respond with ONLY this JSON object:\n"
    '{"delta_gpus": <signed integer, negative to take GPUs away, 0 for no change>, '
    '"evidence": "<the exact phrase from the note that justifies it, or empty>"}')

SYSTEM_SUPPLY_SIGNED = (
    "You are the SUPPLY-side reviewer for a GPU cluster. A market has already allocated the "
    "free pool using each job's numerical bid. Your ONLY question is:\n"
    "  must some capacity be left UNALLOCATED this tick, and how much?\n"
    "Answer non-zero only when the notes carry a fact the market's price does not contain — an "
    "announced arrival, a maintenance or drain window, a power or thermal cap, a reservation "
    "for work that has not been submitted, a correctness limit. Ordinary scarcity is ALREADY in "
    "the ask price.\n"
    "Respond with ONLY this JSON object:\n"
    '{"hold_free_gpus": <0 or a positive integer>, "evidence": "<the phrase that justifies it, '
    'or empty>"}')

PROMPT_VERSION = "signed-v1"


def _job_lines(jobs: list[dict]) -> list[str]:
    return [f"  {j['jid']} (tier={j['tier']}, deadline={j['deadline']}, "
            f"requested={j['requested']}): {j.get('note') or '(no note)'}" for j in jobs]


def _parse_decision(obj: dict | None) -> dict:
    if not obj:
        return {"changes": {}, "hold_free": 0, "justification": "no parse", "_source": "fallback"}
    changes: dict[str, int] = {}
    for jid, v in (obj.get("changes") or {}).items():
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            continue
        if n:
            changes[str(jid)] = n
    try:
        hold = max(0, int(round(float(obj.get("hold_free", 0) or 0))))
    except (TypeError, ValueError):
        hold = 0
    return {"changes": changes, "hold_free": hold,
            "justification": str(obj.get("justification", ""))[:300]}


def gather_signed(jobs: list[dict], supply_note: str, free: int, alloc: dict[str, int],
                  use_llm: bool = True, model: str = DEFAULT_MODEL,
                  cache: dict | None = None, host: str = HOST) -> dict:
    """Reviewers, one per job that carries text, plus one supply call. Proposals, not moves."""
    cache = {} if cache is None else cache
    props: dict = {"demand": [], "supply": None}
    if not use_llm:
        return props
    for j in jobs:
        if not (j.get("note") or "").strip():
            continue                                  # no context -> no call -> no proposal
        user = "\n".join([f"free_gpus: {free}", f"market_gave_this_job: {alloc.get(j['jid'], 0)}",
                          f"market_allocation: {dict(sorted(alloc.items()))}",
                          "this job:", *_job_lines([j])])
        o = _ask(SYSTEM_DEMAND_SIGNED, user, model, host, cache, tag="dem-signed")
        try:
            n = int(round(float((o or {}).get("delta_gpus", 0) or 0)))
        except (TypeError, ValueError):
            n = 0
        if n:
            props["demand"].append({"jid": j["jid"], "delta": n,
                                    "evidence": str((o or {}).get("evidence", ""))[:200]})
    if (supply_note or "").strip():
        user = "\n".join([f"free_gpus: {free}", f"allocated: {sum(alloc.values())}",
                          f"supply note: {supply_note}"])
        o = _ask(SYSTEM_SUPPLY_SIGNED, user, model, host, cache, tag="sup-signed")
        try:
            n = max(0, int(round(float((o or {}).get("hold_free_gpus", 0) or 0))))
        except (TypeError, ValueError):
            n = 0
        if n:
            props["supply"] = {"hold_free": n,
                               "evidence": str((o or {}).get("evidence", ""))[:200]}
    return props


def referee_signed(alloc: dict[str, int], props: dict, free: int, jobs: list[dict],
                   use_llm: bool = True, model: str = DEFAULT_MODEL,
                   cache: dict | None = None, host: str = HOST) -> dict:
    """Rules on the reviewers' proposals. No proposals -> no call, exactly as in Exp 77."""
    cache = {} if cache is None else cache
    if not props["demand"] and not props["supply"]:
        return {"changes": {}, "hold_free": 0, "justification": "no contextual evidence raised",
                "_source": "skip"}
    if not use_llm:
        return {"changes": {}, "hold_free": 0, "justification": "rule: market stands",
                "_source": "rule"}
    lines = [f"free_gpus: {free}", f"market_allocation: {dict(sorted(alloc.items()))}",
             f"unsold_in_pool: {free - sum(alloc.values())}", "proposed changes:"]
    for p in props["demand"]:
        lines.append(f"  demand reviewer, {p['jid']}: {p['delta']:+d} GPUs — "
                     f"evidence: {p['evidence']!r}")
    if props["supply"]:
        lines.append(f"  supply reviewer: hold {props['supply']['hold_free']} GPUs free — "
                     f"evidence: {props['supply']['evidence']!r}")
    out = _parse_decision(_ask(SYSTEM_REFEREE_SIGNED, "\n".join(lines), model, host, cache,
                               tag="ref-signed"))
    return out | {"_source": f"llm:{model}"}


def single_signed(alloc: dict[str, int], jobs: list[dict], supply_note: str, free: int,
                  use_llm: bool = True, model: str = DEFAULT_MODEL,
                  cache: dict | None = None, host: str = HOST) -> dict:
    """One call over the raw notes. Same contract, no reviewer/referee split."""
    cache = {} if cache is None else cache
    if not use_llm:
        return {"changes": {}, "hold_free": 0, "justification": "rule: market stands",
                "_source": "rule"}
    user = "\n".join([f"free_gpus: {free}",
                      f"market_allocation: {dict(sorted(alloc.items()))}",
                      f"unsold_in_pool: {free - sum(alloc.values())}",
                      "job notes:", *_job_lines(jobs),
                      f"supply note: {supply_note or '(none)'}"])
    out = _parse_decision(_ask(SYSTEM_SINGLE_SIGNED, user, model, host, cache, tag="sgl-signed"))
    return out | {"_source": f"llm:{model}"}


def disruption(before: dict[str, int], after: dict[str, int]) -> int:
    """GPUs that actually changed hands (departure 4). A transfer of k counts k, not 2k."""
    keys = set(before) | set(after)
    gained = sum(max(0, after.get(j, 0) - before.get(j, 0)) for j in keys)
    lost = sum(max(0, before.get(j, 0) - after.get(j, 0)) for j in keys)
    return max(gained, lost)


def apply_signed(alloc: dict[str, int], decision: dict, free: int,
                 ranking: list[tuple[float, str]] | None = None,
                 floors: dict[str, int] | None = None,
                 budget: int = SIGNED_BUDGET) -> tuple[dict[str, int], list[str]]:
    """Turn a signed judgement into a feasible allocation, or reject it.

    Returns (final_allocation, violations). A non-empty violations list means the decision was
    REJECTED and the market's allocation stands — never repaired, per the Exp 77 rule.
    """
    changes = {j: v for j, v in decision.get("changes", {}).items() if j in alloc}
    unknown = [j for j in decision.get("changes", {}) if j not in alloc]
    hold = max(0, int(decision.get("hold_free", 0)))
    floors = floors or {}

    new = dict(alloc)
    for jid, v in changes.items():
        new[jid] = max(0, new[jid] + v)

    # capacity available to the jobs after honouring the hold
    ceiling = free - hold
    beneficiaries = {j for j, v in changes.items() if v > 0}
    reduced = {j for j, v in changes.items() if v < 0}

    need = sum(new.values()) - ceiling
    if need > 0:
        # revoke from the market's least-valued units: never from a beneficiary, and never
        # below a job's floor UNLESS the decision explicitly reduced that job (departure 1)
        donors = list(ranking or sorted((1.0, j) for j in new))
        for _v, jid in donors:
            if need <= 0:
                break
            if jid in beneficiaries:
                continue
            floor = 0 if jid in reduced else floors.get(jid, 0)
            take = min(max(0, new.get(jid, 0) - floor), need)
            if take:
                new[jid] -= take
                need -= take

    viol: list[str] = []
    if unknown:
        viol.append(f"unknown job(s): {sorted(unknown)}")
    if need > 0:
        viol.append(f"cannot fund: {need} GPUs short of the requested hold/grants")
    moved = disruption(alloc, new)
    if moved > budget:
        viol.append(f"change budget: {moved} GPUs moved > {budget}")
    if sum(new.values()) > free:
        viol.append(f"infeasible: {sum(new.values())} > {free}")
    return (alloc if viol else new), viol


# --------------------------------------------------------------------------- #
#  DEBATE round (Exp 83). Mirrors referee.rebut's convention: each side reads the
#  opposing positions and may revise ITS OWN ask, arguing only its own corner.
#
#  One deviation, stated: referee.rebut enforces monotone concession (demand may
#  not raise, supply may not raise) to guarantee termination of an open-ended
#  protocol. Here there is exactly ONE rebuttal round, so termination is trivial
#  and a monotonicity rule would only stop a reviewer from correcting itself
#  after reading evidence it had not seen. Revision is therefore free in both
#  directions, and the opening position is kept in the packet so the referee can
#  see who moved and why.
# --------------------------------------------------------------------------- #
SYSTEM_DEMAND_REBUT = (
    "You are the DEMAND-side reviewer for ONE job. You already stated whether it should get "
    "more or less than the market gave it. You can now see what the other reviewers said about "
    "the other jobs, and what the supply side wants held free. The pool is finite: every extra "
    "GPU your job gets comes out of another job on this list.\n"
    "Revise your own position if the other positions change what is justified — you may raise "
    "or lower it — and argue your corner to the referee in one sentence. Only your own job.\n"
    "Respond with ONLY this JSON object:\n"
    '{"delta_gpus": <signed integer, 0 for no change>, "argument": "<one sentence to the '
    'referee>"}')

SYSTEM_SUPPLY_REBUT = (
    "You are the SUPPLY-side reviewer. You already stated how many GPUs must be left "
    "unallocated. You can now see every job's position and the evidence behind it. If the "
    "demand on the table is more urgent than what you are holding capacity for, releasing it "
    "is the right call — but do not release capacity that a stated limit or an announced "
    "arrival genuinely requires.\n"
    "Respond with ONLY this JSON object:\n"
    '{"hold_free_gpus": <integer, 0 for none>, "argument": "<one sentence to the referee>"}')


def _opening_view(props: dict, me: str | None) -> dict:
    """What one reviewer may see: every OTHER position and its evidence, never private state."""
    return {"other_jobs": [{"job": p["jid"], "proposed_delta": p["delta"],
                            "evidence": p["evidence"]}
                           for p in props["demand"] if p["jid"] != me],
            "supply": ({"hold_free": props["supply"]["hold_free"],
                        "evidence": props["supply"]["evidence"]} if props["supply"] else None)}


def debate_signed(jobs: list[dict], supply_note: str, free: int, alloc: dict[str, int],
                  props: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
                  cache: dict | None = None, host: str = HOST) -> dict:
    """One rebuttal round over the openings. Returns the same shape, plus `opening` kept."""
    cache = {} if cache is None else cache
    if not use_llm or (not props["demand"] and not props["supply"]):
        return dict(props, opening=props, debated=False)

    revised: list[dict] = []
    for j in jobs:
        if not (j.get("note") or "").strip():
            continue
        mine = next((p for p in props["demand"] if p["jid"] == j["jid"]), None)
        view = _opening_view(props, j["jid"])
        if not view["other_jobs"] and not view["supply"]:
            if mine:
                revised.append(dict(mine, argument=""))
            continue
        user = json.dumps({"free_gpus": free, "market_allocation": dict(sorted(alloc.items())),
                           "my_job": {"job": j["jid"], "tier": j["tier"],
                                      "deadline": j["deadline"], "note": j.get("note") or "",
                                      "market_gave": alloc.get(j["jid"], 0),
                                      "my_opening_delta": (mine or {}).get("delta", 0)},
                           "other_positions": view}, indent=1)
        o = _ask(SYSTEM_DEMAND_REBUT, user, model, host, cache, tag=f"dem-rebut-{j['jid']}")
        try:
            n = int(round(float((o or {}).get("delta_gpus", (mine or {}).get("delta", 0)) or 0)))
        except (TypeError, ValueError):
            n = (mine or {}).get("delta", 0)
        if n:
            revised.append({"jid": j["jid"], "delta": n,
                            "evidence": (mine or {}).get("evidence", ""),
                            "argument": str((o or {}).get("argument", ""))[:200]})

    sup = props["supply"]
    if sup or props["demand"]:
        user = json.dumps({"free_gpus": free, "allocated": sum(alloc.values()),
                           "supply_note": supply_note or "",
                           "my_opening_hold_free": (sup or {}).get("hold_free", 0),
                           "job_positions": [{"job": p["jid"], "proposed_delta": p["delta"],
                                              "evidence": p["evidence"]}
                                             for p in props["demand"]]}, indent=1)
        o = _ask(SYSTEM_SUPPLY_REBUT, user, model, host, cache, tag="sup-rebut")
        try:
            h = max(0, int(round(float((o or {}).get("hold_free_gpus",
                                                     (sup or {}).get("hold_free", 0)) or 0))))
        except (TypeError, ValueError):
            h = (sup or {}).get("hold_free", 0)
        sup = ({"hold_free": h, "evidence": (sup or {}).get("evidence", ""),
                "argument": str((o or {}).get("argument", ""))[:200]} if h else None)

    return {"demand": revised, "supply": sup, "opening": props, "debated": True}
