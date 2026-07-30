"""The explicit GPU market (elevated research plan §6) — a new allocator arm.

Until now the live simulator (`two_sided_sim`) never ran a market. Its bid entered exactly
once, at `frozen = {j.jid: sum(j.bid()) for j in active}`, i.e. the marginal-value CURVE was
collapsed to one scalar and used as a tie-break for greedy fill. There was no supply ask and
no clearing condition. `pins/mechanism.py` has a real uniform-price auction, but only the
older sims call it.

This module builds the plan's market properly, as an arm that can be compared paired-by-seed
against `negotiated`, `referee` and the floor:

  demand, per job j and per extra GPU k (§6.1):
      b_(j,k) = alpha*dp_hat_(j,k) + beta*R_SLA_j + gamma*W_wait_j - delta*C_resize_(j,k)
  supply, per unit sold (§6.2):
      a_k     = eta1*Scarcity + eta2*Fragmentation + eta3*Reserve + eta4*ArrivalPressure,
                rising as more of the pool is sold
  clearing (§6.3):
      Q_t = max { q : b_(q) >= a_(q) }        b sorted descending, a ascending

`dp_hat` is the term that made this buildable only after Exp 68: it is the job's ACTUAL
marginal useful progress from the counterfactual progress model (`two_sided_sim._rate`), not
a phase-profile lookup. Every term is normalised to [0,1] and the curve is enforced
non-increasing, per the plan.

HEADROOM LIVES IN THE PRICE, NOT IN AN IDLE BLOCK. The first build of this arm returned
`reserve = free - sold`, treating every unsold unit as held-idle headroom. That collapsed the
cluster (util 30%, 8.6/16 jobs finished): in `two_sided_sim` the reserve blocks best-effort
jobs from reaching their BASE, not just their margin, so an unsold margin unit was starving
queued work. The correct mapping is that the supply side expresses scarcity through the ask
(eta3 raises the price when prod arrivals are pending, so fewer margin units sell) and the
units it declines to sell stay AVAILABLE for other jobs' bases — which is what actually helps
an incoming prod job. `hold_unsold=True` restores the idle-block semantics for comparison.
"""
from __future__ import annotations

import json

from pins.negotiation_protocol import NegotiationOutcome
from pins.two_sided_sim import STARVE_TICKS, _rate, _useful_units

# alpha, beta, gamma, delta — demand weights (plan §6.1)
BID_W = (1.0, 0.5, 0.3, 0.5)
# eta1..eta4 — supply weights (plan §6.2)
ASK_W = (0.5, 0.2, 0.3, 0.2)
SLA_RISK = {"behind": 1.0, "ontrack": 0.4, "ahead": 0.0}


def bid_curve(facts: dict, ctx: dict, env: dict, w=BID_W) -> list[float]:
    """The job's non-increasing marginal-value curve over EXTRA (margin) GPUs."""
    a, b, g, d = w
    base, usable = facts.get("base", 0), facts.get("usable", 0)
    if base <= 0 or usable <= 0:
        return []
    law, kappa = env.get("law", "amdahl"), env.get("kappa", 2.0)
    alpha, norm = env.get("alpha", 0.0), env.get("alpha_norm", "c0")
    ceil_use = base + usable

    def useful(g_):
        return _useful_units(g_, base, _rate(g_, base, ceil_use, alpha, norm, law, kappa))

    gains = [useful(base + k) - useful(base + k - 1) for k in range(1, usable + 1)]
    top = max(gains) if gains else 0.0
    if top <= 0:
        return []
    r_sla = SLA_RISK.get(ctx.get("deadline"), 0.4)
    w_wait = min(1.0, facts.get("waited", 0) / STARVE_TICKS)
    # plan §15/§6.1: C_resize prices the CHANGE from what the job holds now, not the size of
    # the award. Exp 75 found the first version charged `k` — so a job was billed the same to
    # KEEP three margin GPUs as to acquire them, the bid had no notion of stability, and the
    # delta term could not damp thrash at fast ticks (the cooldown had to). With
    # m = margin currently held, C(k) = c_fixed*[k != m] + c1*|k - m|, and the k-th unit
    # carries the increment C(k) - C(k-1): NEGATIVE below m (awarding it avoids a shrink),
    # positive above (acquiring costs). c_fixed/c1 are already fractions of a tick's progress,
    # the same scale as the normalised gain.
    c_fix, c1 = env.get("realloc_cost", 0.0), env.get("resize_c1", 0.0)
    m = max(0, facts.get("held", base) - base)
    curve, prev = [], float("inf")
    for k, gain in enumerate(gains, start=1):
        if k <= m:
            adj = -c1                          # keeping a held GPU avoids a shrink
        elif k == m + 1:
            adj = c_fix + c1                   # first acquisition pays the fixed restart too
        else:
            adj = c1
        v = a * (gain / top) + b * r_sla + g * w_wait - d * min(1.0, max(-1.0, adj))
        v = min(max(v, 0.0), prev)            # plan §6.1: bids must be non-increasing
        curve.append(v)
        prev = v
    return curve


def ask_curve(free: int, env: dict, w=ASK_W) -> list[float]:
    """The supply side's price for the q-th GPU sold — rising, so the pool's last units are
    dearest and the RESERVE emerges as what nobody outbid (plan §6.2)."""
    e1, e2, e3, e4 = w
    total = max(env.get("total_gpus", free), 1)
    scarcity = 1.0 - free / total
    frag = 1.0 / (1.0 + free)
    reserve_p = min(1.0, env.get("incoming_prod", 0) / 4.0)
    arrival_p = min(1.0, env.get("n_waiting", 0) / max(env.get("n_active", 1), 1))
    base = e1 * scarcity + e2 * frag + e3 * reserve_p + e4 * arrival_p
    return [base * (q / max(free, 1)) for q in range(1, free + 1)]


def clear_market(demand, free: int, env: dict, bid_w=BID_W, ask_w=ASK_W):
    """Plan §6.3. Returns (margins, reserve, clearing_price, units_sold)."""
    units: list[tuple[float, str, int]] = []
    for j in demand:
        for k, v in enumerate(bid_curve(getattr(j, "facts", None) or {}, j.ctx, env, bid_w),
                              start=1):
            units.append((v, j.jid, k))
    units.sort(key=lambda u: (-u[0], u[1], u[2]))       # descending value, deterministic ties
    asks = ask_curve(free, env, ask_w)
    margins = {j.jid: 0 for j in demand}
    price, sold = 0.0, 0
    for q, (v, jid, _k) in enumerate(units[:free]):
        if v < asks[q]:                                 # single crossing: b descending, a rising
            break
        margins[jid] += 1
        price, sold = asks[q], sold + 1
    return margins, max(0, free - sold), price, sold


# --------------------------------------------------------------------------- #
#  Auction-output validation — every tick, deterministic, no LLM (2026-07-23)   #
# --------------------------------------------------------------------------- #
# Until now `check_allocation` was wired to the REFEREE arm only (referee.py fast path,
# qcache path, and post-ruling), so the market — the arm that wins utilisation, useful
# utilisation and regret — was the one arm applying its clearing unvalidated. Same rule set,
# same fallback convention: violations -> concede the margins (ceilings drop to base, which
# is the floor's behaviour for that tick) and report agreed=False so two_sided_sim counts it
# in `fb`.
#
# SCOPE, so a 0% rejection rate is not over-read: trace_replay builds every DemandJob with
# forecast_cap=0, so check_allocation's rules 2-4 (base unmet / prod-before-besteffort /
# within-tier envy) are inert there and this reduces to rule 1 feasibility, unknown job ids
# and the negativity check added below. Those are exactly the failures a clearing bug or a
# stale `free` would produce, which is what the validator is for.
VALIDATOR_STATS = {"checked": 0, "rejected": 0}


def take_validator_stats() -> dict:
    """Read-and-reset, same convention as referee.take_shell_stats()."""
    out = dict(VALIDATOR_STATS)
    VALIDATOR_STATS.update(checked=0, rejected=0)
    return out


def validate_clearing(margins: dict[str, int], reserve: int, demand, free: int) -> list[str]:
    """Cheap deterministic check of the auction's own output. [] == apply it."""
    from pins.referee import check_allocation      # local: referee imports the sim, market too
    v = [f"negative award {jid}={n}" for jid, n in sorted(margins.items()) if n < 0]
    if reserve < 0:
        v.append(f"negative reserve {reserve}")
    v += check_allocation(margins, reserve, demand, free)
    VALIDATOR_STATS["checked"] += 1
    if v:
        VALIDATOR_STATS["rejected"] += 1
    return v


def make_policy_market(trace=None, seen=None, bid_w=BID_W, ask_w=ASK_W,
                       hold_unsold: bool = False):
    """A `two_sided_sim` policy: the market decides margins AND (implicitly) the reserve."""
    def policy(demand, supply_ctx, free, waiting=None, env=None, **_):
        env = dict(env or {})
        env.setdefault("n_waiting", len(waiting or []))
        env.setdefault("n_active", max(len(demand), 1))
        margins, unsold, price, sold = clear_market(demand, free, env, bid_w, ask_w)
        reserve = unsold if hold_unsold else 0      # see module docstring
        bad = validate_clearing(margins, reserve, demand, free)
        if bad:
            return {}, 0, NegotiationOutcome(
                margins={}, reserve=0, rounds=0, agreed=False,
                transcript=[{"round": 0, "actor": "validator", "free": free,
                             "violations": bad,
                             "why": f"clearing rejected ({'; '.join(bad)}); margins conceded"}])
        out = NegotiationOutcome(
            margins=margins, reserve=reserve, rounds=0, agreed=True,
            transcript=[{"round": 0, "actor": "market", "price": round(price, 4),
                         "units_sold": sold, "free": free,
                         "unsold": unsold, "reserve": reserve,
                         "why": f"cleared {sold}/{free} GPUs at ask {price:.3f}; "
                                f"{unsold} unsold, {reserve} held idle"}])
        if trace is not None and seen is not None:
            sig = f"market|free={free}|sold={sold}|m={sorted(margins.items())}"
            if sig not in seen:
                seen.add(sig)
                trace.append({"policy": "market", "free_gpus": free, "price": price,
                              "units_sold": sold, "reserve": reserve, "margins": margins})
        return margins, reserve, out
    return policy


# --------------------------------------------------------------------------- #
#  The GATED arm (2026-07-23): validated auction by default, debate on trigger   #
# --------------------------------------------------------------------------- #
# The user's architecture:
#
#   demand bids / supply asks -> deterministic clearing -> automatic validation
#     valid and no contextual trigger  -> apply the auction's allocation
#     contextual trigger, ruling valid -> apply the debate arm's allocation
#     otherwise                        -> fall back
#
# The escalation arm is DEBATE (+ the precedent manual when --manual is set), not the plain
# referee: on the only venue with text, debate beats referee 14/31 vs 11/31 and one call
# 9/31 (Exp 83), and 15/31 with the manual. This measures what that arm costs in SLA when it
# sits on top of the auction instead of replacing it.
#
# The trigger mirrors `referee._hard` (Exp 61) rather than importing it, because that closure
# is built inside make_policy_referee and reaching into it would change the existing
# +hardtrig tier's behaviour. Kept deliberately identical so the two are comparable.
GATE_STATS = {"auction": 0, "escalated": 0, "escalated_invalid": 0}


def take_gate_stats() -> dict:
    out = dict(GATE_STATS)
    GATE_STATS.update(auction=0, escalated=0, escalated_invalid=0)
    return out


def _make_trigger(st):
    """The contextual escalation trigger, shared by every gated arm so they fire identically."""
    def _trigger(demand, free) -> bool:
        jids = frozenset(j.jid for j in demand)
        behind = frozenset(j.jid for j in demand if j.ctx.get("deadline") == "behind")
        bucket = lambda f: 0 if f == 0 else (1 if f <= 2 else 2)
        fire = (st["jids"] is None                       # cold start
                or st["fell"]                            # last escalated ruling was infeasible
                or bucket(free) != bucket(st["free"])    # contested-capacity crossing
                or bool(behind - st["behind"])           # a job newly behind deadline
                or any(j.jid not in st["jids"] and j.ctx.get("tier") == "prod"
                       for j in demand))                 # prod arrival
        st["jids"], st["free"], st["behind"] = jids, free, behind
        return fire
    return _trigger


def make_policy_gated(use_llm=True, model="qwen2.5:14b", cache=None, trace=None, seen=None,
                      bid_w=BID_W, ask_w=ASK_W, debate=True, statement_model=None):
    """Validated auction as the default path; the debate arm only on a contextual trigger."""
    from pins.referee import make_policy_referee
    ref = make_policy_referee(use_llm, model, cache, trace, seen,
                              statement_model=statement_model, debate=debate)
    st = {"jids": None, "free": None, "behind": frozenset(), "fell": False}
    _trigger = _make_trigger(st)

    def policy(demand, supply_ctx, free, waiting=None, env=None, **_):
        env = dict(env or {})
        env.setdefault("n_waiting", len(waiting or []))
        env.setdefault("n_active", max(len(demand), 1))
        margins, unsold, price, sold = clear_market(demand, free, env, bid_w, ask_w)
        bad = validate_clearing(margins, 0, demand, free)
        if bad or _trigger(demand, free):
            GATE_STATS["escalated"] += 1
            m, r, out = ref(demand, supply_ctx, free, waiting=waiting, env=env)
            st["fell"] = not getattr(out, "agreed", True)   # infeasible ruling re-fires next tick
            if st["fell"]:
                GATE_STATS["escalated_invalid"] += 1
            return m, r, out
        GATE_STATS["auction"] += 1
        st["fell"] = False
        return margins, 0, NegotiationOutcome(
            margins=margins, reserve=0, rounds=0, agreed=True,
            transcript=[{"round": 0, "actor": "market", "price": round(price, 4),
                         "units_sold": sold, "free": free, "unsold": unsold,
                         "why": f"routine tick: cleared {sold}/{free} at {price:.3f}, "
                                f"validated, no contextual trigger"}])
    return policy


# --------------------------------------------------------------------------- #
#  The CORRECTED arm (Exp 92): the hard-case winner, run in-sim — boundary test  #
# --------------------------------------------------------------------------- #

CORR_STATS = {"escalated": 0, "llm_calls": 0, "proposals": 0, "changed": 0, "rejected": 0}


def take_corr_stats() -> dict:
    out = dict(CORR_STATS)
    CORR_STATS.clear()                  # also drops the rej_* reason keys added on rejection
    CORR_STATS.update(escalated=0, llm_calls=0, proposals=0, changed=0, rejected=0)
    return out


def make_policy_corrected(use_llm=True, model="qwen2.5:14b", cache=None, trace=None, seen=None,
                          bid_w=BID_W, ask_w=ASK_W, debate=True, authored=None):
    """Exp 92 — run the arm that WINS on text exceptions (Exp 83/88/89) inside the simulator.

    Exp 84 measured a *different* debate: `referee.py` re-decides every job from scratch, with no
    text gate, and cost -12.0* prodSLA. The arm that actually wins the hard-case suite is
    `correction_signed`: the market allocates, and reviewers may only propose *corrections*, and
    only for jobs that carry a note (`if not (j.get("note") or "").strip(): continue`). The supply
    call is gated on a supply note the same way.

    The v2020 replay has no text channel at all — `Job` has no note field and no `ctx` key carries
    one — so `supply_note` is mapped to "" honestly rather than synthesised from the numbers.
    Doing otherwise would manufacture the very channel this experiment exists to show is absent.

    PREDICTION (pre-registered): the trigger still fires at the Exp-87 rate, the escalation runs,
    and it makes ZERO LLM calls and ZERO changes -- allocation identical to `market`, tick for
    tick. That is a strictly stronger boundary claim than Exp 84's -12.0*.

    `authored` (Exp 94) switches the note source from the trace (which has none) to the demand and
    supply AGENTS, which write one sentence each on a fired tick. None keeps this arm byte-identical
    to Exp 92.
    """
    from pins.correction_signed import (apply_signed, debate_signed, gather_signed,
                                        referee_signed)
    st = {"jids": None, "free": None, "behind": frozenset(), "fell": False}
    _trigger = _make_trigger(st)
    stats = CORR_STATS if authored is None else auth_stats(authored)
    prev: dict = {"jobs": {}, "free": None}       # Exp 94: the only cross-tick state, see §3

    def _as_dicts(demand, margins, notes):
        return [{"jid": j.jid, "tier": j.ctx.get("tier", "besteffort"),
                 "deadline": j.ctx.get("deadline", "ontrack"),
                 "requested": margins.get(j.jid, 0),
                 "note": notes.get(j.jid, j.ctx.get("note", ""))}   # the trace never sets ctx.note
                for j in demand]

    def policy(demand, supply_ctx, free, waiting=None, env=None, **_):
        env = dict(env or {})
        env.setdefault("n_waiting", len(waiting or []))
        env.setdefault("n_active", max(len(demand), 1))
        margins, unsold, price, sold = clear_market(demand, free, env, bid_w, ask_w)
        bad = validate_clearing(margins, 0, demand, free)
        why = (f"routine tick: cleared {sold}/{free} at {price:.3f}, validated, no trigger")

        if bad or _trigger(demand, free):
            stats["escalated"] += 1
            n0 = len(cache) if isinstance(cache, dict) else 0
            notes, note = {}, ""
            if authored:
                notes, note = author_notes(demand, margins, free, prev, authored,
                                           use_llm=use_llm, model=model, cache=cache)
                stats["notes"] += len(notes) + (1 if note else 0)
            elif isinstance(supply_ctx, dict):
                note = str(supply_ctx.get("note", ""))
            jobs = _as_dicts(demand, margins, notes)
            props = gather_signed(jobs, note, free, margins, use_llm=use_llm,
                                  model=model, cache=cache)
            if debate:
                props = debate_signed(jobs, note, free, margins, props, use_llm=use_llm,
                                      model=model, cache=cache)
            n_props = len(props.get("demand") or []) + (1 if props.get("supply") else 0)
            stats["proposals"] += n_props
            if n_props:
                dec = referee_signed(margins, props, free, jobs, use_llm=use_llm,
                                     model=model, cache=cache)
                new, viol = apply_signed(margins, dec, free)
                if viol:
                    stats["rejected"] += 1            # market's allocation stands, never repaired
                    for v in viol:   # Exp 94 follow-up (a): over-reach ('change budget',
                        k = "rej_" + v.split(":")[0].replace(" ", "_")   # 'infeasible') or
                        stats[k] = stats.get(k, 0) + 1                   # merely 'cannot fund'?
                elif new != margins:
                    stats["changed"] += 1
                    margins = new
                    why = f"escalated: {n_props} proposal(s) applied"
            else:
                why = "escalated: no job carries text -> no proposal, market stands"
            if isinstance(cache, dict):
                stats["llm_calls"] += max(0, len(cache) - n0)

        prev["jobs"] = {j.jid: _job_state(j, margins) for j in demand}
        prev["free"] = free
        st["fell"] = False
        return margins, 0, NegotiationOutcome(
            margins=margins, reserve=0, rounds=0, agreed=True,
            transcript=[{"round": 0, "actor": authored or "corrected", "price": round(price, 4),
                         "units_sold": sold, "free": free, "unsold": unsold, "why": why}])
    return policy


# --------------------------------------------------------------------------- #
#  The AUTHORED text channel (Exp 94): the agents write the notes themselves     #
# --------------------------------------------------------------------------- #
# Exp 92 showed the escalation is a literal no-op in-sim because the v2020 replay carries no
# operator text. This arm GENERATES the channel rather than fabricating one: when a job's
# situation changes between t-1 and t, its demand agent writes one sentence alongside the
# numeric request, and the supply agent writes one about the pool. The sentences land in the
# same `note` fields the authored hard-case suite uses, so the mechanism validated in Exp 82-89
# is untouched -- only the source of the text changes.
#
# THE PLACEBO IS THE DESIGN. `narrated` says WHAT changed and is forbidden to say why;
# `attributed` says WHY. Same model, same authoring condition, therefore the same call count on
# the same ticks. Without it a win could not be separated from "the LLM was handed cross-tick
# history the market lacks", which would be a finding about BID_W, not about text (decision-rule
# branch 3 of the pre-registration).
#
# WHAT THE AGENTS MAY SEE is pinned here so it cannot be argued after the fact: the previous
# allocation, the previous free-GPU count, the previous deadline bucket and the agent's own
# previous request. Nothing else.
#
# The two prompts are deliberately parallel in structure and length (threat-to-validity 3): same
# role sentence, same "in one sentence, state ..." instruction, same prohibition clause, same
# output contract. Only the WHAT/WHY axis differs.
AUTH_STATS: dict[str, dict] = {}


def auth_stats(mode: str) -> dict:
    return AUTH_STATS.setdefault(mode, {"escalated": 0, "notes": 0, "llm_calls": 0,
                                        "proposals": 0, "changed": 0, "rejected": 0})


def take_auth_stats() -> dict[str, dict]:
    """Read-and-reset, per mode, so `narrated` and `attributed` never pool their counters."""
    out = {k: dict(v) for k, v in AUTH_STATS.items()}
    AUTH_STATS.clear()
    return out


_AUTH_TAIL = ('Do not request GPUs and do not argue for an outcome.\n'
              'Respond with ONLY this JSON object:\n{"note": "<one sentence>"}')

SYSTEM_DEMAND_NOTE = {
    "narrated": (
        "You are the DEMAND-side agent for one job in a GPU scheduler. Your job's situation "
        "changed since the previous tick. In one sentence, state WHAT changed: which of your "
        "quantities moved, and in which direction. Do NOT state, imply or guess any cause, "
        "reason or explanation for the change — report the change itself only. " + _AUTH_TAIL),
    "attributed": (
        "You are the DEMAND-side agent for one job in a GPU scheduler. Your job's situation "
        "changed since the previous tick. In one sentence, state WHY it changed: which "
        "circumstance produced the change, and what it means for the job. DO state the cause, "
        "reason or explanation behind the change — not the change alone. " + _AUTH_TAIL),
}

SYSTEM_SUPPLY_NOTE = {
    "narrated": (
        "You are the SUPPLY-side agent for a GPU cluster. The pool's situation changed since the "
        "previous tick. In one sentence, state WHAT changed: which of the pool's quantities "
        "moved, and in which direction. Do NOT state, imply or guess any cause, reason or "
        "explanation for the change — report the change itself only. " + _AUTH_TAIL),
    "attributed": (
        "You are the SUPPLY-side agent for a GPU cluster. The pool's situation changed since the "
        "previous tick. In one sentence, state WHY it changed: which circumstance produced the "
        "change, and what it means for the pool. DO state the cause, reason or explanation "
        "behind the change — not the change alone. " + _AUTH_TAIL),
}


def _job_state(j, margins: dict) -> dict:
    """The pinned per-job state carried across one tick."""
    facts = getattr(j, "facts", None) or {}
    base = facts.get("base", 0)
    return {"held_margin": max(0, facts.get("held", base) - base),
            "deadline": j.ctx.get("deadline", "ontrack"),
            "requested": margins.get(j.jid, 0)}


def _speaks(now: dict, was: dict | None) -> bool:
    """Exp 94 §3 authoring condition: held margin moved, deadline bucket moved, or it is new."""
    return (was is None
            or now["held_margin"] != was["held_margin"]
            or now["deadline"] != was["deadline"])


def author_notes(demand, margins: dict, free: int, prev: dict, mode: str,
                 use_llm: bool = True, model: str = "qwen2.5:14b", cache=None):
    """One note per job whose situation changed, plus one supply note. -> (job_notes, supply_note).

    Identical across modes in WHO speaks and HOW MANY calls are made; only the system prompt
    differs. `mode` is part of the cache tag because correction._ask keys on (tag, user, model)
    and NOT on the system prompt — without it the two arms would share one another's notes.
    """
    from pins.correction import HOST
    from pins.correction_signed import _ask
    if not use_llm or mode not in SYSTEM_DEMAND_NOTE:
        return {}, ""
    cache = {} if cache is None else cache
    notes: dict[str, str] = {}
    for j in demand:
        now, was = _job_state(j, margins), prev["jobs"].get(j.jid)
        if not _speaks(now, was):
            continue
        user = json.dumps({"my_job": j.jid, "now": now,
                           "previously": was or "(this job is new this tick)",
                           "free_gpus_now": free, "free_gpus_previously": prev["free"]}, indent=1)
        o = _ask(SYSTEM_DEMAND_NOTE[mode], user, model, HOST, cache, tag=f"note-{mode}-dem")
        text = str((o or {}).get("note", "")).strip()[:300]
        if text:
            notes[j.jid] = text
    user = json.dumps({"free_gpus_now": free, "free_gpus_previously": prev["free"],
                       "allocated_now": sum(margins.values()),
                       "allocated_previously": sum(s["requested"]
                                                   for s in prev["jobs"].values()) or None,
                       "jobs_now": sorted(margins), "jobs_previously": sorted(prev["jobs"])},
                      indent=1)
    o = _ask(SYSTEM_SUPPLY_NOTE[mode], user, model, HOST, cache, tag=f"note-{mode}-sup")
    return notes, str((o or {}).get("note", "")).strip()[:300]


def make_policy_authored(mode: str, use_llm=True, model="qwen2.5:14b", cache=None, trace=None,
                         seen=None, bid_w=BID_W, ask_w=ASK_W, debate=True):
    """Exp 94 — the Exp-92 escalation, run on notes the agents wrote (`narrated`/`attributed`)."""
    assert mode in SYSTEM_DEMAND_NOTE, f"unknown authoring mode {mode!r}"
    return make_policy_corrected(use_llm, model, cache, trace, seen, bid_w, ask_w,
                                 debate=debate, authored=mode)


# --------------------------------------------------------------------------- #
#  The COMPOSED arm (Exp 73): LLM sets the reserve, the market clears the rest   #
# --------------------------------------------------------------------------- #
def make_policy_composed(use_llm=False, model="qwen2.5:14b", cache=None, trace=None, seen=None,
                         bid_w=BID_W, ask_w=ASK_W):
    """Each layer does only what Exp 68-72 showed it is good at.

    Exp 70/71: the reasoning layer's one distinctive, reproducible effect is PROD-TIER
    PROTECTION (dprodSLA -8.9*/-6.3*), and it pays for it in regret and utilisation.
    Exp 72: the explicit market wins utilisation, useful utilisation and regret everywhere,
    but moves the prod tier not at all.

    So: the LLM (or its rule fallback) sets ONLY the reserve — how much headroom to withhold
    for incoming prod — and the market clears the REMAINING pool into margins on the plan's
    bid/ask rule. The LLM no longer touches the margin, which is where its regret came from.

    The falsifiable claim: if the referee's protection was real reasoning about arrivals, it
    survives this amputation; if it was margin-hoarding, it disappears and the arm collapses
    onto plain `market`.
    """
    from pins.llm_agent import llm_reserve, reserve_amount

    def policy(demand, supply_ctx, free, waiting=None, env=None, **_):
        env = dict(env or {})
        env.setdefault("n_waiting", len(waiting or []))
        env.setdefault("n_active", max(len(demand), 1))
        rd = llm_reserve(supply_ctx, use_llm=use_llm, model=model, cache=cache)
        reserve = min(reserve_amount(rd["reserve"]), free)
        # the market only ever sees what the supply side did not withhold
        margins, unsold, price, sold = clear_market(demand, free - reserve, env, bid_w, ask_w)
        bad = validate_clearing(margins, reserve, demand, free)
        if bad:
            return {}, 0, NegotiationOutcome(
                margins={}, reserve=0, rounds=1, agreed=False,
                transcript=[{"round": 0, "actor": "validator", "free": free,
                             "violations": bad,
                             "why": f"clearing rejected ({'; '.join(bad)}); margins conceded"}])
        out = NegotiationOutcome(
            margins=margins, reserve=reserve, rounds=1, agreed=True,
            transcript=[{"round": 0, "actor": "supply", "level": rd["reserve"],
                         "why": rd.get("justification", ""), "_source": rd.get("_source")},
                        {"round": 1, "actor": "market", "price": round(price, 4),
                         "units_sold": sold, "free": free - reserve, "unsold": unsold,
                         "why": f"reserve {reserve} withheld by {rd.get('_source')}; "
                                f"market cleared {sold}/{free - reserve} at {price:.3f}"}])
        if trace is not None and seen is not None:
            sig = f"composed|free={free}|r={reserve}|sold={sold}"
            if sig not in seen:
                seen.add(sig)
                trace.append({"policy": "composed", "free_gpus": free, "reserve": reserve,
                              "reserve_level": rd["reserve"], "units_sold": sold,
                              "price": price, "margins": margins})
        return margins, reserve, out
    return policy
