"""Rejected v2.2 development prototype for cross-window policy debate.

Version 1 remains frozen in :mod:`pins.policy_debate` for the incomplete, pre-registered
training sweep.  This version is an exploratory successor derived from the first eight complete
training-window pairs.  It fixes the observed role collapse without using evaluation-only state:

* Demand must defend a service-side ordering (deadline or production priority).
* Supply must defend a queue-side ordering (fairness or accumulated wait).
* Both sides independently choose a bounded sizing position.
* A Referee always adjudicates the two ordering positions and may combine either proposed
  ordering with either proposed sizing.  There are at most four code-derived candidates.
* Machine-enforced role guards remember a service or fairness episode until its backlog drains.
  Material production pressure activates Demand; a batch-dominated majority-user burst activates
  Supply.  The active advocate's ordering is binding while the Referee still adjudicates sizing.
* There is no LLM rebuttal round, so a valid epoch always costs exactly three calls rather than
  the old two/four/five-call path.

All decision evidence is observable online.  True duration, warm-up labels, and realised deadline
outcomes remain hidden from the scheduler.  The completed v1 sweep subsequently showed that the
majority-user episode is too broad to serve as a binding cross-window rule.  Keep this separately
named arm for audit and reproduction; do not promote it without replacing that trigger and testing
the replacement on unseen windows (see ``debate_cross_window_review.md``).
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from pins import policy_debate as v1


ROLE_ORDERINGS = {
    "demand": ("auction_deadline", "auction_priority"),
    "supply": ("auction_fairness", "auction_wait"),
}
ROLE_SIZINGS = {
    "demand": ("as_requested", "adaptive"),
    "supply": ("adaptive", "rcon"),
}
ORDERING_EPISODE_MIN_QUEUE = 8
ORDERING_EPISODE_MAJORITY_SHARE = 0.5
ORDERING_EPISODE_PRODUCTION_SHARE = 0.1
ORDERING_EPISODE_RELEASE_QUEUE = 8

OPENING_FIELDS = {"ordering", "default_sizing", "objection", "evidence"}
FINAL_FIELDS = {"ordering", "default_sizing", "job_sizing", "why"}
OPENING_SCHEMA = (
    '{"ordering": "<one role-allowed ordering>", '
    '"default_sizing": "<one role-allowed sizing>", '
    '"objection": "<one sentence describing the harm the other perspective may miss>", '
    '"evidence": "<one sentence citing supplied state>"}'
)

POLICY_DEBATE_V2_DEMAND = (
    "You are the DEMAND advocate for queued jobs. Independently defend service urgency. "
    "Choose ordering only from auction_deadline or auction_priority and sizing only from "
    "as_requested or adaptive. Deadline pressure means a substantial share of deadline budget "
    "has been consumed across the queue; one merely waiting job is not sufficient evidence. "
    "Use deadline_pressure_user_share to distinguish broad deadline pressure from a concentrated "
    "same-user burst. When ordering_episode_role is demand, code detected material production "
    "pressure and your service ordering will be binding until the backlog drains; choose deadline "
    "or priority carefully. Use auction_priority only when production work is materially represented. "
    "State the concrete harm that a capacity/fairness view could overlook. You propose a global "
    "categorical policy, never an allocation or GPU count. Do not invent runtime or arrivals. "
    "Reply with JSON only: " + OPENING_SCHEMA
)

POLICY_DEBATE_V2_SUPPLY = (
    "You are the SUPPLY advocate for shared cluster capacity. Independently defend queue drainage, "
    "user balance, and stable sizing. Choose ordering only from auction_fairness or auction_wait "
    "and sizing only from adaptive or rcon. Use top_user_job_share, top_user_wait_share, "
    "deadline_pressure_user_share, queue trend, and policy tenure. When ordering_episode_role is "
    "supply, you MUST choose auction_fairness: code detected a batch-dominated majority-user burst "
    "and keeps that objection active until the backlog drains below eight jobs. A "
    "concentrated user burst or a "
    "policy held while the queue grows is evidence for auction_fairness; broadly distributed age "
    "without concentration is evidence for auction_wait. Adaptive is the ordinary contention "
    "choice; rcon requires an explicit resize-stability reason. State the concrete starvation or "
    "capacity harm that a deadline/priority view could overlook. You propose a global categorical "
    "policy, never an allocation or GPU count. Do not invent runtime or arrivals. Reply with JSON "
    "only: " + OPENING_SCHEMA
)

POLICY_DEBATE_V2_REFEREE = (
    "You are the REFEREE for two deliberately different GPU scheduling objections. Select one "
    "policy from ALLOWED CANDIDATES exactly; code formed the bounded cross-product of the two "
    "proposed orderings and two proposed sizing actions. Do not reward agreement or compromise for "
    "its own sake. Decide ordering first: favour auction_deadline when deadline-budget consumption "
    "is severe and distributed across users; favour auction_fairness when threatened jobs or wait "
    "are concentrated in a user burst, or when the current ordering has been held while the queue "
    "grew. Use auction_priority only for material production pressure and auction_wait only when "
    "neither deadline nor concentration evidence is decisive. Decide sizing separately: adaptive "
    "is the default under contention, as_requested is justified only with room to honour demand, "
    "and rcon requires resize-instability evidence. Address both objections in the reason. "
    "ORDERING CONSTRAINT in the packet is binding: during an active episode, the selected role's "
    "proposed ordering is the only candidate ordering, although sizing remains for you to decide. "
    "job_sizing must be empty; deterministic code computes all GPU counts. Reply JSON only: "
    '{"ordering": "<allowed ordering>", "default_sizing": "<allowed sizing>", '
    '"job_sizing": {}, "why": "<one sentence naming decisive evidence from both sides>"}.'
)


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def decision_state(pending, free, ctx: dict, bench) -> dict:
    """Return the online evidence that separates deadline from fairness failures."""
    now = float(ctx.get("now", 0.0))
    waits: list[float] = []
    budgets_used: list[float] = []
    user_jobs: Counter[str] = Counter()
    user_wait: Counter[str] = Counter()
    pressure_users: Counter[str] = Counter()
    declared = production = 0
    for job in pending:
        attrs = job.attributes
        user = str(attrs.get("user", ""))
        waited = max(0.0, now - float(job.submit_time))
        deadline = float(attrs.get("deadline_s", now + 1.0))
        budget = max(1.0, deadline - float(job.submit_time))
        used = waited / budget
        waits.append(waited)
        budgets_used.append(used)
        user_jobs[user] += 1
        user_wait[user] += waited
        if used >= 0.5:
            pressure_users[user] += 1
        declared += int(int(attrs.get("req_min", 0)) > 0)
        production += int(attrs.get("tier") == "prod")

    n = len(pending)
    total_wait = sum(user_wait.values())
    pressured = sum(pressure_users.values())
    history = [row for row in ctx.get("sel_history", [])
               if row.get("queue_after") is not None]
    queue_delta = 0
    growth_streak = 0
    if history:
        last = history[-1]
        queue_delta = int(last["queue_after"]) - int(last["queue"])
        for row in reversed(history):
            if int(row["queue_after"]) <= int(row["queue"]):
                break
            growth_streak += 1

    load = bench._load_fields(pending, ctx)
    return {
        "queue_depth": n,
        "free_gpus": len(free),
        "queue_pressure": round(float(load["queue_pressure"]), 3),
        "arrival_load_1h": round(float(load["arrival_load"]), 3),
        "wait_s_p90": round(_quantile(waits, 0.9), 1),
        "deadline_budget_used_p50": round(_quantile(budgets_used, 0.5), 3),
        "deadline_budget_used_p90": round(_quantile(budgets_used, 0.9), 3),
        "deadline_pressure_jobs": pressured,
        "deadline_pressure_user_share": round(
            max(pressure_users.values(), default=0) / max(1, pressured), 3),
        "top_user_job_share": round(max(user_jobs.values(), default=0) / max(1, n), 3),
        "top_user_wait_share": round(
            max(user_wait.values(), default=0.0) / max(1.0, total_wait), 3),
        "waiting_users": len(user_jobs),
        "declared_walltime_share": round(declared / max(1, n), 3),
        "production_share": round(production / max(1, n), 3),
        "queue_delta_last_epoch": queue_delta,
        "queue_growth_streak": growth_streak,
        "current_policy_held_epochs": bench._held_intervals(ctx),
    }


def update_ordering_episode(state: dict, ctx: dict) -> dict:
    """Latch the advocate whose structural objection defines the current contention episode.

    Production work makes service ordering first-class.  Otherwise, a majority-user burst makes
    fairness first-class.  Instantaneous shares fall as other users arrive even while the triggering
    work remains queued, so an episode releases on drainage, not dilution.  The thresholds describe
    the mechanism (material production, majority concentration, non-trivial queue), not a reward.
    """
    previous_role = ctx.get("debate_v2_ordering_episode_role")
    queue_depth = int(state["queue_depth"])
    share = max(float(state["top_user_job_share"]), float(state["top_user_wait_share"]))
    production_share = float(state["production_share"])
    released = bool(previous_role) and queue_depth < ORDERING_EPISODE_RELEASE_QUEUE
    role = None if released else previous_role
    # Service pressure has precedence even if production work arrives after a Supply episode began.
    if (queue_depth >= ORDERING_EPISODE_MIN_QUEUE
            and production_share >= ORDERING_EPISODE_PRODUCTION_SHARE):
        role = "demand"
    elif role is None and queue_depth >= ORDERING_EPISODE_MIN_QUEUE \
            and share >= ORDERING_EPISODE_MAJORITY_SHARE:
        role = "supply"
    triggered = role is not None and role != previous_role
    ctx["debate_v2_ordering_episode_role"] = role
    state.update({
        "ordering_episode_triggered": triggered,
        "ordering_episode_role": role,
        "ordering_episode_release_queue": ORDERING_EPISODE_RELEASE_QUEUE,
        "ordering_episode_reason": (
            "material production pressure latched for Demand" if role == "demand" else
            "batch-dominated majority-user burst latched for Supply" if role == "supply" else
            "inactive"),
    })
    return {
        "triggered": triggered,
        "previous_role": previous_role,
        "released": released,
        "active": role is not None,
        "role": role,
        "required_ordering": None,
    }


def _packet(packet: str, role: str, state: dict) -> str:
    return (
        packet + "\n\nROLE-ALLOWED ORDERINGS: " + ", ".join(ROLE_ORDERINGS[role]) +
        "\nROLE-ALLOWED SIZINGS: " + ", ".join(ROLE_SIZINGS[role]) +
        "\nCROSS-WINDOW DECISION STATE (computed by code):\n" +
        json.dumps(state, sort_keys=True)
    )


def validate_opening(raw: Any, role: str, ctx: dict, bench) -> tuple[dict | None, str | None]:
    if role not in ROLE_ORDERINGS:
        return None, "unknown advocate role"
    if not isinstance(raw, dict):
        return None, "opening is not a JSON object"
    extra = set(raw) - OPENING_FIELDS
    if extra:
        return None, f"unexpected opening fields: {','.join(sorted(extra))}"
    error = v1._validate_pair(raw.get("ordering"), raw.get("default_sizing"), ctx, bench)
    if error:
        return None, error
    if raw["ordering"] not in ROLE_ORDERINGS[role]:
        return None, f"{role} ordering is outside its objection role"
    if raw["default_sizing"] not in ROLE_SIZINGS[role]:
        return None, f"{role} sizing is outside its objection role"
    objection = v1._text(raw.get("objection"))
    evidence = v1._text(raw.get("evidence"))
    if not objection:
        return None, "missing objection"
    if not evidence:
        return None, "missing evidence"
    return {
        "ordering": raw["ordering"],
        "default_sizing": raw["default_sizing"],
        "objection": objection,
        "evidence": evidence,
    }, None


def candidate_records(demand: dict, supply: dict,
                      required_ordering: str | None = None) -> list[dict]:
    """Bounded component-wise adjudication: at most two orderings x two sizings."""
    orderings = dict.fromkeys((demand["ordering"], supply["ordering"]))
    if required_ordering is not None:
        orderings = {required_ordering: None}
    sizings = dict.fromkeys((demand["default_sizing"], supply["default_sizing"]))
    return [
        {"ordering": ordering, "default_sizing": sizing}
        for ordering in orderings
        for sizing in sizings
    ]


def validate_final(raw: Any, pending, ctx: dict, bench,
                   candidates: set[tuple[str, str]]) -> tuple[
                       tuple[str, str] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "referee answer is not a JSON object"
    extra = set(raw) - FINAL_FIELDS
    if extra:
        return None, f"unexpected referee fields: {','.join(sorted(extra))}"
    error = v1._validate_pair(raw.get("ordering"), raw.get("default_sizing"), ctx, bench)
    if error:
        return None, error
    pair = bench._policy_answer(raw, ctx)
    if pair not in candidates:
        return None, "referee policy is outside the component-wise candidate set"
    if raw.get("job_sizing", {}) != {}:
        return None, "referee job_sizing must be empty"
    if not v1._text(raw.get("why")):
        return None, "missing referee reason"
    if bench._materialise_job_sizing(raw, pending, ctx) is None:
        return None, "referee policy failed the shared materialisation validator"
    return pair, None


def arm_policy_debate_v2(pending, free, ctx) -> None:
    """Two independent objections -> one mandatory, component-constrained Referee."""
    from pins import elastisim_bench as bench

    started_total = bench._policy_begin(pending, ctx)
    if started_total is None:
        bench._policy_execute(pending, free, ctx)
        return

    state = decision_state(pending, free, ctx, bench)
    ordering_guard = update_ordering_episode(state, ctx)
    demand_raw = v1._ask(
        ctx, POLICY_DEBATE_V2_DEMAND,
        _packet(bench._packet_policy_demand(pending, ctx), "demand", state),
        "es-policy-debate-v2-demand",
    )
    supply_raw = v1._ask(
        ctx, POLICY_DEBATE_V2_SUPPLY,
        _packet(bench._packet_policy_supply(pending, free, ctx), "supply", state),
        "es-policy-debate-v2-supply",
    )
    demand, demand_error = validate_opening(demand_raw, "demand", ctx, bench)
    supply, supply_error = validate_opening(supply_raw, "supply", ctx, bench)
    if (supply is not None and ordering_guard["role"] == "supply"
            and supply["ordering"] != "auction_fairness"):
        supply = None
        supply_error = "active Supply episode requires ordering=auction_fairness"
    guarded_advocate = demand if ordering_guard["role"] == "demand" else supply
    if ordering_guard["active"]:
        ordering_guard["required_ordering"] = (
            guarded_advocate["ordering"] if guarded_advocate is not None else
            (ctx.get("fallback_ordering") or "auction_deadline")
            if ordering_guard["role"] == "demand" else
            "auction_fairness")
        state["ordering_episode_required_ordering"] = ordering_guard["required_ordering"]
    rejected = []
    if demand_error:
        rejected.append({"stage": "opening", "role": "demand", "reason": demand_error})
    if supply_error:
        rejected.append({"stage": "opening", "role": "supply", "reason": supply_error})

    candidates: list[dict] = []
    referee_called = False
    final_raw = None
    final_error = None
    if demand is not None and supply is not None:
        candidates = candidate_records(
            demand, supply, required_ordering=ordering_guard["required_ordering"])
        candidate_pairs = {(row["ordering"], row["default_sizing"]) for row in candidates}
        referee_called = True
        referee_packet = (
            bench._packet_policy(pending, free, ctx) +
            "\n\nCROSS-WINDOW DECISION STATE (computed by code):\n" +
            json.dumps(state, sort_keys=True) +
            "\n\nDEMAND OBJECTION:\n" + json.dumps(demand, sort_keys=True) +
            "\n\nSUPPLY OBJECTION:\n" + json.dumps(supply, sort_keys=True) +
            "\n\nORDERING CONSTRAINT:\n" + json.dumps(ordering_guard, sort_keys=True) +
            "\n\nALLOWED CANDIDATES:\n" + json.dumps(candidates, sort_keys=True)
        )
        final_raw = v1._ask(
            ctx, POLICY_DEBATE_V2_REFEREE, referee_packet,
            "es-policy-debate-v2-referee",
        )
        _, final_error = validate_final(
            final_raw, pending, ctx, bench, candidate_pairs)
        if final_error:
            rejected.append({"stage": "referee", "role": "referee", "reason": final_error})
    else:
        final_error = "both independent objections are required"

    final_valid = final_error is None
    guard_answer = None
    if ordering_guard["active"] and not final_valid:
        guard_answer = {
            "ordering": ordering_guard["required_ordering"],
            "default_sizing": ctx.get("fallback_sizing") or "adaptive",
            "job_sizing": {},
            "why": f"machine-enforced persistent {ordering_guard['role']} episode guard",
        }
    bench._policy_record(
        pending, ctx, started_total,
        final_raw if final_valid else guard_answer, "policy_debate_v2")
    selected = {
        "ordering": ctx.get("sel_ordering"),
        "default_sizing": ctx.get("sel_sizing"),
    }
    fallback_used = not final_valid
    fallback_source = None
    if fallback_used:
        fallback_source = (
            f"machine_{ordering_guard['role']}_ordering_episode_guard"
            if guard_answer is not None else
            "configured_deterministic_floor"
            if ctx.get("fallback_ordering") and ctx.get("fallback_sizing") else
            "previous_validated_policy"
        )
    ratification = {
        "protocol_version": "2.2",
        "decision_state": state,
        "ordering_guard": ordering_guard,
        "role_orderings": ROLE_ORDERINGS,
        "role_sizings": ROLE_SIZINGS,
        "openings": {"demand": demand, "supply": supply},
        "candidates": candidates,
        "referee_called": referee_called,
        "resolution_source": "referee" if final_valid else fallback_source,
        "selected": selected,
        "final_valid": final_valid,
        "fallback_used": fallback_used,
        "fallback_source": fallback_source,
        "rejected_outputs": rejected,
        "residual_risk": ([f"{final_error}; {fallback_source} executed"]
                          if fallback_used else []),
    }
    ctx["sel_log"][-1]["ratification"] = ratification
    ctx["debate_epochs"] = ctx.get("debate_epochs", 0) + 1
    ctx["debate_referee_calls"] = ctx.get("debate_referee_calls", 0) + int(referee_called)
    ctx["debate_rejected_outputs"] = ctx.get("debate_rejected_outputs", 0) + len(rejected)
    ctx["debate_v2_epochs"] = ctx.get("debate_v2_epochs", 0) + 1
    ctx["debate_v2_one_sided_rejections"] = ctx.get(
        "debate_v2_one_sided_rejections", 0) + int(demand is None or supply is None)
    ctx["debate_v2_fairness_guard_epochs"] = ctx.get(
        "debate_v2_fairness_guard_epochs", 0) + int(ordering_guard["role"] == "supply")
    ctx["debate_v2_fairness_guard_triggers"] = ctx.get(
        "debate_v2_fairness_guard_triggers", 0) + int(
            ordering_guard["triggered"] and ordering_guard["role"] == "supply")
    ctx["debate_v2_fairness_guard_floors"] = ctx.get(
        "debate_v2_fairness_guard_floors", 0) + int(
            guard_answer is not None and ordering_guard["role"] == "supply")
    ctx["debate_v2_demand_guard_epochs"] = ctx.get(
        "debate_v2_demand_guard_epochs", 0) + int(ordering_guard["role"] == "demand")
    ctx["debate_v2_demand_guard_triggers"] = ctx.get(
        "debate_v2_demand_guard_triggers", 0) + int(
            ordering_guard["triggered"] and ordering_guard["role"] == "demand")
    ctx["debate_v2_demand_guard_floors"] = ctx.get(
        "debate_v2_demand_guard_floors", 0) + int(
            guard_answer is not None and ordering_guard["role"] == "demand")
    bench._policy_execute(pending, free, ctx)
