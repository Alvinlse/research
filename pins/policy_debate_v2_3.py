"""Policy debate v2.3: calibrated, persistent objections with parallel openings.

This arm was specified after the completed 24-window v1 training audit.  It is therefore an
in-sample development arm, not confirmatory evidence.  V2.3 keeps v2's disjoint advocate roles,
component-constrained Referee, and strict deterministic floor, but replaces v2.2's unsafe
``majority user => fairness`` veto with a two-stage escalation:

* Demand escalates for material production or severe deadline pressure distributed across users.
* Supply concentration is initially evidence only.  It escalates after two consecutive selection
  epochs in which the queue grows under a service ordering, at least two users are waiting, one
  user owns a majority of jobs or wait, and no Demand emergency is active.
* An escalated objection remains binding until the backlog drains; a Demand emergency can pre-empt
  a Supply episode.  The Referee continues to decide sizing from the advocates' proposed actions.

Demand and Supply openings are independent and are issued concurrently.  The Referee is called
only after both complete, preserving the three-call valid path while reducing opening-round latency.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pins import policy_debate as v1
from pins import policy_debate_v2 as v2


PROTOCOL_VERSION = "2.3"
MIN_QUEUE = 8
PRODUCTION_SHARE = 0.10
DEADLINE_PRESSURE_FRACTION = 0.50
DEADLINE_PRESSURE_USER_SHARE_MAX = 0.50
SUPPLY_CONCENTRATION_SHARE = 0.50
SUPPLY_GROWTH_OBSERVATIONS = 2

OPENING_SCHEMA = (
    '{"ordering": "<one role-allowed ordering>", '
    '"default_sizing": "<one role-allowed sizing>", '
    '"objection": "<specific harm the other side may miss>", '
    '"evidence": "<sentence citing supplied numeric state>"}'
)

DEMAND_PROMPT = (
    "You are the DEMAND advocate for queued jobs. Independently defend service urgency. Choose "
    "ordering only from auction_deadline or auction_priority and sizing only from as_requested or "
    "adaptive. A deadline emergency requires at least half the queue to have consumed half its "
    "deadline budget, with those threatened jobs distributed across users; one late job is not an "
    "emergency. Production share of at least 0.10 is material. When escalation_role is demand, "
    "your ordering is binding until the backlog drains, so distinguish deadline from production "
    "priority carefully. State the concrete service harm Supply may miss. Never invent runtime, "
    "future arrivals, allocations, or GPU counts. Reply JSON only: " + OPENING_SCHEMA
)

SUPPLY_PROMPT = (
    "You are the SUPPLY advocate for shared cluster capacity. Independently defend user balance, "
    "queue drainage, and stable sizing. Choose ordering only from auction_fairness or auction_wait "
    "and sizing only from adaptive or rcon. User concentration alone is evidence, not a veto. A "
    "binding Supply escalation exists only after code observes two consecutive queue-growth epochs "
    "under a service ordering, at least two waiting users, majority job/wait concentration, and no "
    "Demand emergency. When escalation_role is supply, you MUST choose auction_fairness. Otherwise "
    "choose fairness for sustained concentration and wait for broadly distributed age. Adaptive is "
    "the ordinary contention response; rcon needs resize-instability evidence. State the concrete "
    "harm Demand may miss. Never invent runtime, future arrivals, allocations, or GPU counts. Reply "
    "JSON only: " + OPENING_SCHEMA
)

REFEREE_PROMPT = (
    "You are the REFEREE for two deliberately different GPU scheduling objections. Select exactly "
    "one ALLOWED CANDIDATE. Decide ordering before sizing and address both objections. The completed "
    "training audit found fixed deadline better in 14 windows, fairness better in four, and six "
    "ties, so concentration alone is not evidence enough to override service ordering. Code has "
    "already applied the frozen escalation thresholds in ESCALATION. If required_ordering is set, "
    "it is binding. Otherwise prefer deadline for broad severe deadline consumption, priority for "
    "material production, fairness for sustained multi-user concentration with queue deterioration, "
    "and wait only for broadly distributed age without stronger evidence. Decide sizing separately: "
    "adaptive is the contention default, as_requested requires room, and rcon requires instability "
    "evidence. job_sizing must be empty; code computes GPU counts. Reply JSON only: "
    '{"ordering": "<allowed ordering>", "default_sizing": "<allowed sizing>", '
    '"job_sizing": {}, "why": "<sentence resolving both objections with supplied evidence>"}.'
)


def decision_state(pending, free, ctx: dict, bench) -> dict:
    """Add cross-epoch trends and calibrated evidence classes to v2's online state."""
    state = v2.decision_state(pending, free, ctx, bench)
    queue_depth = int(state["queue_depth"])
    previous = ctx.get("debate_v23_previous_queue_depth")
    queue_delta = 0 if previous is None else queue_depth - int(previous)
    growth_streak = (
        int(ctx.get("debate_v23_queue_growth_streak", 0)) + 1 if queue_delta > 0 else 0)
    ctx["debate_v23_previous_queue_depth"] = queue_depth
    ctx["debate_v23_queue_growth_streak"] = growth_streak
    pressure_fraction = float(state["deadline_pressure_jobs"]) / max(1, queue_depth)
    state.update({
        "queue_delta_previous_epoch": queue_delta,
        "queue_growth_streak": growth_streak,
        "deadline_pressure_fraction": round(pressure_fraction, 3),
        "training_fixed_policy_wins": {"deadline": 14, "fairness": 4, "ties": 6},
    })
    return state


def update_escalation(state: dict, ctx: dict) -> dict:
    """Apply the frozen two-stage objection escalation without outcome-only information."""
    queue_depth = int(state["queue_depth"])
    previous_role = ctx.get("debate_v23_escalation_role")
    pressure_fraction = float(state["deadline_pressure_fraction"])
    pressure_user_share = float(state["deadline_pressure_user_share"])
    production_share = float(state["production_share"])
    concentration = max(
        float(state["top_user_job_share"]), float(state["top_user_wait_share"]))

    production_emergency = (
        queue_depth >= MIN_QUEUE and production_share >= PRODUCTION_SHARE)
    distributed_deadline_emergency = (
        queue_depth >= MIN_QUEUE
        and int(state["waiting_users"]) >= 2
        and pressure_fraction >= DEADLINE_PRESSURE_FRACTION
        and pressure_user_share <= DEADLINE_PRESSURE_USER_SHARE_MAX)
    demand_emergency = production_emergency or distributed_deadline_emergency

    supply_observation = (
        queue_depth >= MIN_QUEUE
        and int(state["waiting_users"]) >= 2
        and concentration >= SUPPLY_CONCENTRATION_SHARE
        and production_share < PRODUCTION_SHARE
        and not distributed_deadline_emergency
        and ctx.get("sel_ordering") in v2.ROLE_ORDERINGS["demand"]
        and int(state["queue_delta_previous_epoch"]) > 0)
    observation_streak = (
        int(ctx.get("debate_v23_supply_observation_streak", 0)) + 1
        if supply_observation else 0)
    ctx["debate_v23_supply_observation_streak"] = observation_streak

    released = bool(previous_role) and queue_depth < MIN_QUEUE
    role = None if released else previous_role
    if demand_emergency:
        role = "demand"
    elif role is None and observation_streak >= SUPPLY_GROWTH_OBSERVATIONS:
        role = "supply"
    triggered = role is not None and role != previous_role
    ctx["debate_v23_escalation_role"] = role

    reason = "inactive"
    if role == "demand":
        reason = (
            "material production pressure" if production_emergency else
            "severe user-distributed deadline pressure")
    elif role == "supply":
        reason = "sustained majority-user concentration with queue deterioration"
    state.update({
        "demand_production_emergency": production_emergency,
        "demand_distributed_deadline_emergency": distributed_deadline_emergency,
        "supply_escalation_observation": supply_observation,
        "supply_escalation_observation_streak": observation_streak,
        "escalation_triggered": triggered,
        "escalation_role": role,
        "escalation_reason": reason,
    })
    return {
        "triggered": triggered,
        "previous_role": previous_role,
        "released": released,
        "active": role is not None,
        "role": role,
        "reason": reason,
        "required_ordering": None,
    }


def _packet(packet: str, role: str, state: dict) -> str:
    return (
        packet + "\n\nROLE-ALLOWED ORDERINGS: " + ", ".join(v2.ROLE_ORDERINGS[role]) +
        "\nROLE-ALLOWED SIZINGS: " + ", ".join(v2.ROLE_SIZINGS[role]) +
        "\nCALIBRATED V2.3 STATE (computed by code):\n" +
        json.dumps(state, sort_keys=True))


def parallel_openings(pending, free, ctx: dict, bench, state: dict) -> tuple[Any, Any, float]:
    """Issue independent openings concurrently and merge audit records deterministically."""
    from pins.correction import _ask as ask_model

    requests = {
        "demand": (
            DEMAND_PROMPT,
            _packet(bench._packet_policy_demand(pending, ctx), "demand", state),
            "es-policy-debate-v2-3-demand",
        ),
        "supply": (
            SUPPLY_PROMPT,
            _packet(bench._packet_policy_supply(pending, free, ctx), "supply", state),
            "es-policy-debate-v2-3-supply",
        ),
    }

    def invoke(role: str) -> tuple[Any, list[dict]]:
        system, user, tag = requests[role]
        audit: list[dict] = []
        answer = ask_model(
            system, user, ctx["model"], ctx["host"], ctx["cache"], tag,
            num_predict=ctx.get("num_predict", 400),
            temperature=ctx.get("temperature", 0), seed=ctx.get("llm_seed"), audit=audit)
        return answer, audit

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="debate-v23") as executor:
        futures = {role: executor.submit(invoke, role) for role in ("demand", "supply")}
        results = {role: futures[role].result() for role in ("demand", "supply")}
    wall_s = round(time.monotonic() - started, 6)
    for role in ("demand", "supply"):
        ctx.setdefault("llm_audit", []).extend(results[role][1])
    ctx["calls"] += 2
    return results["demand"][0], results["supply"][0], wall_s


def arm_policy_debate_v2_3(pending, free, ctx) -> None:
    """Parallel independent objections -> mandatory component-constrained Referee."""
    from pins import elastisim_bench as bench

    started_total = bench._policy_begin(pending, ctx)
    if started_total is None:
        bench._policy_execute(pending, free, ctx)
        return

    state = decision_state(pending, free, ctx, bench)
    escalation = update_escalation(state, ctx)
    demand_raw, supply_raw, opening_wall_s = parallel_openings(
        pending, free, ctx, bench, state)
    demand, demand_error = v2.validate_opening(demand_raw, "demand", ctx, bench)
    supply, supply_error = v2.validate_opening(supply_raw, "supply", ctx, bench)
    if supply is not None and escalation["role"] == "supply" \
            and supply["ordering"] != "auction_fairness":
        supply = None
        supply_error = "active Supply escalation requires ordering=auction_fairness"

    guarded = demand if escalation["role"] == "demand" else supply
    if escalation["active"]:
        escalation["required_ordering"] = (
            guarded["ordering"] if guarded is not None else
            (ctx.get("fallback_ordering") or "auction_deadline")
            if escalation["role"] == "demand" else "auction_fairness")
        state["escalation_required_ordering"] = escalation["required_ordering"]

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
        candidates = v2.candidate_records(
            demand, supply, required_ordering=escalation["required_ordering"])
        candidate_pairs = {(x["ordering"], x["default_sizing"]) for x in candidates}
        referee_called = True
        referee_packet = (
            bench._packet_policy(pending, free, ctx) +
            "\n\nCALIBRATED V2.3 STATE:\n" + json.dumps(state, sort_keys=True) +
            "\n\nDEMAND OBJECTION:\n" + json.dumps(demand, sort_keys=True) +
            "\n\nSUPPLY OBJECTION:\n" + json.dumps(supply, sort_keys=True) +
            "\n\nESCALATION:\n" + json.dumps(escalation, sort_keys=True) +
            "\n\nALLOWED CANDIDATES:\n" + json.dumps(candidates, sort_keys=True))
        final_raw = v1._ask(
            ctx, REFEREE_PROMPT, referee_packet, "es-policy-debate-v2-3-referee")
        _, final_error = v2.validate_final(
            final_raw, pending, ctx, bench, candidate_pairs)
        if final_error:
            rejected.append({"stage": "referee", "role": "referee", "reason": final_error})
    else:
        final_error = "both independent objections are required"

    final_valid = final_error is None
    guard_answer = None
    if escalation["active"] and not final_valid:
        guard_answer = {
            "ordering": escalation["required_ordering"],
            "default_sizing": ctx.get("fallback_sizing") or "adaptive",
            "job_sizing": {},
            "why": f"machine-enforced v2.3 {escalation['role']} escalation",
        }
    bench._policy_record(
        pending, ctx, started_total,
        final_raw if final_valid else guard_answer, "policy_debate_v2_3")
    selected = {
        "ordering": ctx.get("sel_ordering"),
        "default_sizing": ctx.get("sel_sizing"),
    }
    fallback_used = not final_valid
    fallback_source = None
    if fallback_used:
        fallback_source = (
            f"machine_{escalation['role']}_v23_escalation"
            if guard_answer is not None else
            "configured_deterministic_floor"
            if ctx.get("fallback_ordering") and ctx.get("fallback_sizing") else
            "previous_validated_policy")
    ratification = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_state": state,
        "escalation": escalation,
        "opening_calls_parallel": True,
        "opening_round_wall_s": opening_wall_s,
        "role_orderings": v2.ROLE_ORDERINGS,
        "role_sizings": v2.ROLE_SIZINGS,
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
    ctx["debate_v23_epochs"] = ctx.get("debate_v23_epochs", 0) + 1
    ctx["debate_v23_parallel_opening_rounds"] = ctx.get(
        "debate_v23_parallel_opening_rounds", 0) + 1
    ctx["debate_v23_supply_escalation_epochs"] = ctx.get(
        "debate_v23_supply_escalation_epochs", 0) + int(escalation["role"] == "supply")
    ctx["debate_v23_supply_escalation_triggers"] = ctx.get(
        "debate_v23_supply_escalation_triggers", 0) + int(
            escalation["triggered"] and escalation["role"] == "supply")
    ctx["debate_v23_demand_escalation_epochs"] = ctx.get(
        "debate_v23_demand_escalation_epochs", 0) + int(escalation["role"] == "demand")
    ctx["debate_v23_demand_escalation_triggers"] = ctx.get(
        "debate_v23_demand_escalation_triggers", 0) + int(
            escalation["triggered"] and escalation["role"] == "demand")
    bench._policy_execute(pending, free, ctx)
