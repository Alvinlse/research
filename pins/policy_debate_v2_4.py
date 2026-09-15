"""Policy debate v2.4: guarded ordering rotation and deterministic fast sizing.

V2.4 is a separate development arm built while the frozen v2.3 sweep runs.  It does not change
v1--v2.3.  Independent Demand and Supply managers propose ordering branches concurrently every
30 simulated minutes, followed by a constrained Referee.  Once per simulated hour, when no service
emergency is present, the clock makes a 30-minute trial eligible.  Demand and Supply independently
vote whether their proposed ordering should receive that trial, and the Referee selects no trial or
one of those requested branches.  Code never picks a branch randomly.  The challenger is retained
only if queue depth does not increase and broad deadline pressure does not materially worsen;
otherwise the incumbent is restored.

Sizing is no longer an LLM output.  A deterministic controller reviews queue and arrival pressure
at the first scheduler invocation at least five simulated minutes after its previous review.  It
enters adaptive at high pressure and returns to as-requested only below a lower threshold, providing
hysteresis.  Actual allocation changes still occur only at ElastiSim work boundaries and remain
subject to the existing resize budget/cooldown and legal job bounds.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pins import policy_debate as v1
from pins import policy_debate_v2 as v2


PROTOCOL_VERSION = "2.4.1"
ORDERING_REVIEW_INTERVAL_S = 1800
SIZING_REVIEW_INTERVAL_S = 300
ADAPTIVE_ENTER_PRESSURE = 0.75
ADAPTIVE_EXIT_PRESSURE = 0.35
ADAPTIVE_EXIT_ARRIVAL_LOAD = 0.50
ORDERING_TRIAL_INTERVAL_S = 3600
ORDERING_TRIAL_DURATION_S = 1800
TRIAL_MAX_DEADLINE_PRESSURE = 0.25
TRIAL_PRESSURE_REGRESSION = 0.10
MIN_TRIAL_QUEUE = 8

OPENING_FIELDS = {"ordering", "request_trial", "objection", "evidence"}
FINAL_FIELDS = {"ordering", "trial_role", "why"}
OPENING_SCHEMA = (
    '{"ordering": "<one role-allowed ordering>", '
    '"request_trial": <true or false>, '
    '"objection": "<specific harm the other branch may miss>", '
    '"evidence": "<sentence citing supplied numeric state>"}'
)

DEMAND_PROMPT = (
    "You are the DEMAND ordering manager. Choose only auction_deadline or auction_priority. Focus "
    "on broad deadline-budget consumption and material production work. One waiting job is not a "
    "deadline emergency. You do not choose sizing or GPU counts; a five-minute deterministic "
    "controller owns sizing. Use request_trial=true only when rotation_eligible_now is true and a "
    "bounded trial of your ordering is justified by the supplied state and trial-learning history. "
    "Treat one outcome as evidence, not a universal rule, and address repeated rollback evidence. "
    "State the service harm the Supply branch may miss. Reply JSON only: " + OPENING_SCHEMA
)
SUPPLY_PROMPT = (
    "You are the SUPPLY ordering manager. Choose only auction_fairness or auction_wait. Focus on "
    "multi-user concentration, accumulated wait, and queue drainage. Concentration is evidence, "
    "not an automatic veto; a deterministic trial manager limits exposure and can roll back. Use "
    "request_trial=true only when rotation_eligible_now is true and a trial of your ordering is "
    "justified by the supplied state and trial-learning history. Treat one outcome as evidence, "
    "not a universal rule, and address repeated rollback evidence. You do not choose sizing or GPU "
    "counts. State the sharing harm the Demand branch may miss. Reply JSON only: " + OPENING_SCHEMA
)
REFEREE_PROMPT = (
    "You are the ordering REFEREE. Choose exactly one ordering from ALLOWED ORDERINGS and address "
    "both objections. Fixed deadline won 14 of the 24 development windows, fairness won four, and "
    "six tied, so do not infer that a concentrated queue automatically needs fairness. Prefer "
    "deadline for broad severe deadline pressure, priority for material production, fairness for "
    "sustained multi-user imbalance with deterioration, and wait for broadly distributed age. "
    "The advocates also vote on a bounded rotation trial. Use the raw prior outcomes and accumulated "
    "per-role trial scorecard to improve later decisions; repeated rollback is negative evidence, "
    "while one result is not a universal rule. The saved incumbent is the ordering executed before "
    "this debate. If you approve a requested trial, choose that advocate's trial_role and its "
    "ordering; the trial manager compares it with the saved incumbent. Choose trial_role only from "
    "ALLOWED TRIAL ROLES; use none unless an eligible, requested trial has evidence. Sizing is "
    "outside your "
    "authority. Reply JSON only: "
    '{"ordering": "<allowed ordering>", "trial_role": "none|demand|supply", '
    '"why": "<one sentence citing decisive state and the rotation decision>"}.'
)


def trial_learning(outcomes: list[dict]) -> dict:
    """Summarise completed trials as compact in-context evidence for all three LLM roles."""
    summary = {
        role: {
            "completed": 0, "accepts": 0, "rollbacks": 0,
            "mean_queue_delta": 0.0, "mean_deadline_pressure_delta": 0.0,
        }
        for role in ("demand", "supply")
    }
    deltas = {role: {"queue": [], "pressure": []} for role in summary}
    for outcome in outcomes:
        trial = outcome.get("trial") or {}
        role = trial.get("role")
        if role not in summary:
            continue
        accepted = outcome.get("action") == "trial_accept"
        summary[role]["completed"] += 1
        summary[role]["accepts"] += int(accepted)
        summary[role]["rollbacks"] += int(not accepted)
        deltas[role]["queue"].append(
            int(outcome.get("end_queue_depth", trial.get("baseline_queue_depth", 0)))
            - int(trial.get("baseline_queue_depth", 0)))
        deltas[role]["pressure"].append(
            float(outcome.get(
                "end_deadline_pressure_fraction",
                trial.get("baseline_deadline_pressure_fraction", 0.0)))
            - float(trial.get("baseline_deadline_pressure_fraction", 0.0)))
    for role, role_deltas in deltas.items():
        if role_deltas["queue"]:
            summary[role]["mean_queue_delta"] = round(
                sum(role_deltas["queue"]) / len(role_deltas["queue"]), 3)
            summary[role]["mean_deadline_pressure_delta"] = round(
                sum(role_deltas["pressure"]) / len(role_deltas["pressure"]), 3)
    return summary


def decision_state(pending, free, ctx: dict, bench) -> dict:
    state = v2.decision_state(pending, free, ctx, bench)
    queue_depth = int(state["queue_depth"])
    state["deadline_pressure_fraction"] = round(
        float(state["deadline_pressure_jobs"]) / max(1, queue_depth), 3)
    state["deterministic_sizing"] = ctx.get("sel_sizing", "adaptive")
    trial = ctx.get("debate_v24_trial")
    last_trial = ctx.get("debate_v24_last_trial_t")
    elapsed = None if last_trial is None else float(ctx.get("now", 0.0)) - float(last_trial)
    state["ordering_trial"] = trial
    outcomes = ctx.get("debate_v24_trial_outcomes", [])
    state["prior_trial_outcomes"] = outcomes[-3:]
    state["trial_learning"] = trial_learning(outcomes)
    state["rotation_eligible_now"] = bool(
        trial is None and elapsed is not None and elapsed >= ORDERING_TRIAL_INTERVAL_S
        and int(state["queue_depth"]) >= MIN_TRIAL_QUEUE
        and int(state["waiting_users"]) >= 2
        and float(state["deadline_pressure_fraction"]) < TRIAL_MAX_DEADLINE_PRESSURE
        and float(state["production_share"]) < 0.10)
    state["rotation_eligible_in_s"] = (
        None if elapsed is None else max(0, round(ORDERING_TRIAL_INTERVAL_S - elapsed)))
    return state


def update_sizing_controller(pending, free, ctx: dict, bench) -> str:
    """Review sizing at five-minute cadence; allocation changes remain work-boundary-only."""
    now = float(ctx.get("now", 0.0))
    last = float(ctx.get("debate_v24_sizing_last", float("-inf")))
    if now - last < SIZING_REVIEW_INTERVAL_S:
        return ctx.get("sel_sizing", "adaptive")
    load = bench._load_fields(pending, ctx)
    pressure = float(load["queue_pressure"])
    arrival = float(load["arrival_load"])
    previous = ctx.get("sel_sizing", "adaptive")
    if pressure >= ADAPTIVE_ENTER_PRESSURE:
        selected = "adaptive"
        reason = "entered adaptive at high queue pressure"
    elif pressure <= ADAPTIVE_EXIT_PRESSURE and arrival <= ADAPTIVE_EXIT_ARRIVAL_LOAD:
        selected = "as_requested"
        reason = "returned to requested sizing below hysteresis floor"
    else:
        selected = previous if previous in ("adaptive", "as_requested") else "adaptive"
        reason = "held sizing inside hysteresis band"
    ctx["debate_v24_sizing_last"] = now
    ctx["sel_sizing"] = selected
    ctx["sizer"] = selected
    active = list(pending) + list(ctx.get("running", []))
    ctx["job_sizing"] = {bench._job_key(job): selected for job in active}
    ctx["debate_v24_sizing_reviews"] = ctx.get("debate_v24_sizing_reviews", 0) + 1
    changed = selected != previous
    ctx["debate_v24_sizing_transitions"] = ctx.get(
        "debate_v24_sizing_transitions", 0) + int(changed)
    ctx.setdefault("debate_v24_sizing_log", []).append({
        "t": int(now), "previous": previous, "selected": selected,
        "queue_pressure": round(pressure, 3), "arrival_load_1h": round(arrival, 3),
        "changed": changed, "reason": reason,
    })
    return selected


def _demand_emergency(state: dict) -> bool:
    return (
        float(state["production_share"]) >= 0.10
        or (
            float(state["deadline_pressure_fraction"]) >= 0.50
            and int(state["waiting_users"]) >= 2
            and float(state["deadline_pressure_user_share"]) <= 0.50))


def manage_rotation(state: dict, demand_ordering: str, supply_ordering: str,
                    referee_ordering: str, trial_role: str, ctx: dict) -> tuple[str, dict]:
    """Execute, evaluate, and roll back bounded ordering-branch trials."""
    now = float(ctx.get("now", 0.0))
    trial = ctx.get("debate_v24_trial")
    incumbent = ctx.get("debate_v24_incumbent_ordering") or referee_ordering
    event = {"action": "referee", "referee_ordering": referee_ordering}

    if _demand_emergency(state):
        if trial:
            event = {"action": "emergency_rollback", "trial": trial}
        incumbent = demand_ordering
        ctx["debate_v24_trial"] = None
        ctx["debate_v24_incumbent_ordering"] = incumbent
        ctx["debate_v24_last_trial_t"] = now
        return incumbent, {**event, "executed": incumbent, "emergency": True}

    if trial and now >= float(trial["end_t"]):
        queue_ok = int(state["queue_depth"]) <= int(trial["baseline_queue_depth"])
        pressure_ok = float(state["deadline_pressure_fraction"]) <= (
            float(trial["baseline_deadline_pressure_fraction"]) + TRIAL_PRESSURE_REGRESSION)
        accepted = queue_ok and pressure_ok
        incumbent = trial["ordering"] if accepted else trial["incumbent"]
        event = {
            "action": "trial_accept" if accepted else "trial_rollback",
            "trial": trial,
            "end_queue_depth": int(state["queue_depth"]),
            "end_deadline_pressure_fraction": float(state["deadline_pressure_fraction"]),
        }
        ctx.setdefault("debate_v24_trial_outcomes", []).append({**event, "t": int(now)})
        ctx["debate_v24_trial"] = None
        ctx["debate_v24_incumbent_ordering"] = incumbent
        ctx["debate_v24_trial_accepts"] = ctx.get(
            "debate_v24_trial_accepts", 0) + int(accepted)
        ctx["debate_v24_trial_rollbacks"] = ctx.get(
            "debate_v24_trial_rollbacks", 0) + int(not accepted)
        return incumbent, {**event, "executed": incumbent, "emergency": False}

    if trial:
        return trial["ordering"], {
            "action": "trial_continue", "trial": trial,
            "executed": trial["ordering"], "emergency": False}

    last_trial = ctx.get("debate_v24_last_trial_t")
    if last_trial is None:
        incumbent = referee_ordering
        ctx["debate_v24_incumbent_ordering"] = incumbent
        ctx["debate_v24_last_trial_t"] = now
        return incumbent, {**event, "executed": incumbent, "emergency": False}

    safe = (
        now - float(last_trial) >= ORDERING_TRIAL_INTERVAL_S
        and int(state["queue_depth"]) >= MIN_TRIAL_QUEUE
        and int(state["waiting_users"]) >= 2
        and float(state["deadline_pressure_fraction"]) < TRIAL_MAX_DEADLINE_PRESSURE
        and float(state["production_share"]) < 0.10)
    if safe and trial_role in ("demand", "supply"):
        challenger = demand_ordering if trial_role == "demand" else supply_ordering
        if challenger == incumbent:
            return incumbent, {
                "action": "trial_same_as_incumbent", "trial_role": trial_role,
                "executed": incumbent, "emergency": False}
        trial = {
            "role": trial_role,
            "ordering": challenger,
            "incumbent": incumbent,
            "start_t": int(now),
            "end_t": int(now + ORDERING_TRIAL_DURATION_S),
            "baseline_queue_depth": int(state["queue_depth"]),
            "baseline_deadline_pressure_fraction": float(
                state["deadline_pressure_fraction"]),
        }
        ctx["debate_v24_trial"] = trial
        ctx["debate_v24_last_trial_t"] = now
        ctx["debate_v24_trials"] = ctx.get("debate_v24_trials", 0) + 1
        return challenger, {
            "action": "trial_start", "trial": trial,
            "executed": challenger, "emergency": False}
    # No trial started: the ordinary Referee decision becomes the new incumbent. This assignment
    # must happen after challenger comparison; doing it before the comparison caused every
    # Referee-approved trial whose ordering matched its advocate to collapse into
    # `trial_same_as_incumbent`, even when the pre-debate incumbent was the opposing branch.
    incumbent = referee_ordering
    ctx["debate_v24_incumbent_ordering"] = incumbent
    return incumbent, {
        **event,
        "action": "trial_not_selected" if safe else event["action"],
        "trial_role": trial_role,
        "executed": incumbent,
        "emergency": False,
    }


def validate_opening(raw: Any, role: str, ctx: dict, bench) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "opening is not a JSON object"
    if set(raw) != OPENING_FIELDS:
        return None, "opening fields do not match the ordering-only schema"
    ordering = raw.get("ordering")
    if ordering not in v2.ROLE_ORDERINGS[role] or ordering not in bench._family_orderings(ctx):
        return None, f"{role} ordering is outside its role"
    if not isinstance(raw.get("request_trial"), bool):
        return None, "request_trial must be boolean"
    objection, evidence = v1._text(raw.get("objection")), v1._text(raw.get("evidence"))
    if not objection or not evidence:
        return None, "opening requires objection and evidence"
    return {
        "ordering": ordering, "request_trial": raw["request_trial"],
        "objection": objection, "evidence": evidence,
    }, None


def validate_final(raw: Any, candidates: set[str],
                   allowed_trial_roles: set[str]) -> tuple[
                       tuple[str, str] | None, str | None]:
    if not isinstance(raw, dict) or set(raw) != FINAL_FIELDS:
        return None, "referee fields do not match the ordering-only schema"
    if raw.get("ordering") not in candidates:
        return None, "referee ordering is outside the advocate candidate set"
    if raw.get("trial_role") not in allowed_trial_roles:
        return None, "referee trial_role was not requested by that advocate"
    if not v1._text(raw.get("why")):
        return None, "missing referee reason"
    return (raw["ordering"], raw["trial_role"]), None


def parallel_openings(pending, free, ctx: dict, bench, state: dict) -> tuple[Any, Any, float]:
    from pins.correction import _ask as ask_model
    requests = {
        "demand": (DEMAND_PROMPT, bench._packet_policy_demand(pending, ctx),
                   "es-policy-debate-v2-4-demand"),
        "supply": (SUPPLY_PROMPT, bench._packet_policy_supply(pending, free, ctx),
                   "es-policy-debate-v2-4-supply"),
    }

    def invoke(role: str) -> tuple[Any, list[dict]]:
        system, packet, tag = requests[role]
        user = packet + "\n\nV2.4 ORDERING STATE:\n" + json.dumps(state, sort_keys=True)
        audit: list[dict] = []
        answer = ask_model(
            system, user, ctx["model"], ctx["host"], ctx["cache"], tag,
            num_predict=ctx.get("num_predict", 400),
            temperature=ctx.get("temperature", 0), seed=ctx.get("llm_seed"), audit=audit)
        return answer, audit

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="debate-v24") as executor:
        futures = {role: executor.submit(invoke, role) for role in ("demand", "supply")}
        results = {role: futures[role].result() for role in ("demand", "supply")}
    for role in ("demand", "supply"):
        ctx.setdefault("llm_audit", []).extend(results[role][1])
    ctx["calls"] += 2
    return results["demand"][0], results["supply"][0], round(
        time.monotonic() - started, 6)


def arm_policy_debate_v2_4(pending, free, ctx) -> None:
    """Thirty-minute ordering debate plus five-minute deterministic sizing review."""
    from pins import elastisim_bench as bench

    # V2.4 is a fixed protocol: callers cannot accidentally turn its ordering debate into the
    # five-minute sizing cadence (or silently stretch it beyond thirty minutes).
    ctx["sel_every"] = ORDERING_REVIEW_INTERVAL_S
    update_sizing_controller(pending, free, ctx, bench)
    started_total = bench._policy_begin(pending, ctx)
    if started_total is None:
        bench._policy_execute(pending, free, ctx)
        return
    state = decision_state(pending, free, ctx, bench)
    demand_raw, supply_raw, opening_wall_s = parallel_openings(
        pending, free, ctx, bench, state)
    demand, demand_error = validate_opening(demand_raw, "demand", ctx, bench)
    supply, supply_error = validate_opening(supply_raw, "supply", ctx, bench)
    rejected = []
    if demand_error:
        rejected.append({"role": "demand", "reason": demand_error})
    if supply_error:
        rejected.append({"role": "supply", "reason": supply_error})

    final_raw = None
    final_error = None
    referee_ordering = None
    trial_role = "none"
    rotation = None
    if demand and supply:
        candidates = {demand["ordering"], supply["ordering"]}
        allowed_trial_roles = {"none"} | {
            role for role, advocate in (("demand", demand), ("supply", supply))
            if advocate["request_trial"] and state["rotation_eligible_now"]}
        packet = (
            bench._packet_policy(pending, free, ctx) +
            "\n\nV2.4 ORDERING STATE:\n" + json.dumps(state, sort_keys=True) +
            "\n\nDEMAND:\n" + json.dumps(demand, sort_keys=True) +
            "\n\nSUPPLY:\n" + json.dumps(supply, sort_keys=True) +
            "\n\nALLOWED ORDERINGS:\n" + json.dumps(sorted(candidates)) +
            "\n\nALLOWED TRIAL ROLES:\n" + json.dumps(sorted(allowed_trial_roles)))
        final_raw = v1._ask(
            ctx, REFEREE_PROMPT, packet, "es-policy-debate-v2-4-referee")
        decision, final_error = validate_final(final_raw, candidates, allowed_trial_roles)
        if decision is not None:
            referee_ordering, trial_role = decision
        if final_error:
            rejected.append({"role": "referee", "reason": final_error})
    else:
        final_error = "both ordering objections are required"

    if referee_ordering is not None:
        ordering, rotation = manage_rotation(
            state, demand["ordering"], supply["ordering"], referee_ordering,
            trial_role, ctx)
        answer = {
            "ordering": ordering,
            "default_sizing": ctx["sel_sizing"],
            "job_sizing": {},
            "why": f"v2.4 {rotation['action']}; referee proposed {referee_ordering}",
        }
    else:
        answer = None
    bench._policy_record(pending, ctx, started_total, answer, "policy_debate_v2_4")
    ratification = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_state": state,
        "opening_calls_parallel": True,
        "opening_round_wall_s": opening_wall_s,
        "openings": {"demand": demand, "supply": supply},
        "referee": final_raw,
        "referee_ordering": referee_ordering,
        "referee_trial_role": trial_role,
        "rotation": rotation,
        "deterministic_sizing": ctx.get("sel_sizing"),
        "final_valid": referee_ordering is not None,
        "fallback_used": referee_ordering is None,
        "rejected_outputs": rejected,
    }
    ctx["sel_log"][-1]["ratification"] = ratification
    ctx["debate_epochs"] = ctx.get("debate_epochs", 0) + 1
    ctx["debate_referee_calls"] = ctx.get("debate_referee_calls", 0) + int(
        demand is not None and supply is not None)
    ctx["debate_rejected_outputs"] = ctx.get("debate_rejected_outputs", 0) + len(rejected)
    ctx["debate_v24_epochs"] = ctx.get("debate_v24_epochs", 0) + 1
    bench._policy_execute(pending, free, ctx)
