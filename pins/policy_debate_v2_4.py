"""Policy debate v2.4.3: phase-correct rotation learning and deterministic fast sizing.

V2.4 is a separate development arm built while the frozen v2.3 sweep runs.  It does not change
v1--v2.3.  Independent Demand and Supply managers propose ordering branches concurrently every
30 simulated minutes, followed by a constrained Referee. Once per simulated hour, when no service
emergency is present, the clock makes a 30-minute trial eligible. Demand and Supply independently
vote whether their proposed ordering should receive that trial, and the Referee chooses only hold,
trial_demand, or trial_supply. Code never picks a branch randomly, and no-trial decisions retain the
incumbent. A successful first phase enters a second 30-minute probation before acceptance.
Completed outcomes report deltas from the phase that actually passed or failed. Two consecutive
rollbacks block that exact incumbent-to-challenger transition for four simulated hours.

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


PROTOCOL_VERSION = "2.4.3"
ORDERING_REVIEW_INTERVAL_S = 1800
SIZING_REVIEW_INTERVAL_S = 300
ADAPTIVE_ENTER_PRESSURE = 0.75
ADAPTIVE_EXIT_PRESSURE = 0.35
ADAPTIVE_EXIT_ARRIVAL_LOAD = 0.50
ORDERING_TRIAL_INTERVAL_S = 3600
ORDERING_TRIAL_DURATION_S = 1800
TRIAL_MAX_DEADLINE_PRESSURE = 0.25
TRIAL_PRESSURE_REGRESSION = 0.05
TRIAL_CONCENTRATION_REGRESSION = 0.02
TRIAL_WAIT_TOLERANCE_S = 300
MIN_TRIAL_QUEUE = 8
TRANSITION_BACKOFF_ROLLBACKS = 2
TRANSITION_BACKOFF_S = 4 * 3600

OPENING_FIELDS = {"ordering", "request_trial", "objection", "evidence"}
FINAL_FIELDS = {"action", "why"}
OPENING_SCHEMA = (
    '{"ordering": "<one role-allowed ordering or abstain>", '
    '"request_trial": <true or false>, '
    '"objection": "<specific harm the other branch may miss>", '
    '"evidence": "<sentence citing supplied numeric state>"}'
)

DEMAND_PROMPT = (
    "You are the DEMAND ordering manager. Choose only auction_deadline, auction_priority, or "
    "abstain. Use abstain with request_trial=false when this state has no defensible Demand-side "
    "candidate; never choose a Supply ordering. Focus "
    "on broad deadline-budget consumption and material production work. One waiting job is not a "
    "deadline emergency. You do not choose sizing or GPU counts; a five-minute deterministic "
    "controller owns sizing. Use request_trial=true only when rotation_eligible_now is true and a "
    "bounded trial of your ordering is justified by the supplied state and trial-learning history. "
    "Treat one outcome as evidence, not a universal rule, and address repeated rollback evidence. "
    "State the service harm the Supply branch may miss. Reply JSON only: " + OPENING_SCHEMA
)
SUPPLY_PROMPT = (
    "You are the SUPPLY ordering manager. Choose only auction_fairness, auction_wait, or abstain. "
    "Use abstain with request_trial=false when this state has no defensible Supply-side candidate; "
    "never choose a Demand ordering. Focus on "
    "multi-user concentration, accumulated wait, and queue drainage. Concentration is evidence, "
    "not an automatic veto; a deterministic trial manager limits exposure and can roll back. Use "
    "request_trial=true only when rotation_eligible_now is true and a trial of your ordering is "
    "justified by the supplied state and trial-learning history. Treat one outcome as evidence, "
    "not a universal rule, and address repeated rollback evidence. You do not choose sizing or GPU "
    "counts. State the sharing harm the Demand branch may miss. Reply JSON only: " + OPENING_SCHEMA
)
REFEREE_PROMPT = (
    "You are the ordering REFEREE. Choose exactly one action from ALLOWED ACTIONS and address both "
    "objections. hold retains the saved incumbent; trial_demand and trial_supply execute the named "
    "advocate's candidate under deterministic trial and rollback control. You never emit an ordering "
    "name. Use raw prior outcomes and the transition scorecard. A rollback is negative evidence "
    "about that exact challenger, never evidence that the incumbent failed; only trial_accept is "
    "positive evidence. A blocked transition is unavailable until its blocked_until_t. Choose a "
    "trial only when that exact action is allowed and evidence beats holding. Sizing is outside "
    "your authority. Reply JSON only: "
    '{"action": "hold|trial_demand|trial_supply", '
    '"why": "<one sentence citing the decisive state, objections, and trial history>"}.'
)


def _phase_deltas(outcome: dict) -> tuple[int, float]:
    """Return deltas from the phase whose checks produced the outcome verdict."""
    checks = outcome.get("checks") or {}
    return int(checks.get("queue_delta", 0)), float(
        checks.get("deadline_pressure_delta", 0.0))


def trial_learning(outcomes: list[dict], now: float | None = None) -> dict:
    """Summarise phase-correct positive evidence and exact-transition rollback history."""
    summary = {
        role: {
            "completed": 0, "accepts": 0, "rollbacks": 0,
            "accepted_evidence_count": 0,
            "mean_accepted_phase_queue_delta": None,
            "mean_accepted_phase_deadline_pressure_delta": None,
        }
        for role in ("demand", "supply")
    }
    accepted_deltas = {role: {"queue": [], "pressure": []} for role in summary}
    for outcome in outcomes:
        trial = outcome.get("trial") or {}
        role = trial.get("role")
        if role not in summary:
            continue
        accepted = outcome.get("action") == "trial_accept"
        summary[role]["completed"] += 1
        summary[role]["accepts"] += int(accepted)
        summary[role]["rollbacks"] += int(not accepted)
        if accepted:
            queue_delta, pressure_delta = _phase_deltas(outcome)
            accepted_deltas[role]["queue"].append(queue_delta)
            accepted_deltas[role]["pressure"].append(pressure_delta)
            summary[role]["accepted_evidence_count"] += 1
    for role, role_deltas in accepted_deltas.items():
        if role_deltas["queue"]:
            summary[role]["mean_accepted_phase_queue_delta"] = round(
                sum(role_deltas["queue"]) / len(role_deltas["queue"]), 3)
            summary[role]["mean_accepted_phase_deadline_pressure_delta"] = round(
                sum(role_deltas["pressure"]) / len(role_deltas["pressure"]), 3)
    transitions: dict[str, dict] = {}
    for outcome in outcomes:
        trial = outcome.get("trial") or {}
        incumbent, challenger = trial.get("incumbent"), trial.get("ordering")
        if not incumbent or not challenger:
            continue
        key = f"{incumbent}->{challenger}"
        item = transitions.setdefault(key, {
            "incumbent": incumbent, "challenger": challenger,
            "completed": 0, "accepts": 0, "rollbacks": 0,
            "consecutive_rollbacks": 0, "last_verdict": None,
            "last_reason": None, "last_phase": None,
            "last_phase_queue_delta": 0,
            "last_phase_deadline_pressure_delta": 0.0,
            "last_outcome_t": None, "blocked_until_t": None,
            "blocked_now": False, "recommendation": "eligible_with_caution",
        })
        accepted = outcome.get("action") == "trial_accept"
        queue_delta, pressure_delta = _phase_deltas(outcome)
        item["completed"] += 1
        item["accepts"] += int(accepted)
        item["rollbacks"] += int(not accepted)
        item["consecutive_rollbacks"] = (
            0 if accepted else item["consecutive_rollbacks"] + 1)
        item["last_verdict"] = "beneficial_challenger" if accepted else "harmful_challenger"
        item["last_reason"] = outcome.get("reason")
        item["last_phase"] = trial.get("phase", "trial")
        item["last_phase_queue_delta"] = queue_delta
        item["last_phase_deadline_pressure_delta"] = round(pressure_delta, 3)
        item["last_outcome_t"] = outcome.get("t")
        item["last_failure_checks"] = None if accepted else outcome.get("checks")
    for item in transitions.values():
        if (item["consecutive_rollbacks"] >= TRANSITION_BACKOFF_ROLLBACKS
                and item["last_outcome_t"] is not None):
            item["blocked_until_t"] = int(item["last_outcome_t"] + TRANSITION_BACKOFF_S)
            item["blocked_now"] = bool(
                now is not None and now < item["blocked_until_t"])
        if item["blocked_now"]:
            item["recommendation"] = "block_harmful_challenger"
        elif item["last_verdict"] == "beneficial_challenger":
            item["recommendation"] = "accepted_positive_evidence"
        elif item["rollbacks"]:
            item["recommendation"] = "negative_challenger_evidence"
    summary["by_transition"] = transitions
    return summary


def transition_status(outcomes: list[dict], incumbent: str, challenger: str,
                      now: float) -> dict:
    """Return the exact-transition evidence and deterministic backoff status."""
    key = f"{incumbent}->{challenger}"
    return trial_learning(outcomes, now)["by_transition"].get(key, {
        "incumbent": incumbent, "challenger": challenger,
        "completed": 0, "accepts": 0, "rollbacks": 0,
        "consecutive_rollbacks": 0, "last_verdict": None,
        "blocked_until_t": None, "blocked_now": False,
        "recommendation": "no_prior_evidence",
    })


def decision_state(pending, free, ctx: dict, bench) -> dict:
    state = v2.decision_state(pending, free, ctx, bench)
    queue_depth = int(state["queue_depth"])
    state["deadline_pressure_fraction"] = round(
        float(state["deadline_pressure_jobs"]) / max(1, queue_depth), 3)
    state["deterministic_sizing"] = ctx.get("sel_sizing", "adaptive")
    state["ordering_incumbent"] = ctx.get(
        "debate_v24_incumbent_ordering", ctx.get("sel_ordering", "auction_deadline"))
    trial = ctx.get("debate_v24_trial")
    last_trial = ctx.get("debate_v24_last_trial_t")
    elapsed = None if last_trial is None else float(ctx.get("now", 0.0)) - float(last_trial)
    state["ordering_trial"] = trial
    outcomes = ctx.get("debate_v24_trial_outcomes", [])
    state["prior_trial_outcomes"] = outcomes[-3:]
    state["trial_learning"] = trial_learning(outcomes, float(ctx.get("now", 0.0)))
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


def _trial_snapshot(state: dict, now: float) -> dict:
    return {
        "t": int(now),
        "queue_depth": int(state["queue_depth"]),
        "queue_delta_last_epoch": int(state.get("queue_delta_last_epoch", 0)),
        "deadline_pressure_fraction": float(state["deadline_pressure_fraction"]),
        "deadline_pressure_jobs": int(state.get("deadline_pressure_jobs", 0)),
        "top_user_wait_share": float(state.get("top_user_wait_share", 0.0)),
        "wait_s_p90": float(state.get("wait_s_p90", 0.0)),
    }


def _trial_checks(trial: dict, state: dict, now: float) -> dict:
    """Role-aware non-regression checks for one trial or probation phase."""
    baseline = trial["phase_baseline"]
    elapsed = max(0.0, now - float(baseline["t"]))
    queue_delta = int(state["queue_depth"]) - int(baseline["queue_depth"])
    prior_delta = int(baseline.get("queue_delta_last_epoch", 0))
    queue_limit = max(0, prior_delta - 1) if prior_delta > 0 else 0
    queue_trend_ok = queue_delta <= queue_limit
    pressure_delta = (
        float(state["deadline_pressure_fraction"])
        - float(baseline["deadline_pressure_fraction"]))
    pressure_ok = pressure_delta <= TRIAL_PRESSURE_REGRESSION
    wait_growth = float(state.get("wait_s_p90", 0.0)) - float(baseline["wait_s_p90"])
    wait_ok = wait_growth <= elapsed + TRIAL_WAIT_TOLERANCE_S

    if trial["role"] == "supply":
        concentration_delta = (
            float(state.get("top_user_wait_share", 0.0))
            - float(baseline["top_user_wait_share"]))
        objective_ok = (
            concentration_delta <= TRIAL_CONCENTRATION_REGRESSION
            and (queue_delta < 0 or concentration_delta < 0))
        objective = {
            "name": "supply_queue_or_concentration",
            "top_user_wait_share_delta": round(concentration_delta, 3),
        }
    else:
        deadline_jobs_delta = (
            int(state.get("deadline_pressure_jobs", 0))
            - int(baseline["deadline_pressure_jobs"]))
        objective_ok = (
            deadline_jobs_delta <= 0
            and (queue_delta < 0 or deadline_jobs_delta < 0 or pressure_delta < 0))
        objective = {
            "name": "demand_deadline_or_queue",
            "deadline_pressure_jobs_delta": deadline_jobs_delta,
        }
    return {
        "passed": bool(queue_trend_ok and pressure_ok and wait_ok and objective_ok),
        "queue_delta": queue_delta,
        "queue_trend_limit": queue_limit,
        "queue_trend_ok": queue_trend_ok,
        "deadline_pressure_delta": round(pressure_delta, 3),
        "deadline_pressure_ok": pressure_ok,
        "wait_s_p90_growth": round(wait_growth),
        "wait_s_p90_limit": round(elapsed + TRIAL_WAIT_TOLERANCE_S),
        "wait_s_p90_ok": wait_ok,
        "objective_ok": objective_ok,
        "objective": objective,
    }


def _complete_trial(trial: dict, state: dict, ctx: dict, accepted: bool,
                    reason: str, checks: dict) -> tuple[str, dict]:
    now = float(ctx.get("now", 0.0))
    incumbent = trial["ordering"] if accepted else trial["incumbent"]
    event = {
        "action": "trial_accept" if accepted else "trial_rollback",
        "reason": reason,
        "trial": trial,
        "checks": checks,
        "end_queue_depth": int(state["queue_depth"]),
        "end_deadline_pressure_fraction": float(state["deadline_pressure_fraction"]),
        "end_top_user_wait_share": float(state.get("top_user_wait_share", 0.0)),
        "end_wait_s_p90": float(state.get("wait_s_p90", 0.0)),
        "executed": incumbent,
        "emergency": reason == "demand_emergency",
    }
    ctx.setdefault("debate_v24_trial_outcomes", []).append({**event, "t": int(now)})
    ctx["debate_v24_trial"] = None
    ctx["debate_v24_incumbent_ordering"] = incumbent
    ctx["debate_v24_last_trial_t"] = now
    ctx["debate_v24_trial_accepts"] = ctx.get(
        "debate_v24_trial_accepts", 0) + int(accepted)
    ctx["debate_v24_trial_rollbacks"] = ctx.get(
        "debate_v24_trial_rollbacks", 0) + int(not accepted)
    return incumbent, event


def settle_expired_trial(state: dict, ctx: dict) -> tuple[str, dict] | None:
    """Settle an expired phase before the next LLM debate so its result is immediately visible."""
    now = float(ctx.get("now", 0.0))
    trial = ctx.get("debate_v24_trial")
    if not trial or now < float(trial["end_t"]):
        return None
    checks = _trial_checks(trial, state, now)
    if _demand_emergency(state) and trial["role"] == "supply":
        return _complete_trial(
            trial, state, ctx, False, "demand_emergency", checks)
    if trial.get("phase", "trial") == "trial" and checks["passed"]:
        # Copy before mutation so the earlier trial_start audit record remains immutable.
        trial = {
            **trial,
            "phase": "probation",
            "phase_baseline": _trial_snapshot(state, now),
            "end_t": int(now + ORDERING_TRIAL_DURATION_S),
        }
        ctx["debate_v24_trial"] = trial
        ctx["debate_v24_trial_probations"] = ctx.get(
            "debate_v24_trial_probations", 0) + 1
        return trial["ordering"], {
            "action": "trial_probation",
            "trial": trial,
            "checks": checks,
            "executed": trial["ordering"],
            "emergency": False,
        }
    accepted = trial.get("phase") == "probation" and checks["passed"]
    reason = "probation_passed" if accepted else (
        "probation_failed" if trial.get("phase") == "probation" else "trial_failed")
    return _complete_trial(trial, state, ctx, accepted, reason, checks)


def manage_rotation(state: dict, demand_ordering: str, supply_ordering: str,
                    referee_action: str, ctx: dict) -> tuple[str, dict]:
    """Hold the incumbent or execute an explicitly approved, bounded branch trial."""
    now = float(ctx.get("now", 0.0))
    incumbent = ctx.get(
        "debate_v24_incumbent_ordering", ctx.get("sel_ordering", "auction_deadline"))
    ctx["debate_v24_incumbent_ordering"] = incumbent
    if _demand_emergency(state):
        trial = ctx.get("debate_v24_trial")
        prior_event = None
        if trial:
            _, prior_event = _complete_trial(
                trial, state, ctx, False, "demand_emergency",
                _trial_checks(trial, state, now))
        incumbent = (
            demand_ordering if demand_ordering != "abstain" else "auction_deadline")
        ctx["debate_v24_trial"] = None
        ctx["debate_v24_incumbent_ordering"] = incumbent
        ctx["debate_v24_last_trial_t"] = now
        return incumbent, {
            "action": "emergency_demand",
            "prior_trial_outcome": prior_event,
            "executed": incumbent,
            "emergency": True,
        }

    settled = settle_expired_trial(state, ctx)
    if settled is not None:
        return settled

    trial = ctx.get("debate_v24_trial")
    if trial:
        return trial["ordering"], {
            "action": "trial_continue", "trial": trial,
            "executed": trial["ordering"], "emergency": False}

    last_trial = ctx.get("debate_v24_last_trial_t")
    if last_trial is None:
        ctx["debate_v24_last_trial_t"] = now
        return incumbent, {
            "action": "hold", "referee_action": referee_action,
            "executed": incumbent, "emergency": False}

    safe = (
        now - float(last_trial) >= ORDERING_TRIAL_INTERVAL_S
        and int(state["queue_depth"]) >= MIN_TRIAL_QUEUE
        and int(state["waiting_users"]) >= 2
        and float(state["deadline_pressure_fraction"]) < TRIAL_MAX_DEADLINE_PRESSURE
        and float(state["production_share"]) < 0.10)
    if safe and referee_action in ("trial_demand", "trial_supply"):
        role = "demand" if referee_action == "trial_demand" else "supply"
        challenger = demand_ordering if role == "demand" else supply_ordering
        if challenger == incumbent:
            return incumbent, {
                "action": "trial_same_as_incumbent", "referee_action": referee_action,
                "executed": incumbent, "emergency": False}
        transition = transition_status(
            ctx.get("debate_v24_trial_outcomes", []), incumbent, challenger, now)
        if transition["blocked_now"]:
            ctx["debate_v24_transition_backoff_holds"] = ctx.get(
                "debate_v24_transition_backoff_holds", 0) + 1
            return incumbent, {
                "action": "transition_backoff_hold",
                "referee_action": referee_action,
                "transition": transition,
                "executed": incumbent,
                "emergency": False,
            }
        baseline = _trial_snapshot(state, now)
        trial = {
            "role": role,
            "ordering": challenger,
            "incumbent": incumbent,
            "phase": "trial",
            "start_t": int(now),
            "end_t": int(now + ORDERING_TRIAL_DURATION_S),
            "baseline_queue_depth": baseline["queue_depth"],
            "baseline_deadline_pressure_fraction": baseline[
                "deadline_pressure_fraction"],
            "baseline_top_user_wait_share": baseline["top_user_wait_share"],
            "baseline_wait_s_p90": baseline["wait_s_p90"],
            "phase_baseline": baseline,
        }
        ctx["debate_v24_trial"] = trial
        ctx["debate_v24_trials"] = ctx.get("debate_v24_trials", 0) + 1
        return challenger, {
            "action": "trial_start", "trial": trial,
            "executed": challenger, "emergency": False}
    return incumbent, {
        "action": "hold",
        "referee_action": referee_action,
        "rotation_eligible": safe,
        "executed": incumbent,
        "emergency": False,
    }


def validate_opening(raw: Any, role: str, ctx: dict, bench) -> tuple[dict | None, str | None]:
    if not isinstance(raw, dict):
        return None, "opening is not a JSON object"
    if set(raw) != OPENING_FIELDS:
        return None, "opening fields do not match the ordering-only schema"
    ordering = raw.get("ordering")
    if ordering != "abstain" and (
            ordering not in v2.ROLE_ORDERINGS[role]
            or ordering not in bench._family_orderings(ctx)):
        return None, f"{role} ordering is outside its role"
    if not isinstance(raw.get("request_trial"), bool):
        return None, "request_trial must be boolean"
    if ordering == "abstain" and raw["request_trial"]:
        return None, "an abstaining advocate cannot request a trial"
    objection, evidence = v1._text(raw.get("objection")), v1._text(raw.get("evidence"))
    if not objection or not evidence:
        return None, "opening requires objection and evidence"
    return {
        "ordering": ordering, "request_trial": raw["request_trial"],
        "objection": objection, "evidence": evidence,
    }, None


def validate_final(raw: Any, allowed_actions: set[str]) -> tuple[str | None, str | None]:
    if not isinstance(raw, dict) or set(raw) != FINAL_FIELDS:
        return None, "referee fields do not match the hold-or-trial schema"
    if raw.get("action") not in allowed_actions:
        return None, "referee action is not eligible and requested"
    if not v1._text(raw.get("why")):
        return None, "missing referee reason"
    return raw["action"], None


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
    """Thirty-minute hold-or-trial debate plus five-minute deterministic sizing."""
    from pins import elastisim_bench as bench

    ctx["sel_every"] = ORDERING_REVIEW_INTERVAL_S
    update_sizing_controller(pending, free, ctx, bench)
    started_total = bench._policy_begin(pending, ctx)
    if started_total is None:
        bench._policy_execute(pending, free, ctx)
        return
    ctx.setdefault(
        "debate_v24_incumbent_ordering",
        ctx.get("sel_ordering", ctx.get("fallback_ordering", "auction_deadline")))
    state = decision_state(pending, free, ctx, bench)
    # Settle before inference, then rebuild state so the outcome reaches this debate.
    settled_before_debate = settle_expired_trial(state, ctx)
    if settled_before_debate is not None:
        state = decision_state(pending, free, ctx, bench)
        state["just_settled_trial"] = settled_before_debate[1]

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
    referee_action = None
    rotation = None
    if demand and supply:
        incumbent = ctx["debate_v24_incumbent_ordering"]
        blocked_trial_requests = {}
        for role, advocate in (("demand", demand), ("supply", supply)):
            if (advocate["request_trial"] and advocate["ordering"] != "abstain"
                    and advocate["ordering"] != incumbent
                    and state["rotation_eligible_now"]):
                transition = transition_status(
                    ctx.get("debate_v24_trial_outcomes", []), incumbent,
                    advocate["ordering"], float(ctx.get("now", 0.0)))
                if transition["blocked_now"]:
                    blocked_trial_requests[role] = transition
        allowed_actions = {"hold"} | {
            f"trial_{role}"
            for role, advocate in (("demand", demand), ("supply", supply))
            if (
                advocate["request_trial"]
                and advocate["ordering"] != "abstain"
                and advocate["ordering"] != incumbent
                and state["rotation_eligible_now"]
                and role not in blocked_trial_requests)
        }
        packet = (
            bench._packet_policy(pending, free, ctx) +
            "\n\nV2.4 ORDERING STATE:\n" + json.dumps(state, sort_keys=True) +
            "\n\nDEMAND:\n" + json.dumps(demand, sort_keys=True) +
            "\n\nSUPPLY:\n" + json.dumps(supply, sort_keys=True) +
            "\n\nALLOWED ACTIONS:\n" + json.dumps(sorted(allowed_actions)))
        final_raw = v1._ask(
            ctx, REFEREE_PROMPT, packet, "es-policy-debate-v2-4-referee")
        referee_action, final_error = validate_final(final_raw, allowed_actions)
        if final_error:
            rejected.append({"role": "referee", "reason": final_error})
    else:
        final_error = "both ordering objections are required"
        blocked_trial_requests = {}

    demand_ordering = demand["ordering"] if demand else "abstain"
    supply_ordering = supply["ordering"] if supply else "abstain"
    effective_action = referee_action or "hold"
    ordering, rotation = manage_rotation(
        state, demand_ordering, supply_ordering, effective_action, ctx)
    # Invalid debates hold a valid incumbent and never invoke the fixed deadline fallback.
    answer = {
        "ordering": ordering,
        "default_sizing": ctx["sel_sizing"],
        "job_sizing": {},
        "why": f"v2.4.3 {rotation['action']}; referee action {effective_action}",
    }
    bench._policy_record(pending, ctx, started_total, answer, "policy_debate_v2_4")
    invalid_hold_used = bool(rejected)
    ratification = {
        "protocol_version": PROTOCOL_VERSION,
        "decision_state": state,
        "settled_before_debate": (
            settled_before_debate[1] if settled_before_debate is not None else None),
        "opening_calls_parallel": True,
        "opening_round_wall_s": opening_wall_s,
        "openings": {"demand": demand, "supply": supply},
        "referee": final_raw,
        "referee_action": referee_action,
        "blocked_trial_requests": blocked_trial_requests,
        "rotation": rotation,
        "deterministic_sizing": ctx.get("sel_sizing"),
        "final_valid": referee_action is not None,
        "fallback_used": False,
        "invalid_hold_used": invalid_hold_used,
        "rejected_outputs": rejected,
    }
    ctx["sel_log"][-1]["ratification"] = ratification
    ctx["debate_epochs"] = ctx.get("debate_epochs", 0) + 1
    ctx["debate_referee_calls"] = ctx.get("debate_referee_calls", 0) + int(
        demand is not None and supply is not None)
    ctx["debate_rejected_outputs"] = ctx.get("debate_rejected_outputs", 0) + len(rejected)
    ctx["debate_v24_epochs"] = ctx.get("debate_v24_epochs", 0) + 1
    ctx["debate_v24_invalid_holds"] = ctx.get(
        "debate_v24_invalid_holds", 0) + int(invalid_hold_used)
    ctx["debate_v24_transition_blocked_requests"] = ctx.get(
        "debate_v24_transition_blocked_requests", 0) + len(blocked_trial_requests)
    bench._policy_execute(pending, free, ctx)
