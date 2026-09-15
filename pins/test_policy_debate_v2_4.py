"""Contract tests for phase-correct rotation learning and sizing in v2.4.3."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from pins import elastisim_bench as bench
from pins import policy_debate_v2_4 as debate
from pins.test_policy_debate_v2 import Job, context


def state(**updates) -> dict:
    base = {
        "queue_depth": 20,
        "queue_delta_last_epoch": 3,
        "waiting_users": 3,
        "production_share": 0.0,
        "deadline_pressure_fraction": 0.0,
        "deadline_pressure_jobs": 0,
        "deadline_pressure_user_share": 0.0,
        "top_user_wait_share": 0.8,
        "wait_s_p90": 1000.0,
    }
    base.update(updates)
    return base


def opening(role: str, ordering: str, request_trial: bool = False) -> dict:
    return {
        "ordering": ordering,
        "request_trial": request_trial,
        "objection": f"{role} branch identifies a distinct harm",
        "evidence": f"numeric {role} evidence",
    }


def referee(action: str = "hold") -> dict:
    return {
        "action": action,
        "why": "both objections and transition history justify this action",
    }


def initialise(ctx: dict) -> None:
    ctx.update(
        now=0,
        debate_v24_last_trial_t=0,
        debate_v24_incumbent_ordering="auction_deadline")


def start_supply_trial(ctx: dict, baseline: dict | None = None) -> tuple[str, dict]:
    ctx["now"] = 3600
    return debate.manage_rotation(
        baseline or state(), "auction_deadline", "auction_fairness",
        "trial_supply", ctx)


def test_v24_is_registered_without_replacing_v23() -> None:
    assert bench.ARMS["policy_debate_v2_4"] is debate.arm_policy_debate_v2_4
    assert bench.ARMS["policy_debate_v2_3"] is not debate.arm_policy_debate_v2_4
    assert debate.PROTOCOL_VERSION == "2.4.3"


def test_sizing_review_uses_five_minute_cadence_and_hysteresis() -> None:
    load = {"queue_pressure": 0.1, "arrival_load": 0.1}
    fake = SimpleNamespace(
        _load_fields=lambda pending, ctx: load,
        _job_key=lambda job: job.identifier)
    ctx = context()
    pending = [Job("p")]
    ctx["running"] = [Job("r")]
    ctx["now"] = 0
    assert debate.update_sizing_controller(pending, [], ctx, fake) == "as_requested"
    assert ctx["job_sizing"] == {"p": "as_requested", "r": "as_requested"}
    load.update(queue_pressure=1.0, arrival_load=1.0)
    ctx["now"] = 299
    assert debate.update_sizing_controller(pending, [], ctx, fake) == "as_requested"
    assert ctx["debate_v24_sizing_reviews"] == 1
    ctx["now"] = 300
    assert debate.update_sizing_controller(pending, [], ctx, fake) == "adaptive"
    load.update(queue_pressure=0.5, arrival_load=0.1)
    ctx["now"] = 600
    assert debate.update_sizing_controller(pending, [], ctx, fake) == "adaptive"


def test_advocates_may_abstain_but_cannot_request_an_abstention_trial() -> None:
    ctx = context()
    got, error = debate.validate_opening(
        opening("demand", "abstain"), "demand", ctx, bench)
    assert error is None and got["ordering"] == "abstain"
    got, error = debate.validate_opening(
        opening("demand", "abstain", True), "demand", ctx, bench)
    assert got is None and "cannot request" in error
    got, error = debate.validate_opening(
        opening("demand", "auction_fairness"), "demand", ctx, bench)
    assert got is None and "outside its role" in error


def test_referee_emits_only_an_allowed_hold_or_trial_action() -> None:
    assert debate.validate_final(referee("hold"), {"hold"})[0] == "hold"
    got, error = debate.validate_final(
        referee("trial_supply"), {"hold", "trial_demand"})
    assert got is None and "not eligible" in error
    got, error = debate.validate_final(
        {"ordering": "auction_deadline", "why": "old schema"}, {"hold"})
    assert got is None and "schema" in error


def test_hold_never_changes_the_incumbent() -> None:
    ctx = {}
    initialise(ctx)
    ctx["now"] = 3600
    selected, event = debate.manage_rotation(
        state(), "auction_priority", "auction_fairness", "hold", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "hold"


def test_rotation_waits_one_hour() -> None:
    ctx = {}
    initialise(ctx)
    ctx["now"] = 3599
    selected, event = debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "trial_supply", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "hold"
    assert ctx.get("debate_v24_trial") is None


def test_requested_rotation_starts_against_saved_incumbent() -> None:
    ctx = {}
    initialise(ctx)
    selected, event = start_supply_trial(ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_start"
    assert event["trial"]["incumbent"] == "auction_deadline"
    assert event["trial"]["phase"] == "trial"


def test_passing_first_phase_enters_probation_not_acceptance() -> None:
    ctx = {}
    initialise(ctx)
    _, start_event = start_supply_trial(ctx)
    ctx["now"] = 5400
    selected, event = debate.manage_rotation(
        state(queue_depth=18, queue_delta_last_epoch=-2,
              top_user_wait_share=0.75, wait_s_p90=2500),
        "auction_deadline", "auction_fairness", "hold", ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_probation"
    assert ctx["debate_v24_trial"]["phase"] == "probation"
    assert start_event["trial"]["phase"] == "trial"
    assert ctx.get("debate_v24_trial_accepts", 0) == 0


def test_passing_probation_accepts_and_starts_cooldown_at_completion() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 5400
    debate.manage_rotation(
        state(queue_depth=18, queue_delta_last_epoch=-2,
              top_user_wait_share=0.75, wait_s_p90=2500),
        "auction_deadline", "auction_fairness", "hold", ctx)
    ctx["now"] = 7200
    selected, event = debate.manage_rotation(
        state(queue_depth=17, queue_delta_last_epoch=-1,
              top_user_wait_share=0.73, wait_s_p90=4000),
        "auction_deadline", "auction_fairness", "hold", ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_accept"
    assert ctx["debate_v24_incumbent_ordering"] == "auction_fairness"
    assert ctx["debate_v24_last_trial_t"] == 7200


def test_first_phase_rolls_back_on_queue_regression() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 5400
    selected, event = debate.manage_rotation(
        state(queue_depth=25, queue_delta_last_epoch=5,
              top_user_wait_share=0.82, wait_s_p90=2500),
        "auction_deadline", "auction_fairness", "hold", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "trial_rollback"
    assert event["reason"] == "trial_failed"


def test_probation_failure_rolls_back() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 5400
    debate.manage_rotation(
        state(queue_depth=18, queue_delta_last_epoch=-2,
              top_user_wait_share=0.75, wait_s_p90=2500),
        "auction_deadline", "auction_fairness", "hold", ctx)
    ctx["now"] = 7200
    selected, event = debate.manage_rotation(
        state(queue_depth=22, queue_delta_last_epoch=4,
              top_user_wait_share=0.8, wait_s_p90=4000),
        "auction_deadline", "auction_fairness", "hold", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "trial_rollback"
    assert event["reason"] == "probation_failed"


def test_demand_emergency_aborts_supply_trial() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 3900
    selected, event = debate.manage_rotation(
        state(production_share=0.1), "auction_priority", "auction_fairness",
        "hold", ctx)
    assert selected == "auction_priority"
    assert event["action"] == "emergency_demand"
    assert ctx["debate_v24_trial"] is None
    assert ctx["debate_v24_trial_rollbacks"] == 1


def test_completed_trial_learning_is_keyed_by_transition() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 5400
    debate.manage_rotation(
        state(queue_depth=25, top_user_wait_share=0.82, wait_s_p90=2500),
        "auction_deadline", "auction_fairness", "hold", ctx)
    learning = debate.trial_learning(ctx["debate_v24_trial_outcomes"])
    assert learning["supply"]["rollbacks"] == 1
    assert learning["by_transition"][
        "auction_deadline->auction_fairness"]["rollbacks"] == 1


def test_rollback_learning_uses_failed_phase_not_original_trial_baseline() -> None:
    outcome = {
        "action": "trial_rollback", "reason": "probation_failed", "t": 7200,
        "trial": {
            "role": "supply", "incumbent": "auction_deadline",
            "ordering": "auction_fairness", "phase": "probation",
            "baseline_queue_depth": 50,
            "baseline_deadline_pressure_fraction": 0.20,
            "phase_baseline": {"queue_depth": 15, "deadline_pressure_fraction": 0.02},
        },
        # The full exposure looks beneficial (50 -> 20), but probation itself failed (+5).
        "end_queue_depth": 20,
        "end_deadline_pressure_fraction": 0.04,
        "checks": {"queue_delta": 5, "deadline_pressure_delta": 0.02,
                   "passed": False},
    }
    learning = debate.trial_learning([outcome], now=7200)
    role = learning["supply"]
    transition = learning["by_transition"]["auction_deadline->auction_fairness"]
    assert role["accepted_evidence_count"] == 0
    assert role["mean_accepted_phase_queue_delta"] is None
    assert transition["last_phase_queue_delta"] == 5
    assert transition["last_phase_deadline_pressure_delta"] == 0.02
    assert transition["last_verdict"] == "harmful_challenger"
    assert transition["recommendation"] == "negative_challenger_evidence"


def rollback_outcome(t: int) -> dict:
    return {
        "action": "trial_rollback", "reason": "trial_failed", "t": t,
        "trial": {
            "role": "supply", "incumbent": "auction_deadline",
            "ordering": "auction_fairness", "phase": "trial",
        },
        "checks": {"queue_delta": 3, "deadline_pressure_delta": 0.01,
                   "passed": False},
    }


def test_two_consecutive_rollbacks_block_only_the_exact_transition() -> None:
    outcomes = [rollback_outcome(100), rollback_outcome(200)]
    blocked = debate.transition_status(
        outcomes, "auction_deadline", "auction_fairness", 201)
    reverse = debate.transition_status(
        outcomes, "auction_fairness", "auction_deadline", 201)
    expired = debate.transition_status(
        outcomes, "auction_deadline", "auction_fairness",
        200 + debate.TRANSITION_BACKOFF_S)
    assert blocked["consecutive_rollbacks"] == 2
    assert blocked["blocked_now"] is True
    assert blocked["recommendation"] == "block_harmful_challenger"
    assert reverse["blocked_now"] is False
    assert expired["blocked_now"] is False


def test_rotation_manager_enforces_transition_backoff() -> None:
    ctx = {}
    initialise(ctx)
    ctx["now"] = 3600
    ctx["debate_v24_trial_outcomes"] = [rollback_outcome(100), rollback_outcome(200)]
    selected, event = debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "trial_supply", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "transition_backoff_hold"
    assert event["transition"]["blocked_now"] is True
    assert ctx["debate_v24_transition_backoff_holds"] == 1


def test_expired_outcome_can_be_added_before_decision_state_is_built() -> None:
    ctx = {}
    initialise(ctx)
    start_supply_trial(ctx)
    ctx["now"] = 5400
    settled = debate.settle_expired_trial(
        state(queue_depth=25, top_user_wait_share=0.82, wait_s_p90=2500), ctx)
    assert settled[1]["action"] == "trial_rollback"
    base = {
        **state(queue_depth=25),
        "queue_pressure": 1.0, "arrival_load_1h": 1.0,
        "free_gpus": 1, "current_policy_held_epochs": 1,
        "deadline_budget_used_p50": 0.0, "deadline_budget_used_p90": 0.0,
        "declared_walltime_share": 1.0,
    }
    with patch.object(debate.v2, "decision_state", return_value=base):
        visible = debate.decision_state([], [], ctx, SimpleNamespace())
    assert visible["prior_trial_outcomes"][-1]["action"] == "trial_rollback"


def test_ordering_openings_are_parallel() -> None:
    barrier = threading.Barrier(2)
    spans = {}

    def fake_ask(*args, **kwargs):
        tag = args[5]
        spans[tag] = [time.monotonic(), None]
        barrier.wait(timeout=1)
        time.sleep(0.05)
        spans[tag][1] = time.monotonic()
        return opening(
            "demand" if tag.endswith("demand") else "supply",
            "auction_deadline" if tag.endswith("demand") else "auction_fairness")

    ctx = context()
    jobs = [Job("a", "u0"), Job("b", "u1")]
    with patch("pins.correction._ask", side_effect=fake_ask):
        debate.parallel_openings(jobs, [object()] * 2, ctx, bench, state())
    first, second = spans.values()
    assert max(first[0], second[0]) < min(first[1], second[1])
    assert ctx["calls"] == 2


def run_arm(replies: dict) -> tuple[dict, list[str], list[tuple[str, str]]]:
    calls, executed = [], []

    def fake_ask(*args, **kwargs):
        calls.append(args[5])
        return replies[args[5]]

    market = ("auction_deadline", "auction_wait", "auction_fairness", "auction_priority")
    old = {name: bench.ARMS[name] for name in market}
    for name in market:
        bench.ARMS[name] = lambda pending, free, ctx: executed.append(
            (ctx["sel_ordering"], ctx["sel_sizing"]))
    try:
        ctx = context()
        with patch("pins.correction._ask", side_effect=fake_ask):
            debate.arm_policy_debate_v2_4([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS.update(old)
    return ctx, calls, executed


def test_valid_arm_epoch_uses_three_calls_and_holds_incumbent() -> None:
    ctx, calls, executed = run_arm({
        "es-policy-debate-v2-4-demand": opening("demand", "auction_deadline"),
        "es-policy-debate-v2-4-supply": opening("supply", "auction_fairness"),
        "es-policy-debate-v2-4-referee": referee("hold"),
    })
    package = ctx["sel_log"][0]["ratification"]
    assert set(calls[:2]) == {
        "es-policy-debate-v2-4-demand", "es-policy-debate-v2-4-supply"}
    assert calls[2] == "es-policy-debate-v2-4-referee"
    assert ctx["calls"] == 3
    assert package["protocol_version"] == "2.4.3"
    assert package["fallback_used"] is False
    assert package["invalid_hold_used"] is False
    assert executed == [("auction_deadline", "as_requested")]


def test_invalid_opening_holds_without_fixed_fallback() -> None:
    ctx, calls, executed = run_arm({
        "es-policy-debate-v2-4-demand": opening("demand", "auction_fairness"),
        "es-policy-debate-v2-4-supply": opening("supply", "auction_fairness"),
    })
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 2
    assert ctx.get("fallbacks", 0) == 0
    assert ctx["debate_v24_invalid_holds"] == 1
    assert package["invalid_hold_used"] is True
    assert executed == [("auction_deadline", "as_requested")]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} policy-debate-v2.4 tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
