"""Contract tests for guarded rotation and fast deterministic sizing in v2.4."""
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
        "queue_depth": 20, "waiting_users": 3, "production_share": 0.0,
        "deadline_pressure_fraction": 0.0, "deadline_pressure_user_share": 0.0,
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


def referee(ordering: str, trial_role: str = "none") -> dict:
    return {
        "ordering": ordering, "trial_role": trial_role,
        "why": "both objections and the rotation vote resolved from supplied state",
    }


def test_v24_is_registered_without_replacing_v23() -> None:
    assert bench.ARMS["policy_debate_v2_4"] is debate.arm_policy_debate_v2_4
    assert bench.ARMS["policy_debate_v2_3"] is not debate.arm_policy_debate_v2_4


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
    assert ctx["debate_v24_sizing_transitions"] == 2


def test_rotation_waits_one_hour_before_first_trial() -> None:
    ctx = {"now": 0}
    selected, event = debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    assert selected == "auction_deadline" and event["action"] == "referee"
    ctx["now"] = 3599
    selected, event = debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    assert selected == "auction_deadline" and event["action"] == "referee"


def test_referee_selected_hourly_rotation_starts_requested_branch_trial() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    selected, event = debate.manage_rotation(
        state(queue_depth=30), "auction_deadline", "auction_fairness",
        "auction_deadline", "supply", ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_start"
    assert ctx["debate_v24_trial"]["end_t"] == 5400


def test_trial_compares_with_predebate_incumbent_not_referee_ordering() -> None:
    """Regression for real traces where Referee ordering and requested advocate naturally match."""
    ctx = {
        "now": 3600,
        "debate_v24_last_trial_t": 0,
        "debate_v24_incumbent_ordering": "auction_priority",
    }
    selected, event = debate.manage_rotation(
        state(queue_depth=50), "auction_priority", "auction_fairness",
        "auction_fairness", "supply", ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_start"
    assert event["trial"]["incumbent"] == "auction_priority"


def test_safe_clock_does_not_select_a_rotation_without_referee_vote() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    selected, event = debate.manage_rotation(
        state(queue_depth=30), "auction_deadline", "auction_fairness",
        "auction_deadline", "none", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "trial_not_selected"
    assert ctx.get("debate_v24_trial") is None


def test_trial_rolls_back_when_queue_regresses() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    debate.manage_rotation(
        state(queue_depth=20), "auction_deadline", "auction_fairness",
        "auction_deadline", "supply", ctx)
    ctx["now"] = 5400
    selected, event = debate.manage_rotation(
        state(queue_depth=21), "auction_deadline", "auction_fairness",
        "auction_deadline", "none", ctx)
    assert selected == "auction_deadline"
    assert event["action"] == "trial_rollback"
    assert ctx["debate_v24_trial_rollbacks"] == 1


def test_trial_is_retained_when_queue_and_pressure_do_not_regress() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    debate.manage_rotation(
        state(queue_depth=20), "auction_deadline", "auction_fairness",
        "auction_deadline", "supply", ctx)
    ctx["now"] = 5400
    selected, event = debate.manage_rotation(
        state(queue_depth=18, deadline_pressure_fraction=0.05),
        "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    assert selected == "auction_fairness"
    assert event["action"] == "trial_accept"
    assert ctx["debate_v24_trial_accepts"] == 1


def test_demand_emergency_aborts_supply_trial() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "supply", ctx)
    ctx["now"] = 3900
    selected, event = debate.manage_rotation(
        state(production_share=0.1), "auction_priority", "auction_fairness",
        "auction_fairness", "none", ctx)
    assert selected == "auction_priority"
    assert event["action"] == "emergency_rollback"
    assert ctx["debate_v24_trial"] is None


def test_llm_schemas_cannot_choose_sizing() -> None:
    ctx = context()
    got, error = debate.validate_opening(
        {**opening("demand", "auction_deadline"), "default_sizing": "adaptive"},
        "demand", ctx, bench)
    assert got is None and "schema" in error
    got, error = debate.validate_final(
        {**referee("auction_deadline"), "default_sizing": "adaptive"},
        {"auction_deadline", "auction_fairness"}, {"none"})
    assert got is None and "schema" in error


def test_referee_cannot_select_an_unrequested_rotation() -> None:
    got, error = debate.validate_final(
        referee("auction_fairness", "supply"),
        {"auction_deadline", "auction_fairness"}, {"none"})
    assert got is None
    assert "not requested" in error


def test_completed_trials_become_referee_learning_evidence() -> None:
    ctx = {"now": 0}
    debate.manage_rotation(
        state(), "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    ctx["now"] = 3600
    debate.manage_rotation(
        state(queue_depth=20), "auction_deadline", "auction_fairness",
        "auction_deadline", "supply", ctx)
    ctx["now"] = 5400
    debate.manage_rotation(
        state(queue_depth=23, deadline_pressure_fraction=0.04),
        "auction_deadline", "auction_fairness", "auction_deadline", "none", ctx)
    learning = debate.trial_learning(ctx["debate_v24_trial_outcomes"])
    assert learning["supply"] == {
        "completed": 1, "accepts": 0, "rollbacks": 1,
        "mean_queue_delta": 3.0, "mean_deadline_pressure_delta": 0.04,
    }
    assert learning["demand"]["completed"] == 0


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


def test_valid_arm_epoch_uses_three_calls_and_deterministic_sizing() -> None:
    replies = {
        "es-policy-debate-v2-4-demand": opening("demand", "auction_deadline"),
        "es-policy-debate-v2-4-supply": opening("supply", "auction_fairness"),
        "es-policy-debate-v2-4-referee": referee("auction_deadline"),
    }
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
    package = ctx["sel_log"][0]["ratification"]
    assert set(calls[:2]) == {
        "es-policy-debate-v2-4-demand", "es-policy-debate-v2-4-supply"}
    assert calls[2] == "es-policy-debate-v2-4-referee"
    assert ctx["calls"] == 3
    assert package["deterministic_sizing"] == "as_requested"
    assert executed == [("auction_deadline", "as_requested")]
    assert ctx["sel_every"] == 1800


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} policy-debate-v2.4 tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
