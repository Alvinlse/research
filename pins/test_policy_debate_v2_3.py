"""Contract and concurrency tests for policy debate v2.3."""
from __future__ import annotations

import threading
import time
from unittest.mock import patch

from pins import elastisim_bench as bench
from pins import policy_debate_v2_3 as debate
from pins.test_policy_debate_v2 import Job, context, final, opening


def state(**updates) -> dict:
    base = {
        "queue_depth": 12,
        "waiting_users": 3,
        "deadline_pressure_fraction": 0.0,
        "deadline_pressure_user_share": 0.0,
        "production_share": 0.0,
        "top_user_job_share": 0.6,
        "top_user_wait_share": 0.7,
        "queue_delta_previous_epoch": 2,
    }
    base.update(updates)
    return base


def run_with(replies: dict[str, dict], jobs: list[Job] | None = None,
             configure=None):
    calls, executed = [], []

    def fake_ask(*args, **kwargs):
        tag = args[5]
        calls.append(tag)
        return replies[tag]

    market = ("auction_deadline", "auction_wait", "auction_fairness", "auction_priority")
    old = {name: bench.ARMS[name] for name in market}
    replacement = lambda pending, free, ctx: executed.append(
        (ctx["sel_ordering"], ctx["sizer"]))
    for name in market:
        bench.ARMS[name] = replacement
    try:
        ctx = context()
        if configure:
            configure(ctx)
        with patch("pins.correction._ask", side_effect=fake_ask):
            debate.arm_policy_debate_v2_3(jobs or [Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS.update(old)
    return ctx, calls, executed


def test_v23_is_registered_separately() -> None:
    assert bench.ARMS["policy_debate_v2_3"] is debate.arm_policy_debate_v2_3
    assert bench.ARMS["policy_debate_v2"] is not debate.arm_policy_debate_v2_3


def test_queue_growth_is_measured_across_selection_epochs() -> None:
    jobs = [Job(str(i), "u0" if i < 5 else "u1") for i in range(8)]
    ctx = context()
    ctx["debate_v23_previous_queue_depth"] = 5
    got = debate.decision_state(jobs, [object()] * 2, ctx, bench)
    assert got["queue_delta_previous_epoch"] == 3
    assert got["queue_growth_streak"] == 1
    ctx["now"] += 1800
    got = debate.decision_state(jobs + [Job("8", "u1")], [object()] * 2, ctx, bench)
    assert got["queue_delta_previous_epoch"] == 1
    assert got["queue_growth_streak"] == 2


def test_concentration_alone_is_not_a_supply_veto() -> None:
    ctx = context()
    got = debate.update_escalation(
        state(queue_delta_previous_epoch=0, top_user_job_share=1.0), ctx)
    assert got["active"] is False
    assert ctx["debate_v23_supply_observation_streak"] == 0


def test_supply_requires_two_consecutive_deterioration_observations() -> None:
    ctx = context()
    first = debate.update_escalation(state(), ctx)
    second = debate.update_escalation(state(queue_delta_previous_epoch=1), ctx)
    assert first["active"] is False
    assert second["triggered"] is True and second["role"] == "supply"
    diluted = debate.update_escalation(
        state(top_user_job_share=0.2, top_user_wait_share=0.2,
              queue_delta_previous_epoch=-1), ctx)
    assert diluted["role"] == "supply"


def test_demand_emergency_preempts_supply_and_drain_releases() -> None:
    ctx = context()
    debate.update_escalation(state(), ctx)
    debate.update_escalation(state(), ctx)
    got = debate.update_escalation(state(production_share=0.1), ctx)
    assert got["previous_role"] == "supply"
    assert got["triggered"] is True and got["role"] == "demand"
    got = debate.update_escalation(state(queue_depth=7), ctx)
    assert got["released"] is True and got["active"] is False


def test_distributed_deadline_pressure_activates_demand() -> None:
    ctx = context()
    got = debate.update_escalation(
        state(deadline_pressure_fraction=0.5,
              deadline_pressure_user_share=0.5), ctx)
    assert got["role"] == "demand"
    assert got["reason"] == "severe user-distributed deadline pressure"


def test_demand_and_supply_openings_overlap() -> None:
    barrier = threading.Barrier(2)
    spans = {}

    def fake_ask(*args, **kwargs):
        tag = args[5]
        spans[tag] = [time.monotonic(), None]
        barrier.wait(timeout=1)
        time.sleep(0.05)
        spans[tag][1] = time.monotonic()
        role = "demand" if tag.endswith("demand") else "supply"
        ordering = "auction_deadline" if role == "demand" else "auction_fairness"
        return opening(role, ordering, "adaptive")

    ctx = context()
    jobs = [Job("a", "u0"), Job("b", "u1")]
    with patch("pins.correction._ask", side_effect=fake_ask):
        _, _, wall_s = debate.parallel_openings(
            jobs, [object()] * 2, ctx, bench, state(queue_depth=2))
    demand_span, supply_span = spans.values()
    assert max(demand_span[0], supply_span[0]) < min(demand_span[1], supply_span[1])
    assert wall_s < 0.09
    assert ctx["calls"] == 2


def test_valid_epoch_uses_parallel_openings_then_referee() -> None:
    replies = {
        "es-policy-debate-v2-3-demand": opening(
            "demand", "auction_deadline", "as_requested"),
        "es-policy-debate-v2-3-supply": opening(
            "supply", "auction_fairness", "adaptive"),
        "es-policy-debate-v2-3-referee": final("auction_deadline", "adaptive"),
    }
    ctx, calls, executed = run_with(replies)
    package = ctx["sel_log"][0]["ratification"]
    assert set(calls[:2]) == {
        "es-policy-debate-v2-3-demand", "es-policy-debate-v2-3-supply"}
    assert calls[2] == "es-policy-debate-v2-3-referee"
    assert ctx["calls"] == 3 and package["opening_calls_parallel"] is True
    assert package["protocol_version"] == "2.3"
    assert len(package["candidates"]) == 4
    assert executed == [("auction_deadline", "adaptive")]


def test_active_supply_escalation_is_machine_enforced() -> None:
    jobs = [Job(str(i), "burst") for i in range(7)] + [Job("other", "other")]
    replies = {
        "es-policy-debate-v2-3-demand": opening(
            "demand", "auction_deadline", "adaptive"),
        "es-policy-debate-v2-3-supply": opening(
            "supply", "auction_fairness", "adaptive"),
        "es-policy-debate-v2-3-referee": final("auction_deadline", "adaptive"),
    }

    def configure(ctx):
        ctx["debate_v23_previous_queue_depth"] = 7
        ctx["debate_v23_supply_observation_streak"] = 1

    ctx, _, executed = run_with(replies, jobs, configure)
    package = ctx["sel_log"][0]["ratification"]
    assert package["escalation"]["role"] == "supply"
    assert package["candidates"] == [
        {"ordering": "auction_fairness", "default_sizing": "adaptive"}]
    assert package["fallback_source"] == "machine_supply_v23_escalation"
    assert executed == [("auction_fairness", "adaptive")]


def test_invalid_opening_uses_configured_floor_without_one_sided_ruling() -> None:
    replies = {
        "es-policy-debate-v2-3-demand": opening(
            "demand", "auction_fairness", "adaptive"),
        "es-policy-debate-v2-3-supply": opening(
            "supply", "auction_fairness", "adaptive"),
    }
    ctx, calls, executed = run_with(replies)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 2 and package["referee_called"] is False
    assert package["fallback_source"] == "configured_deterministic_floor"
    assert executed == [("auction_deadline", "adaptive")]


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} policy-debate-v2.3 tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
