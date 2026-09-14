"""Contract tests for the cross-window revised policy-debate protocol."""
from __future__ import annotations

from unittest.mock import patch

from pins import elastisim_bench as bench
from pins import policy_debate as v1
from pins import policy_debate_v2 as debate


class Job:
    def __init__(self, jid: str = "j0", user: str = "u0", submit: float = 0.0,
                 tier: str = "batch"):
        self.identifier = jid
        self.submit_time = submit
        self.start_time = submit
        self.num_nodes = None
        self.num_nodes_min = 1
        self.num_nodes_max = 4
        self.attributes = {
            "jid": jid, "req_nodes": 2, "req_min": 60, "tier": tier,
            "user": user, "deadline_s": submit + 7200,
        }


def context() -> dict:
    return {
        "now": 5400.0, "sel_every": 0,
        "sel_ordering": "auction_deadline", "sel_sizing": "adaptive",
        "fallback_ordering": "auction_deadline", "fallback_sizing": "adaptive",
        "family": "mkt", "sizes": {}, "running": [], "pool_n": 8,
        "arrivals": [], "sel_history": [], "model": "mock", "host": "mock",
        "cache": {}, "calls": 0,
    }


def opening(role: str, ordering: str, sizing: str) -> dict:
    return {
        "ordering": ordering, "default_sizing": sizing,
        "objection": f"{role} harm remains", "evidence": f"observable {role} evidence",
    }


def final(ordering: str, sizing: str) -> dict:
    return {
        "ordering": ordering, "default_sizing": sizing, "job_sizing": {},
        "why": "deadline distribution and queue concentration resolve both objections",
    }


def run_with(replies: list[dict], jobs: list[Job] | None = None):
    calls, executed = [], []

    def fake_ask(*args, **kwargs):
        calls.append(args[5])
        return replies.pop(0)

    market = ("auction_deadline", "auction_wait", "auction_fairness", "auction_priority")
    old = {name: bench.ARMS[name] for name in market}
    replacement = lambda pending, free, ctx: executed.append(
        (ctx["sel_ordering"], ctx["sizer"]))
    for name in market:
        bench.ARMS[name] = replacement
    try:
        ctx = context()
        with patch("pins.correction._ask", side_effect=fake_ask):
            debate.arm_policy_debate_v2(jobs or [Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS.update(old)
    assert not replies
    return ctx, calls, executed


def test_v2_is_registered_without_mutating_frozen_v1() -> None:
    assert bench.ARMS["policy_debate"] is v1.arm_policy_debate
    assert bench.ARMS["policy_debate_v2"] is debate.arm_policy_debate_v2


def test_role_constraints_preserve_real_ordering_disagreement() -> None:
    ctx = context()
    got, error = debate.validate_opening(
        opening("demand", "auction_fairness", "adaptive"), "demand", ctx, bench)
    assert got is None and "objection role" in error
    got, error = debate.validate_opening(
        opening("supply", "auction_deadline", "adaptive"), "supply", ctx, bench)
    assert got is None and "objection role" in error


def test_referee_always_adjudicates_and_may_combine_components() -> None:
    demand = opening("demand", "auction_deadline", "as_requested")
    supply = opening("supply", "auction_fairness", "adaptive")
    ctx, calls, executed = run_with(
        [demand, supply, final("auction_deadline", "adaptive")])
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == ctx["calls"] == 3
    assert calls[-1] == "es-policy-debate-v2-referee"
    assert package["referee_called"] is True
    assert len(package["candidates"]) == 4
    assert package["resolution_source"] == "referee"
    assert executed == [("auction_deadline", "adaptive")]


def test_one_invalid_advocate_fails_closed_without_one_sided_ratification() -> None:
    invalid_demand = opening("demand", "auction_fairness", "adaptive")
    supply = opening("supply", "auction_fairness", "adaptive")
    ctx, calls, executed = run_with([invalid_demand, supply])
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == ctx["calls"] == 2
    assert package["referee_called"] is False
    assert package["fallback_used"] is True
    assert package["fallback_source"] == "configured_deterministic_floor"
    assert ctx["debate_v2_one_sided_rejections"] == 1
    assert executed == [("auction_deadline", "adaptive")]


def test_referee_cannot_invent_an_unproposed_component() -> None:
    demand = opening("demand", "auction_deadline", "adaptive")
    supply = opening("supply", "auction_fairness", "adaptive")
    ctx, calls, executed = run_with(
        [demand, supply, final("auction_wait", "adaptive")])
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 3
    assert package["fallback_used"] is True
    assert any("candidate set" in row["reason"] for row in package["rejected_outputs"])
    assert executed == [("auction_deadline", "adaptive")]


def test_batch_majority_burst_latches_supply_until_backlog_drains() -> None:
    ctx = context()
    first = {"queue_depth": 8, "top_user_job_share": 0.5, "top_user_wait_share": 0.4,
             "production_share": 0.0}
    guard = debate.update_ordering_episode(first, ctx)
    assert guard["triggered"] is True and guard["role"] == "supply"
    diluted = {"queue_depth": 100, "top_user_job_share": 0.2, "top_user_wait_share": 0.2,
               "production_share": 0.0}
    guard = debate.update_ordering_episode(diluted, ctx)
    assert guard["triggered"] is False and guard["role"] == "supply"
    drained = {"queue_depth": 7, "top_user_job_share": 1.0, "top_user_wait_share": 1.0,
               "production_share": 0.0}
    guard = debate.update_ordering_episode(drained, ctx)
    assert guard["released"] is True and guard["active"] is False


def test_material_production_takes_demand_episode_precedence() -> None:
    ctx = context()
    state = {"queue_depth": 8, "top_user_job_share": 1.0, "top_user_wait_share": 1.0,
             "production_share": 0.125}
    guard = debate.update_ordering_episode(state, ctx)
    assert guard["triggered"] is True and guard["role"] == "demand"
    ctx = context()
    debate.update_ordering_episode(
        {"queue_depth": 8, "top_user_job_share": 1.0, "top_user_wait_share": 1.0,
         "production_share": 0.0}, ctx)
    guard = debate.update_ordering_episode(state, ctx)
    assert guard["previous_role"] == "supply"
    assert guard["triggered"] is True and guard["role"] == "demand"


def test_active_fairness_episode_constrains_referee_and_guard_floor() -> None:
    jobs = [Job(str(i), "burst", 0) for i in range(8)]
    demand = opening("demand", "auction_deadline", "adaptive")
    supply = opening("supply", "auction_fairness", "adaptive")
    ctx, calls, executed = run_with(
        [demand, supply, final("auction_deadline", "adaptive")], jobs)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 3
    assert package["ordering_guard"]["role"] == "supply"
    assert package["candidates"] == [
        {"ordering": "auction_fairness", "default_sizing": "adaptive"}]
    assert package["fallback_source"] == "machine_supply_ordering_episode_guard"
    assert ctx["debate_v2_fairness_guard_floors"] == 1
    assert executed == [("auction_fairness", "adaptive")]


def test_active_demand_episode_constrains_referee_and_guard_floor() -> None:
    jobs = [Job(str(i), "prod-burst", 0, tier="prod") for i in range(8)]
    demand = opening("demand", "auction_deadline", "adaptive")
    supply = opening("supply", "auction_fairness", "adaptive")
    ctx, calls, executed = run_with(
        [demand, supply, final("auction_fairness", "adaptive")], jobs)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 3
    assert package["ordering_guard"]["role"] == "demand"
    assert package["candidates"] == [
        {"ordering": "auction_deadline", "default_sizing": "adaptive"}]
    assert package["fallback_source"] == "machine_demand_ordering_episode_guard"
    assert ctx["debate_v2_demand_guard_floors"] == 1
    assert executed == [("auction_deadline", "adaptive")]


def test_decision_state_exposes_deadline_distribution_and_user_concentration() -> None:
    jobs = [Job("a", "heavy", 0), Job("b", "heavy", 0), Job("c", "other", 3600)]
    ctx = context()
    state = debate.decision_state(jobs, [object()] * 2, ctx, bench)
    assert state["queue_depth"] == 3
    assert state["top_user_job_share"] == 0.667
    assert state["top_user_wait_share"] > state["top_user_job_share"]
    assert state["deadline_pressure_jobs"] == 2
    assert state["deadline_pressure_user_share"] == 1.0
    assert state["deadline_budget_used_p90"] == 0.75


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} policy-debate-v2 tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
