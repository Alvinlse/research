"""Fast contract tests for the bounded ``policy_debate`` arm.

Run with: ``.venv/bin/python -m pins.test_policy_debate``.
No model server, network connection, GPU, or ElastiSim process is required.
"""
from __future__ import annotations

from unittest.mock import patch

from pins import elastisim_bench as bench
from pins import policy_debate as debate


class Job:
    def __init__(self, jid: str = "j0"):
        self.identifier = jid
        self.submit_time = 0.0
        self.start_time = 0.0
        self.num_nodes = None
        self.num_nodes_min = 1
        self.num_nodes_max = 4
        self.attributes = {
            "jid": jid,
            "req_nodes": 2,
            "req_min": 60,
            "tier": "prod",
            "user": "u0",
            "deadline_s": 7200,
        }


def context() -> dict:
    return {
        "now": 3600.0,
        "sel_every": 0,
        "sel_ordering": "auction_deadline",
        "sel_sizing": "adaptive",
        "fallback_ordering": "auction_deadline",
        "fallback_sizing": "adaptive",
        "family": "mkt",
        "sizes": {},
        "running": [],
        "pool_n": 8,
        "arrivals": [],
        "model": "mock",
        "host": "mock",
        "cache": {},
        "calls": 0,
    }


def opening(ordering: str, sizing: str, role: str) -> dict:
    return {
        "ordering": ordering,
        "default_sizing": sizing,
        "red_line": dict(debate.INCENTIVE_PROFILES[role]["red_line"]),
        "evidence": f"observable {role} evidence",
    }


def final(ordering: str, sizing: str) -> dict:
    return {
        "ordering": ordering,
        "default_sizing": sizing,
        "job_sizing": {},
        "why": "the negotiated evidence supports it",
    }


def run_with(replies: list[dict], executed: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    calls = []

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
            debate.arm_policy_debate([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS.update(old)
    assert not replies
    return ctx, calls


def test_arm_is_registered_without_replacing_completed_arm() -> None:
    assert bench.ARMS["policy_debate"] is debate.arm_policy_debate
    assert bench.ARMS["policy_negotiate"] is bench.arm_policy_negotiate
    assert bench.ARMS["policy_debate"] is not bench.ARMS["policy_negotiate"]


def test_opening_agreement_skips_rebuttal_and_records_ratification() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_deadline", "adaptive", "supply")
    executed = []
    ctx, calls = run_with([demand, supply], executed)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == ctx["calls"] == 2
    assert all("rebut" not in tag and "referee" not in tag for tag in calls)
    assert ctx["debate_epochs"] == ctx["debate_opening_agreements"] == 1
    assert ctx["debate_rebuttals"] == 0
    assert ctx["debate_direct_ratifications"] == 1
    assert ctx["debate_referee_calls"] == 0
    assert package["opening_agreement"] is True
    assert package["rebuttal_triggered"] is False
    assert package["referee_called"] is False
    assert package["resolution_source"] == "opening_consensus"
    assert package["fallback_used"] is False
    assert package["concessions"] == package["dissent"] == package["rejected_outputs"] == []
    assert executed == [("auction_deadline", "adaptive")]


def test_disagreement_triggers_one_round_and_accepts_only_concrete_moves() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_wait", "as_requested", "supply")
    demand_accepts = {
        "move": "accept",
        "ordering": "auction_wait",
        "default_sizing": "as_requested",
        "concession": "accept the capacity-protecting opening",
        "evidence": "queue pressure is high",
    }
    supply_counters = {
        "move": "counter",
        "ordering": "auction_wait",
        "default_sizing": "adaptive",
        "concession": "retain pricing but share capacity",
        "evidence": "several jobs can run malleably",
    }
    executed = []
    ctx, calls = run_with(
        [demand, supply, demand_accepts, supply_counters, final("auction_wait", "adaptive")],
        executed,
    )
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == ctx["calls"] == 5
    assert sum("rebut" in tag for tag in calls) == 2
    assert ctx["debate_rebuttals"] == 1
    assert ctx["debate_referee_calls"] == 1
    assert ctx["debate_concessions"] == len(package["concessions"]) == 2
    assert ctx["debate_dissent"] == len(package["dissent"]) == 1
    assert package["rejected_outputs"] == []
    assert package["revisions"]["demand"]["move"] == "accept"
    assert package["revisions"]["supply"]["move"] == "counter"
    assert executed == [("auction_wait", "adaptive")]


def test_two_component_counter_is_rejected_and_opening_is_retained() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_wait", "as_requested", "supply")
    invalid_counter = {
        "move": "counter",
        "ordering": "auction_priority",
        "default_sizing": "greedy",
        "concession": "change everything",
        "evidence": "unsupported",
    }
    supply_retains = {
        "move": "retain",
        "ordering": "auction_wait",
        "default_sizing": "as_requested",
        "concession": "none",
        "evidence": "capacity remains tight",
    }
    ctx, _ = run_with(
        [demand, supply, invalid_counter, supply_retains,
         final("auction_deadline", "adaptive")],
        [],
    )
    package = ctx["sel_log"][0]["ratification"]
    assert package["revisions"]["demand"] is None
    assert any(
        item["role"] == "demand" and "exactly one" in item["reason"]
        for item in package["rejected_outputs"]
    )
    assert ctx["debate_rejected_outputs"] == 1
    # Invalid cross-talk cannot erase or silently rewrite a valid opening.
    assert package["dissent"][0]["preferred"] == {
        "ordering": "auction_wait", "default_sizing": "as_requested"}


def test_invalid_referee_output_executes_floor_and_marks_risk() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_wait", "as_requested", "supply")
    demand_retains = {
        "move": "retain", "ordering": "auction_deadline", "default_sizing": "adaptive",
        "concession": "none", "evidence": "deadline pressure remains decisive",
    }
    supply_retains = {
        "move": "retain", "ordering": "auction_wait", "default_sizing": "as_requested",
        "concession": "none", "evidence": "capacity remains protected",
    }
    # This is a valid market policy, but neither advocate proposed it.
    invalid_final = final("auction_fairness", "adaptive")
    executed = []
    ctx, calls = run_with(
        [demand, supply, demand_retains, supply_retains, invalid_final], executed)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == 5
    assert ctx["sel_invalid"] == ctx["fallbacks"] == 1
    assert package["final_valid"] is False and package["fallback_used"] is True
    assert package["fallback_source"] == "configured_deterministic_floor"
    assert package["selected"] == {
        "ordering": "auction_deadline", "default_sizing": "adaptive"}
    assert any(item["stage"] == "referee" and "surviving" in item["reason"]
               for item in package["rejected_outputs"])
    assert ctx["debate_rejected_outputs"] == 1
    assert "deterministic floor" in package["residual_risk"][0]
    assert executed == [("auction_deadline", "adaptive")]


def test_opening_schema_and_family_are_fail_closed() -> None:
    wrong_family = opening("fairness", "adaptive", "demand")
    got, error = debate.validate_opening(wrong_family, "demand", context(), bench)
    assert got is None and "outside" in error
    with_extra = opening("auction_wait", "adaptive", "demand") | {"gpu_count": 8}
    got, error = debate.validate_opening(with_extra, "demand", context(), bench)
    assert got is None and "unexpected" in error


def test_machine_readable_red_lines_are_exact_and_enforced_when_active() -> None:
    tampered = opening("auction_deadline", "adaptive", "demand")
    tampered["red_line"]["threshold"] = 99
    got, error = debate.validate_opening(tampered, "demand", context(), bench)
    assert got is None and "exactly echo" in error

    violates_demand = opening("auction_wait", "adaptive", "demand")
    got, error = debate.validate_opening(
        violates_demand, "demand", context(), bench, red_line_active=True)
    assert got is None and "requires ordering=auction_deadline" in error

    violates_supply = opening("auction_fairness", "greedy", "supply")
    got, error = debate.validate_opening(
        violates_supply, "supply", context(), bench, red_line_active=True)
    assert got is None and "forbids default_sizing=greedy" in error


def test_rebuttal_consensus_is_ratified_without_a_referee() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_wait", "adaptive", "supply")
    demand_accepts = {
        "move": "accept", "ordering": "auction_wait", "default_sizing": "adaptive",
        "concession": "ordering", "evidence": "current deadline pressure is low",
    }
    supply_retains = {
        "move": "retain", "ordering": "auction_wait", "default_sizing": "adaptive",
        "concession": "none", "evidence": "adaptive sizing protects capacity",
    }
    executed = []
    ctx, calls = run_with([demand, supply, demand_accepts, supply_retains], executed)
    package = ctx["sel_log"][0]["ratification"]
    assert len(calls) == ctx["calls"] == 4
    assert all("referee" not in tag for tag in calls)
    assert package["resolution_source"] == "rebuttal_consensus"
    assert package["referee_called"] is False
    assert package["candidates"] == [{
        "ordering": "auction_wait", "default_sizing": "adaptive",
        "roles": ["demand", "supply"],
    }]
    assert package["dissent"] == []
    assert executed == [("auction_wait", "adaptive")]


def test_move_is_derived_from_the_policy_not_the_advocate_label() -> None:
    demand = opening("auction_deadline", "adaptive", "demand")
    supply = opening("auction_wait", "as_requested", "supply")
    ctx = context()
    # Previously rejected: labelled "accept" but actually a one-component counter.
    mislabelled = {
        "move": "accept", "ordering": "auction_wait", "default_sizing": "adaptive",
        "concession": "concede pricing only", "evidence": "queue pressure is high",
    }
    got, error = debate.validate_revision(mislabelled, demand, supply, ctx, bench, "demand")
    assert error is None and got["move"] == "counter"
    # No label at all is the new contract, and copying the other opening is still an accept.
    unlabelled = {
        "ordering": "auction_wait", "default_sizing": "as_requested",
        "concession": "adopt the supply opening", "evidence": "capacity is the binding constraint",
    }
    got, error = debate.validate_revision(unlabelled, demand, supply, ctx, bench, "demand")
    assert error is None and got["move"] == "accept"
    # The bounded protocol survives: a third policy is still refused.
    invented = {
        "ordering": "auction_priority", "default_sizing": "greedy",
        "concession": "change everything", "evidence": "unsupported",
    }
    got, error = debate.validate_revision(invented, demand, supply, ctx, bench, "demand")
    assert got is None and "exactly one" in error
    assert "move" not in debate.REVISION_SCHEMA


def test_incentive_profiles_share_dimensions_and_are_genuinely_diverse() -> None:
    assert debate.INCENTIVE_DISTANCE_L1 >= 1.0
    for profile in debate.INCENTIVE_PROFILES.values():
        assert tuple(profile["weights"]) == debate.SHARED_DIMENSIONS
        assert abs(sum(profile["weights"].values()) - 1.0) < 1e-9
        assert profile["home_dimension"] == max(
            profile["weights"], key=profile["weights"].get)
        assert set(profile["red_line"]) == {
            "metric", "operator", "threshold", "policy_field",
            "policy_operator", "policy_value",
        }


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    print(f"running {len(tests)} policy-debate tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
