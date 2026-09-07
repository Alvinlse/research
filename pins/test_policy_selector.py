"""Deterministic checks for the constrained multi-LLM policy selector.

Run with: .venv/bin/python -m pins.test_policy_selector
No model server, network access, GPU, or ElastiSim process is required.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from pins import elastisim_bench as bench
from pins import policy_labels
from pins.policy_dataset import canonical_targets, split_windows


class Job:
    def __init__(self, jid: str = "j0", submit: float = 0, tier: str = "batch"):
        self.identifier = jid
        self.submit_time = submit
        self.start_time = submit
        self.num_nodes = None
        self.num_nodes_min = 1
        self.num_nodes_max = 4
        self.attributes = {"req_nodes": 2, "req_min": 60, "tier": tier, "user": "u0"}


def context() -> dict:
    return {"now": 3600.0, "sel_every": 0, "sel_ordering": "market",
            "sel_sizing": "as_requested", "sizes": {}, "running": [], "pool_n": 8,
            "arrivals": [], "model": "mock", "host": "mock", "cache": {}, "calls": 0}


def test_safety_is_semantic_and_enforced() -> None:
    assert "resize_conservative+greedy" in bench.POLICY_UNSAFE
    assert bench._policy_answer({"ordering": "resize_conservative", "sizing": "greedy"}) is None
    assert bench._policy_answer({"ordering": "least_laxity", "sizing": "adaptive"}) == \
        ("least_laxity", "adaptive")
    assert bench._policy_answer({"ordering": "invented", "sizing": "adaptive"}) is None


def test_train_statistics_do_not_read_test_rewards() -> None:
    states = [{"window": f"w{i}", "rewards": {"a": 1.0, "b": 0.0}} for i in range(12)]
    split = split_windows(states, 7)
    train = [s for s in states if split[s["window"]] == "train"]
    before = canonical_targets(train)
    for s in states:
        if split[s["window"]] == "test":
            s["rewards"] = {"a": -1000.0, "b": 1000.0}
    after = canonical_targets(train)
    assert before == after == {"a": len(train)}, (before, after)


def test_opposed_reviews_reach_referee_and_code_executes_choice() -> None:
    calls = []
    replies = iter([
        {"ordering": "least_laxity", "sizing": "adaptive", "statement": "long waits"},
        {"ordering": "market", "sizing": "as_requested", "statement": "capacity is tight"},
        {"ordering": "least_laxity", "sizing": "adaptive", "why": "demand evidence"},
    ])

    def fake_ask(system, user, *args, **kwargs):
        calls.append((system, user, kwargs))
        return next(replies)

    executed = []
    old = bench.ARMS["least_laxity"]
    bench.ARMS["least_laxity"] = lambda pending, free, ctx: executed.append(ctx["sizer"])
    try:
        ctx = context()
        with patch("pins.correction._ask", side_effect=fake_ask):
            bench.arm_policy_negotiate([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS["least_laxity"] = old

    assert ctx["calls"] == 3
    assert len(calls) == 3
    referee_input = calls[2][1]
    assert "DEMAND STATEMENT" in referee_input and "long waits" in referee_input
    assert "SUPPLY STATEMENT" in referee_input and "capacity is tight" in referee_input
    assert (ctx["sel_ordering"], ctx["sel_sizing"]) == ("least_laxity", "adaptive")
    assert executed == ["adaptive"]
    assert ctx["sel_log"][0]["mode"] == "opposed"


def test_invalid_referee_answer_preserves_safe_previous_policy() -> None:
    executed = []
    old = bench.ARMS["market"]
    bench.ARMS["market"] = lambda pending, free, ctx: executed.append(ctx["sizer"])
    try:
        ctx = context()
        replies = iter([{}, {}, {"ordering": "resize_conservative", "sizing": "greedy"}])
        with patch("pins.correction._ask", side_effect=lambda *a, **k: next(replies)):
            bench.arm_policy_negotiate([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS["market"] = old
    assert ctx["sel_invalid"] == 1
    assert (ctx["sel_ordering"], ctx["sel_sizing"]) == ("market", "as_requested")
    assert executed == ["as_requested"]


def test_symmetric_control_has_matched_three_call_budget() -> None:
    calls = []
    replies = iter([
        {"ordering": "fairness", "sizing": "adaptive", "statement": "review A"},
        {"ordering": "fairness", "sizing": "adaptive", "statement": "review B"},
        {"ordering": "fairness", "sizing": "adaptive", "why": "both reviewers"},
    ])
    old = bench.ARMS["fairness"]
    bench.ARMS["fairness"] = lambda *args: None
    try:
        ctx = context()
        with patch("pins.correction._ask",
                   side_effect=lambda *a, **k: calls.append((a, k)) or next(replies)):
            bench.arm_policy_symmetric([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS["fairness"] = old
    assert ctx["calls"] == len(calls) == 3
    assert "REVIEWER_A STATEMENT" in calls[2][0][1]
    assert "REVIEWER_B STATEMENT" in calls[2][0][1]
    assert ctx["sel_log"][0]["mode"] == "symmetric"


def test_bo3_matches_three_call_budget_and_votes() -> None:
    replies = iter([
        {"ordering": "market", "sizing": "adaptive"},
        {"ordering": "fairness", "sizing": "adaptive"},
        {"ordering": "market", "sizing": "adaptive"},
    ])
    executed = []
    old = bench.ARMS["market"]
    bench.ARMS["market"] = lambda pending, free, ctx: executed.append(ctx["sizer"])
    try:
        ctx = context()
        with patch("pins.correction._ask", side_effect=lambda *a, **k: next(replies)):
            bench.arm_policy_bo3([Job()], [object()] * 4, ctx)
    finally:
        bench.ARMS["market"] = old
    assert ctx["calls"] == 3
    assert (ctx["sel_ordering"], ctx["sel_sizing"]) == ("market", "adaptive")
    assert executed == ["adaptive"]


def test_packet_is_richer_without_oracle_fields() -> None:
    packet = bench._packet_policy([Job(submit=1200, tier="prod")], [object()] * 3, context())
    for field in ("wait_s_p90", "asked_gpus_p90", "malleable_waiting", "running_age_s_p90",
                  "laxity_s_min", "negative_laxity_waiting"):
        assert field in packet, field
    assert "_true_dur" not in packet and "_real_wait" not in packet
    assert "user whose queued work has waited longest" in packet


def test_interval_teacher_returns_to_common_continuation() -> None:
    schedules = []

    def fake_run(world, arm, **kwargs):
        schedules.append(json.loads(kwargs["policy_schedule"].read_text()))
        return {"n": 1, "sla10_viol_pct": 0, "mean_bsd": 1, "mean_wait_s": 0,
                "resize_events": 0, "jain_user_wait": 1}

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        world, out = root / "w0", root / "labels"
        world.mkdir(); out.mkdir()
        (out / "w0_packets.jsonl").write_text(json.dumps(
            {"t": 100, "packet": "online state"}) + "\n")
        with patch("pins.policy_labels.run", side_effect=fake_run):
            n = policy_labels.label_world(world, out, every=1800, keep=1, done=set(),
                                          lo=0, hi=1, budget=[1], decision_horizon=1800)
        assert n == 1 and len(schedules) == 1
        assert schedules[0] == [
            {"t": 0, **policy_labels.BASELINE},
            {"t": 100, "ordering": "market", "sizing": "as_requested"},
            {"t": 1900, **policy_labels.BASELINE},
        ]
        row = json.loads((out / "rollouts.0.jsonl").read_text())
        assert row["decision_horizon"] == 1800 and row["key"].endswith("|h1800")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} policy-selector tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
