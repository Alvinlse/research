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


def test_policy_family_is_matched_size_and_gates_both_menu_and_answer() -> None:
    """The 2x2's family factor: same number of actions on each side, and no silent remapping."""
    sizes = {f: len(o) * len(bench.POLICY_MENU["sizing"]) for f, o in bench.POLICY_FAMILY.items()}
    assert len(set(sizes.values())) == 1, sizes          # RQ1 must not be a menu-size contrast
    mkt, nm = {"family": "mkt"}, {"family": "nm"}
    assert bench._family_orderings(mkt) == ["market"]
    assert bench._family_orderings() == list(bench.POLICY_MENU["ordering"])   # unset = full library
    menu = "\n".join(bench._policy_menu_lines(mkt))
    assert "market:" in menu and "least_laxity:" not in menu and "fairness:" not in menu
    # an off-family ordering is rejected, exactly like an invented one -- never remapped
    assert bench._policy_answer({"ordering": "fairness", "sizing": "adaptive"}, mkt) is None
    assert bench._policy_answer({"ordering": "market", "sizing": "adaptive"}, nm) is None
    assert bench._policy_answer({"ordering": "market", "sizing": "adaptive"}, mkt) == \
        ("market", "adaptive")
    # and the pre-selection fallback is INSIDE the family, or the contrast leaks
    for fam, orderings in bench.POLICY_FAMILY.items():
        assert bench._family_orderings({"family": fam})[0] in orderings


def test_by_size_sizer_is_per_job_and_leaves_other_sizers_alone() -> None:
    """Exp 101's 0-token floor: one sizing rule per job, keyed on what the job asked for."""
    small, large = Job("small"), Job("large")
    small.attributes["req_nodes"] = 1
    large.attributes["req_nodes"] = 4
    free = [object()] * 4
    ctx = context() | {"sizer": "by_size"}
    assert bench._size(small, free, ctx, pending_n=2) == 1          # as_requested
    assert bench._size(large, free, ctx, pending_n=2) == 2          # adaptive: shares the free pool
    # the same two jobs under one global sizer are sized identically, which is the contrast
    for rule, want in (("as_requested", (1, 4)), ("greedy", (4, 4))):
        got = tuple(bench._size(j, free, context() | {"sizer": rule}, 2) for j in (small, large))
        assert got == want, (rule, got)


def test_every_sizer_respects_capacity_and_malleable_bounds() -> None:
    """Phase-4 invariants, checked on the one function every arm's allocation goes through."""
    for rule in ("as_requested", "adaptive", "greedy", "by_size", "auto"):
        for lo, hi, asked in ((1, 4, 2), (2, 2, 2), (1, 8, 6), (3, 5, 1)):
            for f in range(0, 9):
                job = Job()
                job.num_nodes_min, job.num_nodes_max = lo, hi
                job.attributes["req_nodes"] = asked
                for pending_n in (1, 3):
                    k = bench._size(job, [object()] * f, context() | {"sizer": rule}, pending_n)
                    assert k >= 0, (rule, lo, hi, f, k)
                    assert k <= f, (rule, lo, hi, f, k)              # never exceeds free capacity
                    if k:
                        assert lo <= k <= hi, (rule, lo, hi, f, k)   # malleable bounds hold
                    elif lo <= f and rule in ("adaptive", "greedy"):
                        # `as_requested` waits for the exact size by design, and `auto`/`by_size`
                        # may resolve to it, so only the two sharing rules must always start a job
                        # that fits at its minimum.
                        raise AssertionError(f"{rule} refused a startable job: {lo}<={f}")


def test_start_at_allocates_once_and_consumes_exactly_k() -> None:
    class Node:
        pass

    class Assignable(Job):
        def __init__(self):
            super().__init__()
            self.assigned = None

        def assign(self, nodes):
            assert self.assigned is None, "a job was allocated twice"
            self.assigned = list(nodes)

        def assign_num_gpus_per_node(self, n):
            self.gpus_per_node = n

    job, free, ctx = Assignable(), [Node() for _ in range(5)], context()
    bench._start_at(job, free, ctx, 3)
    assert len(job.assigned) == 3 and len(free) == 2        # the nodes really left the free pool
    assert ctx["sizes"][job.identifier] == 3
    assert not set(map(id, job.assigned)) & set(map(id, free))


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} policy-selector tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
