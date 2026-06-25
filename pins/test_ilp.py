"""
Deterministic tests for the ILP allocator — run from inside pins/:
    ../.venv/bin/python test_ilp.py
No network, no MCP, no LLM. Verifies the LLMSched-style decider in isolation and,
crucially, against the auction it is meant to be compared with (pins/ilp.py).
"""
from ilp import allocate
from mechanism import clear, welfare
from predictor import marginal_values


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def test_capacity_respected() -> None:
    bids = {"A": marginal_values("train"), "B": marginal_values("train")}
    r = allocate(bids, total_gpus=6, current={})
    assert sum(r.allocation.values()) <= 6, r.allocation
    assert r.status in ("Optimal", "GreedyFallback"), r.status
    print("  capacity ->", r.allocation, "welfare", round(r.welfare, 2), f"({r.status})")


def test_ties_auction_on_welfare() -> None:
    # The headline equivalence: on the 1-D pool, welfare-max is greedy-optimal, so the ILP
    # must MATCH the auction's welfare exactly (no rescale cost -> pure welfare).
    for gpus in (3, 4, 6, 8, 12):
        bids = {"A": marginal_values("train", 1.4), "B": marginal_values("train", 0.9),
                "C": marginal_values("eval", 1.1)}
        a = clear(bids, total_gpus=gpus, current={}, rescale_cost=0.0)
        i = allocate(bids, total_gpus=gpus, current={}, rescale_cost=0.0)
        assert approx(i.welfare, a.welfare), (gpus, i.welfare, a.welfare)
    print("  ties-auction -> ILP welfare == auction welfare across pool sizes")


def test_lambda_penalises_preemption() -> None:
    # With a punishing rescale cost the ILP must not pull GPUs off the incumbent for a tiny gain.
    bids = {"A": marginal_values("train"), "B": marginal_values("eval")}
    start = allocate(bids, total_gpus=4, current={}, rescale_cost=0.0).allocation
    held = allocate(bids, total_gpus=4, current=start, rescale_cost=1000.0)
    assert held.preempted == 0, held
    print("  lambda -> no preemption under high rescale cost; alloc", held.allocation)


def test_fine_grained_preemption() -> None:
    # The behavioural edge over the auction's all-or-nothing gate: the ILP can give up SOME
    # GPUs (those whose marginal value < gain elsewhere - cost) while keeping the rest.
    bids = {"A": marginal_values("train", 1.0), "B": marginal_values("train", 1.3)}
    start = {"A": 6, "B": 0}
    r = allocate(bids, total_gpus=6, current=start, rescale_cost=1.0)
    moved = r.preempted
    assert 0 < moved < 6, (moved, r.allocation)  # partial, not all-or-nothing
    print("  fine-grained -> moved", moved, "of 6 GPUs; alloc", r.allocation)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} ILP tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
