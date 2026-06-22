"""
Deterministic tests for the auctioneer — run with: python -m pins.test_mechanism
No network, no MCP, no LLM. This is how we verify the "decider" in isolation.
"""
from mechanism import clear, welfare
from predictor import marginal_values


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def test_scarcity_goes_to_highest_value() -> None:
    # 4 GPUs, two agents. A (train) values GPUs more than B (eval).
    bids = {"A": marginal_values("train"), "B": marginal_values("eval")}
    r = clear(bids, total_gpus=4, current={"A": 0, "B": 0}, rescale_cost=0.0)
    assert sum(r.allocation.values()) == 4, r.allocation
    # A's high marginal values should win it the majority of the pool.
    assert r.allocation["A"] >= r.allocation["B"], r.allocation
    # Welfare must equal the realised value of the chosen allocation.
    assert approx(r.welfare, welfare(bids, r.allocation))
    print("  scarcity ->", r.allocation, "price", r.price, "welfare", round(r.welfare, 2))


def test_demand_below_supply_price_zero() -> None:
    # Plenty of GPUs: nobody is rationed, clearing price is 0.
    bids = {"A": marginal_values("preprocess"), "B": marginal_values("eval")}
    r = clear(bids, total_gpus=16, current={}, rescale_cost=0.0)
    assert approx(r.price, 0.0), r.price
    # Each agent gets exactly its capacity (curve length).
    assert r.allocation["A"] == len(bids["A"])
    assert r.allocation["B"] == len(bids["B"])
    print("  abundance ->", r.allocation, "price", r.price)


def test_efficiency_total_value() -> None:
    # The greedy fill must be welfare-optimal vs. a hand-picked alternative.
    bids = {"A": marginal_values("train"), "B": marginal_values("train")}
    r = clear(bids, total_gpus=6, current={}, rescale_cost=0.0)
    alt = {"A": 6, "B": 0}  # give everything to A
    assert welfare(bids, r.allocation) >= welfare(bids, alt) - 1e-9
    print("  efficiency ->", r.allocation, "welfare", round(r.welfare, 2),
          ">= alt", round(welfare(bids, alt), 2))


def test_antithrashing_blocks_marginal_move() -> None:
    # Current alloc already near-optimal; a huge rescale cost must block churn.
    bids = {"A": marginal_values("train"), "B": marginal_values("eval")}
    start = clear(bids, total_gpus=4, current={}, rescale_cost=0.0).allocation
    # Re-clear with a punishing rescale cost and a tiny perturbation incentive.
    r = clear(bids, total_gpus=4, current=start, rescale_cost=1000.0)
    assert not r.applied or r.gpus_moved == 0, r
    assert r.allocation == start, (r.allocation, start)
    print("  anti-thrash ->", r.allocation, "applied", r.applied)


def test_phase_change_shifts_demand() -> None:
    # Same job, preprocess -> train, should pull in more GPUs when contested.
    pre = {"A": marginal_values("preprocess"), "B": marginal_values("train")}
    tr = {"A": marginal_values("train"), "B": marginal_values("train")}
    a_pre = clear(pre, total_gpus=8, current={}, rescale_cost=0.0).allocation["A"]
    a_tr = clear(tr, total_gpus=8, current={}, rescale_cost=0.0).allocation["A"]
    assert a_tr > a_pre, (a_pre, a_tr)
    print(f"  phase-change -> A held {a_pre} (preprocess) then {a_tr} (train)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} mechanism tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
