"""
Deterministic tests for the 2-D placement layer — run from inside pins/:
    ../.venv/bin/python test_placement.py
Verifies (1) the auction's count grant can be unplaceable and gets repaired with measurable
loss, and (2) the placement ILP never returns an infeasible/repaired allocation.
"""
from ilp import allocate_placement
from mechanism import welfare
from placement import Cluster, is_feasible, place_ffd
from predictor import marginal_values


def approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def test_ffd_repairs_unplaceable_grant() -> None:
    # 2 nodes x 4. Co-located counts {3,3,2} sum to 8 (<= total) but cannot be packed: after
    # placing the two 3s (one per node) only 1+1 is left, so the 2-job is repaired to 1.
    cluster = Cluster(2, 4)
    counts = {"A": 3, "B": 3, "C": 2}
    placed, node_of = place_ffd(counts, cluster)
    loss = sum(counts.values()) - sum(placed.values())
    assert loss == 1, (placed, loss)
    assert is_feasible(placed, node_of, cluster)
    print("  ffd-repair -> placed", placed, "loss", loss)


def test_ilp_placement_is_always_feasible() -> None:
    # The same contended cluster: the ILP must return a placement that actually fits.
    cluster = Cluster(2, 8)
    bids = {"T1": marginal_values("train", 1.5), "T2": marginal_values("train", 1.2),
            "E": marginal_values("eval", 1.0), "P": marginal_values("preprocess", 1.0)}
    r = allocate_placement(bids, cluster.n_nodes, cluster.gpus_per_node)
    assert r.status in ("Optimal", "GreedyFallback"), r.status
    assert is_feasible(r.allocation, r.detail["node_of"], cluster), (r.allocation, r.detail)
    assert sum(r.allocation.values()) <= cluster.total
    print("  ilp-feasible ->", r.allocation, "nodes", r.detail["node_of"])


def test_ilp_beats_auction_when_fragmented() -> None:
    # Incumbent small jobs fragment both nodes; a high-value train job then cannot be placed at
    # full size by the auction path, but the ILP can co-pack the small jobs to free a node.
    cluster = Cluster(2, 8)
    bids = {"T": marginal_values("train", 1.6),          # wants a whole node
            "E1": marginal_values("eval", 0.9), "E2": marginal_values("eval", 0.9)}
    current = {"E1": 2, "E2": 2}                          # one on each node (fragmented)
    # auction path
    from mechanism import clear
    counts = clear(bids, cluster.total, current=current, rescale_cost=2.0).allocation
    a_placed, _ = place_ffd(counts, cluster)
    a_welf = welfare(bids, a_placed)
    # ILP path
    i = allocate_placement(bids, cluster.n_nodes, cluster.gpus_per_node,
                           current=current, rescale_cost=2.0)
    assert is_feasible(i.allocation, i.detail["node_of"], cluster)
    assert i.welfare >= a_welf - 1e-9, (i.allocation, i.welfare, a_placed, a_welf)
    print(f"  ilp>=auction -> ILP welfare {i.welfare:.2f} (alloc {i.allocation}) "
          f">= auction {a_welf:.2f} (alloc {a_placed})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} placement tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
