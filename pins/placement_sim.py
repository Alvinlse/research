"""
Stage-2, 2-D: does NODE PLACEMENT break the 1-D auction, and does the ILP fix it?

Follow-up to pins/negotiation_sim.py (auction and welfare-ILP tie on a single GPU pool). Here
GPUs live on NODES, jobs are co-located (all GPUs on one node — the NVLink-coupled training
case), and crucially placement is STICKY: a running job cannot migrate for free, so once nodes
fragment the fragments persist. This is the auction's real handicap — it clears a GPU *count*
with no way to express "migrate job X to free a whole node", so a train job (capacity 8) that
needs a full node gets stranded when small jobs hold a slot on every node. The ILP
(pins/ilp.allocate_placement) plans count AND node jointly and may relocate a live job at a
bounded `migrate_cost`, so it can consolidate fragmentation the auction is blind to.

Same workload/metrics/design hinge as negotiation_sim (pure deterministic deciders; bids are
the negotiation candidates). `ploss` = mean GPUs/round won but not placeable (auction's
structural cost; ILP = 0 by construction).

Run:  .venv/bin/python -m pins.placement_sim
"""
from __future__ import annotations

from dataclasses import dataclass

from pins.ilp import allocate_placement
from pins.mechanism import clear, welfare
from pins.negotiation_sim import Job, RESCALE_COST, deadline_bid, make_workload, static_bid
from pins.placement import Cluster, place_sticky

Bids = dict[str, list[float]]
Alloc = dict[str, int]
Home = dict[str, "int | None"]

# Value charged per GPU a live job runs off its current node (relocation cost). Set below a
# train-phase marginal (~8-10) so the ILP WILL pay to migrate small jobs to free a node for a
# high-value train job, but won't churn placements for nothing. Auction cannot migrate at all.
MIGRATE_COST = 1.5


# --------------------------------------------------------------------------- #
#  Sticky, placement-aware schedulers: (bids, cluster, cur, home) -> (alloc, home, ploss)
# --------------------------------------------------------------------------- #
def sched_auction_sticky(bids: Bids, cluster: Cluster, cur: Alloc, home: Home):
    """1-D auction THEN sticky placement. Auction clears counts blind to nodes; incumbents are
    pinned to their node (no migration — the auction cannot express one), new jobs placed FFD,
    overflow repaired. ploss = GPUs lost to fragmentation it could not foresee."""
    counts = clear(bids, cluster.total, current=cur, rescale_cost=RESCALE_COST).allocation
    placed, new_home, loss = place_sticky(counts, cluster, home)
    return placed, new_home, loss


def sched_ilp_sticky(bids: Bids, cluster: Cluster, cur: Alloc, home: Home):
    """ILP that plans count AND node jointly and may MIGRATE a live job at MIGRATE_COST to
    consolidate fragmentation — feasible by construction, so placement loss is always 0."""
    r = allocate_placement(bids, cluster.n_nodes, cluster.gpus_per_node, current=cur,
                           rescale_cost=RESCALE_COST, current_node=home,
                           migrate_cost=MIGRATE_COST)
    return r.allocation, r.detail["node_of"], 0


def sched_greedy_sticky(bids: Bids, cluster: Cluster, cur: Alloc, home: Home):
    """Value-blind foil: greedy-FIFO counts then the SAME sticky placement as the auction."""
    counts = {a: 0 for a in bids}
    left = cluster.total
    for a in sorted(bids):
        give = min(len(bids[a]), left)
        counts[a] = give
        left -= give
        if left <= 0:
            break
    placed, new_home, loss = place_sticky(counts, cluster, home)
    return placed, new_home, loss


STRATEGIES = [
    ("auction+sticky",    static_bid,   sched_auction_sticky),
    ("auction-DL+sticky", deadline_bid, sched_auction_sticky),
    ("ILP-place",         static_bid,   sched_ilp_sticky),
    ("ILP-place-DL",      deadline_bid, sched_ilp_sticky),
    ("greedy+sticky",     static_bid,   sched_greedy_sticky),
]


# --------------------------------------------------------------------------- #
#  Simulator + metrics (carries node state `home` across rounds)               #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    sla_violation_rate: float
    prod_sla_rate: float
    utilisation: float
    welfare: float
    mean_slowdown: float
    placement_loss: float        # mean GPUs/round won but not placeable (fragmentation cost)
    finished: int
    n_jobs: int


def simulate(jobs_proto: list[Job], bid_builder, allocator, cluster: Cluster,
             horizon: int) -> Result:
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    current: Alloc = {}
    home: Home = {}
    busy_sum = busy_steps = 0
    total_welfare = 0.0
    loss_sum = 0.0

    for t in range(horizon):
        active = [j for j in jobs if j.active(t)]
        if not active:
            current, home = {}, {}
            if all(j.done_at is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        demand = sum(j.capacity() for j in active)
        used = sum(current.get(j.jid, 0) for j in active)
        market = {"total_gpus": cluster.total, "free_gpus": cluster.total - used,
                  "contention": "high" if demand > cluster.total else "low"}
        bids = {j.jid: bid_builder(j, t, market) for j in active}
        base = {j.jid: j.bid() for j in active}
        cur = {jid: current.get(jid, 0) for jid in bids}
        cur_home = {jid: home.get(jid) for jid in bids}
        alloc, new_home, ploss = allocator(bids, cluster, cur, cur_home)
        total_welfare += welfare(base, alloc)
        loss_sum += ploss
        busy_sum += sum(alloc.values()) / cluster.total
        busy_steps += 1
        for j in active:
            j.step(alloc.get(j.jid, 0), t)
        current = {jid: alloc.get(jid, 0) for jid in by_id}
        # A job keeps its node only while it actually holds GPUs there; else it releases it.
        home = {jid: (new_home.get(jid) if alloc.get(jid, 0) > 0 else None) for jid in by_id}

    finished = [j for j in jobs if j.done_at is not None]
    def violated(j: Job) -> bool:
        return j.done_at is None or j.done_at > j.deadline
    violations = sum(1 for j in jobs if violated(j))
    prod = [j for j in jobs if j.tier == "prod"]
    prod_viol = sum(1 for j in prod if violated(j))
    slow = [(j.done_at - j.arrival) / j.nominal for j in finished if j.nominal > 0]
    return Result(
        sla_violation_rate=violations / len(jobs),
        prod_sla_rate=prod_viol / max(len(prod), 1),
        utilisation=busy_sum / max(busy_steps, 1),
        welfare=total_welfare,
        mean_slowdown=sum(slow) / max(len(slow), 1),
        placement_loss=loss_sum / max(busy_steps, 1),
        finished=len(finished),
        n_jobs=len(jobs),
    )


def sweep(n_jobs: int, horizon: int, node_counts: list[int], gpus_per_node: int,
          seed: int, arrival_window: int | None = None) -> None:
    # Compress arrivals into `arrival_window` (make_workload spreads them over 0.6*its horizon)
    # so many jobs are concurrent and nodes genuinely fill — that is what makes fragmentation bite.
    jobs = make_workload(n_jobs, seed, arrival_window or horizon)
    n_prod = sum(1 for j in jobs if j.tier == "prod")
    print(f"workload: {n_jobs} jobs ({n_prod} prod / {n_jobs - n_prod} besteffort), "
          f"horizon={horizon}, seed={seed} | co-located jobs, STICKY placement (no free migration)\n")
    print(f"cluster = N nodes x {gpus_per_node} GPUs. Train job (cap 8) needs a WHOLE node; once "
          f"small jobs hold a slot on every node it is stranded.\n"
          f"ploss = mean GPUs/round won but NOT placeable (auction's structural cost; ILP = 0).\n")
    header = (f"{'cluster':>9}  {'strategy':<18} {'SLA-viol':>9} {'prodSLA':>8} "
              f"{'util':>6} {'welfare':>9} {'slowdown':>9} {'ploss':>6} {'done':>7}")
    for nn in node_counts:
        cluster = Cluster(nn, gpus_per_node)
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        rows = [(name, simulate(jobs, bb, alc, cluster, horizon)) for name, bb, alc in STRATEGIES]
        best_sla = min(r.sla_violation_rate for _, r in rows)
        best_prod = min(r.prod_sla_rate for _, r in rows)
        tag = f"{nn}x{gpus_per_node}={cluster.total}"
        for name, r in rows:
            star = "*" if r.sla_violation_rate == best_sla else " "
            pstar = "*" if r.prod_sla_rate == best_prod else " "
            print(f"{tag:>9}  {name:<18} {r.sla_violation_rate:>8.1%}{star}"
                  f"{r.prod_sla_rate:>7.1%}{pstar}{r.utilisation:>6.0%} {r.welfare:>9.0f} "
                  f"{r.mean_slowdown:>9.2f} {r.placement_loss:>6.2f} {r.finished:>3}/{r.n_jobs:<3}")
        print()


def main() -> None:
    # gpus_per_node = 8 so a train job (capacity 8) CAN fit one node; fewer nodes = more contention
    # for whole nodes = more fragmentation under sticky placement. Sweep node count for the curve.
    sweep(n_jobs=40, horizon=400, node_counts=[2, 3, 4, 6], gpus_per_node=8, seed=0,
          arrival_window=60)
    print("'*' = best (lowest) violation rate at that cluster size.")
    print(f"Sticky placement, no free migration; ILP may migrate at MIGRATE_COST={MIGRATE_COST}.")
    print("Expect: ploss > 0 for the auction (stranded by fragmentation), 0 for ILP-place.")


if __name__ == "__main__":
    main()
