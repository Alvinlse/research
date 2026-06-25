"""
ILP allocator — the LLMSched-style "guarantee/optimize" decider, as a drop-in
alternative to the PINS auction (pins/mechanism.py).

Why this exists
---------------
research_plan.md Open-Question #1: should the deterministic decider that consumes
the two-LLM negotiation be (a) the PINS auction or (b) an LLMSched-style ILP? This
module is the (b) arm so the two can be compared head-to-head on the SAME negotiated
bids, in the SAME Stage-2 sweep (pins/negotiation_sim.py), under the SAME metrics.

It is the analogue of LLMSched's ILP: take the proposer's candidate (here, the
marginal-value curves that come out of negotiation) and emit a feasible allocation
that maximises value, with a SOFT penalty `λ` (their term) that rewards keeping the
current placement — i.e. minimal-edit, anti-thrashing. The LLM/negotiation never
decides; this pure, deterministic program does — same design hinge as the auction.

Relationship to the auction
---------------------------
On the single GPU pool with non-increasing curves, *welfare-max is already solved
optimally by the auction's greedy fill* — so on welfare alone the ILP ties it. The
one place they can genuinely differ is preemption granularity: mechanism.clear's
anti-thrashing gate is ALL-OR-NOTHING per round (apply the whole target or keep
current), whereas this ILP does FINE-GRAINED partial preemption — it gives up only
the individual GPUs whose welfare gain beats the per-GPU rescale cost. That is the
behavioural difference the sweep measures.

Pure: no MCP, no network, no LLM. Falls back to a greedy fill if PuLP is absent so
the offline guarantee of the rest of the system is preserved.

Model (per round)
-----------------
    maximise   Σ_{a,k} curve[a][k] · y[a,k]            (value of the k-th GPU to a)
             − rescale_cost · Σ_a preempt[a]           (λ: minimal-edit penalty)
    s.t.       Σ_{a,k} y[a,k] ≤ total_gpus             (pool capacity)
               preempt[a] ≥ current[a] − x[a]          (only GPUs PULLED from a job)
               x[a] = Σ_k y[a,k],   y[a,k] ∈ {0,1}

Because each curve is non-increasing, contiguous fill is automatic (the optimiser
takes a job's high-value units before its low-value ones), so the linearisation is
exact — the integer optimum equals the true piecewise-concave-value optimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Bids = dict[str, list[float]]
Alloc = dict[str, int]

try:
    import pulp
    _HAVE_PULP = True
except Exception:  # pragma: no cover - keep the system runnable offline
    _HAVE_PULP = False


@dataclass
class ILPResult:
    allocation: Alloc
    welfare: float                      # realised value of `allocation` (same metric as auction)
    preempted: int = 0                  # GPUs pulled away from a running job this round
    status: str = "Optimal"
    detail: dict = field(default_factory=dict)


def _welfare(bids: Bids, alloc: Alloc) -> float:
    """Total value realised: sum of each agent's first ``alloc[a]`` marginal values.
    Identical definition to mechanism.welfare so the two deciders are scored alike."""
    total = 0.0
    for agent, n in alloc.items():
        curve = bids.get(agent, [])
        total += sum(curve[: max(0, n)])
    return total


def _greedy_fallback(bids: Bids, total_gpus: int) -> Alloc:
    """Welfare-max greedy fill (the auction's own optimum) — used only if PuLP is missing."""
    items: list[tuple[float, str]] = []
    for a, curve in bids.items():
        items.extend((float(v), a) for v in curve)
    items.sort(key=lambda it: (-it[0], it[1]))
    alloc = {a: 0 for a in bids}
    for _, a in items[: max(0, total_gpus)]:
        alloc[a] += 1
    return alloc


def allocate(
    bids: Bids,
    total_gpus: int,
    current: Alloc | None = None,
    rescale_cost: float = 0.0,
    time_limit: float = 0.05,
) -> ILPResult:
    """Solve one round of GPU allocation as a MILP (LLMSched's optimise/guarantee step).

    Args mirror mechanism.clear so the ILP is a true drop-in:
        bids: agent_id -> non-increasing marginal-value curve (the negotiation candidate).
        total_gpus: size of the contested pool.
        current: agent_id -> GPUs currently held (for the λ preemption penalty).
        rescale_cost: λ — value charged per GPU PULLED from a running job (idle fills free).
        time_limit: CBC budget in seconds (LLMSched runs the ILP under a tight ~50 ms cap).
    """
    current = dict(current or {})
    cur = {a: current.get(a, 0) for a in bids}

    if not _HAVE_PULP:
        alloc = _greedy_fallback(bids, total_gpus)
        return ILPResult(alloc, _welfare(bids, alloc), status="GreedyFallback")

    prob = pulp.LpProblem("pins_alloc", pulp.LpMaximize)
    y: dict[tuple[str, int], "pulp.LpVariable"] = {}
    x: dict[str, "pulp.LpAffineExpression"] = {}
    value_terms = []
    for a, curve in bids.items():
        units = [pulp.LpVariable(f"y_{a}_{k}", cat="Binary") for k in range(len(curve))]
        for k, var in enumerate(units):
            y[(a, k)] = var
            value_terms.append(float(curve[k]) * var)
        x[a] = pulp.lpSum(units) if units else 0

    # Soft minimal-edit penalty (LLMSched's λ): charge only for GPUs taken away.
    preempt = {a: pulp.LpVariable(f"p_{a}", lowBound=0) for a in bids}
    for a in bids:
        prob += preempt[a] >= cur[a] - x[a]

    prob += pulp.lpSum(value_terms) - rescale_cost * pulp.lpSum(preempt.values())
    prob += pulp.lpSum(y.values()) <= total_gpus

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    alloc = {a: int(round(sum(y[(a, k)].value() or 0 for k in range(len(bids[a])))))
             for a in bids}
    preempted = sum(max(0, cur[a] - alloc[a]) for a in bids)
    return ILPResult(
        allocation=alloc,
        welfare=_welfare(bids, alloc),
        preempted=preempted,
        status=pulp.LpStatus[prob.status],
        detail={"rescale_cost": rescale_cost},
    )


def allocate_placement(
    bids: Bids,
    n_nodes: int,
    gpus_per_node: int,
    current: Alloc | None = None,
    rescale_cost: float = 0.0,
    coloc: dict[str, bool] | None = None,
    current_node: dict[str, int | None] | None = None,
    migrate_cost: float = 0.0,
    time_limit: float = 0.5,
) -> ILPResult:
    """2-D allocation: choose GPUs-per-(job,node) jointly, respecting NODE boundaries.

    This is the regime where the 1-D auction structurally cannot compete: it clears a GPU
    *count* blind to which node those GPUs live on, so a co-located job can win 8 GPUs that
    are actually 4-here + 4-there and unplaceable (research discussion 2026-06-22). The ILP
    plans count AND placement together, so every allocation it returns is feasible by
    construction — no repair needed. When `current_node`/`migrate_cost` are given it can also
    decide to MIGRATE a running job to consolidate fragmentation — a move the count-only
    auction cannot even express.

    Args:
        bids: agent_id -> non-increasing marginal-value curve (negotiation candidate).
        n_nodes, gpus_per_node: the cluster shape; total pool = n_nodes * gpus_per_node.
        current: agent_id -> GPUs currently held (for the λ preemption penalty).
        rescale_cost: λ — value charged per GPU pulled from a running job.
        coloc: agent_id -> must all its GPUs sit on ONE node? Defaults to True for every job
            (single-node training — the NVLink-coupling case). False = splittable across nodes.
        current_node: agent_id -> node a running job currently sits on (None if unplaced).
        migrate_cost: value charged per GPU that runs on a node OTHER than the job's current
            one — the cost of relocating a live job. The ILP migrates only when it pays off.
        time_limit: CBC budget in seconds.
    """
    current = dict(current or {})
    cur = {a: current.get(a, 0) for a in bids}
    coloc = coloc if coloc is not None else {a: True for a in bids}
    current_node = current_node or {}

    if not _HAVE_PULP:  # offline fallback: 1-D greedy then deterministic placement repair
        from pins.placement import Cluster, place_ffd
        counts = _greedy_fallback(bids, n_nodes * gpus_per_node)
        placed, node_of = place_ffd(counts, Cluster(n_nodes, gpus_per_node), coloc)
        return ILPResult(placed, _welfare(bids, placed),
                         status="GreedyFallback", detail={"node_of": node_of})

    prob = pulp.LpProblem("pins_placement", pulp.LpMaximize)
    nodes = range(n_nodes)
    g = {(a, m): pulp.LpVariable(f"g_{a}_{m}", lowBound=0, upBound=gpus_per_node, cat="Integer")
         for a in bids for m in nodes}
    y = {(a, k): pulp.LpVariable(f"y_{a}_{k}", cat="Binary")
         for a in bids for k in range(len(bids[a]))}
    x = {a: pulp.lpSum(g[(a, m)] for m in nodes) for a in bids}

    value_terms = [float(bids[a][k]) * y[(a, k)] for a in bids for k in range(len(bids[a]))]
    for a in bids:                                   # tie realised value to the GPU count
        prob += pulp.lpSum(y[(a, k)] for k in range(len(bids[a]))) == x[a]
    for m in nodes:                                  # node capacity — the hard 2-D constraint
        prob += pulp.lpSum(g[(a, m)] for a in bids) <= gpus_per_node

    z: dict[tuple[str, int], "pulp.LpVariable"] = {}
    for a in bids:                                   # co-location: all GPUs on a single node
        if coloc.get(a, True):
            for m in nodes:
                z[(a, m)] = pulp.LpVariable(f"z_{a}_{m}", cat="Binary")
                prob += g[(a, m)] <= gpus_per_node * z[(a, m)]
            prob += pulp.lpSum(z[(a, m)] for m in nodes) <= 1

    preempt = {a: pulp.LpVariable(f"p_{a}", lowBound=0) for a in bids}
    for a in bids:
        prob += preempt[a] >= cur[a] - x[a]

    # Migration penalty: GPUs a live job runs on a node OTHER than its current home are charged
    # migrate_cost. Lets the ILP relocate a job to free a whole node — the count-only auction
    # has no such lever. New/unplaced jobs (no current node) migrate for free.
    migrate_terms = []
    if migrate_cost > 0:
        for a in bids:
            home = current_node.get(a)
            if home is not None and cur[a] > 0:
                migrate_terms += [g[(a, m)] for m in nodes if m != home]

    prob += (pulp.lpSum(value_terms)
             - rescale_cost * pulp.lpSum(preempt.values())
             - migrate_cost * pulp.lpSum(migrate_terms))
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))

    alloc = {a: int(round(sum((g[(a, m)].value() or 0) for m in nodes))) for a in bids}
    node_of = {a: next((m for m in nodes if (g[(a, m)].value() or 0) > 0.5), None) for a in bids}
    preempted = sum(max(0, cur[a] - alloc[a]) for a in bids)
    return ILPResult(
        allocation=alloc,
        welfare=_welfare(bids, alloc),
        preempted=preempted,
        status=pulp.LpStatus[prob.status],
        detail={"node_of": node_of},
    )
