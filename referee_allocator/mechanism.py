"""
PINS auctioneer — the deterministic clearing mechanism (Step 3 / Step 8).

This module is intentionally PURE: no MCP, no network, no LLM. It is the
"mechanism decides" half of the design hinge in research_plan.md:60. Because it
is pure, it is unit-testable in milliseconds (see test_mechanism.py) and you can
later *prove* properties about it (efficiency, no-starvation) for the paper.

Model
-----
A GPU pool has `total_gpus` units. Each job-agent submits a *marginal-value
curve*: a list ``[v(1), v(2), ...]`` where ``v(k)`` is what the agent is willing
to pay for its k-th GPU **in its current predicted phase**. Curves are expected
to be non-increasing (diminishing returns); that is what makes the greedy fill
welfare-optimal. The list length is the agent's capacity (max useful GPUs) this
phase, so a job in its preprocess phase submits a short curve and the same job in
its train phase submits a long one — this is the time-varying demand PINS exploits.

Clearing
--------
1. Pool every marginal bid into one list and sort descending by value.
2. Award the top `total_gpus` units -> the welfare-maximising target allocation.
3. Uniform clearing price = value of the highest *rejected* bid.
4. Anti-thrashing gate: only move GPUs if the welfare gain beats the rescale
   cost of the moves (research_plan.md:58). Otherwise keep the current alloc.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClearResult:
    allocation: dict[str, int]          # agent_id -> GPUs held after clearing
    deltas: dict[str, int]              # agent_id -> change vs. previous alloc
    price: float                        # uniform clearing price
    welfare: float                      # total realised value of `allocation`
    applied: bool                       # False if anti-thrashing gate blocked the move
    gpus_moved: int = 0                 # number of GPUs that changed hands
    detail: dict = field(default_factory=dict)


def welfare(bids: dict[str, list[float]], alloc: dict[str, int]) -> float:
    """Total value realised: sum of each agent's first ``alloc[a]`` marginal values."""
    total = 0.0
    for agent, n in alloc.items():
        curve = bids.get(agent, [])
        total += sum(curve[: max(0, n)])
    return total


def clear(
    bids: dict[str, list[float]],
    total_gpus: int,
    current: dict[str, int] | None = None,
    rescale_cost: float = 0.0,
) -> ClearResult:
    """Run one sealed-bid, uniform-price clearing over `bids`.

    Args:
        bids: agent_id -> non-increasing marginal-value curve for this phase.
        total_gpus: size of the contested pool.
        current: agent_id -> GPUs currently held (for the anti-thrashing gate).
        rescale_cost: value-cost charged per GPU that changes hands.
    """
    current = dict(current or {})
    cur = {a: current.get(a, 0) for a in bids}

    # 1. Pool every marginal bid as (value, agent, k). Deterministic tie-break.
    items: list[tuple[float, str, int]] = []
    for agent, curve in bids.items():
        for k, v in enumerate(curve):
            items.append((float(v), agent, k))
    items.sort(key=lambda it: (-it[0], it[1], it[2]))

    # 2. Welfare-maximising target: award the top `total_gpus` marginal bids.
    target = {a: 0 for a in bids}
    awarded = min(total_gpus, len(items))
    for i in range(awarded):
        _, agent, _ = items[i]
        target[agent] += 1

    # 3. Uniform price = first rejected marginal bid (0 if demand <= supply).
    price = items[total_gpus][0] if len(items) > total_gpus else 0.0

    # 4. Anti-thrashing gate: is the reallocation worth its rescale cost?
    # The disruption cost is for PREEMPTIONS — GPUs pulled away from a job that was using
    # them (forcing it to checkpoint / elastically shrink). Filling IDLE GPUs is free, so a
    # cold start or any move into spare capacity is never blocked; only DISPLACING a running
    # job must clear the bar. (Charging idle fills too would make the gate refuse free
    # capacity from a cold start — see negotiation_sim.)
    w_target = welfare(bids, target)
    w_cur = welfare(bids, cur)
    preempted = sum(max(0, cur[a] - target[a]) for a in bids)   # GPUs taken from jobs
    gpus_moved = sum(max(0, target[a] - cur[a]) for a in bids)  # GPUs reassigned (for reporting)
    cost = preempted * rescale_cost
    applied = (w_target - w_cur) > cost

    final = target if applied else cur
    deltas = {a: final[a] - cur[a] for a in bids}
    return ClearResult(
        allocation=final,
        deltas=deltas,
        price=price,
        welfare=welfare(bids, final),
        applied=applied,
        gpus_moved=gpus_moved if applied else 0,
        detail={
            "welfare_target": round(w_target, 4),
            "welfare_current": round(w_cur, 4),
            "rescale_cost": round(cost, 4),
        },
    )
