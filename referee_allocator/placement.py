"""
Node placement + repair — the step the 1-D auction needs and the ILP folds in for free.

The PINS auction (pins/mechanism.py) clears a GPU *count* per job, blind to which node
those GPUs sit on. On a real cluster GPUs live on nodes, and a co-located job (single-node
training, NVLink-coupled) needs all its GPUs on ONE node. So an auction grant of "8 GPUs"
can be physically unplaceable when those 8 are 4-here + 4-there — exactly the failure
discussed on 2026-06-22. This module does the deterministic best-effort placement and, when
a grant cannot be honoured, the REPAIR (shrink the job to the largest single-node block it
can actually get). The GPUs lost to repair are the auction's structural placement cost; the
ILP (pins/ilp.allocate_placement) avoids them by planning count and node jointly.

Pure: no LLM, no network. First-fit-decreasing — the standard, defensible bin-packer, so the
auction is given a fair placement, not a strawman.
"""
from __future__ import annotations

from dataclasses import dataclass

Alloc = dict[str, int]


@dataclass(frozen=True)
class Cluster:
    n_nodes: int
    gpus_per_node: int

    @property
    def total(self) -> int:
        return self.n_nodes * self.gpus_per_node


def place_ffd(
    counts: Alloc,
    cluster: Cluster,
    coloc: dict[str, bool] | None = None,
) -> tuple[Alloc, dict[str, int | None]]:
    """First-fit-decreasing placement of granted GPU counts onto nodes, with repair.

    Co-located jobs must land entirely on one node; if no node has room for the full grant the
    job is repaired DOWN to the largest single-node free block (possibly 0). Splittable jobs are
    filled greedily across nodes up to their grant. Jobs are placed largest-first (FFD).

    Returns (placed_counts, node_of) where placed_counts[a] <= counts[a]; the deficit
    sum(counts) - sum(placed) is the placement loss the 1-D auction cannot foresee.
    """
    coloc = coloc if coloc is not None else {a: True for a in counts}
    free = [cluster.gpus_per_node] * cluster.n_nodes
    placed: Alloc = {a: 0 for a in counts}
    node_of: dict[str, int | None] = {a: None for a in counts}

    # Largest grants first; deterministic tie-break by id.
    for a in sorted(counts, key=lambda j: (-counts[j], j)):
        want = counts[a]
        if want <= 0:
            continue
        if coloc.get(a, True):
            # Must fit fully on one node; pick the fullest node that still fits (best-fit) so
            # we leave whole nodes open for bigger jobs. If none fits, repair to the largest block.
            fit = [m for m in range(cluster.n_nodes) if free[m] >= want]
            if fit:
                m = min(fit, key=lambda m: free[m])      # tightest node that fits
                free[m] -= want
                placed[a], node_of[a] = want, m
            else:
                m = max(range(cluster.n_nodes), key=lambda m: free[m])
                give = free[m]                            # repair: largest single-node block
                if give > 0:
                    free[m] -= give
                    placed[a], node_of[a] = give, m
        else:
            left = want                                   # splittable: fill across nodes
            for m in sorted(range(cluster.n_nodes), key=lambda m: -free[m]):
                take = min(left, free[m])
                if take > 0:
                    free[m] -= take
                    left -= take
                    placed[a] += take
                    if node_of[a] is None:
                        node_of[a] = m
                if left <= 0:
                    break
    return placed, node_of


def place_sticky(
    counts: Alloc,
    cluster: Cluster,
    home: dict[str, int | None],
    coloc: dict[str, bool] | None = None,
) -> tuple[Alloc, dict[str, int | None], int]:
    """Sticky placement: a running job CANNOT migrate — it stays on its home node and may only
    grow into free space there (or shrink). This is the auction's real handicap: it emits a GPU
    *count* with no way to say "move job X to free a node", so once nodes fragment, the fragments
    persist. New jobs are placed FFD into whatever node space remains; co-located grants that
    cannot fit on a single node are repaired down (the loss).

    Returns (placed, new_home, loss). `home[a]` is the node a already runs on (None if new).
    """
    coloc = coloc if coloc is not None else {a: True for a in counts}
    free = [cluster.gpus_per_node] * cluster.n_nodes
    placed: Alloc = {a: 0 for a in counts}
    new_home: dict[str, int | None] = {a: None for a in counts}
    loss = 0

    # 1. Incumbents first: pinned to home, clamp the grant to that node's free space (no migration).
    for a in sorted(counts):
        m = home.get(a)
        if m is None or counts[a] <= 0:
            continue
        give = min(counts[a], free[m])
        free[m] -= give
        placed[a], new_home[a] = give, m
        loss += counts[a] - give                      # grant we could not honour on the home node

    # 2. New jobs: FFD (largest grant first) into remaining space; co-located must fit one node.
    new_jobs = [a for a in counts if home.get(a) is None and counts[a] > 0]
    for a in sorted(new_jobs, key=lambda j: (-counts[j], j)):
        want = counts[a]
        if coloc.get(a, True):
            fit = [m for m in range(cluster.n_nodes) if free[m] >= want]
            if fit:
                m = min(fit, key=lambda m: free[m])
                free[m] -= want
                placed[a], new_home[a] = want, m
            else:
                m = max(range(cluster.n_nodes), key=lambda m: free[m])
                give = free[m]
                if give > 0:
                    free[m] -= give
                    placed[a], new_home[a] = give, m
                loss += want - give
        else:
            left = want
            for m in sorted(range(cluster.n_nodes), key=lambda m: -free[m]):
                take = min(left, free[m])
                if take > 0:
                    free[m] -= take
                    left -= take
                    placed[a] += take
                    if new_home[a] is None:
                        new_home[a] = m
                if left <= 0:
                    break
            loss += left
    return placed, new_home, loss


def is_feasible(
    counts: Alloc,
    node_of: dict[str, int | None],
    cluster: Cluster,
    coloc: dict[str, bool] | None = None,
) -> bool:
    """True iff `counts` placed at `node_of` respects node capacity and co-location."""
    coloc = coloc if coloc is not None else {a: True for a in counts}
    load = [0] * cluster.n_nodes
    for a, n in counts.items():
        if n <= 0:
            continue
        m = node_of.get(a)
        if coloc.get(a, True):
            if m is None:
                return False
            load[m] += n
        else:
            # splittable feasibility is not tracked per-node here; only co-located is checked.
            pass
    return all(l <= cluster.gpus_per_node for l in load)
