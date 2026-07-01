"""
The FULL locked PINS pipeline, end-to-end, in one world (build follow-up (b), Exp 25).

Exp 22/24 (`pins/two_sided_sim.py`) resolved the demand-margin ⇄ supply-reserve negotiation but
then used a BESPOKE prod-first grant as a stand-in for the decider. The locked architecture
(`research_progress.md` 2026-06-25; `Research/CLAUDE.md` §3) is:

    LLMs reason/bid  →  committed-auction decides (who/how-many)  →  ILP places (where) / guarantees

This module wires the negotiated bids through the REAL deciders so the merged sim matches that
pipeline end-to-end:

  * RATION = the committed-auction (Exp-11 SLA winner): bid-once frozen priority + run-to-completion
    `_serialise` from `pins/negotiation_sim.py` — NOT the per-round `mechanism.clear`, which Exp 9/18A
    already showed spreads + thrashes and loses SLA. The committed-auction IS the locked decider.
  * PLACE = `pins/ilp.allocate_placement` (Exp-18 winner on nodes): plans node assignment jointly and
    migrates live jobs to consolidate fragmentation — feasible by construction (placement loss 0),
    vs the count-only `place_sticky` baseline that strands whole-node train jobs behind small ones.

The world is the Exp-22/24 one (rigid incumbents + stochastic train-work spikes; a margin GPU buys
spike-absorbing speed), now on a NODE cluster so placement bites. The negotiation is the Exp-24
contested-slice version (running train jobs contest the free GPUs for a margin vs the supply reserve).

The experiment is a 2×2 decomposition — negotiation {off, on} × placement {sticky, ILP}:
  * floor       — no negotiation, committed + sticky   (the pre-pipeline baseline)
  * floor+ILP   — no negotiation, committed + ILP       (ILP placement value alone)
  * nego+sticky — negotiation,    committed + sticky    (negotiation value alone)
  * pipeline    — negotiation,    committed + ILP        (the full locked pipeline)

`ploss` = mean GPUs/round won by the auction but not placeable (fragmentation cost; ILP → ~0).

Run:  .venv/bin/python -m pins.pipeline_sim                 # rule fallback (no Ollama)
      .venv/bin/python -m pins.pipeline_sim --llm --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import os

from pins import bridge
from pins.ilp import allocate_placement
from pins.llm_agent import save_cache
from pins.negotiation_protocol import DemandJob, negotiate
from pins.negotiation_sim import Job, _serialise, make_workload
from pins.placement import Cluster, place_sticky
from pins.predictor import PHASE_PROFILES, marginal_values
from pins.two_sided_sim import job_facts
from pins.uncertainty_sim import assign, load_uncertainty_distribution, true_need

HERE = os.path.dirname(os.path.abspath(__file__))
DORDER = {"ahead": 0, "ontrack": 1, "behind": 2}
MIGRATE_COST = 1.5          # value charged per GPU a live job runs off its home node (see Exp 18)


def bid_with_margin(job: Job, phase: str, margin: int) -> list[float]:
    """The job's marginal-value curve for `phase`, extended by `margin` safety-margin units that
    continue the diminishing curve (train phase only). margin=0 reproduces the plain bid.

    `phase` is passed in explicitly (not read from job.phase()): the simulator tracks each job's
    phase in a local `pidx` dict and never mutates the Job object, so job.phase()/job.bid() would
    be stale at the initial phase."""
    curve = marginal_values(phase, job.urgency)
    cap, base, decay = PHASE_PROFILES[phase]
    for k in range(margin if cap > 0 else 0):
        curve.append(round(job.urgency * base * (decay ** (cap + k)), 4))
    return curve


def simulate(jobs_proto: list[Job], cluster: Cluster, horizon: int, u_map: dict, spike_map: dict,
             scale: int, spike_max: float, *, negotiate_on: bool, place: str,
             use_llm: bool, model: str, cache: dict) -> dict:
    """One run of the pipeline on a fresh workload copy.

    RATION: committed-auction — priority frozen to each job's urgency on arrival, serialise
    full blocks by that order over (total − reserve). PLACE: `place` ∈ {sticky, ilp}.
    Negotiation (when on) sizes each running train job's margin and the supply reserve over the
    genuinely-free GPUs (Exp-24 contested slice). Spikes: a train phase's true work is inflated;
    margin GPUs grant rate>1 to absorb it (capped by the spike's usable parallelism)."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}     # realised (spiked) work
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}      # extra GPUs a spike can use
    held = {j.jid: 0 for j in jobs}
    home: dict[str, int | None] = {j.jid: None for j in jobs}
    progress = {j.jid: 0.0 for j in jobs}
    pidx = {j.jid: 0 for j in jobs}
    done_at: dict[str, int | None] = {j.jid: None for j in jobs}
    frozen: dict[str, float] = {}                                    # bid-once committed priority
    busy_sum = 0.0
    busy_steps = 0
    loss_sum = 0.0
    n_fallback = 0
    n_decisions = 0

    def phase_of(j):
        return j.phases[pidx[j.jid]] if pidx[j.jid] < len(j.phases) else "idle"

    def cap0(j):
        return PHASE_PROFILES[phase_of(j)][0]

    def remaining(j):
        return max(0.0, j.need[pidx[j.jid]] - progress[j.jid]) + sum(j.need[pidx[j.jid] + 1:])

    for t in range(horizon):
        active = [j for j in jobs if j.arrival <= t and done_at[j.jid] is None]
        if not active:
            if all(done_at[j.jid] is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue

        for j in active:                                # bid-once: freeze priority on arrival
            frozen.setdefault(j.jid, j.urgency)

        # --- NEGOTIATE the margin (demand) vs reserve (supply) over the contested slice ----------
        margins = {j.jid: 0 for j in active}
        reserve = 0
        if negotiate_on:
            demand_gpus = sum(cap0(j) for j in active)
            con_supply = bridge.contention_bucket(demand_gpus, cluster.total)
            con_demand = "high" if demand_gpus >= cluster.total else "low"
            n_inc = sum(1 for jj in jobs if jj.tier == "prod" and jj.arrival > t)
            supply_ctx = bridge.reserve_ctx(con_supply, n_inc)
            demand: list[DemandJob] = []
            for j in active:                            # only RUNNING train jobs contest a margin
                if phase_of(j) == "train" and held[j.jid] >= cap0(j) > 0:
                    db = bridge.deadline_bucket(remaining(j), j.deadline - t)
                    ctx = bridge.margin_ctx(job_facts(j, u_map[j.jid], spike_max), db, con_demand)
                    rank = DORDER.get(db, 0) * 2 + (1 if j.tier == "prod" else 0)
                    demand.append(DemandJob(j.jid, ctx, 0, True, float(rank)))
            free_now = cluster.total - sum(held[j.jid] for j in active)
            o = negotiate(demand, supply_ctx, free_now, use_llm=use_llm, model=model, cache=cache)
            margins.update(o.margins)
            reserve = o.reserve
            n_decisions += 1
            if not o.agreed:
                n_fallback += 1

        # --- RATION: committed-auction (frozen priority, serialise full blocks over total−reserve) -
        bids = {j.jid: bid_with_margin(j, phase_of(j),
                                       margins.get(j.jid, 0) if phase_of(j) == "train" else 0)
                for j in active}
        order = sorted((j.jid for j in active), key=lambda a: (-frozen.get(a, 0.0), a))
        counts = _serialise(order, bids, max(0, cluster.total - reserve))

        # --- PLACE: sticky (fragments) vs ILP (migrates to consolidate, feasible by construction) --
        cur_counts = {j.jid: held[j.jid] for j in active}
        cur_home = {j.jid: home[j.jid] for j in active}
        if place == "ilp":
            # Feed the committed counts to the placement ILP as flat priority-weighted demands: it
            # places WHERE (and migrates) but does not re-ration. ploss = counts it cannot fit.
            place_bids = {a: [frozen.get(a, 1.0)] * counts[a] for a in counts if counts[a] > 0}
            if place_bids:
                r = allocate_placement(place_bids, cluster.n_nodes, cluster.gpus_per_node,
                                       current={a: cur_counts.get(a, 0) for a in place_bids},
                                       current_node={a: cur_home.get(a) for a in place_bids},
                                       migrate_cost=MIGRATE_COST)
                placed = {j.jid: r.allocation.get(j.jid, 0) for j in active}
                new_home = {j.jid: r.detail["node_of"].get(j.jid) for j in active}
            else:
                placed = {j.jid: 0 for j in active}
                new_home = {j.jid: None for j in active}
            ploss = sum(counts.values()) - sum(placed.values())
        else:
            placed, new_home, ploss = place_sticky(counts, cluster, cur_home)
        loss_sum += ploss
        busy_steps += 1

        # --- advance: margin GPUs buy spike-absorbing speed -------------------------------------
        busy_sum += sum(placed.get(j.jid, 0) for j in active) / cluster.total
        for j in active:
            held[j.jid] = placed.get(j.jid, 0)
            home[j.jid] = new_home.get(j.jid) if held[j.jid] > 0 else None
            c0 = cap0(j)
            g = held[j.jid]
            if c0 == 0:
                rate = 1.0
            else:
                ceil_use = c0 + (useful[j.jid] if phase_of(j) == "train" else 0)
                rate = min(g, ceil_use) / c0
            progress[j.jid] += rate
            while done_at[j.jid] is None and progress[j.jid] >= work[j.jid][pidx[j.jid]] - 1e-9:
                progress[j.jid] -= work[j.jid][pidx[j.jid]]
                pidx[j.jid] += 1
                if pidx[j.jid] >= len(j.phases):
                    done_at[j.jid] = t
                    break
            if done_at[j.jid] is not None:
                held[j.jid] = 0
                home[j.jid] = None

    def violated(j):
        return done_at[j.jid] is None or done_at[j.jid] > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    fin = [j for j in jobs if done_at[j.jid] is not None]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "ploss": loss_sum / max(busy_steps, 1),
        "finished": float(len(fin)),
        "fallback_rate": n_fallback / max(n_decisions, 1),
    }


STRATEGIES = [                              # (name, negotiate_on, place)
    ("floor",       False, "sticky"),
    ("floor+ILP",   False, "ilp"),
    ("nego+sticky", True,  "sticky"),
    ("pipeline",    True,  "ilp"),
]


def sweep(node_counts, gpus_per_node, n_jobs, horizon, seeds, scale, spike_max,
          use_llm, model, arrival_window=None) -> None:
    dist = load_uncertainty_distribution()
    cache: dict = {}
    tag = "rule" if not use_llm else model
    # Compress arrivals into `arrival_window` (make_workload spreads over 0.6× its horizon arg) so
    # many jobs are concurrent and whole-node train jobs genuinely fragment the cluster — the regime
    # where placement bites (mirrors placement_sim.py). Deadlines stay relative to each job's work.
    aw = arrival_window or horizon
    print(f"\n{'='*92}")
    print(f"FULL LOCKED PIPELINE — negotiate → committed-auction → ILP placement; agents={tag}")
    print(f"{'='*92}")
    print(f"{n_jobs} jobs, horizon {horizon}, arrivals≤{int(aw*0.6)}, mean of {len(seeds)} seeds | "
          f"spike_max={spike_max} scale={scale} | nodes×{gpus_per_node} GPUs | migrate={MIGRATE_COST}")
    print("Lower SLA/prodSLA/ploss = better; util higher = better. "
          "floor=no-nego+sticky; pipeline=nego+ILP.\n")
    header = (f"{'cluster':>8}  {'strategy':<12} {'SLA':>7} {'prodSLA':>8} {'util':>6} "
              f"{'ploss':>6} {'fb':>5} {'done':>8}")
    for nn in node_counts:
        cluster = Cluster(nn, gpus_per_node)
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        for name, nego, place in STRATEGIES:
            acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "ploss": 0.0,
                   "finished": 0.0, "fallback_rate": 0.0}
            for s in seeds:
                jobs = make_workload(n_jobs, s, aw)
                u_map, spike_map = assign(jobs, s, dist, spike_max)
                r = simulate(jobs, cluster, horizon, u_map, spike_map, scale, spike_max,
                             negotiate_on=nego, place=place, use_llm=use_llm, model=model,
                             cache=cache)
                for k in acc:
                    acc[k] += r[k]
            results.append((name, {k: v / len(seeds) for k, v in acc.items()}))
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        tagc = f"{nn}x{gpus_per_node}={cluster.total}"
        for name, r in results:
            s1 = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p1 = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{tagc:>8}  {name:<12} {r['sla']:>6.1%}{s1}{r['prod_sla']:>7.1%}{p1}"
                  f"{r['util']:>6.0%} {r['ploss']:>6.2f} {r['fallback_rate']:>4.0%} "
                  f"{r['finished']:>4.1f}/{n_jobs:<3}")
        print()
    print("'*' = best (lowest) at that cluster. floor+ILP vs floor = ILP placement value; "
          "nego+sticky vs floor = negotiation value; pipeline = both.")
    if use_llm:
        save_cache(cache)


def main() -> None:
    ap = argparse.ArgumentParser(description="Full locked pipeline (nego → committed → ILP place)")
    ap.add_argument("--llm", action="store_true", help="use qwen agents (needs Ollama)")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--spike", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--window", type=int, default=80, help="arrival-compression window (contention)")
    a = ap.parse_args()
    sweep([2, 3, 4, 6], gpus_per_node=8, n_jobs=a.jobs, horizon=400, seeds=list(range(a.seeds)),
          scale=a.scale, spike_max=a.spike, use_llm=a.llm, model=a.model, arrival_window=a.window)


if __name__ == "__main__":
    main()
