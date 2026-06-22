"""
Stage-2, Exp 14: does adding a SUPPLY-SIDE agent improve quality of service?

Everything before this (Exp 9-13, pins/negotiation_sim.py) is DEMAND-ONLY: jobs bid /
declare a priority class into a deterministic auctioneer. The thesis (Research/CLAUDE.md,
research_plan.md) proposes a TWO-SIDED negotiation: a demand agent (jobs) AND a supply agent
(the resource pool) with deliberately asymmetric objectives. This module builds the first
supply agent and asks the only question that matters: *does the supply side lower the
SLA-violation rate (at a given utilisation) over the no-supply-agent committed-auction
baseline (the Exp-11 winner)?*

Supply lever = HEADROOM RESERVATION
-----------------------------------
The supply agent holds back `R` GPUs from best-effort jobs so a late-arriving PROD job lands
on idle capacity instead of preempting a running job. The reserved level `R` is the
deterministic stand-in for the *negotiated* outcome: the demand side pushes R -> 0 (use all
capacity now), the supply side pushes R > 0 (protect headroom for future high-priority load).
(The LLM that SETS and JUSTIFIES R, like llm_priority did for committed priority, is the next
experiment; this one establishes whether the mechanism can help at all.)

The load-bearing subtlety
-------------------------
In this simulator preemption is FREE (Job.step loses no progress when GPUs drop) and
committed-auction already serialises prod jobs to the front, so a late prod job already
preempts best-effort at zero cost -> under free preemption, reservation can only HURT (idle
GPUs, no benefit). The supply agent's headroom only buys something when preemption is COSTLY
(realistic: checkpoint / rescale rollback). So `simulate_pre` adds a `preempt_penalty` knob and
we run BOTH regimes; the regime-dependence is the finding. Hypothesis: with costly preemption,
reservation trades a little utilisation for fewer preemption stalls (better overall SLA /
slowdown); with free preemption it does not help.

Pure Python, no LLM / MCP / network — runs in the base .venv, instantly and reproducibly. The
validated Exp 9-13 harness (negotiation_sim.py) is imported, never modified.

Run:  .venv/bin/python -m pins.supply_sim                 # both regimes
      .venv/bin/python -m pins.supply_sim --penalty 1.0   # one regime
      .venv/bin/python -m pins.supply_sim --seeds 16
"""
from __future__ import annotations

import argparse
import random

from pins.negotiation_sim import Job, make_committed_auction, make_workload, sched_greedy

# An allocator here: (bids, total_gpus, current, by_id, t) -> new_alloc. The extra by_id/t
# (vs negotiation_sim's 3-arg scheduler) let the SUPPLY agent read each job's tier and reason
# about future arrivals. Value-blind/demand-only allocators just ignore them.
Bids = dict[str, list[float]]
Alloc = dict[str, int]


# --------------------------------------------------------------------------- #
#  The supply agent: headroom-reserving committed auction                      #
# --------------------------------------------------------------------------- #
def make_reserving_committed(reserve: int, adaptive: bool = False):
    """Committed auction (bid-once, freeze priority, serialise run-to-completion) PLUS a
    supply-side headroom reservation. Two-pass clearing:

      pass 1  serve PROD jobs in frozen-priority order from the full pool.
      pass 2  serve BEST-EFFORT jobs from `max(0, left - R)` -> the held-back R GPUs stay idle
              as headroom a future prod arrival can land on without preempting anyone.

    `adaptive=True`: only reserve while more prod load is still INCOMING (some prod job with
    arrival > t). This is the supply agent using its *predicted load* signal so it does not
    waste headroom once the prod burst is over.

    With R=0 this reduces EXACTLY to committed-auction: on arrival every job is in 'preprocess'
    (cap 1, value=urgency), so the frozen priority equals urgency, and prod (top urgency tier)
    always outranks best-effort -> prod-first two-pass == committed's single priority sort."""
    frozen: dict[str, float] = {}

    def sched(bids: Bids, total_gpus: int, current: Alloc, by_id: dict[str, Job], t: int) -> Alloc:
        for a, curve in bids.items():
            frozen.setdefault(a, sum(curve))            # bid-once: priority set on first sight
        order = sorted(bids, key=lambda a: (-frozen.get(a, 0.0), a))
        prod = [a for a in order if by_id[a].tier == "prod"]
        be = [a for a in order if by_id[a].tier != "prod"]

        alloc = {a: 0 for a in bids}
        left = total_gpus
        for a in prod:                                  # pass 1: prod first, full pool
            g = min(len(bids[a]), left)
            alloc[a] = g
            left -= g
            if left <= 0:
                break

        if adaptive:
            more_prod_coming = any(j.tier == "prod" and j.arrival > t for j in by_id.values())
            r = reserve if more_prod_coming else 0
        else:
            r = reserve
        be_pool = max(0, left - r)                      # pass 2: best-effort, minus reserved headroom
        for a in be:
            if be_pool <= 0:
                break
            g = min(len(bids[a]), be_pool)
            alloc[a] = g
            be_pool -= g
            left -= g
        return alloc

    return sched


# Demand-only / value-blind baselines wrapped to the (…, by_id, t) signature so one simulator
# drives them all. A fresh allocator is built per run so the committed auction's frozen-priority
# map starts clean.
def greedy_factory():
    return lambda bids, tot, cur, by_id, t: sched_greedy(bids, tot, cur)


def committed_factory():
    base = make_committed_auction()
    return lambda bids, tot, cur, by_id, t: base(bids, tot, cur)


# --------------------------------------------------------------------------- #
#  Simulator with a preemption cost                                            #
# --------------------------------------------------------------------------- #
def simulate_pre(jobs_proto: list[Job], allocator, total_gpus: int, horizon: int,
                 preempt_penalty: float) -> dict:
    """Run one allocator over a fresh copy of the workload. Identical to
    negotiation_sim.simulate except for the preemption cost: a job whose allocation DROPS this
    tick (`new < prev`, was running) rolls back progress `preempt_penalty * (prev-new)/capacity`
    — a checkpoint/rescale stall. `preempt_penalty=0` reproduces the Exp 9-13 behaviour."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    current: Alloc = {}
    busy_sum = 0.0
    busy_steps = 0

    for t in range(horizon):
        active = [j for j in jobs if j.active(t)]
        if not active:
            current = {}
            if all(j.done_at is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        bids = {j.jid: j.bid() for j in active}
        cur = {jid: current.get(jid, 0) for jid in bids}
        alloc = allocator(bids, total_gpus, cur, by_id, t)
        busy_sum += sum(alloc.values()) / total_gpus
        busy_steps += 1
        for j in active:
            newg = alloc.get(j.jid, 0)
            prev = current.get(j.jid, 0)
            cap = j.capacity()
            if preempt_penalty > 0 and prev > 0 and newg < prev and cap > 0:
                j.progress = max(0.0, j.progress - preempt_penalty * (prev - newg) / cap)
            j.step(newg, t)
        current = {jid: alloc.get(jid, 0) for jid in by_id}

    finished = [j for j in jobs if j.done_at is not None]

    def violated(j: Job) -> bool:
        return j.done_at is None or j.done_at > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    slow = [(j.done_at - j.arrival) / j.nominal for j in finished if j.nominal > 0]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "slowdown": sum(slow) / max(len(slow), 1),
        "finished": len(finished),
        "n_jobs": len(jobs),
    }


# --------------------------------------------------------------------------- #
#  Multi-seed sweep                                                            #
# --------------------------------------------------------------------------- #
def strategies(reserves: list[int]):
    """(name, allocator_factory) rows: the two foils + reservation at each R, static & adaptive.
    reserve-static(0) doubles as a cross-check that it reproduces committed-auction."""
    rows = [
        ("greedy-FIFO", greedy_factory),
        ("committed-auction", committed_factory),
    ]
    for r in reserves:
        rows.append((f"reserve-static(R={r})", lambda r=r: make_reserving_committed(r, adaptive=False)))
    for r in reserves:
        rows.append((f"reserve-adaptive(R={r})", lambda r=r: make_reserving_committed(r, adaptive=True)))
    return rows


def mean_over_seeds(factory, total_gpus: int, horizon: int, n_jobs: int, seeds: list[int],
                    preempt_penalty: float) -> dict:
    accs = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "slowdown": 0.0, "finished": 0.0}
    for s in seeds:
        jobs = make_workload(n_jobs, s, horizon)
        r = simulate_pre(jobs, factory(), total_gpus, horizon, preempt_penalty)
        for k in accs:
            accs[k] += r[k]
    n = len(seeds)
    out = {k: v / n for k, v in accs.items()}
    out["n_jobs"] = float(n_jobs)
    return out


def sweep(pools: list[int], reserves: list[int], n_jobs: int, horizon: int,
          seeds: list[int], preempt_penalty: float) -> None:
    rows = strategies(reserves)
    print(f"\n{'='*84}")
    print(f"PREEMPTION PENALTY = {preempt_penalty}   "
          f"({'FREE preemption — reservation expected NOT to help' if preempt_penalty == 0 else 'COSTLY preemption — the regime where headroom can pay off'})")
    print(f"{'='*84}")
    print(f"{n_jobs} jobs, horizon {horizon}, mean of {len(seeds)} seeds. "
          f"Lower SLA/prodSLA/slowdown = better; util is the cost of reserving.\n")
    header = (f"{'pool':>4}  {'strategy':<22} {'SLA':>7} {'prodSLA':>8} "
              f"{'util':>6} {'slowdown':>9} {'done':>7}")
    for gpus in pools:
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        results = [(name, mean_over_seeds(f, gpus, horizon, n_jobs, seeds, preempt_penalty))
                   for name, f in rows]
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<22} {r['sla']:>6.1%}{s}{r['prod_sla']:>7.1%}{p}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['finished']:>4.1f}/{r['n_jobs']:<3.0f}")
        print()
    print("'*' = best (lowest) at that pool. reserve-static(R=0) == committed-auction (sanity).")


# --------------------------------------------------------------------------- #
#  Rigid-incumbent regime (non-malleable jobs)                                 #
# --------------------------------------------------------------------------- #
# In the malleable sim above, a late prod job preempts a best-effort incumbent for FREE, so
# reserved headroom is redundant. Here incumbents are RIGID: once a job holds GPUs it keeps them
# run-to-completion and is NEVER involuntarily preempted — it only shrinks VOLUNTARILY when its
# own phase needs fewer (e.g. train cap-8 -> eval cap-2), releasing the excess. Newcomers and
# growth draw ONLY from genuinely free GPUs. This is the gang-scheduled / checkpoint-boundary
# model real HPC schedulers face, and the one place the mechanism predicts headroom should pay:
# a late prod job can't bump a rigid best-effort job, so it lands on reserved headroom or WAITS.
def simulate_rigid(jobs_proto: list[Job], total_gpus: int, horizon: int, strategy: str,
                   reserve: int = 0, adaptive: bool = False, reserve_fn=None) -> dict:
    """One run under rigid (non-preemptable) incumbents. `strategy` in
    {'greedy','committed','reserving'}; reservation only applies to 'reserving' and caps how
    many free GPUs BEST-EFFORT jobs may claim, leaving `reserve` as prod headroom.

    `reserve_fn(contention, incoming_prod) -> int` (e.g. the LLM supply agent) overrides the
    fixed `reserve`/`adaptive` policy: each tick the current contention bucket and incoming-prod
    bucket are computed and handed to it to DECIDE the reservation — the supply agent reasoning
    over live state instead of a hard-coded R."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    held = {j.jid: 0 for j in jobs}                  # GPUs locked to a running job (no preemption)
    frozen: dict[str, float] = {}
    busy_sum = 0.0
    busy_steps = 0

    def grant(order, free_avail, spendable):
        """Give each job in `order` up to its remaining want, bounded by free GPUs AND the
        group's spendable budget (best-effort's budget is capped to leave reserved headroom)."""
        for j in order:
            give = min(j.capacity() - held[j.jid], free_avail, spendable)
            if give > 0:
                held[j.jid] += give
                free_avail -= give
                spendable -= give
        return free_avail

    for t in range(horizon):
        active = [j for j in jobs if j.active(t)]
        if not active:
            if all(j.done_at is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        for j in active:
            frozen.setdefault(j.jid, sum(j.bid()))    # bid-once priority (preprocess => urgency)
            if held[j.jid] > j.capacity():            # voluntary shrink at a phase boundary
                held[j.jid] = j.capacity()
        free = total_gpus - sum(held[j.jid] for j in active)
        wanters = [j for j in active if held[j.jid] < j.capacity()]

        if strategy == "greedy":
            grant(sorted(wanters, key=lambda j: j.jid), free, free)
        elif strategy == "committed":
            grant(sorted(wanters, key=lambda j: (-frozen[j.jid], j.jid)), free, free)
        else:                                         # reserving: prod first (uncapped), then be
            if reserve_fn is not None:                # supply agent reasons over live state
                demand = sum(j.capacity() for j in active)
                ratio = demand / total_gpus
                contention = "scarce" if ratio >= 1.8 else "moderate" if ratio >= 1.0 else "ample"
                n_inc = sum(1 for jj in jobs if jj.tier == "prod" and jj.arrival > t)
                incoming = "none" if n_inc == 0 else "few" if n_inc <= 2 else "many"
                r = reserve_fn(contention, incoming)
            else:
                r = reserve
                if adaptive and not any(jj.tier == "prod" and jj.arrival > t for jj in jobs):
                    r = 0                             # stop reserving once no prod load is incoming
            prod_w = sorted([j for j in wanters if j.tier == "prod"],
                            key=lambda j: (-frozen[j.jid], j.jid))
            be_w = sorted([j for j in wanters if j.tier != "prod"],
                          key=lambda j: (-frozen[j.jid], j.jid))
            free = grant(prod_w, free, free)
            grant(be_w, free, max(0, free - r))

        busy_sum += sum(held[j.jid] for j in active) / total_gpus
        busy_steps += 1
        for j in active:
            j.step(held[j.jid], t)
            if j.done_at is not None:
                held[j.jid] = 0

    finished = [j for j in jobs if j.done_at is not None]

    def violated(j: Job) -> bool:
        return j.done_at is None or j.done_at > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    slow = [(j.done_at - j.arrival) / j.nominal for j in finished if j.nominal > 0]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "slowdown": sum(slow) / max(len(slow), 1),
        "finished": len(finished),
        "n_jobs": float(len(jobs)),
    }


def sweep_rigid(pools, reserves, n_jobs, horizon, seeds) -> None:
    rows = [("greedy-FIFO", "greedy", 0, False),
            ("committed-auction", "committed", 0, False)]
    rows += [(f"reserve-static(R={r})", "reserving", r, False) for r in reserves]
    rows += [(f"reserve-adaptive(R={r})", "reserving", r, True) for r in reserves]

    print(f"\n{'='*84}")
    print("RIGID INCUMBENTS — running jobs are non-preemptable; newcomers use only FREE GPUs")
    print(f"{'='*84}")
    print(f"{n_jobs} jobs, horizon {horizon}, mean of {len(seeds)} seeds. "
          "This is the regime where reserved headroom CAN help a late prod job land.\n")
    header = (f"{'pool':>4}  {'strategy':<22} {'SLA':>7} {'prodSLA':>8} "
              f"{'util':>6} {'slowdown':>9} {'done':>7}")
    for gpus in pools:
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        results = []
        for name, strat, res, adap in rows:
            acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "slowdown": 0.0, "finished": 0.0}
            for s in seeds:
                r = simulate_rigid(make_workload(n_jobs, s, horizon), gpus, horizon, strat, res, adap)
                for k in acc:
                    acc[k] += r[k]
            results.append((name, {k: v / len(seeds) for k, v in acc.items()}))
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<22} {r['sla']:>6.1%}{s}{r['prod_sla']:>7.1%}{p}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['finished']:>4.1f}/{n_jobs:<3}")
        print()
    print("'*' = best (lowest) at that pool. reserve-static(R=0) == committed-auction (sanity).")


# --------------------------------------------------------------------------- #
#  LLM supply agent (Exp 14): the LLM DECIDES the reservation from live state   #
# --------------------------------------------------------------------------- #
def make_llm_reserve_fn(use_llm: bool, model: str, cache: dict, trace: list, seen: set):
    """Wrap llm_reserve as a `(contention, incoming_prod) -> int` policy for simulate_rigid.
    The LLM returns a categorical level (none/light/heavy); code maps it to GPUs. Cached per
    discretised state, so a whole multi-seed sweep costs only a handful of model calls. Records
    each distinct decision + justification into `trace` (the interpretability artifact)."""
    from pins.llm_agent import llm_reserve, reserve_amount, reserve_state_key

    def fn(contention: str, incoming_prod: str) -> int:
        ctx = {"contention": contention, "incoming_prod": incoming_prod}
        d = llm_reserve(ctx, use_llm=use_llm, model=model, cache=cache)
        key = f"{model}|{reserve_state_key(ctx)}"
        if key not in seen:
            seen.add(key)
            trace.append({"model": model, "state": reserve_state_key(ctx),
                          "reserve": d["reserve"], "justification": d["justification"],
                          "_source": d["_source"]})
        return reserve_amount(d["reserve"])

    return fn


def sweep_rigid_llm(pools, n_jobs, horizon, seeds, models, use_llm) -> None:
    """Rigid regime: compare the no-supply baseline, the deterministic adaptive oracle, the rule
    supply agent, and the LLM supply agent at each model size — does the LLM match the oracle's
    win AND correctly DECLINE to reserve where reservation hurts?"""
    import json
    import os

    from pins.llm_agent import save_cache

    cache: dict = {}
    trace: list = []
    seen: set = set()
    # name, (strategy, reserve, adaptive, reserve_fn)
    rows = [
        ("committed-auction", ("committed", 0, False, None)),
        ("reserve-adaptive(R=1)", ("reserving", 1, True, None)),
        ("rule-supply-agent", ("reserving", 0, False,
                               make_llm_reserve_fn(False, "rule", cache, trace, seen))),
    ]
    for m in models:
        fn = make_llm_reserve_fn(use_llm, m, cache, trace, seen)
        rows.append((f"llm-supply({m})", ("reserving", 0, False, fn)))

    print(f"\n{'='*88}")
    print("RIGID INCUMBENTS + LLM SUPPLY AGENT — the LLM decides how much headroom to reserve")
    print(f"{'='*88}")
    print(f"{n_jobs} jobs, horizon {horizon}, mean of {len(seeds)} seeds. The LLM emits a "
          "categorical level (none/light/heavy); code maps it to GPUs.\n")
    header = (f"{'pool':>4}  {'strategy':<24} {'SLA':>7} {'prodSLA':>8} "
              f"{'util':>6} {'slowdown':>9} {'done':>7}")
    for gpus in pools:
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        results = []
        for name, (strat, res, adap, rfn) in rows:
            acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "slowdown": 0.0, "finished": 0.0}
            for s in seeds:
                r = simulate_rigid(make_workload(n_jobs, s, horizon), gpus, horizon, strat,
                                   res, adap, reserve_fn=rfn)
                for k in acc:
                    acc[k] += r[k]
            results.append((name, {k: v / len(seeds) for k, v in acc.items()}))
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<24} {r['sla']:>6.1%}{s}{r['prod_sla']:>7.1%}{p}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['finished']:>4.1f}/{n_jobs:<3}")
        print()

    print("'*' = best (lowest) at that pool.\n")
    print("=== supply-agent decisions (state -> reserve level + justification) ===")
    for d in sorted(trace, key=lambda x: (x["model"], x["state"])):
        print(f"  [{d['model']:<12} {d['state']:<16}] -> {d['reserve']:<6} "
              f"({d['_source']})  {d['justification']}")
    if use_llm:
        save_cache(cache)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_supply_llm.json")
        with open(out, "w") as f:
            json.dump({"models": models, "decisions": trace}, f, indent=2)
        print(f"\nLLM: {len(trace)} decisions recorded; trace -> {out}")


# --------------------------------------------------------------------------- #
#  Exp 15: MIXED malleable + rigid incumbents                                   #
# --------------------------------------------------------------------------- #
# Exp 14 ran two ENDPOINTS: all-malleable (simulate_pre: reservation redundant — a late prod
# job preempts for free) and all-rigid (simulate_rigid: reservation wins but pays idle-GPU
# utilisation). Real HPC is a MIX: some apps resize at runtime (malleable MPI / DMR-DROM),
# most don't (rigid). This experiment fills in the axis between the two endpoints and asks
# whether KNOWING which incumbents are malleable lets the supply agent reserve smarter.
#
# The lever this exposes: the supply agent has TWO sources of headroom for a late prod job —
#   (1) idle GPUs it RESERVED (costs utilisation), and
#   (2) GPUs it can RECLAIM by shrinking a malleable best-effort incumbent on demand (free).
# A malleability-AWARE agent reserves idle headroom only against the RIGID fraction (which it
# cannot reclaim) and lends the reserved pool to malleable best-effort jobs (which it can claw
# back later). A BLIND agent holds R idle against everyone — paying utilisation for headroom it
# did not need. Hypothesis: aware >= blind everywhere, the gap GROWING with the malleable
# fraction phi; aware keeps the rigid-reservation prodSLA win while recovering the util cost.


def malleability_map(jobs: list[Job], frac: float, seed: int) -> dict[str, bool]:
    """Mark a fraction `frac` of jobs malleable (reclaimable), the rest rigid. Each job is
    assigned a fixed uniform draw u_j; malleable iff u_j < frac — so raising phi only ADDS
    malleable jobs (nested), keeping the sweep monotone. phi=0 -> all rigid; phi=1 -> all
    malleable (the two Exp-14 endpoints)."""
    rng = random.Random(f"mal-{seed}")
    u = {j.jid: rng.random() for j in jobs}
    return {jid: (val < frac) for jid, val in u.items()}


def simulate_mixed(jobs_proto: list[Job], total_gpus: int, horizon: int, strategy: str,
                   malleable: dict[str, bool], reserve: int = 0, adaptive: bool = False,
                   aware: bool = False, reserve_fn=None) -> dict:
    """Incumbents are a MIX: `malleable[jid]` jobs can be shrunk involuntarily (reclaimed) by a
    higher-priority job; rigid jobs cannot (they only shrink voluntarily at a phase boundary).
    Reclaim is free (no rollback), matching Exp-14A's free-preemption malleable baseline.

    Generalises both Exp-14 simulators (verified by `check_endpoints`):
      * malleable all-False  ==  simulate_rigid   (no job is ever reclaimable)
      * malleable all-True   ==  the Exp-14A 'reservation redundant' limit (prod reclaims freely)

    `strategy` in {'greedy','committed','reserving'}. 'reserving' holds R GPUs as prod headroom:
      * aware=False (blind): keep R idle away from ALL best-effort.
      * aware=True : keep R idle away from RIGID best-effort only; malleable best-effort may use
        it, because a late prod job can RECLAIM those GPUs on demand."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    held = {j.jid: 0 for j in jobs}                  # GPUs locked to a running job
    frozen: dict[str, float] = {}
    busy_sum = 0.0
    busy_steps = 0

    def grow(order, pool):
        """Grow jobs in `order` into `pool` free GPUs (no reclaim). Returns GPUs left."""
        for j in order:
            give = min(j.capacity() - held[j.jid], pool)
            if give > 0:
                held[j.jid] += give
                pool -= give
        return pool

    def reclaim_for(j, need, active):
        """Pull up to `need` GPUs from held MALLEABLE incumbents of strictly lower frozen
        priority than j (lowest priority first) — the second, free source of headroom."""
        if need <= 0:
            return
        donors = sorted([d for d in active if held[d.jid] > 0 and malleable[d.jid]
                         and frozen[d.jid] < frozen[j.jid]],
                        key=lambda d: (frozen[d.jid], d.jid))
        for d in donors:
            take = min(need, held[d.jid])
            held[d.jid] -= take
            held[j.jid] += take
            need -= take
            if need <= 0:
                break

    for t in range(horizon):
        active = [j for j in jobs if j.active(t)]
        if not active:
            if all(j.done_at is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        for j in active:
            frozen.setdefault(j.jid, sum(j.bid()))   # bid-once priority (preprocess => urgency)
            if held[j.jid] > j.capacity():            # voluntary shrink at a phase boundary
                held[j.jid] = j.capacity()
        free = total_gpus - sum(held[j.jid] for j in active)
        wanters = [j for j in active if held[j.jid] < j.capacity()]

        if strategy == "greedy":                      # value-blind FIFO, free GPUs only (the foil)
            grow(sorted(wanters, key=lambda j: j.jid), free)
        elif strategy == "committed":                 # priority order; reclaim lower-prio malleable
            for j in sorted(wanters, key=lambda j: (-frozen[j.jid], j.jid)):
                free = grow([j], free)
                reclaim_for(j, j.capacity() - held[j.jid], active)
        else:                                         # reserving: prod first, then be minus headroom
            if reserve_fn is not None:                # supply agent reasons over live state
                demand = sum(j.capacity() for j in active)
                ratio = demand / total_gpus
                contention = "scarce" if ratio >= 1.8 else "moderate" if ratio >= 1.0 else "ample"
                n_inc = sum(1 for jj in jobs if jj.tier == "prod" and jj.arrival > t)
                incoming = "none" if n_inc == 0 else "few" if n_inc <= 2 else "many"
                r = reserve_fn(contention, incoming)
            else:
                r = reserve
                if adaptive and not any(jj.tier == "prod" and jj.arrival > t for jj in jobs):
                    r = 0                             # stop reserving once no prod load is incoming
            prod_w = sorted([j for j in wanters if j.tier == "prod"],
                            key=lambda j: (-frozen[j.jid], j.jid))
            be_w = sorted([j for j in wanters if j.tier != "prod"],
                          key=lambda j: (-frozen[j.jid], j.jid))
            for j in prod_w:                          # prod: free GPUs + reclaim malleable on demand
                free = grow([j], free)
                reclaim_for(j, j.capacity() - held[j.jid], active)
            free = total_gpus - sum(held[j.jid] for j in active)   # authoritative free after prod
            if aware:
                # malleable best-effort may use the reserved pool (reclaimable later); rigid
                # best-effort is capped to leave R idle as the only headroom against rigidity.
                grow([j for j in be_w if malleable[j.jid]], free)
                free = total_gpus - sum(held[j.jid] for j in active)
                grow([j for j in be_w if not malleable[j.jid]], max(0, free - r))
            else:
                grow(be_w, max(0, free - r))          # blind: hold R idle away from ALL best-effort

        busy_sum += sum(held[j.jid] for j in active) / total_gpus
        busy_steps += 1
        for j in active:
            j.step(held[j.jid], t)
            if j.done_at is not None:
                held[j.jid] = 0

    finished = [j for j in jobs if j.done_at is not None]

    def violated(j: Job) -> bool:
        return j.done_at is None or j.done_at > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    slow = [(j.done_at - j.arrival) / j.nominal for j in finished if j.nominal > 0]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "slowdown": sum(slow) / max(len(slow), 1),
        "finished": len(finished),
        "n_jobs": float(len(jobs)),
    }


def check_endpoints(n_jobs: int, horizon: int, seeds: list[int]) -> bool:
    """Sanity: simulate_mixed at phi=0 (all rigid) must reproduce simulate_rigid EXACTLY, for
    every strategy. This validates that the mixed simulator is a faithful generalisation and the
    rigid endpoint did not drift."""
    ok = True
    cases = [("greedy", "greedy", 0, False),
             ("committed", "committed", 0, False),
             ("reserving R=2 adaptive", "reserving", 2, True)]
    for pool in (8, 12):
        for label, strat, res, adap in cases:
            for s in seeds:
                jobs = make_workload(n_jobs, s, horizon)
                mal = {j.jid: False for j in jobs}          # phi = 0: all rigid
                a = simulate_mixed(jobs, pool, horizon, strat, mal, res, adap, aware=False)
                b = simulate_rigid(make_workload(n_jobs, s, horizon), pool, horizon, strat, res, adap)
                for k in ("sla", "prod_sla", "util", "slowdown", "finished"):
                    if abs(a[k] - b[k]) > 1e-9:
                        ok = False
                        print(f"  MISMATCH pool={pool} {label} seed={s} {k}: "
                              f"mixed={a[k]:.6f} rigid={b[k]:.6f}")
    print(f"endpoint check (phi=0 == simulate_rigid): {'PASS' if ok else 'FAIL'}")
    return ok


def sweep_mixed(pools, fracs, R, n_jobs, horizon, seeds, adaptive) -> None:
    """For each pool, show prodSLA / util as the malleable fraction phi grows, for the no-supply
    baseline (committed) vs the BLIND and AWARE reserving supply agents. The blind-vs-aware gap
    across phi is the result: does malleability-awareness recover the reservation's util cost
    while keeping its prodSLA win?"""
    tag = "adaptive" if adaptive else "static"
    print(f"\n{'='*84}")
    print(f"MIXED malleable/rigid incumbents — supply agent reserves R={R} ({tag})")
    print(f"{'='*84}")
    print(f"{n_jobs} jobs, horizon {horizon}, mean of {len(seeds)} seeds. Cells = prodSLA% / util% "
          "(lower prodSLA better, higher util better).")
    print("phi = fraction of incumbents that are MALLEABLE (reclaimable). phi=0 all rigid (Exp14B); "
          "phi=1 all malleable (Exp14A).\n")
    strats = [
        ("committed (no supply)", "committed", 0, False),
        ("reserve-BLIND",         "reserving", R, False),
        ("reserve-AWARE",         "reserving", R, True),
    ]
    for pool in pools:
        head = f"{'strategy':<22}" + "".join(f"{f'phi={p:.2f}':>14}" for p in fracs)
        print("-" * len(head))
        print(f"pool {pool}")
        print("-" * len(head))
        print(head)
        print("-" * len(head))
        for name, strat, res, aware in strats:
            cells = []
            for phi in fracs:
                acc = {"prod_sla": 0.0, "util": 0.0}
                for s in seeds:
                    jobs = make_workload(n_jobs, s, horizon)
                    mal = malleability_map(jobs, phi, s)
                    r = simulate_mixed(jobs, pool, horizon, strat, mal, res, adaptive, aware=aware)
                    acc["prod_sla"] += r["prod_sla"]
                    acc["util"] += r["util"]
                n = len(seeds)
                cells.append(f"{acc['prod_sla']/n:>6.0%}/{acc['util']/n:<6.0%}")
            print(f"{name:<22}" + "".join(f"{c:>14}" for c in cells))
        print()
    print("Read across a row: how each strategy responds as more incumbents become malleable.")
    print("AWARE - BLIND gap should widen with phi (aware stops paying idle-util for reclaimable headroom).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 14 — supply-agent (headroom reservation) eval")
    ap.add_argument("--penalty", type=float, default=None,
                    help="run ONE regime at this preemption penalty (default: run both 0.0 and 1.0)")
    ap.add_argument("--rigid", action="store_true",
                    help="run the RIGID-incumbent (non-preemptable) deterministic regime")
    ap.add_argument("--llm", action="store_true",
                    help="rigid regime + LLM supply agent across model sizes (needs Ollama)")
    ap.add_argument("--no-llm", action="store_true",
                    help="with --llm: use the rule supply agent instead of calling Ollama")
    ap.add_argument("--models", default="qwen2.5:3b,qwen2.5:7b,qwen2.5:14b",
                    help="comma-separated Ollama models for the LLM supply agent")
    ap.add_argument("--mixed", action="store_true",
                    help="Exp 15: MIXED malleable/rigid incumbents, blind vs aware reservation")
    ap.add_argument("--check", action="store_true",
                    help="Exp 15: endpoint sanity only (phi=0 must reproduce simulate_rigid)")
    ap.add_argument("--reserve", type=int, default=2, help="(--mixed) headroom R the supply agent holds")
    ap.add_argument("--static", action="store_true",
                    help="(--mixed) static reservation instead of the default adaptive policy")
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds to average")
    a = ap.parse_args()

    seeds = list(range(a.seeds))
    pools = [4, 6, 8, 12, 20]
    reserves = [0, 1, 2, 3]
    n_jobs, horizon = 16, 300
    if a.check:
        check_endpoints(n_jobs, horizon, seeds)
        return
    if a.mixed:
        check_endpoints(n_jobs, horizon, seeds)
        sweep_mixed([6, 8, 12], [0.0, 0.25, 0.5, 0.75, 1.0], a.reserve,
                    n_jobs, horizon, seeds, adaptive=not a.static)
        return
    if a.llm:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
        sweep_rigid_llm(pools, n_jobs, horizon, seeds, models, use_llm=not a.no_llm)
        return
    if a.rigid:
        sweep_rigid(pools, reserves, n_jobs, horizon, seeds)
        return
    penalties = [a.penalty] if a.penalty is not None else [0.0, 1.0]
    for pen in penalties:
        sweep(pools, reserves, n_jobs, horizon, seeds, pen)


if __name__ == "__main__":
    main()
