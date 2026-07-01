"""
The MERGED two-sided simulator (build task #4) — both levers in ONE world.

Exp 14 (supply_sim.py) and Exp 16-17 (uncertainty_sim.py) each isolate a single lever in its own
simulator: supply-side headroom RESERVE (rigid incumbents) and demand-side safety MARGIN
(stochastic spikes). The two-sided thesis only bites when BOTH levers draw on the SAME free pool,
so this module merges the two worlds:

  * rigid, non-preemptable incumbents + a reserve held idle for incoming prod  (from simulate_rigid)
  * train work that SPIKES, where margin GPUs buy the speed to absorb it       (from simulate_stochastic)

Margin and reserve now COMPETE for the same GPUs — exactly the tension the bounded negotiation
(pins/negotiation_protocol.py) resolves. Four policies set (per-job margin, reserve R) each tick
and we score them on the identical stochastic, rigid workload:

  * no-llm     — margins 0, R 0 (the cheap point-forecast heuristic; the floor)
  * isolated   — llm_margin per job + llm_reserve, decided INDEPENDENTLY (today's state of the art)
  * negotiated — the bounded two-sided protocol (margin vs reserve resolved jointly)
  * single-llm — one llm_joint agent decides both at once (Open-Q #5 control)

Each tick builds every job's Stage1Facts and bridges them to the agent ctx (pins/bridge.py),
wiring build task #1 into the loop. Default use_llm=False -> the rule fallback, so the whole sweep
runs with NO Ollama, fast and deterministic. --llm calls qwen. The validated Exp 9-17 harness is
imported, never modified.

Run:  .venv/bin/python -m pins.two_sided_sim                 # rule-fallback comparison (no Ollama)
      .venv/bin/python -m pins.two_sided_sim --llm           # qwen agents
      .venv/bin/python -m pins.two_sided_sim --seeds 16
"""
from __future__ import annotations

import argparse
import json
import os

from pins import bridge
from pins.llm_agent import (llm_margin, llm_reserve, reserve_amount, load_cache, save_cache)
from pins.negotiation_protocol import (DemandJob, HEDGE_GPUS, negotiate, single_llm_plan)
from pins.negotiation_sim import Job, make_workload
from pins.predictor import PHASE_PROFILES
from pins.uncertainty_sim import (assign, assign_gpu, load_gpu_distribution,
                                   load_uncertainty_distribution, true_need)

HERE = os.path.dirname(os.path.abspath(__file__))
DORDER = {"ahead": 0, "ontrack": 1, "behind": 2}


# --------------------------------------------------------------------------- #
#  Per-job Stage-1 facts -> bridged demand ctx (build task #1 in the loop)       #
# --------------------------------------------------------------------------- #
def job_facts(job: Job, u: float, spike_max: float, req_gpu: int) -> bridge.Stage1Facts:
    """Synthesise the Stage-1 facts a real forecaster would emit for this job: P50 runtime = its
    nominal work (steps), P90 = inflated by the plausible spike (u*spike_max), uncertainty = u,
    and `req_gpu` = the REAL Stage-1 predicted GPU request (predict_gpu P50, drawn per job in the
    sweep). The bridge turns these into the agent's qualitative buckets — incl. the request-size
    bucket the demand agent now negotiates over."""
    p50 = job.nominal
    return bridge.Stage1Facts(
        jid=job.jid, runtime_p50=p50, runtime_p10=p50 * (1 - 0.3 * u),
        runtime_p90=p50 * (1 + u * spike_max),
        uncertainty=u, tier=job.tier, req_gpu=req_gpu)


# --------------------------------------------------------------------------- #
#  The four policies: (demand_jobs, supply_ctx, free, ...) -> (margins, reserve) #
# --------------------------------------------------------------------------- #
def policy_none(demand, supply_ctx, free, **_):
    return {j.jid: 0 for j in demand}, 0, None


def make_policy_isolated(use_llm, model, cache, trace, seen):
    def policy(demand, supply_ctx, free, **_):
        margins = {}
        for j in demand:
            if j.is_train:
                d = llm_margin(j.ctx, use_llm=use_llm, model=model, cache=cache)
                margins[j.jid] = HEDGE_GPUS[d["hedge"]]
                _record(trace, seen, "isolated-demand", j.ctx, d["hedge"], d)
            else:
                margins[j.jid] = 0
        rd = llm_reserve(supply_ctx, use_llm=use_llm, model=model, cache=cache)
        _record(trace, seen, "isolated-supply", supply_ctx, rd["reserve"], rd)
        return margins, reserve_amount(rd["reserve"]), None
    return policy


def make_policy_negotiated(use_llm, model, cache, trace, seen):
    def policy(demand, supply_ctx, free, **_):
        o = negotiate(demand, supply_ctx, free, use_llm=use_llm, model=model, cache=cache)
        _record_outcome(trace, seen, "negotiated", o)
        return o.margins, o.reserve, o
    return policy


def make_policy_single(use_llm, model, cache, trace, seen):
    def policy(demand, supply_ctx, free, **_):
        o = single_llm_plan(demand, supply_ctx, free, use_llm=use_llm, model=model, cache=cache)
        _record_outcome(trace, seen, "single-llm", o)
        return o.margins, o.reserve, o
    return policy


def _record(trace, seen, tag, ctx, level, d):
    key = f"{tag}|{'|'.join(str(v) for v in ctx.values())}"
    if key not in seen:
        seen.add(key)
        trace.append({"policy": tag, "state": key, "decision": level,
                      "why": d["justification"], "_source": d["_source"]})


def _record_outcome(trace, seen, tag, o):
    sig = f"{tag}|agreed={o.agreed}|r={o.reserve}|m={sorted(o.margins.items())}"
    if sig not in seen:
        seen.add(sig)
        trace.append({"policy": tag, "agreed": o.agreed, "rounds": o.rounds,
                      "reserve": o.reserve, "margins": o.margins, "transcript": o.transcript})


# --------------------------------------------------------------------------- #
#  Simulator: rigid incumbents + stochastic spikes, margin buys speed           #
# --------------------------------------------------------------------------- #
def simulate(jobs_proto: list[Job], policy, total_gpus: int, horizon: int,
             u_map: dict, spike_map: dict, scale: int, spike_max: float,
             cap_map: dict[str, int]) -> dict:
    """One run of a policy on a fresh workload copy. Rigid: a running job is never involuntarily
    preempted; it only shrinks VOLUNTARILY to its ceiling (cap0 + this tick's negotiated margin).
    Spikes: a train phase's true work is inflated; margin GPUs grant rate>1 to absorb it, capped at
    the spike's usable parallelism `useful = round(u*scale)`. Deadlines come from NOMINAL work, so
    an unabsorbed spike is what threatens the SLA — and reserved headroom protects late prod jobs."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}     # realised (spiked) work
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}      # extra GPUs a spike can use
    held = {j.jid: 0 for j in jobs}                                  # rigid: locked to the job
    progress = {j.jid: 0.0 for j in jobs}
    pidx = {j.jid: 0 for j in jobs}
    done_at: dict[str, int | None] = {j.jid: None for j in jobs}
    busy_sum = 0.0
    busy_steps = 0
    n_fallback = 0
    n_decisions = 0

    def phase_of(j):
        return j.phases[pidx[j.jid]] if pidx[j.jid] < len(j.phases) else "idle"

    def cap0(j):
        ph = phase_of(j)
        # Train-phase base = the job's REAL Stage-1 predicted GPU request (forecast_cap); other
        # phases keep the profile (preprocess I/O-bound, eval moderate). This is where the
        # predicted requested GPU enters the negotiation as the non-negotiable base.
        return cap_map[j.jid] if ph == "train" else PHASE_PROFILES[ph][0]

    def remaining(j):
        return max(0.0, j.need[pidx[j.jid]] - progress[j.jid]) + sum(j.need[pidx[j.jid] + 1:])

    for t in range(horizon):
        active = [j for j in jobs if j.arrival <= t and done_at[j.jid] is None]
        if not active:
            if all(done_at[j.jid] is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue

        # --- build the demand table + supply ctx (via the bridge) and run the policy ----------
        demand_gpus = sum(cap0(j) for j in active)
        con_supply = bridge.contention_bucket(demand_gpus, total_gpus)
        con_demand = "high" if demand_gpus >= total_gpus else "low"
        n_inc = sum(1 for jj in jobs if jj.tier == "prod" and jj.arrival > t)
        supply_ctx = bridge.reserve_ctx(con_supply, n_inc)
        # Contested slice: only jobs already RUNNING their full base contest the free GPUs for a
        # speed-up margin — a waiting/ramping job can't spend a margin GPU, it needs base first (the
        # auction's job). So the margin table is the running train jobs, contesting `free_now` (the
        # GPUs genuinely free this tick) against the supply reserve. Base demand is not negotiable and
        # never enters `want`, so the negotiation no longer false-fallbacks on base oversubscription.
        demand: list[DemandJob] = []
        for j in active:
            db = bridge.deadline_bucket(remaining(j), j.deadline - t)
            ctx = bridge.margin_ctx(job_facts(j, u_map[j.jid], spike_max, cap_map[j.jid]),
                                    db, con_demand)
            rank = DORDER.get(db, 0) * 2 + (1 if j.tier == "prod" else 0)
            if phase_of(j) == "train" and held[j.jid] >= cap0(j) > 0:
                demand.append(DemandJob(j.jid, ctx, 0, True, float(rank)))
        free_now = total_gpus - sum(held[j.jid] for j in active)
        margins, reserve, outcome = policy(demand, supply_ctx, free_now)
        n_decisions += 1
        if outcome is not None and not getattr(outcome, "agreed", True):
            n_fallback += 1

        # --- rigid allocation with the negotiated ceilings ------------------------------------
        ceiling = {j.jid: cap0(j) + (margins.get(j.jid, 0) if phase_of(j) == "train" else 0)
                   for j in active}
        for j in active:                                   # voluntary shrink to the new ceiling
            if held[j.jid] > ceiling[j.jid]:
                held[j.jid] = ceiling[j.jid]
        free = total_gpus - sum(held[j.jid] for j in active)
        frozen = {j.jid: sum(j.bid()) for j in active}     # bid-once priority (preprocess=urgency)
        wanters = [j for j in active if held[j.jid] < ceiling[j.jid]]
        prod_w = sorted([j for j in wanters if j.tier == "prod"], key=lambda j: (-frozen[j.jid], j.jid))
        be_w = sorted([j for j in wanters if j.tier != "prod"], key=lambda j: (-frozen[j.jid], j.jid))

        def grant(order, pool):
            nonlocal free
            for j in order:
                give = min(ceiling[j.jid] - held[j.jid], pool, free)
                if give > 0:
                    held[j.jid] += give
                    pool -= give
                    free -= give
            return pool

        grant(prod_w, free)                                # prod first, full free pool
        grant(be_w, max(0, free - reserve))                # best-effort, minus reserved headroom

        # --- advance: margin GPUs buy spike-absorbing speed ------------------------------------
        busy_sum += sum(held[j.jid] for j in active) / total_gpus
        busy_steps += 1
        for j in active:
            c0 = cap0(j)
            g = held[j.jid]
            if c0 == 0:
                rate = 1.0
            else:
                ceil_use = c0 + (useful[j.jid] if phase_of(j) == "train" else 0)
                rate = min(g, ceil_use) / c0               # margin -> rate can exceed 1
            progress[j.jid] += rate
            while done_at[j.jid] is None and progress[j.jid] >= work[j.jid][pidx[j.jid]] - 1e-9:
                progress[j.jid] -= work[j.jid][pidx[j.jid]]
                pidx[j.jid] += 1
                if pidx[j.jid] >= len(j.phases):
                    done_at[j.jid] = t
                    break
            if done_at[j.jid] is not None:
                held[j.jid] = 0                            # release on completion

    def violated(j):
        return done_at[j.jid] is None or done_at[j.jid] > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    fin = [j for j in jobs if done_at[j.jid] is not None]
    slow = [(done_at[j.jid] - j.arrival) / j.nominal for j in fin if j.nominal > 0]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "slowdown": sum(slow) / max(len(slow), 1),
        "finished": float(len(fin)),
        "n_jobs": float(len(jobs)),
        "fallback_rate": n_fallback / max(n_decisions, 1),
    }


# --------------------------------------------------------------------------- #
#  Sweep                                                                        #
# --------------------------------------------------------------------------- #
def sweep(pools, n_jobs, horizon, seeds, scale, spike_max, use_llm, model) -> None:
    dist = load_uncertainty_distribution()
    gpu_dist = load_gpu_distribution()
    src = "results_quantile.json" if os.path.exists(
        os.path.join(HERE, "forecast", "results_quantile.json")) else "fallback"
    gpu_p = os.path.join(HERE, "eval", "results_gpu.json")
    gpu_src = ("results_gpu.json" if os.path.exists(gpu_p)
               and json.load(open(gpu_p)).get("per_job_gpu") else "fallback")
    cache: dict = load_cache()     # warm-start from disk so re-runs are Ollama-free
    trace: list = []
    seen: set = set()
    tag = "rule" if not use_llm else model

    rows = [
        ("no-llm",     lambda: policy_none),
        ("isolated",   lambda: make_policy_isolated(use_llm, model, cache, trace, seen)),
        ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, trace, seen)),
        ("single-llm", lambda: make_policy_single(use_llm, model, cache, trace, seen)),
    ]

    print(f"\n{'='*86}")
    print(f"TWO-SIDED MERGED SIM — rigid incumbents + demand spikes; agents={tag}")
    print(f"{'='*86}")
    print(f"{n_jobs} jobs, horizon {horizon}, mean of {len(seeds)} seeds | spike_max={spike_max} "
          f"scale={scale} | uncertainty dist: {src} ({len(dist)} vals) | "
          f"predicted-GPU dist: {gpu_src} ({len(gpu_dist)} vals)")
    print("Lower SLA/prodSLA/slowdown = better; util shown; fb = negotiation fallback rate.\n")
    header = (f"{'pool':>4}  {'policy':<12} {'SLA':>7} {'prodSLA':>8} {'util':>6} "
              f"{'slowdown':>9} {'fb':>6} {'done':>8}")
    for gpus in pools:
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        for name, factory in rows:
            acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "slowdown": 0.0,
                   "finished": 0.0, "fallback_rate": 0.0}
            for s in seeds:
                jobs = make_workload(n_jobs, s, horizon)
                u_map, spike_map = assign(jobs, s, dist, spike_max)
                cap_map = assign_gpu(jobs, s, gpu_dist)
                r = simulate(jobs, factory(), gpus, horizon, u_map, spike_map, scale,
                             spike_max, cap_map)
                for k in acc:
                    acc[k] += r[k]
            results.append((name, {k: v / len(seeds) for k, v in acc.items()}))
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s1 = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p1 = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<12} {r['sla']:>6.1%}{s1}{r['prod_sla']:>7.1%}{p1}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['fallback_rate']:>5.0%} "
                  f"{r['finished']:>4.1f}/{n_jobs:<3}")
        print()
        if use_llm:
            save_cache(cache)      # checkpoint per pool: a killed run resumes, not restarts
    print("'*' = best (lowest) at that pool. no-llm = point-forecast floor; isolated = today's "
          "independent agents;\nnegotiated = bounded protocol; single-llm = one agent both objectives.")

    out = os.path.join(HERE, "results_two_sided.json")
    with open(out, "w") as f:
        json.dump({"agents": tag, "use_llm": use_llm, "spike_max": spike_max, "scale": scale,
                   "decisions": trace}, f, indent=2)
    if use_llm:
        save_cache(cache)
    print(f"\n{len(trace)} distinct decisions/transcripts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merged two-sided sim (margin vs reserve)")
    ap.add_argument("--llm", action="store_true", help="use qwen agents (needs Ollama)")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--spike", type=float, default=0.6, help="train work inflates up to u*spike")
    ap.add_argument("--scale", type=int, default=3, help="margin GPUs a full-uncertainty spike can use")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    # Pools sit in the CONTENDED regime for the real predicted-GPU caps (mean ~2.35 quarter-GPU
    # units); the old [6,8,12] left pool 8/12 near-idle once flat-8 caps were replaced by the trace.
    sweep([3, 4, 6], n_jobs=16, horizon=300, seeds=list(range(a.seeds)),
          scale=a.scale, spike_max=a.spike, use_llm=a.llm, model=a.model)


if __name__ == "__main__":
    main()
