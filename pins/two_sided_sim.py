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
from pins.negotiation_protocol import (DemandJob, HEDGE_GPUS, NegotiationOutcome, negotiate,
                                       single_llm_plan)
from pins.negotiation_sim import Job, make_workload
from pins.predictor import PHASE_PROFILES
from pins.uncertainty_sim import (assign, assign_gpu, load_gpu_distribution,
                                   load_uncertainty_distribution, true_need)

HERE = os.path.dirname(os.path.abspath(__file__))
DORDER = {"ahead": 0, "ontrack": 1, "behind": 2}
TTF_HORIZON = 2      # Exp 39: "imminent" release = believed remaining work <= this many ticks


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


def make_policy_single_ilp(use_llm, model, cache, trace, seen):
    """LLMSched-architecture arm: ONE joint-objective LLM proposes (margins, reserve), the
    evaluator verifies it, and an infeasible proposal is REPAIRED by the min-edit ILP
    (pins/ilp.py) instead of falling back to the floor wholesale — the reference paper's
    propose->guarantee spine. `agreed=False` marks a repaired tick, so the fb column reads
    as the ILP repair rate (the analogue of the referee's fallback rate)."""
    from pins.ilp import allocate
    from pins.referee import check_allocation

    def policy(demand, supply_ctx, free, **_):
        o = single_llm_plan(demand, supply_ctx, free, use_llm=use_llm, model=model, cache=cache)
        margins, reserve = dict(o.margins), o.reserve
        repaired = False
        if check_allocation(margins, reserve, demand, free):
            # Constant curves rank prod/behind margins above besteffort above the reserve;
            # capacity = this tick's free pool. Nothing is "current" (margins are per-tick),
            # so rescale_cost stays 0 and the ILP is a pure value-max knapsack over the
            # LLM's own proposal — the minimum edit that fits.
            bids = {j.jid: [2.0 + j.concede_rank] * margins[j.jid]
                    for j in demand if margins.get(j.jid, 0) > 0}
            if reserve > 0:
                bids["_reserve"] = [1.0] * reserve
            r = allocate(bids, free)
            margins = {jid: r.allocation.get(jid, 0) for jid in margins}
            reserve = r.allocation.get("_reserve", 0)
            repaired = True
        out = NegotiationOutcome(margins=margins, reserve=reserve, rounds=o.rounds,
                                 agreed=not repaired, transcript=o.transcript)
        _record_outcome(trace, seen, "single-ilp", out)
        return margins, reserve, out
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
             cap_map: dict[str, int], true_cap_map: dict[str, int] | None = None,
             belief_work: dict[str, float] | str | None = None,
             ttf_work: dict[str, float] | str | None = None,
             dyn_cap_map: dict[str, int] | None = None, dyn_after: int = 3) -> dict:
    """One run of a policy on a fresh workload copy. Rigid: a running job is never involuntarily
    preempted; it only shrinks VOLUNTARILY to its ceiling (cap0 + this tick's negotiated margin).
    Spikes: a train phase's true work is inflated; margin GPUs grant rate>1 to absorb it, capped at
    the spike's usable parallelism `useful = round(u*scale)`. Deadlines come from NOMINAL work, so
    an unabsorbed spike is what threatens the SLA — and reserved headroom protects late prod jobs.

    `true_cap_map` (Exp 30): if given, `cap_map` is the demand the agents REQUEST/negotiate over
    (e.g. a Stage-1 prediction) while progress dynamics run on the job's TRUE train demand —
    under-prediction starves the job (rate<1 even fully granted), over-prediction hogs GPUs it
    cannot convert into progress. If None, request == truth (Exp 27/28 behaviour, unchanged).

    `belief_work` (Exp 38): what the DEMAND AGENT believes a job's total work is, for its
    behind/ontrack/ahead deadline bucket only — dynamics and SLA stay on the truth. None =
    oracle (true remaining, the pre-Exp-38 behaviour, unchanged); "blind" = no time signal
    (every job reads "ontrack"); {jid: predicted total ticks} = Stage-1 predicted runtime.

    `ttf_work` (Exp 39): the SUPPLY AGENT's time-to-free signal — held GPUs of running jobs
    whose believed remaining work is <= TTF_HORIZON ticks enter reserve_ctx as a `release`
    bucket (imminent releases substitute for idle reserve). None = no signal (pre-Exp-39,
    byte-identical); "oracle" = true realised remaining; {jid: predicted total ticks} =
    Stage-1 runtime. Remaining WORK is the proxy for remaining TIME (assumes rate ~1).

    `dyn_cap_map` (Exp 45): a DYNAMIC train-phase cap — once a job has RUN for `dyn_after`
    ticks (telemetry observed), its allocation base switches from cap_map (the static
    admission request) to dyn_cap_map (a telemetry-corrected estimate of true need). The
    user's declared request stays fixed; only the system's belief moves. A falling cap
    triggers the existing voluntary-shrink path; a rising one makes the job a wanter again.
    Negotiation facts (job_facts req_gpu) intentionally stay on cap_map — the margin layer
    still negotiates over the admission request. None = pre-Exp-45, byte-identical."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}     # realised (spiked) work
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}      # extra GPUs a spike can use
    held = {j.jid: 0 for j in jobs}                                  # rigid: locked to the job
    ran = {j.jid: 0 for j in jobs}                                   # ticks run (telemetry seen)
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
        if ph != "train":
            return PHASE_PROFILES[ph][0]
        if dyn_cap_map is not None and ran[j.jid] >= dyn_after:
            return dyn_cap_map[j.jid]                      # telemetry-corrected base (Exp 45)
        return cap_map[j.jid]

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
        upcoming = None
        if ttf_work is not None:                         # Exp 39: imminent-release signal
            def ttf_rem(j):
                if ttf_work == "oracle":                 # true realised remaining work
                    return (work[j.jid][pidx[j.jid]] - progress[j.jid]
                            + sum(work[j.jid][pidx[j.jid] + 1:]))
                return max(0.0, ttf_work[j.jid] - (sum(j.need[:pidx[j.jid]]) + progress[j.jid]))
            upcoming = sum(held[j.jid] for j in active
                           if held[j.jid] > 0 and ttf_rem(j) <= TTF_HORIZON)
        supply_ctx = bridge.reserve_ctx(con_supply, n_inc, upcoming)
        # Contested slice: only jobs already RUNNING their full base contest the free GPUs for a
        # speed-up margin — a waiting/ramping job can't spend a margin GPU, it needs base first (the
        # auction's job). So the margin table is the running train jobs, contesting `free_now` (the
        # GPUs genuinely free this tick) against the supply reserve. Base demand is not negotiable and
        # never enters `want`, so the negotiation no longer false-fallbacks on base oversubscription.
        demand: list[DemandJob] = []
        for j in active:
            if belief_work is None:                        # oracle time signal (unchanged)
                rem = remaining(j)
            elif belief_work == "blind":                   # no time signal
                rem = None
            else:                                          # believed remaining = predicted - done
                rem = max(0.0, belief_work[j.jid]
                          - (sum(j.need[:pidx[j.jid]]) + progress[j.jid]))
            db = "ontrack" if rem is None else bridge.deadline_bucket(rem, j.deadline - t)
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
        tcap = true_cap_map or cap_map
        for j in active:
            # dynamics run on the TRUE demand; cap0/ceiling above used the (possibly predicted)
            # requested demand — the gap is exactly prediction error hitting outcomes
            c0 = tcap[j.jid] if phase_of(j) == "train" else PHASE_PROFILES[phase_of(j)][0]
            g = held[j.jid]
            if g > 0:
                ran[j.jid] += 1                            # a tick of telemetry accrues
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
        "done_at": dict(done_at),   # per-job completion tick (None = unfinished);
    }                               # results files are unaffected (METRICS whitelist)


# --------------------------------------------------------------------------- #
#  EASY-backfilling baseline (Exp 40)                                            #
# --------------------------------------------------------------------------- #
def simulate_backfill(jobs_proto: list[Job], total_gpus: int, horizon: int,
                      u_map: dict, spike_map: dict, scale: int, spike_max: float,
                      cap_map: dict[str, int], true_cap_map: dict[str, int] | None = None,
                      belief_work: dict[str, float] | str = "oracle") -> dict:
    """The classical FCFS + EASY-backfilling scheduler (research_plan.md 'Baselines' row) on the
    SAME workload/dynamics as `simulate`, so results are seed-paired with every policy tier.

    Deliberately NOT a `policy`: EASY is a different allocation DISCIPLINE — all-or-nothing
    grants at the requested cap, hold to completion, tier/urgency-blind FCFS order, a
    reservation for the queue head, and backfill of jobs that provably (by runtime ESTIMATE)
    don't delay that reservation. The margins/reserve hook can't express that, so it gets a
    sibling loop; the progress dynamics below are copied from `simulate` verbatim so the
    discipline is the only difference. No margins ever: EASY has no spike-absorption lever.

    `belief_work` is the runtime estimate the reservation/backfill rule runs on — the input
    EASY cannot exist without, and exactly the live Stage-2 role Exp 38/39 concluded is left
    for the Stage-1 runtime predictor: {jid: predicted total ticks} (easy-pred) or "oracle" =
    the realised spiked work (easy-oracle prices the prediction error).

    Written for the trace-replay workload (single 'train' phase, request = cap_map); multi-
    phase `make_workload` jobs would need a per-phase request schedule EASY doesn't model."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}
    held = {j.jid: 0 for j in jobs}
    progress = {j.jid: 0.0 for j in jobs}
    pidx = {j.jid: 0 for j in jobs}
    done_at: dict[str, int | None] = {j.jid: None for j in jobs}
    started: set[str] = set()
    busy_sum = 0.0
    busy_steps = 0

    def req(j) -> int:                                   # all-or-nothing request, held to completion
        return max(1, cap_map[j.jid])

    def believed_total(j) -> float:                      # scheduler's estimate of TOTAL work (ticks)
        return sum(work[j.jid]) if belief_work == "oracle" else belief_work[j.jid]

    def believed_remaining(j) -> float:                  # running job: estimate minus done (rate~1)
        if belief_work == "oracle":
            return (work[j.jid][pidx[j.jid]] - progress[j.jid]
                    + sum(work[j.jid][pidx[j.jid] + 1:]))
        return max(1.0, belief_work[j.jid] - (sum(j.need[:pidx[j.jid]]) + progress[j.jid]))

    for t in range(horizon):
        active = [j for j in jobs if j.arrival <= t and done_at[j.jid] is None]
        if not active:
            if all(done_at[j.jid] is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue

        # --- EASY scheduling pass: FCFS start, head reservation, estimate-gated backfill -------
        running = [j for j in active if j.jid in started]
        queue = sorted((j for j in active if j.jid not in started),
                       key=lambda j: (j.arrival, j.jid))
        free = total_gpus - sum(held[j.jid] for j in running)
        while queue and req(queue[0]) <= free:           # plain FCFS while the head fits
            head = queue.pop(0)
            started.add(head.jid)
            held[head.jid] = req(head)
            running.append(head)
            free -= req(head)
        if queue:                                        # head blocked: reserve its start
            need = req(queue[0]) - free                  # GPUs the head still lacks
            t_reserve = float("inf")
            shadow = free                                # spare GPUs AT the reserved start
            acc = 0
            for j in sorted(running, key=believed_remaining):
                acc += held[j.jid]
                if acc >= need:                          # earliest believed time free >= request
                    t_reserve = believed_remaining(j)
                    shadow = free + acc - req(queue[0])
                    break
            for k in queue[1:]:                          # backfill: must not delay the reservation
                rq = req(k)
                if rq <= free and (believed_total(k) <= t_reserve or rq <= shadow):
                    started.add(k.jid)
                    held[k.jid] = rq
                    free -= rq
                    if believed_total(k) > t_reserve:    # outlives the reservation: eats the spare
                        shadow -= rq

        # --- advance: dynamics copied from `simulate` (no margins, grant == request) ------------
        busy_sum += sum(held[j.jid] for j in active) / total_gpus
        busy_steps += 1
        tcap = true_cap_map or cap_map
        for j in active:
            c0 = tcap[j.jid] if j.phases[min(pidx[j.jid], len(j.phases) - 1)] == "train" \
                else PHASE_PROFILES[j.phases[min(pidx[j.jid], len(j.phases) - 1)]][0]
            g = held[j.jid]
            if c0 == 0:
                rate = 1.0
            else:
                rate = min(g, c0 + useful[j.jid]) / c0
            progress[j.jid] += rate
            while done_at[j.jid] is None and progress[j.jid] >= work[j.jid][pidx[j.jid]] - 1e-9:
                progress[j.jid] -= work[j.jid][pidx[j.jid]]
                pidx[j.jid] += 1
                if pidx[j.jid] >= len(j.phases):
                    done_at[j.jid] = t
                    break
            if done_at[j.jid] is not None:
                held[j.jid] = 0

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
        "fallback_rate": 0.0,
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
