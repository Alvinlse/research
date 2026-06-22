"""
Stage-1 -> Stage-2 bridge (Exp 16): does uncertainty-sized SAFETY MARGIN help the demand agent?

research_plan.md's prediction co-contribution claims *"uncertainty sizes the demand agent's
safety margin; the auction rations it."* Exps 1-8 built the predictor (now with quantile
uncertainty, pins/forecast/model_quantile.py); Exps 9-15 built the negotiation. This module
finally CONNECTS them and runs the plan's required *no-uncertainty ablation* (research_plan.md:
252): fix the margin vs size it from the quantile width.

The mechanism that makes a margin matter
----------------------------------------
A job's forecast capacity is C0 GPUs. Its TRUE train work can SPIKE above the forecast
(stragglers / extra iterations), by an amount bounded by its Stage-1 uncertainty `u` — the
case the point forecast was blind to. To finish the spiked work before its deadline a job must
run FASTER, i.e. use margin GPUs beyond C0. The demand agent requests that margin in its bid
(pins/predictor.marginal_values, sized by `u`); the committed auction (the Exp-11/12 winner)
rations it. Three bid policies, scored on the SAME stochastic workload:

  * no-margin          : bid C0 only (the point-forecast demand agent) — cannot absorb a spike.
  * fixed-margin       : bid C0 + 1 for EVERY job (a blanket headroom) — mis-targeted: wastes
                         capacity on low-uncertainty jobs, raising contention.
  * uncertainty-sized  : bid C0 + round(u*scale) — margin goes exactly where the spike risk is.

Hypothesis: uncertainty-sized wins SLA at a given pool because it targets scarce margin at the
jobs that will actually spike, where fixed-margin over-subscribes and no-margin under-absorbs.

Pure Python, base .venv. The validated Exp 9-13 harness (negotiation_sim.py) is imported, never
modified; uncertainty + stochastic true-work are carried in external maps (as supply_sim.py did
for malleability). Uses the real Stage-1 uncertainty distribution from results_quantile.json if
present, else a uniform fallback.

Run:  .venv/bin/python -m pins.uncertainty_sim
      .venv/bin/python -m pins.uncertainty_sim --spike 0.6 --scale 3
"""
from __future__ import annotations

import argparse
import json
import os
import random

from pins.negotiation_sim import Job, make_committed_auction, make_workload, sched_greedy
from pins.predictor import PHASE_PROFILES, marginal_values

Alloc = dict[str, int]
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  Stage-1 -> Stage-2: per-job uncertainty + realised demand spike             #
# --------------------------------------------------------------------------- #
def load_uncertainty_distribution() -> list[float]:
    """The empirical per-job uncertainty values from the quantile forecaster (Stage-1). Falls
    back to a plausible spread if the artifact has not been generated yet."""
    p = os.path.join(HERE, "forecast", "results_quantile.json")
    if os.path.exists(p):
        with open(p) as f:
            vals = list(json.load(f).get("per_job_uncertainty", {}).values())
        if vals:
            return [float(v) for v in vals]
    return [0.02, 0.05, 0.08, 0.10, 0.12, 0.18, 0.25, 0.34]      # fallback spread


def assign(jobs: list[Job], seed: int, dist: list[float], spike_max: float):
    """Give each job a true uncertainty `u` (drawn from the Stage-1 distribution) and a realised
    train-work spike fraction in [0, u*spike_max] (uncertainty BOUNDS the spike; the forecast
    point estimate is blind to it). Both deterministic per seed."""
    rng = random.Random(f"unc-{seed}")
    u = {j.jid: rng.choice(dist) for j in jobs}
    spike = {j.jid: u[j.jid] * spike_max * rng.random() for j in jobs}
    return u, spike


def true_need(job: Job, spike: float) -> list[float]:
    """The job's REAL work per phase: train phases inflated by the realised spike; others nominal."""
    return [w * (1.0 + spike) if ph == "train" else w for ph, w in zip(job.phases, job.need)]


# --------------------------------------------------------------------------- #
#  Stochastic-demand simulator (margin GPUs buy speed to absorb a spike)        #
# --------------------------------------------------------------------------- #
def simulate_stochastic(jobs_proto: list[Job], allocator, total_gpus: int, horizon: int,
                        bid_u, u_map: dict[str, float], spike_map: dict[str, float],
                        scale: int) -> dict:
    """Run one bid policy on a fresh copy of the workload under stochastic true demand.

    `bid_u(jid, deadline_bucket, contention) -> float` is the uncertainty the demand agent
    DECLARES (0 / fixed / true `u` / LLM-decided); it sizes the margin in the bid curve. The two
    extra args let the LLM policy reason about the job's state (deterministic policies ignore them).
    A spiked train phase can usefully absorb up to `C0 + round(u*scale)` GPUs (its true parallelism
    under the spike), so margin GPUs grant speedup `rate = min(alloc, C0+margin_useful)/C0` (can
    exceed 1); without them a spiked job runs at rate<=1 and finishes its inflated work late.
    Deadlines come from the NOMINAL work (make_workload), so an unabsorbed spike is what threatens
    the SLA."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}      # realised true work
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}       # extra GPUs a spike can use
    progress = {j.jid: 0.0 for j in jobs}
    pidx = {j.jid: 0 for j in jobs}
    done_at: dict[str, int | None] = {j.jid: None for j in jobs}
    current: Alloc = {}
    busy_sum = 0.0
    busy_steps = 0

    def phase_of(j):
        return j.phases[pidx[j.jid]] if pidx[j.jid] < len(j.phases) else "idle"

    def cap0(j):
        return PHASE_PROFILES[phase_of(j)][0]

    def active(j, t):
        return j.arrival <= t and done_at[j.jid] is None

    for t in range(horizon):
        act = [j for j in jobs if active(j, t)]
        if not act:
            current = {}
            if all(done_at[j.jid] is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        # contention right now (spare capacity <-> demand below pool) for the agent's judgement
        demand = sum(cap0(j) for j in act)
        contention = "high" if demand >= total_gpus else "low"

        def dbucket(j):                                # the agent's view: NOMINAL remaining vs time
            rem = max(0.0, j.need[pidx[j.jid]] - progress[j.jid]) + sum(j.need[pidx[j.jid] + 1:])
            ratio = rem / max(1, j.deadline - t)
            return "behind" if ratio > 1.0 else "ahead" if ratio < 0.6 else "ontrack"

        # demand bids: forecast curve + an uncertainty-sized margin (policy decides the size).
        # Margin is only useful in the spike-prone train phase, so request it only there.
        bids = {j.jid: marginal_values(
            phase_of(j), j.urgency,
            uncertainty=bid_u(j.jid, dbucket(j), contention) if phase_of(j) == "train" else 0.0,
            margin_scale=scale) for j in act}
        cur = {jid: current.get(jid, 0) for jid in bids}
        alloc = allocator(bids, total_gpus, cur)
        busy_sum += sum(alloc.values()) / total_gpus
        busy_steps += 1
        for j in act:
            g = alloc.get(j.jid, 0)
            c0 = cap0(j)
            if c0 == 0:
                rate = 1.0
            else:
                ceiling = c0 + (useful[j.jid] if phase_of(j) == "train" else 0)  # spike parallelism
                rate = min(g, ceiling) / c0                          # margin GPUs -> rate can be >1
            progress[j.jid] += rate
            while done_at[j.jid] is None and progress[j.jid] >= work[j.jid][pidx[j.jid]] - 1e-9:
                progress[j.jid] -= work[j.jid][pidx[j.jid]]
                pidx[j.jid] += 1
                if pidx[j.jid] >= len(j.phases):
                    done_at[j.jid] = t
                    break
        current = {jid: alloc.get(jid, 0) for jid in by_id}

    def violated(j):
        return done_at[j.jid] is None or done_at[j.jid] > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    fin = [j for j in jobs if done_at[j.jid] is not None]
    return {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "finished": len(fin),
        "n_jobs": float(len(jobs)),
    }


# --------------------------------------------------------------------------- #
#  Bid policies (the ablation) + sweep                                          #
# --------------------------------------------------------------------------- #
def policies(u_map: dict[str, float], fixed_u: float):
    """(name, bid_u) rows. Each bid_u takes (jid, deadline_bucket, contention); the deterministic
    policies ignore the state args. fixed-margin declares a constant uncertainty for everyone
    (blanket margin); uncertainty-sized declares each job's true Stage-1 uncertainty."""
    return [
        ("no-margin",         lambda jid, db, con: 0.0),
        ("fixed-margin",      lambda jid, db, con: fixed_u),
        ("uncertainty-sized", lambda jid, db, con: u_map[jid]),
    ]


def make_llm_policy(u_map, tier_of, scale, spike_max, cache, trace, seen, use_llm, model):
    """The LLM demand agent (Exp 17): from its discretised state (uncertainty bucket, SPIKE-RISK
    bucket, deadline, contention, tier) the LLM decides a categorical HEDGE; code maps it to an
    effective uncertainty (pins/llm_agent.margin_uncertainty) fed to marginal_values. SPIKE RISK =
    the plausible relative over-run (severity of a miss), the signal Exp 17 found missing — it lets
    the agent hedge under contention when the tail is heavy. Cached per state, out of the hot loop;
    records each distinct decision + justification (the interpretability artifact)."""
    from pins.llm_agent import (llm_margin, margin_state_key, margin_uncertainty,
                                spike_risk_bucket, uncertainty_bucket)

    def bid_u(jid, db, con):
        u = u_map[jid]
        ctx = {"uncertainty": uncertainty_bucket(u),
               "spike_risk": spike_risk_bucket(u * spike_max),   # expected over-run severity
               "deadline": db, "contention": con, "tier": tier_of[jid]}
        d = llm_margin(ctx, use_llm=use_llm, model=model, cache=cache)
        key = margin_state_key(ctx)
        if key not in seen:
            seen.add(key)
            trace.append({"state": key, "hedge": d["hedge"],
                          "justification": d["justification"], "_source": d["_source"]})
        return margin_uncertainty(d["hedge"], u, scale)

    return bid_u


def sweep(pools, n_jobs, horizon, seeds, spike_max, scale, fixed_u,
          llm_row: bool = False, use_llm: bool = True, model: str = "qwen2.5:3b") -> None:
    dist = load_uncertainty_distribution()
    src = "results_quantile.json" if os.path.exists(
        os.path.join(HERE, "forecast", "results_quantile.json")) else "fallback"
    cache: dict = {}
    trace: list = []
    seen: set = set()
    names = ["no-margin", "fixed-margin", "uncertainty-sized"]
    llm_name = f"llm-margin({'rule' if not use_llm else model})"
    if llm_row:
        names.append(llm_name)

    print(f"\n{'='*78}")
    print("UNCERTAINTY-SIZED SAFETY MARGIN — demand agent, committed auction")
    print(f"{'='*78}")
    print(f"{n_jobs} jobs, horizon {horizon}, {len(seeds)} seeds | spike_max={spike_max} "
          f"margin_scale={scale} | uncertainty dist: {src} ({len(dist)} values)")
    print("Lower SLA/prodSLA = better; util shown. fixed-margin = +1 GPU for all "
          f"(fixed_u={fixed_u}); llm-margin = LLM hedges none/some/heavy from its state.\n")
    header = f"{'pool':>4}  {'policy':<22} {'SLA':>7} {'prodSLA':>8} {'util':>6} {'done':>8}"
    n = len(seeds)
    for gpus in pools:
        print("-" * len(header)); print(header); print("-" * len(header))
        results = [[name, {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "finished": 0.0}]
                   for name in names]
        for s in seeds:
            jobs = make_workload(n_jobs, s, horizon)
            u_map, spike_map = assign(jobs, s, dist, spike_max)
            plist = list(policies(u_map, fixed_u))
            if llm_row:
                tier_of = {j.jid: j.tier for j in jobs}
                plist.append((llm_name, make_llm_policy(u_map, tier_of, scale, spike_max, cache,
                                                        trace, seen, use_llm, model)))
            for row, (_name, bid_u) in zip(results, plist):
                r = simulate_stochastic(jobs, make_committed_auction(), gpus, horizon,
                                        bid_u, u_map, spike_map, scale)
                for k in row[1]:
                    row[1][k] += r[k]
        best_sla = min(acc["sla"] for _, acc in results)
        best_prod = min(acc["prod_sla"] for _, acc in results)
        for name, acc in results:
            a = {k: v / n for k, v in acc.items()}
            s1 = "*" if abs(a["sla"] - best_sla / n) < 1e-9 else " "
            p1 = "*" if abs(a["prod_sla"] - best_prod / n) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<22} {a['sla']:>6.1%}{s1}{a['prod_sla']:>7.1%}{p1}"
                  f"{a['util']:>6.0%} {a['finished']:>4.1f}/{n_jobs:<3}")
        print()
    print("'*' = best (lowest) at that pool. no-margin = point-forecast demand agent (Exp-8 era).")

    if llm_row:
        print("\n=== LLM hedge decisions (state -> hedge + justification) ===")
        for d in sorted(trace, key=lambda x: x["state"]):
            print(f"  [{d['state']:<28}] -> {d['hedge']:6s} ({d['_source']})  {d['justification']}")
        from pins.llm_agent import save_cache
        save_cache(cache)
        out = os.path.join(HERE, "results_uncertainty_llm.json")
        with open(out, "w") as f:
            json.dump({"model": model, "use_llm": use_llm, "decisions": trace}, f, indent=2)
        print(f"\nLLM: {len(trace)} distinct decisions; trace -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 16/17 — uncertainty-sized safety margin")
    ap.add_argument("--spike", type=float, default=0.6,
                    help="spike_max: a job's true train work can inflate up to u*spike_max")
    ap.add_argument("--scale", type=int, default=3, help="margin GPUs at full uncertainty")
    ap.add_argument("--fixed-u", type=float, default=0.34,
                    help="uncertainty the fixed-margin policy declares for every job")
    ap.add_argument("--llm", action="store_true",
                    help="Exp 17: add the LLM demand agent that decides the hedge (needs Ollama)")
    ap.add_argument("--no-llm", action="store_true",
                    help="with --llm: use the rule hedge instead of calling Ollama")
    ap.add_argument("--model", default="qwen2.5:3b", help="Ollama model for the LLM demand agent")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    sweep([6, 8, 12], n_jobs=16, horizon=300, seeds=list(range(a.seeds)),
          spike_max=a.spike, scale=a.scale, fixed_u=a.fixed_u,
          llm_row=a.llm, use_llm=not a.no_llm, model=a.model)


if __name__ == "__main__":
    main()
