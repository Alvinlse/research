"""
Exp 28 — TRACE REPLAY: real Alibaba gpu-v2020 jobs drive the two-sided sim.

Exp 27 sampled only the per-job GPU cap from Stage-1 predictions; arrivals, durations and
their correlations stayed synthetic (`make_workload`). This replays contiguous windows of
REAL v2020 jobs so (arrival, duration, GPU demand) come JOINTLY from the trace:

  * ONE CLOCK: tick = TICK_S seconds for BOTH arrivals and durations, so the real
    arrival-vs-duration time relationship survives (an affine stretch of arrivals with a
    separate duration scale would destroy exactly the joint structure a replay is for),
  * arrival  = the job's real first-task start within a random 10-hour trace window,
  * work     = the job's real wall-clock duration in ticks (median ≈ 9 at TICK_S=120),
  * cap      = the job's real total requested GPU in quarter-GPU quanta
               (sum over tasks of plan_gpu*inst_num / 25), clipped to CAP_CLIP,
  * THINNING: the full PAI cluster (~6.5k GPUs) sees ~370 arrivals/hour; a 1-2 GPU pool
    obviously cannot replay them all, so we randomly sample n_jobs of the window's arrivals
    (thinning a near-Poisson stream preserves its statistics while scaling the load).

What the trace does NOT have — deadlines, urgency, tiers — keeps the EXACT `make_workload`
recipe (seeded), so the only change vs Exp 27 is the workload dynamics: a clean ablation.
The validated simulator and all four policies are imported from `two_sided_sim`, unmodified.

Needs `data/alibaba-gpu-v2020/replay_jobs.csv` (job_name, arrival, dur, quanta), built once:
sort pai_task_table Terminated GPU tasks by job, arrival=min start, dur=max end - min start.

Run:  .venv/bin/python -m pins.trace_replay                    # rule tier, no Ollama
      .venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from pins.llm_agent import load_cache, save_cache
from pins.negotiation_sim import Job
from pins.two_sided_sim import (make_policy_isolated, make_policy_negotiated,
                                make_policy_single, policy_none, simulate)
from pins.uncertainty_sim import assign, load_uncertainty_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_CSV = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020", "replay_jobs.csv")

TICK_S = 120         # one sim tick = 2 real minutes -> median trace job ≈ 9 ticks of work
WORK_CLAMP = (1, 60)
CAP_CLIP = 8         # quanta (= 2 GPUs); ~80% of trace jobs are below; keeps pools sane
ARRIVAL_FRAC = 0.6   # sample arrivals from the first 60% of the horizon (make_workload conv.)


def load_trace(path: str = REPLAY_CSV) -> list[tuple[int, int, int]]:
    """(arrival_s, dur_s, quanta) per job, sorted by real arrival."""
    with open(path) as f:
        rows = [(int(float(r["arrival"])), int(float(r["dur"])), int(r["quanta"]))
                for r in csv.DictReader(f)]
    rows.sort()
    return rows


def make_trace_workload(trace, n_jobs: int, seed: int, horizon: int
                        ) -> tuple[list[Job], dict[str, int]]:
    """n_jobs real jobs thinned from a random 10-hour trace window, on ONE clock.

    Real (jointly, per job): arrival time within the window, duration, GPU quanta — all in
    the same TICK_S units, so burstiness vs job length is the trace's own. Synthetic (same
    seeded recipe as make_workload, because the trace has none): urgency, deadline, tier.
    """
    rng = random.Random(f"replay-{seed}")
    window_s = int(horizon * ARRIVAL_FRAC) * TICK_S        # arrivals within first 60%
    t_lo, t_hi = trace[0][0], trace[-1][0] - window_s
    t0 = rng.randrange(t_lo, t_hi)
    in_win = [r for r in trace if t0 <= r[0] < t0 + window_s]
    if len(in_win) < n_jobs:                               # sparse stretch of the trace: reroll
        return make_trace_workload(trace, n_jobs, seed + 7919, horizon)
    window = sorted(rng.sample(in_win, n_jobs))            # thin the arrival stream

    jobs: list[Job] = []
    cap_map: dict[str, int] = {}
    for i, (arr_s, dur_s, quanta) in enumerate(window):
        arrival = (arr_s - t0) // TICK_S
        work = float(min(WORK_CLAMP[1], max(WORK_CLAMP[0], round(dur_s / TICK_S))))
        urgency = round(rng.uniform(0.6, 2.2), 3)          # make_workload recipe verbatim
        slack = max(1.15, min(2.4, 2.5 - 0.65 * urgency))
        deadline = arrival + int(round(work * slack))
        tier = "prod" if urgency >= 1.667 else "besteffort"
        j = Job(f"r{i:02d}", arrival, ["train"], [work], urgency, deadline, tier)
        jobs.append(j)
        cap_map[j.jid] = min(quanta, CAP_CLIP)
    return jobs, cap_map


def sweep(pools, n_jobs, horizon, seeds, scale, spike_max, use_llm, model) -> None:
    trace = load_trace()
    dist = load_uncertainty_distribution()
    cache: dict = load_cache()
    decisions: list = []
    seen: set = set()
    tag = "rule" if not use_llm else model

    rows = [
        ("no-llm",     lambda: policy_none),
        ("isolated",   lambda: make_policy_isolated(use_llm, model, cache, decisions, seen)),
        ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen)),
        ("single-llm", lambda: make_policy_single(use_llm, model, cache, decisions, seen)),
    ]

    print(f"\n{'='*86}")
    print(f"TRACE REPLAY (Alibaba gpu-v2020) — two-sided sim on real jobs; agents={tag}")
    print(f"{'='*86}")
    print(f"{n_jobs} real jobs/window from {len(trace):,} trace jobs, horizon {horizon}, "
          f"mean of {len(seeds)} seeds (window per seed) | spike_max={spike_max} scale={scale} "
          f"| caps = real plan_gpu quanta clipped at {CAP_CLIP}")
    header = (f"{'pool':>4}  {'policy':<12} {'SLA':>7} {'prodSLA':>8} {'util':>6} "
              f"{'slowdown':>9} {'fb':>6} {'done':>8}")
    for gpus in pools:
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        for name, factory in rows:
            acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "slowdown": 0.0,
                   "finished": 0.0, "fallback_rate": 0.0}
            for s in seeds:
                jobs, cap_map = make_trace_workload(trace, n_jobs, s, horizon)
                cap_map = {k: min(v, gpus) for k, v in cap_map.items()}  # feasible at this pool
                u_map, spike_map = assign(jobs, s, dist, spike_max)
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
            save_cache(cache)

    out = os.path.join(HERE, "results_trace_replay.json")
    with open(out, "w") as f:
        json.dump({"agents": tag, "use_llm": use_llm, "spike_max": spike_max, "scale": scale,
                   "cap_clip": CAP_CLIP, "decisions": decisions}, f, indent=2)
    if use_llm:
        save_cache(cache)
    print(f"{len(decisions)} distinct decisions/transcripts -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace replay: real v2020 jobs in the two-sided sim")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--spike", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pools", default="4,6,8", help="real caps are heavier (median 4 quanta)")
    a = ap.parse_args()
    sweep([int(p) for p in a.pools.split(",")], n_jobs=16, horizon=300,
          seeds=list(range(a.seeds)), scale=a.scale, spike_max=a.spike,
          use_llm=a.llm, model=a.model)


if __name__ == "__main__":
    main()
