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
PRED_CSV = os.path.join(HERE, "eval", "pred_job_gpu.csv")

TICK_S = 120         # one sim tick = 2 real minutes -> median trace job ≈ 9 ticks of work
WORK_CLAMP = (1, 60)
CAP_CLIP = 8         # quanta (= 2 GPUs); ~80% of trace jobs are below; keeps pools sane
ARRIVAL_FRAC = 0.6   # sample arrivals from the first 60% of the horizon (make_workload conv.)


def load_trace(path: str = REPLAY_CSV) -> list[tuple[int, int, int, str]]:
    """(arrival_s, dur_s, quanta, job_name) per job, sorted by real arrival.

    Sort key excludes job_name so tie order (stable = CSV order) is byte-identical to the
    Exp-28 3-tuple version — same seeds keep sampling the same windows/jobs."""
    with open(path) as f:
        rows = [(int(float(r["arrival"])), int(float(r["dur"])), int(r["quanta"]), r["job_name"])
                for r in csv.DictReader(f)]
    rows.sort(key=lambda r: r[:3])
    return rows


def load_predicted_quanta(path: str = PRED_CSV, quantile: str = "p50") -> dict[str, float]:
    """Exp 30/31: job_name -> Stage-1 predicted quanta (test-split jobs of predict_gpu).

    Exp 31: `quantile` picks WHERE on the predicted [p10,p90] interval the agents request —
    the newsvendor knob prediction intervals buy that a point predictor cannot: under-request
    starves the job, over-request hoards GPUs it cannot use. Coverage (the key set) is the
    same for every quantile, so windows — and therefore seeds — stay perfectly paired."""
    with open(path) as f:
        return {r["job_name"]: float(r[quantile]) for r in csv.DictReader(f)}


def make_trace_workload(trace, n_jobs: int, seed: int, horizon: int, pred=None, oracle=False
                        ) -> tuple[list[Job], dict[str, int], dict[str, int]]:
    """n_jobs real jobs thinned from a random 10-hour trace window, on ONE clock.

    Real (jointly, per job): arrival time within the window, duration, GPU quanta — all in
    the same TICK_S units, so burstiness vs job length is the trace's own. Synthetic (same
    seeded recipe as make_workload, because the trace has none): urgency, deadline, tier.

    Exp 30: `pred` = {job_name: predicted quanta} restricts the window to jobs the Stage-1
    predictor has predictions for (its test split) and returns cap_map = PREDICTED demand
    (what the agents request/negotiate over) next to true_cap_map = the trace's real demand
    (what the job actually needs to progress). `oracle=True` keeps the same restricted window
    but requests the truth — the matched-window control. pred=None: Exp 28/29, request==truth.
    """
    rng = random.Random(f"replay-{seed}")
    window_s = int(horizon * ARRIVAL_FRAC) * TICK_S        # arrivals within first 60%
    t_lo, t_hi = trace[0][0], trace[-1][0] - window_s
    t0 = rng.randrange(t_lo, t_hi)
    in_win = [r for r in trace if t0 <= r[0] < t0 + window_s and (pred is None or r[3] in pred)]
    if len(in_win) < n_jobs:                               # sparse stretch of the trace: reroll
        return make_trace_workload(trace, n_jobs, seed + 7919, horizon, pred, oracle)
    window = sorted(rng.sample(in_win, n_jobs), key=lambda r: r[:3])   # thin the arrival stream

    jobs: list[Job] = []
    cap_map: dict[str, int] = {}
    true_cap_map: dict[str, int] = {}
    for i, (arr_s, dur_s, quanta, name) in enumerate(window):
        arrival = (arr_s - t0) // TICK_S
        work = float(min(WORK_CLAMP[1], max(WORK_CLAMP[0], round(dur_s / TICK_S))))
        urgency = round(rng.uniform(0.6, 2.2), 3)          # make_workload recipe verbatim
        slack = max(1.15, min(2.4, 2.5 - 0.65 * urgency))
        deadline = arrival + int(round(work * slack))
        tier = "prod" if urgency >= 1.667 else "besteffort"
        j = Job(f"r{i:02d}", arrival, ["train"], [work], urgency, deadline, tier)
        jobs.append(j)
        true_cap_map[j.jid] = min(quanta, CAP_CLIP)
        cap_map[j.jid] = true_cap_map[j.jid] if pred is None or oracle else \
            min(max(1, round(pred[name])), CAP_CLIP)
    return jobs, cap_map, true_cap_map


METRICS = ("sla", "prod_sla", "util", "slowdown", "finished", "fallback_rate")
RESULTS = os.path.join(HERE, "results_trace_replay.json")


def t95(df: int) -> float:
    """Two-sided 95% Student-t critical value (coarse table; df>=1)."""
    for lo, t in ((60, 2.000), (30, 2.042), (20, 2.086), (10, 2.228), (5, 2.571), (2, 4.303)):
        if df >= lo:
            return t
    return 12.71


def paired_ci(diffs: list[float]) -> tuple[float, float]:
    """Mean paired difference and 95% CI half-width."""
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, float("inf")
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    return m, t95(n - 1) * (var / n) ** 0.5


def print_paired_vs_floor(per_seed_pool: dict[str, list[dict]]) -> None:
    """Per-policy paired diffs vs the no-llm floor (same seed = same workload+spikes)."""
    floor = per_seed_pool["no-llm"]
    for name, rows_ in per_seed_pool.items():
        if name == "no-llm":
            continue
        parts = []
        for metric, label, pct in (("sla", "dSLA", True), ("prod_sla", "dprodSLA", True),
                                   ("slowdown", "dslow", False)):
            diffs = [a[metric] - b[metric] for a, b in zip(rows_, floor)]
            m, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(m) else " "
            parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
        print(f"      {name:<12} vs floor:  " + "  ".join(parts))
    print(f"      (* = 95% CI excludes 0, paired by seed, n={len(floor)})")


def load_results() -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
        if "tiers" in data:
            return data
    return {"tiers": {}}


def cross_tier_stats(policy: str = "negotiated") -> None:
    """Paired-by-seed comparison of one policy across tiers (e.g. the 3b vs 14b inversion)."""
    tiers = load_results()["tiers"]
    tags = [t for t in tiers if policy in next(iter(tiers[t]["per_seed"].values()), {})]
    if len(tags) < 2:
        print(f"need >=2 tiers with per-seed data for '{policy}' in {RESULTS}; have {tags}")
        return
    pools = sorted({int(p) for t in tags for p in tiers[t]["per_seed"]})
    print(f"\nCross-tier paired stats for policy '{policy}' (95% CI, paired by seed):")
    for a in tags:
        for b in tags:
            if a >= b:
                continue
            print(f"  {a} - {b}:")
            for pool in pools:
                ra = tiers[a]["per_seed"].get(str(pool), {}).get(policy)
                rb = tiers[b]["per_seed"].get(str(pool), {}).get(policy)
                if not ra or not rb or len(ra) != len(rb):
                    continue
                parts = []
                for metric, label, pct in (("sla", "dSLA", True), ("prod_sla", "dprodSLA", True),
                                           ("slowdown", "dslow", False)):
                    diffs = [x[metric] - y[metric] for x, y in zip(ra, rb)]
                    m, h = paired_ci(diffs)
                    u = 100.0 if pct else 1.0
                    sig = "*" if h < abs(m) else " "
                    parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
                print(f"    pool {pool:>2} (n={len(ra)}):  " + "  ".join(parts))


def sweep(pools, n_jobs, horizon, seeds, scale, spike_max, use_llm, model,
          caps_mode: str = "real", quantile: str = "p50") -> None:
    trace = load_trace()
    pred = load_predicted_quanta(quantile=quantile) if caps_mode != "real" else None
    oracle = caps_mode == "oracle"
    dist = load_uncertainty_distribution()
    cache: dict = load_cache()
    decisions: list = []
    seen: set = set()
    suffix = {"real": "", "predicted": "+pred", "oracle": "+oracle"}[caps_mode]
    if caps_mode == "predicted" and quantile != "p50":
        suffix = f"+pred-{quantile}"          # 'rule+pred' stays the Exp-30 P50 tier
    tag = ("rule" if not use_llm else model) + suffix

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
    all_per_seed: dict[str, dict[str, list[dict]]] = {}
    for gpus in pools:
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        per_seed_pool: dict[str, list[dict]] = {}
        for name, factory in rows:
            per_seed: list[dict] = []
            for s in seeds:
                jobs, cap_map, tcap = make_trace_workload(trace, n_jobs, s, horizon, pred, oracle)
                cap_map = {k: min(v, gpus) for k, v in cap_map.items()}  # feasible at this pool
                tcap = {k: min(v, gpus) for k, v in tcap.items()}
                u_map, spike_map = assign(jobs, s, dist, spike_max)
                r = simulate(jobs, factory(), gpus, horizon, u_map, spike_map, scale,
                             spike_max, cap_map, true_cap_map=tcap)
                per_seed.append({k: r[k] for k in METRICS})
            per_seed_pool[name] = per_seed
            results.append((name, {k: sum(row[k] for row in per_seed) / len(seeds)
                                   for k in METRICS}))
        all_per_seed[str(gpus)] = per_seed_pool
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s1 = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p1 = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<12} {r['sla']:>6.1%}{s1}{r['prod_sla']:>7.1%}{p1}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['fallback_rate']:>5.0%} "
                  f"{r['finished']:>4.1f}/{n_jobs:<3}")
        print_paired_vs_floor(per_seed_pool)
        print()
        if use_llm:
            save_cache(cache)

    data = load_results()   # merge per tier: rule / 3b / 14b runs no longer clobber each other
    data["tiers"][tag] = {"use_llm": use_llm, "spike_max": spike_max, "scale": scale,
                          "cap_clip": CAP_CLIP, "n_seeds": len(seeds),
                          "per_seed": all_per_seed, "decisions": decisions}
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    if use_llm:
        save_cache(cache)
    print(f"{len(decisions)} distinct decisions/transcripts -> {RESULTS} (tier '{tag}')")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace replay: real v2020 jobs in the two-sided sim")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--spike", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--pools", default="4,6,8", help="real caps are heavier (median 4 quanta)")
    ap.add_argument("--stats", action="store_true",
                    help="no sim: cross-tier paired stats from results_trace_replay.json")
    ap.add_argument("--caps", default="real", choices=("real", "predicted", "oracle"),
                    help="Exp 30: agents request Stage-1 PREDICTED demand (dynamics stay true); "
                         "'oracle' = truth requested on the same prediction-covered windows")
    ap.add_argument("--quantile", default="p50", choices=("p10", "p50", "p90"),
                    help="Exp 31: which predicted quantile the agents request (--caps predicted)")
    a = ap.parse_args()
    if a.stats:
        cross_tier_stats()
        return
    sweep([int(p) for p in a.pools.split(",")], n_jobs=16, horizon=300,
          seeds=list(range(a.seeds)), scale=a.scale, spike_max=a.spike,
          use_llm=a.llm, model=a.model, caps_mode=a.caps, quantile=a.quantile)


if __name__ == "__main__":
    main()
