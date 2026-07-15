"""
Exp 49 — REFEREE ON REAL JOBS: the referee-LLM allocator scored on Alibaba gpu-v2020 windows.

The smoke scenes in pins/referee.py were hand-made. Here every scene is a window of REAL
v2020 jobs via the validated Exp-28 sampler (`trace_replay.make_trace_workload`): base
demand = the job's real requested quarter-GPU quanta, tiers/deadlines = the same seeded
recipe every prior experiment uses (the trace has none). Each window is clearing-pointed at
three pool sizes — surplus / exact / shortfall of its own total base demand — so rationing
is tested where it actually bites.

Per scene the demand/supply statements are gathered once (statement model fixed at 3b,
cached by bucket) and each referee model decides the same submissions: a paired,
statements-held-fixed model ablation. Scored by the deterministic evaluator
(`referee.check_allocation` — reports, never repairs):

  feas%      — scenes with zero rule violations
  over       — mean GPUs awarded beyond the pool (rule-1 overcommit)
  waste      — mean GPUs left idle while some job's base was unmet (rationing efficiency)
  prodcov    — mean fraction of prod-tier base demand actually served

Run:  .venv/bin/python -m pins.referee_eval --models rule                  # sanity, no LLM
      .venv/bin/python -m pins.referee_eval --models qwen2.5:3b,qwen2.5:14b,gemma2:27b
"""
from __future__ import annotations

import argparse
import json
import os

from pins import bridge
from pins.llm_agent import load_cache, save_cache
from pins.negotiation_protocol import DemandJob
from pins.referee import referee_decide
from pins.trace_replay import load_trace, make_trace_workload
from pins.two_sided_sim import job_facts
from pins.uncertainty_sim import assign, load_uncertainty_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_referee_eval.json")
SPIKE_MAX = 0.6                       # the Exp 28+ default world
POOL_FACTORS = (1.25, 1.0, 0.75)      # surplus / exact / shortfall of the scene's base demand


def make_scene(trace, seed: int, n_jobs: int) -> tuple[list[DemandJob], dict, int]:
    """One real window -> (demand table, supply ctx, total base quanta)."""
    jobs, cap_map, _, _ = make_trace_workload(trace, n_jobs, seed, horizon=300)
    dist = load_uncertainty_distribution()
    u_map, _ = assign(jobs, seed, dist, SPIKE_MAX)
    base_total = sum(cap_map.values())
    demand = []
    for j in jobs:
        rem = sum(j.need)                                  # scene clears at each job's arrival
        db = bridge.deadline_bucket(rem, j.deadline - j.arrival)
        con = "high" if base_total >= max(1, round(base_total * min(POOL_FACTORS))) else "low"
        ctx = bridge.margin_ctx(job_facts(j, u_map[j.jid], SPIKE_MAX, cap_map[j.jid]), db, con)
        demand.append(DemandJob(j.jid, ctx, forecast_cap=cap_map[j.jid]))
    n_prod = sum(1 for j in jobs if j.tier == "prod")
    supply_ctx = bridge.reserve_ctx(bridge.contention_bucket(base_total, base_total), n_prod)
    return demand, supply_ctx, base_total


def score(o, demand: list[DemandJob], free: int) -> dict:
    """Evaluator-side metrics for one referee decision (violations already in `o`)."""
    total = sum(o.alloc.values()) + o.reserve
    unmet = sum(max(0, j.forecast_cap - o.alloc.get(j.jid, 0)) for j in demand)
    prod_base = sum(j.forecast_cap for j in demand if j.ctx.get("tier") == "prod")
    prod_got = sum(min(o.alloc.get(j.jid, 0), j.forecast_cap)
                   for j in demand if j.ctx.get("tier") == "prod")
    return {"feasible": o.feasible,
            "over": max(0, total - free),
            "waste": max(0, free - total) if unmet > 0 else 0,
            "prodcov": prod_got / prod_base if prod_base else 1.0,
            "n_violations": len(o.violations)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Referee allocator on real v2020 windows")
    ap.add_argument("--models", default="rule,qwen2.5:3b,qwen2.5:14b,gemma2:27b")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    models = a.models.split(",")

    trace = load_trace()
    cache = load_cache()
    scenes = []
    for s in range(a.seeds):
        demand, supply_ctx, base_total = make_scene(trace, s, a.n_jobs)
        for f in POOL_FACTORS:
            scenes.append((s, f, demand, supply_ctx, max(2, round(base_total * f))))
    print(f"{len(scenes)} scenes ({a.seeds} real windows x {len(POOL_FACTORS)} pool factors), "
          f"{a.n_jobs} real jobs each; statements at qwen2.5:3b\n")

    results: dict = {}
    header = f"{'referee':<14} {'pool':>8} {'feas%':>6} {'over':>6} {'waste':>6} {'prodcov':>8}"
    print(header); print("-" * len(header))
    for model in models:
        per_factor: dict[float, list[dict]] = {f: [] for f in POOL_FACTORS}
        rows_out = []
        for s, f, demand, supply_ctx, free in scenes:
            o = referee_decide(demand, supply_ctx, free, use_llm=(model != "rule"),
                               model=model, statement_model="qwen2.5:3b", cache=cache)
            m = score(o, demand, free)
            per_factor[f].append(m)
            rows_out.append({"seed": s, "factor": f, "free": free, "alloc": o.alloc,
                             "reserve": o.reserve, "violations": o.violations,
                             "justification": o.justification, **m})
            save_cache(cache)                      # LLM calls are slow: persist as we go
        for f in POOL_FACTORS:
            rows = per_factor[f]
            n = len(rows)
            print(f"{model:<14} {f:>7.2f}x "
                  f"{100 * sum(r['feasible'] for r in rows) / n:>5.0f}% "
                  f"{sum(r['over'] for r in rows) / n:>6.2f} "
                  f"{sum(r['waste'] for r in rows) / n:>6.2f} "
                  f"{sum(r['prodcov'] for r in rows) / n:>8.2f}")
        results[model] = rows_out
        print()

    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"per-scene detail -> {RESULTS}")


if __name__ == "__main__":
    main()
