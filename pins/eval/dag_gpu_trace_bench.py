"""
Stage-1 (DAG track) Exp 21B — REAL Alibaba v2018 DAG topologies, executed for ACTUAL GPU.

Exp 21A used synthetic layered DAGs and found the heaviest-layer rule predicts peak-concurrent
GPU to ~3% (the cascade was unneeded) — but clean layering is exactly what makes that easy.
Real production DAGs are irregular, so this runs the experiment on the ACTUAL v2018 task graphs
we extracted (`extract_dag.py`), to see whether the simple depth-level rule survives or whether
cross-level overlap finally forces runtime-aware reasoning.

The anonymisation constraint (unavoidable): v2018 tasks carry no code/op, only normalised
`plan_mem`. So we cannot run "the task" — we run an executable STAND-IN per node, sized to that
node's demand. Concretely, and to keep the GPU cost bounded:
  * Measure a small LIBRARY of CNN configs once on the A100 -> (config, peak_gb, duration).
  * Each trace node is RANK-MATCHED: its plan_mem percentile -> the library config at the same
    GPU-footprint percentile. Preserves the trace's per-node demand ORDERING (what drives the
    concurrent peak) without trusting plan_mem's absolute units.
  * per-node MEMORY  = library measured peak (real GPU measurement, sized to the request)
  * per-node DURATION = the node's REAL trace duration (end-start) — the real task length that
    governs which tasks overlap. (Library measure-time is a probe artifact, not used here.)
  * topology = the REAL v2018 edges; depth = the real longest-path level.

So: real graph shape + real durations + measured GPU memory. Honest caveat — the per-node
WORKLOAD is a representative stand-in, not Alibaba's actual (unknowable) op.

Run (GPU venv; .venv torch is broken):
  .venv-forecast/bin/python -m pins.eval.dag_gpu_trace_bench --n-jobs 80
  .venv-forecast/bin/python -m pins.eval.dag_gpu_trace_bench --n-jobs 80 --max-parallel 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from pins.eval.dag_gpu_bench import GRID, measure_task, simulate_peak_concurrent

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "..", "data", "alibaba-v2018")
NODES = os.path.join(DATA, "dag_nodes.csv.gz")
EDGES = os.path.join(DATA, "dag_edges.csv.gz")
OUT = os.path.join(HERE, "results_dag_gpu_trace.json")


def build_library(device, steps):
    """Measure every GRID config once; return (peaks, durs, cfgs) sorted by GPU footprint."""
    lib = []
    grids = [{"width": w, "blocks": b, "res": r, "batch": bs}
             for w in GRID["width"] for b in GRID["blocks"]
             for r in GRID["res"] for bs in GRID["batch"]]
    print(f"measuring library of {len(grids)} configs on {torch.cuda.get_device_name(0)} ...", flush=True)
    for cfg in grids:
        peak, dur = measure_task(cfg, device, steps)
        lib.append((peak, dur, cfg))
    lib.sort(key=lambda t: t[0])
    peaks = np.array([t[0] for t in lib])
    print(f"  library GPU footprint span {peaks.min():.2f}-{peaks.max():.2f} GB", flush=True)
    return peaks, [t[2] for t in lib]


def load_trace_dags(n_jobs, min_n, max_n, seed):
    """Sample real multi-task v2018 jobs with edges, positive durations, acyclic, size in range."""
    edges_all = pd.read_csv(EDGES)
    nodes_all = pd.read_csv(NODES)
    rng = np.random.default_rng(seed)
    cand = rng.permutation(edges_all.job_name.unique())[:30000]
    cand = set(cand.tolist())
    nA = nodes_all[nodes_all.job_name.isin(cand)]
    eA = edges_all[edges_all.job_name.isin(cand)]
    edict = {jn: sub for jn, sub in eA.groupby("job_name")}
    out = []
    for jn, nd in nA.groupby("job_name"):
        if not (min_n <= len(nd) <= max_n) or jn not in edict:
            continue
        if nd.plan_mem.isna().any() or (nd.duration <= 0).any() or (nd.depth < 0).any():
            continue
        out.append((jn, nd.reset_index(drop=True), edict[jn]))
        if len(out) >= n_jobs:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=80)
    ap.add_argument("--min-nodes", type=int, default=4)
    ap.add_argument("--max-nodes", type=int, default=25)
    ap.add_argument("--max-parallel", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA — run on the A100 node via .venv-forecast")
    device = "cuda:0"

    lib_peaks, lib_cfgs = build_library(device, args.steps)
    L = len(lib_peaks)

    print(f"\nsampling real v2018 DAGs ({args.min_nodes}-{args.max_nodes} nodes) ...", flush=True)
    dags = load_trace_dags(args.n_jobs, args.min_nodes, args.max_nodes, args.seed)
    print(f"  {len(dags)} jobs selected", flush=True)

    # global rank-match: node plan_mem percentile -> library footprint percentile
    all_mem = np.concatenate([nd.plan_mem.to_numpy() for _, nd, _ in dags])
    order = all_mem.argsort()
    pct = np.empty(len(all_mem)); pct[order] = np.linspace(0, 1, len(all_mem))
    lib_idx_all = np.round(pct * (L - 1)).astype(int)

    rows = []
    cur = 0
    for jn, nd, ed in dags:
        k = len(nd)
        lib_idx = lib_idx_all[cur:cur + k]; cur += k
        local = {tid: i for i, tid in enumerate(nd.task_id.tolist())}      # task_id -> 0..k-1
        mem = {i: float(lib_peaks[lib_idx[i]]) for i in range(k)}
        dur = {i: float(nd.duration.iloc[i]) for i in range(k)}
        edges = [(local[s], local[d]) for s, d in zip(ed.src, ed.dst)
                 if s in local and d in local]
        peak, makespan = simulate_peak_concurrent(list(range(k)), edges, mem, dur, args.max_parallel)
        naive_sum = sum(mem.values())
        naive_max = max(mem.values())
        by_level: dict[int, float] = {}
        for i in range(k):                                                # real depth-level sum
            lv = int(nd.depth.iloc[i]); by_level[lv] = by_level.get(lv, 0) + mem[i]
        level_sum = max(by_level.values())
        rows.append({"job": jn, "n_nodes": k, "n_edges": len(edges), "depth": int(nd.depth.max()),
                     "peak_concurrent_gb": peak, "naive_sum": naive_sum,
                     "naive_max": naive_max, "level_sum": level_sum})

    truth = np.array([r["peak_concurrent_gb"] for r in rows])
    nn = np.array([r["n_nodes"] for r in rows]); dd = np.array([r["depth"] for r in rows])
    print(f"\n{len(rows)} real DAGs | nodes {nn.min()}-{nn.max()} (med {int(np.median(nn))}) | "
          f"depth med {int(np.median(dd))} max {dd.max()} | "
          f"peak-concurrent {truth.min():.2f}/{np.median(truth):.2f}/{truth.max():.2f} GB")
    print(f"\n{'baseline':12} {'MAE(GB)':>9} {'MAPE%':>8} {'within1.5x':>11}")
    for name in ("naive_sum", "naive_max", "level_sum"):
        pred = np.array([r[name] for r in rows])
        ratio = np.maximum(pred, 1e-6) / np.maximum(truth, 1e-6)
        print(f"{name:12} {np.mean(np.abs(pred-truth)):9.3f} "
              f"{np.mean(np.abs(pred-truth)/truth)*100:8.1f} "
              f"{np.mean((ratio>=1/1.5)&(ratio<=1.5))*100:10.1f}%")

    json.dump({"device": torch.cuda.get_device_name(0), "n_jobs": len(rows),
               "max_parallel": args.max_parallel, "library_configs": L, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
