"""
Stage-1 (DAG track) Exp 21 — a MEASURED DAG GPU benchmark, because no public trace
pairs precedence DAGs with actual GPU.

Verified across every Alibaba release: v2018 has DAGs but no GPU; gpu-v2020 has measured
GPU util but only PS/worker gangs (no precedence); gpu-v2023/2025 have neither (request-only
scheduling snapshots). So to study "does DAG topology drive ACTUAL GPU demand?" we must mint
the labels — execute real GPU workloads arranged in a DAG and measure. This mirrors Exp 1-7's
A100 closed loop (real CNNs, `torch.cuda.max_memory_allocated`), lifted from single tasks to
workflows.

THE TARGET worth predicting is NOT per-task GPU memory — Exp 4 already nails that to 0.04 GB
with deterministic code, and the DAG is irrelevant to it. The genuinely DAG-dependent,
unsolved quantity is the WORKFLOW's **peak concurrent GPU memory**: how much VRAM the job
needs at once, which is a function of (a) topology — which tasks MAY run in parallel, (b)
durations — whether parallel branches actually OVERLAP in time, and (c) per-task footprint.
It is emphatically NOT the sum of per-task memory (over-counts: not everything co-resides) nor
the max single task (under-counts: parallel branches do co-reside). That gap is the experiment.

WHAT THIS FILE DOES (Part A — the measured dataset + the "naive aggregation fails" result):
  1. Generate layered DAGs of CNN tasks (parallel nodes within a layer CAN co-run).
  2. Execute each distinct task config on the A100, measure (peak_gb, duration_s). Memoised by
     config, so dozens of unique measurements cover hundreds of nodes.
  3. Compute the workflow peak-concurrent GPU via a list-scheduler over the measured
     (mem, duration) under a parallelism cap — the deterministic ground truth for the schedule.
  4. Score three structure-blind/partial baselines against it: naive_sum, naive_max, and a
     topology-aware-but-duration-blind layer_sum. Their error motivates Part B (predict it from
     DAG + GBT-runtime, the cascade).

SCOPE / honesty: Part A measures per-task footprints for real and DERIVES the overlap by
simulation over measured durations (a list-scheduler). It does not yet capture GPU CONTENTION
slowdown when tasks truly co-run (which would change durations hence overlap) — that is Part C,
deferred, exactly as Exp 14 staged malleable->rigid. `max_parallel` defaults high so the peak
is TOPOLOGY-limited (the pure DAG effect), not resource-throttled.

Run (needs the GPU venv — .venv torch is broken, ncclCommResume):
  .venv-forecast/bin/python -m pins.eval.dag_gpu_bench --n-dags 40
  .venv-forecast/bin/python -m pins.eval.dag_gpu_bench --n-dags 40 --max-parallel 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_dag_gpu_bench.json")

# Task config grid: param-light, activation-dominated (Exp 1-7 regime). width x blocks x res x
# batch spans ~0.3-6 GB so a DAG's parallel branches can plausibly co-exceed 40 GB or not.
GRID = {"width": [32, 64, 96, 128], "blocks": [3, 4], "res": [64, 96], "batch": [64, 128, 256]}


# --- task primitive: a plain VGG-ish CNN train step (mirrors predict_cnn.SimpleCNN) ----------
class SimpleCNN(nn.Module):
    def __init__(self, width=64, blocks=3, in_ch=3, n_classes=10):
        super().__init__()
        layers, c, w = [], in_ch, width
        for _ in range(blocks):
            layers += [nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c, w = w, w * 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def measure_task(cfg, device, steps=5):
    """Train a few steps; return (peak_gb, duration_s). This IS the ground truth (Exp 1-7)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = SimpleCNN(cfg["width"], cfg["blocks"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    b, res = cfg["batch"], cfg["res"]
    torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(steps):
        x = torch.randn(b, 3, res, res, device=device)
        y = torch.randint(0, 10, (b,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    dur = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return peak, dur


# --- DAG generation: layered, with real parallelism within a layer ---------------------------
def gen_dag(rng):
    """Return (nodes, edges). nodes[i] = {id, cfg, layer}; edges = (src,dst) precedence."""
    n_layers = int(rng.integers(2, 6))
    nodes, edges, prev = [], [], []
    nid = 0
    for L in range(n_layers):
        width = int(rng.integers(1, 4))                      # 1-3 parallel tasks in this layer
        cur = []
        for _ in range(width):
            cfg = {k: int(rng.choice(v)) for k, v in GRID.items()}
            nodes.append({"id": nid, "cfg": cfg, "layer": L})
            if prev:                                         # depend on >=1 random parent
                k = int(rng.integers(1, len(prev) + 1))
                for p in rng.choice(prev, size=k, replace=False):
                    edges.append((int(p), nid))
            cur.append(nid)
            nid += 1
        prev = cur
    return nodes, edges


# --- workflow peak-concurrent GPU via list-scheduling over measured (mem, dur) ----------------
def simulate_peak_concurrent(nodes, edges, mem, dur, max_parallel):
    """Greedy list scheduler: a task starts once all parents finish and a slot is free; run up
    to `max_parallel` at once. Peak = max over time of summed mem of co-running tasks."""
    n = len(nodes)
    preds = {i: [] for i in range(n)}
    for s, d in edges:
        preds[d].append(s)
    done = np.zeros(n, bool)
    finish = np.full(n, -1.0)        # finish time, -1 = not started
    running = {}                     # id -> finish_time
    t = 0.0
    peak = 0.0
    started = np.zeros(n, bool)
    while not done.all():
        # launch any ready task while a slot is free
        for i in range(n):
            if (not started[i] and len(running) < max_parallel
                    and all(done[p] for p in preds[i])):
                running[i] = t + max(dur[i], 1e-6)
                started[i] = True
        peak = max(peak, sum(mem[i] for i in running))      # concurrent footprint now
        if not running:                                     # nothing runnable -> deadlock guard
            break
        t = min(running.values())                           # advance to next finish
        for i in [i for i, f in running.items() if f <= t + 1e-9]:
            done[i] = True
            finish[i] = running.pop(i)
    return peak, float(finish.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dags", type=int, default=40)
    ap.add_argument("--max-parallel", type=int, default=64, help="concurrency cap (high = topology-limited)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA — run on the A100 node via .venv-forecast")
    device = "cuda:0"
    print(f"device: {torch.cuda.get_device_name(0)} | {args.n_dags} DAGs | "
          f"max_parallel={args.max_parallel}\n", flush=True)

    rng = np.random.default_rng(args.seed)
    cache: dict[tuple, tuple] = {}                          # cfg-key -> (mem, dur), memoised

    def meas(cfg):
        key = (cfg["width"], cfg["blocks"], cfg["res"], cfg["batch"])
        if key not in cache:
            cache[key] = measure_task(cfg, device, args.steps)
        return cache[key]

    rows = []
    for d in range(args.n_dags):
        nodes, edges = gen_dag(rng)
        mem = {nd["id"]: meas(nd["cfg"])[0] for nd in nodes}
        dur = {nd["id"]: meas(nd["cfg"])[1] for nd in nodes}
        peak, makespan = simulate_peak_concurrent(nodes, edges, mem, dur, args.max_parallel)
        # baselines for the workflow peak-concurrent target
        naive_sum = sum(mem.values())                       # everything co-resides (over)
        naive_max = max(mem.values())                       # only the biggest (under)
        by_layer: dict[int, float] = {}
        for nd in nodes:                                    # topology-aware, duration-blind
            by_layer[nd["layer"]] = by_layer.get(nd["layer"], 0) + mem[nd["id"]]
        layer_sum = max(by_layer.values())
        rows.append({"dag": d, "n_nodes": len(nodes), "n_edges": len(edges),
                     "peak_concurrent_gb": peak, "makespan_s": makespan,
                     "naive_sum": naive_sum, "naive_max": naive_max, "layer_sum": layer_sum})
        print(f"  dag {d:2d}: {len(nodes):2d} nodes/{len(edges):2d} edges | "
              f"peak {peak:5.2f} GB | sum {naive_sum:5.2f} max {naive_max:5.2f} "
              f"layer {layer_sum:5.2f}", flush=True)

    truth = np.array([r["peak_concurrent_gb"] for r in rows])
    print(f"\n{len(cache)} distinct task configs measured | peak-concurrent "
          f"min/median/max = {truth.min():.2f}/{np.median(truth):.2f}/{truth.max():.2f} GB")
    print(f"\n{'baseline':12} {'MAE(GB)':>9} {'MAPE%':>8} {'within1.5x':>11}")
    for name in ("naive_sum", "naive_max", "layer_sum"):
        pred = np.array([r[name] for r in rows])
        ratio = np.maximum(pred, 1e-6) / np.maximum(truth, 1e-6)
        print(f"{name:12} {np.mean(np.abs(pred-truth)):9.3f} "
              f"{np.mean(np.abs(pred-truth)/truth)*100:8.1f} "
              f"{np.mean((ratio>=1/1.5)&(ratio<=1.5))*100:10.1f}%")

    json.dump({"device": torch.cuda.get_device_name(0), "n_dags": args.n_dags,
               "max_parallel": args.max_parallel, "n_configs_measured": len(cache),
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
