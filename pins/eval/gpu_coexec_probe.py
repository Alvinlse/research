"""
Stage-1 (DAG track) Exp 21C — does GPU UTILIZATION co-execute non-additively?

The memory result (Exp 21A/B) showed peak concurrent GPU *memory* is additive: a deterministic
heaviest-level sum predicts it to ~3%, so a learned model is over-engineering. But GPU
UTILIZATION is a different beast — two 60%-util tasks co-running do NOT give 120%; they contend
for SMs / memory bandwidth and the device saturates, while each task SLOWS DOWN by an amount
that depends on the MIX of co-resident kernels. If realized co-run utilization (and slowdown)
can't be predicted from per-task solo stats by a simple rule, THAT is the relational regime
where a DAG -> graph-attention -> util model has a baseline worth beating.

This probe MEASURES that non-additivity directly (no model yet):
  * SOLO: run each task config alone, measure its duration + mean device util (nvidia-smi).
  * CO-RUN: launch k tasks as concurrent SUBPROCESSES (each its own CUDA context, real
    contention — how a scheduler actually co-locates jobs), started together via a Barrier so
    they overlap. Measure realized device util in the overlap window and each task's slowdown.
  * Compare against the additive baselines a simple rule would use:
      util_pred  = min(100, sum(solo_util))           # additive-with-cap
      makespan_pred = max(solo_dur)                    # perfect parallelism (no interference)
  * The GAP (realized vs predicted, and whether the slowdown depends on which tasks co-run) is
    the evidence for/against the learned model.

Honest scope: only CNN (conv-bound) tasks here, so this establishes saturation + slowdown
non-additivity; MIX-dependence across task TYPES (compute- vs bandwidth-bound) is the stronger
follow-up that most justifies attention. Utilization is sampled at the DEVICE level (nvidia-smi
gives whole-GPU util, which is exactly the contended quantity).

Run (GPU venv; .venv torch is broken):
  .venv-forecast/bin/python -m pins.eval.gpu_coexec_probe
  .venv-forecast/bin/python -m pins.eval.gpu_coexec_probe --quick      # tiny smoke test
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results_gpu_coexec.json")

# Compute-heavier configs (high solo util) so co-running genuinely contends; moderate memory so
# up to 3 co-fit in 40 GB. Varied size for a spread of solo utilisation.
CO_CONFIGS = [
    {"id": "s",  "width": 64,  "blocks": 3, "res": 96,  "batch": 128},
    {"id": "m",  "width": 96,  "blocks": 3, "res": 96,  "batch": 128},
    {"id": "l",  "width": 128, "blocks": 3, "res": 128, "batch": 128},
    {"id": "d",  "width": 96,  "blocks": 4, "res": 96,  "batch": 128},
    {"id": "w",  "width": 64,  "blocks": 3, "res": 96,  "batch": 256},
]


def _worker(cfg, run_s, barrier, q, idx):
    """Run for a fixed wall-window in a private CUDA context; report steps completed + timestamps.
    Fixed duration => all co-runners overlap fully, giving a long, stable utilisation window and
    a throughput-based slowdown (robust to per-step timing noise)."""
    import torch
    import torch.nn as nn
    from pins.eval.dag_gpu_bench import SimpleCNN
    dev = "cuda:0"
    torch.cuda.set_device(0)
    model = SimpleCNN(cfg["width"], cfg["blocks"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    b, res = cfg["batch"], cfg["res"]

    def step():
        x = torch.randn(b, 3, res, res, device=dev)
        y = torch.randint(0, 10, (b,), device=dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()

    for _ in range(3):                      # warmup: CUDA init + cudnn autotune (excluded)
        step()
    torch.cuda.synchronize()
    barrier.wait()                          # all co-runners start the timed region together
    t0 = time.time()
    n = 0
    while time.time() - t0 < run_s:
        step()
        n += 1
    torch.cuda.synchronize()
    t1 = time.time()
    q.put((idx, n, t0, t1))                 # steps completed + window


class Sampler:
    """Stream whole-device utilisation via nvidia-smi; timestamp each sample on our clock."""
    def __init__(self, period_ms=50):
        self.samples: list[tuple[float, float, float]] = []   # (t, util%, mem_MiB)
        self.period_ms = period_ms

    def start(self):
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
            stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            try:
                u, m = line.split(",")
                self.samples.append((time.time(), float(u), float(m)))
            except Exception:
                pass

    def stop(self):
        self.p.terminate()

    def window(self, t0, t1):
        us = [u for t, u, _ in self.samples if t0 <= t <= t1]
        return float(np.mean(us)) if us else float("nan")


def run_set(ctx, cfgs, run_s, sampler):
    """Co-run len(cfgs) tasks for run_s seconds each; return (throughputs, overlap_util).
    throughput[i] = steps/s of task i (solo or contended)."""
    k = len(cfgs)
    barrier = ctx.Barrier(k)
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(c, run_s, barrier, q, i)) for i, c in enumerate(cfgs)]
    for p in procs:
        p.start()
    res = [q.get() for _ in range(k)]
    for p in procs:
        p.join()
    res.sort(key=lambda r: r[0])
    tps = [n / (t1 - t0) for (_, n, t0, t1) in res]
    starts = [r[2] for r in res]
    ends = [r[3] for r in res]
    overlap_util = sampler.window(max(starts), min(ends))   # window where ALL k overlap
    return tps, overlap_util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-s", type=float, default=3.0, help="fixed wall-window per task (s)")
    ap.add_argument("--trials", type=int, default=12, help="number of random co-run sets")
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2 configs, 1 pair (smoke test)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA — run on the A100 node via .venv-forecast")
    ctx = mp.get_context("spawn")
    configs = CO_CONFIGS[:2] if args.quick else CO_CONFIGS
    rng = np.random.default_rng(args.seed)

    sampler = Sampler()
    sampler.start()
    time.sleep(0.5)
    print(f"device sampling live | {len(configs)} configs\n", flush=True)

    # 1) SOLO baselines: throughput (steps/s) + mean device util alone
    solo = {}
    print("SOLO  id   tput(st/s)  util%")
    for c in configs:
        tps, util = run_set(ctx, [c], args.run_s, sampler)
        solo[c["id"]] = {"tput": tps[0], "util": util}
        print(f"      {c['id']:3} {tps[0]:10.1f} {util:6.1f}", flush=True)

    # 2) CO-RUN trials
    trials = [configs[:2]] if args.quick else [
        [configs[i] for i in rng.choice(len(configs), size=int(rng.integers(2, args.max_k + 1)),
                                        replace=False)]
        for _ in range(args.trials)]

    print(f"\nCO-RUN (full overlap)            util: realized  add-cap | slowdown (tput_solo/tput_co)")
    rows = []
    for s in trials:
        ids = [c["id"] for c in s]
        tps, util = run_set(ctx, s, args.run_s, sampler)
        add_cap = min(100.0, sum(solo[i]["util"] for i in ids))
        slow = [solo[i]["tput"] / tp if tp > 0 else float("nan") for i, tp in zip(ids, tps)]
        mean_slow = float(np.mean(slow))
        rows.append({"set": ids, "k": len(s), "util_realized": util, "util_add_cap": add_cap,
                     "slowdown_mean": mean_slow, "slowdown_per_task": slow})
        print(f"  {'+'.join(ids):16} k={len(s)}   {util:7.1f}  {add_cap:7.1f} | "
              f"mean {mean_slow:5.2f}x  per-task {[round(x,2) for x in slow]}", flush=True)

    sampler.stop()

    # 3) the headline non-additivity numbers
    ur = np.array([r["util_realized"] for r in rows])
    ua = np.array([r["util_add_cap"] for r in rows])
    sd = np.array([r["slowdown_mean"] for r in rows])
    # mix-dependence: do same-k sets differ in slowdown? (the relational signal attention captures)
    by_k = {k: sd[[r["k"] == k for r in rows]] for k in sorted({r["k"] for r in rows})}
    print(f"\nrealized co-run util:  mean {np.nanmean(ur):.1f}%   (additive-cap predicts {np.nanmean(ua):.1f}%)")
    print(f"slowdown (1.0=no interference): mean {np.nanmean(sd):.2f}x  max {np.nanmax(sd):.2f}x")
    for k, v in by_k.items():
        if len(v) > 1:
            print(f"   k={k}: slowdown spread {np.nanmin(v):.2f}-{np.nanmax(v):.2f}x "
                  f"(std {np.nanstd(v):.2f}) over {len(v)} sets -> mix-dependent")
    verdict = "NON-additive (model justified)" if np.nanmean(sd) > 1.15 else "roughly additive"
    print(f"=> util additive-rule error {np.nanmean(np.abs(ur-ua)):.1f} pts; slowdown "
          f"{np.nanmean(sd):.2f}x => {verdict}")

    json.dump({"device": torch.cuda.get_device_name(0), "solo": solo, "trials": rows},
              open(args.out, "w"), indent=2)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
