"""Measured GPU usage per replay job — the SCORING ground truth for Exp 78 (tier 3).

`pai_sensor_table.csv` (1.06 GB) records what each worker actually did:

    job_name, task_name, inst_name, worker_name, machine, gpu_name, cpu_usage,
    gpu_wrk_util, avg_mem, max_mem, avg_gpu_wrk_mem, max_gpu_wrk_mem, read, write, ...

This module aggregates `gpu_wrk_util` (and GPU memory) per job so the correction layer's
judgement can be scored against MEASUREMENT rather than against predicates we wrote ourselves.

DIRECTION OF USE — the thing that must not be got backwards. This data is ground truth ONLY.
It is never an input to the bid, to `facts["usable"]`, or to the demand reviewer's prompt. The
reviewer sees only the role/instance context (`job_context.csv`), which is what a human
operator would have BEFORE the job runs; the sensor table is what we, afterwards, use to check
whether its judgement was right. Feeding measured utilisation into the bid would mean the
reviewer was reading a fact already priced in — which would test double-counting resistance,
not exception handling, while looking like a win.

Output: data/alibaba-gpu-v2020/job_usage.csv
    job_name, n_workers, gpu_util_mean, gpu_util_max, gpu_mem_mean, gpu_mem_max, cpu_mean
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020")
REPLAY = os.path.join(DATA, "replay_jobs.csv")
SENSOR = os.path.join(DATA, "pai_sensor_table.csv")
OUT = os.path.join(DATA, "job_usage.csv")

I_JOB, I_CPU, I_GPU_UTIL, I_GPU_MEM_AVG, I_GPU_MEM_MAX = 0, 6, 7, 10, 11


def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    want = {r["job_name"] for r in csv.DictReader(open(REPLAY))}
    print(f"replay jobs: {len(want):,}")

    acc: dict[str, dict] = {}
    seen = 0
    with open(SENSOR) as f:
        for row in csv.reader(f):
            seen += 1
            if seen % 1000000 == 0:
                print(f"\r  scanned {seen:,} sensor rows, matched {len(acc):,} jobs",
                      end="", flush=True)
            if len(row) <= I_GPU_MEM_MAX:
                continue
            jid = row[I_JOB]
            if jid not in want:
                continue
            a = acc.setdefault(jid, {"util": [], "mem": [], "cpu": [], "n": 0})
            a["n"] += 1
            for key, idx in (("util", I_GPU_UTIL), ("mem", I_GPU_MEM_MAX), ("cpu", I_CPU)):
                v = _f(row[idx])
                if v is not None:
                    a[key].append(v)
    print(f"\r  scanned {seen:,} sensor rows, matched {len(acc):,} jobs")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_name", "n_workers", "gpu_util_mean", "gpu_util_max",
                    "gpu_mem_mean", "gpu_mem_max", "cpu_mean"])
        for jid, a in acc.items():
            u, m, c = a["util"], a["mem"], a["cpu"]
            w.writerow([jid, a["n"],
                        f"{st.mean(u):.3f}" if u else "",
                        f"{max(u):.3f}" if u else "",
                        f"{st.mean(m):.3f}" if m else "",
                        f"{max(m):.3f}" if m else "",
                        f"{st.mean(c):.3f}" if c else ""])
    print(f"coverage: {len(acc):,}/{len(want):,} ({len(acc)/max(len(want),1):.1%}) -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
