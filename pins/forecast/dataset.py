"""
Loader + aligner for the MIT Supercloud dynamic-forecasting task.

Goal (research_plan.md §2, warm-job numeric forecaster): given a running task's recent
telemetry, forecast its GPU/CPU/memory usage 5 minutes ahead. The two raw streams are
sampled at different rates and cover different wall-clock windows, so this module produces
one ALIGNED multivariate frame per job on a common 10 s grid:

    columns = [gpu_util, gpu_mem_gb, cpu_util, mem_gb]   (the 4 forecast channels)

Alignment rules (see the data nuance discovered while sampling):
  - CPU `<jobid>-timeseries.csv` is already on a 10 s grid (EpochTime steps of 10 s).
  - GPU `<jobid>-<node>.csv` is ~100 ms; we resample it to 10 s by MEAN over each bin.
  - The two are INNER-joined on absolute wall-clock time (10 s bins), dropping the
    non-overlapping tails (the monitors start/stop at different times).
Multi-row aggregation per 10 s bin: utilization channels -> mean; memory channels -> mean
across sub-samples (per-GPU / per-series). v1 caveat: for multi-GPU/multi-node jobs this
under-counts total memory; most sampled jobs are single-GPU. Documented, fix later.

This is pure deterministic prep — no LLM, no model. It is the substrate the persistence
baseline and the attention+LLM forecaster both consume.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "..", "..", "data", "supercloud-sample")
BIN_S = 10                      # common grid: 10 s (the CPU cadence)
CHANNELS = ["gpu_util", "gpu_mem_gb", "cpu_util", "mem_gb"]
HORIZON = 30                    # 30 steps * 10 s = 5 min forecast horizon


def _labels(sample_dir: str) -> dict[str, str]:
    p = os.path.join(sample_dir, "labelled_jobids.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype={"id_job": str})
    return dict(zip(df["id_job"], df["model"]))


def load_aligned(jobid: str, sample_dir: str = SAMPLE_DIR) -> pd.DataFrame:
    """Return the aligned 10 s multivariate frame for one job (index = unix-time bin)."""
    # ---- CPU (10 s): CPUUtilization, RSS(KB) -> mem_gb ----
    cpu = pd.read_csv(os.path.join(sample_dir, "cpu", f"{jobid}-timeseries.csv"))
    cpu = cpu[cpu["ElapsedTime"] >= 0]                      # drop the -4 setup rows
    cpu["bin"] = (cpu["EpochTime"] // BIN_S) * BIN_S
    cpu_g = cpu.groupby("bin").agg(cpu_util=("CPUUtilization", "mean"),
                                   rss_kb=("RSS", "sum")).reset_index()
    cpu_g["mem_gb"] = cpu_g["rss_kb"] / 1e6                 # KB -> GB (1e6 KB = 1 GB)

    # ---- GPU (~100 ms): resample to 10 s by mean ----
    frames = []
    for f in sorted(glob.glob(os.path.join(sample_dir, "gpu", f"{jobid}-*.csv"))):
        g = pd.read_csv(f)
        g["bin"] = (g["timestamp"] // BIN_S) * BIN_S
        frames.append(g)
    gpu = pd.concat(frames, ignore_index=True)
    gpu_g = gpu.groupby("bin").agg(gpu_util=("utilization_gpu_pct", "mean"),
                                   gpu_mem_mb=("memory_used_MiB", "mean")).reset_index()
    gpu_g["gpu_mem_gb"] = gpu_g["gpu_mem_mb"] / 1024.0

    # ---- correct the CPU/GPU timezone offset, then inner-join ----
    # The two monitors log the SAME runtime but in different zones: GPU `timestamp` trails
    # CPU `EpochTime` by an exact whole-hour offset (14400 s = EDT in summer; 5 h in winter
    # under DST). Auto-detect the hour shift that maximises bin overlap rather than hardcoding,
    # so it survives DST and any sub-window case. Deterministic — just set arithmetic.
    cb = set(cpu_g["bin"].astype(int))
    gb = set(gpu_g["bin"].astype(int))
    off = max(range(-12, 13), key=lambda h: len(cb & {x + h * 3600 for x in gb}))
    gpu_g["bin"] = gpu_g["bin"] + off * 3600

    m = pd.merge(gpu_g, cpu_g, on="bin", how="inner").sort_values("bin").reset_index(drop=True)
    return m[["bin"] + CHANNELS]


def load_labels(sample_dir: str = SAMPLE_DIR) -> dict[str, str]:
    """jobid -> model-type label for a labelled sample (data/.../labels.csv).

    Written by `fetch_supercloud.py --labelled-only`. Falls back to the full catalog
    `labelled_jobids.csv` if a per-sample labels.csv isn't present."""
    for fname in ("labels.csv", "labelled_jobids.csv"):
        p = os.path.join(sample_dir, fname)
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={"id_job": str})
            return dict(zip(df["id_job"], df["model"]))
    return {}


def load_all(sample_dir: str = SAMPLE_DIR, min_len: int = 60) -> dict[str, pd.DataFrame]:
    """Load every joint job; keep those with >= min_len aligned steps (need history+horizon)."""
    ids = [l.strip() for l in open(os.path.join(sample_dir, "joint_jobids.txt")) if l.strip()]
    out = {}
    for jid in ids:
        try:
            df = load_aligned(jid, sample_dir)
        except Exception as e:                              # noqa: BLE001 — skip malformed jobs, report
            print(f"  skip {jid}: {type(e).__name__} {e}")
            continue
        if len(df) >= min_len:
            out[jid] = df
    return out


def windows(series: np.ndarray, lookback: int, horizon: int = HORIZON, stride: int = 5):
    """Rolling-origin (history, future) pairs. series: (T, C)."""
    T = len(series)
    for t in range(lookback, T - horizon + 1, stride):
        yield series[t - lookback:t], series[t:t + horizon]


if __name__ == "__main__":      # quick smoke / data sanity
    labels = _labels(SAMPLE_DIR)
    data = load_all()
    lens = np.array([len(v) for v in data.values()])
    print(f"loaded {len(data)} jobs with >=60 aligned 10s-steps "
          f"(of 100); len: min={lens.min()} median={int(np.median(lens))} max={lens.max()}")
    jid = list(data)[0]
    df = data[jid]
    print(f"\nexample job {jid}  (label: {labels.get(jid, '—')})  rows={len(df)}")
    print(df[CHANNELS].describe().round(2).to_string())
    print(f"\nlabelled jobs in sample: "
          f"{sum(1 for j in data if j in labels)} / {len(data)}")
