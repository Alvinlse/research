"""
Baselines + the shared evaluation harness for dynamic resource forecasting.

The whole static Stage-1 result was only credible because a *dumb baseline* (mean / params
heuristic) was there to beat. Same discipline here: before any LLM/attention model, establish
what trivial forecasters score, so "the model helps" means something.

Task: from `lookback` steps of history (10 s grid), predict the next `HORIZON`=30 steps
(= 5 min) of all 4 channels [gpu_util, gpu_mem_gb, cpu_util, mem_gb], rolling-origin.

Baselines:
  - persistence : hold the last observed value flat for 5 min (the canonical TS baseline).
  - moving_avg  : repeat the mean of the last k steps.

A forecaster is any `f(history: (L,C)) -> (HORIZON,C)`. `evaluate()` scores per-channel MAE
in native units plus a scale-normalised nMAE (MAE / per-channel std) so channels on wildly
different scales (cpu_util in the thousands, gpu_mem in GB) aggregate into one comparable number.
"""
from __future__ import annotations

import numpy as np

from pins.forecast.dataset import CHANNELS, HORIZON, load_all, windows


# ------------------------------- baselines ----------------------------------
def persistence(history: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    return np.repeat(history[-1:][:], horizon, axis=0)


def moving_avg(history: np.ndarray, horizon: int = HORIZON, k: int = 6) -> np.ndarray:
    return np.repeat(history[-k:].mean(axis=0, keepdims=True), horizon, axis=0)


# ----------------------------- eval harness ---------------------------------
def split_jobs(jobs: dict, frac_train: float = 0.7, seed: int = 0):
    ids = sorted(jobs)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = int(len(ids) * frac_train)
    return ids[:n], ids[n:]


def channel_scales(jobs: dict, ids: list[str]) -> np.ndarray:
    """Per-channel std over the training jobs — used to normalise MAE across channels."""
    allv = np.concatenate([jobs[j][CHANNELS].to_numpy() for j in ids], axis=0)
    return allv.std(axis=0) + 1e-6


def evaluate(predict_fn, jobs: dict, ids: list[str], lookback: int,
             scales: np.ndarray, horizon: int = HORIZON, stride: int = 5) -> dict:
    """Per-channel MAE (native) + normalised nMAE, averaged over all rolling windows."""
    abs_err = np.zeros(len(CHANNELS))
    n = 0
    for j in ids:
        series = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        for hist, fut in windows(series, lookback, horizon, stride):
            pred = predict_fn(hist)
            abs_err += np.abs(pred - fut).mean(axis=0)   # mean over horizon, per channel
            n += 1
    mae = abs_err / max(n, 1)
    nmae = mae / scales
    return {"n_windows": n,
            "mae": {c: round(float(m), 4) for c, m in zip(CHANNELS, mae)},
            "nmae": {c: round(float(m), 4) for c, m in zip(CHANNELS, nmae)},
            "nmae_mean": round(float(nmae.mean()), 4)}


def _print(name: str, r: dict) -> None:
    mae = r["mae"]
    print(f"{name:16} nMAE_mean={r['nmae_mean']:.3f} | "
          + "  ".join(f"{c}={mae[c]:.3f}({r['nmae'][c]:.2f})" for c in CHANNELS))


if __name__ == "__main__":
    LOOKBACK = 30
    jobs = load_all()
    tr, te = split_jobs(jobs)
    scales = channel_scales(jobs, tr)
    print(f"{len(jobs)} jobs | train={len(tr)} test={len(te)} | lookback={LOOKBACK} "
          f"horizon={HORIZON} (5 min) | metric: MAE(nMAE), nMAE=MAE/std\n")
    print("per-channel MAE in native units; (.) = normalised by train-set std\n")
    _print("persistence", evaluate(persistence, jobs, te, LOOKBACK, scales))
    _print("moving_avg(k=6)", evaluate(moving_avg, jobs, te, LOOKBACK, scales))
