"""
Stage-1 DYNAMIC, with UNCERTAINTY: a quantile-regression forecaster.

Exp 8 (pins/forecast/model.py) forecasts a single trajectory (a POINT estimate, L1 loss).
research_plan.md's prediction co-contribution needs more than a point: *"forecast each task's
resource demand over the next T timesteps WITH AN EXPLICIT UNCERTAINTY ESTIMATE, which sizes
the safety margin the demand agent bids for."* This module adds that — predict P10/P50/P90 per
channel via pinball (quantile) loss, so the interval [P10,P90] is the uncertainty the Stage-2
demand agent consumes (pins/predictor.marginal_values, pins/uncertainty_sim.py).

Design (keeps the Exp-8 hinge and tricks):
  * Same small Transformer encoder over the lookback history; the LLM is NOT in this loop.
  * P50 is still predicted as a RESIDUAL from persistence (Exp-8's anchor), so flat channels
    degenerate to persistence.
  * The two interval edges are predicted as NON-NEGATIVE offsets from P50 via softplus, so by
    construction P10 <= P50 <= P90 — quantile crossing is impossible.
  * Pinball loss on the standardised residual (sum over the 3 quantiles).

Reported beside the Exp-8 MAE gate: P50 nMAE (must still beat persistence — uncertainty must
not cost accuracy), empirical COVERAGE of [P10,P90] (target ~80%), and mean interval WIDTH
(sharpness). Writes per-job uncertainty -> results_quantile.json for the Stage-2 bridge.

Run (in the forecast venv):  .venv-forecast/bin/python -m pins.forecast.model_quantile
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pins.forecast.baselines import (channel_scales, evaluate, persistence,
                                     split_jobs, _print)
from pins.forecast.dataset import CHANNELS, HORIZON, load_all, windows
from pins.forecast.model import _make_xy

C = len(CHANNELS)
QUANTILES = (0.1, 0.5, 0.9)                 # P10 / P50 / P90
Q = len(QUANTILES)
DEMAND_CHANNELS = ("gpu_util", "gpu_mem_gb")  # the channels whose uncertainty sizes the GPU bid
HERE = os.path.dirname(os.path.abspath(__file__))


class QuantileForecaster(nn.Module):
    """Transformer encoder over history -> per (horizon, channel): a P50 residual plus two
    non-negative half-widths. P10 = P50 - lo, P90 = P50 + hi (monotone by construction)."""

    def __init__(self, lookback: int, horizon: int = HORIZON, d_model: int = 64,
                 nhead: int = 4, layers: int = 2):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.inp = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, horizon * C * 3)   # [resid50, lo_raw, hi_raw] per (h,c)

    def forward(self, x):                          # x: (B, L, C) standardised
        h = self.tr(self.inp(x) + self.pos)
        y = self.head(h[:, -1]).view(-1, self.horizon, C, 3)
        p50 = y[..., 0]
        lo = F.softplus(y[..., 1])                 # >= 0 : how far P10 sits below P50
        hi = F.softplus(y[..., 2])                 # >= 0 : how far P90 sits above P50
        # stack as the three quantiles (z-space residuals), guaranteed non-crossing
        return torch.stack([p50 - lo, p50, p50 + hi], dim=-1)   # (B, H, C, Q)


def pinball_loss(pred_q, target):
    """Mean pinball/quantile loss. pred_q: (..., Q) residual-quantile predictions; target: (...)
    the realised residual. For quantile tau, penalise under/over-prediction asymmetrically."""
    taus = torch.tensor(QUANTILES, device=pred_q.device).view(*([1] * (pred_q.dim() - 1)), Q)
    e = target.unsqueeze(-1) - pred_q                       # (..., Q)
    return torch.maximum(taus * e, (taus - 1.0) * e).mean()


def train_model(jobs, tr, lookback, mean, std, dstd, device, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    X, Y = _make_xy(jobs, tr, lookback)
    last = X[:, -1:, :]
    Xz = torch.tensor((X - mean) / std, device=device)
    Rz = torch.tensor((Y - last) / dstd, device=device)     # standardised residual target
    model = QuantileForecaster(lookback).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, bs = len(Xz), 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = pinball_loss(model(Xz[idx]), Rz[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 15 == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  train pinball(z) = {tot/n:.4f}")
    return model.eval()


def coverage_width(predict_q, jobs, ids, lookback, scales, horizon=HORIZON, stride=5):
    """Empirical coverage of [P10,P90] and mean interval width per channel, over all windows.
    coverage target = P90-P10 = 0.8. Width normalised by per-channel std for an aggregate."""
    inside = np.zeros(C)
    width = np.zeros(C)
    per_job = {}
    n_tot = 0
    for j in ids:
        series = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        jin = np.zeros(C); jw = np.zeros(C); jn = 0
        for hist, fut in windows(series, lookback, horizon, stride):
            p10, _p50, p90 = predict_q(hist)                # each (horizon, C)
            jin += ((fut >= p10) & (fut <= p90)).mean(axis=0)
            jw += (p90 - p10).mean(axis=0)
            jn += 1
        if jn:
            inside += jin; width += jw; n_tot += jn
            per_job[j] = (jw / jn) / scales                 # normalised width per channel
    cov = inside / max(n_tot, 1)
    w = width / max(n_tot, 1)
    return cov, w, per_job


def job_uncertainty(per_job_norm_width: dict[str, np.ndarray]) -> dict[str, float]:
    """Collapse a job's per-channel normalised interval width into one uncertainty scalar in
    ~[0,1]: the mean normalised width over the GPU-demand channels, clipped. This is the signal
    the demand agent uses to size its safety margin (pins/predictor.marginal_values)."""
    idx = [CHANNELS.index(c) for c in DEMAND_CHANNELS]
    out = {}
    for j, w in per_job_norm_width.items():
        out[j] = float(np.clip(np.mean([w[i] for i in idx]), 0.0, 1.0))
    return out


def conformal_q(predict_q, jobs, cal_ids, lookback, alpha=0.2, horizon=HORIZON, stride=5):
    """Split-conformal (CQR, Romano et al. 2019) per-channel width adjustment. On a held-out
    CALIBRATION set, the conformity score is how far the truth fell OUTSIDE [P10,P90]
    (`max(P10-y, y-P90)`; negative if comfortably inside). Adding the finite-sample (1-alpha)
    quantile of those scores to each interval edge gives test coverage ~ 1-alpha with a
    distribution-free guarantee. Q<0 is allowed: if the raw intervals were too WIDE, conformal
    SHRINKS them. Per-channel because the channels live on different scales."""
    scores = [[] for _ in range(C)]
    for j in cal_ids:
        series = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        for hist, fut in windows(series, lookback, horizon, stride):
            p10, _p50, p90 = predict_q(hist)
            e = np.maximum(p10 - fut, fut - p90)                # (H,C); >0 outside, <0 inside
            for c in range(C):
                scores[c].extend(e[:, c].tolist())
    Q = np.zeros(C, dtype=np.float32)
    for c in range(C):
        s = np.sort(np.asarray(scores[c]))
        n = len(s)
        k = min(max(int(np.ceil((n + 1) * (1 - alpha))) - 1, 0), n - 1)
        Q[c] = s[k]
    return Q


def _report_cov(name, predict_q, jobs, ids, lookback, scales):
    cov, width, per_job = coverage_width(predict_q, jobs, ids, lookback, scales)
    print(f"\n=== {name} (interval [P10,P90], target coverage 0.80) ===")
    for i, c in enumerate(CHANNELS):
        print(f"{c:12} coverage={cov[i]:.2f}  mean_width={width[i]:.3f} "
              f"(norm {width[i]/scales[i]:.2f})")
    print(f"{'aggregate':12} coverage={cov.mean():.2f}  norm_width={(width/scales).mean():.2f}")
    return cov, width, per_job


def main():
    LOOKBACK = 30
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    jobs = load_all()
    tr_all, te = split_jobs(jobs)
    # carve a CALIBRATION split out of train: fit the model on `fit`, conformalise on `cal`.
    order = list(tr_all)
    np.random.default_rng(1).shuffle(order)
    ncal = max(10, len(tr_all) // 4)
    cal, fit = order[:ncal], order[ncal:]
    scales = channel_scales(jobs, fit)

    Xtr, Ytr = _make_xy(jobs, fit, LOOKBACK)
    mean = Xtr.reshape(-1, C).mean(0).astype(np.float32)
    std = (Xtr.reshape(-1, C).std(0) + 1e-6).astype(np.float32)
    dstd = ((Ytr - Xtr[:, -1:, :]).reshape(-1, C).std(0) + 1e-6).astype(np.float32)

    print(f"{len(jobs)} jobs | fit={len(fit)} cal={len(cal)} test={len(te)} | device={device} | "
          f"lookback={LOOKBACK} horizon={HORIZON} (5 min) | quantiles={QUANTILES}\n")
    model = train_model(jobs, fit, LOOKBACK, mean, std, dstd, device)

    mz = torch.tensor(mean, device=device); sz = torch.tensor(std, device=device)
    dz = torch.tensor(dstd, device=device)

    @torch.no_grad()
    def predict_q(hist: np.ndarray):
        x = (torch.tensor(hist, device=device) - mz) / sz
        rq = model(x.unsqueeze(0))[0] * dz.unsqueeze(-1)        # (H, C, Q) de-standardised resid
        anchor = torch.tensor(hist[-1], device=device).unsqueeze(0).unsqueeze(-1)  # (1,C,1)
        vq = (anchor + rq).cpu().numpy()                        # (H, C, Q) value quantiles
        return vq[..., 0], vq[..., 1], vq[..., 2]               # p10, p50, p90 each (H,C)

    # conformal width adjustment from the calibration split
    Qc = conformal_q(predict_q, jobs, cal, LOOKBACK)

    def predict_q_cal(hist):
        p10, p50, p90 = predict_q(hist)
        return p10 - Qc, p50, p90 + Qc                          # calibrated interval

    def predict_p50(hist):                                       # for the Exp-8 MAE gate
        return predict_q(hist)[1]

    print("\n=== P50 accuracy gate (must still beat persistence) ===")
    print("per-channel MAE in native units; (.) = normalised by train-set std\n")
    _print("persistence", evaluate(persistence, jobs, te, LOOKBACK, scales))
    _print("quantile P50", evaluate(predict_p50, jobs, te, LOOKBACK, scales))

    _report_cov("RAW uncertainty", predict_q, jobs, te, LOOKBACK, scales)
    print(f"\nconformal per-channel width add (native): "
          + "  ".join(f"{c}={Qc[i]:+.3f}" for i, c in enumerate(CHANNELS)))
    cov, width, per_job = _report_cov("CALIBRATED uncertainty", predict_q_cal, jobs, te,
                                      LOOKBACK, scales)

    uncert = job_uncertainty(per_job)                            # from CALIBRATED widths
    vals = np.array(list(uncert.values()))
    print(f"\nper-job uncertainty (calibrated norm width over {DEMAND_CHANNELS}): "
          f"min={vals.min():.2f} median={np.median(vals):.2f} max={vals.max():.2f}")

    out = os.path.join(HERE, "results_quantile.json")
    with open(out, "w") as f:
        json.dump({
            "quantiles": list(QUANTILES),
            "calibrated": True,
            "conformal_width_add": {c: round(float(Qc[i]), 4) for i, c in enumerate(CHANNELS)},
            "coverage": {c: round(float(cov[i]), 3) for i, c in enumerate(CHANNELS)},
            "width_native": {c: round(float(width[i]), 4) for i, c in enumerate(CHANNELS)},
            "aggregate_coverage": round(float(cov.mean()), 3),
            "per_job_uncertainty": {j: round(u, 3) for j, u in uncert.items()},
            "uncertainty_summary": {"min": round(float(vals.min()), 3),
                                    "median": round(float(np.median(vals)), 3),
                                    "max": round(float(vals.max()), 3)},
        }, f, indent=2)
    print(f"\ncalibrated uncertainty artifact -> {out}  ({len(uncert)} jobs)")


if __name__ == "__main__":
    main()
