"""
The deterministic 'decider' of the dynamic predictor: an attention-based forecaster.

Maps `lookback` steps of recent multivariate history -> the next HORIZON=30 steps (5 min)
of all 4 channels, using a small Transformer encoder over the history (self-attention across
time). This is the "attention-based workload prediction" class that `research_plan.md` names
as the baseline-to-beat; here it is OUR deterministic numeric model. The LLM is NOT in this
loop — per the project hinge it will sit ON TOP, emitting structured regime facts that adjust
this forecast (pins/forecast/llm_facts.py, next), never emitting the numbers itself.

Channels span orders of magnitude (cpu_util in thousands, gpu_mem in GB), so we standardise
per-channel on the TRAIN set; the model works in z-space and predict() de-standardises.

Run:  .venv/bin/python -m pins.forecast.model            # train + eval vs the baseline gate
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pins.forecast.baselines import (channel_scales, evaluate, moving_avg,
                                     persistence, split_jobs, _print)
from pins.forecast.dataset import CHANNELS, HORIZON, load_all, windows

C = len(CHANNELS)


class AttnForecaster(nn.Module):
    def __init__(self, lookback: int, horizon: int = HORIZON, d_model: int = 64,
                 nhead: int = 4, layers: int = 2):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.inp = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, horizon * C)

    def forward(self, x):                          # x: (B, L, C) standardised
        h = self.tr(self.inp(x) + self.pos)        # (B, L, d)
        y = self.head(h[:, -1])                    # last-step summary -> (B, horizon*C)
        return y.view(-1, self.horizon, C)


def _make_xy(jobs, ids, lookback, stride=3):
    X, Y = [], []
    for j in ids:
        s = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        for hist, fut in windows(s, lookback, HORIZON, stride):
            X.append(hist); Y.append(fut)
    return np.asarray(X), np.asarray(Y)


def train_model(jobs, tr, lookback, mean, std, dstd, device, epochs=40, lr=1e-3, seed=0):
    """Predict the RESIDUAL from persistence: target = future - last_history_value.

    Anchoring to persistence means the model only has to learn the *change*; on flat
    channels it learns ~0 and degenerates to persistence (can't do worse), spending capacity
    only where the signal actually moves. Inputs are standardised on raw stats; residual
    targets on their own per-channel std `dstd` (residual mean is ~0)."""
    torch.manual_seed(seed)
    X, Y = _make_xy(jobs, tr, lookback)
    last = X[:, -1:, :]                            # (N,1,C) persistence anchor
    Xz = torch.tensor((X - mean) / std, device=device)
    Rz = torch.tensor((Y - last) / dstd, device=device)   # standardised residual target
    model = AttnForecaster(lookback).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()                            # match the MAE eval metric
    n, bs = len(Xz), 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xz[idx]), Rz[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  train L1(z) = {tot/n:.4f}")
    return model.eval()


def main():
    LOOKBACK = 30
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    jobs = load_all()
    tr, te = split_jobs(jobs)
    scales = channel_scales(jobs, tr)

    # per-channel standardisation from TRAIN windows; residual std for the delta target
    Xtr, Ytr = _make_xy(jobs, tr, LOOKBACK)
    mean = Xtr.reshape(-1, C).mean(0).astype(np.float32)
    std = (Xtr.reshape(-1, C).std(0) + 1e-6).astype(np.float32)
    dstd = ((Ytr - Xtr[:, -1:, :]).reshape(-1, C).std(0) + 1e-6).astype(np.float32)

    print(f"{len(jobs)} jobs | train={len(tr)} test={len(te)} | device={device} | "
          f"lookback={LOOKBACK} horizon={HORIZON} (5 min)\n")
    model = train_model(jobs, tr, LOOKBACK, mean, std, dstd, device)

    mz = torch.tensor(mean, device=device); sz = torch.tensor(std, device=device)
    dz = torch.tensor(dstd, device=device)

    @torch.no_grad()
    def predict_attn(hist: np.ndarray) -> np.ndarray:
        x = (torch.tensor(hist, device=device) - mz) / sz
        resid = model(x.unsqueeze(0))[0] * dz          # de-standardise residual
        return (torch.tensor(hist[-1], device=device) + resid).cpu().numpy()  # + persistence anchor

    print("\nper-channel MAE in native units; (.) = normalised by train-set std\n")
    _print("persistence", evaluate(persistence, jobs, te, LOOKBACK, scales))
    _print("moving_avg(k=6)", evaluate(moving_avg, jobs, te, LOOKBACK, scales))
    _print("attn (ours)", evaluate(predict_attn, jobs, te, LOOKBACK, scales))


if __name__ == "__main__":
    main()
