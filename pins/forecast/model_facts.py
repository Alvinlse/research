"""
Step 3 of the LLM-on-top forecaster: condition the deterministic attention forecaster
(pins/forecast/model.py) on a STATIC COVARIATE derived from each job's MODEL-TYPE LABEL through
pins/forecast/llm_facts.py. This is the dynamic-task analogue of the §5/§6 cold-start story.

The hinge is unchanged (CLAUDE.md, research_plan.md §6): the LLM never emits a number. It emits
categorical regime FACTS for the job's architecture; `llm_facts.facts_to_vec` encodes them to a
fixed vector `s`; here a single Linear projects `s` into d_model and ADDS it to every history-step
embedding before the Transformer (the Temporal-Fusion-Transformer "static covariate" pattern).
The network still predicts every number; the facts only *condition* it.

Ablation (the point — mirrors the §6 H2 "does class-conditioning help?" question):
  plain  : no covariate                       -> reproduces model.py, the gate to beat
  onehot : one-hot of the label               -> can memorise per-label behaviour, but CANNOT
                                                 transfer to rare classes (few train examples)
  facts  : LLM regime-facts vector (shared)   -> a rare class (e.g. gnn, n=5) inherits its
                                                 family's regime, so it should beat onehot there

All three share the exact architecture and the residual-from-persistence target of model.py;
only the covariate differs, so any gap is attributable to the conditioning signal.

Run: .venv-forecast/bin/python -m pins.forecast.model_facts            # facts vs onehot vs plain
     .venv-forecast/bin/python -m pins.forecast.model_facts --no-llm   # facts via rule fallback
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from pins.forecast import llm_facts
from pins.forecast.baselines import (channel_scales, evaluate, moving_avg,
                                     persistence, split_jobs, _print)
from pins.forecast.dataset import CHANNELS, HORIZON, load_all, load_labels, windows

C = len(CHANNELS)
SAMPLE_DIR = "data/supercloud-labelled"


class FactsForecaster(nn.Module):
    """AttnForecaster (model.py) + an optional static covariate added to every step.

    cov_dim == 0 reproduces the plain model exactly (no projection, covariate ignored)."""

    def __init__(self, lookback: int, cov_dim: int, horizon: int = HORIZON,
                 d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.lookback, self.horizon, self.cov_dim = lookback, horizon, cov_dim
        self.inp = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        self.cov = nn.Linear(cov_dim, d_model) if cov_dim > 0 else None
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, horizon * C)

    def forward(self, x, s=None):                       # x:(B,L,C)  s:(B,cov_dim)
        h = self.inp(x) + self.pos                       # (B,L,d)
        if self.cov is not None and s is not None:
            h = h + self.cov(s).unsqueeze(1)             # broadcast static -> every step
        h = self.tr(h)
        return self.head(h[:, -1]).view(-1, self.horizon, C)


def _make_xy_cov(jobs, ids, lookback, cov_map, stride=3):
    """Rolling (history, future, static-covariate) triples. cov is constant within a job."""
    X, Y, S = [], [], []
    for j in ids:
        s = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        cov = cov_map[j]
        for hist, fut in windows(s, lookback, HORIZON, stride):
            X.append(hist); Y.append(fut); S.append(cov)
    return np.asarray(X), np.asarray(Y), np.asarray(S, dtype=np.float32)


def train(jobs, tr, lookback, cov_map, cov_dim, mean, std, dstd, device,
          epochs=40, lr=1e-3, seed=0):
    """Residual-from-persistence target (model.py's winning design), now conditioned on `s`."""
    torch.manual_seed(seed)
    X, Y, S = _make_xy_cov(jobs, tr, lookback, cov_map)
    last = X[:, -1:, :]
    Xz = torch.tensor((X - mean) / std, device=device)
    Rz = torch.tensor((Y - last) / dstd, device=device)
    Sz = torch.tensor(S, device=device)
    model = FactsForecaster(lookback, cov_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    n, bs = len(Xz), 256
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xz[idx], Sz[idx]), Rz[idx])
            loss.backward(); opt.step()
    return model.eval()


def eval_conditioned(model, jobs, ids, lookback, scales, cov_map, mean, std, dstd, device):
    """Like baselines.evaluate but feeds each job's static covariate. Same MAE/nMAE metric."""
    mz = torch.tensor(mean, device=device); sz = torch.tensor(std, device=device)
    dz = torch.tensor(dstd, device=device)

    @torch.no_grad()
    def predict(hist, cov):
        x = (torch.tensor(hist, device=device) - mz) / sz
        s = torch.tensor(cov, device=device).unsqueeze(0)
        resid = model(x.unsqueeze(0), s)[0] * dz
        return (torch.tensor(hist[-1], device=device) + resid).cpu().numpy()

    abs_err = np.zeros(C); n = 0
    for j in ids:
        series = jobs[j][CHANNELS].to_numpy().astype(np.float32)
        for hist, fut in windows(series, lookback, HORIZON, stride=5):
            abs_err += np.abs(predict(hist, cov_map[j]) - fut).mean(axis=0)
            n += 1
    mae = abs_err / max(n, 1); nmae = mae / scales
    return {"n_windows": n,
            "mae": {c: round(float(m), 4) for c, m in zip(CHANNELS, mae)},
            "nmae": {c: round(float(m), 4) for c, m in zip(CHANNELS, nmae)},
            "nmae_mean": round(float(nmae.mean()), 4)}


def build_cov_maps(jobs, labels, use_llm, model_name):
    """Return {mode: (jobid->vec, cov_dim)} for plain / onehot / facts."""
    ids = list(jobs)
    uniq = sorted({labels[j] for j in ids})
    idx = {lab: i for i, lab in enumerate(uniq)}

    # facts: one LLM call per distinct label -> 12-d regime vector
    fact_vecs = llm_facts.label_vectors([labels[j] for j in ids], use_llm, model_name)
    facts_map = {j: fact_vecs[labels[j]] for j in ids}

    def onehot(lab):
        v = np.zeros(len(uniq), dtype=np.float32); v[idx[lab]] = 1.0; return v
    onehot_map = {j: onehot(labels[j]) for j in ids}

    zero_map = {j: np.zeros(1, dtype=np.float32) for j in ids}
    return {"plain": (zero_map, 0),
            "onehot": (onehot_map, len(uniq)),
            "facts": (facts_map, llm_facts.FEATURE_DIM)}


def run(lookback, jobs, tr, te, scales, cov_maps, device, tag=""):
    Xtr, Ytr, _ = _make_xy_cov(jobs, tr, lookback, cov_maps["plain"][0])
    mean = Xtr.reshape(-1, C).mean(0).astype(np.float32)
    std = (Xtr.reshape(-1, C).std(0) + 1e-6).astype(np.float32)
    dstd = ((Ytr - Xtr[:, -1:, :]).reshape(-1, C).std(0) + 1e-6).astype(np.float32)

    print(f"\n=== lookback={lookback} ({lookback*10}s history) {tag} ===")
    _print("persistence", evaluate(persistence, jobs, te, lookback, scales))
    _print("moving_avg(k=6)", evaluate(moving_avg, jobs, te, lookback, scales))
    for mode in ("plain", "onehot", "facts"):
        cov_map, cov_dim = cov_maps[mode]
        model = train(jobs, tr, lookback, cov_map, cov_dim, mean, std, dstd, device)
        r = eval_conditioned(model, jobs, te, lookback, scales, cov_map, mean, std, dstd, device)
        _print(f"attn+{mode}", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="facts via rule fallback (no Ollama)")
    ap.add_argument("--model", default=llm_facts.DEFAULT_MODEL)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    jobs = load_all(SAMPLE_DIR, min_len=60)
    labels = load_labels(SAMPLE_DIR)
    jobs = {j: d for j, d in jobs.items() if j in labels}     # keep only labelled
    tr, te = split_jobs(jobs)
    scales = channel_scales(jobs, tr)
    cov_maps = build_cov_maps(jobs, labels, not args.no_llm, args.model)

    print(f"{len(jobs)} labelled jobs | train={len(tr)} test={len(te)} | device={device} | "
          f"facts={'rule' if args.no_llm else args.model}")
    print("per-channel MAE(nMAE); nMAE=MAE/std; lower is better. Watch gpu_util/cpu_util "
          "(the dynamic channels) and the cold-start block.")

    run(30, jobs, tr, te, scales, cov_maps, device, tag="(warm: 5 min history)")
    run(6,  jobs, tr, te, scales, cov_maps, device, tag="(COLD-START: 1 min history)")


if __name__ == "__main__":
    main()
