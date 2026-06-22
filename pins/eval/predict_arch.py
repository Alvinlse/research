"""
Stage-1 prediction — does the deterministic estimator GENERALISE across architectures?

`predict_cnn.py --deterministic` won the gate on a VGG-style CNN family (mem MAE 0.04GB,
100% within 1.5x) and survived fp16/bf16. Open question (research_progress.md, next-step 2):
does the same LLM-shapes -> code-sum + calibrated-overhead recipe hold for **skip-connection
ResNets** and a **small Transformer LM**, with ONE global (a,b)?

Two changes from predict_cnn.py:
  1. Architecture-agnostic activation extractor: instead of analytically replaying one known
     recipe (`feature_map_elements`), we register forward hooks on every LEAF module and sum
     its output activations. This handles residual adds, attention blocks, embeddings, etc.
     (In production the LLM emits these per-layer shapes; here we read them off the model we
     build anyway to measure truth.)
  2. Three model families in one pool; (a,b) is calibrated GLOBALLY, leave-one-out, so a job
     is never predicted with constants fit on itself.

Design principle unchanged (CLAUDE.md): the LLM would supply shapes; deterministic code does
all arithmetic and the calibration. No LLM number in the loop.

Run:  .venv/bin/python -m pins.eval.predict_arch
      .venv/bin/python -m pins.eval.predict_arch --precision fp16
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pins.eval.predict_cnn import SimpleCNN
from pins.eval.predict_resources import heuristic_baseline, score, fmt

HERE = os.path.dirname(__file__)


# ------------------------------ ResNet family -------------------------------
class BasicBlock(nn.Module):
    """Standard ResNet basic block — the skip connection is the point of the test."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.r1 = nn.ReLU(inplace=True)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                      nn.BatchNorm2d(cout))
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.r1(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return self.r2(out + idt)


class SmallResNet(nn.Module):
    def __init__(self, width: int = 64, stages=(2, 2, 2), n_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        blocks = []
        cin = width
        w = width
        for si, n in enumerate(stages):
            for bi in range(n):
                stride = 2 if (bi == 0 and si > 0) else 1
                blocks.append(BasicBlock(cin, w, stride))
                cin = w
            w *= 2
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ---------------------------- Transformer family ----------------------------
class TinyLM(nn.Module):
    """A small GPT-ish encoder LM. Activation memory here is dominated by the
    attention score matrix (B, heads, seq, seq) — which scales with seq^2, a regime
    a per-layer-OUTPUT proxy cannot see (scores are never a module output)."""

    def __init__(self, vocab: int = 2000, d_model: int = 256, nhead: int = 4,
                 layers: int = 4, seq: int = 128):
        super().__init__()
        self.seq = seq
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):                       # x: (B, seq) long
        h = self.emb(x) + self.pos
        return self.head(self.tr(h))            # (B, seq, vocab)


# -------------------------------- the jobs ----------------------------------
def _img_sample(b, res, device):
    return torch.randn(b, 3, res, res, device=device), \
        torch.randint(0, 10, (b,), device=device)


def _lm_sample(b, seq, vocab, device):
    return torch.randint(0, vocab, (b, seq), device=device), \
        torch.randint(0, vocab, (b, seq), device=device)


# Each job: builder + sample fn + loss kind + metadata for the heuristic baseline.
def make_jobs() -> list[dict]:
    V = 2000
    J = []
    # VGG-CNN family
    for wid, blk, res, bs in [(64, 3, 64, 256), (96, 3, 128, 64)]:
        J.append({"id": f"cnn-w{wid}-b{blk}-{res}px-bs{bs}", "family": "cnn",
                  "build": (lambda w=wid, b=blk: SimpleCNN(width=w, blocks=b)),
                  "sample": (lambda d, b=bs, r=res: _img_sample(b, r, d)),
                  "loss": "cls", "batch": bs, "seq": None})
    # ResNet family (skip connections)
    for wid, stg, res, bs in [(64, (2, 2, 2), 64, 128),
                              (64, (2, 2, 2, 2), 96, 64),
                              (128, (2, 2, 2), 64, 128)]:
        sname = "".join(map(str, stg))
        J.append({"id": f"res-w{wid}-{sname}-{res}px-bs{bs}", "family": "resnet",
                  "build": (lambda w=wid, s=stg: SmallResNet(width=w, stages=s)),
                  "sample": (lambda d, b=bs, r=res: _img_sample(b, r, d)),
                  "loss": "cls", "batch": bs, "seq": None})
    # Transformer LM family — incl. long-seq stress (probes the seq^2 attention blind spot)
    for dm, nl, sq, bs in [(256, 4, 128, 32), (384, 6, 128, 16), (256, 4, 256, 16),
                           (256, 4, 512, 32), (384, 6, 1024, 16)]:
        J.append({"id": f"lm-d{dm}-l{nl}-s{sq}-bs{bs}", "family": "transformer",
                  "build": (lambda d=dm, l=nl, s=sq: TinyLM(d_model=d, layers=l, seq=s, vocab=V)),
                  "sample": (lambda d, b=bs, s=sq, v=V: _lm_sample(b, s, v, d)),
                  "loss": "lm", "batch": bs, "seq": sq, "layers": nl, "nhead": 4})
    return J


# --------------------- measurement + generic activation ---------------------
def params_m_of(job) -> float:
    m = job["build"]()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    del m
    return n


def measure_peak_gb(job, precision: str, steps: int, device: str) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = job["build"]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))
    for _ in range(steps):
        x, y = job["sample"](device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            if job["loss"] == "lm":
                loss = F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1))
            else:
                loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return round(peak, 3)


def activation_elems_per_sample(job, device: str) -> int:
    """Architecture-agnostic: sum every leaf module's output activation elements.

    This is the deterministic stand-in for the LLM's per-layer shape list — general
    enough for residual adds, embeddings, attention output projections, etc. NOTE it
    sees module OUTPUTS only, so it cannot see a transformer's internal (B,h,seq,seq)
    attention scores — that seq^2 blind spot is supplied analytically by
    attention_elems_per_sample() and added on top of this proxy."""
    model = job["build"]().to(device).train()
    x, _ = job["sample"](device)
    total = [0]
    handles = []

    def hook(_m, _inp, out):
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if torch.is_tensor(t):
                total[0] += t.numel()

    for m in model.modules():
        if len(list(m.children())) == 0:          # leaf modules only (no double count)
            handles.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    b = job["batch"]
    del model
    torch.cuda.empty_cache()
    return int(total[0] / b)


def attention_elems_per_sample(job) -> int:
    """Analytic attention-score term the leaf-output hook structurally cannot see.

    Self-attention materialises a (B, nhead, seq, seq) score matrix per layer (the
    fp32 math backend). That tensor is an *internal* intermediate — never a module
    output — so activation_elems_per_sample() misses it; this is the §6 long-context
    blind spot. The count is closed-form from metadata (no LLM, no hook), per sample:
        nhead * seq^2  per layer  ->  layers * nhead * seq^2
    Zero for non-attention families (seq is None). Folded into the activation term so
    the single global (a, b) absorbs its retention/precision factor like any other.
    (Caveat: fp16/bf16 flash kernels may not materialise scores — see precision note.)"""
    if job.get("seq") is None:
        return 0
    return int(job["layers"] * job["nhead"] * job["seq"] ** 2)


# ---------------------- deterministic predictor + calib ---------------------
def _terms(job, params_m: float, act_ps: int, attn_ps: int,
           precision: str) -> tuple[float, float]:
    bpp = 2 if precision in ("fp16", "bf16") else 4
    param_term = 4 * params_m * 1e6 * bpp / 1e9          # weights+grad+Adam (exact)
    act_raw = job["batch"] * bpp * (act_ps + attn_ps) / 1e9   # +seq^2 attention scores
    return param_term, act_raw


def _calibrate(rows, precision):
    A, y = [], []
    for r in rows:
        pt, ar = _terms(r["job"], r["params_m"], r["act_ps"], r["attn_ps"], precision)
        A.append([ar, 1.0])
        y.append(r["truth"] - pt)
    (a, b), *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--no-attn", action="store_true",
                    help="ablate the analytic seq^2 attention term (Exp 6 behaviour). Used to "
                         "show the term is still NEEDED under fp16/bf16 — i.e. the score matrix "
                         "is materialised even under autocast, so it must NOT be backend-gated.")
    ap.add_argument("--out", default=os.path.join(HERE, "results_arch.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — needs the A100.")
    device = "cuda:0"
    print(f"device: {torch.cuda.get_device_name(0)} | precision: {args.precision} | "
          f"GLOBAL (a,b) across CNN+ResNet+Transformer, leave-one-out\n")

    jobs = make_jobs()
    rows = []
    for job in jobs:
        pm = params_m_of(job)
        act = activation_elems_per_sample(job, device)
        attn = 0 if args.no_attn else attention_elems_per_sample(job)
        truth = measure_peak_gb(job, args.precision, args.steps, device)
        rows.append({"job": job, "params_m": pm, "act_ps": act,
                     "attn_ps": attn, "truth": truth})

    det_preds, heur_preds, jb = [], [], []
    for i, r in enumerate(rows):
        a, b = _calibrate([rr for j, rr in enumerate(rows) if j != i], args.precision)
        pt, ar = _terms(r["job"], r["params_m"], r["act_ps"], r["attn_ps"], args.precision)
        det = max(0.05, pt + a * ar + b)
        meta = {"params_m": round(r["params_m"], 2),
                "precision": args.precision, "training_mode": "from_scratch",
                "batch": r["job"]["batch"], "seq": r["job"]["seq"],
                "truth": {"peak_mem_gb": r["truth"], "gpus": 1}}
        jb.append(meta)
        det_preds.append({"peak_mem_gb": round(det, 2),
                          "recommended_gpus": max(1, int(-(-det // 38)))})
        heur_preds.append(heuristic_baseline(meta))
        r["det"] = round(det, 2)

    mean_mem = round(float(np.mean([r["truth"] for r in rows])), 2)
    mean_preds = [{"peak_mem_gb": mean_mem, "recommended_gpus": 1} for _ in rows]

    print(f"{'job':24} {'family':12} {'params':>8} {'MEASURED':>9} {'DET':>7} {'heur':>7}")
    print("-" * 76)
    for r, dp, hp in zip(rows, det_preds, heur_preds):
        print(f"{r['job']['id']:24} {r['job']['family']:12} {r['params_m']:7.2f}M "
              f"{r['truth']:8.2f}G {dp['peak_mem_gb']:6.2f}G {hp['peak_mem_gb']:6.2f}G")

    # per-family MAE to see WHERE it holds or breaks
    fams = sorted({r["job"]["family"] for r in rows})
    print("\nper-family mem MAE (deterministic):")
    for fam in fams:
        idx = [i for i, r in enumerate(rows) if r["job"]["family"] == fam]
        mae = float(np.mean([abs(det_preds[i]["peak_mem_gb"] - rows[i]["truth"]) for i in idx]))
        print(f"  {fam:12}: {mae:.2f} GB  (n={len(idx)})")

    print("\n" + "=" * 76)
    print(f"DETERMINISTIC (LOOCV global) : {fmt(score(det_preds, jb))}")
    print(f"HEUR (params rule)           : {fmt(score(heur_preds, jb))}")
    print(f"MEAN (no prediction)         : {fmt(score(mean_preds, jb))}")
    print("=" * 76)
    dm, hm = score(det_preds, jb), score(heur_preds, jb)
    print(f"\nBEATS-HEURISTIC gate (mem MAE): {'PASS' if dm['mem_MAE'] < hm['mem_MAE'] else 'FAIL'} "
          f"({dm['mem_MAE']:.2f} vs {hm['mem_MAE']:.2f} GB)")

    with open(args.out, "w") as f:
        json.dump({"precision": args.precision, "device": torch.cuda.get_device_name(0),
                   "rows": [{"id": r["job"]["id"], "family": r["job"]["family"],
                             "params_m": round(r["params_m"], 2), "act_ps": r["act_ps"],
                             "attn_ps": r["attn_ps"],
                             "truth_gb": r["truth"], "deterministic_gb": r["det"]}
                            for r in rows],
                   "metrics": {"deterministic": dm, "heuristic": hm,
                               "mean": score(mean_preds, jb)}}, f, indent=2)
    print(f"\nper-job results -> {args.out}")


if __name__ == "__main__":
    main()
