"""
Stage-1 prediction REDONE on REAL architectures (Exp 1-7 used synthetic CNNs only).

The model list is externally justified: it is exactly the DNN families observed in the
MIT Supercloud labelled trace (data/labelled_jobids_full.csv) — resnet50/101/152,
vgg11/16/19, inception3, bert-base-uncased, distilbert-base-uncased, and the U-Net
family U{depth}-{base_filters}. GNNs (schnet/dimenet/pna) are skipped (would need PyG).
Ground truth is measured on our A100 so batch/resolution/precision are controlled.

Arms (same comparison as Exp 1 vs Exp 4/7):
  - raw LLM number (Exp-1 arm) — NOTE these are FAMOUS model names, so the LLM may
    calibrate better than on anonymous synthetic CNNs; that is part of the test.
  - deterministic LOOCV (Exp-4/7 arm): param term exact, hook-counted activations
    + analytic seq^2 attention term, global (a,b) leave-one-out.
  - params heuristic + mean baselines.

Transformers use attn_implementation="eager" so the (B,h,seq,seq) score matrix is
materialised, matching the Exp-7 analytic term (flash/sdpa kernels would not).

Run:  cd Research && .venv-forecast/bin/python -m pins.eval.predict_real
      (main .venv torch is broken; .venv-forecast has torch 2.6.0+cu124 + torchvision
       + transformers. --skip-llm to run without Ollama.)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pins.eval.predict_arch import (_calibrate, _img_sample, _terms,
                                    activation_elems_per_sample,
                                    attention_elems_per_sample, measure_peak_gb,
                                    params_m_of)
from pins.eval.predict_cnn import SYSTEM
from pins.eval.predict_resources import (fmt, heuristic_baseline,
                                         parse_prediction, score)

HERE = os.path.dirname(__file__)


# ------------------------------ U-Net (Supercloud U{depth}-{filters}) -------
class UNet(nn.Module):
    """Plain conv U-Net, channels doubling per level — the Supercloud U3/U4/U5 family."""

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout),
                             nn.ReLU(inplace=True))

    def __init__(self, depth: int = 4, base: int = 64, in_ch: int = 3, n_classes: int = 2):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([self._block(in_ch if i == 0 else chs[i - 1], chs[i])
                                  for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.mid = self._block(chs[depth - 1], chs[depth])
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2)
                                 for i in reversed(range(depth))])
        self.dec = nn.ModuleList([self._block(chs[i] * 2, chs[i])
                                  for i in reversed(range(depth))])
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for u, d, s in zip(self.up, self.dec, reversed(skips)):
            x = d(torch.cat([u(x), s], dim=1))
        return self.head(x)


# ------------------------------ HF wrappers ---------------------------------
class _HFWrap(nn.Module):
    """Return plain logits so predict_arch's measure/hook helpers work unchanged."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, input_ids):
        return self.m(input_ids=input_ids).logits


def _build_bert():
    from transformers import AutoModelForSequenceClassification, BertConfig
    return _HFWrap(AutoModelForSequenceClassification.from_config(
        BertConfig(), attn_implementation="eager"))


def _build_distilbert():
    from transformers import AutoModelForSequenceClassification, DistilBertConfig
    return _HFWrap(AutoModelForSequenceClassification.from_config(
        DistilBertConfig(), attn_implementation="eager"))


def _seg_sample(b, res, device):
    return torch.randn(b, 3, res, res, device=device), \
        torch.randint(0, 2, (b, res, res), device=device)


def _tok_sample(b, seq, vocab, device):
    return torch.randint(0, vocab, (b, seq), device=device), \
        torch.randint(0, 2, (b,), device=device)


# -------------------------------- the jobs ----------------------------------
def make_jobs() -> list[dict]:
    from torchvision import models as tvm
    J = []

    def img(jid, fam, name, build, res, bs):
        J.append({"id": jid, "family": fam, "build": build,
                  "sample": (lambda d, b=bs, r=res: _img_sample(b, r, d)),
                  "loss": "cls", "batch": bs, "seq": None, "res": res,
                  "prompt_name": name})

    img("resnet50-224-bs64", "resnet", "ResNet-50 (torchvision)",
        lambda: tvm.resnet50(weights=None), 224, 64)
    img("resnet101-224-bs32", "resnet", "ResNet-101 (torchvision)",
        lambda: tvm.resnet101(weights=None), 224, 32)
    img("resnet152-224-bs32", "resnet", "ResNet-152 (torchvision)",
        lambda: tvm.resnet152(weights=None), 224, 32)
    img("vgg11-224-bs64", "vgg", "VGG-11 (torchvision)",
        lambda: tvm.vgg11(weights=None), 224, 64)
    img("vgg16-224-bs64", "vgg", "VGG-16 (torchvision)",
        lambda: tvm.vgg16(weights=None), 224, 64)
    img("vgg19-224-bs32", "vgg", "VGG-19 (torchvision)",
        lambda: tvm.vgg19(weights=None), 224, 32)
    img("inception3-299-bs64", "inception", "Inception-v3 (torchvision)",
        lambda: tvm.inception_v3(weights=None, aux_logits=False, init_weights=True),
        299, 64)

    # U-Nets: Supercloud labels U{depth}-{base_filters}
    for depth, base, res, bs in [(3, 32, 256, 16), (4, 64, 256, 8), (5, 128, 256, 4)]:
        J.append({"id": f"U{depth}-{base}-{res}-bs{bs}", "family": "unet",
                  "build": (lambda dp=depth, ba=base: UNet(depth=dp, base=ba)),
                  "sample": (lambda d, b=bs, r=res: _seg_sample(b, r, d)),
                  "loss": "cls", "batch": bs, "seq": None, "res": res,
                  "prompt_name": f"U-Net (depth {depth}, base filters {base}, "
                                 f"2-class segmentation)"})

    # Transformers (eager attention -> scores materialised; analytic term applies)
    for name, build, layers, seq, bs in [
            ("bert-base-uncased (HuggingFace)", _build_bert, 12, 128, 32),
            ("bert-base-uncased (HuggingFace)", _build_bert, 12, 512, 16),
            ("distilbert-base-uncased (HuggingFace)", _build_distilbert, 6, 128, 64)]:
        short = name.split("-")[0] if "distil" not in name else "distilbert"
        J.append({"id": f"{short}-s{seq}-bs{bs}", "family": "transformer",
                  "build": build,
                  "sample": (lambda d, b=bs, s=seq: _tok_sample(b, s, 30522, d)),
                  "loss": "cls", "batch": bs, "seq": seq, "res": None,
                  "layers": layers, "nhead": 12, "prompt_name": name})
    return J


# ------------------------------ raw-LLM arm ----------------------------------
def make_prompt(job: dict, params_m: float, precision: str) -> str:
    shape = (f"Image resolution: {job['res']}x{job['res']}x3\n" if job["res"]
             else f"Sequence length: {job['seq']}\n")
    return (
        f"Framework: pytorch\n"
        f"Model: {job['prompt_name']} ({round(params_m, 1)}M parameters)\n"
        f"Training mode: from_scratch\n"
        f"Batch size: {job['batch']}\n"
        f"{shape}"
        f"Precision: {precision}\n"
        f"Dataset: synthetic\n\n"
        "Estimate peak_mem_gb and recommended_gpus."
    )


def query_llm(client, model: str, prompt: str, retries: int = 2) -> dict | None:
    for _ in range(retries):
        try:
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
            )
            pred = parse_prediction(resp.message.content)
            if pred:
                return pred
        except Exception as e:
            print(f"    ! ollama error: {type(e).__name__}: {e}")
            time.sleep(1)
    return None


# ---------------------------------- main -------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--models", default="qwen2.5:3b,qwen2.5:7b,qwen2.5:14b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default=os.path.join(HERE, "results_real.json"))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — needs the A100.")
    device = "cuda:0"
    print(f"device: {torch.cuda.get_device_name(0)} | precision: {args.precision} | "
          f"REAL architectures (Supercloud labelled families), global LOOCV (a,b)\n")

    jobs = make_jobs()
    rows = []
    for job in jobs:
        pm = params_m_of(job)
        act = activation_elems_per_sample(job, device)
        attn = attention_elems_per_sample(job)
        truth = measure_peak_gb(job, args.precision, args.steps, device)
        print(f"  measured {job['id']:24} params {pm:7.1f}M  truth {truth:6.2f} GB")
        rows.append({"job": job, "params_m": pm, "act_ps": act,
                     "attn_ps": attn, "truth": truth})

    det_preds, heur_preds, jb = [], [], []
    for i, r in enumerate(rows):
        a, b = _calibrate([rr for j, rr in enumerate(rows) if j != i], args.precision)
        pt, ar = _terms(r["job"], r["params_m"], r["act_ps"], r["attn_ps"], args.precision)
        det = max(0.05, pt + a * ar + b)
        meta = {"params_m": round(r["params_m"], 2), "precision": args.precision,
                "truth": {"peak_mem_gb": r["truth"], "gpus": 1}}
        jb.append(meta)
        det_preds.append({"peak_mem_gb": round(det, 2),
                          "recommended_gpus": max(1, int(-(-det // 38)))})
        heur_preds.append(heuristic_baseline(meta))
        r["det"] = round(det, 2)

    mean_mem = round(float(np.mean([r["truth"] for r in rows])), 2)
    mean_preds = [{"peak_mem_gb": mean_mem, "recommended_gpus": 1} for _ in rows]

    llm_preds: dict[str, list] = {}
    if not args.skip_llm:
        import ollama
        client = ollama.Client(host=args.host)
        try:
            avail = {m.model for m in client.list().models}
        except Exception as e:
            print(f"! ollama unreachable ({e}); skipping the LLM arm")
            avail = set()
        for model in args.models.split(","):
            if model not in avail:
                print(f"! {model} not available in ollama; skipped")
                continue
            preds = []
            for r in rows:
                p = query_llm(client, model,
                              make_prompt(r["job"], r["params_m"], args.precision))
                preds.append(p or {"peak_mem_gb": mean_mem, "recommended_gpus": 1})
                r.setdefault("llm", {})[model] = p["peak_mem_gb"] if p else None
            llm_preds[model] = preds

    print(f"\n{'job':24} {'family':12} {'params':>8} {'MEASURED':>9} {'DET':>7} {'heur':>7}"
          + "".join(f" {m.split(':')[1]:>7}" for m in llm_preds))
    print("-" * (76 + 8 * len(llm_preds)))
    for i, (r, dp, hp) in enumerate(zip(rows, det_preds, heur_preds)):
        llm_cols = "".join(f" {llm_preds[m][i]['peak_mem_gb']:6.1f}G" for m in llm_preds)
        print(f"{r['job']['id']:24} {r['job']['family']:12} {r['params_m']:7.1f}M "
              f"{r['truth']:8.2f}G {dp['peak_mem_gb']:6.2f}G {hp['peak_mem_gb']:6.2f}G"
              + llm_cols)

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
    for m, preds in llm_preds.items():
        print(f"RAW LLM {m:20} : {fmt(score(preds, jb))}")
    print("=" * 76)
    dm, hm = score(det_preds, jb), score(heur_preds, jb)
    print(f"\nBEATS-HEURISTIC gate (mem MAE): {'PASS' if dm['mem_MAE'] < hm['mem_MAE'] else 'FAIL'} "
          f"({dm['mem_MAE']:.2f} vs {hm['mem_MAE']:.2f} GB)")

    with open(args.out, "w") as f:
        json.dump({"precision": args.precision, "device": torch.cuda.get_device_name(0),
                   "rows": [{"id": r["job"]["id"], "family": r["job"]["family"],
                             "params_m": round(r["params_m"], 2), "act_ps": r["act_ps"],
                             "attn_ps": r["attn_ps"], "truth_gb": r["truth"],
                             "deterministic_gb": r["det"], "llm_gb": r.get("llm", {})}
                            for r in rows],
                   "metrics": {"deterministic": dm, "heuristic": hm,
                               "mean": score(mean_preds, jb),
                               **{f"llm:{m}": score(p, jb) for m, p in llm_preds.items()}}},
                  f, indent=2)
    print(f"\nper-job results -> {args.out}")


if __name__ == "__main__":
    main()
