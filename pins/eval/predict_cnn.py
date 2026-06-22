"""
Stage-1 prediction, *closed-loop* on real CNNs (research_plan.md gate, Month 3/5).

`predict_resources.py` scores the LLM + heuristic against **approximate community
truth** baked into benchmark.json (see its `_note`: "replace with measured
profiling for the Month-5 real validation"). This module closes that loop for a
couple of *small CNNs we can actually run*: it asks the same three predictors for
each model's peak VRAM, then **trains the model on the GPU and MEASURES the real
peak** (`torch.cuda.max_memory_allocated`). Prediction vs. ground truth, measured.

Why CNNs are a sharp test (research_plan.md:10 — "demand varies, metadata is thin"):
CNN memory is dominated by *activations* (~ batch x resolution x width), NOT by
parameter count. The heuristic only knows params x bytes, so it is structurally
blind to a small-param / high-resolution / big-batch job. We therefore include
the image **resolution** in the LLM prompt (the benchmark schema omits it) and
pick two configs that are param-light but activation-heavy — the regime where
"the LLM reasons / code decides" should matter most.

Design principle (CLAUDE.md): the LLM only emits structured numbers; this script
(deterministic code) measures truth and scores. No LLM in the hot loop.

Run (after torch is installed and an A100 is visible):
    .venv/bin/python -m pins.eval.predict_cnn
    .venv/bin/python -m pins.eval.predict_cnn --model qwen2.5:7b --steps 8
    .venv/bin/python -m pins.eval.predict_cnn --precision fp16
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

# Reuse the exact baselines + scoring the headline eval uses, so numbers are
# comparable across the two scripts.
from pins.eval.predict_resources import heuristic_baseline, parse_prediction, score, fmt

HERE = os.path.dirname(__file__)


# --------------------------- the CNNs under test ----------------------------
class SimpleCNN(nn.Module):
    """A plain VGG-ish conv classifier with tunable width/depth/resolution.

    Kept deliberately simple and param-light so that *activation* memory (a
    function of batch and resolution) dominates — the case the params-only
    heuristic cannot see.
    """

    def __init__(self, in_ch: int = 3, width: int = 64, blocks: int = 3,
                 n_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_ch
        w = width
        for _ in range(blocks):
            layers += [
                nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c = w
            w = w * 2
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# Two "jobs": both param-light, but very different activation pressure.
#   cnnA  -> moderate width, 64x64 input, big batch  (activation-heavy, tiny params)
#   cnnB  -> wider/deeper,   96x96 input, mid batch  (more params AND activations)
CNN_CONFIGS = [
    {"id": "cnnA-64px-b256", "width": 64,  "blocks": 3, "res": 64, "batch": 256},
    {"id": "cnnB-128w-96px", "width": 128, "blocks": 4, "res": 96, "batch": 128},
]


def build(cfg: dict) -> nn.Module:
    return SimpleCNN(width=cfg["width"], blocks=cfg["blocks"])


# Six varied nets (width x depth x resolution x batch) for a real trend, not a
# 2-point anecdote. All fit in 40 GB and train in a few steps on the A100.
DET_CONFIGS = [
    {"id": "w32-b3-64px-bs128",  "width": 32,  "blocks": 3, "res": 64,  "batch": 128},
    {"id": "w64-b3-64px-bs256",  "width": 64,  "blocks": 3, "res": 64,  "batch": 256},
    {"id": "w64-b4-96px-bs128",  "width": 64,  "blocks": 4, "res": 96,  "batch": 128},
    {"id": "w128-b4-96px-bs128", "width": 128, "blocks": 4, "res": 96,  "batch": 128},
    {"id": "w96-b3-128px-bs64",  "width": 96,  "blocks": 3, "res": 128, "batch": 64},
    {"id": "w64-b5-128px-bs64",  "width": 64,  "blocks": 5, "res": 128, "batch": 64},
]


def feature_map_elements(cfg: dict) -> tuple[int, int]:
    """Replay the architecture to sum per-sample output elements.

    This is the deterministic stand-in for the LLM's per-layer SHAPE trace (which
    we verified it gets right): conv outputs (2 per block, spatial kept by pad-1)
    and the pooled output (spatial halved) for each block. Returns
    (conv_elements, pool_elements) for ONE sample. Mirrors SimpleCNN exactly.
    """
    spatial, width = cfg["res"], cfg["width"]
    conv_elems = pool_elems = 0
    for _ in range(cfg["blocks"]):
        conv_elems += 2 * width * spatial * spatial   # two 3x3(pad1) convs keep H,W
        spatial //= 2                                 # 2x2 maxpool halves H,W
        pool_elems += width * spatial * spatial
        width *= 2                                    # each block doubles channels
    return conv_elems, pool_elems


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def metadata(cfg: dict, params_m: float, precision: str) -> dict:
    """Human-facing metadata the predictors are allowed to see (no truth)."""
    return {
        "id": cfg["id"],
        "framework": "pytorch",
        "model": f"Custom CNN (VGG-style, {cfg['blocks']} blocks, width {cfg['width']})",
        "params_m": round(params_m, 2),
        "batch": cfg["batch"],
        "resolution": f"{cfg['res']}x{cfg['res']}x3",   # the field the schema lacked
        # Architecture facts (known at submission) — let a reasoning pass walk the layers.
        "blocks": cfg["blocks"],
        "width": cfg["width"],
        "res_px": cfg["res"],
        "in_ch": 3,
        "n_classes": 10,
        "seq": None,
        "precision": precision,
        "training_mode": "from_scratch",
        "dataset": "synthetic 3-channel images",
    }


# ----------------------------- the LLM prompt -------------------------------
SYSTEM = (
    "You are an HPC resource estimator for deep-learning training jobs. "
    "Given a job's metadata, estimate the GPU memory it needs. "
    'Respond with ONLY a JSON object: {"peak_mem_gb": <number>, "recommended_gpus": <integer>}. '
    "peak_mem_gb is the TOTAL GPU memory in GB for the training run (weights + gradients "
    "+ optimizer states + ACTIVATIONS). For CNNs, activation memory scales with batch size "
    "and image resolution and can dominate parameter memory. "
    "recommended_gpus is how many 40 GB A100 GPUs are needed to fit it."
)


def make_prompt(job: dict) -> str:
    return (
        f"Framework: {job['framework']}\n"
        f"Model: {job['model']} ({job['params_m']}M parameters)\n"
        f"Training mode: {job['training_mode']}\n"
        f"Batch size: {job['batch']}\n"
        f"Image resolution: {job['resolution']}\n"
        f"Precision: {job['precision']}\n"
        f"Dataset: {job['dataset']}\n\n"
        "Estimate peak_mem_gb and recommended_gpus."
    )


def query_llm(client, model: str, job: dict, retries: int = 2) -> dict | None:
    for _ in range(retries):
        try:
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": make_prompt(job)}],
            )
            pred = parse_prediction(resp.message.content)
            if pred:
                return pred
        except Exception as e:
            print(f"    ! ollama error: {type(e).__name__}: {e}")
            time.sleep(1)
    return None


# --------------------- extract-then-compute HYBRID predictor ----------------
# The design-consistent option (research_plan.md:60, CLAUDE.md): the LLM never
# emits the final GB number (we showed it can't calibrate that). Instead it emits
# STRUCTURED FACTS — the things metadata-reasoning is actually good at (training
# mode -> trainable fraction, optimizer behaviour, an activation-intensity
# estimate) — and the deterministic formula below turns those facts into GB.
FACTS_SYSTEM = (
    "You are a deep-learning systems analyst. You do NOT estimate total memory. "
    "Instead, from the job metadata you extract structured FACTS that a formula "
    "will use. Respond with ONLY this JSON object:\n"
    '{"trainable_fraction": <0..1>, "bytes_per_param": <2 or 4>, '
    '"optimizer_multiplier": <number>, "activation_mb_per_sample": <number>}\n'
    "Definitions:\n"
    "- trainable_fraction: fraction of parameters that receive gradients "
    "(1.0 for from_scratch/full finetune; ~0.01-0.05 for LoRA/adapters).\n"
    "- bytes_per_param: 4 for fp32/tf32, 2 for fp16/bf16.\n"
    "- optimizer_multiplier: extra copies of each TRAINABLE param the optimizer "
    "stores (Adam keeps 2 moments -> 2; SGD-momentum -> 1; plain SGD -> 0).\n"
    "- activation_mb_per_sample: megabytes of forward activations stored for ONE "
    "input sample (must be backprop'd). For a CNN this grows with image resolution "
    "and channel width; for a transformer with sequence length. Estimate it for "
    "ONE sample, NOT the whole batch."
)


def make_facts_prompt(job: dict) -> str:
    return (
        f"Framework: {job['framework']}\n"
        f"Model: {job['model']} ({job['params_m']}M parameters)\n"
        f"Training mode: {job['training_mode']}\n"
        f"Batch size: {job['batch']}\n"
        f"Image resolution: {job['resolution']}\n"
        f"Precision: {job['precision']}\n"
        f"Dataset: {job['dataset']}\n\n"
        "Extract the structured facts."
    )


def parse_facts(text: str) -> dict | None:
    import re
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    try:
        return {
            "trainable_fraction": min(1.0, max(0.0, float(obj["trainable_fraction"]))),
            "bytes_per_param": float(obj["bytes_per_param"]),
            "optimizer_multiplier": max(0.0, float(obj["optimizer_multiplier"])),
            "activation_mb_per_sample": max(0.0, float(obj["activation_mb_per_sample"])),
        }
    except Exception:
        return None


def hybrid_compute(facts: dict, job: dict) -> dict:
    """Deterministic formula: structured FACTS -> peak GB. No LLM arithmetic."""
    P = job["params_m"] * 1e6
    bpp = facts["bytes_per_param"]
    trainable = P * facts["trainable_fraction"]
    weights = P * bpp                                   # resident model weights
    grads = trainable * bpp                             # one grad per trainable param
    optim = trainable * bpp * facts["optimizer_multiplier"]   # Adam moments etc.
    activ = job["batch"] * facts["activation_mb_per_sample"] * 1e6
    total = (weights + grads + optim + activ) * 1.10    # ~10% workspace/frag slack
    mem_gb = total / 1e9
    return {"peak_mem_gb": round(mem_gb, 2),
            "recommended_gpus": max(1, int(-(-mem_gb // 38)))}  # ceil(mem/38)


def query_hybrid(client, model: str, job: dict, retries: int = 2):
    """Returns (prediction, facts) or (None, None)."""
    for _ in range(retries):
        try:
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 150},
                messages=[{"role": "system", "content": FACTS_SYSTEM},
                          {"role": "user", "content": make_facts_prompt(job)}],
            )
            facts = parse_facts(resp.message.content)
            if facts:
                return hybrid_compute(facts, job), facts
        except Exception as e:
            print(f"    ! ollama error: {type(e).__name__}: {e}")
            time.sleep(1)
    return None, None


# --------------- reasoning HYBRID: walk the layers, then extract --------------
# The previous hybrid's only bad fact was activation_mb_per_sample (the LLM
# guessed instead of computing feature-map sizes). Here we hand the model the
# architecture and make it REASON layer-by-layer in free text first, then emit
# the facts JSON last. Not a forced-JSON call — chain-of-thought needs room.
REASONING_SYSTEM = (
    "You are a deep-learning systems analyst. Think step by step IN TEXT, then end "
    "with one JSON object. Compute activation memory by walking the network layer "
    "by layer: for every layer whose output is kept for backprop, work out the "
    "output tensor shape (channels x height x width) for ONE sample, its element "
    "count, and its size in MB (elements x bytes_per_param / 1e6). Sum these to get "
    "activation_mb_per_sample. Remember 3x3 conv with padding 1 keeps H,W; a 2x2 "
    "maxpool halves H,W; each block here doubles the channel width.\n"
    "After your reasoning, output EXACTLY ONE final line that is a JSON object:\n"
    '{"activation_mb_per_sample": <number>, "trainable_fraction": <0..1>, '
    '"bytes_per_param": <2 or 4>, "optimizer_multiplier": <number>}'
)


def make_reasoning_prompt(job: dict) -> str:
    bpp = 2 if job["precision"] in ("fp16", "bf16") else 4
    return (
        "Estimate the per-sample activation memory of this CNN training job.\n\n"
        f"Architecture: VGG-style CNN, {job['blocks']} blocks, base width {job['width']}.\n"
        f"Input: {job['in_ch']} x {job['res_px']} x {job['res_px']} per sample.\n"
        "Each block = [Conv3x3(pad1) -> BN -> ReLU -> Conv3x3(pad1) -> BN -> ReLU "
        "-> MaxPool2x2]. Block k uses width = base_width * 2^k (k=0,1,...). "
        "After the blocks: AdaptiveAvgPool to 1x1 -> Linear to "
        f"{job['n_classes']} classes.\n"
        f"Precision: {job['precision']} ({bpp} bytes per element). "
        f"Total parameters: {job['params_m']}M. Training mode: {job['training_mode']} "
        "with Adam.\n\n"
        "Walk every conv/BN/ReLU output layer, compute its CxHxW and MB for one "
        "sample, sum them, and report the facts JSON at the end."
    )


def _extract_last_json(text: str) -> dict | None:
    """Find the LAST balanced {...} block that parses and has the facts keys.

    Reasoning output is long free text with the JSON at the end (and possibly
    stray braces mid-reasoning), so scan from the end for the last valid object.
    """
    starts = [i for i, c in enumerate(text) if c == "{"]
    for s in reversed(starts):
        depth = 0
        for e in range(s, len(text)):
            if text[e] == "{":
                depth += 1
            elif text[e] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[s:e + 1])
                    except Exception:
                        break
                    if "activation_mb_per_sample" in obj:
                        return obj
                    break
    return None


def query_hybrid_reasoning(client, model: str, job: dict, retries: int = 2):
    """Reasoning pass -> (prediction, facts, raw_reasoning) or (None, None, text)."""
    last_text = ""
    for _ in range(retries):
        try:
            resp = client.chat(
                model=model,                      # NO format=json: let it think
                options={"temperature": 0, "num_predict": 3000},
                messages=[{"role": "system", "content": REASONING_SYSTEM},
                          {"role": "user", "content": make_reasoning_prompt(job)}],
            )
            last_text = resp.message.content
            obj = _extract_last_json(last_text)
            if obj is None:
                continue
            facts = parse_facts(json.dumps(obj))   # reuse clamping/validation
            if facts:
                return hybrid_compute(facts, job), facts, last_text
        except Exception as e:
            print(f"    ! ollama error: {type(e).__name__}: {e}")
            time.sleep(1)
    return None, None, last_text


# --------------------------- the ground-truth probe -------------------------
def measure_peak_gb(cfg: dict, precision: str, steps: int, device: str) -> float:
    """Actually train the CNN for a few steps and return measured peak VRAM (GB).

    This IS the ground truth — the thing benchmark.json only approximates.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = build(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    b, res = cfg["batch"], cfg["res"]
    for _ in range(steps):
        x = torch.randn(b, 3, res, res, device=device)
        y = torch.randint(0, 10, (b,), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return round(peak, 3)


def deterministic_predict_gb(cfg: dict, params_m: float, precision: str,
                             a: float, b: float) -> float:
    """LLM-shapes -> code sums -> calibrated GB. No LLM number anywhere.

    peak ~= param_term (exact) + a * activation_raw + b
      param_term : weights + grad + Adam(2 moments) = 4 * P * bytes  (exact)
      activation_raw : batch * bytes * (conv + pool feature-map elements)
      a : activation retention factor (BN/ReLU buffers, autograd save) — calibrated
      b : fixed framework/cudnn-workspace overhead (GB) — calibrated
    """
    bpp = 2 if precision in ("fp16", "bf16") else 4
    P = params_m * 1e6
    param_term = 4 * P * bpp / 1e9
    conv_e, pool_e = feature_map_elements(cfg)
    act_raw = cfg["batch"] * bpp * (conv_e + pool_e) / 1e9
    return param_term + a * act_raw + b


def _calibrate(rows: list[dict], precision: str) -> tuple[float, float]:
    """Least-squares fit of (a, b) on (peak - param_term) ~ a*act_raw + b."""
    bpp = 2 if precision in ("fp16", "bf16") else 4
    A, y = [], []
    for r in rows:
        P = r["params_m"] * 1e6
        param_term = 4 * P * bpp / 1e9
        conv_e, pool_e = feature_map_elements(r["cfg"])
        act_raw = r["cfg"]["batch"] * bpp * (conv_e + pool_e) / 1e9
        A.append([act_raw, 1.0])
        y.append(r["truth"] - param_term)
    (a, b), *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return float(a), float(b)


def run_deterministic_eval(args) -> None:
    """Final predictor: per-layer shapes -> deterministic sum + LOOCV-calibrated
    overhead. Evaluated leave-one-out so it never sees the config it predicts."""
    device = "cuda:0"
    print(f"device: {torch.cuda.get_device_name(0)} | precision: {args.precision} | "
          f"deterministic (LLM-shapes -> code sum), leave-one-out calibrated\n")

    # 1) measure ground truth for all configs
    rows = []
    for cfg in DET_CONFIGS:
        m = build(cfg); params_m = count_params_m(m); del m
        truth = measure_peak_gb(cfg, args.precision, args.steps, device)
        rows.append({"cfg": cfg, "params_m": params_m, "truth": truth})

    # 2) leave-one-out: calibrate (a,b) on the other 5, predict the held-out one
    det_preds, heur_preds, jobs = [], [], []
    for i, r in enumerate(rows):
        a, b = _calibrate([rr for j, rr in enumerate(rows) if j != i], args.precision)
        det_gb = deterministic_predict_gb(r["cfg"], r["params_m"], args.precision, a, b)
        det_gb = max(0.05, det_gb)
        job = metadata(r["cfg"], r["params_m"], args.precision)
        job["truth"] = {"peak_mem_gb": r["truth"], "gpus": 1}
        jobs.append(job)
        det_preds.append({"peak_mem_gb": round(det_gb, 2),
                          "recommended_gpus": max(1, int(-(-det_gb // 38)))})
        heur_preds.append(heuristic_baseline(job))
        r["det"], r["ab"] = round(det_gb, 2), (round(a, 3), round(b, 3))

    mean_mem = round(float(np.mean([r["truth"] for r in rows])), 2)
    mean_preds = [{"peak_mem_gb": mean_mem, "recommended_gpus": 1} for _ in rows]

    print(f"{'config':22} {'params':>8} {'MEASURED':>9} {'DETERM':>8} {'heur':>8} {'mean':>7}")
    print("-" * 72)
    for r, dp, hp in zip(rows, det_preds, heur_preds):
        print(f"{r['cfg']['id']:22} {r['params_m']:7.2f}M {r['truth']:8.2f}G "
              f"{dp['peak_mem_gb']:7.2f}G {hp['peak_mem_gb']:7.2f}G {mean_mem:6.1f}G")

    print("\n" + "=" * 72)
    print(f"DETERMINISTIC (LOOCV) : {fmt(score(det_preds, jobs))}")
    print(f"HEUR (params rule)    : {fmt(score(heur_preds, jobs))}")
    print(f"MEAN (no prediction)  : {fmt(score(mean_preds, jobs))}")
    print("=" * 72)
    det_m, heur_m = score(det_preds, jobs), score(heur_preds, jobs)
    verdict = "PASS" if det_m["mem_MAE"] < heur_m["mem_MAE"] else "FAIL"
    print(f"\nBEATS-HEURISTIC gate (mem MAE): {verdict} "
          f"({det_m['mem_MAE']:.2f} vs {heur_m['mem_MAE']:.2f} GB)")

    with open(args.out, "w") as f:
        json.dump({"precision": args.precision, "device": torch.cuda.get_device_name(0),
                   "rows": [{"id": r["cfg"]["id"], "params_m": round(r["params_m"], 2),
                             "truth_gb": r["truth"], "deterministic_gb": r["det"],
                             "calib_ab": r["ab"]} for r in rows],
                   "metrics": {"deterministic": det_m, "heuristic": heur_m,
                               "mean": score(mean_preds, jobs)}}, f, indent=2)
    print(f"\nper-job results -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b", help="ollama model for the LLM predictor")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--steps", type=int, default=6, help="training steps per measurement")
    ap.add_argument("--reasoning", action="store_true",
                    help="hybrid extracts activation by layer-by-layer reasoning (slower)")
    ap.add_argument("--show-reasoning", action="store_true",
                    help="print the model's full layer-by-layer reasoning trace")
    ap.add_argument("--deterministic", action="store_true",
                    help="final predictor: LLM-shapes -> code sum + LOOCV-calibrated overhead "
                         "(no LLM number); evaluated on 6 CNNs, leave-one-out")
    ap.add_argument("--out", default=os.path.join(HERE, "results_cnn.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this probe needs the A100.")

    if args.deterministic:
        run_deterministic_eval(args)
        return

    device = "cuda:0"
    print(f"device: {torch.cuda.get_device_name(0)} | precision: {args.precision} | "
          f"LLM: {args.model}\n")

    import ollama
    client = ollama.Client(host=args.host)

    # 1) measure ground truth, 2) collect metadata, 3) ask predictors
    jobs, llm_preds, heur_preds, hyb_preds, hyb_facts = [], [], [], [], []
    print(f"{'job':18} {'params':>7} {'MEASURED':>9} {'LLM':>8} {'heur':>8} {'hybrid':>8}")
    print("-" * 66)
    for cfg in CNN_CONFIGS:
        m = build(cfg)
        params_m = count_params_m(m)
        del m
        truth_gb = measure_peak_gb(cfg, args.precision, args.steps, device)

        job = metadata(cfg, params_m, args.precision)
        job["truth"] = {"peak_mem_gb": round(truth_gb, 2), "gpus": 1}
        jobs.append(job)

        lp = query_llm(client, args.model, job) or {"peak_mem_gb": float("nan"),
                                                    "recommended_gpus": 1}
        hp = heuristic_baseline(job)
        if args.reasoning:
            yp, facts, trace = query_hybrid_reasoning(client, args.model, job)
            if args.show_reasoning:
                print(f"\n----- reasoning trace [{cfg['id']}] -----\n{trace}\n"
                      f"----- end trace -----\n")
        else:
            yp, facts = query_hybrid(client, args.model, job)
        yp = yp or {"peak_mem_gb": float("nan"), "recommended_gpus": 1}
        llm_preds.append(lp)
        heur_preds.append(hp)
        hyb_preds.append(yp)
        hyb_facts.append(facts)
        print(f"{cfg['id']:18} {params_m:6.2f}M {truth_gb:8.2f}G "
              f"{lp['peak_mem_gb']:7.2f}G {hp['peak_mem_gb']:7.2f}G {yp['peak_mem_gb']:7.2f}G")

    # mean baseline over THIS measured set (no-information predictor)
    mean_mem = round(float(np.mean([j["truth"]["peak_mem_gb"] for j in jobs])), 2)
    mean_preds = [{"peak_mem_gb": mean_mem, "recommended_gpus": 1} for _ in jobs]

    print("\nLLM-extracted facts (what the hybrid computed from):")
    for j, f in zip(jobs, hyb_facts):
        print(f"  {j['id']:18} {f}")

    print("\n" + "=" * 66)
    print(f"LLM    (raw number, {args.model:10}): {fmt(score(llm_preds, jobs))}")
    print(f"HYBRID (facts->formula)          : {fmt(score(hyb_preds, jobs))}")
    print(f"HEUR   (params rule)             : {fmt(score(heur_preds, jobs))}")
    print(f"MEAN   (no prediction)           : {fmt(score(mean_preds, jobs))}")
    print("=" * 66)

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "precision": args.precision,
                   "device": torch.cuda.get_device_name(0),
                   "rows": [{"id": j["id"], "params_m": j["params_m"],
                             "resolution": j["resolution"], "batch": j["batch"],
                             "truth": j["truth"], "llm": lp, "heuristic": hp,
                             "hybrid": yp, "hybrid_facts": fa}
                            for j, lp, hp, yp, fa in
                            zip(jobs, llm_preds, heur_preds, hyb_preds, hyb_facts)],
                   "metrics": {"llm": score(llm_preds, jobs),
                               "hybrid": score(hyb_preds, jobs),
                               "heuristic": score(heur_preds, jobs),
                               "mean": score(mean_preds, jobs)}}, f, indent=2)
    print(f"\nper-job results -> {args.out}")


if __name__ == "__main__":
    main()
