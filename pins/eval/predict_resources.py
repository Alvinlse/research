"""
Stage-1 cold-start evaluation: how well can qwen2.5:3b predict a job's GPU needs
from metadata alone? (research_plan.md gate, Month 3.)

The LLM sees ONLY human-facing metadata (framework, model, params, batch, seq,
precision, training mode, dataset) — never the ground-truth numbers — and must
emit {peak_mem_gb, recommended_gpus}. We score it against two baselines it has to
beat:
  * MEAN     : predict the benchmark's average for every job (the "no prediction"
               baseline — this IS the gate in research_plan.md:127).
  * HEURISTIC: a simple non-LLM rule from params x bytes/param (knows nothing about
               LoRA vs full finetune or activation pressure).

Run:  .venv/bin/python -m pins.eval.predict_resources
      .venv/bin/python -m pins.eval.predict_resources --model qwen2.5:3b --repeats 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

import numpy as np

HERE = os.path.dirname(__file__)
BENCH_PATH = os.path.join(HERE, "benchmark.json")

SYSTEM = (
    "You are an HPC resource estimator for deep-learning training jobs. "
    "Given a job's metadata, estimate the GPU resources it needs. "
    "Respond with ONLY a JSON object of the form "
    '{"peak_mem_gb": <number>, "recommended_gpus": <integer>}. '
    "peak_mem_gb is the TOTAL GPU memory in GB the training run needs across all GPUs "
    "(model weights + gradients + optimizer states + activations). "
    "recommended_gpus is how many 40 GB A100 GPUs are needed to fit it. "
    "Account for training mode: LoRA/adapter training uses far less memory than full "
    "fine-tuning of the same model."
)


def make_prompt(job: dict) -> str:
    seq = job.get("seq")
    seq_line = f"Sequence length: {seq}\n" if seq else ""
    return (
        f"Framework: {job['framework']}\n"
        f"Model: {job['model']} ({job['params_m']}M parameters)\n"
        f"Training mode: {job['training_mode']}\n"
        f"Batch size: {job['batch']}\n"
        f"{seq_line}"
        f"Precision: {job['precision']}\n"
        f"Dataset: {job['dataset']}\n\n"
        "Estimate peak_mem_gb and recommended_gpus."
    )


def parse_prediction(text: str) -> dict | None:
    """Robustly pull {peak_mem_gb, recommended_gpus} out of the model output."""
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
        mem = float(obj["peak_mem_gb"])
        gpus = int(round(float(obj["recommended_gpus"])))
        return {"peak_mem_gb": mem, "recommended_gpus": max(1, gpus)}
    except Exception:
        return None


def query_llm(client, model: str, job: dict, retries: int = 2) -> dict | None:
    for _ in range(retries):
        try:
            resp = client.chat(
                model=model,
                format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": make_prompt(job)},
                ],
            )
            pred = parse_prediction(resp.message.content)
            if pred:
                return pred
        except Exception as e:
            print(f"    ! ollama error: {type(e).__name__}: {e}")
            time.sleep(1)
    return None


# ----------------------------- baselines ------------------------------------
BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2}


def heuristic_baseline(job: dict) -> dict:
    """Non-LLM rule: total mem ~ params x bytes/param x training-overhead factor."""
    p_b = job["params_m"] / 1000.0          # billions of params
    b = BYTES.get(job["precision"], 2)
    # Full training (Adam mixed precision) ~ 16-18 bytes/param; LoRA freezes the
    # base model so only a tiny fraction trains. The heuristic only "knows" params.
    overhead = 18                            # bytes/param for full Adam training
    mem = p_b * overhead
    mem = max(mem, p_b * b + 2)              # at least weights + a little
    mem = mem + 2                            # crude activation slack
    gpus = max(1, math.ceil(mem / 38))
    return {"peak_mem_gb": round(mem, 1), "recommended_gpus": gpus}


def mean_baseline(jobs: list[dict]) -> dict:
    """No-information baseline: predict the dataset average for every job."""
    mem = float(np.mean([j["truth"]["peak_mem_gb"] for j in jobs]))
    gpus = int(round(np.mean([j["truth"]["gpus"] for j in jobs])))
    return {"peak_mem_gb": round(mem, 1), "recommended_gpus": max(1, gpus)}


# ------------------------------- metrics ------------------------------------
def _avg_rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    rp, rt = _avg_rank(pred), _avg_rank(truth)
    if rp.std() == 0 or rt.std() == 0:
        return float("nan")
    return float(np.corrcoef(rp, rt)[0, 1])


def score(preds: list[dict], jobs: list[dict]) -> dict:
    pm = np.array([p["peak_mem_gb"] for p in preds], float)
    tm = np.array([j["truth"]["peak_mem_gb"] for j in jobs], float)
    pg = np.array([p["recommended_gpus"] for p in preds], float)
    tg = np.array([j["truth"]["gpus"] for j in jobs], float)
    ratio = np.maximum(pm, 1e-6) / np.maximum(tm, 1e-6)
    return {
        "mem_MAE": float(np.mean(np.abs(pm - tm))),
        "mem_MAPE": float(np.mean(np.abs(pm - tm) / tm) * 100),
        "mem_within_1.5x": float(np.mean((ratio >= 1 / 1.5) & (ratio <= 1.5)) * 100),
        "mem_spearman": spearman(pm, tm),
        "gpu_exact_acc": float(np.mean(pg == tg) * 100),
        "gpu_within_1": float(np.mean(np.abs(pg - tg) <= 1) * 100),
    }


def fmt(m: dict) -> str:
    return (f"mem MAE={m['mem_MAE']:5.1f}GB  MAPE={m['mem_MAPE']:5.1f}%  "
            f"within1.5x={m['mem_within_1.5x']:5.1f}%  rho={m['mem_spearman']:.2f}  | "
            f"gpu exact={m['gpu_exact_acc']:5.1f}%  within1={m['gpu_within_1']:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--repeats", type=int, default=1, help="LLM queries per job (averaged)")
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()

    import ollama
    client = ollama.Client(host=args.host)

    with open(BENCH_PATH) as f:
        jobs = json.load(f)["jobs"]
    print(f"benchmark: {len(jobs)} jobs | model: {args.model}\n")

    mean_pred = mean_baseline(jobs)
    llm_preds, heur_preds, mean_preds, rows = [], [], [], []
    fails = 0
    t0 = time.time()

    print(f"{'job':16} {'truth(GB,gpu)':>14} {'LLM':>13} {'heur':>13} {'mean':>11}")
    print("-" * 72)
    for job in jobs:
        # average repeated LLM queries (temp=0 so usually identical)
        mems, gps = [], []
        for _ in range(args.repeats):
            pred = query_llm(client, args.model, job)
            if pred:
                mems.append(pred["peak_mem_gb"])
                gps.append(pred["recommended_gpus"])
        if mems:
            lp = {"peak_mem_gb": round(float(np.mean(mems)), 1),
                  "recommended_gpus": int(round(float(np.mean(gps))))}
        else:
            fails += 1
            lp = {"peak_mem_gb": mean_pred["peak_mem_gb"],   # fall back to mean on failure
                  "recommended_gpus": mean_pred["recommended_gpus"]}
        hp = heuristic_baseline(job)
        llm_preds.append(lp); heur_preds.append(hp); mean_preds.append(mean_pred)
        t = job["truth"]
        rows.append({"id": job["id"], "truth": t, "llm": lp, "heuristic": hp})
        truth_s = f"{t['peak_mem_gb']},{t['gpus']}"
        llm_s = f"{lp['peak_mem_gb']},{lp['recommended_gpus']}"
        heur_s = f"{hp['peak_mem_gb']},{hp['recommended_gpus']}"
        mean_s = f"{mean_pred['peak_mem_gb']},{mean_pred['recommended_gpus']}"
        print(f"{job['id']:16} {truth_s:>14} {llm_s:>13} {heur_s:>13} {mean_s:>11}")

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"LLM  ({args.model:12}) : {fmt(score(llm_preds, jobs))}")
    print(f"HEUR (params rule)      : {fmt(score(heur_preds, jobs))}")
    print(f"MEAN (no prediction)    : {fmt(score(mean_preds, jobs))}")
    print("=" * 72)
    if fails:
        print(f"note: {fails}/{len(jobs)} LLM predictions failed to parse (fell back to MEAN)")
    print(f"elapsed {dt:.1f}s ({dt/max(1,len(jobs)*args.repeats):.2f}s/query)")

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "metrics": {
            "llm": score(llm_preds, jobs),
            "heuristic": score(heur_preds, jobs),
            "mean": score(mean_preds, jobs),
        }, "rows": rows, "failures": fails}, f, indent=2)
    print(f"per-job results -> {args.out}")

    # The gate (research_plan.md:127): LLM must beat the no-prediction baseline.
    llm_m, mean_m = score(llm_preds, jobs), score(mean_preds, jobs)
    verdict = "PASS" if llm_m["mem_MAE"] < mean_m["mem_MAE"] else "FAIL"
    print(f"\nGATE (LLM beats no-prediction on mem MAE): {verdict} "
          f"({llm_m['mem_MAE']:.1f} vs {mean_m['mem_MAE']:.1f} GB)")


if __name__ == "__main__":
    main()
