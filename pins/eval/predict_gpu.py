"""
Stage-1 (GPU track) — does job metadata predict a task's REQUESTED GPU (`plan_gpu`)?

Context. The DAG track (predict_dag.py) forecasts requested MEMORY on Alibaba v2018 and
found the task's own submission metadata dominates topology. But v2018 has no GPU field at
all. The Alibaba cluster-trace-gpu-v2020 (PAI MLaaS) is the trace that carries requested
GPU: `plan_gpu` is FRACTIONAL (25 = quarter of a GPU, 100 = one, 800 = eight), so unlike a
1/2/4/8 count it has real spread (median 50, mean 68, up to 800) — a genuine regression
target. This trace has NO precedence DAG (jobs are flat PS/worker gangs), so this is a
DIFFERENT track from v2018: "predict requested GPU from job/task metadata", not from topology.

TARGET = plan_gpu (per-task requested GPU %), tasks with plan_gpu>0 only (17.8% are NaN =
CPU-only; dropped — the question is "how much GPU", not "any GPU"). REQUESTED, not measured
(actual util lives in pai_sensor_table, deferred). Same honest framing as the DAG track:
forecast a task's GPU demand from its request context.

THE EXPERIMENT (beat-the-baseline gate, conventions lifted from predict_dag.py). All emit
P10/P50/P90 on log1p(plan_gpu):
  * GLOBAL   — global plan_gpu quantiles (no-information floor; must be beaten).
  * GBT-NUM  — GBT quantile regressor on the raw co-requested NUMBERS a user submits
               alongside the GPU ask: [inst_num, plan_cpu, plan_mem]. The strong baseline.
  * GBT-FULL — SAME model + SEMANTIC tags: [gpu_type, task_name(role/framework), workload].
The gate is GBT-FULL vs GBT-NUM: do the semantic/categorical tags add signal over the raw
co-requested CPU/mem? (Analog of gbt-dag vs gbt-nodag.) The workload tag is joined
task.job_name -> pai_job_table.inst_id -> pai_group_tag_table.workload (sparse; missing =
"none"). gpu_type is part of the request bundle (user picks type + amount) — a mild-leakage
feature, isolated in the FULL arm on purpose.

Model = sklearn HistGradientBoostingRegressor(loss="quantile"), one fit per quantile on
log1p(plan_gpu) (skewed), predictions sorted monotone. Split BY JOB (siblings never straddle
train/test). Deterministic, no LLM.

Run:  .venv/bin/python -m pins.eval.predict_gpu               # all plan_gpu>0 tasks
      .venv/bin/python -m pins.eval.predict_gpu --no-cpu-mem  # drop co-request; tags+inst only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "alibaba-gpu-v2020")
TASK = os.path.join(DATA_DIR, "pai_task_table.csv")
JOB = os.path.join(DATA_DIR, "pai_job_table.csv")
GTAG = os.path.join(DATA_DIR, "pai_group_tag_table.csv")
OUT = os.path.join(HERE, "results_gpu.json")

QUANTILES = (0.10, 0.50, 0.90)
NUM_FEATS = ["inst_num", "plan_cpu", "plan_mem"]
TAG_FEATS = ["gpu_type", "task_name", "workload"]
CATEGORICAL = {"gpu_type", "task_name", "workload"}

TASK_COLS = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
GTAG_COLS = ["inst_id", "user", "gpu_type_spec", "group", "workload"]


# --------------------------- data + features --------------------------------
def build_features() -> pd.DataFrame:
    """Load tasks (plan_gpu>0), join the sparse workload tag, encode categoricals."""
    t = pd.read_csv(TASK, header=None, names=TASK_COLS,
                    usecols=["job_name", "task_name", "inst_num", "plan_cpu", "plan_mem",
                             "plan_gpu", "gpu_type"])
    t = t[(t.plan_gpu.notna()) & (t.plan_gpu > 0)].copy()
    print(f"  {len(t):,} GPU tasks across {t.job_name.nunique():,} jobs", flush=True)

    # workload tag: task.job_name -> job.inst_id -> group_tag.workload (sparse; miss -> "none")
    job = pd.read_csv(JOB, header=None, names=JOB_COLS, usecols=["job_name", "inst_id"])
    gt = pd.read_csv(GTAG, header=None, names=GTAG_COLS, usecols=["inst_id", "workload"])
    gt = gt.dropna(subset=["inst_id"]).drop_duplicates("inst_id")
    j2w = job.merge(gt, on="inst_id", how="left")[["job_name", "workload"]].drop_duplicates("job_name")
    t = t.merge(j2w, on="job_name", how="left")
    cov = t.workload.notna().mean() * 100
    print(f"  workload tag coverage: {cov:.1f}%", flush=True)

    # encode categoricals as int codes (HistGBT categorical); NaN -> its own code
    for c in CATEGORICAL:
        t[c] = t[c].fillna("__na__").astype("category").cat.codes
    return t.reset_index(drop=True)


def split_by_job(t: pd.DataFrame, frac_test: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    jobs = t.job_name.unique()
    test_jobs = set(rng.choice(jobs, size=int(len(jobs) * frac_test), replace=False))
    mask = t.job_name.isin(test_jobs)
    return t[~mask].reset_index(drop=True), t[mask].reset_index(drop=True)


# ------------------------------- predictors ---------------------------------
def global_predict(train, test):
    q = [float(train.plan_gpu.quantile(x)) for x in QUANTILES]
    return tuple(np.full(len(test), v) for v in q)


def gbt_predict(train, test, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat_mask = [f in CATEGORICAL for f in feats]
    Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
    ytr = np.log1p(train.plan_gpu.to_numpy())
    preds = []
    for qv in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=qv, max_iter=200, learning_rate=0.1,
            categorical_features=cat_mask, random_state=0)
        m.fit(Xtr, ytr)
        preds.append(np.expm1(m.predict(Xte)))
    P = np.sort(np.clip(np.vstack(preds), 0, None), axis=0)
    return P[0], P[1], P[2]


# ------------------------------- metrics ------------------------------------
def _avg_rank(x):
    order = x.argsort(); r = np.empty_like(order, float); r[order] = np.arange(len(x)); return r


def spearman(pred, truth):
    rp, rt = _avg_rank(pred), _avg_rank(truth)
    return float("nan") if rp.std() == 0 or rt.std() == 0 else float(np.corrcoef(rp, rt)[0, 1])


def score(p10, p50, p90, truth):
    p50 = np.maximum(p50, 1e-6); truth = np.maximum(truth, 1e-6)
    ratio = p50 / truth
    inside = (truth >= p10) & (truth <= p90)
    return {
        "MAE": float(np.mean(np.abs(p50 - truth))),
        "MdAE": float(np.median(np.abs(p50 - truth))),
        "within_2x_pct": float(np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100),
        "log_rmse": float(np.sqrt(np.mean(np.log(ratio) ** 2))),
        "spearman": spearman(p50, truth),
        "coverage": float(np.mean(inside)),
        "width": float(np.mean(p90 - p10)),
    }


def fmt(m):
    return (f"MAE={m['MAE']:.4f}  MdAE={m['MdAE']:.4f}  within2x={m['within_2x_pct']:5.1f}%  "
            f"logRMSE={m['log_rmse']:.3f}  rho={m['spearman']:+.3f}  "
            f"cov={m['coverage']:.2f}  width={m['width']:.3f}")


# --------------------------------- driver -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cpu-mem", action="store_true",
                    help="drop co-requested plan_cpu/plan_mem — tests if inst_num + tags alone "
                         "predict the GPU ask (harder, less co-request leakage)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not os.path.exists(TASK):
        raise SystemExit(f"missing {TASK} — run data/fetch_alibaba_gpu.py first")
    num_feats = [f for f in NUM_FEATS if not (args.no_cpu_mem and f in ("plan_cpu", "plan_mem"))]

    print("building features ...", flush=True)
    t0 = time.time()
    t = build_features()
    train, test = split_by_job(t)
    print(f"  train {len(train):,} / test {len(test):,} tasks "
          f"| plan_gpu median {t.plan_gpu.median():.1f} max {t.plan_gpu.max():.0f} "
          f"| {time.time()-t0:.1f}s\n", flush=True)

    results = {}
    preds = {}
    runs = [("global", lambda: global_predict(train, test)),
            ("gbt-num", lambda: gbt_predict(train, test, num_feats)),
            ("gbt-full", lambda: gbt_predict(train, test, num_feats + TAG_FEATS))]
    truth = test.plan_gpu.to_numpy()
    for name, fn in runs:
        tt = time.time()
        p10, p50, p90 = fn()
        preds[name] = (p10, p50, p90)
        results[name] = score(p10, p50, p90, truth)
        print(f"{name:9}: {fmt(results[name])}   ({time.time()-tt:.1f}s)", flush=True)

    # Per-job predicted-GPU sample for the Stage-2 negotiation (two_sided_sim). The best model's
    # (gbt-full) actual P10/P50/P90 predictions ARE the "predicted requested GPU" the demand agent
    # negotiates over; a bounded, seeded sample is enough to draw an empirical distribution from.
    p10, p50, p90 = preds["gbt-full"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(p50), size=min(2000, len(p50)), replace=False)
    per_job_gpu = [[round(float(p10[i]), 2), round(float(p50[i]), 2), round(float(p90[i]), 2)]
                   for i in idx]

    base, full = results["gbt-num"], results["gbt-full"]
    d_mae = (base["MAE"] - full["MAE"]) / base["MAE"] * 100
    d_log = (base["log_rmse"] - full["log_rmse"]) / base["log_rmse"] * 100
    verdict = "PASS — semantic tags add signal" if full["MAE"] < base["MAE"] else \
              "FAIL — tags do not help over raw co-request"
    print(f"\nGATE (gbt-full vs gbt-num): {verdict}")
    print(f"  MAE {base['MAE']:.4f} -> {full['MAE']:.4f} ({d_mae:+.1f}%)   "
          f"logRMSE {base['log_rmse']:.3f} -> {full['log_rmse']:.3f} ({d_log:+.1f}%)   "
          f"rho {base['spearman']:+.3f} -> {full['spearman']:+.3f}")

    json.dump({"n_train": len(train), "n_test": len(test),
               "no_cpu_mem": args.no_cpu_mem, "metrics": results,
               "delta_mae_pct": d_mae, "delta_logrmse_pct": d_log,
               "per_job_gpu": per_job_gpu},          # [P10,P50,P90] plan_gpu% for Stage-2
              open(args.out, "w"), indent=2)
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
