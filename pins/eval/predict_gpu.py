"""
Stage-1 (GPU track) — predict a task's resources from its submission metadata.

RETARGET (2026-07-08). The original target, `plan_gpu`, is a USER-DECLARED field: it sits in
the submission script next to plan_cpu/plan_mem, so at decision time the scheduler already
has it — predicting it is imputation, not demand prediction (the Exp 30/31 caveat). The
genuinely unknown-at-submission quantities in this trace are the task's RUNTIME and its
ACTUAL usage (pai_sensor_table). So the harness now takes `--target`:
  * plan_gpu  — the original track, byte-identical (keeps Exp 30/31 reproducible).
  * runtime   — end_time − start_time (s), Terminated tasks only.
  * gpu_util  — ACTUAL GPU utilization % (sensor gpu_wrk_util, mean over workers).
  * gpu_mem   — ACTUAL peak GPU memory GB (sensor max_gpu_wrk_mem, max over workers).
For the non-plan_gpu targets, plan_gpu joins the features — the rule is "declared fields
are features, not targets". Population stays plan_gpu>0 (GPU tasks, the Stage-2 world).

Run:  .venv/bin/python -m pins.eval.predict_gpu --target runtime
      .venv/bin/python -m pins.eval.predict_gpu --target gpu_util   # needs pai_sensor_table
      (data/fetch_alibaba_gpu.py --tables pai_sensor_table)

Original context below.

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
SENSOR = os.path.join(DATA_DIR, "pai_sensor_table.csv")
OUT = os.path.join(HERE, "results_gpu.json")

TARGETS = ("plan_gpu", "runtime", "gpu_util", "gpu_mem")
QUANTILES = (0.10, 0.50, 0.90)
NUM_FEATS = ["inst_num", "plan_cpu", "plan_mem"]
TAG_FEATS = ["gpu_type", "task_name", "workload"]
CATEGORICAL = {"gpu_type", "task_name", "workload"}

TASK_COLS = ["job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]
JOB_COLS = ["job_name", "inst_id", "user", "status", "start_time", "end_time"]
GTAG_COLS = ["inst_id", "user", "gpu_type_spec", "group", "workload"]
SENSOR_COLS = ["job_name", "task_name", "worker_name", "inst_id", "machine", "gpu_name",
               "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem", "avg_gpu_wrk_mem",
               "max_gpu_wrk_mem", "read", "write", "read_count", "write_count"]


# --------------------------- data + features --------------------------------
def build_features(target: str = "plan_gpu") -> pd.DataFrame:
    """Load GPU tasks (plan_gpu>0), attach the target as column `y`, join the sparse
    workload tag, encode categoricals. For target != plan_gpu the declared plan_gpu is a
    FEATURE (it is known at submission); the target is what is not."""
    usecols = ["job_name", "task_name", "inst_num", "plan_cpu", "plan_mem",
               "plan_gpu", "gpu_type"]
    if target == "runtime":
        usecols += ["status", "start_time", "end_time"]
    t = pd.read_csv(TASK, header=None, names=TASK_COLS, usecols=usecols)
    t = t[(t.plan_gpu.notna()) & (t.plan_gpu > 0)].copy()
    print(f"  {len(t):,} GPU tasks across {t.job_name.nunique():,} jobs", flush=True)

    if target == "plan_gpu":
        t["y"] = t.plan_gpu
    elif target == "runtime":
        t = t[(t.status == "Terminated") & t.start_time.notna() & t.end_time.notna()].copy()
        t["y"] = t.end_time - t.start_time
        t = t[t.y > 0]
        print(f"  runtime: {len(t):,} Terminated tasks | median {t.y.median():.0f}s", flush=True)
    else:  # gpu_util / gpu_mem — ACTUAL usage, aggregated over the task's workers
        s = pd.read_csv(SENSOR, header=None, names=SENSOR_COLS,
                        usecols=["job_name", "task_name", "gpu_wrk_util", "max_gpu_wrk_mem"])
        agg = s.groupby(["job_name", "task_name"], as_index=False).agg(
            gpu_util=("gpu_wrk_util", "mean"), gpu_mem=("max_gpu_wrk_mem", "max"))
        n0 = len(t)
        t = t.merge(agg, on=["job_name", "task_name"], how="inner")
        t = t[t[target].notna()].copy()   # keep zeros: idle-GPU tasks ARE the signal
        t["y"] = t[target]
        print(f"  sensor join: {len(t):,}/{n0:,} tasks ({100*len(t)/n0:.1f}%) carry {target}",
              flush=True)

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
    q = [float(train.y.quantile(x)) for x in QUANTILES]
    return tuple(np.full(len(test), v) for v in q)


def gbt_predict(train, test, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat_mask = [f in CATEGORICAL for f in feats]
    Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
    ytr = np.log1p(train.y.to_numpy())
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
    ap.add_argument("--target", default="plan_gpu", choices=TARGETS,
                    help="what the GBT predicts. plan_gpu = the original declared-field track "
                         "(kept reproducible); runtime / gpu_util / gpu_mem are UNKNOWN at "
                         "submission — for these, plan_gpu becomes a feature")
    ap.add_argument("--no-cpu-mem", action="store_true",
                    help="drop co-requested plan_cpu/plan_mem — tests if inst_num + tags alone "
                         "predict the GPU ask (harder, less co-request leakage)")
    ap.add_argument("--out", default=None,
                    help="results json (default: results_gpu.json / results_<target>.json)")
    args = ap.parse_args()
    # 'runtime' gets a results_gpu_ prefix: plain results_runtime.json belongs to
    # predict_runtime.py (the Supercloud runtime eval, Exp 18-21) — don't clobber it.
    out = args.out or os.path.join(HERE, {"plan_gpu": "results_gpu.json",
                                          "runtime": "results_gpu_runtime.json"}
                                   .get(args.target, f"results_{args.target}.json"))
    if not os.path.exists(TASK):
        raise SystemExit(f"missing {TASK} — run data/fetch_alibaba_gpu.py first")
    if args.target in ("gpu_util", "gpu_mem") and not os.path.exists(SENSOR):
        raise SystemExit(f"missing {SENSOR} — run "
                         "data/fetch_alibaba_gpu.py --tables pai_sensor_table first")
    num_feats = [f for f in NUM_FEATS if not (args.no_cpu_mem and f in ("plan_cpu", "plan_mem"))]
    if args.target != "plan_gpu":
        num_feats.append("plan_gpu")   # declared fields are features, not targets

    print(f"building features (target={args.target}) ...", flush=True)
    t0 = time.time()
    t = build_features(args.target)
    train, test = split_by_job(t)
    print(f"  train {len(train):,} / test {len(test):,} tasks "
          f"| {args.target} median {t.y.median():.1f} max {t.y.max():.0f} "
          f"| {time.time()-t0:.1f}s\n", flush=True)

    results = {}
    preds = {}
    runs = [("global", lambda: global_predict(train, test)),
            ("gbt-num", lambda: gbt_predict(train, test, num_feats)),
            ("gbt-full", lambda: gbt_predict(train, test, num_feats + TAG_FEATS))]
    truth = test.y.to_numpy()
    for name, fn in runs:
        tt = time.time()
        p10, p50, p90 = fn()
        preds[name] = (p10, p50, p90)
        results[name] = score(p10, p50, p90, truth)
        print(f"{name:9}: {fmt(results[name])}   ({time.time()-tt:.1f}s)", flush=True)

    payload = {"n_train": len(train), "n_test": len(test), "target": args.target,
               "no_cpu_mem": args.no_cpu_mem, "metrics": results}

    if args.target == "plan_gpu":
        # Per-job predicted-GPU sample for the Stage-2 negotiation (two_sided_sim). The best model's
        # (gbt-full) actual P10/P50/P90 predictions ARE the "predicted requested GPU" the demand agent
        # negotiates over; a bounded, seeded sample is enough to draw an empirical distribution from.
        p10, p50, p90 = preds["gbt-full"]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(p50), size=min(2000, len(p50)), replace=False)
        payload["per_job_gpu"] = [[round(float(p10[i]), 2), round(float(p50[i]), 2),
                                   round(float(p90[i]), 2)] for i in idx]

        # Exp 30: per-JOB predicted quanta keyed by job_name, so trace_replay can negotiate over the
        # PREDICTION for the very jobs it replays (test set only — train jobs would leak). Aggregated
        # the same way replay_jobs.csv aggregates the truth: sum(plan_gpu*inst_num)/25 quarter-GPU.
        agg = test[["job_name", "inst_num"]].copy()
        for col, p in (("p10", p10), ("p50", p50), ("p90", p90)):
            agg[col] = p * agg.inst_num / 25.0
        per_job = agg.groupby("job_name")[["p10", "p50", "p90"]].sum().round(3)
        jobs_csv = os.path.join(HERE, "pred_job_gpu.csv")
        per_job.to_csv(jobs_csv)
        print(f"per-job predicted quanta ({len(per_job):,} test jobs) -> {jobs_csv}")

    if args.target == "gpu_util":
        # Exp 36: per-job predicted USAGE quanta + the usage TRUTH, keyed by job_name, so
        # trace_replay can run a usage-based replay world (true need = actual usage; request =
        # declared plan / predicted usage / oracle usage). Same aggregation recipe as
        # replay_jobs.csv / pred_job_gpu.csv: sum(util*inst_num)/25 quarter-GPU quanta.
        p10, p50, p90 = preds["gbt-full"]
        agg = test[["job_name", "inst_num"]].copy()
        for col, p in (("p10", p10), ("p50", p50), ("p90", p90), ("truth", truth)):
            agg[col] = p * agg.inst_num / 25.0
        per_job = agg.groupby("job_name")[["p10", "p50", "p90", "truth"]].sum().round(4)
        usage_csv = os.path.join(HERE, "pred_job_usage.csv")
        per_job.to_csv(usage_csv)
        print(f"per-job predicted+true usage quanta ({len(per_job):,} test jobs) -> {usage_csv}")

    base, full = results["gbt-num"], results["gbt-full"]
    d_mae = (base["MAE"] - full["MAE"]) / base["MAE"] * 100
    d_log = (base["log_rmse"] - full["log_rmse"]) / base["log_rmse"] * 100
    verdict = "PASS — semantic tags add signal" if full["MAE"] < base["MAE"] else \
              "FAIL — tags do not help over raw co-request"
    print(f"\nGATE (gbt-full vs gbt-num): {verdict}")
    print(f"  MAE {base['MAE']:.4f} -> {full['MAE']:.4f} ({d_mae:+.1f}%)   "
          f"logRMSE {base['log_rmse']:.3f} -> {full['log_rmse']:.3f} ({d_log:+.1f}%)   "
          f"rho {base['spearman']:+.3f} -> {full['spearman']:+.3f}")

    payload["delta_mae_pct"] = d_mae
    payload["delta_logrmse_pct"] = d_log
    json.dump(payload, open(out, "w"), indent=2)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
