"""
Stage-1 (DAG track) — does DAG TOPOLOGY predict a task's resource demand?

Context. Exp 19 found the MIT Supercloud trace has NO workflow structure (every job is a
single script, requests are a flat template), so "predict resource from DAG" is not
expressible there. The Alibaba cluster-trace-v2018 IS DAG-native: `extract_dag.py` turned
`batch_task.csv` into per-task nodes + dependency edges (4.2M jobs, 48% with >=1 edge).
This module asks the modelling question that extraction set up: given a task's position in
its job DAG and the demand of its UPSTREAM tasks, can we predict the resource it requests?

TARGET = plan_mem (per-task requested memory; 322 distinct values, the richest demand
signal — plan_cpu has only 16, duration is contaminated with negatives). These are
REQUESTED resources (no batch_instance/actual-usage file exists), so the honest framing is
"forecast a task's demand from its DAG context before it is submitted" — useful to the
supply/demand agents, and exactly the signal Supercloud lacked.

THE EXPERIMENT (a clean ablation, the lab's "beat-the-baseline" gate). Three predictors,
all emitting P10/P50/P90 (uncertainty story, conventions from predict_runtime.py):
  * GLOBAL    — global plan_mem quantiles (no-information floor; must be beaten).
  * GBT-NODAG — gradient-boosted quantile regressor on features knowable WITHOUT the DAG:
                the task's own [instances, plan_cpu, stage_type]. A strong baseline.
  * GBT-DAG   — the SAME model + topology features: [depth, in_degree, out_degree,
                n_tasks_in_job, parent_mem_{mean,max}, parent_cpu_mean].
The whole question is GBT-DAG vs GBT-NODAG: any gain is attributable to topology ALONE
(every non-DAG feature is in both arms). No leakage: `duration` (an outcome) is excluded;
parent features come from UPSTREAM tasks (lower depth), known at submit time.

Model = sklearn HistGradientBoostingRegressor(loss="quantile") on log1p(plan_mem) (skewed
target), one fit per quantile, predictions sorted to stay monotone. Split BY JOB (siblings
never straddle train/test). Deterministic throughout — no LLM (Exp 19 showed the LLM does
not earn its cost as a numeric predictor; the DAG signal, if any, is structural).

Run:  .venv/bin/python -m pins.eval.predict_dag                 # 500k-job sample (fast)
      .venv/bin/python -m pins.eval.predict_dag --full          # all 4.2M jobs
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
DATA_DIR = os.path.join(ROOT, "data", "alibaba-v2018")
NODES = os.path.join(DATA_DIR, "dag_nodes.csv.gz")
EDGES = os.path.join(DATA_DIR, "dag_edges.csv.gz")
OUT = os.path.join(HERE, "results_dag.json")

QUANTILES = (0.10, 0.50, 0.90)
NODAG_FEATS = ["instances", "plan_cpu", "stage_type"]
DAG_FEATS = ["depth", "in_degree", "out_degree", "n_tasks_in_job",
             "parent_mem_mean", "parent_mem_max", "parent_cpu_mean"]
CATEGORICAL = {"stage_type"}                  # the rest are numeric


# --------------------------- data + features --------------------------------
def build_features(limit_jobs: int | None) -> pd.DataFrame:
    """Load nodes/edges, sample BY JOB, attach topology + upstream-demand features."""
    nodes = pd.read_csv(NODES)
    nodes = nodes[(nodes.plan_mem.notna()) & (nodes.depth >= 0)].copy()   # need target; drop cyclic
    if limit_jobs:
        rng = np.random.default_rng(0)
        jobs = nodes.job_name.unique()
        keep = set(rng.choice(jobs, size=min(limit_jobs, len(jobs)), replace=False))
        nodes = nodes[nodes.job_name.isin(keep)].copy()
    print(f"  {len(nodes):,} tasks across {nodes.job_name.nunique():,} jobs", flush=True)

    # stage_type = the task_name prefix letters (M/R/J/...), encoded as int category codes
    pref = nodes.task_name.str.extract(r"^([A-Za-z]+)")[0].fillna("?")
    nodes["stage_type"] = pref.astype("category").cat.codes

    edges = pd.read_csv(EDGES)
    edges = edges[edges.job_name.isin(nodes.job_name.unique())]

    # degrees
    indeg = edges.groupby(["job_name", "dst"]).size().rename("in_degree")
    outdeg = edges.groupby(["job_name", "src"]).size().rename("out_degree")
    nodes = nodes.merge(indeg.reset_index().rename(columns={"dst": "task_id"}),
                        on=["job_name", "task_id"], how="left")
    nodes = nodes.merge(outdeg.reset_index().rename(columns={"src": "task_id"}),
                        on=["job_name", "task_id"], how="left")
    nodes[["in_degree", "out_degree"]] = nodes[["in_degree", "out_degree"]].fillna(0)

    # job size
    nodes["n_tasks_in_job"] = nodes.groupby("job_name").task_id.transform("size")

    # upstream-demand: join each edge's src to its node's plan_mem/plan_cpu, aggregate per dst
    pm = nodes[["job_name", "task_id", "plan_mem", "plan_cpu"]].rename(
        columns={"task_id": "src", "plan_mem": "src_mem", "plan_cpu": "src_cpu"})
    pe = edges.merge(pm, on=["job_name", "src"], how="left")
    agg = pe.groupby(["job_name", "dst"]).agg(
        parent_mem_mean=("src_mem", "mean"),
        parent_mem_max=("src_mem", "max"),
        parent_cpu_mean=("src_cpu", "mean")).reset_index().rename(columns={"dst": "task_id"})
    nodes = nodes.merge(agg, on=["job_name", "task_id"], how="left")   # roots -> NaN (handled natively)
    return nodes.reset_index(drop=True)


def split_by_job(nodes: pd.DataFrame, frac_test: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    jobs = nodes.job_name.unique()
    test_jobs = set(rng.choice(jobs, size=int(len(jobs) * frac_test), replace=False))
    mask = nodes.job_name.isin(test_jobs)
    return nodes[~mask].reset_index(drop=True), nodes[mask].reset_index(drop=True)


# ------------------------------- predictors ---------------------------------
# Each returns (p10, p50, p90) arrays aligned to `test`, in raw plan_mem units.

def global_predict(train, test):
    q = [float(train.plan_mem.quantile(x)) for x in QUANTILES]
    return tuple(np.full(len(test), v) for v in q)


def gbt_predict(train, test, feats):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cat_mask = [f in CATEGORICAL for f in feats]
    Xtr, Xte = train[feats].to_numpy(float), test[feats].to_numpy(float)
    ytr = np.log1p(train.plan_mem.to_numpy())
    preds = []
    for qv in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=qv, max_iter=200, learning_rate=0.1,
            categorical_features=cat_mask, random_state=0)
        m.fit(Xtr, ytr)
        preds.append(np.expm1(m.predict(Xte)))
    P = np.sort(np.clip(np.vstack(preds), 0, None), axis=0)        # enforce p10<=p50<=p90
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
        "coverage": float(np.mean(inside)),       # target ~0.80
        "width": float(np.mean(p90 - p10)),       # sharpness (lower better)
    }


def fmt(m):
    return (f"MAE={m['MAE']:.4f}  MdAE={m['MdAE']:.4f}  within2x={m['within_2x_pct']:5.1f}%  "
            f"logRMSE={m['log_rmse']:.3f}  rho={m['spearman']:+.3f}  "
            f"cov={m['coverage']:.2f}  width={m['width']:.3f}")


# --------------------------------- driver -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use all jobs (default: 500k-job sample)")
    ap.add_argument("--limit-jobs", type=int, default=500_000)
    ap.add_argument("--no-cpu", action="store_true",
                    help="drop the co-requested plan_cpu — isolates whether topology predicts "
                         "demand from upstream STRUCTURE alone (forecast-before-submit setting)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if not os.path.exists(NODES):
        raise SystemExit(f"missing {NODES} — run pins.eval.extract_dag first")
    nodag_feats = [f for f in NODAG_FEATS if not (args.no_cpu and f == "plan_cpu")]

    print("building features ...", flush=True)
    t0 = time.time()
    nodes = build_features(None if args.full else args.limit_jobs)
    train, test = split_by_job(nodes)
    print(f"  train {len(train):,} / test {len(test):,} tasks "
          f"| plan_mem median {nodes.plan_mem.median():.3f} max {nodes.plan_mem.max():.2f} "
          f"| {time.time()-t0:.1f}s\n", flush=True)

    results = {}
    runs = [("global", lambda: global_predict(train, test)),
            ("gbt-nodag", lambda: gbt_predict(train, test, nodag_feats)),
            ("gbt-dag", lambda: gbt_predict(train, test, nodag_feats + DAG_FEATS))]
    truth = test.plan_mem.to_numpy()
    for name, fn in runs:
        t = time.time()
        p10, p50, p90 = fn()
        results[name] = score(p10, p50, p90, truth)
        print(f"{name:11}: {fmt(results[name])}   ({time.time()-t:.1f}s)", flush=True)

    base, dag = results["gbt-nodag"], results["gbt-dag"]
    d_mae = (base["MAE"] - dag["MAE"]) / base["MAE"] * 100
    d_log = (base["log_rmse"] - dag["log_rmse"]) / base["log_rmse"] * 100
    verdict = "PASS — DAG topology adds signal" if dag["MAE"] < base["MAE"] else \
              "FAIL — topology does not help over no-DAG features"
    print(f"\nGATE (gbt-dag vs gbt-nodag): {verdict}")
    print(f"  MAE {base['MAE']:.4f} -> {dag['MAE']:.4f} ({d_mae:+.1f}%)   "
          f"logRMSE {base['log_rmse']:.3f} -> {dag['log_rmse']:.3f} ({d_log:+.1f}%)   "
          f"rho {base['spearman']:+.3f} -> {dag['spearman']:+.3f}")

    json.dump({"n_train": len(train), "n_test": len(test),
               "full": args.full, "no_cpu": args.no_cpu, "metrics": results,
               "delta_mae_pct": d_mae, "delta_logrmse_pct": d_log},
              open(args.out, "w"), indent=2)
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
