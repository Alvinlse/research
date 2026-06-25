"""
Stage-1 cold-start RUNTIME prediction on the real MIT Supercloud trace.

Why runtime (not GPU memory / utilization)? Measured on the 150 labelled jobs with
telemetry, this workload has almost no per-job GPU slack to harvest (median GPU util
~92%, no idle allocated GPUs, negligible phase variation) and `tres_req` is a flat
copy-paste template — see memory `supercloud-profiling-data-reality`. The real
inefficiency in the trace is QUEUEING (median wait ~15 h). So the utilization win here
is CLUSTER-LEVEL scheduling (packing/ordering), and the Stage-1 prediction that feeds
it is each job's wall-clock RUNTIME WITH AN UNCERTAINTY INTERVAL: when will this job
free its GPUs (and how confidently) so the next can be placed.

Task: predict runtime (minutes) from thin submission-time metadata (model name +
requested resources + time limit) for the 3,430 labelled DNN jobs, joined from the
public scheduler log (`data/slurm-log.csv`). Ground truth = time_end - time_start.

Point predictors:
  * MEAN      : global median runtime (no-information floor).
  * HEURISTIC : a calibrated fraction of the user's requested time limit (metadata-only).
  * RETRIEVAL : per-model runtime quantiles (P10/P50/P90) from TRAIN; global-quantile
                fallback for an unseen model. P50 is the point; [P10,P90] the interval.
                The strong baseline to beat.
  * LLM       : qwen reasons from the model name + resources to a runtime DISTRIBUTION
                {p10,p50,p90} (cached per distinct prompt; only ~dozens of inputs).

Two evaluations:
  1. IN-DISTRIBUTION (5-fold, random): retrieval is expected to be strong.
  2. OUT-OF-DISTRIBUTION (leave-one-model-family-out): retrieval has no same-model
     history -> degrades to global quantiles; the LLM's world knowledge should degrade
     less. The differentiator (cf. the reference paper's <8% degradation under shift).

Metrics favour the scheduling-relevant quantities: Spearman rank (backfill ordering),
within-2x, and log-space error on a heavy tail; plus interval COVERAGE of [P10,P90]
(target ~0.80) and WIDTH (sharpness) for the uncertainty story.

Design principle (CLAUDE.md): the LLM emits numbers it REASONS to; deterministic code
clamps and scores. Quantile/coverage conventions mirror pins/forecast/model_quantile.py.

Run:  .venv/bin/python -m pins.eval.predict_runtime --no-llm           # baselines only
      .venv/bin/python -m pins.eval.predict_runtime                    # + LLM (3b)
      .venv/bin/python -m pins.eval.predict_runtime --models qwen2.5:3b,qwen2.5:7b,qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SLURM_LOG = os.path.join(ROOT, "data", "slurm-log.csv")
LABELS = os.path.join(ROOT, "data", "labelled_jobids_full.csv")
JOBS_CACHE = os.path.join(ROOT, "data", "runtime_jobs.csv")
LLM_CACHE = os.path.join(HERE, "llm_runtime_cache.json")

STATE_COMPLETED = 3                      # slurm job_state enum: 3 = COMPLETED
MIN_RUNTIME, MAX_RUNTIME = 1.0, 3000.0   # minutes; also the clamp band for predictions
QUANTILES = (0.10, 0.50, 0.90)           # P10 / P50 / P90  (matches model_quantile.py)

# Map a model name to a coarse architecture FAMILY for the leave-family-out split.
FAMILY_RULES = [
    ("vision", ("vgg", "resnet", "inception", "conv")),
    ("nlp",    ("bert", "distilbert")),
    ("unet",   ("u3", "u4", "u5")),
    ("gnn",    ("schnet", "pna", "dimenet", "nnconv")),
]


def family_of(model: str) -> str:
    m = model.lower()
    for fam, pres in FAMILY_RULES:
        if any(m.startswith(p) for p in pres):
            return fam
    return "other"


def clamp(v: float) -> float:
    return float(min(MAX_RUNTIME, max(MIN_RUNTIME, v)))


# --------------------------- data preparation -------------------------------
def _tres(s: str, key: int) -> int:
    m = re.search(rf"(?:^|,){key}=(\d+)", str(s))
    return int(m.group(1)) if m else 0


def build_jobs() -> pd.DataFrame:
    """Join the scheduler log to the labelled-job map and derive the runtime target."""
    lab = pd.read_csv(LABELS, dtype={"id_job": str})
    labmap = dict(zip(lab.id_job, lab.model))
    cols = ["id_job", "cpus_req", "nodes_alloc", "state", "timelimit",
            "time_start", "time_end", "tres_req"]
    df = pd.read_csv(SLURM_LOG, dtype={"id_job": str}, usecols=cols)
    df = df[df.id_job.isin(labmap)].copy()
    df["model"] = df.id_job.map(labmap)
    df["family"] = df.model.map(family_of)
    df["req_gpu"] = df.tres_req.apply(lambda s: _tres(s, 1001) + _tres(s, 1002))
    df["req_cpu"] = df.tres_req.apply(lambda s: _tres(s, 1))
    df["req_mem_gb"] = df.tres_req.apply(lambda s: _tres(s, 2) / 1024.0)
    df["timelimit_min"] = pd.to_numeric(df.timelimit, errors="coerce")
    df["runtime_min"] = (df.time_end - df.time_start) / 60.0
    df = df[(df.state == STATE_COMPLETED) &
            (df.runtime_min >= MIN_RUNTIME) & (df.runtime_min <= MAX_RUNTIME)]
    keep = ["id_job", "model", "family", "req_gpu", "req_cpu", "req_mem_gb",
            "timelimit_min", "runtime_min"]
    return df[keep].reset_index(drop=True)


def load_jobs(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not os.path.exists(JOBS_CACHE):
        jobs = build_jobs()
        jobs.to_csv(JOBS_CACHE, index=False)
        return jobs
    return pd.read_csv(JOBS_CACHE, dtype={"id_job": str})


# ------------------------------- predictors ---------------------------------
# Every predictor returns a (p10, p50, p90) tuple of arrays aligned to `test`.
# Point-only predictors set p10=p50=p90 (a degenerate, full-confidence interval).

def mean_predict(train: pd.DataFrame, test: pd.DataFrame):
    med = clamp(float(train.runtime_min.median()))
    p = np.full(len(test), med)
    return p, p.copy(), p.copy()


def heuristic_predict(train: pd.DataFrame, test: pd.DataFrame):
    """Non-LLM, metadata-only: a calibrated fraction of the requested time limit."""
    glob = clamp(float(train.runtime_min.median()))
    tl, rt = train.timelimit_min.to_numpy(), train.runtime_min.to_numpy()
    ok = np.isfinite(tl) & (tl > 0) & (tl < MAX_RUNTIME * 5)
    frac = float(np.median(rt[ok] / tl[ok])) if ok.any() else 0.5
    out = []
    for tlim in test.timelimit_min:
        out.append(clamp(frac * tlim) if np.isfinite(tlim) and 0 < tlim < MAX_RUNTIME * 5
                   else glob)
    p = np.array(out, float)
    return p, p.copy(), p.copy()


def retrieval_predict(train: pd.DataFrame, test: pd.DataFrame):
    """Per-model empirical P10/P50/P90 from train; global quantiles as fallback."""
    g10, g50, g90 = (clamp(float(train.runtime_min.quantile(q))) for q in QUANTILES)
    q = train.groupby("model").runtime_min.quantile(list(QUANTILES)).unstack()
    by_model = {m: (clamp(r[QUANTILES[0]]), clamp(r[QUANTILES[1]]), clamp(r[QUANTILES[2]]))
                for m, r in q.iterrows()}
    p10, p50, p90 = [], [], []
    for m in test.model:
        a, b, c = by_model.get(m, (g10, g50, g90))
        p10.append(a); p50.append(b); p90.append(c)
    return np.array(p10), np.array(p50), np.array(p90)


# ------------------------------- LLM path -----------------------------------
SYSTEM = (
    "You are an HPC job-runtime estimator for deep-learning TRAINING jobs on the MIT "
    "Supercloud cluster (NVIDIA V100 GPUs). Given a job's submission metadata, estimate "
    "its total wall-clock runtime IN MINUTES as a distribution: a most-likely value p50 "
    "and an 80% interval [p10, p90]. Use your knowledge of how expensive each named model "
    "is to train and how the requested GPU count changes it. Respond with ONLY a JSON "
    'object: {"p10_min": <number>, "p50_min": <number>, "p90_min": <number>} with '
    "p10 <= p50 <= p90."
)


def llm_prompt(j) -> str:
    tl = (f"{int(j.timelimit_min)} min" if np.isfinite(j.timelimit_min) and j.timelimit_min > 0
          else "unspecified")
    return (
        f"Model trained: {j.model}\n"
        f"GPUs requested: {int(j.req_gpu)} (V100)\n"
        f"CPUs requested: {int(j.req_cpu)}\n"
        f"Host memory requested: {j.req_mem_gb:.0f} GB\n"
        f"User time limit: {tl}\n\n"
        "Estimate the runtime distribution in minutes."
    )


def parse_quantiles(text: str):
    """Return clamped, monotone (p10,p50,p90) or None."""
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
        vals = sorted(clamp(float(obj[k])) for k in ("p10_min", "p50_min", "p90_min"))
        return vals[0], vals[1], vals[2]
    except Exception:
        return None


def llm_predict(client, model_name: str, jobs: pd.DataFrame, cache: dict,
                fallback, retries: int = 2):
    """One LLM call per DISTINCT prompt. fallback = (p10,p50,p90) used on parse failure."""
    p10, p50, p90 = [], [], []
    for _, j in jobs.iterrows():
        prompt = llm_prompt(j)
        key = f"{model_name}|{prompt}"
        if key not in cache:
            val = None
            for _ in range(retries):
                try:
                    resp = client.chat(
                        model=model_name, format="json",
                        options={"temperature": 0, "num_predict": 80},
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": prompt}])
                    val = parse_quantiles(resp.message.content)
                    if val is not None:
                        break
                except Exception as e:
                    print(f"    ! ollama error: {type(e).__name__}: {e}")
                    time.sleep(1)
            cache[key] = val
            json.dump(cache, open(LLM_CACHE, "w"))   # persist per new call: timeout-safe, resumable
        a, b, c = cache[key] if cache[key] is not None else fallback
        p10.append(a); p50.append(b); p90.append(c)
    return np.array(p10), np.array(p50), np.array(p90)


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


def score(p10, p50, p90, truth) -> dict:
    p50 = np.maximum(p50, 1e-6); truth = np.maximum(truth, 1e-6)
    ratio = p50 / truth
    inside = (truth >= p10) & (truth <= p90)
    return {
        "MAE_min": float(np.mean(np.abs(p50 - truth))),
        "MdAE_min": float(np.median(np.abs(p50 - truth))),
        "MAPE_pct": float(np.mean(np.abs(p50 - truth) / truth) * 100),
        "within_2x_pct": float(np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100),
        "log_rmse": float(np.sqrt(np.mean(np.log(ratio) ** 2))),
        "spearman": spearman(p50, truth),
        "coverage": float(np.mean(inside)),          # target ~0.80
        "width_min": float(np.mean(p90 - p10)),      # sharpness (lower better)
    }


def fmt_point(m: dict) -> str:
    return (f"MAE={m['MAE_min']:6.1f}m  MdAE={m['MdAE_min']:6.1f}m  within2x={m['within_2x_pct']:5.1f}%  "
            f"logRMSE={m['log_rmse']:.2f}  rho={m['spearman']:+.2f}")


def fmt_interval(m: dict) -> str:
    return f"coverage={m['coverage']:.2f} (target 0.80)  width={m['width_min']:6.1f}m"


# --------------------------- evaluation drivers -----------------------------
def kfold_indices(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


DETERMINISTIC = {"mean": mean_predict, "heuristic": heuristic_predict,
                 "retrieval": retrieval_predict}


def eval_in_distribution(jobs, client, llm_models, cache, k: int = 5) -> dict:
    n = len(jobs)
    folds = kfold_indices(n, k)
    names = list(DETERMINISTIC) + llm_models
    P = {nm: [np.zeros(n), np.zeros(n), np.zeros(n)] for nm in names}
    for f in folds:
        test, train = jobs.iloc[f], jobs.drop(jobs.index[f])
        gq = tuple(clamp(float(train.runtime_min.quantile(q))) for q in QUANTILES)
        for nm, fn in DETERMINISTIC.items():
            a, b, c = fn(train, test)
            P[nm][0][f], P[nm][1][f], P[nm][2][f] = a, b, c
        for mdl in llm_models:
            a, b, c = llm_predict(client, mdl, test, cache, gq)
            P[mdl][0][f], P[mdl][1][f], P[mdl][2][f] = a, b, c
    truth = jobs.runtime_min.to_numpy()
    return {nm: score(P[nm][0], P[nm][1], P[nm][2], truth) for nm in names}


def eval_ood(jobs, client, llm_models, cache) -> dict:
    fams = sorted(f for f in jobs.family.unique() if f != "other")
    res = {}
    for fam in fams:
        test, train = jobs[jobs.family == fam], jobs[jobs.family != fam]
        gq = tuple(clamp(float(train.runtime_min.quantile(q))) for q in QUANTILES)
        truth = test.runtime_min.to_numpy()
        row = {"n": int(len(test))}
        for nm in ("mean", "retrieval"):
            a, b, c = DETERMINISTIC[nm](train, test)
            row[nm] = score(a, b, c, truth)
        for mdl in llm_models:
            a, b, c = llm_predict(client, mdl, test, cache, gq)
            row[mdl] = score(a, b, c, truth)
        res[fam] = row
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5:3b",
                    help="comma-separated ollama models for the LLM predictor (size ablation)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true", help="baselines only (no Ollama)")
    ap.add_argument("--rebuild", action="store_true", help="re-parse slurm-log.csv")
    ap.add_argument("--out", default=os.path.join(HERE, "results_runtime.json"))
    args = ap.parse_args()

    jobs = load_jobs(rebuild=args.rebuild)
    print(f"{len(jobs)} completed labelled jobs | "
          f"runtime min/median/max = {jobs.runtime_min.min():.0f}/"
          f"{jobs.runtime_min.median():.0f}/{jobs.runtime_min.max():.0f} min | "
          f"{jobs.model.nunique()} models, {jobs.family.nunique()} families\n")

    llm_models = [] if args.no_llm else [m.strip() for m in args.models.split(",") if m.strip()]
    client = cache = None
    if llm_models:
        import ollama
        client = ollama.Client(host=args.host)
        cache = json.load(open(LLM_CACHE)) if os.path.exists(LLM_CACHE) else {}

    print("=" * 92)
    print("IN-DISTRIBUTION (5-fold, random split)")
    print("=" * 92)
    indist = eval_in_distribution(jobs, client, llm_models, cache)
    for nm in list(DETERMINISTIC) + llm_models:
        label = nm.upper() if nm in DETERMINISTIC else f"LLM {nm}"
        line = f"{label:20}: {fmt_point(indist[nm])}"
        if nm == "retrieval" or nm in llm_models:        # interval-bearing predictors
            line += f"   | {fmt_interval(indist[nm])}"
        print(line)

    print("\n" + "=" * 92)
    print("OUT-OF-DISTRIBUTION (leave-one-model-family-out)  [does the predictor survive shift?]")
    print("=" * 92)
    ood = eval_ood(jobs, client, llm_models, cache)
    cols = ["retrieval"] + llm_models
    print(f"{'held-out':10}{'n':>5}   " + "  ".join(f"{c[:14]:>22}" for c in cols))
    print(f"{'':10}{'':>5}   " + "  ".join(f"{'within2x/rho/cov':>22}" for _ in cols))
    print("-" * (15 + 24 * len(cols)))
    for fam, row in ood.items():
        cells = []
        for c in cols:
            m = row[c]
            cells.append(f"{m['within_2x_pct']:5.1f}/{m['spearman']:+.2f}/{m['coverage']:.2f}".rjust(22))
        print(f"{fam:10}{row['n']:>5}   " + "  ".join(cells))

    if cache is not None:
        json.dump(cache, open(LLM_CACHE, "w"), indent=2)
    json.dump({"models": llm_models, "n_jobs": len(jobs),
               "in_distribution": indist, "ood_leave_family_out": ood},
              open(args.out, "w"), indent=2)
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
