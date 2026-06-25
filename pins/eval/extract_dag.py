"""
Stage-1 (DAG track) — extract per-job task DAGs from the Alibaba cluster-trace-v2018.

WHY this file exists. The MIT Supercloud trace has no workflow structure: every job is a
single training script, `tres_req` is a flat template, and there are zero task
dependencies (verified). So "predict the required resource from the job's DAG" is simply
not expressible there. Alibaba v2018 IS DAG-native: a batch *job* is a set of *tasks*
with explicit precedence, and each task carries its REQUESTED resources (`plan_cpu`,
`plan_mem`). This module turns the raw `batch_task.csv` into a per-job task DAG so
downstream work can ask: does DAG topology + per-node requests predict the resource a
task/job actually needs?

THE ENCODING (the whole trick). `batch_task.csv` has no header; columns (in order) are:

    task_name, instance_num, job_name, task_type, status, start_time, end_time,
    plan_cpu, plan_mem

A job = all rows sharing `job_name` (e.g. "j_3418309"). The DAG is reconstructed purely
from each task's `task_name`:

    <prefix><id>_<dep1>_<dep2>...      e.g.  M1 | M2_1 | R4_2 | M5_3_4
      |       |    \_____________ ids this task DEPENDS ON (edges dep->id)
      |       \__________________ this task's id within the job
      \__________________________ stage type (M/R/J...), IRRELEVANT to dependencies

Only the NUMBERS matter. `M1` is a root (no deps). `M5_3_4` depends on tasks 3 and 4.
Randomly-named tasks ("task_<base64>", "MergeTask", ...) don't match the numeric pattern
and are treated as INDEPENDENT singletons (a node, no edges; given a synthetic negative id).

OUTPUT (compact, vectorized — no 4M networkx graphs in RAM):
  * data/alibaba-v2018/dag_nodes.csv.gz  — one row per task:
        job_name, task_id, task_name, instances, plan_cpu, plan_mem, duration, depth
  * data/alibaba-v2018/dag_edges.csv.gz  — one row per dependency: job_name, src, dst
  * data/alibaba-v2018/dag_stats.json    — prevalence summary (the make-or-break number)
These node/edge tables load straight into pandas or PyTorch-Geometric for modeling.

`depth` is the node's longest-path distance from a root, computed globally by iterative
relaxation over the edge table (max ~tens of iterations = the deepest DAG).

Design principle (CLAUDE.md): deterministic code parses/decides structure; no LLM here.

Run:  .venv/bin/python -m pins.eval.extract_dag                 # full file
      .venv/bin/python -m pins.eval.extract_dag --limit 2000000 # quick slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "alibaba-v2018"
BATCH_TASK_CSV = DATA_DIR / "batch_task.csv"
NODES_OUT = DATA_DIR / "dag_nodes.csv.gz"
EDGES_OUT = DATA_DIR / "dag_edges.csv.gz"
STATS_OUT = DATA_DIR / "dag_stats.json"

COLS = ["task_name", "instance_num", "job_name", "task_type", "status",
        "start_time", "end_time", "plan_cpu", "plan_mem"]
USE = ["task_name", "instance_num", "job_name", "start_time", "end_time", "plan_cpu", "plan_mem"]

# prefix letters, the task id, then zero or more _<dep> groups, nothing else.
TASK_RE = r"^[A-Za-z]+(\d+)((?:_\d+)+)?$"


def load(limit=None) -> pd.DataFrame:
    print(f"reading {BATCH_TASK_CSV} (limit={limit}) ...", flush=True)
    df = pd.read_csv(BATCH_TASK_CSV, header=None, names=COLS, usecols=USE, nrows=limit,
                     dtype={"task_name": str, "job_name": str})
    print(f"  {len(df):,} task rows, {df['job_name'].nunique():,} jobs", flush=True)
    return df


def build_tables(df: pd.DataFrame):
    """Vectorized parse -> (nodes, edges) DataFrames."""
    ext = df["task_name"].str.extract(TASK_RE)            # [id, dep_str]
    df = df.assign(
        task_id=pd.to_numeric(ext[0], errors="coerce"),
        dep_str=ext[1],
        duration=df["end_time"] - df["start_time"],
    )

    # unparsed (random/base64/MergeTask) -> independent singletons with synthetic neg ids
    unp = df["task_id"].isna()
    n_unp = int(unp.sum())
    if n_unp:
        df.loc[unp, "task_id"] = -(df.loc[unp].groupby("job_name").cumcount() + 1)
    df["task_id"] = df["task_id"].astype(np.int64)

    nodes = df[["job_name", "task_id", "task_name", "instance_num",
                "plan_cpu", "plan_mem", "duration"]].rename(columns={"instance_num": "instances"})

    # edges: explode the dependency string of every task that has one
    dep = df.loc[df["dep_str"].notna(), ["job_name", "task_id", "dep_str"]].copy()
    dep["src"] = dep["dep_str"].str.strip("_").str.split("_")
    dep = dep.explode("src")
    dep["src"] = pd.to_numeric(dep["src"], errors="coerce").astype("Int64")
    edges = (dep[["job_name", "src", "task_id"]]
             .rename(columns={"task_id": "dst"}).dropna())
    edges["src"] = edges["src"].astype(np.int64)
    edges = edges[edges["src"] != edges["dst"]]           # no self-loops

    # keep only edges whose src actually exists as a task in that job
    valid = nodes[["job_name", "task_id"]].rename(columns={"task_id": "src"})
    before = len(edges)
    edges = edges.merge(valid, on=["job_name", "src"], how="inner")
    dropped = before - len(edges)
    edges = edges.drop_duplicates(["job_name", "src", "dst"]).reset_index(drop=True)
    return nodes, edges, n_unp, dropped


def compute_depth(nodes: pd.DataFrame, edges: pd.DataFrame, maxit: int = 64):
    """Longest-path depth per node (0 = root), by global iterative relaxation.

    DAGs converge within `max-depth` iterations. A non-convergence at `maxit` means a
    cycle (a few jobs have malformed self/back-referencing task_names) — those jobs'
    nodes are set to depth -1 and excluded from depth stats. Returns (depth, n_cyclic)."""
    depth = np.zeros(len(nodes), dtype=np.int32)
    cyclic_jobs = 0
    if len(edges):
        # map (job_name, task_id) -> positional node index via merge (C-speed, not a py dict)
        pos = nodes[["job_name", "task_id"]].reset_index(drop=True)
        pos["pos"] = np.arange(len(pos), dtype=np.int64)
        s = (edges[["job_name", "src"]].merge(
                pos.rename(columns={"task_id": "src"}), on=["job_name", "src"], how="left")
             )["pos"].to_numpy()
        d = (edges[["job_name", "dst"]].merge(
                pos.rename(columns={"task_id": "dst"}), on=["job_name", "dst"], how="left")
             )["pos"].to_numpy()
        last_better = None
        for it in range(maxit):
            upd = pd.Series(depth[s] + 1).groupby(d).max()   # best new depth per dst
            better = upd.values > depth[upd.index.values]
            if not better.any():
                print(f"  depth converged in {it} iters (max depth {depth.max()})", flush=True)
                last_better = None
                break
            depth[upd.index.values[better]] = upd.values[better]
            last_better = upd.index.values[better]
            if (it + 1) % 8 == 0:
                print(f"    iter {it+1}: max depth so far {depth.max()}", flush=True)
        if last_better is not None:                          # hit cap -> cycle(s) present
            bad = set(nodes.iloc[last_better]["job_name"])
            mask = nodes["job_name"].isin(bad).values
            depth[mask] = -1
            cyclic_jobs = len(bad)
            print(f"  WARNING: {cyclic_jobs} cyclic job(s) -> depth set to -1", flush=True)
    return pd.Series(depth, index=nodes.index, name="depth"), cyclic_jobs


def summarize(nodes, edges, n_unp, dropped, cyclic_jobs) -> dict:
    sz = nodes.groupby("job_name").size()
    n_jobs = len(sz)
    multitask = int((sz >= 2).sum())
    jobs_with_edges = edges["job_name"].nunique()
    job_depth = nodes.groupby("job_name")["depth"].max()
    ed = job_depth[job_depth > 0]

    st = {
        "jobs_total": int(n_jobs),
        "tasks_total": int(len(nodes)),
        "single_task_jobs": int(n_jobs - multitask),
        "multitask_jobs": multitask,
        "multitask_pct": round(100 * multitask / n_jobs, 2),
        "jobs_with_dep_edge": int(jobs_with_edges),
        "jobs_with_dep_edge_pct": round(100 * jobs_with_edges / n_jobs, 2),
        "edges_total": int(len(edges)),
        "unparsed_singleton_tasks": int(n_unp),
        "edges_dropped_missing_src": int(dropped),
        "cyclic_jobs": int(cyclic_jobs),
        "tasks_per_job": {"mean": round(float(sz.mean()), 2), "median": int(sz.median()),
                          "p90": int(sz.quantile(.9)), "p99": int(sz.quantile(.99)),
                          "max": int(sz.max())},
        "dag_depth_jobs_with_edges": ({} if ed.empty else
            {"mean": round(float(ed.mean()), 2), "median": int(ed.median()),
             "p90": int(ed.quantile(.9)), "max": int(ed.max())}),
    }

    print(f"\n{'='*60}\nDAG PREVALENCE (full trace)\n{'='*60}")
    print(f"jobs total              : {st['jobs_total']:,}")
    print(f"tasks total             : {st['tasks_total']:,}")
    print(f"single-task jobs        : {st['single_task_jobs']:,} "
          f"({100*st['single_task_jobs']/n_jobs:.1f}%)")
    print(f"multi-task jobs (>=2)    : {st['multitask_jobs']:,} ({st['multitask_pct']}%)")
    print(f"jobs with >=1 dep edge   : {st['jobs_with_dep_edge']:,} ({st['jobs_with_dep_edge_pct']}%)")
    print(f"dependency edges total   : {st['edges_total']:,}")
    print(f"\ntasks/job  : {st['tasks_per_job']}")
    print(f"DAG depth  : {st['dag_depth_jobs_with_edges']}  (jobs with edges)")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="read only N rows (quick look)")
    args = ap.parse_args()
    if not BATCH_TASK_CSV.exists():
        sys.exit(f"missing {BATCH_TASK_CSV} — extract batch_task.tar.gz first")

    df = load(args.limit)
    print("parsing task_name -> nodes + edges ...", flush=True)
    nodes, edges, n_unp, dropped = build_tables(df)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges "
          f"({n_unp:,} unparsed singletons, {dropped:,} edges dropped w/ missing src)", flush=True)

    print("computing longest-path depth ...", flush=True)
    nodes = nodes.copy()
    depth, cyclic_jobs = compute_depth(nodes, edges)
    nodes["depth"] = depth.values

    st = summarize(nodes, edges, n_unp, dropped, cyclic_jobs)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(NODES_OUT, index=False)
    edges.to_csv(EDGES_OUT, index=False)
    STATS_OUT.write_text(json.dumps(st, indent=2))
    print(f"\nsaved:\n  {NODES_OUT}\n  {EDGES_OUT}\n  {STATS_OUT}", flush=True)


if __name__ == "__main__":
    main()
