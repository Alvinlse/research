"""Attach REAL trace context to the replay jobs (Exp 78, tier 3).

`replay_jobs.csv` carries only `job_name, arrival, dur, quanta` — no text, because operator
notes are exactly what a public trace never records. But `pai_task_table.csv` carries fields
the scheduler's bid never sees and that a human operator plainly would:

    job_name, task_name, inst_num, status, start, end, plan_cpu, plan_mem, plan_gpu, gpu_type

`task_name` is the ROLE (tensorflow / ps / worker / PyTorchWorker / evaluator /
TensorboardTask / ReduceTask / xComputeWorker / ReduceTask ...), `inst_num` the instance
count, `gpu_type` the requested hardware class. None of it is in
`facts = {base, usable, waited, held}`, so it is genuine out-of-model context ON REAL JOBS —
the channel the correction layer needs and the hard-case suite could only simulate.

Why this matters for the experiment: the demand reviewer's question ("does this job need more
GPU than its numerical bid suggests?") has checkable answers here. A `ps` (parameter server)
or a `TensorboardTask` gains ~nothing from a margin GPU; a multi-instance `PyTorchWorker`
plausibly does. And the ground truth is not authored by us — it can be read from the trace's
own measured usage.

Output: data/alibaba-gpu-v2020/job_context.csv
    job_name, roles, n_tasks, inst_total, plan_gpu_max, gpu_type, note

`note` is the free-text rendering handed to the demand reviewer. It states only what the trace
says; it does NOT editorialise about whether the job deserves more GPU (that is the reviewer's
judgement under test, and writing the answer into the prompt would be the strawman).
"""
from __future__ import annotations

import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020")
REPLAY = os.path.join(DATA, "replay_jobs.csv")
TASKS = os.path.join(DATA, "pai_task_table.csv")
OUT = os.path.join(DATA, "job_context.csv")

TASK_COLS = ("job_name", "task_name", "inst_num", "status", "start_time", "end_time",
             "plan_cpu", "plan_mem", "plan_gpu", "gpu_type")


def note_for(roles: list[str], inst: int, gpu_type: str, plan_gpu: float) -> str:
    """Render the trace's own facts as the note a submission would carry. Descriptive only."""
    role_txt = ", ".join(f"{r}x{c}" if c > 1 else r
                         for r, c in collections.Counter(roles).most_common())
    parts = [f"task roles: {role_txt}"]
    if inst:
        parts.append(f"{inst} instance{'s' if inst != 1 else ''} in total")
    if gpu_type and gpu_type != "MISC":
        parts.append(f"requested GPU class {gpu_type}")
    if plan_gpu:
        parts.append(f"planned GPU {plan_gpu:g}% per instance")
    return "; ".join(parts) + "."


def main() -> None:
    want = set()
    with open(REPLAY) as f:
        for row in csv.DictReader(f):
            want.add(row["job_name"])
    print(f"replay jobs: {len(want):,}")

    agg: dict[str, dict] = {}
    seen = 0
    with open(TASKS) as f:
        for row in csv.reader(f):
            seen += 1
            if len(row) < len(TASK_COLS):
                continue
            jid = row[0]
            if jid not in want:
                continue
            a = agg.setdefault(jid, {"roles": [], "inst": 0.0, "plan_gpu": 0.0, "gpu": ""})
            a["roles"].append(row[1])
            try:
                a["inst"] += float(row[2] or 0)
            except ValueError:
                pass
            try:
                a["plan_gpu"] = max(a["plan_gpu"], float(row[8] or 0))
            except ValueError:
                pass
            if row[9] and not a["gpu"]:
                a["gpu"] = row[9]
            if seen % 500000 == 0:
                print(f"\r  scanned {seen:,} task rows, matched {len(agg):,}", end="", flush=True)
    print(f"\r  scanned {seen:,} task rows, matched {len(agg):,} jobs")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_name", "roles", "n_tasks", "inst_total", "plan_gpu_max",
                    "gpu_type", "note"])
        for jid, a in agg.items():
            roles = a["roles"]
            w.writerow([jid, "|".join(sorted(set(roles))), len(roles), int(a["inst"]),
                        f"{a['plan_gpu']:g}", a["gpu"],
                        note_for(roles, int(a["inst"]), a["gpu"], a["plan_gpu"])])

    cov = len(agg) / max(len(want), 1)
    roles = collections.Counter(r for a in agg.values() for r in set(a["roles"]))
    print(f"coverage: {len(agg):,}/{len(want):,} replay jobs ({cov:.1%}) -> {OUT}")
    print("distinct roles across matched jobs:", dict(roles.most_common(10)))
    multi = sum(1 for a in agg.values() if len(set(a["roles"])) > 1)
    print(f"jobs with >1 distinct role: {multi:,} ({multi / max(len(agg), 1):.1%})")


if __name__ == "__main__":
    sys.exit(main())
