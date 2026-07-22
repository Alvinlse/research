"""Worker-holdout split on Alibaba v2020 — a leakage-free venue for runtime-evidence reasoning.

Exp 78 showed the reviewer's signal was the ROLE LABEL, and exactly as informative as a lookup
table on that field. The proposed fix is to remove role names and reason from runtime behaviour
instead. But the runtime quantity we score against (`gpu_wrk_util`) is the same quantity we
would be showing as evidence — so a strong result would be the model reading the answer back.
And `pai_sensor_table` has no timestamp (16 fields, one row per WORKER), so there is no
temporal split available: no trend, no progress-per-interval, no resize history.

What the trace does support is a SPATIAL split. A multi-worker job has several instances; show
statistics from some of them, score against the others:

    EVIDENCE workers  (shown)  -> gpu_util, cpu_usage, gpu memory, I/O, worker count
    HELD-OUT workers  (truth)  -> mean gpu_util, never shown

The split is deterministic (sorted by instance id, alternating) so the same job always yields
the same split, and no worker appears on both sides.

WHAT THIS CAN AND CANNOT TEST. It tests whether behaviour observed on part of a job supports a
judgement about the rest of it. It does NOT test "would an extra GPU help": the trace records
behaviour only under the allocation that actually occurred (elevated plan §3.1), so marginal
benefit has no measured ground truth here at all — that is what the plan's saturating model
simulates. Measured utilisation answers "did it use what it had".

Output: data/alibaba-gpu-v2020/worker_holdout.csv
    job_name, n_ev, n_ho, ev_util_mean, ev_util_max, ev_util_std, ev_cpu_mean,
    ev_mem_mean, ev_mem_max, ev_read, ev_write, ho_util_mean, roles, quanta
"""
from __future__ import annotations

import collections
import csv
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020")
SENSOR = os.path.join(DATA, "pai_sensor_table.csv")
OUT = os.path.join(DATA, "worker_holdout.csv")
MIN_WORKERS = 4                     # need at least 2 a side for a meaningful split

I_JOB, I_INST, I_CPU, I_UTIL, I_MEM_AVG, I_MEM_MAX, I_READ, I_WRITE = 0, 2, 6, 7, 10, 11, 12, 13


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ctx = {r["job_name"]: r for r in
           csv.DictReader(open(os.path.join(DATA, "job_context.csv")))}
    rep = {r["job_name"]: r for r in
           csv.DictReader(open(os.path.join(DATA, "replay_jobs.csv")))}

    workers: dict[str, list] = collections.defaultdict(list)
    seen = 0
    with open(SENSOR) as f:
        for row in csv.reader(f):
            seen += 1
            if seen % 1000000 == 0:
                print(f"\r  {seen:,} rows, {len(workers):,} jobs", end="", flush=True)
            if len(row) <= I_WRITE or row[I_JOB] not in rep:
                continue
            u = _f(row[I_UTIL])
            if u is None:
                continue
            workers[row[I_JOB]].append((row[I_INST], u, _f(row[I_CPU]) or 0.0,
                                        _f(row[I_MEM_MAX]) or 0.0, _f(row[I_READ]) or 0.0,
                                        _f(row[I_WRITE]) or 0.0))
    print(f"\r  {seen:,} rows, {len(workers):,} jobs with usable workers")

    n_out = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_name", "n_ev", "n_ho", "ev_util_mean", "ev_util_max", "ev_util_std",
                    "ev_cpu_mean", "ev_mem_mean", "ev_mem_max", "ev_read", "ev_write",
                    "ho_util_mean", "roles", "quanta"])
        for jid, ws in workers.items():
            if len(ws) < MIN_WORKERS:
                continue
            ws.sort(key=lambda t: t[0])              # deterministic: sorted by instance id
            ev, ho = ws[0::2], ws[1::2]              # alternating, no worker on both sides
            eu = [x[1] for x in ev]
            w.writerow([jid, len(ev), len(ho),
                        f"{st.mean(eu):.3f}", f"{max(eu):.3f}",
                        f"{st.pstdev(eu):.3f}",
                        f"{st.mean(x[2] for x in ev):.3f}",
                        f"{st.mean(x[3] for x in ev):.3f}",
                        f"{max(x[3] for x in ev):.3f}",
                        f"{st.mean(x[4] for x in ev):.1f}",
                        f"{st.mean(x[5] for x in ev):.1f}",
                        f"{st.mean(x[1] for x in ho):.3f}",
                        ctx.get(jid, {}).get("roles", ""),
                        rep.get(jid, {}).get("quanta", "")])
            n_out += 1
    print(f"jobs with >={MIN_WORKERS} workers: {n_out:,} -> {OUT}")


if __name__ == "__main__":
    main()
