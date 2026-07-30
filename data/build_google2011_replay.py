"""Build a PINS replay CSV from the Google cluster-data 2011 trace (ClusterData2011_2).

The v2020 Alibaba replay gives real (arrival, duration, GPU demand) but NOTHING about
priority or deadlines, so `tier` and `deadline` were drawn from one synthetic uniform — the
confound Exp 96 measured (tier and deadline tightness were literally the same variable).
This trace carries **priority (0-11)** and **scheduling class (0-3)** per task, plus SUBMIT /
SCHEDULE / FINISH events, so tier stops being synthetic and the real queueing delay is
observable.

    https://github.com/google/cluster-data  (data: gs://clusterdata-2011-2, public over HTTPS)

Output columns, a superset of the Alibaba replay so `trace_replay.load_trace` keeps working:

    job_name, arrival, dur, quanta, priority, sched_class, real_wait

  * arrival  = first SUBMIT, seconds from trace start (trace time is microseconds)
  * real_wait= first SCHEDULE - first SUBMIT, seconds -- what Borg actually delivered
  * dur      = last FINISH - first SCHEDULE, seconds of RUN time (wait excluded)
  * quanta   = summed CPU request over the job's tasks, in QUANTUM units (below)
  * priority = max over tasks. Google's bands: 0-1 free, 2-8 normal, 9-10 production,
               11 monitoring. `tier = prod iff priority >= 9` is then a REAL label.

Only jobs with a complete SUBMIT -> SCHEDULE -> FINISH are emitted (the Alibaba extractor's
"Terminated" filter). Jobs whose SUBMIT is at time 0 already existed when the trace started
and are dropped: their arrival and wait are unknowable.

WHAT THIS TRACE IS NOT: CPU/memory only. The 2011 fleet predates GPU scheduling, so a result
measured here is a CPU-cluster result. Recorded at the top of the file so it cannot be
forgotten downstream.

  .venv/bin/python data/build_google2011_replay.py --parts 120
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "google2011")
OUT = os.path.join(HERE, "google-cluster-2011", "replay_jobs.csv")
URL = "https://storage.googleapis.com/clusterdata-2011-2/task_events/part-{:05d}-of-00500.csv.gz"

# 1 quantum := 1/128 of the largest machine's CPU. Chosen so the CLIP below retains ~80% of
# jobs whole, matching the Alibaba replay's quarter-GPU quantum + clip 8 convention rather
# than inventing a second one. Re-derive with --calibrate if the part count changes a lot.
QUANTUM = 1.0 / 128
CLIP = 8                     # same CAP_CLIP as trace_replay
EV_SUBMIT, EV_SCHEDULE, EV_FINISH = 0, 1, 4


def fetch(parts: int) -> None:
    os.makedirs(RAW, exist_ok=True)
    for i in range(parts):
        dest = os.path.join(RAW, os.path.basename(URL.format(i)))
        if not os.path.exists(dest):
            urllib.request.urlretrieve(URL.format(i), dest)


def aggregate() -> dict:
    """One record per job ID, folded over every task event of that job."""
    jobs: dict[str, dict] = collections.defaultdict(
        lambda: {"sub": None, "sch": None, "fin": None, "cpu": {}, "pri": 0, "sc": 0})
    for f in sorted(glob.glob(os.path.join(RAW, "part-*.csv.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                p = line.rstrip("\n").split(",")
                if len(p) < 10:
                    continue
                t, jid, tidx, ev = int(p[0]), p[2], p[3], int(p[5])
                j = jobs[jid]
                if ev == EV_SUBMIT and (j["sub"] is None or t < j["sub"]):
                    j["sub"] = t
                elif ev == EV_SCHEDULE and (j["sch"] is None or t < j["sch"]):
                    j["sch"] = t
                elif ev == EV_FINISH:
                    j["fin"] = max(j["fin"] or 0, t)
                if p[9]:
                    # per TASK, not summed over events: a task re-scheduled after eviction
                    # emits its request again and would otherwise be counted twice
                    j["cpu"][tidx] = float(p[9])
                if p[8]:
                    j["pri"] = max(j["pri"], int(p[8]))
                if p[7]:
                    j["sc"] = max(j["sc"], int(p[7]))
    return jobs


def rows(jobs: dict) -> list[dict]:
    out = []
    for jid, j in jobs.items():
        if not (j["sub"] and j["sch"] and j["fin"]) or not j["cpu"]:
            continue                       # incomplete lifecycle (or spans a part boundary)
        if j["fin"] <= j["sch"] or j["sch"] < j["sub"]:
            continue
        cpu = sum(j["cpu"].values())
        out.append({"job_name": f"g{jid}",
                    "arrival": j["sub"] // 1_000_000,
                    "dur": max(1, (j["fin"] - j["sch"]) // 1_000_000),
                    "quanta": min(CLIP, max(1, round(cpu / QUANTUM))),
                    "priority": j["pri"],
                    "sched_class": j["sc"],
                    "real_wait": (j["sch"] - j["sub"]) // 1_000_000})
    out.sort(key=lambda r: (r["arrival"], r["job_name"]))
    return out


def report(out: list[dict]) -> None:
    def q(key, p):
        a = sorted(r[key] for r in out)
        return a[int(len(a) * p)]
    pri = collections.Counter(r["priority"] for r in out)
    prod = sum(c for k, c in pri.items() if k >= 9) / max(len(out), 1)
    span = (out[-1]["arrival"] - out[0]["arrival"]) / 3600 if out else 0
    unclipped = sum(1 for r in out if r["quanta"] < CLIP) / max(len(out), 1)
    print(f"{len(out):,} jobs over {span:.1f} h")
    for k in ("dur", "quanta", "real_wait"):
        print(f"  {k:9s} p10 {q(k, .1):>6}  p50 {q(k, .5):>6}  p90 {q(k, .9):>6}  max {q(k, .999):>6}")
    print(f"  priority  {dict(sorted(pri.items()))}")
    print(f"  prod share (priority>=9) {prod:.1%}   below the clip {unclipped:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=120, help="task_events parts to use (of 500)")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        fetch(a.parts)
    out = rows(aggregate())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    report(out)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
