"""
Build the SECOND trace for the replay world (the Exp-43/44 external-validity caveat):
MIT Supercloud (TX-GAIA) scheduler log -> data/supercloud/replay_jobs.csv, the same
(job_name, arrival, dur, quanta) schema `pins/trace_replay.py` replays for Alibaba v2020.

Why Supercloud: a genuinely different regime — university HPC batch (Slurm, whole V100s,
median job ~2.1 h) vs Alibaba's cloud PAI (fractional GPUs, median ~18 min) — and the log
(`data/slurm-log.csv`) is already on disk from the forecast track.

Mapping (mirrors the v2020 build documented in trace_replay.py's module docstring):
  * keep state==3 (COMPLETED) jobs with allocated GPUs and a positive makespan,
  * arrival = time_start (the job's real start, like v2020's first-task start),
  * dur     = time_end - time_start (wall-clock, seconds),
  * quanta  = GPUs * 4 (whole-GPU allocations in quarter-GPU quanta; v2020's plan_gpu/25).
    GPUs come from TRES id 1002 in tres_alloc — id 1001 appears on only 6 rows and the
    1002 value distribution (1 and 2 dominate; TX-GAIA nodes carry 2 V100s) identifies it.
CAP_CLIP stays with the replayer. Durations are 14x v2020's, so the replayer pairs this
trace with a larger tick (TICK_S 900 vs 120) to keep median work ~9 ticks — same sim
regime, different world (see --trace in trace_replay.py).

Run:  .venv/bin/python data/build_supercloud_replay.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "slurm-log.csv")
OUT_DIR = os.path.join(HERE, "supercloud")
OUT = os.path.join(OUT_DIR, "replay_jobs.csv")
GPU_TRES = "1002"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, skipped = [], 0
    with open(SRC) as f:
        for r in csv.DictReader(f):
            tres = dict(kv.split("=") for kv in r["tres_alloc"].split(",") if "=" in kv)
            if r["state"] != "3" or GPU_TRES not in tres:
                continue
            start, end = int(r["time_start"]), int(r["time_end"])
            gpus = int(tres[GPU_TRES])
            if start <= 0 or end <= start or gpus <= 0:
                skipped += 1
                continue
            rows.append((start, end - start, gpus * 4, r["id_job"]))
    rows.sort(key=lambda x: x[:3])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_name", "arrival", "dur", "quanta"])
        for arr, dur, quanta, name in rows:
            w.writerow([name, arr, dur, quanta])
    durs = sorted(dur for _, dur, _, _ in rows)
    span_h = (rows[-1][0] - rows[0][0]) / 3600
    print(f"{len(rows)} completed GPU jobs -> {OUT} ({skipped} degenerate skipped)")
    print(f"dur median {durs[len(durs)//2]}s  p10/p90 {durs[len(durs)//10]}/"
          f"{durs[9*len(durs)//10]}s | span {span_h:.0f}h "
          f"(~{len(rows)/span_h:.0f} GPU jobs/h)")


if __name__ == "__main__":
    main()
