"""Extract the REAL per-job labels the Alibaba v2020 replay never carried (Exp 98).

Exp 96 showed the synthetic recipe made `tier` and deadline tightness the same random
variable (measured point-biserial -0.78). The Google 2011 trace fixes that by shipping two
independent real fields (priority, scheduling class). This is the same method applied to the
GPU trace we actually need: pull what v2020 really records and stop inventing `tier`.

    tier  <- REAL. A job whose instance carries a registered `workload` tag in
             pai_group_tag_table (bert / ctr / nmt / inception / graphlearn / ...) is a named,
             recurring production pipeline; an unlabelled job is ad-hoc. 12.2% of terminated
             GPU jobs, close to Google's 7.8% production share.

    tightness <- STILL SYNTHETIC, and this is the honest limitation. v2020's only per-job role
             field (`task_name`) is degenerate for this purpose: 691,792 jobs share one class
             against 23,412 and 57. There is no field that varies deadline urgency, so slack
             must still be drawn -- but now from its OWN stream, independent of tier, which is
             what removes the Exp 96 confound.

Also emitted because they are real and cheap to carry: gpu_type_spec (a hard placement
constraint, 1.3% of jobs), the role class, and gpu_type.

This writes a SIDE-CAR keyed by job_name. `replay_jobs.csv` is NOT rebuilt: every committed
v2020 tier depends on its exact window sampling, and regenerating it risks a silent drift that
would strand a month of results.

  .venv/bin/python data/build_v2020_labels.py
"""
from __future__ import annotations

import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "alibaba-gpu-v2020")
OUT = os.path.join(RAW, "job_labels.csv")

# pai_task_table.task_name -> tightness class. Kept (and emitted) so the degeneracy is visible
# in the data rather than asserted in a docstring; 0 = auxiliary, 1 = training, 2 = batch compute.
ROLE_CLASS = {"evaluator": 0, "TensorboardTask": 0, "ps": 1, "worker": 1, "tensorflow": 1,
              "PyTorchWorker": 1, "xComputeWorker": 2, "ReduceTask": 2}


def main() -> None:
    # pai_job_table: job_name, inst_id, user, status, start_time, end_time
    inst_of, user_of = {}, {}
    with open(os.path.join(RAW, "pai_job_table.csv")) as f:
        for r in csv.reader(f):
            if len(r) >= 3:
                inst_of[r[0]], user_of[r[0]] = r[1], r[2]

    # pai_group_tag_table: inst_id, user, gpu_type_spec, group, workload
    tag = {}
    with open(os.path.join(RAW, "pai_group_tag_table.csv")) as f:
        for r in csv.reader(f):
            if len(r) >= 5:
                tag[r[0]] = (r[2], r[3], r[4])

    # pai_task_table: job_name, task_name, inst_num, status, start, end, cpu, mem, plan_gpu, gpu_type
    roles: dict[str, set] = collections.defaultdict(set)
    gputype: dict[str, str] = {}
    with open(os.path.join(RAW, "pai_task_table.csv")) as f:
        for r in csv.reader(f):
            if len(r) >= 10 and r[3] == "Terminated" and float(r[8] or 0) > 0:
                roles[r[0]].add(r[1])
                gputype[r[0]] = r[9]

    rows, prod = [], 0
    for job, rs in roles.items():
        spec, group, workload = tag.get(inst_of.get(job, ""), ("", "", ""))
        is_prod = bool(workload)
        prod += is_prod
        rows.append({"job_name": job,
                     "tier": "prod" if is_prod else "besteffort",
                     "workload": workload,
                     "gpu_type_spec": spec,
                     "gpu_type": gputype.get(job, ""),
                     "role_class": max((ROLE_CLASS.get(x, 1) for x in rs), default=1)})
    rows.sort(key=lambda r: r["job_name"])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    wl = collections.Counter(r["workload"] for r in rows if r["workload"])
    rc = collections.Counter(r["role_class"] for r in rows)
    print(f"{len(rows):,} terminated GPU jobs -> {OUT}")
    print(f"  prod (has a workload tag): {prod:,} = {prod/len(rows):.2%}")
    print(f"  workloads: {dict(wl.most_common(8))}")
    print(f"  role_class: {dict(sorted(rc.items()))}  <- degenerate, see the module docstring")


if __name__ == "__main__":
    main()
