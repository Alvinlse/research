"""ElastiSim bench: Supercloud GPU jobs replayed through ElastiSim, five scheduler arms.

    .venv/bin/python -m pins.elastisim_bench build --day 157 --hours 12 --pool 256 --out runs/es_d157
    .venv/bin/python -m pins.elastisim_bench run --world runs/es_d157 --arm fcfs|firstfit|sjf|single|debate

World (whole-GPU HPC batch; jobs are rigid by default, MALLEABLE under --elastic-frac):
  * one ElastiSim node == one GPU (Supercloud jobs are 82% 1x1, 16% 1x2; a job needing g GPUs
    becomes a rigid job with num_nodes=g, 1 GPU per node), so the allocation unit is a GPU;
  * each job is ONE gpu task, flops = true_runtime * flops_per_gpu with pattern "uniform" (every
    rank gets the full value -> the job runs exactly its trace runtime on any node count);
  * walltime = Slurm timelimit (sentinels -> 0 = unlimited); the scheduler sees the REQUESTED
    limit, never the true runtime (same information as a real batch system).
Arms: fcfs (strict head-of-queue), firstfit (backfill without reservation), easy (FCFS + EASY
backfilling on requested limits, --est-default for undeclared), sjf (requested walltime, first
fit), tier_fcfs / tier_sjf (prod first, reserving), single (one LLM call decides the start list),
debate (proposer + critic). The validator only enforces feasibility; an invalid/empty LLM answer
falls back to firstfit. Arrivals are Slurm eligible times; `summary` prints the real waits of the
same jobs (p50/p90/mean/max/frac>1h) as calibration targets; `sweep` runs floors over many windows.
Metric names: sla10 = turnaround > 10x true runtime; ta_over_req_limit = turnaround > requested
limit (NOT an ElastiSim walltime kill -- only completed jobs are replayed, so none ever fires).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import functools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ES_ROOT = Path(os.environ.get("ELASTISIM_ROOT", "/import/gp-home.ciero/kimseng/elastisim"))
ES_BIN = ES_ROOT / "env/bin/elastisim"
SLURM_LOG = Path(__file__).resolve().parent.parent / "data/slurm-log.csv"
GPU_TRES = "1002"
FLOPS_PER_GPU = 1e12          # "1Tf" in the platform; flops = seconds * 1e12
TIMELIMIT_SENTINELS = {4294967295, 525600}   # Slurm "unlimited" and the 1-year default

# ---------------------------------------------------------------- build
@functools.lru_cache(maxsize=1)
def _trace_rows() -> tuple[dict, ...]:
    """The whole filtered trace, submit-sorted. Cached: a sweep builds many windows out of it."""
    rows = []
    with open(SLURM_LOG) as f:
        for r in csv.DictReader(f):
            if r["state"] != "3" or GPU_TRES + "=" not in r["tres_alloc"]:
                continue
            tres = dict(kv.split("=") for kv in r["tres_alloc"].split(",") if "=" in kv)
            try:
                su, s, e, tl = (int(r["time_submit"]), int(r["time_start"]),
                                int(r["time_end"]), int(r["timelimit"]))
                el = int(r["time_eligible"] or 0)
            except ValueError:
                continue
            if su <= 0 or s <= 0 or e <= s:
                continue
            # arrival = Slurm eligible time (after holds/dependencies), not submit: 6.2% of GPU jobs were
            # held, and a held job is not the scheduler's wait. real_wait is measured the same way.
            arr = min(max(el, su), s)
            rows.append(dict(jid=r["id_job"], submit=arr, dur=e - s, gpus=int(tres[GPU_TRES]),
                             timelimit_min=0 if tl in TIMELIMIT_SENTINELS else tl,
                             priority=r["priority"], partition=r["partition"], user=r["id_user"],
                             real_wait=s - arr, real_wait_from_submit=s - su))
    rows.sort(key=lambda x: x["submit"])
    return tuple(rows)


def load_window(day: int, hours: float, offset_h: float = 0, warmup_h: float = 0) -> list[dict]:
    """The measured window's jobs plus `warmup_h` of PRIOR arrivals, so no window starts on an empty
    cluster. Submits are shifted so warm-up occupies [0, warmup) and the measured window is
    [warmup, warmup+hours) -- ElastiSim needs submit_time >= 0. Warm-up jobs are flagged, but the
    flag is invisible to the scheduler (they are ordinary pending work); only scoring reads it."""
    rows = _trace_rows()
    t0 = rows[0]["submit"] + day * 86400 + int(offset_h * 3600)
    w0 = t0 - int(warmup_h * 3600)
    win = [dict(x) for x in rows if w0 <= x["submit"] < t0 + hours * 3600]   # copy: _trace_rows is cached
    for x in win:
        x["warmup"] = x["submit"] < t0
        x["submit"] -= w0
    return win


def _rand_starts(hours: float, seed: int, accepted: list[int]):
    """Random integer start hours, never within `hours` of an already-ACCEPTED start. `accepted` is
    read live, so overlap is checked against the windows we KEPT -- a rejected candidate blocks
    nothing. Overlapping windows share jobs, which would break the independence the 95% CI assumes."""
    rows = _trace_rows()
    hi = int((rows[-1]["submit"] - rows[0]["submit"]) / 3600 - hours)
    rng = random.Random(seed)
    while True:
        h = rng.randint(0, hi)
        if all(abs(h - a) >= hours for a in accepted):
            yield h


def real_stats(jobs: list[dict]) -> dict:
    """What the real scheduler did to these same jobs: the calibration targets (several, not one)."""
    w = sorted(j["real_wait"] for j in jobs)
    q = lambda p: w[int(p * (len(w) - 1))]
    return {"real_p50_wait_s": q(0.5), "real_p90_wait_s": q(0.9), "real_mean_wait_s": round(statistics.mean(w)),
            "real_max_wait_s": w[-1], "real_frac_wait_gt_1h": round(sum(x > 3600 for x in w) / len(w), 3)}


def build(day: int, hours: float, pool: int, out: Path, offset_h: float = 0, load: float = 0.0,
          warmup_h: float = 0, elastic_frac: float = 0.0, par_frac: float = 1.0,
          max_scale: float = 4.0, elastic_seed: int = 0, resize_points: int = 10) -> dict:
    """pool=0 with load>0 sizes the pool by offered load (exploratory path only: it makes load an
    OUTPUT, so windows cannot be stratified by it). The measured design passes a fixed pool and
    lets each window's own demand set its load. Meta describes the MEASURED window, not the warm-up."""
    out = out.resolve()
    (out / "in").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(exist_ok=True)
    jobs = load_window(day, hours, offset_h, warmup_h)
    scored = [j for j in jobs if not j["warmup"]]
    if not scored:
        return {}
    if pool <= 0:
        pool = max(8, round(sum(j["dur"] * j["gpus"] for j in scored) / 3600 / hours / load))
    # Elastic jobs are MALLEABLE: the scheduler picks their launch size and may resize them at
    # `resize_points - 1` equally spaced work boundaries. Dividing each iteration's work by the
    # number of points preserves the original total work exactly.
    # Amdahl anchored on the observed point -- a job seen at n0 nodes for `dur` seconds gets
    #     runtime(n) = dur * (s + (1-s)/n) / (s + (1-s)/n0)
    # so runtime(n0) == dur EXACTLY (the rigid replay is this model at the observed size) while
    # s=1 reproduces rigid at every size and s=0 gives linear speed-up. With ALL_RANKS ("total")
    # ElastiSim divides the work by the node count, so the flops FORMULA must be
    #     size(n) = n * FLOPS_PER_GPU * runtime(n) = a_par*n + a_ser,
    # which is linear in num_nodes -- exprtk substitutes num_nodes at assignment time.
    rng = random.Random(elastic_seed)
    elastic = {j["jid"]: rng.random() < elastic_frac for j in jobs}

    def amdahl(j):
        n0, s = max(1, j["gpus"]), par_frac
        A = FLOPS_PER_GPU * j["dur"] / (s + (1 - s) / n0)
        return {"a_par": repr(A * s), "a_ser": repr(A * (1 - s))}

    (out / "in/application_model.json").write_text(json.dumps({
        "phases": [{"iterations": 1, "scheduling_point": False, "tasks": [
            {"type": "gpu", "name": "train", "flops": "flops", "computation_pattern": "uniform"}]}]},
        indent=1))
    resize_points = max(1, resize_points)
    (out / "in/application_model_elastic.json").write_text(json.dumps({
        "phases": [{"iterations": resize_points, "scheduling_point": resize_points > 1, "tasks": [
            {"type": "gpu", "name": "train", "flops": "(a_par*num_nodes + a_ser)/resize_points",
             "computation_pattern": "total"}]}]}, indent=1))
    (out / "in/jobs.json").write_text(json.dumps({"jobs": [{
        **({"type": "malleable", "num_nodes_min": 1,
            "num_nodes_max": min(pool, max(1, int(max_scale * j["gpus"]))),
            "num_gpus_per_node_min": 1, "num_gpus_per_node_max": 1,
            "application_model": str(out / "in/application_model_elastic.json"),
            "arguments": amdahl(j) | {"resize_points": resize_points}}
           if elastic[j["jid"]] else
           {"type": "rigid", "num_nodes": j["gpus"], "num_gpus_per_node": 1,
            "application_model": str(out / "in/application_model.json"),
            "arguments": {"flops": j["dur"] * FLOPS_PER_GPU}}),
        "submit_time": j["submit"],
        "walltime": j["timelimit_min"] * 60,
        "attributes": {"jid": j["jid"], "gpus": j["gpus"], "req_min": j["timelimit_min"],
                       "req_nodes": j["gpus"],      # the size it really ran at -> the as_requested sizer
                       "elastic": int(elastic[j["jid"]]),
                       # Slurm multifactor score: 10000-11000 is fairshare/age noise, the +100000 bump is a QoS
                       # class (9.1% of GPU jobs) -> that bit is the tier; the raw number is not shown to arms
                       "tier": "prod" if int(j["priority"] or 0) >= 100000 else "batch",
                       "priority": j["priority"], "partition": j["partition"], "user": j["user"],
                       "_warmup": int(j["warmup"]),  # scoring-only: no arm ever reads it.
                       # int, not bool: ElastiSim's attribute mapper rejects bool (Utility.cpp:159)
                       "_true_dur": j["dur"], "_real_wait": j["real_wait"]}}
        for j in jobs]}, indent=0))
    (out / "in/platform.xml").write_text(f"""<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="Cluster" routing="Full">
    <zone id="BS_zone" routing="Full">
      <host id="Batch_system" speed="0Gf"><prop id="batch_system" value="true"/></host>
    </zone>
    <cluster id="Compute" prefix="Node_" radical="0-{pool - 1}" suffix="" speed="1Gf" bw="100Gbps" lat="50us">
      <prop id="num_gpus" value="1"/><prop id="flops_per_gpu" value="1Tf"/>
      <prop id="node_local_bb" value="false"/><prop id="pfs_targets" value="PFS"/>
    </cluster>
    <zone id="PFS_zone" routing="Full">
      <host id="PFS" speed="0Gf"><prop id="pfs_host" value="true"/></host>
    </zone>
    <link id="PFS_read" bandwidth="80GBps" latency="500us"/>
    <link id="PFS_write" bandwidth="50GBps" latency="500us"/>
    <zoneRoute src="PFS_zone" dst="Compute" gw_src="PFS" gw_dst="Node_Compute_router" symmetrical="NO">
      <link_ctn id="PFS_read"/></zoneRoute>
    <zoneRoute src="Compute" dst="PFS_zone" gw_src="Node_Compute_router" gw_dst="PFS" symmetrical="NO">
      <link_ctn id="PFS_write"/></zoneRoute>
  </zone>
</platform>
""")
    meta = {"day": day, "offset_h": offset_h, "hours": hours, "warmup_h": warmup_h, "pool": pool,
            "n_jobs": len(scored), "n_warmup": len(jobs) - len(scored),
            "elastic_frac": elastic_frac, "par_frac": par_frac, "max_scale": max_scale,
            "resize_points": resize_points,
            "n_elastic": sum(elastic[j["jid"]] for j in scored),
            "n_prod": sum(j["priority"].isdigit() and int(j["priority"]) >= 100000 for j in scored),
            "gpu_hours": sum(j["dur"] * j["gpus"] for j in scored) / 3600, **real_stats(scored)}
    meta["offered_load"] = round(meta["gpu_hours"] / hours / pool, 2)
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"built {out}: {len(scored)} scored jobs (+{len(jobs)-len(scored)} warm-up), "
          f"pool {pool} GPUs, offered load {meta['offered_load']}x")
    return meta


# ---------------------------------------------------------------- arms
# A MALLEABLE job has no num_nodes -- only num_nodes_min/max -- so every arm initially decides:
# which job to start, and how large to make it. The sizing rule is a separate axis from the
# ordering rule, and both baselines and LLM arms are sized by one of these, so no arm gets a lever
# the others lack. `as_requested` reproduces the rigid world exactly and is the control.
def _sizes(job):
    """(min, max) legal node counts. A rigid job has exactly one legal size."""
    n = getattr(job, "num_nodes", None)
    return (n, n) if n is not None else (job.num_nodes_min, job.num_nodes_max)


def _size(job, free, ctx, pending_n: int = 1) -> int:
    """Nodes to give this job right now, 0 if it cannot start. Rigid jobs are unaffected."""
    lo, hi = _sizes(job)
    f = len(free)
    if lo > f:
        return 0
    if lo == hi:
        return lo
    rule = ctx.get("sizer", "as_requested")
    if rule == "auto":
        # Per-TICK policy selection, the deterministic control a selector has to beat. The
        # window-level winner tracked offered load, so the instantaneous analogue is how deep the
        # queue is against what is free right now. Above the threshold share capacity, below it
        # honour what the owner asked for.
        if ctx.get("switch_on") == "load":
            pressure = ctx.get("pending_demand", pending_n) / max(1, ctx.get("pool_n", 1))
        else:
            pressure = pending_n / max(1, f)
        rule = "adaptive" if pressure >= ctx.get("switch_at", 1.0) else "as_requested"
        ctx["switch_adaptive"] = ctx.get("switch_adaptive", 0) + (rule == "adaptive")
        ctx["switch_calls"] = ctx.get("switch_calls", 0) + 1
    if rule == "greedy":                       # take the most that fits
        return min(hi, f)
    if rule == "adaptive":                     # share free capacity with everyone else waiting
        return max(lo, min(hi, f, f // max(1, pending_n)))
    # as_requested: demand exactly the size it really ran at, and WAIT if it is not free -- this is
    # the control, so it must behave identically to the rigid world (see the runtime==trace check).
    req = max(lo, min(hi, int(job.attributes.get("req_nodes", lo))))
    return req if req <= f else 0


def _fit(job, free, ctx, pending_n: int = 1):
    return _size(job, free, ctx, pending_n) > 0


def _start_at(job, free, ctx, k: int):
    """Start at an explicitly chosen size (the LLM arms). _validate has already made k legal."""
    job.assign(free[:k]); del free[:k]
    ctx.setdefault("sizes", {})[job.identifier] = k
    ctx.setdefault("size_history", {}).setdefault(job.identifier, []).append((ctx["now"], k))
    if getattr(job, "num_nodes", None) is None:
        job.assign_num_gpus_per_node(1)


def _start(job, free, ctx, pending_n: int = 1):
    k = _size(job, free, ctx, pending_n)
    job.assign(free[:k]); del free[:k]
    ctx.setdefault("sizes", {})[job.identifier] = k
    ctx.setdefault("size_history", {}).setdefault(job.identifier, []).append((ctx["now"], k))
    if getattr(job, "num_nodes", None) is None:      # elastic: GPUs per node is ours to set too
        job.assign_num_gpus_per_node(1)              # one node == one GPU in this world


def _resize_malleable(job, free, pending, ctx) -> int:
    """Resize one running job at an ElastiSim scheduling point.

    Queue pressure shrinks it to its legal minimum so waiting work can run; an empty queue lets it
    expand into idle GPUs. ElastiSim pauses the job at the scheduling point and applies the changed
    node set before its next work chunk. Return the signed GPU change for accounting/tests.
    """
    lo, hi = _sizes(job)
    before = len(job.assigned_nodes)
    last = ctx.setdefault("last_resize", {}).get(job.identifier, float("-inf"))
    if ctx.get("now", 0) - last < ctx.get("resize_cooldown", 0):
        ctx["resize_cooldown_blocks"] = ctx.get("resize_cooldown_blocks", 0) + 1
        return 0
    target = lo if pending else min(hi, before + len(free))
    budget = max(0, ctx.get("resize_budget", CORRECT_BUDGET))
    target = max(before - budget, min(before + budget, target))
    if target < before:
        job.remove(job.assigned_nodes[target:])
    elif target > before:
        job.assign(free[:target - before])
    delta = target - before
    if delta:
        ctx["resize_events"] = ctx.get("resize_events", 0) + 1
        ctx["resized_gpus"] = ctx.get("resized_gpus", 0) + abs(delta)
        ctx.setdefault("sizes", {})[job.identifier] = target
        ctx.setdefault("size_history", {}).setdefault(job.identifier, []).append((ctx["now"], target))
        ctx["last_resize"][job.identifier] = ctx.get("now", 0)
    return delta


def _near_limit_protection_proven(job, now) -> bool:
    """The zero-LLM control for the only structured resize exception currently permitted."""
    lim = int(job.attributes.get("req_min", 0)) * 60
    elapsed = max(0, int(now - job.start_time))
    return job.attributes.get("tier") == "prod" and lim > 0 and elapsed * 100 >= lim * 80


def _resize_with_deterministic_protection(job, free, pending, ctx) -> int:
    """Protect a near-limit production job; otherwise execute the ordinary resize baseline."""
    lo, _ = _sizes(job)
    if pending and len(job.assigned_nodes) > lo and _near_limit_protection_proven(job, ctx["now"]):
        ctx["protected_events"] = ctx.get("protected_events", 0) + 1
        return 0
    return _resize_malleable(job, free, pending, ctx)


def arm_fcfs(pending, free, ctx):
    for job in pending:
        if not _fit(job, free, ctx, len(pending)):
            break
        _start(job, free, ctx, len(pending))


def arm_firstfit(pending, free, ctx):
    for job in pending:
        if _fit(job, free, ctx, len(pending)):
            _start(job, free, ctx, len(pending))


def arm_sjf(pending, free, ctx):
    """NOT shortest-job-first: requested walltime is ~uncorrelated with true runtime on this trace
    (487 jobs share 9 distinct values; the one significant Spearman is -0.73). Label it
    'requested-walltime ordering' in every write-up. See arm_declared_first for the decomposition."""
    key = lambda j: (int(j.attributes["req_min"]) or 10 ** 9, j.submit_time)
    arm_firstfit(sorted(pending, key=key), free, ctx)


def arm_declared_first(pending, free, ctx):
    """Control for arm_sjf: keeps ONLY its declared-before-undeclared split, FCFS within each group.
    firstfit -> declared_first isolates what the declared/undeclared bit is worth; declared_first ->
    sjf isolates what requested-walltime ORDERING adds on top. All three route through arm_firstfit,
    so packing (skip a non-fitting job and keep scanning) is held constant -- arm_fcfs does not, and
    comparing against it confounds ordering with head-of-line bypass."""
    key = lambda j: (int(j.attributes["req_min"]) == 0, j.submit_time)
    arm_firstfit(sorted(pending, key=key), free, ctx)


def arm_easy(pending, free, ctx):
    """FCFS + EASY backfilling (Lifka 1995): the queue head gets a reservation at the shadow time
    (earliest moment enough GPUs are expected free, from running jobs' ESTIMATED ends); a later job may
    backfill only if it will finish before the shadow time or fits in the GPUs the head will not need.
    Estimates = requested walltime; undeclared (sentinel) limits use ctx['est_default'] -- a stated site
    default, because with an infinite estimate the shadow time is infinite and EASY collapses to first-fit."""
    now, est = ctx["now"], ctx["est"]
    i = 0
    while i < len(pending) and _fit(pending[i], free, ctx, len(pending)):
        _start(pending[i], free, ctx, len(pending)); i += 1
    if i >= len(pending):
        return
    head = pending[i]
    st = ctx.setdefault("easy_stats", {"blocked": 0, "shadow_now": 0, "backfilled": 0, "no_free": 0})
    st["blocked"] += 1
    if not free:
        st["no_free"] += 1
    avail, shadow, extra = len(free), float("inf"), 0
    for t_end, k in sorted((max(now, j.start_time + est(j)), len(j.assigned_nodes)) for j in ctx["running"]):
        avail += k
        if avail >= _sizes(head)[0]:
            shadow, extra = t_end, avail - _sizes(head)[0]
            break
    st["shadow_now"] += shadow <= now
    for j in pending[i + 1:]:
        if not _fit(j, free, ctx, len(pending)):
            continue
        if now + est(j) <= shadow:
            _start(j, free, ctx, len(pending)); st["backfilled"] += 1
        elif _sizes(j)[0] <= extra:
            k = _size(j, free, ctx, len(pending)); _start(j, free, ctx, len(pending)); extra -= k; st["backfilled"] += 1


def _prod_first(j):
    return j.attributes.get("tier") != "prod"


def _prod_reserving(order, free, ctx):
    """Prod jobs in order, strictly: a prod job that does not fit HOLDS the free GPUs (no batch backfill),
    otherwise 1-GPU batch jobs grab every single free GPU and a 2-GPU prod job never assembles a pair.
    Once every prod job is placed or the head prod job is blocked, batch jobs first-fit the remainder."""
    for job in order:
        if job.attributes.get("tier") == "prod":
            if not _fit(job, free, ctx, len(order)):
                return
            _start(job, free, ctx, len(order))
    for job in order:
        if job.attributes.get("tier") != "prod" and _fit(job, free, ctx, len(order)):
            _start(job, free, ctx, len(order))


def arm_tier_fcfs(pending, free, ctx):     # prod first (reserving), FCFS within tier
    _prod_reserving(sorted(pending, key=lambda j: (_prod_first(j), j.submit_time)), free, ctx)


def arm_tier_sjf(pending, free, ctx):      # prod first (reserving), requested-walltime SJF within tier
    key = lambda j: (_prod_first(j), int(j.attributes["req_min"]) or 10 ** 9, j.submit_time)
    _prod_reserving(sorted(pending, key=key), free, ctx)


# Both roles must describe the SAME world. Before this was factored out, the critic was never
# told jobs were malleable and was asked for bare ids, so it silently reverted every size the
# proposer chose back to the requested one (transcript: proposer emitted [id,gpus] pairs on
# 339/348 decisions, the critic on 23/348) -- it was deleting half the decision, not reviewing it.
WORLD = ("A job shown as `gpus=N` is rigid and needs exactly N GPUs for its whole run. A job shown "
         "as `gpus=LO-HI` is MALLEABLE: it starts at a size in that range and may be resized at "
         "later work boundaries; `asked=` is the size its owner originally requested. Scaling is sublinear -- "
         "doubling a job's GPUs does NOT halve its runtime, so a large allocation costs more machine "
         "time than it saves, while too small an allocation risks the job exceeding its requested "
         "walltime and being killed. You only know each job's REQUESTED walltime, never its true "
         "runtime.")
REPLY = ("Reply with JSON only: {\"start\": [[job id, gpus to give it], ...] in start order, "
         "\"why\": \"one line\"}. A bare job id means: use `asked`.")

SYSTEM = ("You are the batch scheduler of a GPU cluster. " + WORLD +
          " Goal: minimise mean waiting time and bounded slowdown without wasting GPU-hours. " + REPLY)
# The critic is NOT asked to re-check capacity or idle GPUs: the validator already enforces both,
# and asking for them drags the answer back toward plain first-fit (measured: debate matched
# first-fit on 17.2% of decisions against a single model's 5.1%). It is asked for the judgements a
# second opinion can actually add -- sizing, and who is being made to wait.
CRITIC = ("You are a second scheduler reviewing a colleague's start list for the same queue. " +
          WORLD +
          " Feasibility is already guaranteed by a downstream validator, so do NOT spend your answer "
          "re-checking that the list fits. Judge instead: (a) is each job's SIZE right -- would a "
          "smaller allocation serve the queue better, or is one so small it risks a walltime kill; "
          "(b) is any long-waiting or high-priority job being passed over. Keep the colleague's list "
          "where it is sound and change only what you can justify; if you would change nothing, "
          "return it unchanged. " + REPLY)
TIER_NOTE = (" Jobs carry tier=prod (operator-granted high-priority QoS, must not be starved) or tier=batch; "
             "protect prod jobs' waiting time first, then optimise the rest.")
# --- three-role negotiation ------------------------------------------------------------------
# The debate arm gives both agents the SAME objective and the SAME information, so the second call
# is theatre: measured on the stress window it overrode 329/349 proposals and cost 17% of mean wait.
# These roles are ASYMMETRIC on purpose. In an elastic world the conflict is real -- demand wants a
# job large (it finishes sooner), supply wants it small (sublinear scaling makes a big allocation
# 2.5x less efficient per device-hour at s=0.7) -- so there is something to negotiate about.
# NOTE the supply objective is device-hours per completed job, NOT utilisation: the occupancy-
# maximising rule measured worst on every service axis and held the highest occupancy in 12/12
# windows, so an advocate told to "keep GPUs busy" would be arguing for the worst known policy.
DEMAND = ("You represent the JOBS WAITING in the queue. " + WORLD +
          " Your objective is to minimise how long jobs wait and how badly they are slowed down. "
          "Argue for the start list and sizes that serve the queue best; a larger allocation is "
          "worth arguing for when it genuinely finishes a job sooner, and a job that has waited a "
          "long time deserves to start. You are an ADVOCATE, not the decider -- a referee will "
          "weigh your proposal against the cluster's. " + REPLY)
SUPPLY = ("You represent the CLUSTER's capacity. " + WORLD +
          " Your objective is to minimise the device-hours spent per completed job, and to keep "
          "capacity free for work that has not arrived yet. Because scaling is sublinear, a large "
          "allocation buys a little speed for a lot of capacity. Keeping devices BUSY is explicitly "
          "not your goal -- completing work cheaply is. Argue for the start list and sizes that "
          "waste least. You are an ADVOCATE, not the decider -- a referee will weigh your proposal "
          "against the queue's. " + REPLY)
REFEREE = ("You are the referee. Two advocates have read the same queue as you: one speaks for the "
           "waiting jobs and wants short waits, the other speaks for the cluster and wants each job "
           "completed for the fewest device-hours. " + WORLD +
           " Both are partial and both may overstate. Decide the final start list and sizes "
           "yourself: take what is right from each, reject special pleading, and say in one line "
           "which side you followed and why. " + REPLY)

# --- correction arm: the LLM corrects a deterministic anchor, it never allocates -------------
# Ported from pins/correction_signed.py, whose governing rule is "nobody generates an allocation
# from scratch": deterministic code decides, the agents may only propose a SIGNED correction, and
# an unusable answer leaves the anchor standing. That last property is why this arm exists here.
# The from-scratch arms (single/debate/negotiate) fall back to firstfit and can therefore finish
# WORSE than the heuristic they are meant to improve -- measured: debate 89,979s against
# as_requested's 66,908s on the stress window. Anchored on as_requested (the best sizing rule over
# 12 windows), a rejected correction costs nothing, so the arm can only help or be neutral.
CORRECT_BUDGET = 6      # GPUs that may change hands, measured as max(gained, lost): L1 double-
                        # counts a transfer, and a transfer is exactly what restraint needs.
CORRECTOR = ("A deterministic scheduler has already chosen which jobs to start and how large to "
             "make each one. It is competent and you should usually leave it alone. " + WORLD +
             " You may propose a SIGNED correction to its plan, and nothing else -- you never "
             "produce a schedule. Reply with JSON only: "
             '{"changes": {"<job id>": <signed integer GPUs, negative to take some or all away>}, '
             '"hold_free": <integer GPUs to leave deliberately unallocated, 0 if none>, '
             '"why": "one line"}. '
             "Correct only what you can justify: a job sized so small it risks exceeding its "
             "requested walltime, a job given more than sublinear scaling can repay, a job that has "
             "waited far too long, or capacity worth holding for imminent higher-priority work. "
             "Empty changes with hold_free 0 is the correct answer for an ordinary scene.")


def _anchor_plan(pending, free, ctx):
    """What the deterministic sizer would do right now: the plan the corrector may edit."""
    plan, used = [], 0
    for j in pending:
        k = _size(j, free[used:], ctx, len(pending))
        if k and used + k <= len(free):
            plan.append([j, k]); used += k
    return plan


def _anchor_line(plan, free) -> str:
    """State the baseline AND the capacity arithmetic. Without `unsold` and each job's headroom the
    model has to subtract the anchor from free_gpus and accumulate its own deltas to stay feasible
    -- which is exactly the arithmetic the signed contract exists to take away from it."""
    used = sum(k for _, k in plan)
    rows = "; ".join(f"job {j.identifier} -> {k} GPU (may go up to {_sizes(j)[1]})"
                     for j, k in plan[:PACKET_CAP]) or "nothing"
    return (f"the deterministic baseline for this moment: {rows}\n"
            f"it allocates {used} of the {len(free)} free GPUs, leaving {len(free) - used} unsold.")


def _apply_correction(plan, ans, pending, free):
    """Delegate to pins.correction_signed: the same parser, the same max(gained,lost) disruption
    budget, and the same funding rule (revoke from the least-valued units, never from a
    beneficiary, never below a floor unless the decision explicitly reduced that job). Reusing it
    rather than reimplementing keeps the two studies on ONE contract -- an earlier local version
    lacked the donor logic and funded grants by simply filling until capacity ran out.

    A non-empty violations list means the decision is REJECTED and the anchor stands, which is why
    this arm cannot finish worse than its deterministic baseline."""
    from pins.correction_signed import _parse_decision, apply_signed
    by_id = {str(j.identifier): j for j in pending}
    # every pending job is in the allocation, at 0 if the anchor does not start it -- otherwise a
    # correction that PROMOTES a waiting job would be rejected as naming an unknown id
    alloc = {jid: 0 for jid in by_id}
    for j, k in plan:
        alloc[str(j.identifier)] = k
    floors = {jid: _sizes(j)[0] for jid, j in by_id.items()}
    # least-valued donates first: batch before prod, then SHORTEST-waiting. submit_time ascending
    # is oldest-first, i.e. longest-waiting -- the opposite of what we want -- so negate it.
    ranking = sorted((float(j.attributes.get("tier") == "prod") * 1e9 - j.submit_time, jid)
                     for jid, j in by_id.items())
    dec = _parse_decision(ans)
    if dec.get("_source") == "fallback":     # nothing parseable came back
        return plan, "no parse"
    # Normalise every delta so the resulting size is within [0, hi] BEFORE funding: apply_signed
    # has no per-job maximum, so an over-large ask would otherwise be costed against capacity and
    # the disruption budget as phantom GPUs that a later clip removes.
    dec["changes"] = {jid: n for jid, n in (
        (jid, max(-alloc.get(jid, 0), min(n, _sizes(by_id[jid])[1] - alloc.get(jid, 0))))
        for jid, n in dec["changes"].items() if jid in by_id) if n}
    if not dec["changes"] and not dec["hold_free"]:
        return plan, "no change"             # the right answer for an ordinary scene
    new_alloc, viol = apply_signed(alloc, dec, len(free),
                                   ranking=ranking, floors=floors, budget=CORRECT_BUDGET)
    if viol:
        return plan, "; ".join(viol)[:40]
    out = []
    for jid, k in new_alloc.items():
        j = by_id[jid]
        lo, hi = _sizes(j)
        if k >= lo:                       # below its legal minimum means "not started"
            out.append((j, min(k, hi)))
    return out, "applied"


# --- combined arm: negotiation that emits a CORRECTION, not an allocation --------------------
# This is pins/correction_signed.py's winning shape ported to the elastic world: two reviewers
# with opposed interests argue, a referee rules, and the ruling is a signed EDIT to a
# deterministic plan rather than a plan of its own. It answers the multi-agent question (unlike
# `correct`, which is single-agent) while keeping the property `negotiate` lacks -- an unusable
# ruling leaves the anchor standing. That is safety against unusable actions, not a guarantee of improvement --
        # an accepted correction can still hurt, and it perturbs every later queue state.
# Each role is told it does NOT produce an allocation, and that the anchor's ordinary sizing is
# already correct: an advocate should speak only where it has a reason the sizer could not see.
_ANCHOR_RULE = (" A deterministic scheduler has ALREADY chosen which jobs to start and how large "
                "to make each one, sizing each job as its owner requested. That baseline is "
                "competent. Ordinary urgency, an ordinary size and ordinary scarcity are ALREADY "
                "in it and are NOT reasons to change anything. You do NOT produce a schedule.\n"
                # correction_signed's contract, dropped in the first port: without it the model
                # tries to balance the books itself and asks for GPUs that do not exist (7 of 11
                # smoke rulings were unfundable).
                "You do NOT need to make the numbers add up. Code funds every grant: it decides "
                "which jobs give up capacity, refuses anything it cannot pay for, and leaves the "
                "baseline standing if your judgement cannot be funded. Say what SHOULD change and "
                "why; do not do the arithmetic.")
DEMAND_SIGNED = ("You are the DEMAND-side reviewer, speaking for the waiting jobs. " + WORLD +
                 _ANCHOR_RULE +
                 " Your only question is: which jobs should get MORE than the baseline gave them, "
                 "or LESS (possibly nothing)? Answer non-zero only with a reason the sizer could "
                 "not see -- a job sized so small it risks exceeding its requested walltime, a job "
                 "that has waited far longer than its peers, or a job given more than sublinear "
                 "scaling can repay. Reply with JSON only: "
                 '{"changes": {"<job id>": <signed integer GPUs>}, "why": "one line"}. '
                 "An empty changes object is the correct answer for an ordinary queue.")
SUPPLY_SIGNED = ("You are the SUPPLY-side reviewer, speaking for the cluster. " + WORLD +
                 _ANCHOR_RULE +
                 " Your only question is: must some capacity be left UNALLOCATED this round, and "
                 "how much? Answer non-zero only when holding capacity back is worth more than "
                 "using it now -- imminent higher-priority work, or allocations so oversized that "
                 "sublinear scaling wastes the machine. Keeping devices BUSY is explicitly not "
                 "your goal; completing work cheaply is. Reply with JSON only: "
                 '{"hold_free": <0 or a positive integer>, "why": "one line"}. '
                 "0 is the correct answer for an ordinary queue.")
REFEREE_SIGNED = ("You are the REFEREE. " + WORLD + _ANCHOR_RULE +
                  " Two reviewers have proposed changes to the baseline: one speaks for the "
                  "waiting jobs, one for the cluster. Both may overstate. You decide only how much "
                  "of what they proposed is justified -- you may adopt a change in full, shrink it, "
                  "or reject it. Reply with JSON only: "
                  '{"changes": {"<job id>": <signed integer GPUs>}, '
                  '"hold_free": <integer, 0 if none>, "why": "one line"}. '
                  "Empty changes with hold_free 0 is the correct answer when neither reviewer has "
                  "made a case.")

# --- neg_v2: asymmetric INFORMATION, rankings and ceilings, verifiable reasons -------------
# Built from what the earlier arms measured. (1) demand and supply read the same packet and agreed
# half the time, so they now see DIFFERENT views: demand the job-level view, supply the cluster-
# level view (running jobs and their elapsed time, unsold capacity, true queue depth). (2) 74% of
# signed rulings were unfundable because the model asked for GPUs that did not exist, so no role
# emits an amount any more: demand emits a RANKING, supply emits a CEILING, the referee decides how
# far down the ranking to go within the ceiling, and code sizes every grant -- there is nothing to
# be "short of". (3) "imminent higher-priority work" was recited as a blanket licence because the
# packet could never refute it, so every permitted reason now names a field the packet carries.
_V2_WORLD = ("Jobs are GPU jobs. A job shown as gpus=N is rigid. A job shown as gpus=LO-HI is "
             "MALLEABLE: it starts within that range and may resize at later work boundaries; "
             "asked= is what its owner requested. Scaling "
             "is sublinear, so more GPUs finish a job somewhat sooner at a proportionally larger "
             "cost in machine time; too few risk the job exceeding its requested walltime and being "
             "killed. True runtimes are unknown to everyone. A deterministic baseline has ALREADY "
             "planned this round, sizing each started job as its owner asked. You never produce a "
             "schedule and never state amounts of GPUs -- code does all sizing arithmetic.")
DEMAND_V2 = ("You speak for the WAITING JOBS. " + _V2_WORLD +
             " You see the queue only: you do not know how much capacity is free. Rank the jobs "
             "that most deserve MORE than the baseline gives them (a larger size, or to be started "
             "when the baseline left them waiting), best case first. Use only these reasons, each "
             "checkable in the list: waited_x_median is high (the job has waited far longer than its "
             "peers); tier=prod is waiting while batch jobs are being started; a malleable job is "
             "started well below asked=; a malleable job is still WAITING and could be started "
             "sooner if it accepted fewer GPUs than asked= (rank it if its wait justifies a "
             "smaller start). Leave out any job with no such reason. Reply with JSON "
             'only: {"rank": [job ids, most deserving first], "why": "one line"}. An empty rank '
             "is the correct answer for an ordinary queue.")
SUPPLY_V2 = ("You speak for the CLUSTER. " + _V2_WORLD +
             " You see the cluster only: what is running, for how long, and against what limit; "
             "how much of the free pool the baseline commits and how much is unsold; and how deep "
             "the queue is. You do not see individual jobs' pleading. Decide the CEILING: the most "
             "GPUs that should be committed to new starts this round. Commit less than the baseline "
             "only for a reason checkable here -- a running job is near its limit or has run far "
             "longer than others and will free enough soon for something the baseline could not "
             "fit, or the queue is so much deeper than shown that a burst is due. Commit more than "
             "the baseline when GPUs sit unsold with a deep queue. Reply with JSON only: "
             '{"commit_max": <integer GPUs>, "why": "one line"}. Restating the baseline\'s own '
             "commitment is the correct answer for an ordinary cluster.")
REFEREE_V2 = ("You are the REFEREE. " + _V2_WORLD +
              " The queue's advocate has ranked which jobs deserve more; the cluster's advocate has "
              "set a ceiling on GPUs to commit. Each saw only its own side. You see the baseline "
              "and both. Decide how far down the ranking to honour and what ceiling to apply: adopt "
              "each in full, cut it back, or reject it. Favouring a WAITING job starts it with "
              "whatever fits under the ceiling, which may be fewer GPUs than it asked for. Reply with JSON only: "
              '{"favour": [job ids to give more, in order, possibly empty], '
              '"commit_max": <integer GPUs, or -1 to keep the baseline\'s commitment>, '
              '"why": "one line naming which advocate you followed"}.')

DEMAND_RESIZE_V2 = ("You represent WAITING jobs reviewing a deterministic live resize. " + _V2_WORLD +
                    " Code already plans to shrink the current running job just enough to unblock "
                    "work. State whether a waiting production job is blocked; ordinary batch demand "
                    "is not an exception. Reply with JSON only: "
                    '{"waiting_prod_blocked": <true or false>, "evidence_job": <job id or -1>}.')
SUPPLY_RESIZE_V2 = ("You review whether the CURRENT running job needs protection from a deterministic "
                    "shrink. " + _V2_WORLD + " Protection is exceptional: use current_prod_near_limit "
                    "only when the packet proves the current job is tier=prod and at least 80% through "
                    "its requested walltime. Otherwise use none. Reply with JSON only: "
                    '{"reason": "current_prod_near_limit" or "none", "evidence_job": <current id>}.')
REFEREE_RESIZE_V2 = ("You referee an exception to a deterministic live resize. " + _V2_WORLD +
                     " The baseline shrink HAPPENS unless the current job has the single permitted "
                     "protection: reason=current_prod_near_limit, supported by its packet fields. "
                     "Waiting production work argues FOR the shrink, never for protection. Do not "
                     "invent another reason. Reply with JSON only: "
                     '{"protect_current": <true or false>, "reason": "current_prod_near_limit" or "none", '
                     '"evidence_job": <current id>}.')
TEXT_RESIZE_SINGLE = ("You review a deterministic live resize using an operator or owner note "
                      "that is not represented in the numerical scheduler fields. The note is a "
                      "synthetic counterfactual for this experiment, but treat its contents as "
                      "authoritative. Choose protect only when it explicitly says shrinking the "
                      "CURRENT job would invalidate work, lose non-recoverable progress, or violate "
                      "an external operational constraint. Choose shrink when it explicitly permits "
                      "or requests releasing GPUs. Choose default for status, convenience, accounting, "
                      "or any note without a scheduling instruction. Reply with JSON only: "
                      '{"action": "protect|shrink|default", "evidence_job": "<trace job id>", '
                      '"reason": "one short evidence phrase"}.')


# The text channel's three roles. The advocates are opposed on purpose: one is paid to release the
# GPUs, the other to keep them, and only the referee sees both notes and the packet together.
TEXT_RESIZE_DEMAND = ("You represent the WAITING jobs. A note from the CURRENT running job's owner "
                      "is quoted below and code already plans to shrink that job to unblock the "
                      "queue. Argue whether the note really forbids the shrink, or whether it is "
                      "status, convenience or accounting text that should not stop it. Be sceptical: "
                      "vague urgency is not an operational constraint. Reply with JSON only: "
                      '{"note_blocks_shrink": <true or false>, "why": "one short phrase"}.')
TEXT_RESIZE_SUPPLY = ("You represent the CURRENT running job. Its owner or operator note is quoted "
                      "below and code plans to shrink it. State whether the note names work that "
                      "shrinking would destroy or an external constraint it would break, or whether "
                      "it instead PERMITS releasing GPUs. Do not invent a constraint the note does "
                      "not state. Reply with JSON only: "
                      '{"claim": "protect|shrink|default", "why": "one short phrase"}.')
TEXT_RESIZE_REFEREE = ("You referee a deterministic live resize on the strength of an operator or "
                       "owner note that the numerical fields do not represent. The note is a "
                       "synthetic counterfactual for this experiment, but treat its contents as "
                       "authoritative. Two advocates have argued opposite sides and each saw only "
                       "its own; you see both and the packet. Choose protect only when the note "
                       "explicitly says shrinking the CURRENT job would invalidate work, lose "
                       "non-recoverable progress, or violate an external operational constraint. "
                       "Choose shrink when it explicitly permits or requests releasing GPUs. Choose "
                       "default for status, convenience, accounting, or any note without a "
                       "scheduling instruction. An advocate's confidence is not evidence. "
                       "Reply with JSON only: "
                       '{"action": "protect|shrink|default", "evidence_job": "<trace job id>", '
                       '"reason": "one short evidence phrase naming which advocate you followed"}.')


def _packet_demand(pending, now, plan=()):
    """The job-level view, including baseline status but no capacity arithmetic."""
    waits = sorted(max(0.0, now - j.submit_time) for j in pending)
    med = waits[len(waits) // 2] if waits else 1.0
    lines = [f"now={int(now)}s pending={len(pending)} (showing {min(len(pending), PACKET_CAP)}, in no "
             f"particular order). queue median wait = {int(med)}s"]
    shown = list(pending[:PACKET_CAP]); random.Random(int(now)).shuffle(shown)
    base = {j.identifier: k for j, k in plan}
    for j in shown:
        a = j.attributes; lo, hi = _sizes(j); w = max(0.0, now - j.submit_time)
        size = f"gpus={lo}" if lo == hi else f"gpus={lo}-{hi} asked={a.get('req_nodes', lo)}"
        status = f"start@{base[j.identifier]}" if j.identifier in base else "WAIT"
        lines.append(f"id={j.identifier} {size} baseline={status} tier={a.get('tier', '?')} waited_s={int(w)} "
                     f"waited_x_median={w / max(1.0, med):.1f}")
    return "\n".join(lines)


def _packet_supply(plan, pending, free, ctx, now):
    """The cluster-level view. Running jobs show elapsed time and requested limit -- what a real
    system knows -- never true runtime. Requested walltimes are 100-600x over-stated on this
    trace, so the release picture is honest but weak; that is a property of the data."""
    used = sum(k for _, k in plan)
    run = sorted(ctx.get("running", []), key=lambda j: j.start_time)
    lines = [f"now={int(now)}s free_gpus={len(free)} queue_depth={len(pending)} "
             f"(the queue's advocate saw at most {PACKET_CAP} of them)",
             f"baseline commits {used} GPUs to {len(plan)} new starts. UNSOLD={len(free) - used} GPUs remain free after it.",
             f"running jobs: {len(run)}, holding {sum(len(j.assigned_nodes) for j in run)} GPUs"]
    for j in run[:PACKET_CAP]:
        el = int(now - j.start_time); lim = int(j.attributes.get("req_min", 0)) * 60
        lines.append(f"  running id={j.identifier} gpus={len(j.assigned_nodes)} elapsed_s={el} "
                     + (f"limit_s={lim} ({100 * el // max(1, lim)}% of limit)" if lim else "limit=UNLIMITED"))
    return "\n".join(lines)


def _decision_ids(ans, field, valid):
    """Parse an ordered id list while preserving order and ignoring hallucinated jobs."""
    out = []
    for raw in (ans.get(field, []) if isinstance(ans, dict) else []):
        try:
            jid = int(str(raw).strip().removeprefix("id="))
        except (TypeError, ValueError):
            continue
        if jid in valid and jid not in out:
            out.append(jid)
    return out


def _llm_resize_decide(job, pending, free, ctx) -> int:
    """Review an exception to the deterministic resize; invalid/weak debate changes nothing."""
    from pins.correction import _ask
    now = ctx["now"]
    last = ctx.setdefault("last_resize", {}).get(job.identifier, float("-inf"))
    if now - last < ctx.get("resize_cooldown", 0):
        ctx["resize_cooldown_blocks"] += 1
        return 0
    lo, _ = _sizes(job)
    if not pending or len(job.assigned_nodes) <= lo:
        return _resize_malleable(job, free, pending, ctx)

    waiting_prod = any(j.attributes.get("tier") == "prod" for j in pending)
    potential_move = min(len(job.assigned_nodes) - lo, ctx.get("resize_budget", CORRECT_BUDGET))
    # Ordinary one-GPU batch-to-batch pressure is exactly what the deterministic policy handles.
    # Spend three calls only when tier or disruption makes overriding that policy plausible.
    contested = (job.attributes.get("tier") == "prod" or waiting_prod or potential_move >= 2)
    if not contested:
        return _resize_malleable(job, free, pending, ctx)

    plan = _anchor_plan(pending, free, ctx)
    demand_packet = _packet_demand(pending, now, plan)
    lim = int(job.attributes.get("req_min", 0)) * 60
    elapsed = max(0, int(now - job.start_time))
    pct = 100 * elapsed // max(1, lim) if lim else 0
    supply_packet = (f"now={int(now)}s free_gpus={len(free)} queue_depth={len(pending)}\n"
                     f"CURRENT id={job.identifier} tier={job.attributes.get('tier', '?')} "
                     f"current={len(job.assigned_nodes)} min={lo} "
                     f"asked={job.attributes.get('req_nodes', lo)} elapsed_s={elapsed} "
                     + (f"limit_s={lim} pct_of_limit={pct}" if lim else "limit=UNLIMITED"))
    d = _ask(DEMAND_RESIZE_V2, demand_packet, ctx["model"], ctx["host"], ctx["cache"],
             "es-d2-resize", num_predict=300)
    sup = _ask(SUPPLY_RESIZE_V2, supply_packet, ctx["model"], ctx["host"],
               ctx["cache"], "es-s2-resize", num_predict=300)
    ctx["calls"] += 2
    user = (demand_packet + "\n\nCURRENT DONOR:\n" + supply_packet
            + f"\n\ndemand advocate: {json.dumps(d) if d else 'none'}"
            + f"\n\nsupply advocate: {json.dumps(sup) if sup else 'none'}")
    ans = _ask(REFEREE_RESIZE_V2, user, ctx["model"], ctx["host"], ctx["cache"],
               "es-r2-resize", num_predict=300)
    ctx["calls"] += 1
    before = len(job.assigned_nodes)
    reason = str((ans or {}).get("reason", "none"))
    try:
        evidence = int((ans or {}).get("evidence_job", -1))
    except (TypeError, ValueError):
        evidence = -1
    proven = _near_limit_protection_proven(job, now)
    protected = ((ans or {}).get("protect_current") is True
                 and reason == "current_prod_near_limit"
                 and evidence == job.identifier and proven)
    if protected:
        ctx["protected_events"] = ctx.get("protected_events", 0) + 1
    delta = 0 if protected else _resize_malleable(job, free, pending, ctx)
    outcome = "protected" if protected else ("baseline resized" if delta else "baseline unchanged")
    ctx["transcript"].write(json.dumps({
        "t": now, "event": "resize", "free": len(free), "pending": len(pending),
        "resizing_job": job.identifier, "size_before": before, "size_after": before + delta,
        "demand": d, "supply": sup, "proposal": ans, "protection_proven": proven,
        "outcome": outcome}) + "\n")
    ctx["transcript"].flush()
    return delta


def _llm_resize_single_decide(job, pending, free, ctx, use_text=False, debate=False) -> int:
    """Single-agent control for the three-role resize review in ``neg_v2``.

    It uses the same escalation gate, evidence, permitted exception, and validator, but asks the
    referee directly once instead of first collecting demand and supply opinions.
    """
    from pins.correction import _ask
    now = ctx["now"]
    last = ctx.setdefault("last_resize", {}).get(job.identifier, float("-inf"))
    if now - last < ctx.get("resize_cooldown", 0):
        ctx["resize_cooldown_blocks"] += 1
        return 0
    lo, _ = _sizes(job)
    if not pending or len(job.assigned_nodes) <= lo:
        return _resize_malleable(job, free, pending, ctx)

    waiting_prod = any(j.attributes.get("tier") == "prod" for j in pending)
    potential_move = min(len(job.assigned_nodes) - lo, ctx.get("resize_budget", CORRECT_BUDGET))
    contested = (job.attributes.get("tier") == "prod" or waiting_prod or potential_move >= 2)
    # The narrow gate consults the model only where the deterministic policy was already going to
    # move GPUs, so a job whose note PERMITS a shrink is never asked and the shrink half of the
    # exception suite cannot be scored at all. `--gate wide` asks about any job that could still
    # move, which is the condition the semantic question actually needs.
    if not contested and ctx.get("gate") != "wide":
        return _resize_malleable(job, free, pending, ctx)

    plan = _anchor_plan(pending, free, ctx)
    demand_packet = _packet_demand(pending, now, plan)
    lim = int(job.attributes.get("req_min", 0)) * 60
    elapsed = max(0, int(now - job.start_time))
    pct = 100 * elapsed // max(1, lim) if lim else 0
    supply_packet = (f"now={int(now)}s free_gpus={len(free)} queue_depth={len(pending)}\n"
                     f"CURRENT id={job.identifier} tier={job.attributes.get('tier', '?')} "
                     f"current={len(job.assigned_nodes)} min={lo} "
                     f"asked={job.attributes.get('req_nodes', lo)} elapsed_s={elapsed} "
                     + (f"limit_s={lim} pct_of_limit={pct}" if lim else "limit=UNLIMITED"))
    trace_jid = str(job.attributes.get("jid", job.identifier))
    note = ctx.get("text_exceptions", {}).get(trace_jid, {}).get("note", "") if use_text else ""
    user = demand_packet + "\n\nCURRENT DONOR:\n" + supply_packet
    if use_text:
        user += f"\ntrace_job_id={trace_jid}\nSYNTHETIC NOTE: {note or 'No note supplied.'}"
    d = sup = None
    if debate:
        # Two opposed readings of the SAME note, then a referee that sees both. Three calls.
        d = _ask(TEXT_RESIZE_DEMAND, user, ctx["model"], ctx["host"], ctx["cache"],
                 "es-text-demand", num_predict=300)
        sup = _ask(TEXT_RESIZE_SUPPLY, user, ctx["model"], ctx["host"], ctx["cache"],
                   "es-text-supply", num_predict=300)
        ctx["calls"] += 2
        user += (f"\n\nqueue advocate: {json.dumps(d) if d else 'none'}"
                 f"\n\njob advocate: {json.dumps(sup) if sup else 'none'}")
    system_prompt = (TEXT_RESIZE_REFEREE if debate else TEXT_RESIZE_SINGLE) if use_text else REFEREE_RESIZE_V2
    tag = ("es-text-debate-resize" if debate else "es-text-single-resize") if use_text else "es-single-resize"
    ans = _ask(system_prompt, user, ctx["model"], ctx["host"], ctx["cache"], tag, num_predict=300)
    ctx["calls"] += 1
    before = len(job.assigned_nodes)
    reason = str((ans or {}).get("reason", "none"))
    try:
        evidence = int((ans or {}).get("evidence_job", -1))
    except (TypeError, ValueError):
        evidence = -1
    proven = _near_limit_protection_proven(job, now)
    if use_text:
        action = str((ans or {}).get("action", "default")).strip().lower()
        evidence_raw = str((ans or {}).get("evidence_job", "")).strip()
        protected = action == "protect" and evidence_raw in (trace_jid, str(job.identifier))
    else:
        action = "protect" if (ans or {}).get("protect_current") is True else "default"
        protected = ((ans or {}).get("protect_current") is True
                     and reason == "current_prod_near_limit"
                     and evidence == job.identifier and proven)
    if protected:
        ctx["protected_events"] = ctx.get("protected_events", 0) + 1
    delta = 0 if protected else _resize_malleable(job, free, pending, ctx)
    outcome = "protected" if protected else ("baseline resized" if delta else "baseline unchanged")
    ctx["transcript"].write(json.dumps({
        "t": now, "event": "resize", "free": len(free), "pending": len(pending),
        "resizing_job": job.identifier, "size_before": before, "size_after": before + delta,
        "trace_job_id": trace_jid, "synthetic_note": note or None, "model_action": action,
        "elapsed_s": elapsed, "demand": d, "supply": sup,
        "proposal": ans, "protection_proven": proven, "outcome": outcome}) + "\n")
    ctx["transcript"].flush()
    return delta


def _sizing_ambiguous(plan, pending, free) -> bool:
    """Escalate on a real sizing CHOICE, not on scarcity and not on slack. Three shapes, each an
    actual decision the anchor resolved by rule: (1) a started job was squeezed below what it
    asked for and there is room to give some back; (2) a waiting job could start now if it
    accepted less than it asked for; (3) a prod job waits while batch jobs run. A job started at
    its asked size with GPUs to spare is NOT a choice -- growing it past asked is the greedy trap
    measured worst on every axis -- and an earlier version fired on exactly that."""
    asked = lambda j: max(_sizes(j)[0], min(_sizes(j)[1], int(j.attributes.get("req_nodes", _sizes(j)[0]))))
    unsold = len(free) - sum(k for _, k in plan)
    started = {j.identifier for j, _ in plan}
    waiting = [j for j in pending if j.identifier not in started]
    if unsold > 0 and any(k < asked(j) for j, k in plan):
        return True
    if any(_sizes(j)[0] <= unsold < asked(j) for j in waiting):
        return True
    return (any(j.attributes.get("tier") == "prod" for j in waiting)
            and any(j.attributes.get("tier") != "prod" for j, _ in plan))


def _build_from_ruling(plan, ans, pending, free, ctx):
    """Turn (favour list, ceiling) into sizes. All arithmetic here. Grow favoured jobs toward
    asked/hi from unsold capacity in order, promote favoured waiting jobs, then enforce the ceiling
    by revoking from the least-valued (shortest-waiting batch first). Bounded by CORRECT_BUDGET via
    correction_signed.disruption; over budget or unparseable -> the anchor stands."""
    from pins.correction_signed import disruption
    if not isinstance(ans, dict):
        return plan, "no parse"
    by_id = {j.identifier: j for j in pending}
    fav = []
    for x in (ans.get("favour") or []):
        try:
            i = int(str(x).strip().removeprefix("id="))
        except (TypeError, ValueError):
            continue
        if i in by_id and i not in fav:
            fav.append(i)
    try:
        cm = int(ans.get("commit_max", -1))
    except (TypeError, ValueError):
        cm = -1
    base_used = sum(k for _, k in plan)
    ceiling = base_used if cm < 0 else max(0, min(len(free), cm))
    if not fav and ceiling == base_used:
        return plan, "no change"
    alloc = {j.identifier: k for j, k in plan}
    asked = lambda j: max(_sizes(j)[0], min(_sizes(j)[1], int(j.attributes.get("req_nodes", _sizes(j)[0]))))
    # Funding order: unsold capacity first, then a TRANSFER from the least-valued non-favoured
    # started job (batch before prod, shortest-waiting first). These are pending jobs choosing who
    # STARTS this tick, so revoking one is "wait one more tick", not preemption -- which is why a
    # start may be revoked to zero here while correction_signed's floors protect a running base.
    donors = sorted((i for i in alloc if i not in fav),
                    key=lambda i: (by_id[i].attributes.get("tier") == "prod", -by_id[i].submit_time))
    room = ceiling - sum(alloc.values())
    for i in fav:
        snapshot, room_before = dict(alloc), room
        j = by_id[i]; lo, hi = _sizes(j); cur = alloc.get(i, 0)
        # target is what the owner ASKED for, started or not: the only licensed reason to favour
        # a started job is that it was squeezed below asked, and growing past asked is greedy.
        target = asked(j)
        need = max(0, target - cur); got = 0
        take = min(max(room, 0), need); got += take; room -= take; need -= take
        for dnr in donors:                              # transfer, least-valued first
            if need <= 0:
                break
            t = min(alloc.get(dnr, 0), need); alloc[dnr] = alloc.get(dnr, 0) - t; got += t; need -= t
        if cur == 0 and got < lo:                       # cannot start it legally: give it all back
            alloc, room, got = snapshot, room_before, 0
        alloc[i] = cur + got
    # enforce the ceiling: least-valued non-favoured donates
    over = sum(alloc.values()) - ceiling
    for dnr in donors:
        if over <= 0:
            break
        t = min(alloc.get(dnr, 0), over); alloc[dnr] = alloc.get(dnr, 0) - t; over -= t
    # A partially revoked pending start below its legal minimum is no allocation at all. Normalize
    # before disruption accounting so dropping a 4-GPU rigid job can never masquerade as a 1-GPU edit.
    alloc = {i: (k if k >= _sizes(by_id[i])[0] else 0) for i, k in alloc.items()}
    if sum(alloc.values()) > ceiling:
        return plan, "unfunded ceiling"
    before = {str(j.identifier): k for j, k in plan}
    after = {str(i): k for i, k in alloc.items() if k > 0}
    if disruption(before, after) > CORRECT_BUDGET:
        return plan, "over budget"
    if after == before:
        return plan, "no change"        # a ruling that changed nothing must not count as one
    out = [(by_id[i], k) for i, k in alloc.items() if k > 0]
    return out, "applied"


PACKET_CAP = 40
# v1 (day-157 single/debate runs): "req_min=0" for undeclared limits -- the 14b model read 0 as SHORTEST and
# returned the whole queue ignoring free_gpus (transcript autopsy 2026-09-03). v2 spells both out.
PACKET_VERSION = os.environ.get("ES_PACKET", "v2")


def _packet(pending, free, now, order: str = "submit"):
    """`order` controls DISPLAY order only; the same jobs are shown either way, and every row
    already carries waited_s, so shuffling loses no information. The default labels the list
    "first N by submit", which reads as an endorsement of arrival order -- the critic's answers
    matched plain first-fit on 5.1 / 17.2 / 29.5% of decisions as it was told more, and arrival
    priming is the leading hypothesis. `shuffled` removes the cue so the effect can be measured
    rather than assumed. Seeded on the sim clock, so a run stays reproducible."""
    shown = pending[:PACKET_CAP]
    if order == "shuffled":
        shown = list(shown); random.Random(int(now)).shuffle(shown)
        head = (f"now={int(now)}s free_gpus={len(free)} pending={len(pending)} "
                f"(showing {len(shown)} of them, IN NO PARTICULAR ORDER -- the list order carries no "
                f"priority, use the fields). You can start jobs totalling at most {len(free)} GPUs "
                f"now; list only those, in the order you want them started.")
        lines = [head]
    elif PACKET_VERSION == "v1":
        lines = [f"now={int(now)}s free_gpus={len(free)} pending={len(pending)} (showing first {PACKET_CAP} by submit)"]
    else:
        lines = [f"now={int(now)}s free_gpus={len(free)} pending={len(pending)} (showing first {PACKET_CAP} by submit). "
                 f"You can start jobs totalling at most {len(free)} GPUs now; list only those, in start order."]
    for j in shown:
        a = j.attributes
        # worlds built before the tier label keep the old priority field so old runs replay byte-identically
        cls = f"tier={a['tier']}" if "tier" in a else f"priority={a['priority']}"
        req = (f"req_min={a['req_min']}" if PACKET_VERSION == "v1" else
               (f"req_limit={a['req_min']}min" if int(a["req_min"]) else "req_limit=UNLIMITED(no estimate)"))
        # A MALLEABLE job has a legal RANGE, not a size. Reporting num_nodes_min here (as an earlier
        # revision did) tells the model every elastic job needs 1 GPU, which is false and makes the
        # packet describe a cluster that does not exist.
        lo, hi = _sizes(j)
        size = f"gpus={lo}" if lo == hi else f"gpus={lo}-{hi} asked={a.get('req_nodes', lo)}"
        lines.append(f"id={j.identifier} {size} {req} "
                     f"waited_s={int(now - j.submit_time)} {cls} partition={a['partition']}")
    return "\n".join(lines)


def transcript_stats(path: Path) -> dict:
    """Reviewer-requested LLM wrapper accounting: what the model proposed vs what the validator let through,
    and how often the final pick equals what first-fit would have done on the same packet."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    # Resize-debate records share the transcript but have no start-list packet.
    L = [x for l in open(path) if "packet" in (x := json.loads(l)) and "picked" in x]
    if not L:
        return {}
    same = dropped = proposed = 0
    for x in L:
        rows = [l.split() for l in x["packet"].splitlines()[1:]]
        # a malleable job prints "gpus=LO-HI"; the first-fit reference below only needs the smallest
        # size that lets it start, so take LO. Rigid jobs print "gpus=N", where LO == N.
        lo = lambda tok: int(tok[5:].split("-")[0])
        ids = [int(r[0][3:]) for r in rows]; g = {int(r[0][3:]): lo(r[1]) for r in rows}
        free, ff = x["free"], []
        for i in ids:
            if g[i] <= free:
                ff.append(i); free -= g[i]
        same += x["picked"] == ff
        # Only direct start-list arms go through _validate. Ranking/correction arms include the
        # deterministic anchor in `picked`, so subtracting it from a non-existent start proposal
        # produced nonsense such as ids_dropped_by_validator=-1.
        if "start" in (x["proposal"] or {}):
            p = (x["proposal"] or {}).get("start") or []
            p = [e[0] if isinstance(e, (list, tuple)) and e else e for e in p]
            proposed += len(p); dropped += max(0, len(p) - len(x["picked"] or []))
    return {"decisions": len(L), "ids_proposed": proposed, "ids_dropped_by_validator": dropped,
            "invalid_answers": sum(x["picked"] is None for x in L),
            "critic_changed": sum(1 for x in L if x["critic"] and (x["critic"].get("start") != (x["proposal"] or {}).get("start"))),
            "pick_eq_firstfit_pct": round(100 * same / len(L), 1)}


def text_exception_stats(transcript: Path, labels_path: Path, world: Path,
                         history_path: Path | None = None) -> dict:
    """Score a run's OUTCOME against every planted exception in the world.

    The denominator is the set of exceptions the world contains, NOT the number of times the
    scheduler chose to consult the LLM.  Scoring only the consulted cases lets an arm that never
    escalates report perfect recall on the two cases it happened to look at, which is the fire-rate
    trap: the metric then measures the gate's silence, not the reasoning.  Compliance here is read
    off the allocation trajectory, so a zero-call arm is scored on exactly the same footing.
    """
    if not labels_path.exists():
        return {}
    labels = json.loads(labels_path.read_text()).get("labels", {})
    if not labels:
        return {}
    jobs = json.loads((world / "in/jobs.json").read_text())["jobs"]
    hp = history_path or transcript.with_name(
        transcript.name.replace("_transcript.jsonl", "_size_history.json"))
    histories = json.loads(hp.read_text()) if hp.exists() else {}
    if not histories:
        return {}

    # index -> trace job id, the join between the answer key and the simulator's own numbering
    jid_of = {str(i): str(j["attributes"]["jid"]) for i, j in enumerate(jobs)}
    mins = {str(i): j.get("num_nodes_min", j.get("num_nodes", 1)) for i, j in enumerate(jobs)}
    shrunk, ran, movable = set(), set(), set()
    for idx, events in histories.items():
        jid = jid_of.get(str(idx))
        if jid is None:
            continue
        ran.add(jid)
        if any(events[n + 1][1] < events[n][1] for n in range(len(events) - 1)):
            shrunk.add(jid)
        # A job that never rose above its legal minimum had nothing to release and nothing to lose,
        # so neither a protect nor a shrink note on it can be honoured OR violated. Scoring those
        # cases as misses charges an arm for decisions the world never offered it.
        if any(g > mins[str(idx)] for _, g in events):
            movable.add(jid)

    planted = {a: [j for j, v in labels.items()
                   if v.get("expected_action") == a and j in ran]
               for a in ("protect", "shrink")}
    # A protect note is honoured by never shrinking the job; a shrink note by actually shrinking it.
    violations = [j for j in planted["protect"] if j in shrunk]
    complied = [j for j in planted["shrink"] if j in shrunk]

    rows = [json.loads(line) for line in transcript.read_text().splitlines()
            if line.strip()] if transcript.exists() else []
    asked = {r["trace_job_id"] for r in rows
             if r.get("model_action") and r.get("trace_job_id") in labels}
    n_planted = len(planted["protect"]) + len(planted["shrink"])
    asked_exc = [r for r in rows if r.get("trace_job_id") in labels and r.get("model_action")
                 and labels[r["trace_job_id"]].get("expected_action") in ("protect", "shrink")]

    act = {a: [j for j in planted[a] if j in movable] for a in ("protect", "shrink")}
    act_viol = [j for j in act["protect"] if j in shrunk]
    act_comp = [j for j in act["shrink"] if j in shrunk]
    n_act = len(act["protect"]) + len(act["shrink"])
    pct = lambda x, d: round(100 * x / d, 1) if d else None
    return {
        # --- ACTIONABLE denominators: cases where the job ever held more than its minimum ---
        "protect_actionable": len(act["protect"]),
        "shrink_actionable": len(act["shrink"]),
        "actionable_compliance_pct": pct(
            (len(act["protect"]) - len(act_viol)) + len(act_comp), n_act),
        "shrink_actionable_pct": pct(len(act_comp), len(act["shrink"])),
        # --- outcome, over every planted exception (the honest denominators) ---
        "planted_protect": len(planted["protect"]),
        "protect_violations": len(violations),
        "protect_compliance_pct": pct(len(planted["protect"]) - len(violations),
                                      len(planted["protect"])),
        "planted_shrink": len(planted["shrink"]),
        "shrink_taken": len(complied),
        "shrink_compliance_pct": pct(len(complied), len(planted["shrink"])),
        "exception_compliance_pct": pct(
            (len(planted["protect"]) - len(violations)) + len(complied), n_planted),
        # --- how much of the planted set the gate ever showed the model ---
        "exceptions_escalated": len(asked_exc),
        "exceptions_escalated_pct": pct(len(asked_exc), n_planted),
        # --- the old call-conditioned figure, kept but named so it cannot pose as recall ---
        "asked_decisions": len(asked),
        "asked_accuracy_pct": pct(sum(labels[r["trace_job_id"]]["expected_action"]
                                      == r["model_action"] for r in asked_exc), len(asked_exc)),
    }


def _validate(ans, pending, free):
    """Feasibility only: keep (job, size) pairs that are pending, legal and fit, in the LLM's order.
    An entry is either a bare id (use the size its owner asked for) or [id, gpus]. The size is
    CLAMPED into the job's legal range rather than rejected -- the LLM proposes, this decides."""
    if not ans or not isinstance(ans.get("start"), list):
        return None
    by_id = {j.identifier: j for j in pending}
    picks, used = [], 0
    for x in ans["start"]:
        n = None
        if isinstance(x, (list, tuple)):
            # models emit [id, gpus], and also bare [id] -- treat the latter as an unsized pick
            # rather than letting int([id]) throw and silently empty the whole decision.
            if len(x) == 2:
                x, n = x
            elif len(x) == 1:
                x = x[0]
            else:
                continue
        # models echo the packet's literal "id=43" rather than 43, so strip the label before
        # parsing. Left unhandled this silently empties every pick list (interface, not reasoning).
        if isinstance(x, str):
            x = x.strip().removeprefix("id=")
        try:
            j = by_id.get(int(x))
            n = int(n) if n is not None else None
        except (TypeError, ValueError):
            continue
        if not j or any(j is p for p, _ in picks):
            continue
        lo, hi = _sizes(j)
        if lo == hi:
            k = lo                                  # rigid: one legal size
        else:
            # clamp into the legal range AND into what is still free: an over-ask becomes the most
            # the job may legally have right now, rather than dropping a job that could have started.
            k = max(lo, min(hi, len(free) - used,
                            n if n is not None else int(j.attributes.get("req_nodes", lo))))
        if used + k <= len(free):
            picks.append((j, k)); used += k
    return picks


def _llm_decide(pending, free, ctx, mode: str):
    from pins.correction import _ask
    tiered = "tier" in pending[0].attributes
    T = TIER_NOTE if tiered else ""
    system, critic = SYSTEM + T, CRITIC + T
    packet = _packet(pending, free, ctx["now"], ctx.get("packet_order", "submit"))
    if mode == "negotiate":
        # two advocates with OPPOSED objectives, then a referee that sees the packet AND both
        d = _ask(DEMAND + T, packet, ctx["model"], ctx["host"], ctx["cache"], "es-demand", num_predict=300)
        sup = _ask(SUPPLY + T, packet, ctx["model"], ctx["host"], ctx["cache"], "es-supply", num_predict=300)
        ctx["calls"] += 2
        user = (packet + f"\n\nqueue advocate proposes: {json.dumps(d) if d else 'none'}"
                       + f"\n\ncluster advocate proposes: {json.dumps(sup) if sup else 'none'}")
        ans = _ask(REFEREE + T, user, ctx["model"], ctx["host"], ctx["cache"], "es-referee", num_predict=300)
        ctx["calls"] += 1
        picks = _validate(ans, pending, free)
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "demand": d, "supply": sup, "proposal": ans, "critic": None,
            "picked": None if picks is None else [j.identifier for j, _ in picks],
            "sizes": None if picks is None else [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    if mode == "neg_signed":
        # negotiation that emits a CORRECTION: demand and supply argue about the deterministic
        # plan, the referee rules, and apply_signed funds the ruling. A rejected ruling leaves
        # the anchor standing. That is safety against unusable actions, not a guarantee of improvement --
        # an accepted correction can still hurt, and it perturbs every later queue state.
        plan = _anchor_plan(pending, free, ctx)
        base = packet + "\n\n" + _anchor_line(plan, free)
        d = _ask(DEMAND_SIGNED + T, base, ctx["model"], ctx["host"], ctx["cache"], "es-dsigned", num_predict=300)
        sup = _ask(SUPPLY_SIGNED + T, base, ctx["model"], ctx["host"], ctx["cache"], "es-ssigned", num_predict=300)
        ctx["calls"] += 2
        user = (base + f"\n\ndemand-side reviewer proposes: {json.dumps(d) if d else 'none'}"
                     + f"\n\nsupply-side reviewer proposes: {json.dumps(sup) if sup else 'none'}")
        ans = _ask(REFEREE_SIGNED + T, user, ctx["model"], ctx["host"], ctx["cache"], "es-rsigned", num_predict=300)
        ctx["calls"] += 1
        picks, how = _apply_correction(plan, ans, pending, free)
        ctx.setdefault("correct_outcome", {}).setdefault(how, 0)
        ctx["correct_outcome"][how] += 1
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "anchor": [[j.identifier, k] for j, k in plan], "demand": d, "supply": sup,
            "proposal": ans, "critic": None, "outcome": how,
            "picked": [j.identifier for j, _ in picks], "sizes": [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    if mode == "neg_v2":
        plan = _anchor_plan(pending, free, ctx)
        dpk = _packet_demand(pending, ctx["now"], plan)
        spk = _packet_supply(plan, pending, free, ctx, ctx["now"])
        d = _ask(DEMAND_V2 + T, dpk, ctx["model"], ctx["host"], ctx["cache"], "es-d2", num_predict=300)
        sup = _ask(SUPPLY_V2 + T, spk, ctx["model"], ctx["host"], ctx["cache"], "es-s2", num_predict=300)
        ctx["calls"] += 2
        user = (packet + "\n\n" + _anchor_line(plan, free)
                + f"\n\nqueue's advocate (saw jobs only): {json.dumps(d) if d else 'none'}"
                + f"\n\ncluster's advocate (saw cluster only): {json.dumps(sup) if sup else 'none'}")
        ans = _ask(REFEREE_V2 + T, user, ctx["model"], ctx["host"], ctx["cache"], "es-r2", num_predict=300)
        ctx["calls"] += 1
        picks, how = _build_from_ruling(plan, ans, pending, free, ctx)
        ctx.setdefault("correct_outcome", {}).setdefault(how, 0)
        ctx["correct_outcome"][how] += 1
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "packet_demand": dpk, "packet_supply": spk,
            "anchor": [[j.identifier, k] for j, k in plan], "demand": d, "supply": sup,
            "proposal": ans, "critic": None, "outcome": how,
            "picked": [j.identifier for j, _ in picks], "sizes": [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    if mode in ("correct3", "sham", "neg_v2"):
        # correct3: THE causal control for neg_signed -- same anchor, same signed contract, same
        #   3 calls, but no roles and no argument. neg_signed minus correct3 is what the
        #   negotiation itself buys; neg_signed minus the anchor only measures the whole package.
        # sham: neg_signed's referee with the advocates' proposals replaced by empty ones. If it
        #   performs the same, the advocates contributed nothing and the arm is a 1-call
        #   correction paying for 3.
        plan = _anchor_plan(pending, free, ctx)
        base = packet + "\n\n" + _anchor_line(plan, free)
        if mode == "sham":
            user = (base + "\n\ndemand-side reviewer proposes: none"
                         + "\n\nsupply-side reviewer proposes: none")
            ans = _ask(REFEREE_SIGNED + T, user, ctx["model"], ctx["host"], ctx["cache"],
                       "es-sham", num_predict=300)
            ctx["calls"] += 3        # charged the same 3 calls, so the budget comparison is honest
            cands = [ans]
        else:
            cands = [_ask(CORRECTOR + T, base, ctx["model"], ctx["host"], ctx["cache"],
                          f"es-c3-{k}", num_predict=300, temperature=0.8) for k in range(3)]
            ctx["calls"] += 3
        # deterministic aggregation: the modal funded outcome, ties broken by sample order
        outs = []
        for a in cands:
            pk, how = _apply_correction(plan, a, pending, free)
            outs.append((tuple(sorted((j.identifier, n) for j, n in pk)), pk, how))
        from collections import Counter
        win = Counter(o[0] for o in outs).most_common(1)[0][0]
        picks, how = next((o[1], o[2]) for o in outs if o[0] == win)
        ctx.setdefault("correct_outcome", {}).setdefault(how, 0)
        ctx["correct_outcome"][how] += 1
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "anchor": [[j.identifier, k] for j, k in plan], "proposal": cands[0], "critic": None,
            "n_samples": len(cands), "outcome": how,
            "picked": [j.identifier for j, _ in picks], "sizes": [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    if mode == "correct":
        plan = _anchor_plan(pending, free, ctx)
        user = packet + "\n\n" + _anchor_line(plan, free)
        ans = _ask(CORRECTOR + T, user, ctx["model"], ctx["host"], ctx["cache"], "es-correct",
                   num_predict=300)
        ctx["calls"] += 1
        picks, how = _apply_correction(plan, ans, pending, free)
        ctx.setdefault("correct_outcome", {}).setdefault(how, 0)
        ctx["correct_outcome"][how] += 1
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "anchor": [[j.identifier, k] for j, k in plan], "proposal": ans, "critic": None,
            "outcome": how,
            "picked": [j.identifier for j, _ in picks], "sizes": [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    if mode == "bo3":
        # budget-matched control: 3 samples of ONE model, self-consistency. temperature must be
        # raised AND the tag varied per sample -- the cache key ignores temperature (correction._ask).
        cands = []
        for k in range(3):
            a = _ask(system, packet, ctx["model"], ctx["host"], ctx["cache"], f"es-bo3-{k}",
                     num_predict=300, temperature=0.8)
            ctx["calls"] += 1
            v = _validate(a, pending, free)
            if v:
                cands.append((tuple((j.identifier, n) for j, n in v), v, a))
        picks = None; ans = None
        if cands:
            from collections import Counter
            win = Counter(c[0] for c in cands).most_common(1)[0][0]
            picks, ans = next((c[1], c[2]) for c in cands if c[0] == win)
        ctx["transcript"].write(json.dumps({
            "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
            "proposal": ans, "critic": None, "n_samples": len(cands),
            "picked": None if picks is None else [j.identifier for j, _ in picks],
            "sizes": None if picks is None else [k for _, k in picks]}) + "\n")
        ctx["transcript"].flush()
        return picks
    ans = _ask(system, packet, ctx["model"], ctx["host"], ctx["cache"], "es-propose", num_predict=300)
    ctx["calls"] += 1
    if ctx["calls"] % 50 == 0:
        print(f"  {ctx['calls']} calls, sim t={ctx['now'] / 3600:.1f}h, {len(pending)} pending", flush=True)
    final = None
    picks = _validate(ans, pending, free)
    if mode == "debate":
        user = packet + f"\n\ncolleague's proposal: {json.dumps(ans) if ans else 'none'}"
        final = _ask(critic, user, ctx["model"], ctx["host"], ctx["cache"], "es-critic", num_predict=300)
        ctx["calls"] += 1
        fp = _validate(final, pending, free)
        if fp is not None:
            ctx["critic_changed"] += (final or {}).get("start") != (ans or {}).get("start")
            picks = fp
    ctx["transcript"].write(json.dumps({
        "t": ctx["now"], "free": len(free), "pending": len(pending), "packet": packet,
        "proposal": ans, "critic": final,
        "picked": None if picks is None else [j.identifier for j, _ in picks],
        "sizes": None if picks is None else [k for _, k in picks]}) + "\n")
    ctx["transcript"].flush()
    return picks


def arm_llm(pending, free, ctx, mode="single"):
    # Two gates. "scarcity" (default, every earlier run): call only when the waiting work exceeds
    # free capacity. "ambiguity": call only when the anchor faces a real sizing choice -- fires on
    # ticks where a decision can matter rather than merely on how busy the cluster is.
    if ctx.get("gate", "scarcity") == "ambiguity":
        if not _sizing_ambiguous(_anchor_plan(pending, free, ctx), pending, free):
            ctx["trivial"] += 1
            return arm_firstfit(pending, free, ctx)
    elif sum(_sizes(j)[0] for j in pending) <= len(free):   # nothing to ration -> no call (the gate)
        ctx["trivial"] += 1
        return arm_firstfit(pending, free, ctx)
    picks = _llm_decide(pending, free, ctx, mode)
    # an EMPTY pick list idles the cluster while jobs fit -- never right here, and in practice it
    # means the answer failed to parse. Treat it as a fallback, not as a decision to hold.
    if not picks and mode != "neg_v2":
        ctx["fallbacks"] += 1
        return arm_firstfit(pending, free, ctx)
    for j, k in picks:
        _start_at(j, free, ctx, k)


# ---------------------------------------------------------------------------------------------
# Policy selection: the referee picks a scheduling POLICY, not an allocation. Both halves of the
# decision are named, so it can combine an ordering rule with a sizing rule -- the pair, not either
# alone, is what the deterministic sweep showed to matter.
POLICY_MENU = {
    "ordering": {
        "fcfs": "strict arrival order; a job that does not fit blocks everything behind it",
        "firstfit": "arrival order, but skip a job that does not fit and try the next",
        "easy": "arrival order with backfilling behind the head job's reservation",
        "sjf": "shortest requested walltime first (this trace's walltimes are a weak runtime signal)",
        "declared_first": "jobs that declared a walltime before those that did not",
        "tier_fcfs": "production tier first, arrival order within tier",
        "tier_sjf": "production tier first, shortest requested walltime within tier",
        "market": "sealed-bid uniform-price auction: each job bids the runtime seconds each extra "
                  "GPU would save it, weighted by tier and time already queued, and the clearing "
                  "awards the free GPUs to the highest marginal bids",
    },
    "sizing": {
        "as_requested": "give each job the size its owner asked for; wait until that many are free",
        "adaptive": "share the free GPUs across the waiting queue; each job may start smaller",
        "greedy": "give each job the largest legal size that fits right now",
    },
}
# Constraints the cluster imposes on any answer. These are checked in code after the reply, so a
# ruling that breaks one is discarded rather than trusted -- the LLM proposes, the validator disposes.
POLICY_RULES = (
    "1. Never exceed the pool: allocations are capped by free GPUs and by each job's legal maximum.\n"
    "2. Never place a malleable job below its minimum or above its maximum size.\n"
    "3. Scaling is sublinear: 4x the GPUs is far less than 4x the speed, so `greedy` buys a little "
    "speed for a lot of capacity and starves the queue behind it.\n"
    "4. Production-tier jobs may be favoured, but starving the batch queue is a failure.\n"
    "5. You choose a POLICY that the scheduler then applies deterministically. You never name a "
    "job or a GPU count yourself."
)
POLICY_SELECT = ("You choose the scheduling policy for a GPU cluster for the next interval. " + WORLD +
                 "\n\nYou pick ONE ordering rule and ONE sizing rule; the pair is the policy. "
                 "The cluster state and the menu are given below, with the rules you must respect. "
                 "Re-decide BOTH halves from the state each time. The current policy is shown for "
                 "reference only; it carries no presumption of being right, and the ordering rule "
                 "matters as much as the sizing rule. Consider what the queue you are shown will "
                 "look like several intervals from now, not only right now: a policy that is "
                 "comfortable while the cluster is quiet can build a backlog that never clears. "
                 "If a record of your own recent decisions is shown, treat it as evidence: a policy "
                 "you have already held while the queue grew is one to reconsider, not to repeat. "
                 "Reply with JSON only: "
                 '{"ordering": "<name from the menu>", "sizing": "<name from the menu>", '
                 '"why": "one line naming the state feature that drove the choice"}.')



def _load_fields(pending, ctx) -> dict:
    """Two DIFFERENT quantities that were previously conflated under one name.

    `queue_pressure` is waiting GPU demand over pool size -- instantaneous backlog. It was called
    `offered_load`, which it is not: the window-level offered load used for stratification is
    submitted GPU-hours per window-hour per GPU, and the two disagree most exactly when the queue is
    empty but arrivals are heavy.
    `arrival_load` is the trailing arrival rate over the last hour, in GPUs demanded per pool GPU.
    It is computable online from what the scheduler has already seen, and unlike backlog it does not
    read zero on a quiet cluster that is filling up.
    """
    pool = max(1, ctx.get("pool_n", 1))
    now = ctx.get("now", 0)
    window = 3600.0
    recent = sum(g for t, g in ctx.get("arrivals", ()) if now - window <= t <= now)
    return {"queue_pressure": sum(max(1, int(j.attributes.get("req_nodes", 1))) for j in pending) / pool,
            "arrival_load": recent / pool}


def _packet_policy(pending, free, ctx) -> str:
    """Cluster state plus the menu and the constraints: everything the choice may depend on."""
    pool = ctx.get("pool_n", 0)
    demand = sum(max(1, int(j.attributes.get("req_nodes", 1))) for j in pending)
    prod = sum(1 for j in pending if j.attributes.get("tier") == "prod")
    running = ctx.get("running", [])
    declared = sum(1 for j in pending if int(j.attributes.get("req_min", 0)) > 0)
    lines = [f"now={int(ctx['now'])}s pool={pool} free={len(free)} running={len(running)}",
             f"queue_depth={len(pending)} waiting_demand={demand} "
             f"queue_pressure={_load_fields(pending, ctx)['queue_pressure']:.2f} "
             f"arrival_load_1h={_load_fields(pending, ctx)['arrival_load']:.2f} "
             f"prod_waiting={prod} declared_walltime={declared}",
             f"current_policy=ordering:{ctx.get('sel_ordering', 'firstfit')} "
             f"sizing:{ctx.get('sel_sizing', 'as_requested')}",
             "", "ORDERING MENU:"]
    lines += [f"  {k}: {v}" for k, v in POLICY_MENU["ordering"].items()]
    lines += ["", "SIZING MENU:"] + [f"  {k}: {v}" for k, v in POLICY_MENU["sizing"].items()]
    hist = ctx.get("sel_history", [])
    if hist:
        # The scratchpad: what you chose, and what the cluster did next. Without it the referee has
        # no evidence that any choice ever mattered, and a selector with no feedback has no reason
        # to change -- which is exactly the collapse onto one policy we measured.
        lines += ["", "YOUR RECENT DECISIONS AND WHAT FOLLOWED:"]
        for h in hist[-4:]:
            after = (f"queue_depth {h['queue']}->{h['queue_after']}, "
                     f"{h['started_after']} jobs started, running {h['running']}->{h['run_after']}"
                     if h.get("queue_after") is not None else "outcome not yet observed")
            # Field names here MUST match the state block above verbatim. Exp 51's manual arc
            # found the model copies the vocabulary it is given and invents fields when the same
            # quantity is paraphrased, so `queue_depth` and `offered_load` are not renamed.
            lines.append(f"  t={h['t']}s offered_load={h['load']:.2f}: chose "
                         f"ordering:{h['ordering']} sizing:{h['sizing']}  ->  {after}")
        lines.append("  A choice that left the queue growing is evidence against repeating it.")
    lines += ["", "CLUSTER RULES:", POLICY_RULES]
    return "\n".join(lines)


def arm_policy_select(pending, free, ctx):
    """Re-choose the policy at most every `interval` seconds, then apply it deterministically."""
    from pins.correction import _ask
    now = ctx["now"]
    if now - ctx.get("sel_last", float("-inf")) >= ctx.get("sel_every", 0):
        ctx["sel_last"] = now
        started_total = len(ctx.get("sizes", {}))
        hist = ctx.setdefault("sel_history", [])
        if hist and hist[-1].get("queue_after") is None:   # close the previous entry's outcome
            hist[-1]["queue_after"] = len(pending)
            hist[-1]["run_after"] = len(ctx.get("running", []))
            hist[-1]["started_after"] = started_total - hist[-1]["started_total"]
        ans = _ask(POLICY_SELECT, _packet_policy(pending, free, ctx), ctx["model"], ctx["host"],
                   ctx["cache"], "es-policy-select", num_predict=200)
        ctx["calls"] += 1
        o = str((ans or {}).get("ordering", "")).strip()
        z = str((ans or {}).get("sizing", "")).strip()
        ok = o in POLICY_MENU["ordering"] and z in POLICY_MENU["sizing"]
        if ok:
            ctx["sel_ordering"], ctx["sel_sizing"] = o, z
        else:
            ctx["sel_invalid"] = ctx.get("sel_invalid", 0) + 1
        hist.append({"t": int(now), "ordering": ctx.get("sel_ordering"), "sizing": ctx.get("sel_sizing"),
                     "load": round(sum(max(1, int(j.attributes.get("req_nodes", 1))) for j in pending)
                                   / max(1, ctx.get("pool_n", 1)), 2),
                     "queue": len(pending), "running": len(ctx.get("running", [])),
                     "started_total": started_total, "queue_after": None})
        ctx.setdefault("sel_log", []).append(
            {"t": int(now), "ordering": o, "sizing": z, "valid": ok,
             "why": str((ans or {}).get("why", ""))[:120],
             "load": round(sum(max(1, int(j.attributes.get("req_nodes", 1))) for j in pending)
                           / max(1, ctx.get("pool_n", 1)), 2)})
        ctx.setdefault("sel_counts", {})[f"{ctx.get('sel_ordering')}+{ctx.get('sel_sizing')}"] = \
            ctx.setdefault("sel_counts", {}).get(f"{ctx.get('sel_ordering')}+{ctx.get('sel_sizing')}", 0) + 1
    ctx["sizer"] = ctx.get("sel_sizing", "as_requested")
    ARMS[ctx.get("sel_ordering", "firstfit")](pending, free, ctx)


def _bid_curve(job, ctx) -> list[float]:
    """Marginal value of the k-th GPU to this job: the SECONDS OF RUNTIME it saves.

    The old world imported `PHASE_PROFILES` because its jobs had no known speed-up law. Here they
    do -- the Amdahl model this world is built on -- so the bid is read off the job's own physics
    instead of a tuned profile. It is non-increasing by construction, which is exactly the
    diminishing-returns shape `mechanism.clear` assumes, and it makes the auction's currency
    commensurable with the metric: a GPU is worth the waiting time it removes.
    """
    lo, hi = _sizes(job)
    a = job.attributes
    dur = float(a.get("_true_dur", 0)) or 1.0
    n0 = max(1, int(a.get("gpus", 1)))
    sf = ctx.get("par_frac", 1.0)
    T = lambda n: dur * (sf + (1 - sf) / n) / (sf + (1 - sf) / n0)
    # Urgency is the private valuation: production work outbids batch, and a job that has been
    # queueing longer bids more, which is the deterministic laxity lever Exp 99 found.
    waited = max(0.0, ctx.get("now", 0) - getattr(job, "submit_time", 0))
    urgency = (3.0 if a.get("tier") == "prod" else 1.0) * (1.0 + waited / 3600.0)
    # CURRENCY: value DENSITY, not value retired. Bidding the seconds a GPU saves maximises total
    # time saved, which makes the longest job the highest bidder -- backwards for mean waiting time,
    # where clearing short work first releases capacity sooner. So each bid is divided by the
    # resource-time it consumes. This is a choice fixed by the objective, not a tuned constant.
    #   first `lo` GPUs: the job goes from not running to running, worth T(lo) over the lo*T(lo)
    #   GPU-seconds it will occupy -- i.e. 1/lo, scaled by urgency: short jobs are not penalised,
    #   long ones no longer outbid them simply for being long.
    first = urgency / max(1, lo)
    curve = [first] * lo
    #   beyond the minimum: the speed-up bought, per GPU-second that growth costs.
    curve += [urgency * (T(k - 1) - T(k)) / max(1e-9, k * T(k)) for k in range(lo + 1, hi + 1)]
    return [round(max(0.0, v), 6) for v in curve]


def arm_market(pending, free, ctx):
    """Sealed-bid uniform-price clearing over the FREE GPUs, reusing `pins.mechanism.clear`.

    Ported from the two-sided world as a start-selection rule: it never preempts a running job,
    so it composes with the same resize policy as every other ordering arm here.
    """
    from pins.mechanism import clear
    if not free or not pending:
        return
    bids = {str(j.identifier): _bid_curve(j, ctx) for j in pending}
    bids = {k: v for k, v in bids.items() if v}
    if not bids:
        return
    res = clear(bids, total_gpus=len(free))
    ctx["market_price"] = res.price
    ctx["market_clearings"] = ctx.get("market_clearings", 0) + 1
    award = res.allocation
    # The auction decides WHO runs and in what order; the SIZING rule decides how many GPUs, exactly
    # as it does for every other ordering arm. An earlier version allocated the auction's award
    # directly via _start_at, which never read ctx["sizer"] -- so market+as_requested,
    # market+adaptive and market+greedy were one implementation under three names, and the menu was
    # not the 24 distinct pairs it advertised. `_start` routes through `_size`, restoring the
    # ordering-by-sizing factorisation that makes the arms comparable.
    ranked = sorted((j for j in pending if award.get(str(j.identifier), 0) >= _sizes(j)[0]),
                    key=lambda j: -sum(bids.get(str(j.identifier), [0])[:award.get(str(j.identifier), 0)]))
    for job in ranked:
        if not _fit(job, free, ctx, len(pending)):
            continue                     # cannot be sized from what is free: it waits
        _start(job, free, ctx, len(pending))


# ---------------------------------------------------------------------------------------------
# Rule synthesis: the model writes a POLICY-SELECTION PROGRAM once; code executes it every tick.
# This is the manual/precedent idea with the compliance problem removed. Exp 51-53 could not
# separate "the referee will not follow the rule" from "the rule is worthless", because the rule
# was advisory text handed to another model (3b complied 11%, 14b 72%). Here the rule IS the
# scheduler, so compliance is 100% by construction and the CONTENT is what gets measured.
#
# A TYPED FIELD TABLE, not prose. Exp 51's costliest lesson: paraphrase a field and the model
# invents one ("incoming_prod_count" for a categorical), so the legal vocabulary is stated exactly
# as the evaluator spells it.
RULE_FIELDS = {
    "offered_load": "float >= 0. INSTANTANEOUS backlog: waiting GPU demand divided by pool size. "
                    "Reads ~0 whenever the queue is empty, even if heavy arrivals are imminent.",
    "arrival_load": "float >= 0. Trailing 1 hour arrival rate: GPUs demanded by newly submitted jobs "
                    "divided by pool size. Unlike offered_load this does NOT collapse on a quiet queue.",
    "queue_depth": "int >= 0. Number of jobs waiting right now.",
    "free_gpus": "int >= 0. GPUs free right now.",
    "prod_waiting": "int >= 0. Number of waiting jobs in the production tier.",
    "declared_walltime": "int >= 0. Number of waiting jobs that declared a walltime limit.",
}
RULE_OPS = (">=", ">", "<=", "<", "==")
RULE_MAX_BRANCHES = 5


def _validate_rule(rule) -> tuple[bool, str]:
    """Structural + vocabulary check. A rule that fails is not repaired, it is refused."""
    if not isinstance(rule, dict):
        return False, "not an object"
    branches, default = rule.get("branches"), rule.get("default")
    if not isinstance(branches, list) or not branches:
        return False, "no branches"
    if len(branches) > RULE_MAX_BRANCHES:
        return False, f"{len(branches)} branches > {RULE_MAX_BRANCHES}"
    def _pair(d, where):
        if not isinstance(d, dict):
            return f"{where}: not an object"
        if d.get("ordering") not in POLICY_MENU["ordering"]:
            return f"{where}: unknown ordering {d.get('ordering')!r}"
        if d.get("sizing") not in POLICY_MENU["sizing"]:
            return f"{where}: unknown sizing {d.get('sizing')!r}"
        return ""
    err = _pair(default, "default")
    if err:
        return False, err
    for i, b in enumerate(branches):
        if not isinstance(b, dict):
            return False, f"branch {i}: not an object"
        cond = b.get("if")
        if not (isinstance(cond, list) and len(cond) == 3):
            return False, f"branch {i}: `if` must be [field, op, number]"
        f, op, v = cond
        if f not in RULE_FIELDS:
            return False, f"branch {i}: unknown field {f!r}"
        if op not in RULE_OPS:
            return False, f"branch {i}: unknown operator {op!r}"
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, f"branch {i}: threshold {v!r} is not a number"
        err = _pair(b.get("then"), f"branch {i}")
        if err:
            return False, err
    return True, ""


def _rule_state(pending, free, ctx) -> dict:
    """The observable vocabulary, computed exactly as RULE_FIELDS describes it."""
    pool = max(1, ctx.get("pool_n", 1))
    lf = _load_fields(pending, ctx)
    return {
        "offered_load": lf["queue_pressure"],
        "arrival_load": lf["arrival_load"],
        "queue_depth": len(pending),
        "free_gpus": len(free),
        "prod_waiting": sum(1 for j in pending if j.attributes.get("tier") == "prod"),
        "declared_walltime": sum(1 for j in pending if int(j.attributes.get("req_min", 0)) > 0),
    }


def _eval_rule(rule, state, ctx=None) -> tuple[str, str]:
    """First matching branch wins; otherwise the default. Per-branch fire counts are recorded so a
    rule whose precondition never holds is visible rather than silently inert (Exp 51's vacuous P1)."""
    import operator
    OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}
    for i, b in enumerate(rule["branches"]):
        f, op, v = b["if"]
        if OPS[op](state[f], v):
            if ctx is not None:
                ctx.setdefault("rule_fires", {})[str(i)] = ctx.setdefault("rule_fires", {}).get(str(i), 0) + 1
            return b["then"]["ordering"], b["then"]["sizing"]
    if ctx is not None:
        ctx.setdefault("rule_fires", {})["default"] = ctx.setdefault("rule_fires", {}).get("default", 0) + 1
    return rule["default"]["ordering"], rule["default"]["sizing"]


RULE_SYNTH = ("You write the scheduling policy for a GPU cluster ONCE, as a small decision rule. "
              + WORLD +
              "\n\nYou do NOT schedule jobs and you will not be asked again: code evaluates your rule "
              "at every scheduling point and applies the policy it selects. Write the rule you would "
              "defend across a whole day of this workload, not one moment of it.\n"
              "Each branch tests ONE field against ONE number. The first matching branch wins, so "
              "order them from the most specific condition to the least. Use only the fields listed; "
              "a field you invent makes the rule invalid and it will be refused.\n"
              "Reply with JSON only:\n"
              '{"branches": [{"if": ["<field>", "<op>", <number>], '
              '"then": {"ordering": "<name>", "sizing": "<name>"}}, ...], '
              '"default": {"ordering": "<name>", "sizing": "<name>"}, '
              '"why": "one line for the shape of the rule"}')


def _packet_synth(ctx) -> str:
    """What a scheduler operator knows before the day starts: the machine, the menu, the vocabulary."""
    lines = [f"CLUSTER: {ctx.get('pool_n', 0)} GPUs, jobs are malleable and may be resized while running.",
             "", "FIELDS you may test (use these names EXACTLY):"]
    lines += [f"  {k}: {v}" for k, v in RULE_FIELDS.items()]
    lines += [f"  operators: {', '.join(RULE_OPS)}    at most {RULE_MAX_BRANCHES} branches", ""]
    lines += ["ORDERING MENU:"] + [f"  {k}: {v}" for k, v in POLICY_MENU["ordering"].items()]
    lines += ["", "SIZING MENU:"] + [f"  {k}: {v}" for k, v in POLICY_MENU["sizing"].items()]
    lines += ["", "CLUSTER RULES:", POLICY_RULES]
    return "\n".join(lines)


def arm_rule_synth(pending, free, ctx):
    """Synthesise the rule on the first call (or load a frozen one), then execute it deterministically."""
    from pins.correction import _ask
    rule = ctx.get("rule")
    if rule is None:
        ans = _ask(RULE_SYNTH, _packet_synth(ctx), ctx["model"], ctx["host"], ctx["cache"],
                   "es-rule-synth", num_predict=500)
        ctx["calls"] += 1
        ok, why = _validate_rule(ans)
        if ok:
            rule = ans
        else:                       # refused, not repaired: the fallback is a named constant
            ctx["rule_invalid"] = why
            rule = {"branches": [], "default": {"ordering": "firstfit", "sizing": "as_requested"}}
        ctx["rule"] = rule
        ctx["rule_text"] = ans
    o, z = _eval_rule(rule, _rule_state(pending, free, ctx), ctx)
    ctx["sizer"] = z
    ARMS[o](pending, free, ctx)


ARMS = {"fcfs": arm_fcfs, "firstfit": arm_firstfit, "easy": arm_easy, "sjf": arm_sjf,
        "declared_first": arm_declared_first,
        "tier_fcfs": arm_tier_fcfs, "tier_sjf": arm_tier_sjf,
        "market": arm_market,
        # Resize controls keep start selection at first-fit. `protect` is the zero-call structured
        # rule; `resize_debate` differs only by asking the three-role exception reviewer.
        "protect": arm_firstfit, "resize_single": arm_firstfit, "text_single": arm_firstfit,
        "text_debate": arm_firstfit,
        "policy_select": arm_policy_select,
        "rule_synth": arm_rule_synth,
        "resize_debate": arm_firstfit,
        "single": lambda p, f, c: arm_llm(p, f, c, "single"),
        "debate": lambda p, f, c: arm_llm(p, f, c, "debate"),
        "negotiate": lambda p, f, c: arm_llm(p, f, c, "negotiate"),
        "bo3": lambda p, f, c: arm_llm(p, f, c, "bo3"),
        "correct": lambda p, f, c: arm_llm(p, f, c, "correct"),
        "neg_signed": lambda p, f, c: arm_llm(p, f, c, "neg_signed"),
        "correct3": lambda p, f, c: arm_llm(p, f, c, "correct3"),
        "sham": lambda p, f, c: arm_llm(p, f, c, "sham"),
        "neg_v2": lambda p, f, c: arm_llm(p, f, c, "neg_v2")}


# ---------------------------------------------------------------- run
def run(world: Path, arm: str, model: str = "qwen2.5:14b", interval: int = 300, tag: str = "",
        est_default: int = 86400, quiet: bool = False, sizer: str = "as_requested",
        switch_at: float = 1.0, switch_on: str = "queue", sel_every: int = 1800,
        rule: Path | None = None, rule_out: Path | None = None,
        packet_order: str = "submit", gate: str = "scarcity", resize_cooldown: int = 600,
        resize_budget: int = CORRECT_BUDGET, text_exceptions: Path | None = None,
        text_labels: Path | None = None) -> dict:
    from elastisim_python import JobState, NodeState, pass_algorithm
    from pins.llm_agent import HOST
    world = world.resolve()
    tag = tag or arm
    stats = world / f"out/{tag}_job_statistics.csv"
    url = f"ipc:///tmp/es_{os.getpid()}_{tag}.ipc"
    cfg = world / f"out/{tag}_config.json"
    cfg.write_text(json.dumps({
        "jobs_file": str(world / "in/jobs.json"), "platform_file": str(world / "in/platform.xml"),
        "zmq_url": url, "schedule_on_job_submit": True, "schedule_on_job_finalize": True,
        "schedule_on_scheduling_point": True, "scheduling_interval": interval, "min_scheduling_interval": 0,
        "allow_oversubscription": False, "clip_evolving_requests": True, "forward_io_information": False,
        "sensing": False, "sensing_interval": 0, "log_task_times": False,
        "pfs_read_links": ["PFS_read"], "pfs_write_links": ["PFS_write"],
        "job_statistics": str(stats), "node_utilization": str(world / f"out/{tag}_node_util.csv")}))
    _wmeta = json.loads((world / "meta.json").read_text())
    ctx = {"par_frac": _wmeta.get("par_frac", 1.0),
           "model": model, "host": HOST, "cache": {}, "calls": 0, "fallbacks": 0, "critic_changed": 0, "trivial": 0,
           "invocations": 0, "now": 0.0,
           "transcript": open(world / f"out/{tag}_transcript.jsonl", "w"),   # one line per LLM decision
           "est": lambda j: (int(j.attributes["req_min"]) * 60) or est_default, "running": [],
           "sizer": sizer, "packet_order": packet_order, "gate": gate, "switch_at": switch_at,
           "switch_on": switch_on,
           "resize_events": 0, "resized_gpus": 0, "size_history": {}, "last_resize": {},
           "resize_cooldown": resize_cooldown, "resize_budget": resize_budget,
           "resize_cooldown_blocks": 0, "protected_events": 0,
           "sel_every": sel_every, "sel_ordering": "firstfit", "sel_sizing": "as_requested",
           "rule": json.loads(rule.read_text()) if rule else None}
    notes_path = text_exceptions or world / "in/text_exceptions.json"
    if arm in ("text_single", "text_debate") and notes_path.exists():
        note_doc = json.loads(notes_path.read_text())
        ctx["text_exceptions"] = {str(x["job_id"]): x for x in note_doc.get("notes", [])}
    fn = ARMS[arm]

    def schedule(jobs, nodes, system):
        ctx["invocations"] += 1; ctx["now"] = system["time"]
        pending = sorted((j for j in jobs if j.state == JobState.PENDING), key=lambda j: (j.submit_time, j.identifier))
        ctx["running"] = [j for j in jobs if j.state == JobState.RUNNING]
        # Waiting DEMAND against cluster size: the per-tick analogue of offered load, which is the
        # feature that predicted the per-window winner. Queue COUNT is not the same quantity.
        ctx["pending_demand"] = sum(max(1, int(j.attributes.get("req_nodes", 1))) for j in pending)
        ctx["pool_n"] = len(nodes)
        # Arrivals, so the packet can carry a load figure that does NOT read ~0 on an empty queue.
        # `queue_pressure` is instantaneous backlog; it collapses exactly when a quiet cluster is
        # about to be hit, which is when the long-horizon choice matters most.
        seen = ctx.setdefault("seen_jobs", set())
        for j in jobs:
            if j.identifier not in seen:
                seen.add(j.identifier)
                ctx.setdefault("arrivals", []).append(
                    (float(getattr(j, "submit_time", ctx["now"])),
                     max(1, int(j.attributes.get("req_nodes", 1)))))
        free = [n for n in nodes if n.state == NodeState.FREE]
        # A malleable job pauses at each work boundary and may change size before continuing. Keep
        # this deterministic: debate still decides waiting-job starts, never reconfiguration maths.
        if system["invocation_type"].name == "INVOKE_SCHEDULING_POINT":
            if arm in ("neg_v2", "resize_debate") and pending:
                _llm_resize_decide(system["job"], pending, free, ctx)
            elif arm == "resize_single" and pending:
                _llm_resize_single_decide(system["job"], pending, free, ctx)
            elif arm == "text_single" and pending:
                _llm_resize_single_decide(system["job"], pending, free, ctx, use_text=True)
            elif arm == "text_debate" and pending:
                _llm_resize_single_decide(system["job"], pending, free, ctx, use_text=True,
                                          debate=True)
            elif arm == "protect":
                _resize_with_deterministic_protection(system["job"], free, pending, ctx)
            else:
                _resize_malleable(system["job"], free, pending, ctx)
            return
        if pending and free:
            fn(pending, free, ctx)

    env = {**os.environ, "LD_LIBRARY_PATH": str(ES_ROOT / "env/lib")}
    log = open(world / f"out/{tag}_sim.log", "w")
    sim = subprocess.Popen([str(ES_BIN), str(cfg), "--log=root.thresh:warning"], env=env, stdout=log, stderr=log)
    time.sleep(1.5)
    if sim.poll() is not None:   # fail fast instead of blocking on the socket forever
        sys.exit(f"elastisim exited {sim.returncode}; see {log.name}")
    t = time.time()
    try:
        pass_algorithm(schedule, url)
    finally:
        sim.wait(timeout=60)
        ctx["transcript"].close()
    (world / f"out/{tag}_sizes.json").write_text(json.dumps({str(k): v for k, v in ctx.get("sizes", {}).items()}))
    if ctx.get("rule") is not None:
        (world / f"out/{tag}_rule.json").write_text(json.dumps(
            {"rule": ctx["rule"], "raw": ctx.get("rule_text"), "fires": ctx.get("rule_fires", {}),
             "invalid": ctx.get("rule_invalid")}, indent=1))
        if rule_out and not rule_out.exists():
            rule_out.write_text(json.dumps(ctx["rule"], indent=1))
    if ctx.get("sel_log"):
        (world / f"out/{tag}_policy_log.json").write_text(json.dumps(ctx["sel_log"], indent=1))
    (world / f"out/{tag}_size_history.json").write_text(json.dumps(
        {str(k): v for k, v in ctx["size_history"].items()}))
    res = summarise(stats, world)
    # Scored for EVERY arm, not just the text one. The planted exceptions exist in the world
    # whether or not an arm reads the notes, so the 0-call floor is the baseline to beat.
    semantic = text_exception_stats(world / f"out/{tag}_transcript.jsonl",
                                    text_labels or world / "in/text_exception_labels.json",
                                    world)
    llm_arms = ("single", "debate", "negotiate", "bo3", "correct", "neg_signed",
                "correct3", "sham", "neg_v2", "resize_single", "text_single", "resize_debate",
                "text_debate", "policy_select", "rule_synth")
    res.update(arm=arm, sizer=sizer, switch_at=switch_at, switch_on=switch_on,
               rule_frozen=bool(rule), rule_invalid=ctx.get("rule_invalid"),
               rule_fires=ctx.get("rule_fires", {}),
               market_clearings=ctx.get("market_clearings", 0),
               sel_every=sel_every, sel_invalid=ctx.get("sel_invalid", 0),
               sel_counts=ctx.get("sel_counts", {}), packet_order=packet_order, gate=gate,
               switch_adaptive=ctx.get("switch_adaptive", 0), switch_calls=ctx.get("switch_calls", 0),
               model=model if arm in llm_arms else None, interval=interval,
               packet=PACKET_VERSION if arm in llm_arms else None,
               est_default=est_default if arm == "easy" else None,
               **transcript_stats(world / f"out/{tag}_transcript.jsonl"),
               invocations=ctx["invocations"], llm_calls=ctx["calls"], fallbacks=ctx["fallbacks"],
               # NOT critic_changed: transcript_stats() already returns that key, and passing both
               # raises TypeError *after* the simulation has finished, discarding the whole run.
               critic_changed_applied=ctx["critic_changed"], trivial=ctx["trivial"],
               resize_events=ctx["resize_events"], resized_gpus=ctx["resized_gpus"],
               resize_cooldown=resize_cooldown, resize_budget=resize_budget,
               resize_cooldown_blocks=ctx["resize_cooldown_blocks"],
               protected_events=ctx["protected_events"],
               wall_s=round(time.time() - t),
               **ctx.get("easy_stats", {}),
               **semantic,
               **{f"correct_{k.replace(' ', '_')}": v
                  for k, v in ctx.get("correct_outcome", {}).items()})
    with open(world / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    if not quiet:
        print(json.dumps(res))
    return res


# Slowdown thresholds: a job "violates SLA-k" when turnaround > k x its TRUE runtime. The trace has
# no deadline field (29 columns, none of them a due date) and the only user-stated bound, timelimit,
# is 100-600x over-stated -- so this is a manufactured stand-in, and it is an ORACLE metric: true
# runtime is unknown to every arm at submit time, so no arm can target it directly. 10 is the PINS
# operating point (--slack-mult 10), inherited for comparability; 2 and 5 are reported alongside it
# so a result can be shown not to hinge on where the threshold was put.
SLACKS = (2, 5, 10)


def summarise(stats: Path, world: Path) -> dict:
    meta = json.loads((world / "meta.json").read_text())
    jobs = json.loads((world / "in/jobs.json").read_text())["jobs"]
    rows = list(csv.DictReader(open(stats)))
    sf = stats.with_name(stats.name.replace("_job_statistics.csv", "_sizes.json"))
    sizes = json.loads(sf.read_text()) if sf.exists() else {}
    hf = stats.with_name(stats.name.replace("_job_statistics.csv", "_size_history.json"))
    histories = json.loads(hf.read_text()) if hf.exists() else {}

    def allocation_seconds(jid, start, end, left=float("-inf"), right=float("inf")):
        """Integrate node count over time; old moldable runs fall back to their one chosen size."""
        events = histories.get(jid)
        if not events:
            g = sizes.get(jid, jobs[int(jid)].get("num_nodes", 1))
            return g * max(0.0, min(end, right) - max(start, left))
        total = 0.0
        for n, (t, g) in enumerate(events):
            seg_end = events[n + 1][0] if n + 1 < len(events) else end
            total += g * max(0.0, min(seg_end, end, right) - max(t, start, left))
        return total
    W = meta["hours"] * 3600
    a = meta.get("warmup_h", 0) * 3600            # the MEASURED interval is [a, a+W)
    wait, bsd, busy_win, gpu_s, late_req, n_req = [], [], 0.0, 0.0, 0, 0
    late = dict.fromkeys(SLACKS, 0)
    killed = 0
    alloc = 0.0        # GPU-seconds the scheduler SPENT: a decision once jobs are moldable
    tier = {t: {"n": 0, "wait": 0.0, **dict.fromkeys(SLACKS, 0)} for t in ("prod", "batch")}
    for r in rows:
        j = jobs[int(r["ID"])]
        sub, st, en, run, ta = (float(r["Submit Time"]), float(r["Start Time"]),
                                float(r["End Time"]), float(r["Makespan"]), float(r["Turnaround Time"]))
        # occupancy counts EVERY job running in the measured interval, warm-up included: those
        # nodes really are busy. Every other metric below is measured-window arrivals only.
        busy_win += allocation_seconds(r["ID"], st, en, a, a + W)
        if j["attributes"].get("_warmup"):
            continue
        wait.append(float(r["Wait Time"]))
        bsd.append(max(1.0, ta / max(10.0, run)))
        spent = allocation_seconds(r["ID"], st, en)
        gpu_s += spent
        alloc += spent
        # A job killed at its walltime finishes EARLY, so every wait-based metric flatters it.
        # Downsizing a malleable job can push it past its limit, so this must be first-class.
        dead = r["Status"] != "completed"
        killed += dead
        for k in SLACKS:
            late[k] += dead or ta > k * run
        t = tier.get(j["attributes"].get("tier"))
        if t is not None:
            t["n"] += 1; t["wait"] += wait[-1]
            for k in SLACKS:
                t[k] += dead or ta > k * run
        if j["walltime"] > 0:                        # TURNAROUND vs the requested limit -- not an ElastiSim kill
            n_req += 1                               # (only completed jobs are replayed, so no job ever hits it)
            late_req += ta > j["walltime"]
    scored = [r for r in rows if not jobs[int(r["ID"])]["attributes"].get("_warmup")]
    if not scored:
        return {"n": 0}
    span = max(float(r["End Time"]) for r in scored) - min(float(r["Submit Time"]) for r in scored)
    wait.sort()
    return {"n": len(scored), "completed": sum(r["Status"] == "completed" for r in scored),
            "n_warmup": len(rows) - len(scored),
            "killed_pct": round(100 * killed / len(scored), 1),
            "alloc_gpu_h": round(alloc / 3600, 1),
            **{f"sla{k}_viol_pct": round(100 * late[k] / len(scored), 1) for k in SLACKS},
            "ta_over_req_limit_pct": round(100 * late_req / max(1, n_req), 1), "n_with_req_limit": n_req,
            "util_win": round(busy_win / (meta["pool"] * W), 3), "util_span": round(gpu_s / (meta["pool"] * span), 3),
            "mean_wait_s": round(statistics.mean(wait)), "p50_wait_s": round(wait[len(wait) // 2]),
            "p90_wait_s": round(wait[int(0.9 * (len(wait) - 1))]), "max_wait_s": round(wait[-1]),
            "mean_bsd": round(statistics.mean(bsd), 2), "span_h": round(span / 3600, 1),
            **{f"{name}_{m}": v for name, t in tier.items() if t["n"] for m, v in
               [("n", t["n"]), ("mean_wait_s", round(t["wait"] / t["n"]))]
               + [(f"sla{k}_viol_pct", round(100 * t[k] / t["n"], 1)) for k in SLACKS]}}


COLS = ([f"sla{k}_viol_pct" for k in SLACKS]
        + ["killed_pct", "alloc_gpu_h", "ta_over_req_limit_pct", "util_win", "mean_wait_s", "p50_wait_s", "p90_wait_s",
           "max_wait_s", "mean_bsd"]
        + [f"prod_sla{k}_viol_pct" for k in SLACKS] + ["prod_mean_wait_s"]
        + [f"batch_sla{k}_viol_pct" for k in SLACKS] + ["batch_mean_wait_s"])


def _table(rows: list[tuple[str, dict]], cols=COLS, w=18) -> None:
    cols = [c for c in cols if any(c in r for _, r in rows)]
    print("arm".ljust(10) + "".join(c[-w + 1:].rjust(w) for c in cols))
    for name, r in rows:
        print(name.ljust(10) + "".join(str(r.get(c, "")).rjust(w) for c in cols))


def summary(world: Path) -> None:
    """Recompute every arm's metrics from its saved job_statistics.csv (no re-run needed)."""
    meta = json.loads((world / "meta.json").read_text())
    print("real (same jobs): " + " ".join(f"{k[5:]}={v}" for k, v in meta.items() if k.startswith("real_")))
    rows = [(s.name.replace("_job_statistics.csv", ""), summarise(s, world))
            for s in sorted(world.glob("out/*_job_statistics.csv")) if s.stat().st_size >= 100]  # skip running arms
    _table(rows)
    for name, _ in rows:
        ts = transcript_stats(world / f"out/{name}_transcript.jsonl")
        if ts:
            print(f"  {name} LLM wrapper: " + " ".join(f"{k}={v}" for k, v in ts.items()))


def census(hours: float, pool: int, min_jobs: int = 100, out: Path | None = None) -> list[dict]:
    """Every hourly window start described by WORKLOAD properties alone -- no scheduler is ever run.
    Window selection is made from this table, so it cannot see a result (design spec §3)."""
    rows = _trace_rows()
    subs = [r["submit"] for r in rows]
    cands = []
    for h in range(int((subs[-1] - subs[0] - hours * 3600) / 3600) + 1):
        t0 = subs[0] + h * 3600
        w = rows[bisect.bisect_left(subs, t0):bisect.bisect_left(subs, t0 + hours * 3600)]
        if len(w) < min_jobs:
            continue
        gh = sum(j["dur"] * j["gpus"] for j in w) / 3600
        d = sorted(j["dur"] for j in w)
        cands.append(dict(h=h, day=h // 24, off=h % 24, n=len(w), gpu_h=round(gh, 1),
                          demand=round(gh / hours, 1), load=round(gh / hours / pool, 2),
                          prod=round(sum(int(j["priority"] or 0) >= 100000 for j in w) / len(w), 3),
                          med_s=d[len(d) // 2], p90_s=d[int(0.9 * (len(d) - 1))],
                          g1=round(sum(j["gpus"] == 1 for j in w) / len(w), 2),
                          gmax=max(j["gpus"] for j in w)))
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    print(f"{len(cands)} candidate {hours:g}h windows with >= {min_jobs} jobs "
          f"(of {int((subs[-1]-subs[0]-hours*3600)/3600)+1} hourly starts, trace "
          f"{(subs[-1]-subs[0])/86400:.1f} days); load = demand / {pool} GPUs")
    for f in ("load", "prod"):
        v = [c[f] for c in cands]
        print(f"  {f:6} " + " ".join(f"p{p}={q(v, p/100):.2f}" for p in (5, 10, 25, 50, 75, 90, 95, 99)))
    if out:
        out.write_text(json.dumps(cands))
        print(f"  wrote {out}")
    return cands


BASELINE = "easy"   # EASY backfilling -- what a production batch system actually runs
DELTA_COLS = ([f"sla{k}_viol_pct" for k in SLACKS] + [f"prod_sla{k}_viol_pct" for k in SLACKS]
              + ["killed_pct", "alloc_gpu_h", "mean_wait_s", "p90_wait_s", "mean_bsd"])


def report(rows: dict[str, list[dict]], hours: float, note: str) -> None:
    """Three views of the same windows: the mean ± 95% CI, the spread across windows, and how OFTEN
    an arm beats the baseline -- a mean win carried by two outlier windows is not a win."""
    arms = sorted((a for a in rows if rows[a]), key=lambda a: (a != BASELINE, a))
    if not arms:
        print("no windows"); return
    ci = lambda v: (f"{statistics.mean(v):.0f}±{1.96 * statistics.stdev(v) / len(v) ** 0.5:.0f}"
                    if len(v) > 1 else f"{v[0]:.0f}")
    ci1 = lambda v: (f"{statistics.mean(v):.1f}±{1.96 * statistics.stdev(v) / len(v) ** 0.5:.1f}"
                     if len(v) > 1 else f"{v[0]:.1f}")
    fmt = lambda c: ci1 if "pct" in c or c == "mean_bsd" or "util" in c else ci
    n = len(rows[arms[0]])
    print(f"\n{n} windows, {hours:g} h, {note}; mean ± 95% CI across windows")
    _table([(a, {c: fmt(c)([r[c] for r in rows[a] if c in r])
                 for c in COLS if any(c in r for r in rows[a])}) for a in arms], w=16)

    print("\ndistribution across windows (p10 / p50 / p90)")
    pct = []
    for a in arms:
        d = {}
        for c in COLS:
            v = sorted(r[c] for r in rows[a] if c in r)
            if v:
                q = lambda p: v[int(p * (len(v) - 1))]
                d[c] = f"{q(.1):g}/{q(.5):g}/{q(.9):g}"
        pct.append((a, d))
    _table(pct, w=22)

    if BASELINE in arms and n > 1:
        base = {r["win"]: r for r in rows[BASELINE]}
        # pair only on windows both arms actually finished -- rows are appended per arm as they run
        pair = lambda a, c: [(r[c], base[r["win"]][c]) for r in rows[a]
                             if c in r and c in base.get(r["win"], {})]
        print(f"\npaired delta vs {BASELINE} (arm − {BASELINE}), mean ± 95% CI")
        _table([(a, {c: fmt(c)([x - y for x, y in pair(a, c)]) for c in DELTA_COLS if pair(a, c)})
                 for a in arms if a != BASELINE], cols=DELTA_COLS, w=16)
        print(f"\nwin-rate vs {BASELINE} (windows won / windows paired; ties count as losses)")
        wins = lambda c, p: sum((x > y) if "util" in c else (x < y) for x, y in p)   # only util is higher-is-better
        _table([(a, {c: f"{wins(c, pair(a, c))}/{len(pair(a, c))}" for c in COLS if pair(a, c)})
                 for a in arms if a != BASELINE], w=16)


def sweep(days: list[int], hours: float, load: float, arms: list[str], out: Path, min_jobs: int = 100,
          sample: int = 0, seed: int = 0, pool: int = 0, warmup_h: float = 0,
          manifest: Path | None = None) -> None:
    """Floors over many windows: build each (pool by offered load), run every arm, report the
    distribution across windows. With --sample N, N non-overlapping windows are drawn at RANDOM
    instead of walking the day grid (a step of 7 days is locked to one weekday, and always starts
    at midnight). N counts windows KEPT: candidates with < min_jobs jobs are redrawn, not counted
    -- half of all 12 h windows in this trace are that thin. LLM arms cost ~1 h/window each."""
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(manifest.read_text()) if manifest else None
    if man:            # the frozen pre-registered set: its parameters win over the CLI
        pool, hours, warmup_h = man["pool"], man["hours"], man["warmup_h"]
        print(f"manifest {manifest}: {len(man['windows'])} windows, pool {pool} GPUs, "
              f"{hours:g}h measured + {warmup_h:g}h warm-up, seed {man['seed']}")
    rows: dict[str, list[dict]] = {a: [] for a in arms}
    accepted: list[int] = []

    def candidates():
        if man:
            yield from ((w["day"], w["off"]) for w in man["windows"])
        elif not sample:
            yield from ((d, 0) for d in days)
        else:
            starts = _rand_starts(hours, seed, accepted)
            for _ in range(200 * sample):
                if len(accepted) >= sample:
                    return
                yield divmod(next(starts), 24)

    n_skip = 0
    for day, off in candidates():
        key = f"d{day}" if not (sample or man) else f"d{day}h{off}"
        meta = build(day, hours, pool, out / key, offset_h=off, load=load, warmup_h=warmup_h)
        if not meta or meta["n_jobs"] < min_jobs:
            print(f"  {key}: {meta.get('n_jobs', 0)} jobs, skipped"); n_skip += 1; continue
        accepted.append(day * 24 + off)
        for a in arms:
            r = run(out / key, a, quiet=True)
            r.update(win=key, day=day, offset_h=off)
            rows[a].append(r)
            with open(out / "sweep.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
    if sample:
        print(f"\ndrew {len(accepted)}/{sample} windows (seed {seed}); {n_skip} rejected for < {min_jobs} jobs")
    report(rows, hours, (f"pool {pool} GPUs fixed" if pool > 0 else f"pool sized to offered load {load}x")
           + (f", {warmup_h:g}h warm-up" if warmup_h else ""))


def sweep_report(out: Path, rebuild: bool = False) -> None:
    """Re-print a sweep's tables from sweep.jsonl. The rows reach disk before the report does, and
    a long sweep on this login node gets reaped between the two.
    --rebuild instead RECOMPUTES every row from the saved job_statistics.csv files, which is how a
    metric added to summarise() reaches runs that predate it -- no simulation is re-run."""
    seen: dict[tuple, dict] = {}
    if rebuild:
        for s in sorted(out.glob("d*/out/*_job_statistics.csv")):
            if s.stat().st_size < 100:               # an arm that died mid-run
                continue
            win, arm = s.parent.parent.name, s.name.replace("_job_statistics.csv", "")
            seen[(arm, win)] = {**summarise(s, s.parent.parent), "arm": arm, "win": win}
        print(f"rebuilt {len(seen)} arm-runs from saved job statistics")
    for line in (out / "sweep.jsonl").read_text().splitlines() if not rebuild else []:
        r = json.loads(line)
        r.setdefault("win", f"d{r.get('day')}")      # rows written before --sample existed
        seen[(r["arm"], r["win"])] = r               # last write wins: a re-run supersedes
    rows: dict[str, list[dict]] = {}
    for (arm, _), r in seen.items():
        rows.setdefault(arm, []).append(r)
    m = next((p for w in {r["win"] for rs in rows.values() for r in rs}     # a REPORTED window, not
              if (p := out / w / "meta.json").exists()), None)              # whatever glob finds first
    meta = json.loads(m.read_text()) if m else {}
    report(rows, meta.get("hours", 0), f"pool {meta.get('pool', '?')} GPUs"
           + (f", {meta['warmup_h']:g}h warm-up" if meta.get("warmup_h") else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--day", type=int, default=157); b.add_argument("--hours", type=float, default=24)
    b.add_argument("--pool", type=int, default=80); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--offset-h", type=float, default=0, help="window start offset within the day (12 = pm half)")
    b.add_argument("--warmup-h", type=float, default=12, help="hours of PRIOR arrivals, run but not scored")
    b.add_argument("--elastic-frac", type=float, default=0.0, help="fraction of jobs made MALLEABLE")
    b.add_argument("--par-frac", type=float, default=1.0,
                   help="Amdahl serial fraction s; 1=no scaling speedup, 0=linear speedup")
    b.add_argument("--max-scale", type=float, default=4.0, help="an elastic job may take up to max_scale x its observed size")
    b.add_argument("--elastic-seed", type=int, default=0)
    b.add_argument("--resize-points", type=int, default=10,
                   help="work chunks per malleable job; resizing is possible between chunks")
    r = sub.add_parser("run"); r.add_argument("--world", type=Path, required=True); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--model", default="qwen2.5:14b"); r.add_argument("--interval", type=int, default=300); r.add_argument("--tag", default="")
    r.add_argument("--est-default", type=int, default=86400, help="EASY runtime estimate (s) for jobs with no declared limit")
    r.add_argument("--sizer", choices=["as_requested", "greedy", "adaptive", "auto"], default="as_requested")
    r.add_argument("--rule", type=Path, help="rule_synth: execute this FROZEN rule instead of writing one")
    r.add_argument("--rule-out", type=Path, help="rule_synth: save the synthesised rule here")
    r.add_argument("--sel-every", type=int, default=1800,
                   help="policy_select: seconds between policy re-selections")
    r.add_argument("--switch-on", choices=["queue", "load"], default="queue",
                   help="auto sizer signal: queue depth over free GPUs, or waiting demand over pool")
    r.add_argument("--switch-at", type=float, default=1.0,
                   help="auto sizer: queue-depth/free-GPU ratio above which it shares capacity")
    r.add_argument("--packet-order", choices=["submit", "shuffled"], default="submit",
                   help="display order of the queue in the LLM packet; shuffled removes the arrival-order cue")
    r.add_argument("--gate", choices=["scarcity", "ambiguity", "wide"], default="scarcity",
                   help="when to call the LLM: scarcity = waiting work exceeds free (default); ambiguity = the anchor faces a real sizing choice")
    r.add_argument("--resize-cooldown", type=int, default=600,
                   help="minimum simulated seconds between accepted resizes of one job")
    r.add_argument("--resize-budget", type=int, default=CORRECT_BUDGET,
                   help="maximum GPUs moved from one job at a resize point")
    r.add_argument("--text-exceptions", type=Path,
                   help="scheduler-visible synthetic notes (text_single only)")
    r.add_argument("--text-labels", type=Path,
                   help="evaluation-only labels, opened after the run (text_single only)")
    b.add_argument("--load", type=float, default=0, help="size the pool by offered load when --pool 0")
    s = sub.add_parser("summary"); s.add_argument("--world", type=Path, required=True)
    w = sub.add_parser("sweep"); w.add_argument("--days", default="3:227:7", help="start:stop:step of window start days")
    w.add_argument("--hours", type=float, default=24); w.add_argument("--load", type=float, default=3.93)
    w.add_argument("--arms", default="easy,fcfs,firstfit,declared_first,sjf,tier_fcfs,tier_sjf")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--sample", type=int, default=0, help="draw N random non-overlapping windows instead of --days")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--pool", type=int, default=0, help="fixed pool; 0 = size by --load (makes load an OUTPUT)")
    w.add_argument("--warmup-h", type=float, default=0)
    w.add_argument("--manifest", type=Path, help="frozen window set (pins/windows12.json); its pool/hours/warmup win")
    c = sub.add_parser("census"); c.add_argument("--hours", type=float, default=24)
    c.add_argument("--pool", type=int, default=80); c.add_argument("--min-jobs", type=int, default=100)
    c.add_argument("--out", type=Path)
    sr = sub.add_parser("sweep-report"); sr.add_argument("--out", type=Path, required=True)
    sr.add_argument("--rebuild", action="store_true", help="recompute rows from saved job statistics")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.day, a.hours, a.pool, a.out, a.offset_h, a.load, a.warmup_h,
              a.elastic_frac, a.par_frac, a.max_scale, a.elastic_seed, a.resize_points)
    elif a.cmd == "summary":
        summary(a.world)
    elif a.cmd == "census":
        census(a.hours, a.pool, a.min_jobs, a.out)
    elif a.cmd == "sweep":
        sweep(list(range(*map(int, a.days.split(":")))), a.hours, a.load, a.arms.split(","), a.out,
              sample=a.sample, seed=a.seed, pool=a.pool, warmup_h=a.warmup_h, manifest=a.manifest)
    elif a.cmd == "sweep-report":
        sweep_report(a.out, a.rebuild)
    else:
        run(a.world, a.arm, a.model, a.interval, a.tag, a.est_default, sizer=a.sizer,
            switch_at=a.switch_at, switch_on=a.switch_on, sel_every=a.sel_every,
            rule=a.rule, rule_out=a.rule_out,
            packet_order=a.packet_order, gate=a.gate, resize_cooldown=a.resize_cooldown,
            resize_budget=a.resize_budget, text_exceptions=a.text_exceptions,
            text_labels=a.text_labels)
