"""ElastiSim bench: Supercloud GPU jobs replayed through ElastiSim, five scheduler arms.

    .venv/bin/python -m pins.elastisim_bench build --day 157 --hours 12 --pool 256 --out runs/es_d157
    .venv/bin/python -m pins.elastisim_bench run --world runs/es_d157 --arm fcfs|firstfit|sjf|single|debate

World (whole-GPU HPC batch; jobs are rigid by default, MOLDABLE under --elastic-frac):
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
          max_scale: float = 4.0, elastic_seed: int = 0) -> dict:
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
    # Elastic jobs are MOLDABLE: the scheduler picks the size at launch, fixed thereafter.
    # Amdahl anchored on the observed point -- a job seen at n0 nodes for `dur` seconds gets
    #     runtime(n) = dur * (s + (1-s)/n) / (s + (1-s)/n0)
    # so runtime(n0) == dur EXACTLY (the rigid replay is this model at the observed size) while
    # s=1 reproduces rigid at every size and s=0 gives linear speed-up. With ALL_RANKS ("total")
    # ElastiSim divides the work by the node count, so the flops FORMULA must be
    #     size(n) = n * FLOPS_PER_GPU * runtime(n) = a_par*n + a_ser,
    # which is linear in num_nodes -- exprtk substitutes num_nodes at assignment time.
    rng = random.Random(elastic_seed)
    elastic = {j["jid"]: (par_frac < 1.0 and rng.random() < elastic_frac) for j in jobs}

    def amdahl(j):
        n0, s = max(1, j["gpus"]), par_frac
        A = FLOPS_PER_GPU * j["dur"] / (s + (1 - s) / n0)
        return {"a_par": repr(A * s), "a_ser": repr(A * (1 - s))}

    (out / "in/application_model.json").write_text(json.dumps({
        "phases": [{"iterations": 1, "scheduling_point": False, "tasks": [
            {"type": "gpu", "name": "train", "flops": "flops", "computation_pattern": "uniform"}]}]},
        indent=1))
    (out / "in/application_model_elastic.json").write_text(json.dumps({
        "phases": [{"iterations": 1, "scheduling_point": False, "tasks": [
            {"type": "gpu", "name": "train", "flops": "a_par*num_nodes + a_ser",
             "computation_pattern": "total"}]}]}, indent=1))
    (out / "in/jobs.json").write_text(json.dumps({"jobs": [{
        **({"type": "moldable", "num_nodes_min": 1,
            "num_nodes_max": min(pool, max(1, int(max_scale * j["gpus"]))),
            "num_gpus_per_node_min": 1, "num_gpus_per_node_max": 1,
            "application_model": str(out / "in/application_model_elastic.json"),
            "arguments": amdahl(j)}
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
            "n_elastic": sum(elastic[j["jid"]] for j in scored),
            "n_prod": sum(j["priority"].isdigit() and int(j["priority"]) >= 100000 for j in scored),
            "gpu_hours": sum(j["dur"] * j["gpus"] for j in scored) / 3600, **real_stats(scored)}
    meta["offered_load"] = round(meta["gpu_hours"] / hours / pool, 2)
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"built {out}: {len(scored)} scored jobs (+{len(jobs)-len(scored)} warm-up), "
          f"pool {pool} GPUs, offered load {meta['offered_load']}x")
    return meta


# ---------------------------------------------------------------- arms
# A MOLDABLE job has no num_nodes -- only num_nodes_min/max -- so every arm makes two decisions:
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
    if getattr(job, "num_nodes", None) is None:
        job.assign_num_gpus_per_node(1)


def _start(job, free, ctx, pending_n: int = 1):
    k = _size(job, free, ctx, pending_n)
    job.assign(free[:k]); del free[:k]
    ctx.setdefault("sizes", {})[job.identifier] = k
    if getattr(job, "num_nodes", None) is None:      # moldable: GPUs per node is ours to set too
        job.assign_num_gpus_per_node(1)              # one node == one GPU in this world


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
# told jobs were moldable and was asked for bare ids, so it silently reverted every size the
# proposer chose back to the requested one (transcript: proposer emitted [id,gpus] pairs on
# 339/348 decisions, the critic on 23/348) -- it was deleting half the decision, not reviewing it.
WORLD = ("A job shown as `gpus=N` is rigid and needs exactly N GPUs for its whole run. A job shown "
         "as `gpus=LO-HI` is MOLDABLE: its size is chosen once, at start, and cannot change "
         "afterwards; `asked=` is the size its owner originally requested. Scaling is sublinear -- "
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
PACKET_CAP = 40
# v1 (day-157 single/debate runs): "req_min=0" for undeclared limits -- the 14b model read 0 as SHORTEST and
# returned the whole queue ignoring free_gpus (transcript autopsy 2026-09-03). v2 spells both out.
PACKET_VERSION = os.environ.get("ES_PACKET", "v2")


def _packet(pending, free, now):
    if PACKET_VERSION == "v1":
        lines = [f"now={int(now)}s free_gpus={len(free)} pending={len(pending)} (showing first {PACKET_CAP} by submit)"]
    else:
        lines = [f"now={int(now)}s free_gpus={len(free)} pending={len(pending)} (showing first {PACKET_CAP} by submit). "
                 f"You can start jobs totalling at most {len(free)} GPUs now; list only those, in start order."]
    for j in pending[:PACKET_CAP]:
        a = j.attributes
        # worlds built before the tier label keep the old priority field so old runs replay byte-identically
        cls = f"tier={a['tier']}" if "tier" in a else f"priority={a['priority']}"
        req = (f"req_min={a['req_min']}" if PACKET_VERSION == "v1" else
               (f"req_limit={a['req_min']}min" if int(a["req_min"]) else "req_limit=UNLIMITED(no estimate)"))
        # A MOLDABLE job has a legal RANGE, not a size. Reporting num_nodes_min here (as an earlier
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
    L = [json.loads(l) for l in open(path)]
    same = dropped = proposed = 0
    for x in L:
        rows = [l.split() for l in x["packet"].splitlines()[1:]]
        # a moldable job prints "gpus=LO-HI"; the first-fit reference below only needs the smallest
        # size that lets it start, so take LO. Rigid jobs print "gpus=N", where LO == N.
        lo = lambda tok: int(tok[5:].split("-")[0])
        ids = [int(r[0][3:]) for r in rows]; g = {int(r[0][3:]): lo(r[1]) for r in rows}
        free, ff = x["free"], []
        for i in ids:
            if g[i] <= free:
                ff.append(i); free -= g[i]
        same += x["picked"] == ff
        p = (x["proposal"] or {}).get("start") or []
        p = [e[0] if isinstance(e, (list, tuple)) and e else e for e in p]
        proposed += len(p); dropped += len(p) - len(x["picked"] or [])
    return {"decisions": len(L), "ids_proposed": proposed, "ids_dropped_by_validator": dropped,
            "invalid_answers": sum(x["picked"] is None for x in L),
            "critic_changed": sum(1 for x in L if x["critic"] and (x["critic"].get("start") != (x["proposal"] or {}).get("start"))),
            "pick_eq_firstfit_pct": round(100 * same / len(L), 1)}


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
        if isinstance(x, (list, tuple)) and len(x) == 2:
            x, n = x
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


def _llm_decide(pending, free, ctx, debate: bool):
    from pins.correction import _ask
    tiered = "tier" in pending[0].attributes
    system, critic = SYSTEM + (TIER_NOTE if tiered else ""), CRITIC + (TIER_NOTE if tiered else "")
    packet = _packet(pending, free, ctx["now"])
    ans = _ask(system, packet, ctx["model"], ctx["host"], ctx["cache"], "es-propose", num_predict=300)
    ctx["calls"] += 1
    if ctx["calls"] % 50 == 0:
        print(f"  {ctx['calls']} calls, sim t={ctx['now'] / 3600:.1f}h, {len(pending)} pending", flush=True)
    final = None
    picks = _validate(ans, pending, free)
    if debate:
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


def arm_llm(pending, free, ctx, debate=False):
    if sum(_sizes(j)[0] for j in pending) <= len(free):   # nothing to ration -> no call (the gate)
        ctx["trivial"] += 1
        return arm_firstfit(pending, free, ctx)
    picks = _llm_decide(pending, free, ctx, debate)
    # an EMPTY pick list idles the cluster while jobs fit -- never right here, and in practice it
    # means the answer failed to parse. Treat it as a fallback, not as a decision to hold.
    if not picks:
        ctx["fallbacks"] += 1
        return arm_firstfit(pending, free, ctx)
    for j, k in picks:
        _start_at(j, free, ctx, k)


ARMS = {"fcfs": arm_fcfs, "firstfit": arm_firstfit, "easy": arm_easy, "sjf": arm_sjf,
        "declared_first": arm_declared_first,
        "tier_fcfs": arm_tier_fcfs, "tier_sjf": arm_tier_sjf,
        "single": lambda p, f, c: arm_llm(p, f, c, False),
        "debate": lambda p, f, c: arm_llm(p, f, c, True)}


# ---------------------------------------------------------------- run
def run(world: Path, arm: str, model: str = "qwen2.5:14b", interval: int = 300, tag: str = "",
        est_default: int = 86400, quiet: bool = False, sizer: str = "as_requested") -> dict:
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
        "schedule_on_scheduling_point": False, "scheduling_interval": interval, "min_scheduling_interval": 0,
        "allow_oversubscription": False, "clip_evolving_requests": True, "forward_io_information": False,
        "sensing": False, "sensing_interval": 0, "log_task_times": False,
        "pfs_read_links": ["PFS_read"], "pfs_write_links": ["PFS_write"],
        "job_statistics": str(stats), "node_utilization": str(world / f"out/{tag}_node_util.csv")}))
    ctx = {"model": model, "host": HOST, "cache": {}, "calls": 0, "fallbacks": 0, "critic_changed": 0, "trivial": 0,
           "invocations": 0, "now": 0.0,
           "transcript": open(world / f"out/{tag}_transcript.jsonl", "w"),   # one line per LLM decision
           "est": lambda j: (int(j.attributes["req_min"]) * 60) or est_default, "running": [],
           "sizer": sizer}
    fn = ARMS[arm]

    def schedule(jobs, nodes, system):
        ctx["invocations"] += 1; ctx["now"] = system["time"]
        pending = sorted((j for j in jobs if j.state == JobState.PENDING), key=lambda j: (j.submit_time, j.identifier))
        ctx["running"] = [j for j in jobs if j.state == JobState.RUNNING]
        free = [n for n in nodes if n.state == NodeState.FREE]
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
    res = summarise(stats, world)
    res.update(arm=arm, sizer=sizer, model=model if arm in ("single", "debate") else None, interval=interval,
               packet=PACKET_VERSION if arm in ("single", "debate") else None,
               est_default=est_default if arm == "easy" else None,
               **transcript_stats(world / f"out/{tag}_transcript.jsonl"),
               invocations=ctx["invocations"], llm_calls=ctx["calls"], fallbacks=ctx["fallbacks"],
               # NOT critic_changed: transcript_stats() already returns that key, and passing both
               # raises TypeError *after* the simulation has finished, discarding the whole run.
               critic_changed_applied=ctx["critic_changed"], trivial=ctx["trivial"],
               wall_s=round(time.time() - t),
               **ctx.get("easy_stats", {}))
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
    sf = stats.with_name(stats.name.replace("_job_statistics.csv", "_sizes.json"))   # moldable: the
    sizes = json.loads(sf.read_text()) if sf.exists() else {}                        # chosen size
    W = meta["hours"] * 3600
    a = meta.get("warmup_h", 0) * 3600            # the MEASURED interval is [a, a+W)
    wait, bsd, busy_win, gpu_s, late_req, n_req = [], [], 0.0, 0.0, 0, 0
    late = dict.fromkeys(SLACKS, 0)
    killed = 0
    alloc = 0.0        # GPU-seconds the scheduler SPENT: a decision once jobs are moldable
    tier = {t: {"n": 0, "wait": 0.0, **dict.fromkeys(SLACKS, 0)} for t in ("prod", "batch")}
    for r in rows:
        j = jobs[int(r["ID"])]
        g, sub, st, en, run, ta = (sizes.get(r["ID"], j.get("num_nodes", 1)), float(r["Submit Time"]), float(r["Start Time"]),
                                   float(r["End Time"]), float(r["Makespan"]), float(r["Turnaround Time"]))
        # occupancy counts EVERY job running in the measured interval, warm-up included: those
        # nodes really are busy. Every other metric below is measured-window arrivals only.
        busy_win += g * max(0.0, min(en, a + W) - max(st, a))
        if j["attributes"].get("_warmup"):
            continue
        wait.append(float(r["Wait Time"]))
        bsd.append(max(1.0, ta / max(10.0, run)))
        gpu_s += run * g
        alloc += g * run
        # A job killed at its walltime finishes EARLY, so every wait-based metric flatters it.
        # Downsizing a moldable job can push it past its limit, so this must be first-class.
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
    b.add_argument("--elastic-frac", type=float, default=0.0, help="fraction of jobs made MOLDABLE")
    b.add_argument("--par-frac", type=float, default=1.0, help="Amdahl parallel fraction s; 1=rigid, 0=linear")
    b.add_argument("--max-scale", type=float, default=4.0, help="an elastic job may take up to max_scale x its observed size")
    b.add_argument("--elastic-seed", type=int, default=0)
    r = sub.add_parser("run"); r.add_argument("--world", type=Path, required=True); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--model", default="qwen2.5:14b"); r.add_argument("--interval", type=int, default=300); r.add_argument("--tag", default="")
    r.add_argument("--est-default", type=int, default=86400, help="EASY runtime estimate (s) for jobs with no declared limit")
    r.add_argument("--sizer", choices=["as_requested", "greedy", "adaptive"], default="as_requested")
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
              a.elastic_frac, a.par_frac, a.max_scale, a.elastic_seed)
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
        run(a.world, a.arm, a.model, a.interval, a.tag, a.est_default, sizer=a.sizer)
