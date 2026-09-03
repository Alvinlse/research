"""ElastiSim bench: Supercloud GPU jobs replayed through ElastiSim, five scheduler arms.

    .venv/bin/python -m pins.elastisim_bench build --day 157 --hours 12 --pool 256 --out runs/es_d157
    .venv/bin/python -m pins.elastisim_bench run --world runs/es_d157 --arm fcfs|firstfit|sjf|single|debate

World (ponytail: whole-GPU HPC batch, nothing malleable yet):
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


def load_window(day: int, hours: float, offset_h: float = 0) -> list[dict]:
    rows = _trace_rows()
    t0 = rows[0]["submit"] + day * 86400 + int(offset_h * 3600)
    win = [dict(x) for x in rows if t0 <= x["submit"] < t0 + hours * 3600]   # copy: _trace_rows is cached
    for x in win:
        x["submit"] -= t0
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


def build(day: int, hours: float, pool: int, out: Path, offset_h: float = 0, load: float = 0.0) -> dict:
    """pool=0 with load>0 sizes the pool by offered load: pool = GPU-hours / hours / load (min 8)."""
    out = out.resolve()
    (out / "in").mkdir(parents=True, exist_ok=True)
    (out / "out").mkdir(exist_ok=True)
    jobs = load_window(day, hours, offset_h)
    if not jobs:
        return {}
    if pool <= 0:
        pool = max(8, round(sum(j["dur"] * j["gpus"] for j in jobs) / 3600 / hours / load))
    (out / "in/application_model.json").write_text(json.dumps({
        "phases": [{"iterations": 1, "scheduling_point": False, "tasks": [
            {"type": "gpu", "name": "train", "flops": "flops", "computation_pattern": "uniform"}]}]},
        indent=1))
    (out / "in/jobs.json").write_text(json.dumps({"jobs": [{
        "type": "rigid", "submit_time": j["submit"], "num_nodes": j["gpus"], "num_gpus_per_node": 1,
        "walltime": j["timelimit_min"] * 60,
        "application_model": str(out / "in/application_model.json"),
        "arguments": {"flops": j["dur"] * FLOPS_PER_GPU},
        "attributes": {"jid": j["jid"], "gpus": j["gpus"], "req_min": j["timelimit_min"],
                       # Slurm multifactor score: 10000-11000 is fairshare/age noise, the +100000 bump is a QoS
                       # class (9.1% of GPU jobs) -> that bit is the tier; the raw number is not shown to arms
                       "tier": "prod" if int(j["priority"] or 0) >= 100000 else "batch",
                       "priority": j["priority"], "partition": j["partition"], "user": j["user"],
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
    meta = {"day": day, "offset_h": offset_h, "hours": hours, "pool": pool, "n_jobs": len(jobs),
            "n_prod": sum(j["priority"].isdigit() and int(j["priority"]) >= 100000 for j in jobs),
            "gpu_hours": sum(j["dur"] * j["gpus"] for j in jobs) / 3600, **real_stats(jobs)}
    meta["offered_load"] = round(meta["gpu_hours"] / hours / pool, 2)
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"built {out}: {len(jobs)} jobs, pool {pool} GPUs, offered load {meta['offered_load']}x")
    return meta


# ---------------------------------------------------------------- arms
def _fit(job, free):
    return job.num_nodes <= len(free)


def _start(job, free):
    job.assign(free[:job.num_nodes]); del free[:job.num_nodes]


def arm_fcfs(pending, free, ctx):
    for job in pending:
        if not _fit(job, free):
            break
        _start(job, free)


def arm_firstfit(pending, free, ctx):
    for job in pending:
        if _fit(job, free):
            _start(job, free)


def arm_sjf(pending, free, ctx):
    key = lambda j: (int(j.attributes["req_min"]) or 10 ** 9, j.submit_time)
    arm_firstfit(sorted(pending, key=key), free, ctx)


def arm_easy(pending, free, ctx):
    """FCFS + EASY backfilling (Lifka 1995): the queue head gets a reservation at the shadow time
    (earliest moment enough GPUs are expected free, from running jobs' ESTIMATED ends); a later job may
    backfill only if it will finish before the shadow time or fits in the GPUs the head will not need.
    Estimates = requested walltime; undeclared (sentinel) limits use ctx['est_default'] -- a stated site
    default, because with an infinite estimate the shadow time is infinite and EASY collapses to first-fit."""
    now, est = ctx["now"], ctx["est"]
    i = 0
    while i < len(pending) and _fit(pending[i], free):
        _start(pending[i], free); i += 1
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
        if avail >= head.num_nodes:
            shadow, extra = t_end, avail - head.num_nodes
            break
    st["shadow_now"] += shadow <= now
    for j in pending[i + 1:]:
        if not _fit(j, free):
            continue
        if now + est(j) <= shadow:
            _start(j, free); st["backfilled"] += 1
        elif j.num_nodes <= extra:
            _start(j, free); extra -= j.num_nodes; st["backfilled"] += 1


def _prod_first(j):
    return j.attributes.get("tier") != "prod"


def _prod_reserving(order, free):
    """Prod jobs in order, strictly: a prod job that does not fit HOLDS the free GPUs (no batch backfill),
    otherwise 1-GPU batch jobs grab every single free GPU and a 2-GPU prod job never assembles a pair.
    Once every prod job is placed or the head prod job is blocked, batch jobs first-fit the remainder."""
    for job in order:
        if job.attributes.get("tier") == "prod":
            if not _fit(job, free):
                return
            _start(job, free)
    for job in order:
        if job.attributes.get("tier") != "prod" and _fit(job, free):
            _start(job, free)


def arm_tier_fcfs(pending, free, ctx):     # prod first (reserving), FCFS within tier
    _prod_reserving(sorted(pending, key=lambda j: (_prod_first(j), j.submit_time)), free)


def arm_tier_sjf(pending, free, ctx):      # prod first (reserving), requested-walltime SJF within tier
    key = lambda j: (_prod_first(j), int(j.attributes["req_min"]) or 10 ** 9, j.submit_time)
    _prod_reserving(sorted(pending, key=key), free)


SYSTEM = ("You are the batch scheduler of a GPU cluster. Jobs are rigid: a job needs exactly `gpus` "
          "GPUs for its whole run, and you only know its REQUESTED walltime (req_min, 0 = unlimited), "
          "not the true runtime. Goal: minimise mean waiting time and bounded slowdown while keeping "
          "GPUs busy. Reply with JSON only: {\"start\": [job ids in start order], \"why\": \"one line\"}.")
CRITIC = ("You are a second scheduler reviewing a colleague's start list for the same queue. Check "
          "it fits free_gpus, does not starve long-waiting jobs, and does not leave GPUs idle when a "
          "job fits. Return the FINAL list as JSON only: {\"start\": [job ids], \"why\": \"one line\"}.")
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
        lines.append(f"id={j.identifier} gpus={j.num_nodes} {req} "
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
        ids = [int(r[0][3:]) for r in rows]; g = {int(r[0][3:]): int(r[1][5:]) for r in rows}
        free, ff = x["free"], []
        for i in ids:
            if g[i] <= free:
                ff.append(i); free -= g[i]
        same += x["picked"] == ff
        p = (x["proposal"] or {}).get("start") or []
        proposed += len(p); dropped += len(p) - len(x["picked"] or [])
    return {"decisions": len(L), "ids_proposed": proposed, "ids_dropped_by_validator": dropped,
            "invalid_answers": sum(x["picked"] is None for x in L),
            "critic_changed": sum(1 for x in L if x["critic"] and (x["critic"].get("start") != (x["proposal"] or {}).get("start"))),
            "pick_eq_firstfit_pct": round(100 * same / len(L), 1)}


def _validate(ans, pending, free):
    """Feasibility only: keep ids that are pending and fit, in the LLM's order."""
    if not ans or not isinstance(ans.get("start"), list):
        return None
    by_id = {j.identifier: j for j in pending}
    picks = []
    for x in ans["start"]:
        try:
            j = by_id.get(int(x))
        except (TypeError, ValueError):
            continue
        if j and j not in picks and j.num_nodes <= len(free) - sum(p.num_nodes for p in picks):
            picks.append(j)
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
        "picked": None if picks is None else [j.identifier for j in picks]}) + "\n")
    ctx["transcript"].flush()
    return picks


def arm_llm(pending, free, ctx, debate=False):
    if sum(j.num_nodes for j in pending) <= len(free):   # nothing to ration -> no call (the gate)
        ctx["trivial"] += 1
        return arm_firstfit(pending, free, ctx)
    picks = _llm_decide(pending, free, ctx, debate)
    if picks is None:
        ctx["fallbacks"] += 1
        return arm_firstfit(pending, free, ctx)
    for j in picks:
        _start(j, free)


ARMS = {"fcfs": arm_fcfs, "firstfit": arm_firstfit, "easy": arm_easy, "sjf": arm_sjf,
        "tier_fcfs": arm_tier_fcfs, "tier_sjf": arm_tier_sjf,
        "single": lambda p, f, c: arm_llm(p, f, c, False),
        "debate": lambda p, f, c: arm_llm(p, f, c, True)}


# ---------------------------------------------------------------- run
def run(world: Path, arm: str, model: str = "qwen2.5:14b", interval: int = 300, tag: str = "",
        est_default: int = 86400, quiet: bool = False) -> dict:
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
           "est": lambda j: (int(j.attributes["req_min"]) * 60) or est_default, "running": []}
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
    res = summarise(stats, world)
    res.update(arm=arm, model=model if arm in ("single", "debate") else None, interval=interval,
               packet=PACKET_VERSION if arm in ("single", "debate") else None,
               est_default=est_default if arm == "easy" else None,
               **transcript_stats(world / f"out/{tag}_transcript.jsonl"),
               invocations=ctx["invocations"], llm_calls=ctx["calls"], fallbacks=ctx["fallbacks"],
               critic_changed=ctx["critic_changed"], trivial=ctx["trivial"], wall_s=round(time.time() - t),
               **ctx.get("easy_stats", {}))
    with open(world / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    if not quiet:
        print(json.dumps(res))
    return res


SLACK = 10   # project's rebased operating point (--slack-mult 10): deadline = submit + SLACK * true runtime


def summarise(stats: Path, world: Path) -> dict:
    meta = json.loads((world / "meta.json").read_text())
    jobs = json.loads((world / "in/jobs.json").read_text())["jobs"]
    rows = list(csv.DictReader(open(stats)))
    W = meta["hours"] * 3600                      # arrival window: util is measured here only
    wait, bsd, busy_win, gpu_s, late, late_req, n_req = [], [], 0.0, 0.0, 0, 0, 0
    tier = {"prod": {"n": 0, "late": 0, "wait": 0.0}, "batch": {"n": 0, "late": 0, "wait": 0.0}}
    for r in rows:
        j = jobs[int(r["ID"])]
        g, sub, st, en, run, ta = (j["num_nodes"], float(r["Submit Time"]), float(r["Start Time"]),
                                   float(r["End Time"]), float(r["Makespan"]), float(r["Turnaround Time"]))
        wait.append(float(r["Wait Time"]))
        bsd.append(max(1.0, ta / max(10.0, run)))
        gpu_s += run * g
        busy_win += g * max(0.0, min(en, W) - max(st, 0.0))
        late += ta > SLACK * run
        t = tier.get(j["attributes"].get("tier"))
        if t is not None:
            t["n"] += 1; t["late"] += ta > SLACK * run; t["wait"] += wait[-1]
        if j["walltime"] > 0:                        # TURNAROUND vs the requested limit -- not an ElastiSim kill
            n_req += 1                               # (only completed jobs are replayed, so no job ever hits it)
            late_req += ta > j["walltime"]
    span = max(float(r["End Time"]) for r in rows) - min(float(r["Submit Time"]) for r in rows)
    wait.sort()
    return {"n": len(rows), "completed": sum(r["Status"] == "completed" for r in rows),
            "sla10_viol_pct": round(100 * late / len(rows), 1),
            "ta_over_req_limit_pct": round(100 * late_req / max(1, n_req), 1), "n_with_req_limit": n_req,
            "util_win": round(busy_win / (meta["pool"] * W), 3), "util_span": round(gpu_s / (meta["pool"] * span), 3),
            "mean_wait_s": round(statistics.mean(wait)), "p50_wait_s": round(wait[len(wait) // 2]),
            "p90_wait_s": round(wait[int(0.9 * (len(wait) - 1))]), "max_wait_s": round(wait[-1]),
            "mean_bsd": round(statistics.mean(bsd), 2), "span_h": round(span / 3600, 1),
            **{f"{k}_{m}": v for k, t in tier.items() if t["n"] for m, v in
               (("n", t["n"]), ("sla10_viol_pct", round(100 * t["late"] / t["n"], 1)),
                ("mean_wait_s", round(t["wait"] / t["n"])))}}


COLS = ["sla10_viol_pct", "ta_over_req_limit_pct", "util_win", "mean_wait_s", "p50_wait_s", "p90_wait_s",
        "max_wait_s", "mean_bsd", "prod_sla10_viol_pct", "prod_mean_wait_s", "batch_sla10_viol_pct", "batch_mean_wait_s"]


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


BASELINE = "easy"   # EASY backfilling -- what a production batch system actually runs
DELTA_COLS = ["sla10_viol_pct", "prod_sla10_viol_pct", "mean_wait_s", "p90_wait_s", "max_wait_s", "mean_bsd"]


def report(rows: dict[str, list[dict]], hours: float, load: float) -> None:
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
    print(f"\n{n} windows, {hours} h, pool sized to offered load {load}x; mean ± 95% CI across windows")
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
          sample: int = 0, seed: int = 0) -> None:
    """Floors over many windows: build each (pool by offered load), run every arm, report the
    distribution across windows. With --sample N, N non-overlapping windows are drawn at RANDOM
    instead of walking the day grid (a step of 7 days is locked to one weekday, and always starts
    at midnight). N counts windows KEPT: candidates with < min_jobs jobs are redrawn, not counted
    -- half of all 12 h windows in this trace are that thin. LLM arms cost ~1 h/window each."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = {a: [] for a in arms}
    accepted: list[int] = []

    def candidates():
        if not sample:
            yield from ((d, 0) for d in days)
            return
        starts = _rand_starts(hours, seed, accepted)
        for _ in range(200 * sample):
            if len(accepted) >= sample:
                return
            yield divmod(next(starts), 24)

    n_skip = 0
    for day, off in candidates():
        key = f"d{day}h{off}" if sample else f"d{day}"
        meta = build(day, hours, 0, out / key, offset_h=off, load=load)
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
    report(rows, hours, load)


def sweep_report(out: Path) -> None:
    """Re-print a sweep's tables from sweep.jsonl. The rows reach disk before the report does, and
    a long sweep on this login node gets reaped between the two."""
    seen: dict[tuple, dict] = {}
    for line in (out / "sweep.jsonl").read_text().splitlines():
        r = json.loads(line)
        r.setdefault("win", f"d{r.get('day')}")      # rows written before --sample existed
        seen[(r["arm"], r["win"])] = r               # last write wins: a re-run supersedes
    rows: dict[str, list[dict]] = {}
    for (arm, _), r in seen.items():
        rows.setdefault(arm, []).append(r)
    m = next((p for w in {r["win"] for rs in rows.values() for r in rs}     # a REPORTED window, not
              if (p := out / w / "meta.json").exists()), None)              # whatever glob finds first
    meta = json.loads(m.read_text()) if m else {}
    report(rows, meta.get("hours", 0), meta.get("offered_load", 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--day", type=int, default=157); b.add_argument("--hours", type=float, default=12)
    b.add_argument("--pool", type=int, default=256); b.add_argument("--out", type=Path, required=True)
    b.add_argument("--offset-h", type=float, default=0, help="window start offset within the day (12 = pm half)")
    r = sub.add_parser("run"); r.add_argument("--world", type=Path, required=True); r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--model", default="qwen2.5:14b"); r.add_argument("--interval", type=int, default=300); r.add_argument("--tag", default="")
    r.add_argument("--est-default", type=int, default=86400, help="EASY runtime estimate (s) for jobs with no declared limit")
    b.add_argument("--load", type=float, default=0, help="size the pool by offered load when --pool 0")
    s = sub.add_parser("summary"); s.add_argument("--world", type=Path, required=True)
    w = sub.add_parser("sweep"); w.add_argument("--days", default="3:227:7", help="start:stop:step of window start days")
    w.add_argument("--hours", type=float, default=12); w.add_argument("--load", type=float, default=3.93)
    w.add_argument("--arms", default="fcfs,firstfit,easy,sjf,tier_fcfs,tier_sjf"); w.add_argument("--out", type=Path, required=True)
    w.add_argument("--sample", type=int, default=0, help="draw N random non-overlapping windows instead of --days")
    w.add_argument("--seed", type=int, default=0)
    sr = sub.add_parser("sweep-report"); sr.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.day, a.hours, a.pool, a.out, a.offset_h, a.load)
    elif a.cmd == "summary":
        summary(a.world)
    elif a.cmd == "sweep":
        sweep(list(range(*map(int, a.days.split(":")))), a.hours, a.load, a.arms.split(","), a.out,
              sample=a.sample, seed=a.seed)
    elif a.cmd == "sweep-report":
        sweep_report(a.out)
    else:
        run(a.world, a.arm, a.model, a.interval, a.tag, a.est_default)
