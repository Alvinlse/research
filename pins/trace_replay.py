"""
Exp 28 — TRACE REPLAY: real Alibaba gpu-v2020 jobs drive the two-sided sim.

Exp 27 sampled only the per-job GPU cap from Stage-1 predictions; arrivals, durations and
their correlations stayed synthetic (`make_workload`). This replays contiguous windows of
REAL v2020 jobs so (arrival, duration, GPU demand) come JOINTLY from the trace:

  * ONE CLOCK: tick = TICK_S seconds for BOTH arrivals and durations, so the real
    arrival-vs-duration time relationship survives (an affine stretch of arrivals with a
    separate duration scale would destroy exactly the joint structure a replay is for),
  * arrival  = the job's real first-task start within a random 10-hour trace window,
  * work     = the job's real wall-clock duration in ticks (median ≈ 9 at TICK_S=120),
  * cap      = the job's real total requested GPU in quarter-GPU quanta
               (sum over tasks of plan_gpu*inst_num / 25), clipped to CAP_CLIP,
  * THINNING: the full PAI cluster (~6.5k GPUs) sees ~370 arrivals/hour; a 1-2 GPU pool
    obviously cannot replay them all, so we randomly sample n_jobs of the window's arrivals
    (thinning a near-Poisson stream preserves its statistics while scaling the load).

What the trace does NOT have — deadlines, urgency, tiers — keeps the EXACT `make_workload`
recipe (seeded), so the only change vs Exp 27 is the workload dynamics: a clean ablation.
The validated simulator and all four policies are imported from `two_sided_sim`, unmodified.

Needs `data/alibaba-gpu-v2020/replay_jobs.csv` (job_name, arrival, dur, quanta), built once:
sort pai_task_table Terminated GPU tasks by job, arrival=min start, dur=max end - min start.

Run:  .venv/bin/python -m pins.trace_replay                    # rule tier, no Ollama
      .venv/bin/python -m pins.trace_replay --llm --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

from pins.llm_agent import load_cache, save_cache
from pins.negotiation_sim import Job
from pins.two_sided_sim import (make_policy_isolated, make_policy_negotiated,
                                make_policy_single, policy_none, simulate,
                                simulate_backfill)
from pins.uncertainty_sim import assign, load_uncertainty_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_CSV = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020", "replay_jobs.csv")
PRED_CSV = os.path.join(HERE, "eval", "pred_job_gpu.csv")
USAGE_CSV = os.path.join(HERE, "eval", "pred_job_usage.csv")
MEM_CSV = os.path.join(HERE, "eval", "pred_job_mem.csv")
RUNTIME_CSV = os.path.join(HERE, "eval", "pred_job_runtime.csv")

TICK_S = 120         # one sim tick = 2 real minutes -> median trace job ≈ 9 ticks of work
# Second trace (Exp 43/44 external-validity caveat): MIT Supercloud slurm log — university
# HPC batch vs v2020's cloud PAI. Jobs are ~14x longer, so its tick is 900 s to keep median
# work ≈ 9 ticks: SAME sim regime, different world. Built by data/build_supercloud_replay.py.
TRACES = {"v2020": (REPLAY_CSV, TICK_S),
          "supercloud": (os.path.join(HERE, "..", "data", "supercloud", "replay_jobs.csv"), 900)}
WORK_CLAMP = (1, 60)
CAP_CLIP = 8         # quanta (= 2 GPUs); ~80% of trace jobs are below; keeps pools sane
ARRIVAL_FRAC = 0.6   # sample arrivals from the first 60% of the horizon (make_workload conv.)


def load_trace(path: str = REPLAY_CSV) -> list[tuple[int, int, int, str]]:
    """(arrival_s, dur_s, quanta, job_name) per job, sorted by real arrival.

    Sort key excludes job_name so tie order (stable = CSV order) is byte-identical to the
    Exp-28 3-tuple version — same seeds keep sampling the same windows/jobs."""
    with open(path) as f:
        rows = [(int(float(r["arrival"])), int(float(r["dur"])), int(r["quanta"]), r["job_name"])
                for r in csv.DictReader(f)]
    rows.sort(key=lambda r: r[:3])
    return rows


def load_predicted_quanta(path: str = PRED_CSV, quantile: str = "p50") -> dict[str, float]:
    """Exp 30/31: job_name -> Stage-1 predicted quanta (test-split jobs of predict_gpu).

    Exp 31: `quantile` picks WHERE on the predicted [p10,p90] interval the agents request —
    the newsvendor knob prediction intervals buy that a point predictor cannot: under-request
    starves the job, over-request hoards GPUs it cannot use. Coverage (the key set) is the
    same for every quantile, so windows — and therefore seeds — stay perfectly paired.

    'prod-p90' is the per-job newsvendor rule (Exp 31 finding 3): values become (p50, p90)
    tuples and make_trace_workload picks p90 for prod jobs, p50 for best-effort."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if quantile == "prod-p90":
        return {r["job_name"]: (float(r["p50"]), float(r["p90"])) for r in rows}
    return {r["job_name"]: float(r[quantile]) for r in rows}


def load_usage_quanta(path: str = USAGE_CSV, quantile: str = "p50"
                      ) -> tuple[dict[str, float], dict[str, float]]:
    """Exp 36: (pred, truth) usage-quanta maps from predict_gpu --target gpu_util.

    THE USAGE WORLD: the trace's plan_gpu is a DECLARATION, and Exp 35 measured how little it
    tracks reality (median ask 50% of a GPU vs median actual util 0.22%, rho 0.23). Here the
    job's TRUE need (progress denominator) becomes its measured usage; what changes between
    arms is only what the agents REQUEST: the declaration, the Stage-1 usage prediction, or
    the usage oracle. Same csv carries pred + truth, so all arms share one key set and seeds
    stay window-paired."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    truth = {r["job_name"]: float(r["truth"]) for r in rows}
    if quantile == "prod-p90":
        pred = {r["job_name"]: (float(r["p50"]), float(r["p90"])) for r in rows}
    else:
        pred = {r["job_name"]: float(r[quantile]) for r in rows}
    return pred, truth


def load_runtime_pred(path: str = RUNTIME_CSV) -> dict[str, float]:
    """Exp 38: job_name -> Stage-1 predicted runtime (s, P50) from predict_gpu --target runtime."""
    with open(path) as f:
        return {r["job_name"]: float(r["p50"]) for r in csv.DictReader(f)}


def make_trace_workload(trace, n_jobs: int, seed: int, horizon: int, pred=None, oracle=False,
                        true_map=None, declared=False, time_pred=None, time_mode=None,
                        tick: int = TICK_S, quantum: int = 1
                        ) -> tuple[list[Job], dict[str, int], dict[str, int],
                                   dict[str, float] | str | None]:
    """n_jobs real jobs thinned from a random 10-hour trace window, on ONE clock.

    Real (jointly, per job): arrival time within the window, duration, GPU quanta — all in
    the same TICK_S units, so burstiness vs job length is the trace's own. Synthetic (same
    seeded recipe as make_workload, because the trace has none): urgency, deadline, tier.

    Exp 30: `pred` = {job_name: predicted quanta} restricts the window to jobs the Stage-1
    predictor has predictions for (its test split) and returns cap_map = PREDICTED demand
    (what the agents request/negotiate over) next to true_cap_map = the trace's real demand
    (what the job actually needs to progress). `oracle=True` keeps the same restricted window
    but requests the truth — the matched-window control. pred=None: Exp 28/29, request==truth.

    Exp 36 (usage world): `true_map` = {job_name: usage quanta} replaces the trace's plan_gpu
    quanta as the TRUE need; `declared=True` makes the agents request the trace's plan_gpu
    declaration anyway (today's practice — the over-provisioning baseline).

    Exp 38 (time belief): `time_pred` = {job_name: predicted runtime s} restricts the window
    to runtime-covered jobs (all three time arms share it → seed-paired); `time_mode` picks
    the demand agent's deadline signal: "predicted" = belief map from time_pred, "blind" =
    no signal, "oracle" = true work on the same restricted windows. None = pre-Exp-38.

    Exp 48 (allocation quantum): `quantum` = pool units per trace quarter-GPU quantum
    divisor — 1 = the native quarter-GPU units (byte-identical to every prior tier), 4 =
    WHOLE-GPU units: a job's cap becomes max(1, round(quanta/4)), so every sub-GPU job
    rounds up to a full card and the pool/margins/reserve all move in whole GPUs. Window
    sampling is quantum-independent, so quarter/whole tiers stay seed-paired.
    """
    rng = random.Random(f"replay-{seed}")
    window_s = int(horizon * ARRIVAL_FRAC) * tick        # arrivals within first 60%
    t_lo, t_hi = trace[0][0], trace[-1][0] - window_s
    t0 = rng.randrange(t_lo, t_hi)
    in_win = [r for r in trace if t0 <= r[0] < t0 + window_s and (pred is None or r[3] in pred)
              and (time_pred is None or r[3] in time_pred)]
    if len(in_win) < n_jobs:                               # sparse stretch of the trace: reroll
        # BUG until Exp 37: this reroll dropped true_map/declared, silently reverting the
        # rerolled seed to the plan world (request==prediction, plan truth) in every arm —
        # 5/32 seeds in the Exp 36/37 sweeps. Forward EVERYTHING.
        return make_trace_workload(trace, n_jobs, seed + 7919, horizon, pred, oracle,
                                   true_map, declared, time_pred, time_mode, tick, quantum)
    window = sorted(rng.sample(in_win, n_jobs), key=lambda r: r[:3])   # thin the arrival stream

    jobs: list[Job] = []
    cap_map: dict[str, int] = {}
    true_cap_map: dict[str, int] = {}
    belief: dict[str, float] | str | None = "blind" if time_mode == "blind" else \
        {} if time_mode == "predicted" else None           # "oracle"/None -> true remaining
    clip = max(1, round(CAP_CLIP / quantum))               # same physical clip in pool units
    for i, (arr_s, dur_s, quanta, name) in enumerate(window):
        quanta = min(max(1, round(quanta / quantum)), clip)  # trace quanta -> pool units
        arrival = (arr_s - t0) // tick
        work = float(min(WORK_CLAMP[1], max(WORK_CLAMP[0], round(dur_s / tick))))
        urgency = round(rng.uniform(0.6, 2.2), 3)          # make_workload recipe verbatim
        slack = max(1.15, min(2.4, 2.5 - 0.65 * urgency))
        deadline = arrival + int(round(work * slack))
        tier = "prod" if urgency >= 1.667 else "besteffort"
        j = Job(f"r{i:02d}", arrival, ["train"], [work], urgency, deadline, tier)
        jobs.append(j)
        true_cap_map[j.jid] = min(quanta, CAP_CLIP) if true_map is None else \
            min(max(1, round(true_map[name])), CAP_CLIP)
        if declared:
            cap_map[j.jid] = min(quanta, CAP_CLIP)         # request the plan_gpu declaration
        elif pred is None or oracle:
            cap_map[j.jid] = true_cap_map[j.jid]
        else:
            p = pred[name]
            if isinstance(p, tuple):                       # prod-p90: hedge only prod jobs
                p = p[1] if tier == "prod" else p[0]
            cap_map[j.jid] = min(max(1, round(p)), CAP_CLIP)
        if time_mode == "predicted":                       # believed total work, same clamp as work
            belief[j.jid] = float(min(WORK_CLAMP[1],
                                      max(WORK_CLAMP[0], round(time_pred[name] / tick))))
    return jobs, cap_map, true_cap_map, belief


METRICS = ("sla", "prod_sla", "util", "slowdown", "finished", "fallback_rate")
RESULTS = os.environ.get("PINS_RESULTS", os.path.join(HERE, "results_trace_replay.json"))


def t95(df: int) -> float:
    """Two-sided 95% Student-t critical value (coarse table; df>=1)."""
    for lo, t in ((60, 2.000), (30, 2.042), (20, 2.086), (10, 2.228), (5, 2.571), (2, 4.303)):
        if df >= lo:
            return t
    return 12.71


def t90(df: int) -> float:
    """Two-sided 90% Student-t critical value (= one-sided 5%; the TOST CI)."""
    for lo, t in ((60, 1.671), (30, 1.697), (20, 1.725), (10, 1.812), (5, 2.015), (2, 2.920)):
        if df >= lo:
            return t
    return 6.314


def paired_ci(diffs: list[float], tcrit=t95) -> tuple[float, float]:
    """Mean paired difference and 95% CI half-width (pass tcrit=t90 for the TOST CI)."""
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, float("inf")
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    return m, tcrit(n - 1) * (var / n) ** 0.5


# Exp 43 TOST equivalence: 90% CI of the paired diff inside ±margin ⇒ the two arms are
# statistically EQUIVALENT within that margin at α=0.05 (stronger than a ns 95% CI, which
# is only failure-to-find-a-difference). Margin ±3 SLA pts pre-registered 2026-07-10
# (below the smallest effect of interest, the −4..−8 pt prodSLA protection); ±2 stricter.
EQ_MARGINS = (3.0, 2.0)


def tost_line(diffs_by_label: list[tuple[str, list[float]]]) -> str:
    """One 'TOST: ...' line for the given per-metric paired diffs (already in pts)."""
    parts = []
    for label, diffs in diffs_by_label:
        m, h = paired_ci(diffs, t90)
        tags = " ".join(f"EQ±{g:.0f}" for g in EQ_MARGINS if -g < m - h and m + h < g)
        parts.append(f"{label} 90%CI[{m-h:+5.1f},{m+h:+5.1f}] {tags or 'not-equiv'}")
    return "TOST: " + "   ".join(parts)


def print_paired_vs_floor(per_seed_pool: dict[str, list[dict]]) -> None:
    """Per-policy paired diffs vs the no-llm floor (same seed = same workload+spikes)."""
    floor = per_seed_pool["no-llm"]
    for name, rows_ in per_seed_pool.items():
        if name == "no-llm":
            continue
        parts = []
        for metric, label, pct in (("sla", "dSLA", True), ("prod_sla", "dprodSLA", True),
                                   ("util", "dutil", True), ("slowdown", "dslow", False)):
            diffs = [a[metric] - b[metric] for a, b in zip(rows_, floor)]
            m, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(m) else " "
            parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
        print(f"      {name:<12} vs floor:  " + "  ".join(parts))
    print(f"      (* = 95% CI excludes 0, paired by seed, n={len(floor)})")


def load_results() -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
        if "tiers" in data:
            return data
    return {"tiers": {}}


def cross_tier_stats(policy: str = "negotiated") -> None:
    """Paired-by-seed comparison of one policy across tiers (e.g. the 3b vs 14b inversion)."""
    tiers = load_results()["tiers"]
    tags = [t for t in tiers if policy in next(iter(tiers[t]["per_seed"].values()), {})]
    if len(tags) < 2:
        print(f"need >=2 tiers with per-seed data for '{policy}' in {RESULTS}; have {tags}")
        return
    pools = sorted({int(p) for t in tags for p in tiers[t]["per_seed"]})
    print(f"\nCross-tier paired stats for policy '{policy}' (95% CI, paired by seed):")
    for a in tags:
        for b in tags:
            if a >= b:
                continue
            print(f"  {a} - {b}:")
            for pool in pools:
                ra = tiers[a]["per_seed"].get(str(pool), {}).get(policy)
                rb = tiers[b]["per_seed"].get(str(pool), {}).get(policy)
                if not ra or not rb or len(ra) != len(rb):
                    continue
                parts, eq = [], []
                for metric, label, pct in (("sla", "dSLA", True), ("prod_sla", "dprodSLA", True),
                                           ("slowdown", "dslow", False)):
                    diffs = [x[metric] - y[metric] for x, y in zip(ra, rb)]
                    m, h = paired_ci(diffs)
                    u = 100.0 if pct else 1.0
                    sig = "*" if h < abs(m) else " "
                    parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
                    if pct:
                        eq.append((label, [d * u for d in diffs]))
                print(f"    pool {pool:>2} (n={len(ra)}):  " + "  ".join(parts))
                print(f"                    {tost_line(eq)}")


def compare_tiers(spec: str) -> None:
    """Exp 40/41: paired-by-seed deltas between two ARBITRARY tier/policy arms, e.g.
    --compare 'qwen2.5:3b+time-predicted/negotiated,easy+time-predicted/easy-pred'.
    Valid only between tiers built on the same windows (same suffix -> same seeds)."""
    (ta, pa), (tb, pb) = (s.rsplit("/", 1) for s in spec.split(","))
    tiers = load_results()["tiers"]
    A, B = tiers[ta]["per_seed"], tiers[tb]["per_seed"]
    print(f"\nPaired stats: {ta}/{pa}  MINUS  {tb}/{pb} (negative = first is better):")
    for pool in sorted(set(A) & set(B), key=int):
        ra, rb = A[pool].get(pa), B[pool].get(pb)
        if not ra or not rb or len(ra) != len(rb):
            continue
        parts, eq = [], []
        for metric, label, pct in (("sla", "dSLA", True), ("prod_sla", "dprodSLA", True),
                                   ("util", "dutil", True), ("slowdown", "dslow", False)):
            diffs = [x[metric] - y[metric] for x, y in zip(ra, rb)]
            m, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(m) else " "
            parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
            if pct:
                eq.append((label, [d * u for d in diffs]))
        print(f"  pool {pool:>2} (n={len(ra)}):  " + "  ".join(parts))
        print(f"                  {tost_line(eq)}")


def sweep(pools, n_jobs, horizon, seeds, scale, spike_max, use_llm, model,
          caps_mode: str = "real", quantile: str = "p50", truth_mode: str = "plan",
          time_mode: str | None = None, ttf_mode: str | None = None,
          baseline: str | None = None, dyncap: str | None = None,
          trace_name: str = "v2020", quantum: int = 1, referee: bool = False,
          manual: str | None = None) -> None:
    assert not (time_mode and ttf_mode), "--time and --ttf are separate experiments"
    assert quantum == 1 or (caps_mode == "real" and truth_mode == "plan" and
                            not (time_mode or ttf_mode or baseline or dyncap)), \
        "--quantum whole supports only the base world (caps in native trace quanta otherwise)"
    assert dyncap is None or (caps_mode == "predicted" and truth_mode == "plan" and
                              baseline is None), \
        "--dyncap is the Exp-30 world only: needs --caps predicted, plan truth, no baseline"
    # The Stage-1 prediction CSVs are keyed by v2020 job names — the second trace replays
    # the BASE world only (request == truth, oracle time), the Exp 28/29 recipe.
    assert trace_name == "v2020" or (caps_mode == "real" and truth_mode == "plan" and
                                     not (time_mode or ttf_mode or baseline or dyncap)), \
        "--trace supercloud supports only the base world (no pred/time/ttf/baseline/dyncap)"
    assert baseline is None or time_mode == "predicted", \
        "--baseline needs --time predicted (EASY's runtime estimate + window pairing)"
    trace_path, tick = TRACES[trace_name]
    trace = load_trace(trace_path)
    true_map = None
    declared = False
    if truth_mode != "plan":                  # Exp 36 usage / Exp 37 mem: true need measured
        path = USAGE_CSV if truth_mode == "usage" else MEM_CSV
        pred, true_map = load_usage_quanta(path, quantile=quantile)  # pred keys restrict windows
        declared = caps_mode == "real"        # 'real' here = request the plan_gpu declaration
    else:
        pred = load_predicted_quanta(quantile=quantile) if caps_mode != "real" else None
    oracle = caps_mode == "oracle"
    time_pred = load_runtime_pred() if (time_mode or ttf_mode) else None  # restricts windows too
    dist = load_uncertainty_distribution()
    cache: dict = load_cache()
    decisions: list = []
    seen: set = set()
    suffix = {"real": "", "predicted": "+pred", "oracle": "+oracle"}[caps_mode]
    if caps_mode == "predicted" and quantile != "p50":
        suffix = f"+pred-{quantile}"          # 'rule+pred' stays the Exp-30 P50 tier
    if truth_mode != "plan":
        suffix = ("+decl" if caps_mode == "real" else suffix) + "@" + truth_mode
    if time_mode:
        suffix += f"+time-{time_mode}"
    if ttf_mode:
        suffix += f"+ttf-{ttf_mode}"      # control (no signal, same windows) = the time-oracle tier
    if dyncap:
        suffix += f"+dyn-{dyncap}"        # Exp 45: telemetry-corrected dynamic cap
    if trace_name != "v2020":
        suffix += f"+{trace_name}"        # second trace: tiers must not collide with v2020
    if quantum != 1:
        suffix += "+qwhole"               # Exp 48: whole-GPU allocation quantum
    if referee:
        suffix += "+referee"              # Exp 50: referee-LLM allocator arm
    manual = manual or os.environ.get("PINS_MANUAL")   # flag and env are the same arm
    if manual:
        assert referee, "--manual is a referee-arm option"
        from pins.referee import load_manual
        load_manual(manual)               # Exp 51: precedent block + cache-key hash
        # tiers must never mix rulings across manuals or with vanilla (results merge per tier)
        suffix += ("+manual-learned" if "learned" in os.path.basename(manual) else "+manual")
    if n_jobs != 16:
        suffix += f"+n{n_jobs}"           # Exp 48: scale-up tiers must not collide
    tag = (baseline or ("rule" if not use_llm else model)) + suffix

    if baseline == "easy":        # Exp 40: rows are allocation DISCIPLINES, not policies
        rows = [("no-llm", lambda: policy_none), ("easy-pred", None), ("easy-oracle", None)]
    elif baseline == "qlearn":    # Exp 41: learned table in the LLM's own policy interface
        from pins.qlearn import load_table, make_policy_qlearn
        table = load_table()
        rows = [("no-llm", lambda: policy_none),
                ("qlearn", lambda: make_policy_qlearn(table))]
    elif referee:                 # Exp 50: referee-LLM decides vs the code-decides arms
        from pins.referee import make_policy_referee
        rows = [
            ("no-llm",     lambda: policy_none),
            ("referee",    lambda: make_policy_referee(use_llm, model, cache, decisions, seen,
                                                       statement_model="qwen2.5:3b")),
            ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen)),
        ]
    else:
        rows = [
            ("no-llm",     lambda: policy_none),
            ("isolated",   lambda: make_policy_isolated(use_llm, model, cache, decisions, seen)),
            ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen)),
            ("single-llm", lambda: make_policy_single(use_llm, model, cache, decisions, seen)),
        ]

    print(f"\n{'='*86}")
    print(f"TRACE REPLAY ({trace_name}) — two-sided sim on real jobs; agents={tag}")
    print(f"{'='*86}")
    print(f"{n_jobs} real jobs/window from {len(trace):,} trace jobs, horizon {horizon}, "
          f"mean of {len(seeds)} seeds (window per seed) | spike_max={spike_max} scale={scale} "
          f"| truth={truth_mode} caps={caps_mode} clip={CAP_CLIP}")
    header = (f"{'pool':>4}  {'policy':<12} {'SLA':>7} {'prodSLA':>8} {'util':>6} "
              f"{'slowdown':>9} {'fb':>6} {'done':>8}")
    all_per_seed: dict[str, dict[str, list[dict]]] = {}
    for gpus in pools:
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        per_seed_pool: dict[str, list[dict]] = {}
        for name, factory in rows:
            per_seed: list[dict] = []
            for s in seeds:
                # ttf 'predicted' reuses the belief-map construction; the map routes to the
                # SUPPLY signal (ttf_work) instead of the demand deadline belief (belief_work)
                mk_mode = time_mode or ("predicted" if ttf_mode == "predicted" else None)
                jobs, cap_map, tcap, belief = make_trace_workload(trace, n_jobs, s, horizon, pred,
                                                                  oracle, true_map, declared,
                                                                  time_pred, mk_mode, tick,
                                                                  quantum)
                cap_map = {k: min(v, gpus) for k, v in cap_map.items()}  # feasible at this pool
                tcap = {k: min(v, gpus) for k, v in tcap.items()}
                dyn_map = None
                if dyncap:   # Exp 45: telemetry cap = truth ('oracle') or a ±25% GBT-like read
                    drng = random.Random(f"dyn-{s}")     # same map across policy rows: paired
                    dyn_map = {k: v if dyncap == "oracle" else
                               min(max(1, round(v * drng.uniform(0.75, 1.25))), gpus)
                               for k, v in tcap.items()}
                u_map, spike_map = assign(jobs, s, dist, spike_max)
                ttf = "oracle" if ttf_mode == "oracle" else belief if ttf_mode == "predicted" else None
                if name.startswith("easy"):
                    r = simulate_backfill(jobs, gpus, horizon, u_map, spike_map, scale,
                                          spike_max, cap_map, true_cap_map=tcap,
                                          belief_work="oracle" if name == "easy-oracle"
                                          else belief)
                else:
                    r = simulate(jobs, factory(), gpus, horizon, u_map, spike_map, scale,
                                 spike_max, cap_map, true_cap_map=tcap,
                                 belief_work=belief if time_mode else None, ttf_work=ttf,
                                 dyn_cap_map=dyn_map)
                per_seed.append({k: r[k] for k in METRICS})
                if use_llm:
                    save_cache(cache)   # per-seed: a reaped run loses at most one seed
            per_seed_pool[name] = per_seed
            results.append((name, {k: sum(row[k] for row in per_seed) / len(seeds)
                                   for k in METRICS}))
        all_per_seed[str(gpus)] = per_seed_pool
        best_sla = min(r["sla"] for _, r in results)
        best_prod = min(r["prod_sla"] for _, r in results)
        for name, r in results:
            s1 = "*" if abs(r["sla"] - best_sla) < 1e-9 else " "
            p1 = "*" if abs(r["prod_sla"] - best_prod) < 1e-9 else " "
            print(f"{gpus:>4}  {name:<12} {r['sla']:>6.1%}{s1}{r['prod_sla']:>7.1%}{p1}"
                  f"{r['util']:>6.0%} {r['slowdown']:>9.2f} {r['fallback_rate']:>5.0%} "
                  f"{r['finished']:>4.1f}/{n_jobs:<3}")
        print_paired_vs_floor(per_seed_pool)
        print()
        if use_llm:
            save_cache(cache)

    data = load_results()   # merge per tier: rule / 3b / 14b runs no longer clobber each other
    prev = data["tiers"].get(tag)
    if prev:                 # resume: keep pools already done for this tier
        all_per_seed = {**prev["per_seed"], **all_per_seed}
        decisions = prev["decisions"] + decisions
    data["tiers"][tag] = {"use_llm": use_llm, "spike_max": spike_max, "scale": scale,
                          "cap_clip": CAP_CLIP, "n_seeds": len(seeds),
                          "per_seed": all_per_seed, "decisions": decisions}
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    if use_llm:
        save_cache(cache)
    print(f"{len(decisions)} distinct decisions/transcripts -> {RESULTS} (tier '{tag}')")


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace replay: real v2020 jobs in the two-sided sim")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--spike", type=float, default=0.6)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed (exclusive end stays --seeds); lets parallel "
                         "cache-warmers split one sweep into disjoint seed blocks")
    ap.add_argument("--pools", default="4,6,8", help="real caps are heavier (median 4 quanta)")
    ap.add_argument("--n-jobs", type=int, default=16,
                    help="jobs thinned per window (scale-up knob; tier gets a +nN suffix)")
    ap.add_argument("--quantum", default="quarter", choices=("quarter", "whole"),
                    help="Exp 48: smallest negotiable element — 'quarter' = the trace's native "
                         "quarter-GPU quanta (pool units = quanta, unchanged); 'whole' = whole "
                         "GPUs (caps round up to full cards, pool/margins/reserve in GPUs)")
    ap.add_argument("--stats", action="store_true",
                    help="no sim: cross-tier paired stats from results_trace_replay.json")
    ap.add_argument("--caps", default="real", choices=("real", "predicted", "oracle"),
                    help="Exp 30: agents request Stage-1 PREDICTED demand (dynamics stay true); "
                         "'oracle' = truth requested on the same prediction-covered windows")
    ap.add_argument("--quantile", default="p50", choices=("p10", "p50", "p90", "prod-p90"),
                    help="Exp 31: which predicted quantile the agents request (--caps predicted); "
                         "'prod-p90' = per-job rule: prod hedges to p90, best-effort stays p50")
    ap.add_argument("--truth", default="plan", choices=("plan", "usage", "mem"),
                    help="Exp 36: 'usage' = a job's TRUE need is its measured GPU usage "
                         "(pred_job_usage.csv); --caps then picks the request: real=the plan_gpu "
                         "DECLARATION, predicted=Stage-1 usage prediction, oracle=true usage. "
                         "Exp 37: 'mem' = need is peak GPU-memory residency in card quanta "
                         "(pred_job_mem.csv) — the pessimistic right-sizing truth")
    ap.add_argument("--ttf", default=None, choices=("predicted", "oracle"),
                    help="Exp 39: supply-side time-to-free — the reserve agent sees held GPUs of "
                         "running jobs predicted to free within TTF_HORIZON ticks ('release' "
                         "bucket). Control = the matching time-oracle tier (same windows, no "
                         "signal). 'oracle' = true remaining work, 'predicted' = Stage-1 runtime")
    ap.add_argument("--time", default=None, choices=("predicted", "blind", "oracle"),
                    help="Exp 38: the demand agent's deadline signal — 'predicted' = Stage-1 "
                         "runtime P50 (pred_job_runtime.csv), 'blind' = none (always ontrack), "
                         "'oracle' = true work on the same runtime-covered windows. Default "
                         "(absent) = pre-Exp-38 oracle on unrestricted windows")
    ap.add_argument("--referee", action="store_true",
                    help="Exp 50: swap the policy rows for the referee-LLM allocator vs the "
                         "no-llm floor and the negotiated (code-decides) arm; infeasible "
                         "referee ticks fall back to the floor and count in fb")
    ap.add_argument("--manual", default=None, metavar="PATH",
                    help="Exp 51: referee precedent manual (e.g. pins/referee_manual.md or "
                         "the self-authored pins/referee_manual_learned.md) — appended to the "
                         "referee's system prompt; tier gets '+manual'. PINS_MANUAL=<path> "
                         "is equivalent. Requires --referee")
    ap.add_argument("--baseline", default=None, choices=("easy", "qlearn"),
                    help="Exp 40/41: classical baselines instead of the LLM policies — "
                         "'easy' = FCFS + EASY backfilling (easy-pred uses the Stage-1 runtime "
                         "P50 as its reservation estimate, easy-oracle the true spiked work); "
                         "'qlearn' = the trained tabular-Q policy (pins/qlearn.py). "
                         "Requires --time predicted (pairs windows with the model tiers)")
    ap.add_argument("--trace", default="v2020", choices=tuple(TRACES),
                    help="which trace to replay: 'supercloud' = MIT Supercloud slurm log "
                         "(HPC batch, tick 900s), base world only — the second-trace "
                         "external-validity check")
    ap.add_argument("--dyncap", default=None, choices=("oracle", "noisy"),
                    help="Exp 45: dynamic cap — after a job has run 3 ticks, its allocation "
                         "base switches from the Stage-1 predicted request to a telemetry-"
                         "corrected estimate ('oracle' = true need, 'noisy' = truth ±25%%). "
                         "Requires --caps predicted (plan truth); user request stays fixed")
    ap.add_argument("--compare", default=None, metavar="TIER/POL,TIER/POL",
                    help="no sim: paired deltas between two tier/policy arms sharing windows, "
                         "e.g. 'qwen2.5:3b+time-predicted/negotiated,easy+time-predicted/easy-pred'")
    a = ap.parse_args()
    if a.stats:
        cross_tier_stats()
        return
    if a.compare:
        compare_tiers(a.compare)
        return
    sweep([int(p) for p in a.pools.split(",")], n_jobs=a.n_jobs, horizon=300,
          seeds=list(range(a.seed_start, a.seeds)), scale=a.scale, spike_max=a.spike,
          use_llm=a.llm, model=a.model, caps_mode=a.caps, quantile=a.quantile,
          truth_mode=a.truth, time_mode=a.time, ttf_mode=a.ttf, baseline=a.baseline,
          dyncap=a.dyncap, trace_name=a.trace, quantum={"quarter": 1, "whole": 4}[a.quantum],
          referee=a.referee, manual=a.manual)


if __name__ == "__main__":
    main()
