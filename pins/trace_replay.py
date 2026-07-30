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
import time

from pins.llm_agent import load_cache, save_cache, take_tokens
from pins.negotiation_sim import Job
from pins.two_sided_sim import (set_tick, make_policy_isolated, make_policy_negotiated,
                                make_policy_single, policy_none, simulate,
                                simulate_backfill)
from pins.uncertainty_sim import assign, load_uncertainty_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_CSV = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020", "replay_jobs.csv")
PRED_CSV = os.path.join(HERE, "eval", "pred_job_gpu.csv")
USAGE_CSV = os.path.join(HERE, "eval", "pred_job_usage.csv")
MEM_CSV = os.path.join(HERE, "eval", "pred_job_mem.csv")
RUNTIME_CSV = os.path.join(HERE, "eval", "pred_job_runtime.csv")

TICK_S = 120         # the REFERENCE tick every pre-Exp-97 tier was measured at (2 real minutes);
#                      the seconds-denominated constants below are stated against it
STD_TICK_S = 30      # Exp 97 standard: 30 s slices -> median trace job ~36 ticks of work, so a
#                      job's lifetime is ~36 decision points instead of ~9
# Second trace (Exp 43/44 external-validity caveat): MIT Supercloud slurm log — university
# HPC batch vs v2020's cloud PAI. Jobs are ~14x longer, so its tick is 900 s to keep median
# work ≈ 9 ticks: SAME sim regime, different world. Built by data/build_supercloud_replay.py.
TRACES = {"v2020": (REPLAY_CSV, STD_TICK_S),
          "supercloud": (os.path.join(HERE, "..", "data", "supercloud", "replay_jobs.csv"), 900)}
WORK_CLAMP = (1, 60)   # ticks, at the reference TICK_S=120 — derive with work_clamp(tick)
# ---- the tick is a RESOLUTION knob, not a workload change --------------------------------------
# These three were denominated in ticks, so shortening the tick used to shrink the arrival window,
# the simulated span and the longest representable job all at once (a 30 s tick quartered every one
# of them). In seconds they are invariant, and `--tick` only changes how finely the same 6 hours of
# arrivals and 13.3 hours of cluster time are sliced.
ARRIVAL_SPAN_S = int(300 * 0.6) * 120      # 6 h of arrivals   (= the old 180 ticks x 120 s)
HORIZON_S = 400 * 120                      # 13.3 h simulated  (= the old 400 ticks x 120 s)
WORK_CLAMP_S = (WORK_CLAMP[0] * 120, WORK_CLAMP[1] * 120)   # 2 min .. 2 h of real work per job


def work_clamp(tick: int) -> tuple[float, float]:
    """WORK_CLAMP in ticks of `tick` seconds. At tick=120 this is exactly the old (1, 60)."""
    return max(1.0, WORK_CLAMP_S[0] / tick), WORK_CLAMP_S[1] / tick
# ---- the Exp 97 STANDARD operating point (2026-07-30) --------------------------------------
# Defaults describe a cluster someone would actually run. The pre-Exp-97 defaults (pool 4/6/8 =
# 1-2 GPUs, horizon 300, slack x1) put the floor at 40-54% missed deadlines before any policy
# ran, and are reachable with explicit flags -- every such tier keeps its old name.
STD_POOL = 32          # quarter-GPU quanta = 8 GPUs, one node
LOAD_PER_QUANTUM = 3   # jobs per pool quantum: the last ratio where every job still finishes
HORIZON = 400          # ticks; arrivals stay in the first 180, the rest is drain (600 == 400)
SLACK_MULT = 10.0      # deadline slack multiplier; below ~8 the floor is deadline-bound, not
#                        policy-bound, and above ~10 the floor stops moving (starvation-bound)
CAP_CLIP = 8         # quanta (= 2 GPUs); ~80% of trace jobs are below; keeps pools sane
ARRIVAL_FRAC = 0.6   # sample arrivals from the first 60% of the horizon (make_workload conv.)
PROD_RATE = (2.2 - 1.667) / (2.2 - 0.6)   # Exp 96: the correlated rule's own prod marginal (1/3)


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
                        tick: int = TICK_S, quantum: int = 1, stratify: bool = False,
                        burst_s: int | None = None, slack_mult: float = 1.0,
                        decorrelate: bool = False
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

    `burst_s` (arrival SURGE): half the window's jobs are re-stamped into one random
    burst_s-second interval, the rest keep the trace's own arrival times — so a single seed
    contains both a quiet baseline and a contended spike. Drawn from its own rng stream, so
    every non-burst tier keeps byte-identical windows.

    Phase A (`stratify=True`, manual_author only): replace the uniform draw with a
    within-window stratification — systematic over trace GPU-quanta (base_gpus spread) and a
    fixed prod count with band-spread urgencies (tier + deadline spread). OFF by default so
    every eval tier keeps its byte-identical seed-paired windows.
    """
    rng = random.Random(f"replay-{seed}")
    # Arrival span is pinned to the 300-tick default (byte-identical there) so that --horizon adds
    # DRAIN time only; scaling it with the horizon would thin arrivals and change contention too.
    window_s = ARRIVAL_SPAN_S                             # 6 h of arrivals, whatever the tick
    t_lo, t_hi = trace[0][0], trace[-1][0] - window_s
    t0 = rng.randrange(t_lo, t_hi)
    in_win = [r for r in trace if t0 <= r[0] < t0 + window_s and (pred is None or r[3] in pred)
              and (time_pred is None or r[3] in time_pred)]
    if len(in_win) < n_jobs:                               # sparse stretch of the trace: reroll
        # BUG until Exp 37: this reroll dropped true_map/declared, silently reverting the
        # rerolled seed to the plan world (request==prediction, plan truth) in every arm —
        # 5/32 seeds in the Exp 36/37 sweeps. Forward EVERYTHING.
        return make_trace_workload(trace, n_jobs, seed + 7919, horizon, pred, oracle,
                                   true_map, declared, time_pred, time_mode, tick, quantum,
                                   stratify, burst_s, slack_mult, decorrelate)
    if stratify:
        # Phase A diversity (opt-in; eval path keeps rng.sample so its seeds stay byte-paired):
        # systematic draw over the trace's GPU-quanta distribution so each window spans
        # small..large base_gpus instead of an accidental all-similar-size clump.
        rows = sorted(in_win, key=lambda r: r[2])
        step = len(rows) / n_jobs
        window = sorted((rng.choice(rows[int(i * step):int((i + 1) * step)] or rows)
                         for i in range(n_jobs)), key=lambda r: r[:3])
    else:
        window = sorted(rng.sample(in_win, n_jobs), key=lambda r: r[:3])   # thin the arrival stream
    if burst_s:
        brng = random.Random(f"burst-{seed}")          # own stream: other tiers stay byte-paired
        t_b = brng.randrange(t0, t0 + max(1, window_s - burst_s))
        surge = set(brng.sample(range(len(window)), len(window) // 2))
        window = sorted(((t_b + brng.randrange(burst_s) if i in surge else r[0], *r[1:])
                         for i, r in enumerate(window)), key=lambda r: r[:3])
    if stratify:
        # tier + deadline diversity: fixed prod count (keeps the recipe's ~1/3 prod MEAN but
        # kills the binomial variance that yields all-besteffort windows) with urgencies
        # spread across each tier's band, so deadline tightness (behind/ontrack/ahead) also
        # spans its range within every window. Shuffled to decorrelate from arrival order.
        prod_n = round(n_jobs * (2.2 - 1.667) / (2.2 - 0.6))
        urg = [1.667 + (2.2 - 1.667) * (k + 0.5) / prod_n for k in range(prod_n)] + \
              [0.6 + (1.667 - 0.6) * (k + 0.5) / (n_jobs - prod_n)
               for k in range(n_jobs - prod_n)]
        rng.shuffle(urg)

    trng = random.Random(f"tier-{seed}")   # Exp 96: own stream -> other tiers stay byte-paired
    jobs: list[Job] = []
    cap_map: dict[str, int] = {}
    true_cap_map: dict[str, int] = {}
    belief: dict[str, float] | str | None = "blind" if time_mode == "blind" else \
        {} if time_mode == "predicted" else None           # "oracle"/None -> true remaining
    clip = max(1, round(CAP_CLIP / quantum))               # same physical clip in pool units
    for i, (arr_s, dur_s, quanta, name) in enumerate(window):
        quanta = min(max(1, round(quanta / quantum)), clip)  # trace quanta -> pool units
        arrival = (arr_s - t0) // tick
        wc_lo, wc_hi = work_clamp(tick)
        work = float(min(wc_hi, max(wc_lo, round(dur_s / tick))))
        urgency = round(urg[i] if stratify else rng.uniform(0.6, 2.2), 3)  # make_workload recipe
        slack = max(1.15, min(2.4, 2.5 - 0.65 * urgency)) * slack_mult
        deadline = arrival + int(round(work * slack))
        # Exp 96: the default rule makes tier and deadline tightness ONE variable — prod <=>
        # urgency >= 1.667 <=> slack in [1.15, 1.42] — so "prod protection" and "tight-deadline
        # protection" are inseparable. `decorrelate` draws the label independently at the SAME
        # marginal, P(prod) = (2.2-1.667)/(2.2-0.6) = 1/3, from its own stream so every window
        # stays byte-identical to the correlated world and the two are exactly paired.
        tier = (("prod" if trng.random() < PROD_RATE else "besteffort") if decorrelate
                else ("prod" if urgency >= 1.667 else "besteffort"))
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
            belief[j.jid] = float(min(wc_hi, max(wc_lo, round(time_pred[name] / tick))))
    return jobs, cap_map, true_cap_map, belief


METRICS = ("sla", "prod_sla", "tight_sla", "util", "slowdown", "finished", "fallback_rate",
           "churn_gpu", "churn_jobs", "wait", "wait_p95", "wait_full",
           # elevated plan §4/§14: occupancy is NOT productive work, so report useful
           # utilisation and waste separately, plus regret vs the per-tick oracle.
           "u_useful", "u_waste", "regret",
           # §5 normalised lateness; §16 starvation tail + Jain share fairness
           "lateness", "starved", "wait_max", "jain")
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
                                   ("util", "dutil", True), ("u_useful", "duseful", True),
                                   ("regret", "dregret", True), ("slowdown", "dslow", False)):
            diffs = [a[metric] - b[metric] for a, b in zip(rows_, floor)]
            m, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(m) else " "
            parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
        print(f"      {name:<12} vs floor:  " + "  ".join(parts))
    print(f"      (* = 95% CI excludes 0, paired by seed, n={len(floor)})")


# Elevated plan §18: the vs-floor block is a FAMILY of tests (arms x metrics). Reporting each
# at alpha=.05 inflates the family error rate — with 3 arms x 6 metrics, ~1 in 4 runs shows a
# spurious star. Holm is the plan's prescribed correction: uniformly more powerful than
# Bonferroni, and valid under any dependence structure (which paired scheduling metrics have
# in abundance — SLA, lateness and regret all move with the same seed's contention).
HOLM_METRICS = (("sla", "SLA"), ("prod_sla", "prodSLA"), ("util", "util"),
                ("u_useful", "useful"), ("regret", "regret"), ("lateness", "lateness"))


def _paired_p(diffs: list[float]) -> tuple[float, str]:
    """Paired p-value: t-test when the differences look normal, Wilcoxon otherwise (plan §18)."""
    from scipy import stats
    d = [x for x in diffs]
    if len(d) < 3 or max(d) - min(d) < 1e-12:
        return (1.0, "degenerate")
    normal = len(d) < 8 or stats.shapiro(d).pvalue > 0.05
    if normal:
        return (float(stats.ttest_1samp(d, 0.0).pvalue), "t")
    return (float(stats.wilcoxon(d).pvalue), "wilcoxon")


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment. Returns {key: adjusted p}, monotone by construction."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        out[k] = running
    return out


def print_holm(per_seed_pool: dict[str, list[dict]]) -> None:
    """The vs-floor family, Holm-corrected — which stars survive multiplicity (plan §18)."""
    floor = per_seed_pool.get("no-llm")
    if not floor or len(floor) < 3:
        return
    raw, tests = {}, {}
    for name, rows_ in per_seed_pool.items():
        if name == "no-llm":
            continue
        for metric, label in HOLM_METRICS:
            if metric not in rows_[0]:
                continue
            diffs = [a[metric] - b[metric] for a, b in zip(rows_, floor)]
            p, how = _paired_p(diffs)
            raw[(name, label)] = p
            tests[(name, label)] = how
    if not raw:
        return
    adj = holm(raw)
    survivors = sorted((k for k, v in adj.items() if v < 0.05), key=lambda k: adj[k])
    print(f"      Holm (family of {len(raw)} vs-floor tests): "
          + (", ".join(f"{a}/{m} p={adj[(a, m)]:.3f}[{tests[(a, m)]}]" for a, m in survivors)
             if survivors else "NOTHING survives correction"))


def print_budget(per_seed_pool: dict[str, list[dict]]) -> None:
    """Elevated plan §13: the inference bill of each arm, so an outcome difference can be read
    against the budget that bought it. C_LLM = calls + tokens + wall-clock, reported in full
    rather than collapsed into one weighted number — the weights (a, b, c) are venue-specific,
    the three components are not."""
    if not any(row.get("llm_calls") for rows_ in per_seed_pool.values() for row in rows_):
        return                                   # rule tier / cache-only run: no bill to show
    print(f"      {'arm':<12} {'calls/seed':>11} {'tokens/seed':>12} {'wall s/seed':>12}")
    for name, rows_ in per_seed_pool.items():
        n = max(len(rows_), 1)
        print(f"      {name:<12} {sum(r.get('llm_calls', 0) for r in rows_) / n:>11.1f} "
              f"{sum(r.get('llm_tokens', 0) for r in rows_) / n:>12.0f} "
              f"{sum(r.get('llm_wall', 0) for r in rows_) / n:>12.1f}")


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
                                   ("util", "dutil", True), ("u_useful", "duseful", True),
                                   ("regret", "dregret", True), ("slowdown", "dslow", False)):
            diffs = [x[metric] - y[metric] for x, y in zip(ra, rb)]
            m, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(m) else " "
            parts.append(f"{label} {m*u:+6.1f} ±{h*u:4.1f}{sig}")
            if pct:
                eq.append((label, [d * u for d in diffs]))
        print(f"  pool {pool:>2} (n={len(ra)}):  " + "  ".join(parts))
        print(f"                  {tost_line(eq)}")


def sweep(pools, n_jobs, horizon, seeds, scale, spike_max, use_llm, model, tick_s: int = 0,
          caps_mode: str = "real", quantile: str = "p50", truth_mode: str = "plan",
          time_mode: str | None = None, ttf_mode: str | None = None,
          baseline: str | None = None, dyncap: str | None = None,
          trace_name: str = "v2020", quantum: int = 1, referee: bool = False,
          manual: str | None = None, single_ilp: bool = False,
          no_think: bool = False, debate: bool = False,
          advocates: str = "qwen2.5:3b", realloc_cost: float = 0.0, alpha: float = 0.0,
          alpha_norm: str = "c0", trigger: str = "bucket", theta: float | None = None,
          stale: str = "fresh", extend: bool = False, no_argue: bool = False,
          prev_input: bool = False, fast_negotiate: bool = False,
          burst_s: int | None = None, hard_trigger: bool = False,
          slack_mult: float = 1.0, admit: bool = False,
          law: str = "amdahl", kappa: float = 2.0, samples: int = 0,
          cooldown: int = 0, resize_c1: float = 0.0, gamma: float | None = None,
          phi: float = 0.0, qcache: float | None = None, market: bool = False,
          gated: bool = False, corrected: bool = False, authored: str = "",
          decorrelate: bool = False,
          composed: bool = False, bid_info: str = "exact") -> None:
    assert not (referee and single_ilp), "--referee and --single-ilp are separate arms"
    assert all(m in ("narrated", "attributed") for m in authored.split(",") if m), \
        "--authored takes narrated and/or attributed"
    assert not debate or referee, "--debate is a referee-arm round"
    assert not (fast_negotiate and extend), "--fast-negotiate replaces the --extend replay path"
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
    tick = tick_s or tick                      # --tick overrides the trace's native resolution
    set_tick(tick)                             # keep the wall-clock thresholds wall-clock
    horizon = horizon or max(1, HORIZON_S // tick)   # 0 = the standard span at this resolution
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
    if theta is not None:
        # every controller-shell variant needs its own tier, or a later ungated run silently
        # overwrites the gated rows under a shared key (the Exp-59 unpaired-comparison trap)
        suffix += f"+theta{theta:g}"
    if extend:
        suffix += "+extend"
    if fast_negotiate:
        assert theta is not None, "--fast-negotiate needs --theta"
        suffix += "+fastneg"              # Exp 59: fast mode auctions instead of replaying
    if stale != "fresh":
        suffix += f"+stale-{stale}"
    if no_argue:
        suffix += "+noargue"
    if prev_input:
        suffix += "+previn"
    if advocates != "qwen2.5:3b":
        assert referee, "--advocates is a referee-arm option"
        suffix += "+adv" + advocates.split(":")[-1]   # tiers must not mix advocate scales
    if no_think:
        assert referee, "--no-think is a referee-arm ablation"
        suffix += "+nothink"              # hybrid reasoner with the thinking channel disabled
    if single_ilp:
        suffix += "+single-ilp"           # Exp 54: single LLM proposes, ILP repairs
    manual = manual or os.environ.get("PINS_MANUAL")   # flag and env are the same arm
    if manual:
        assert referee or gated, "--manual needs the referee or gated arm"
        from pins.referee import load_manual
        load_manual(manual)               # Exp 51: precedent block + cache-key hash
        # tiers must never mix rulings across manuals or with vanilla (results merge per tier)
        suffix += ("+manual-learned" if "learned" in os.path.basename(manual) else "+manual")
    if gated:
        suffix += "+gated"
    if corrected:
        suffix += "+corrected"   # Exp 92: the hard-case winner (signed correction) as the escalation
    if tick != TICK_S:
        # a different tick is a different sim: 4x the policy invocations over the same 13.3 h,
        # and every tick-denominated metric (wait, slowdown) changes units with it
        suffix += f"+t{tick}"
    if horizon != 300:
        # Exp 97: the horizon is part of a tier's identity, like slack_mult and n_jobs already are
        # (below). Without this a rebase at a longer drain lands on the same key and silently
        # overwrites rows it is not comparable with -- the Exp 59 trap with a new face.
        suffix += f"+h{horizon}"
    if decorrelate:
        # Exp 96: same jobs, same deadlines, only the prod LABEL is redrawn — must never merge
        # with the correlated rows, which are a different world under the same arm names
        suffix += "+decorr"
    if authored:
        # Exp 94: both modes belong in ONE tier — the placebo contrast is paired within seed
        suffix += "+authored-" + authored.replace(",", "-")
    if hard_trigger:
        assert debate, "--hard-trigger gates the debate round"
        suffix += "+hardtrig"             # E3: deliberation gated, ruling still per-tick
    if admit:
        suffix += "+admit"                # Exp 63: policy also rations ADMISSION, not just margin
    if slack_mult != 1.0:
        suffix += f"+slack{slack_mult:g}"   # deadline-tightness calibration knob
    if burst_s:
        suffix += f"+burst{burst_s}"      # arrival surge: separate windows, separate tier
    if n_jobs != 16:                      # Exp 48: scale-up tiers must not collide
        # Exp 97: 0 = AUTO (LOAD_PER_QUANTUM x pool). The count then differs per pool, so the tier
        # records the RATIO; the pool itself is already a key in the per_seed block.
        suffix += f"+n{LOAD_PER_QUANTUM}x" if not n_jobs else f"+n{n_jobs}"
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
                                                       statement_model=advocates,
                                                       think=not no_think, trigger=trigger,
                                                       theta=theta, stale=stale,
                                                       extend=extend, no_argue=no_argue,
                                                       prev_input=prev_input,
                                                       fast_negotiate=fast_negotiate,
                                                       admit=admit, qcache=qcache)),
            ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen, admit=admit)),
        ]
        if samples:               # plan §13: the EQUALLY FUNDED centralised control. One LLM
            # decides margin+reserve from the same joint state, but with `samples` independent
            # draws aggregated by median — the knob that buys it the referee arm's token budget.
            rows.append(("single-llm", lambda: make_policy_single(
                use_llm, model, cache, decisions, seen, samples=samples)))
        if debate:                # cross-talk round, same referee stage -> isolates the round.
            # Advocates run at `advocates` in BOTH rows: the round is the only difference, so a
            # debate win cannot be charged to a stronger advocate model.
            rows.insert(2, ("debate", lambda: make_policy_referee(
                use_llm, model, cache, decisions, seen, statement_model=advocates,
                think=not no_think, debate=True, trigger=trigger, theta=theta,
                stale=stale, extend=extend, fast_negotiate=fast_negotiate,
                hard_trigger=hard_trigger, admit=admit, gamma=gamma)))
    elif single_ilp:              # Exp 54: LLMSched spine — one LLM proposes, the ILP repairs
        from pins.two_sided_sim import make_policy_single_ilp
        rows = [
            ("no-llm",     lambda: policy_none),
            ("single-ilp", lambda: make_policy_single_ilp(use_llm, model, cache, decisions, seen)),
            ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen, admit=admit)),
        ]
    else:
        rows = [
            ("no-llm",     lambda: policy_none),
            ("isolated",   lambda: make_policy_isolated(use_llm, model, cache, decisions, seen)),
            ("negotiated", lambda: make_policy_negotiated(use_llm, model, cache, decisions, seen, admit=admit)),
            ("single-llm", lambda: make_policy_single(use_llm, model, cache, decisions, seen)),
        ]

    if market:                    # Exp 72 (plan §6): the explicit bid/ask market arm
        from pins.market import make_policy_market
        rows.append(("market", lambda: make_policy_market(decisions, seen)))
    if gated:                     # validated auction by default, debate arm on trigger
        from pins.market import make_policy_gated
        rows.append(("gated", lambda: make_policy_gated(use_llm, model, cache, decisions, seen,
                                                        debate=True)))
    if corrected:                 # Exp 92 boundary test: signed-correction escalation in-sim
        from pins.market import make_policy_corrected
        rows.append(("corrected", lambda: make_policy_corrected(use_llm, model, cache,
                                                                decisions, seen, debate=True)))
    if authored:                  # Exp 94: same escalation, notes written by the agents
        from pins.market import make_policy_authored
        for mode in authored.split(","):
            rows.append((mode, lambda m=mode: make_policy_authored(m, use_llm, model, cache,
                                                                   decisions, seen, debate=True)))
    if composed:                  # Exp 73: LLM sets the reserve, the market clears the rest
        from pins.market import make_policy_composed
        rows.append(("composed", lambda: make_policy_composed(use_llm, model, cache,
                                                              decisions, seen)))

    print(f"\n{'='*86}")
    print(f"TRACE REPLAY ({trace_name}) — two-sided sim on real jobs; agents={tag}")
    print(f"{'='*86}")
    jobs_of = (lambda q: n_jobs) if n_jobs else (lambda q: LOAD_PER_QUANTUM * q)
    print(f"{n_jobs or f'{LOAD_PER_QUANTUM}x pool'} real jobs/window from {len(trace):,} trace "
          f"jobs, tick {tick}s, horizon {horizon} ticks = {horizon*tick/3600:.1f} h "
          f"({ARRIVAL_SPAN_S//tick} arrival + drain), slack x{slack_mult:g}, "
          f"mean of {len(seeds)} seeds (window per seed) | spike_max={spike_max} scale={scale} "
          f"| truth={truth_mode} caps={caps_mode} clip={CAP_CLIP}")
    print(f"pool is in QUARTER-GPU QUANTA: {', '.join(f'{q}q = {q/4:g} GPU' for q in pools)}")
    header = (f"{'pool':>4}  {'policy':<12} {'SLA':>7} {'prodSLA':>8} {'tight':>6} {'util':>6} "
              f"{'useful':>7} {'regret':>7} {'slowdown':>9} {'wait':>6} {'fb':>6} {'done':>8}")
    all_per_seed: dict[str, dict[str, list[dict]]] = {}
    for gpus in pools:
        n_jobs = jobs_of(gpus)          # AUTO: load scales with the pool, or the explicit override
        print("-" * len(header)); print(header); print("-" * len(header))
        results = []
        per_seed_pool: dict[str, list[dict]] = {}
        for name, factory in rows:
            per_seed: list[dict] = []
            _t0 = time.time()
            for i, s in enumerate(seeds):
                # ttf 'predicted' reuses the belief-map construction; the map routes to the
                # SUPPLY signal (ttf_work) instead of the demand deadline belief (belief_work)
                mk_mode = time_mode or ("predicted" if ttf_mode == "predicted" else None)
                jobs, cap_map, tcap, belief = make_trace_workload(trace, n_jobs, s, horizon, pred,
                                                                  oracle, true_map, declared,
                                                                  time_pred, mk_mode, tick,
                                                                  quantum, burst_s=burst_s,
                                                                  slack_mult=slack_mult,
                                                                  decorrelate=decorrelate)
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
                                 dyn_cap_map=dyn_map, realloc_cost=realloc_cost, alpha=alpha,
                                 alpha_norm=alpha_norm, law=law, kappa=kappa,
                                 cooldown=cooldown, resize_c1=resize_c1, phi=phi,
                                 bid_info=bid_info)
                tk = take_tokens()          # marginal inference cost of THIS (seed, arm)
                from pins.referee import take_shell_stats, take_qcache_stats
                sh = take_shell_stats()     # fast/llm tick split of the controller shell
                qs = take_qcache_stats()    # §12 reuse + FALSE-reuse rate
                per_seed.append({k: r[k] for k in METRICS}
                                | {"llm_calls": float(tk["calls"]),
                                   "llm_tokens": float(tk["prompt"] + tk["completion"]),
                                   "llm_wall": float(tk.get("wall", 0.0)),
                                   "shell_fast": float(sh["fast"]),
                                   "shell_llm": float(sh["llm"]),
                                   "shell_debate": float(sh["debate"])} | qs)
                if use_llm:
                    save_cache(cache)   # per-seed: a reaped run loses at most one seed
                el, done = time.time() - _t0, i + 1
                eta = el / done * (len(seeds) - done)
                print(f"\r      {gpus}q/{gpus//4}gpu {name:<11} {done:>2}/{len(seeds)} seeds "
                      f"| {el:5.0f}s elapsed ~{eta:5.0f}s left", end="", flush=True)
            print()
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
                  f"{r['tight_sla']:>6.1%}"        # Exp 96: the laxity stratum, prod's de-confound
                  f"{r['util']:>6.0%} {r['u_useful']:>7.0%} {r['regret']:>7.0%}"
                  f" {r['slowdown']:>9.2f} {r['wait']:>6.1f} "
                  f"{r['fallback_rate']:>5.0%} "
                  f"{r['finished']:>4.1f}/{n_jobs:<3}")
        print_paired_vs_floor(per_seed_pool)
        print_holm(per_seed_pool)
        print_budget(per_seed_pool)
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
    # atomic replace: concurrent sweeps interleaving writes tore this file once (2026-07-22,
    # 4h of results nearly lost at merge). Last-write-wins can still drop a tier; torn JSON can't.
    with open(RESULTS + ".tmp", "w") as f:
        json.dump(data, f, indent=2)
    os.replace(RESULTS + ".tmp", RESULTS)
    if use_llm:
        save_cache(cache)
    print(f"{len(decisions)} distinct decisions/transcripts -> {RESULTS} (tier '{tag}')")
    def _rej_reasons(c: dict) -> str:
        """Exp 94 left ~96% of attributed's proposals rejected with the reason counted but never
        named. Splitting the tally says whether the causal channel OVER-REACHES (change budget /
        infeasible) or is merely UNFUNDABLE (cannot fund) — different follow-ups entirely.

        Counts REASONS, not rejections: one rejected ruling can trip two constraints, so these
        sum to >= `rejected`."""
        r = {k[4:].replace("_", " "): v for k, v in c.items() if k.startswith("rej_")}
        return ("  [" + ", ".join(f"{k} {v}" for k, v in sorted(r.items())) + "]") if r else ""

    if corrected:                 # Exp 92: did the escalation ever have text to act on?
        from pins.market import take_corr_stats
        c = take_corr_stats()
        print(f"corrected: {c['escalated']} escalations, {c['llm_calls']} LLM calls, "
              f"{c['proposals']} proposals, {c['changed']} ticks changed, {c['rejected']} rejected"
              + _rej_reasons(c))
    if authored:                  # Exp 94: notes/calls per arm — the budget match is the control
        from pins.market import take_auth_stats
        for mode, c in take_auth_stats().items():
            print(f"{mode}: {c['escalated']} escalations, {c['notes']} notes authored, "
                  f"{c['llm_calls']} LLM calls, {c['proposals']} proposals, "
                  f"{c['changed']} ticks changed, {c['rejected']} rejected" + _rej_reasons(c))
    if gated:                     # how often did the contextual trigger actually fire?
        from pins.market import take_gate_stats
        g = take_gate_stats()
        tot = g["auction"] + g["escalated"] or 1
        print(f"gate: {g['auction']} auction ticks, {g['escalated']} escalated "
              f"({g['escalated']/tot:.1%}), {g['escalated_invalid']} escalated rulings infeasible")


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
    ap.add_argument("--pools", default="32",
                    help="pool sizes in QUARTER-GPU QUANTA (4 quanta = 1 GPU). Default 32 = 8 "
                         "GPUs, one node — the Exp 97 standard. The old 4,6,8 default was 1-2 "
                         "physical GPUs, which no operator runs.")
    ap.add_argument("--n-jobs", type=int, default=0,
                    help="jobs thinned per window; 0 (default) = AUTO at LOAD_PER_QUANTUM x pool, "
                         "the last ratio at which every job still finishes (Exp 97 ladder). Set a "
                         "number to override; the tier then gets a +nN suffix.")
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
    ap.add_argument("--no-think", action="store_true",
                    help="referee ablation: run a hybrid reasoner with think=False (distinct tier+cache)")
    ap.add_argument("--single-ilp", action="store_true",
                    help="Exp 54: LLMSched-spine arm — ONE joint-objective LLM proposes "
                         "(margins, reserve), the evaluator verifies, an infeasible proposal "
                         "is min-edit REPAIRED by pins/ilp.py (fb column = repair rate)")
    ap.add_argument("--realloc-cost", type=float, default=0.0, metavar="F",
                    help="Exp 57: fraction of a tick's progress a job loses on any tick its "
                         "allocation changed (checkpoint/restart/reload). 0.0 = the zero-cost "
                         "upper bound Exp 22-56b assumed; try 0.1/0.25/0.5 as sensitivity.")
    ap.add_argument("--alpha", type=float, default=0.0, metavar="A",
                    help="Exp 57: diminishing-returns coefficient in P(g)=g/(1+A*(g-1)). "
                         "0.0 = the linear scaling Exp 22-56b assumed; try 0.1/0.3 as "
                         "sensitivity. Cannot be calibrated from v2020 (no counterfactuals).")
    ap.add_argument("--corrected", action="store_true",
                    help="Exp 92 boundary test: validated auction + the SIGNED-CORRECTION "
                         "escalation that wins the hard-case suite (Exp 83/88/89). It only acts "
                         "on jobs carrying text; the trace has none, so the prediction is a no-op.")
    ap.add_argument("--decorrelate", action="store_true",
                    help="Exp 96: draw the prod/besteffort label INDEPENDENTLY of urgency at the "
                         "same marginal (1/3), instead of prod <=> urgency >= 1.667. Breaks the "
                         "tie between tier and deadline tightness that makes 'prod-tier "
                         "protection' and 'tight-deadline protection' the same claim. Own RNG "
                         "stream, so windows stay byte-identical to the correlated world and the "
                         "two are exactly paired. Tier suffix +decorr.")
    ap.add_argument("--authored", default="", metavar="MODES",
                    help="Exp 94: add the agent-authored text channel — the Exp-92 escalation "
                         "run on notes the demand/supply agents write when their situation "
                         "changes. 'narrated' (placebo) states WHAT changed and may not say "
                         "why; 'attributed' states WHY. Pass both comma-separated so the "
                         "placebo contrast is paired within seed. Needs --llm.")
    ap.add_argument("--gated", action="store_true",
                    help="validated auction every tick; escalate to the debate arm only on a "
                         "contextual trigger (validator reject, prod arrival, job newly behind, "
                         "capacity crossing). Add --manual for the precedent docs")
    ap.add_argument("--market", action="store_true",
                    help="elevated plan S6: add the explicit-market arm — per-GPU marginal "
                         "bids (alpha*dp_hat + beta*SLA risk + gamma*wait - delta*resize) vs a "
                         "rising supply ask, cleared at b>=a. The unsold remainder IS the "
                         "reserve. Deterministic, no LLM.")
    ap.add_argument("--bid-info", default="exact", choices=("exact", "bucket"),
                    help="Exp 74 information ablation: what the market knows about a job's "
                         "usable extra parallelism. 'exact' = the true integer (Exp 72); "
                         "'bucket' = only the spike-risk bucket the LLM arms see. Dynamics "
                         "are unchanged either way.")
    ap.add_argument("--composed", action="store_true",
                    help="Exp 73: the composed arm — the LLM (or its rule fallback) sets ONLY "
                         "the reserve, the plan's market clears the remaining pool into "
                         "margins. Tests whether the reasoning layer's prod-tier protection "
                         "survives when it no longer controls the margin.")
    ap.add_argument("--qcache", type=float, default=None, metavar="THR",
                    help="elevated plan S12: quality-aware similarity cache on the referee arm. "
                         "Reuse a past ruling when sim*quality*age-decay > THR, adapted by job "
                         "category and RE-VALIDATED. Quality is scored on decision-time "
                         "signals only (fill, validator verdict, churn). Sweep .70/.80/.90/.95; "
                         "watch qcache_false, not the hit rate.")
    ap.add_argument("--phi", type=float, default=0.0, metavar="P",
                    help="elevated plan S16: starvation protection — add P*waiting_fraction to "
                         "a job's grant priority so long waits eventually win a tie. 0 = off.")
    ap.add_argument("--gamma", type=float, default=None, metavar="THETA",
                    help="elevated plan S8: gate the debate round on the SERIOUSNESS score "
                         "Gamma_t (mean of SLA risk, clearing ambiguity, uncertainty, "
                         "starvation, job-set churn) exceeding THETA, or any hard trigger. "
                         "Generalises --hard-trigger; sweep .1/.2/.3/.4/.5 per the plan.")
    ap.add_argument("--cooldown", type=int, default=0, metavar="K",
                    help="elevated plan S15: forbid a job's ceiling from moving within K ticks "
                         "of its last move. 0 (default) = unconstrained resizing, as in "
                         "Exp 22-68. Sweep 1/2/5 per the plan.")
    ap.add_argument("--resize-c1", type=float, default=0.0, metavar="C",
                    help="elevated plan S15: per-GPU term of the resize penalty, so the cost "
                         "is --realloc-cost + C*|delta g| instead of a flat fraction.")
    ap.add_argument("--samples", type=int, default=0, metavar="N",
                    help="elevated plan S13: add an EQUAL-BUDGET single-LLM control to a "
                         "--referee run, funded with N independent samples per decision "
                         "(median-aggregated). Pick N so its c_llm matches the referee arm's; "
                         "0 (default) omits the control entirely.")
    ap.add_argument("--law", default="amdahl", choices=("amdahl", "sat"),
                    help="counterfactual progress model (elevated plan S3): 'amdahl' = the "
                         "Exp 57 law all results through Exp 63 used (default, unchanged); "
                         "'sat' = the plan's primary saturating law p(g)=1-exp(-kappa*g). "
                         "A conclusion counts only if it survives BOTH.")
    ap.add_argument("--kappa", type=float, default=2.0, metavar="K",
                    help="scaling sharpness of --law sat; kappa_j = K/c0 so each job "
                         "saturates on the scale of its own base demand (rate==1 at c0). "
                         "Larger K = earlier saturation = margin GPUs buy less.")
    ap.add_argument("--alpha-norm", default="c0", choices=("c0", "unit"),
                    help="Exp 57: where the scaling law is normalised. 'c0' = rate 1.0 at the "
                         "job's base demand (alpha reprices margin only). 'unit' = the plan's "
                         "literal P(1) law, which also tightens every deadline. They disagree "
                         "in sign on world difficulty; report both.")
    ap.add_argument("--trigger", default="bucket", choices=("bucket", "delta"),
                    help="Exp 57: referee scene-cache key. 'bucket' keys every job_id "
                         "separately; 'delta' is role-indexed and identity-free, so a ruling "
                         "transfers to any scene of the same shape.")
    ap.add_argument("--theta", type=float, default=None, metavar="T",
                    help="Exp 57 controller shell: delta-trigger threshold (try 0.15). A tick "
                         "whose state change is below T re-executes the standing ruling "
                         "deterministically instead of re-invoking the referee. Method "
                         "verdict: the BUDGET variant (-4.5* at 1/7 the consultations); "
                         "per-tick (theta off) stays the headline method. Default off.")
    ap.add_argument("--extend", action="store_true",
                    help="Exp 57 shell: fast mode also awards ARRIVALS by the standing "
                         "ruling's own per-tier exemplar and rule-3/4 order; only departures "
                         "and new prod-behind jobs then fire the trigger.")
    ap.add_argument("--fast-negotiate", action="store_true",
                    help="Exp 59: fast mode (below theta) calls the cheap bounded-concession "
                         "negotiate() auction on the live tick instead of replaying/extending "
                         "the standing ruling; the referee still rules every tick past theta. "
                         "Mutually exclusive with --extend. Requires --theta.")
    ap.add_argument("--admit", action="store_true",
                    help="Exp 63: give the decision layer an ADMISSION lever — which waiting "
                         "jobs start this tick and which keep waiting — on top of the "
                         "margin/reserve split. Only never-started jobs can be deferred, so the "
                         "rigid no-preemption invariant holds. Tier suffix +admit.")
    ap.add_argument("--tick", type=int, default=0, metavar="S",
                    help="seconds per tick; 0 (default) = the trace's native 120 s. This is a "
                         "RESOLUTION knob only: the arrival span (6 h), the simulated span (13.3 h) "
                         "and the work clamp (2 min..2 h) are denominated in seconds, so a shorter "
                         "tick slices the same workload more finely instead of shrinking it. Costs "
                         "per-tick work proportionally — at 30 s every arm pays 4x the policy "
                         "invocations, and a per-tick LLM arm pays 4x the calls. Tier suffix +tS.")
    ap.add_argument("--horizon", type=int, default=0, metavar="T",
                    help=f"simulated ticks; 0 (default) = HORIZON_S/tick (= {HORIZON} at 120 s). "
                         "The ARRIVAL span is pinned at 6 h regardless, so this only adds DRAIN "
                         "time. At the pre-Exp-97 300 ticks ~2.5%% of jobs never finish and count "
                         "as SLA violations by censoring alone. Tier suffix +hT.")
    ap.add_argument("--slack-mult", type=float, default=SLACK_MULT, metavar="M",
                    help=f"scale every job's deadline slack by M (default {SLACK_MULT:g}). The raw "
                         "recipe gives median 1.55x the job's OWN work (p10 = 1.00x -- unmeetable "
                         "in a shared queue at 76%% utilisation), which floors SLA violations near "
                         "55%% regardless of policy. Pass 1 to reproduce pre-Exp-97 tiers. Tier "
                         "suffix +slackM.")
    ap.add_argument("--burst", type=int, default=None, metavar="S",
                    help="arrival SURGE: half the window's jobs are re-stamped into one "
                         "random S-second interval, the rest keep the trace's own arrival "
                         "times, so each seed holds a quiet baseline AND a contended spike. "
                         "Tier gets a +burstS suffix (windows differ, so it never pairs "
                         "with a non-burst tier).")
    ap.add_argument("--hard-trigger", action="store_true",
                    help="E3: gate the DELIBERATION, not the ruling — the debate round fires "
                         "only on a hard trigger (prod arrival, free-GPU bucket crossing, a "
                         "job newly behind, or a fallback last tick); the referee still rules "
                         "every tick on live numbers. Routine ticks reuse the plain referee's "
                         "cached ruling. Requires --debate; tier suffix +hardtrig.")
    ap.add_argument("--prev-input", action="store_true",
                    help="E1 (Exp 58): the referee also receives the allocation that "
                         "actually executed last tick plus a change-cost rule 6. Prompt "
                         "variant is keyed separately (|prev:<digest>); default prompts "
                         "and caches untouched.")
    ap.add_argument("--no-argue", action="store_true",
                    help="Exp 57f ablation: strip every statement's justification before the "
                         "referee rules — numbers-only scenes isolate what the two-sided "
                         "advocacy contributes. Separate cache tag (|noarg).")
    ap.add_argument("--stale", default="fresh", choices=("fresh", "one"),
                    help="Exp 57: 'one' = pipelined epochs — the referee decides on the "
                         "previous tick's statements, evaluator validates against the live "
                         "tick. REJECTED as method (erodes -7.6* to -2.5 ns, fb x5); kept "
                         "for the staleness ablation.")
    ap.add_argument("--referee", action="store_true",
                    help="Exp 50: swap the policy rows for the referee-LLM allocator vs the "
                         "no-llm floor and the negotiated (code-decides) arm; infeasible "
                         "referee ticks fall back to the floor and count in fb")
    ap.add_argument("--advocates", default="qwen2.5:3b", metavar="MODEL",
                    help="model for the demand/supply statements in the referee AND debate "
                         "rows (default qwen2.5:3b, as Exp 50). Holding it fixed across both "
                         "rows is what makes the debate contrast the round alone")
    ap.add_argument("--debate", action="store_true",
                    help="add a 'debate' row: both sides read each other's statements and "
                         "revise (asks may only shrink) before the SAME referee rules, so the "
                         "referee vs debate contrast isolates the cross-talk round. Advocates "
                         "run at --model. Requires --referee")
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
    sweep([int(p) for p in a.pools.split(",")], n_jobs=a.n_jobs, horizon=a.horizon,
          seeds=list(range(a.seed_start, a.seeds)), scale=a.scale, spike_max=a.spike,
          tick_s=a.tick,
          use_llm=a.llm, model=a.model, caps_mode=a.caps, quantile=a.quantile,
          truth_mode=a.truth, time_mode=a.time, ttf_mode=a.ttf, baseline=a.baseline,
          dyncap=a.dyncap, trace_name=a.trace, quantum={"quarter": 1, "whole": 4}[a.quantum],
          gated=a.gated, corrected=a.corrected, authored=a.authored,
          decorrelate=a.decorrelate, referee=a.referee, manual=a.manual, single_ilp=a.single_ilp, debate=a.debate,
          no_think=a.no_think, advocates=a.advocates, realloc_cost=a.realloc_cost,
          alpha=a.alpha, alpha_norm=a.alpha_norm, trigger=a.trigger, theta=a.theta,
          stale=a.stale, extend=a.extend, no_argue=a.no_argue, prev_input=a.prev_input,
          burst_s=a.burst, hard_trigger=a.hard_trigger, slack_mult=a.slack_mult,
          admit=a.admit, law=a.law, kappa=a.kappa, samples=a.samples,
          cooldown=a.cooldown, resize_c1=a.resize_c1, gamma=a.gamma, phi=a.phi,
          qcache=a.qcache, market=a.market, composed=a.composed, bid_info=a.bid_info,
          fast_negotiate=a.fast_negotiate)


if __name__ == "__main__":
    main()
