"""Per-job policy labels: is there headroom a WINDOW-level selector cannot see?

`policy_labels --perjob` kept every job's own outcome under every one of the 15 policies, for all
589 states. The window sweep already answered "pick one policy per epoch" (negative: a tree loses
to the best fixed policy 0/5 folds, `runs/headroom_verdict.md`). The finer question this data can
answer is whether the loss was one of GRAIN: a window summary averages over jobs that the policies
treat very differently, so a per-job assigner could in principle beat any per-window choice.

Three ceilings are reported, each one strictly above the last:

  fixed       -- the best single policy pooled over every state. The thing to beat.
  state       -- the best policy per state. This is the ceiling the window-level selector chased.
  group       -- the best policy per (state, job group), groups built from features known at
                 decision time (tier, elasticity, size, already-queued). The ceiling for ANY
                 per-job assigner that routes on those features.
  job         -- the best policy per (state, job). The absolute ceiling.

`group` and `job` are NOT achievable: a rollout applies one ordering+sizing to the whole cluster,
so the labels cannot price a heterogeneous schedule, and the per-job deviations are mostly
zero-sum by construction (one job's earlier start is another's later one). That is exactly why the
zero-sum decomposition below is reported next to the ceilings rather than left implicit: the
question is not whether the job-level ceiling is above the state-level one -- it must be -- but
whether the excess is a common gain that some mechanism could collect, or a tug-of-war that sums
to zero across the jobs sharing the cluster.

The per-job cost is the reward's own job-level terms, so `mean_j cost_j` is IDENTICAL to the
window reward minus its two window-level terms (fairness, resize churn); nothing is re-weighted
here. Those two terms are quantified separately so the exclusion is auditable.

    .venv/bin/python -m pins.perjob_analyse --labels runs/labels_perjob --worlds runs/holdout
    .venv/bin/python -m pins.perjob_analyse --selftest
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from pins.elastisim_bench import POLICY_MENU, POLICY_UNSAFE
from pins.policy_labels import OPERATING_POINTS

ORDERINGS = list(POLICY_MENU["ordering"])
SIZINGS = list(POLICY_MENU["sizing"])
ACTIONS = [f"{o}+{z}" for o in ORDERINGS for z in SIZINGS]
SAFE = [a for a in ACTIONS if a not in POLICY_UNSAFE]


def job_cost(wait: float, ta: float, run: float, ok: int, w: dict) -> float:
    """The reward's three job-level terms, for ONE job, in the reward's own units.

    `reward()` in policy_labels is -(sla10*viol_frac + bsd*(mean_bsd-1)/10 + wait_h*mean_wait/86400
    + resize + unfair). Each of the first three is a mean over jobs, so this function is the
    per-job summand and mean_j job_cost == those three terms exactly.
    """
    bsd = max(1.0, ta / max(10.0, run))
    viol = 1.0 if (not ok or ta > 10 * run) else 0.0
    return w["sla10"] * viol + w["bsd"] * (bsd - 1) / 10 + w["wait_h"] * wait / 86400


def load(labels: Path, w: dict) -> dict:
    """{(window, t): {"ids": (J,), "cost"/"wait"/"viol"/"bsd": (15, J)}}, policies in ACTIONS order."""
    raw: dict = {}
    for f in sorted(glob.glob(str(labels / "perjob.*.jsonl*"))):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                raw.setdefault((r["window"], r["t"]), {})[f"{r['ordering']}+{r['sizing']}"] = r["jobs"]
    out = {}
    for key, byact in raw.items():
        if len(byact) != len(ACTIONS):          # a shard died mid-state; the state is unusable
            continue
        ids = sorted(j[0] for j in byact[ACTIONS[0]])
        cost, wait, viol, bsd = (np.zeros((len(ACTIONS), len(ids))) for _ in range(4))
        ok_ids = True
        for i, a in enumerate(ACTIONS):
            d = {j[0]: j[1:] for j in byact[a]}
            if sorted(d) != ids:                # policies must be compared on the same job set
                ok_ids = False
                break
            for k, jid in enumerate(ids):
                wt, ta, run, ok = d[jid]
                cost[i, k] = job_cost(wt, ta, run, ok, w)
                wait[i, k] = wt
                viol[i, k] = 1.0 if (not ok or ta > 10 * run) else 0.0
                bsd[i, k] = max(1.0, ta / max(10.0, run))
        if ok_ids:
            out[key] = {"ids": ids, "cost": cost, "wait": wait, "viol": viol, "bsd": bsd}
    return out


@lru_cache(maxsize=None)
def _jobs(world: Path) -> list[dict]:
    return json.loads((world / "in/jobs.json").read_text())["jobs"]


def groups_for(world: Path, ids: list[int], t: int) -> dict[str, np.ndarray]:
    """Group labels per job, from features a referee would know at the decision epoch `t`."""
    jobs = _jobs(world)
    # FIXED buckets, not per-state quantiles: a median split would give the same label a different
    # meaning in every state, which silently makes the pooled static rule below incoherent.
    bucket = lambda g: "gpus=1" if g <= 1 else "gpus2-8" if g <= 8 else "gpus>8"
    return {
        "tier": np.array([jobs[i]["attributes"].get("tier", "?") for i in ids]),
        "elastic": np.array([f"elastic={jobs[i]['attributes'].get('elastic', 0)}" for i in ids]),
        "size": np.array([bucket(jobs[i]["attributes"].get("gpus", 1)) for i in ids]),
        "queued": np.array([f"submitted_by_t={int(jobs[i]['submit_time'] <= t)}" for i in ids]),
    }


def ceilings(data: dict, worlds: Path, menu: list[str], metric: str = "cost") -> dict:
    """fixed / state / group / job ceilings, all as a MEAN PER JOB of `metric` (lower is better)."""
    idx = [ACTIONS.index(a) for a in menu]
    keys = sorted(data)
    # `fixed` is pooled the way the window study pooled it: the mean over states of the state's
    # mean-per-job cost, so every state weighs the same regardless of how many jobs it holds.
    per_state = np.array([[data[k][metric][i].mean() for i in idx] for k in keys])   # (S, |menu|)
    fixed_i = int(per_state.mean(axis=0).argmin())
    res = {"fixed_policy": menu[fixed_i], "fixed": float(per_state[:, fixed_i].mean()),
           "state": float(per_state.min(axis=1).mean())}
    job_vals, grp_vals = [], {g: [] for g in ("tier", "elastic", "size", "queued", "all4")}
    for k in keys:
        m = data[k][metric][idx]                                   # (|menu|, J)
        job_vals.append(m.min(axis=0).mean())
        g = groups_for(worlds / k[0], data[k]["ids"], k[1])
        g["all4"] = np.array(["|".join(x) for x in zip(*(g[c] for c in ("tier", "elastic", "size", "queued")))])
        for name, lab in g.items():
            tot = 0.0
            for u in np.unique(lab):
                sel = lab == u
                tot += m[:, sel].mean(axis=1).min() * sel.sum()    # best policy for THIS group
            grp_vals[name].append(tot / m.shape[1])
    res["job"] = float(np.mean(job_vals))
    res["group"] = {g: float(np.mean(v)) for g, v in grp_vals.items()}
    return res


def sizing_only(data: dict, worlds: Path, metric: str = "cost") -> dict:
    """The one heterogeneous family that a mechanism could actually implement.

    `ordering` is a property of the queue and is necessarily global; `sizing` is applied to a job
    when it is placed, so DIFFERENT jobs may legally get different sizings. This prices that
    restricted family: one global ordering per state, chosen jointly with a per-job (or per-group)
    sizing, plus the fully static rule that fixes both once for the whole trace.

    The composability caveat still holds -- every rollout is homogeneous, so a heterogeneous
    schedule's true interactions are unpriced -- but unlike the per-job ceiling this family has an
    implementation path, so its ceiling is the one worth quoting.
    """
    per_job, per_grp = [], []
    pooled: dict = {}          # (ordering, group) -> [state-mean cost per sizing, ...]
    for k in sorted(data):
        m = data[k][metric]
        g = groups_for(worlds / k[0], data[k]["ids"], k[1])
        lab = np.array(["|".join(x) for x in zip(*(g[c] for c in ("tier", "elastic", "size", "queued")))])
        units, counts = np.unique(lab, return_counts=True)
        best_j, best_g = np.inf, np.inf
        for o in ORDERINGS:
            sub = m[[ACTIONS.index(f"{o}+{z}") for z in SIZINGS]]   # (|SIZINGS|, J)
            best_j = min(best_j, float(sub.min(axis=0).mean()))
            per_unit = np.array([sub[:, lab == u].mean(axis=1) for u in units])   # (G, |SIZINGS|)
            best_g = min(best_g, float((per_unit.min(axis=1) * counts).sum() / sub.shape[1]))
            for u, c, row in zip(units, counts, per_unit):
                # Weight = the group's share of THIS state, so the pooled rule below is scored on
                # the same equal-per-state footing as every ceiling in `ceilings()`.
                pooled.setdefault((o, u), []).append((row, c / sub.shape[1]))
        per_job.append(best_j)
        per_grp.append(best_g)
    # Static rule: ONE ordering for the whole trace and one sizing per group, both fixed before any
    # state is seen. Job-count weighted, so it is the same quantity as the ceilings above.
    static, rules, S = {}, {}, len(data)
    for o in ORDERINGS:
        total = 0.0
        for (oo, u), rows in pooled.items():
            if oo != o:
                continue
            # Group u's contribution to the equal-per-state mean, per sizing: its share-weighted
            # cost summed over the states it appears in, divided by ALL states.
            v = np.array([r for r, _ in rows]).T @ np.array([c for _, c in rows]) / S
            total += float(v.min())
            rules.setdefault(o, {})[u] = SIZINGS[int(v.argmin())]
        static[o] = total
    best_o = min(static, key=static.get)
    return {"per_job": float(np.mean(per_job)), "per_group": float(np.mean(per_grp)),
            "static_rule": static[best_o], "static_ordering": best_o, "rule": rules[best_o]}


def zero_sum(data: dict, menu: list[str], metric: str = "cost") -> dict:
    """Split each state's per-job matrix into a common policy effect and a per-job deviation.

    cost[p, j] = m[p] + d[p, j] with sum_j d[p, j] == 0 for every p. `state` gain uses m only;
    everything above it lives in d, whose per-job minimum can never be collected for all jobs at
    once because d sums to zero over the jobs sharing the cluster.
    """
    idx = [ACTIONS.index(a) for a in menu]
    common, dev = [], []
    for k in sorted(data):
        m = data[k][metric][idx]
        mu = m.mean(axis=1, keepdims=True)                 # (|menu|, 1) the state-level effect
        common.append(float(mu.min() - mu.mean()))         # best vs average policy, common part
        dev.append(float((m - mu).min(axis=0).mean()))     # per-job pick of the deviation, <= 0
    return {"common_spread": float(np.mean(common)), "deviation_floor": float(np.mean(dev)),
            "deviation_share": float(np.mean(dev) / (np.mean(dev) + np.mean(common)))}


def concentration(data: dict, menu: list[str], fixed: str, metric: str = "cost") -> dict:
    """Per-job regret of the best FIXED policy against each job's own best, and where it sits."""
    idx = [ACTIONS.index(a) for a in menu]
    f = ACTIONS.index(fixed)
    reg = []
    for k in sorted(data):
        m = data[k][metric]
        reg += list(m[f] - m[idx].min(axis=0))
    reg = np.array(sorted(reg, reverse=True))
    n = len(reg)
    return {"n_jobs": n, "mean_regret": float(reg.mean()),
            "hurt_frac": float((reg > 1e-9).mean()),
            "top10pct_share": float(reg[: n // 10].sum() / reg.sum()) if reg.sum() else 0.0,
            "top1pct_share": float(reg[: max(1, n // 100)].sum() / reg.sum()) if reg.sum() else 0.0}


def window_terms(labels: Path, w: dict) -> dict:
    """How much of the reward the excluded window-level terms carry, and whether they move argmax."""
    rows: dict = {}
    for f in sorted(glob.glob(str(labels / "rollouts.*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows.setdefault((r["window"], r["t"]), {})[f"{r['ordering']}+{r['sizing']}"] = r
    flips, shares = 0, []
    for key, byact in rows.items():
        if len(byact) != len(ACTIONS):
            continue
        full, jobonly = {}, {}
        for a, r in byact.items():
            wl = (w["resize"] * r["resize_events"] / max(1, r["n"])
                  + w["unfair"] * (1 - r["jain_user_wait"]) / 10)
            job = (w["sla10"] * r["sla10_viol_pct"] / 100 + w["bsd"] * (r["mean_bsd"] - 1) / 10
                   + w["wait_h"] * r["mean_wait_s"] / 86400)
            full[a], jobonly[a] = job + wl, job
            shares.append(wl / (job + wl) if job + wl else 0.0)
        flips += min(full, key=full.get) != min(jobonly, key=jobonly.get)
    return {"n_states": len(rows), "window_term_share": float(statistics.mean(shares)),
            "argmax_flips": flips, "argmax_flip_rate": flips / max(1, len(rows))}


def selftest() -> None:
    w = OPERATING_POINTS["balanced"]
    # mean_j job_cost must equal the reward's three job-level terms on the same jobs
    from pins.policy_labels import reward
    jobs = [(100.0, 200.0, 100.0, 1), (50_000.0, 60_000.0, 100.0, 1), (0.0, 10.0, 10.0, 0)]
    bsd = [max(1.0, ta / max(10.0, run)) for _, ta, run, _ in jobs]
    r = {"n": 3, "sla10_viol_pct": 100 * sum(not ok or ta > 10 * run for _, ta, run, ok in jobs) / 3,
         "mean_bsd": statistics.mean(bsd), "mean_wait_s": statistics.mean(x[0] for x in jobs),
         "jain_user_wait": 1.0, "resize_events": 0}
    assert abs(statistics.mean(job_cost(*j, w) for j in jobs) - -reward(r, w)) < 1e-9
    # a synthetic state: policy 0 best on average, every job's own best is another policy, and the
    # deviations are exactly zero-sum -> job ceiling below state ceiling, deviation_share == 1
    S = {("w", 0): {"ids": [0, 1], "cost": np.zeros((len(ACTIONS), 2))}}
    S[("w", 0)]["cost"][0] = [1.0, 1.0]
    S[("w", 0)]["cost"][1] = [0.0, 2.0]
    S[("w", 0)]["cost"][2] = [2.0, 0.0]
    S[("w", 0)]["cost"][3:] = 5.0
    z = zero_sum(S, ACTIONS)
    assert z["common_spread"] < 0 and z["deviation_floor"] < 0
    idx = [ACTIONS.index(a) for a in ACTIONS]
    assert abs(S[("w", 0)]["cost"][idx].min(axis=0).mean() - 0.0) < 1e-9   # job ceiling = 0
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("runs/labels_perjob"))
    ap.add_argument("--worlds", type=Path, default=Path("runs/holdout"))
    ap.add_argument("--weights", choices=sorted(OPERATING_POINTS), default="balanced")
    ap.add_argument("--out", type=Path, default=Path("runs/perjob_report.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    w = OPERATING_POINTS[a.weights]
    data = load(a.labels, w)
    print(f"{len(data)} states | {len({k[0] for k in data})} windows | "
          f"{sum(len(v['ids']) for v in data.values())} job-rollouts per policy")

    rep = {"operating_point": a.weights, "n_states": len(data),
           "excluded_window_terms": window_terms(a.labels, w)}
    for name, menu in (("safe", SAFE), ("full", ACTIONS)):
        rep[name] = {m: ceilings(data, a.worlds, menu, m) for m in ("cost", "wait", "viol", "bsd")}
        rep[name]["zero_sum"] = zero_sum(data, menu)
        rep[name]["concentration"] = concentration(data, menu, rep[name]["cost"]["fixed_policy"])
    rep["sizing_only"] = {m: sizing_only(data, a.worlds, m) for m in ("cost", "wait")}

    for name in ("safe", "full"):
        c = rep[name]
        print(f"\n=== {name} menu ({len(SAFE) if name == 'safe' else len(ACTIONS)} actions), "
              f"best fixed = {c['cost']['fixed_policy']}")
        print(f"{'metric':<8} {'fixed':>10} {'state':>10} {'group(all4)':>12} {'job':>10} "
              f"{'state gain':>11} {'job gain':>10}")
        for m, unit in (("cost", "reward"), ("wait", "s"), ("viol", "frac"), ("bsd", "x")):
            d = c[m]
            print(f"{m:<8} {d['fixed']:>10.4f} {d['state']:>10.4f} {d['group']['all4']:>12.4f} "
                  f"{d['job']:>10.4f} {100*(d['fixed']-d['state'])/abs(d['fixed']):>10.1f}% "
                  f"{100*(d['fixed']-d['job'])/abs(d['fixed']):>9.1f}%   ({unit})")
        g = c["cost"]["group"]
        print("  group ceilings (cost):", " ".join(f"{k}={v:.4f}" for k, v in g.items()))
        d = c["cost"]
        print(f"  of the job-ceiling gain, the part a per-STATE choice already collects: "
              f"{100*(d['fixed']-d['state'])/(d['fixed']-d['job']):.0f}%  "
              f"(remainder {100*(d['state']-d['job'])/(d['fixed']-d['job']):.0f}% is per-job "
              f"redistribution, which no single-policy schedule can collect)")
        z, k = c["zero_sum"], c["concentration"]
        print(f"  zero-sum: common {z['common_spread']:.4f} | per-job deviation floor "
              f"{z['deviation_floor']:.4f} | deviation share {100*z['deviation_share']:.0f}%")
        print(f"  per-job regret of best fixed: mean {k['mean_regret']:.4f} | hurt "
              f"{100*k['hurt_frac']:.0f}% of {k['n_jobs']} jobs | top10% hold "
              f"{100*k['top10pct_share']:.0f}% | top1% hold {100*k['top1pct_share']:.0f}%")
    print("\n=== the implementable family: global ordering, per-job SIZING")
    for m in ("cost", "wait"):
        s, f = rep["sizing_only"][m], rep["safe"][m]
        print(f"{m:<8} fixed {f['fixed']:>10.4f} | state {f['state']:>10.4f} | "
              f"static rule {s['static_rule']:>10.4f} | per-group {s['per_group']:>10.4f} | "
              f"per-job {s['per_job']:>10.4f}   (ordering {s['static_ordering']})")
    r = rep["sizing_only"]["cost"]["rule"]
    print(f"  static rule uses {len(set(r.values()))} distinct sizings over {len(r)} groups: "
          + ", ".join(f"{k}->{v}" for k, v in sorted(r.items())[:4]) + (" ..." if len(r) > 4 else ""))
    e = rep["excluded_window_terms"]
    print(f"\nexcluded window-level terms: {100*e['window_term_share']:.0f}% of total cost, "
          f"argmax flips on {100*e['argmax_flip_rate']:.0f}% of states")
    a.out.write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
