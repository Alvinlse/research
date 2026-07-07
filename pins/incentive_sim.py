"""
Exp 32 — THE INCENTIVE LAYER: per-user budgets + uniform contention pricing.

Exp 13 left the open problem: the committed auction trusts declared priority classes, a
best-effort job lying 'critical' collapses prod protection to the greedy floor, and a flat
per-job budget cannot fix it — any cost on the CLASS hits a liar and an honest declarer
identically, and it taxed honest jobs even when nothing was contested (the control cratered).
Its post-mortem named the fix: exogenous PER-USER budgets (fair-share style), which needs
multi-job agents. This builds exactly that, plus second-price-flavoured charging:

  * USERS own budgets, jobs don't. A user's jobs share one purse, so a lie on one job has an
    opportunity cost inside the liar's own portfolio: the inflated best-effort job buys
    contested service it pays for tick after tick, draining what the user's genuinely
    critical job needs later (insolvency = every job of that user demoted to the bottom).
  * PAY-YOUR-CLAIM, BILLED ONLY ON CONTESTED SERVED TICKS: a job pays its own declared
    class cost per tick, but only while it actually HOLDS GPUs and someone else WAITS.
    Nobody pays in uncontested regimes (the exact failure of Exp 13's flat budget, which
    charged every active tick), and waiting costs nothing (you pay for what the claim GOT
    you). The truthful/liar asymmetry the flat budget lacked comes from the portfolio: an
    honest user's 'critical' claims sit on genuinely tight jobs — short residence, small
    spend — while a liar's sit on long, loose best-effort jobs whose contested residence
    bleeds the purse at 4/tick. (A uniform second-price variant — price set by the highest
    waiting class, charged to all served — was tried first and REJECTED: it socialises the
    lie's cost, honest heavy users drain faster than the liar.)

The measurement is also upgraded: Exp 13 scored system SLA only, but incentive
compatibility is a BEST-RESPONSE question. Part A holds everyone else honest, lets ONE user
inflate its best-effort jobs, and compares the DEVIATOR'S OWN jobs' outcomes paired by
seed — the deviation gain the mechanism must remove. Part B (all users lie) checks system
robustness; Part C (truthful with vs without budgets) checks the honest world is unharmed.

Run:  .venv/bin/python -m pins.incentive_sim            # 32 seeds, pools {6,8,12}
      .venv/bin/python -m pins.incentive_sim --seeds 8  # quick look
"""
from __future__ import annotations

import argparse
import json
import os
import random

from pins.llm_agent import priority_weight
from pins.negotiation_sim import (PRIO_CLASS_COST, Job, _serialise, _truthful_class,
                                  make_workload, simulate)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_incentive.json")

N_USERS = 4


def assign_users(jobs: list[Job], seed: int) -> dict[str, int]:
    """jid -> user id: seeded shuffle + round-robin so each user gets a tier mix."""
    rng = random.Random(f"users-{seed}")
    jids = [j.jid for j in jobs]
    rng.shuffle(jids)
    return {jid: i % N_USERS for i, jid in enumerate(jids)}


def declare_for(job: Job, users: dict[str, int], liars: set[int]) -> str:
    """Truthful class, except users in `liars` inflate their best-effort jobs to 'critical'
    (the damaging Exp-13 deviation; their prod jobs were already declaring truthfully high)."""
    if users[job.jid] in liars and job.tier == "besteffort":
        return "critical"
    return _truthful_class(job)


def make_user_budget_committed(users: dict[str, int], liars: set[int],
                               budget: tuple[float, float] | None):
    """Committed auction + the incentive layer. budget=None reproduces Exp 13's unpriced
    declared-committed (the vulnerable mechanism) for the same declarations.
    budget=(B, r): purse starts at B, earns r per tick, capped back at B (scrip income —
    bursty honest spending recovers between contested stretches; a liar's sustained
    contested drain outruns income)."""
    cls: dict[str, str] = {}
    weight: dict[str, float] = {}
    b0, income = budget if budget is not None else (0.0, 0.0)
    purse: dict[int, float] = {u: b0 for u in set(users.values())}

    def bid_builder(job: Job, t: int, market: dict) -> list[float]:
        if job.jid not in cls:
            c = declare_for(job, users, liars)
            cls[job.jid], weight[job.jid] = c, priority_weight(c)
        return job.bid()

    def alloc_factory():
        def sched(bids, total_gpus, current):
            def eff(a: str) -> float:
                if budget is not None and purse[users[a]] <= 0:
                    return 0.0                      # user insolvent -> whole portfolio demoted
                return weight.get(a, 2.0)
            order = sorted(bids, key=lambda a: (-eff(a), a))
            alloc = _serialise(order, bids, total_gpus)
            if budget is not None:
                contested = any(alloc.get(a, 0) == 0 and len(bids[a]) > 0 for a in bids)
                if contested:                       # pay your own claim, served ticks only
                    for a in bids:
                        if alloc.get(a, 0) > 0:
                            purse[users[a]] -= PRIO_CLASS_COST[cls.get(a, "normal")]
                for u in purse:                     # scrip income, capped at the initial purse
                    purse[u] = min(b0, purse[u] + income)
            return alloc
        return sched

    return bid_builder, alloc_factory


# --------------------------------------------------------------------------- #
#  Metrics + stats                                                              #
# --------------------------------------------------------------------------- #
def violated(j: Job) -> bool:
    return j.done_at is None or j.done_at > j.deadline


def user_rates(jobs: list[Job], users: dict[str, int], uid: int) -> dict:
    """The deviator's own outcomes: all its jobs, and split by TRUE tier."""
    mine = [j for j in jobs if users[j.jid] == uid]
    be = [j for j in mine if j.tier == "besteffort"]
    prod = [j for j in mine if j.tier == "prod"]
    rate = lambda js: sum(1 for j in js if violated(j)) / max(len(js), 1)
    return {"all": rate(mine), "be": rate(be), "prod": rate(prod)}


def t95(df: int) -> float:
    for lo, t in ((60, 2.000), (30, 2.042), (20, 2.086), (10, 2.228), (5, 2.571), (2, 4.303)):
        if df >= lo:
            return t
    return 12.71


def paired_ci(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, float("inf")
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    return m, t95(n - 1) * (var / n) ** 0.5


def fmt(diffs: list[float], label: str) -> str:
    m, h = paired_ci(diffs)
    sig = "*" if h < abs(m) else " "
    return f"{label} {m*100:+6.1f} ±{h*100:4.1f}{sig}"


# --------------------------------------------------------------------------- #
#  Sweep                                                                        #
# --------------------------------------------------------------------------- #
def run_world(jobs, users, liars, budget, gpus, horizon):
    bb, alf = make_user_budget_committed(users, liars, budget)
    res, out_jobs = simulate(jobs, bb, alf(), gpus, horizon, return_jobs=True)
    return res, out_jobs


def sweep(pools, budgets, n_jobs, horizon, seeds) -> None:
    print(f"\n{'='*88}")
    print(f"INCENTIVE LAYER (Exp 32) — per-user budgets + uniform contention pricing")
    print(f"{'='*88}")
    print(f"{n_jobs} jobs / {N_USERS} users, horizon {horizon}, {len(seeds)} seeds, "
          f"committed auction on DECLARED classes | deviation = user's best-effort jobs "
          f"claim 'critical'")
    data: dict = {"budgets": {}, "n_seeds": len(seeds)}

    for budget in budgets:
        tag = "none" if budget is None else \
            (f"{budget[0]:g}+{budget[1]:g}" if budget[1] else f"{budget[0]:g}")
        print(f"\n--- budget = {tag} " + "-" * 60)
        per_pool: dict = {}
        for gpus in pools:
            # per-seed worlds: honest / one deviator (user 0) / all liars
            dev_gain_all, dev_gain_be, dev_gain_prod = [], [], []
            victim_prod = []            # honest users' prod jobs: what the deviator does TO them
            sys_prod_honest, sys_prod_liars, sys_sla_honest, sys_sla_liars = [], [], [], []
            for s in seeds:
                jobs = make_workload(n_jobs, s, horizon)
                users = assign_users(jobs, s)
                res_h, j_hon = run_world(jobs, users, set(), budget, gpus, horizon)
                _, j_dev = run_world(jobs, users, {0}, budget, gpus, horizon)
                r_hon = user_rates(j_hon, users, 0)
                r_dev = user_rates(j_dev, users, 0)
                dev_gain_all.append(r_dev["all"] - r_hon["all"])
                dev_gain_be.append(r_dev["be"] - r_hon["be"])
                dev_gain_prod.append(r_dev["prod"] - r_hon["prod"])
                vic = lambda js: [j for j in js if users[j.jid] != 0 and j.tier == "prod"]
                rate = lambda js: sum(1 for j in js if violated(j)) / max(len(js), 1)
                victim_prod.append(rate(vic(j_dev)) - rate(vic(j_hon)))
                res_l, _ = run_world(jobs, users, set(users.values()), budget, gpus, horizon)
                sys_prod_honest.append(res_h.prod_sla_rate)
                sys_prod_liars.append(res_l.prod_sla_rate)
                sys_sla_honest.append(res_h.sla_violation_rate)
                sys_sla_liars.append(res_l.sla_violation_rate)
            print(f"  pool {gpus:>2} | deviator (user 0) gain, lie vs honest (− = lying pays):")
            print(f"           {fmt(dev_gain_all, 'd(all)')}  {fmt(dev_gain_be, 'd(BE)')}  "
                  f"{fmt(dev_gain_prod, 'd(own prod)')}  {fmt(victim_prod, 'd(victim prod)')}")
            lp = [b - a for a, b in zip(sys_prod_honest, sys_prod_liars)]
            ls = [b - a for a, b in zip(sys_sla_honest, sys_sla_liars)]
            print(f"           all-liars vs honest system: {fmt(lp, 'dprodSLA')}  "
                  f"{fmt(ls, 'dSLA')}   (honest prodSLA "
                  f"{sum(sys_prod_honest)/len(seeds):.1%})")
            per_pool[str(gpus)] = {
                "dev_gain_all": dev_gain_all, "dev_gain_be": dev_gain_be,
                "dev_gain_prod": dev_gain_prod, "victim_prod": victim_prod,
                "sys_prod_honest": sys_prod_honest, "sys_prod_liars": sys_prod_liars,
                "sys_sla_honest": sys_sla_honest, "sys_sla_liars": sys_sla_liars,
            }
        data["budgets"][tag] = per_pool

    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nper-seed data -> {RESULTS}")
    print("Read: d(BE) < 0 means the lie helped the deviator's best-effort jobs; the layer "
          "works if\nd(all) >= 0 with budgets (lying does not pay NET) while budget=none "
          "reproduces Exp 13's gain.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 32: incentive layer for the committed auction")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--pools", default="6,8,12")
    ap.add_argument("--budgets", default="none,60,120,240",
                    help="comma list: 'none' | B | B+r (initial purse + income/tick, cap B)")
    a = ap.parse_args()

    def parse_budget(b: str):
        if b == "none":
            return None
        return (float(b.split("+")[0]), float(b.split("+")[1])) if "+" in b else (float(b), 0.0)

    budgets = [parse_budget(b) for b in a.budgets.split(",")]
    sweep([int(p) for p in a.pools.split(",")], budgets, n_jobs=16, horizon=300,
          seeds=list(range(a.seeds)))


if __name__ == "__main__":
    main()
