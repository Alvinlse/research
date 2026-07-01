"""
Exp 26 — CLOSE THE LOOP: a reflective margin agent that learns from its own outcomes.

Every LLM agent so far (Exp 10/12/14/17/24) is STATELESS across episodes: it maps a discretised
state to a categorical decision, CACHED per state — the same state always yields the same hedge,
forever, regardless of whether it worked. The system never sees the consequence of its choices.
Exp 25 made the cost vivid: 14b confidently bid aggressive margins that backfired, with nothing to
tell it to stop.

This module closes that loop for the demand-side MARGIN agent (the Exp 16/17 world: a single GPU
pool, stochastic train spikes, committed auction). The cycle is:

    run episode (current policy) -> attribute per-state outcomes -> LLM REFLECTS & revises -> re-run

The reflection is the interpretability edge over RL: the policy update is a readable sentence
("no spikes at 96% util -> margin wasted, lower hedge"), not a gradient. Hinge intact — the agent
emits only a categorical hedge; deterministic code owns the GPU count (margin_uncertainty).

The demonstration deliberately starts from the WEAK model (qwen2.5:3b), which Exp 17/24 showed
over-hedges and self-harms. The question: can reflection make a small model CORRECT its own
systematic mistakes over cycles — approaching the 14b / deterministic-oracle policy — with an
auditable trail, and WITHOUT a bigger model?

Run:  .venv/bin/python -m pins.reflective_sim --no-llm           # rule-reflect (no Ollama), fast
      .venv/bin/python -m pins.reflective_sim --llm --model qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
import os

from pins.llm_agent import (llm_margin, llm_margin_reflect, margin_state_key, margin_uncertainty,
                            spike_risk_bucket, uncertainty_bucket, _rule_margin)
from pins.negotiation_sim import Job, make_committed_auction, make_workload
from pins.predictor import PHASE_PROFILES, marginal_values
from pins.uncertainty_sim import assign, load_uncertainty_distribution, true_need

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE_THRESH = 0.05          # a job "actually spiked" if its realised over-run exceeds this


def simulate(jobs_proto, allocator, total_gpus, horizon, hedge_of, u_map, spike_map, scale,
             tier_of, spike_max) -> tuple[dict, dict]:
    """One episode under a margin policy. `hedge_of(ctx) -> hedge` is the policy (a pure function of
    the discretised state). Returns (metrics, per_state) where per_state[skey] accumulates the jobs
    that hit that state during TRAIN and the cluster utilisation while its hedge was active, so the
    caller can attribute deadline/spike outcomes to each state for reflection."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]
    by_id = {j.jid: j for j in jobs}
    work = {j.jid: true_need(j, spike_map[j.jid]) for j in jobs}
    useful = {j.jid: round(u_map[j.jid] * scale) for j in jobs}
    progress = {j.jid: 0.0 for j in jobs}
    pidx = {j.jid: 0 for j in jobs}
    done_at: dict[str, int | None] = {j.jid: None for j in jobs}
    current: dict[str, int] = {}
    busy_sum = 0.0
    busy_steps = 0
    # state attribution: skey -> {jids:set, util_sum:float, ticks:int, hedge:str}
    per_state: dict[str, dict] = {}

    def phase_of(j):
        return j.phases[pidx[j.jid]] if pidx[j.jid] < len(j.phases) else "idle"

    def cap0(j):
        return PHASE_PROFILES[phase_of(j)][0]

    for t in range(horizon):
        act = [j for j in jobs if j.arrival <= t and done_at[j.jid] is None]
        if not act:
            current = {}
            if all(done_at[j.jid] is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        demand = sum(cap0(j) for j in act)
        contention = "high" if demand >= total_gpus else "low"
        util_now = sum(current.get(j.jid, 0) for j in act) / total_gpus

        def dbucket(j):
            rem = max(0.0, j.need[pidx[j.jid]] - progress[j.jid]) + sum(j.need[pidx[j.jid] + 1:])
            ratio = rem / max(1, j.deadline - t)
            return "behind" if ratio > 1.0 else "ahead" if ratio < 0.6 else "ontrack"

        bids = {}
        for j in act:
            if phase_of(j) == "train":
                u = u_map[j.jid]
                ctx = {"uncertainty": uncertainty_bucket(u),
                       "spike_risk": spike_risk_bucket(u * spike_max),
                       "deadline": dbucket(j),
                       "contention": contention, "tier": tier_of[j.jid]}
                hedge = hedge_of(ctx)
                skey = margin_state_key(ctx)
                ps = per_state.setdefault(skey, {"jids": set(), "util_sum": 0.0, "ticks": 0,
                                                 "hedge": hedge, "ctx": ctx})
                ps["jids"].add(j.jid)
                ps["util_sum"] += util_now
                ps["ticks"] += 1
                ps["hedge"] = hedge
                bids[j.jid] = marginal_values(phase_of(j), j.urgency,
                                              uncertainty=margin_uncertainty(hedge, u, scale),
                                              margin_scale=scale)
            else:
                bids[j.jid] = marginal_values(phase_of(j), j.urgency)
        cur = {jid: current.get(jid, 0) for jid in bids}
        alloc = allocator(bids, total_gpus, cur)
        busy_sum += sum(alloc.values()) / total_gpus
        busy_steps += 1
        for j in act:
            g = alloc.get(j.jid, 0)
            c0 = cap0(j)
            if c0 == 0:
                rate = 1.0
            else:
                ceiling = c0 + (useful[j.jid] if phase_of(j) == "train" else 0)
                rate = min(g, ceiling) / c0
            progress[j.jid] += rate
            while done_at[j.jid] is None and progress[j.jid] >= work[j.jid][pidx[j.jid]] - 1e-9:
                progress[j.jid] -= work[j.jid][pidx[j.jid]]
                pidx[j.jid] += 1
                if pidx[j.jid] >= len(j.phases):
                    done_at[j.jid] = t
                    break
        current = {jid: alloc.get(jid, 0) for jid in by_id}

    def violated(j):
        return done_at[j.jid] is None or done_at[j.jid] > j.deadline

    prod = [j for j in jobs if j.tier == "prod"]
    metrics = {
        "sla": sum(1 for j in jobs if violated(j)) / len(jobs),
        "prod_sla": sum(1 for j in prod if violated(j)) / max(len(prod), 1),
        "util": busy_sum / max(busy_steps, 1),
        "finished": float(sum(1 for j in jobs if done_at[j.jid] is not None)),
    }
    # attribute outcomes to each state from this episode's done_at / spikes
    miss = {j.jid: violated(j) for j in jobs}
    spk = {j.jid: spike_map[j.jid] > SPIKE_THRESH for j in jobs}
    for skey, ps in per_state.items():
        jids = ps["jids"]
        ps["n"] = len(jids)
        ps["missed"] = sum(1 for jid in jids if miss[jid])
        ps["spiked"] = sum(1 for jid in jids if spk[jid])
        ps["util"] = ps["util_sum"] / max(ps["ticks"], 1)
    return metrics, per_state


def merge_experience(states_list: list[dict]) -> dict:
    """Pool per-state attribution across the seeds of one cycle into one experience per state."""
    merged: dict[str, dict] = {}
    for ps in states_list:
        for skey, e in ps.items():
            m = merged.setdefault(skey, {"ctx": e["ctx"], "hedge": e["hedge"], "n": 0,
                                         "missed": 0, "spiked": 0, "util_sum": 0.0, "ticks": 0})
            m["hedge"] = e["hedge"]
            m["n"] += e["n"]; m["missed"] += e["missed"]; m["spiked"] += e["spiked"]
            m["util_sum"] += e["util"] * e["n"]; m["ticks"] += e["n"]
    for m in merged.values():
        m["util"] = m["util_sum"] / max(m["ticks"], 1)
    return merged


def evaluate(hedge_of, seeds, dist, pool, n_jobs, horizon, spike_max, scale):
    """Run `hedge_of` on all seeds; return (mean metrics, merged per-state experience)."""
    acc = {"sla": 0.0, "prod_sla": 0.0, "util": 0.0, "finished": 0.0}
    states_list = []
    for s in seeds:
        jobs = make_workload(n_jobs, s, horizon)
        u_map, spike_map = assign(jobs, s, dist, spike_max)
        tier_of = {j.jid: j.tier for j in jobs}
        m, ps = simulate(jobs, make_committed_auction(), pool, horizon, hedge_of,
                         u_map, spike_map, scale, tier_of, spike_max)
        for k in acc:
            acc[k] += m[k]
        states_list.append(ps)
    return {k: v / len(seeds) for k, v in acc.items()}, merge_experience(states_list)


def run(pool, n_jobs, horizon, seeds, spike_max, scale, n_cycles, use_llm, model) -> None:
    dist = load_uncertainty_distribution()
    tag = "rule-reflect" if not use_llm else f"reflect:{model}"

    print(f"\n{'='*82}")
    print(f"REFLECTIVE MARGIN AGENT — close the loop; agent={tag}")
    print(f"{'='*82}")
    print(f"pool {pool}, {n_jobs} jobs, horizon {horizon}, {len(seeds)} seeds/cycle | "
          f"spike_max={spike_max} scale={scale} | {n_cycles} cycles")
    print("Lower SLA/prodSLA = better. Cycle 0 = cold policy (no reflection yet).\n")

    # --- reference lines (static policies, no reflection) -------------------------------------
    none_fn = lambda ctx: "none"
    rule_fn = lambda ctx: _rule_margin(ctx)["hedge"]
    refs = {"no-margin": none_fn, "rule-oracle": rule_fn}
    cold14: dict[str, str] = {}
    def cold14_fn(ctx):
        k = margin_state_key(ctx)
        if k not in cold14:
            cold14[k] = llm_margin(ctx, use_llm=True, model="qwen2.5:14b", cache={})["hedge"]
        return cold14[k]
    ref_metrics = {}
    for name, fn in refs.items():
        ref_metrics[name], _ = evaluate(fn, seeds, dist, pool, n_jobs, horizon, spike_max, scale)
    if use_llm:                                          # 14b static is itself an Ollama policy
        ref_metrics["14b-static"], _ = evaluate(
            cold14_fn, seeds, dist, pool, n_jobs, horizon, spike_max, scale)

    print("Reference (static) policies:")
    for name, m in ref_metrics.items():
        print(f"  {name:<14} SLA {m['sla']:6.1%}  prodSLA {m['prod_sla']:6.1%}  util {m['util']:4.0%}")
    print()

    # --- the reflective loop ------------------------------------------------------------------
    policy: dict[str, str] = {}                          # skey -> hedge (the learned policy)
    cold_seen: set = set()
    cold_cache: dict = {}
    def policy_fn(ctx):                                  # lazy cold-start, then the learned policy
        k = margin_state_key(ctx)
        if k not in policy:
            policy[k] = llm_margin(ctx, use_llm=use_llm, model=model, cache=cold_cache)["hedge"]
            cold_seen.add(k)
        return policy[k]

    trajectory = []
    reflections = []
    print("Reflective trajectory:")
    hdr = f"  {'cycle':>5}  {'SLA':>7} {'prodSLA':>8} {'util':>6} {'changes':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for c in range(n_cycles):
        m, exp = evaluate(policy_fn, seeds, dist, pool, n_jobs, horizon, spike_max, scale)
        trajectory.append(m)
        changes = 0
        if c < n_cycles - 1:                            # reflect, except after the last eval
            for skey, e in sorted(exp.items()):
                rev = llm_margin_reflect(e["ctx"], e, use_llm=use_llm, model=model)
                if rev["hedge"] != policy.get(skey):
                    changes += 1
                    reflections.append({"cycle": c, "state": skey, "from": policy.get(skey),
                                        "to": rev["hedge"], "n": e["n"], "missed": e["missed"],
                                        "spiked": e["spiked"], "util": round(e["util"], 2),
                                        "why": rev["justification"], "_source": rev["_source"]})
                policy[skey] = rev["hedge"]
        print(f"  {c:>5}  {m['sla']:>6.1%} {m['prod_sla']:>7.1%} {m['util']:>6.0%} {changes:>8}")
    print()

    # --- did the learned policy converge to the deterministic oracle? -------------------------
    agree = sum(1 for k in policy if policy[k] == _rule_margin(exp[k]["ctx"])["hedge"]) \
        if exp else 0
    print(f"Final policy: {len(policy)} states; {agree}/{len(policy)} agree with the rule-oracle hedge.")

    print("\nSample reflections (the auditable policy updates):")
    for r in reflections[:12]:
        print(f"  c{r['cycle']} [{r['state']:<28}] {r['from']}->{r['to']}  "
              f"(missed {r['missed']}/{r['n']}, spiked {r['spiked']}, util {r['util']:.0%})  {r['why']}")

    out = os.path.join(HERE, "results_reflective.json")
    with open(out, "w") as f:
        json.dump({"agent": tag, "pool": pool, "spike_max": spike_max,
                   "reference": ref_metrics, "trajectory": trajectory,
                   "final_policy": policy, "reflections": reflections}, f, indent=2)
    if use_llm:
        from pins.llm_agent import save_cache
        save_cache(cold_cache)
    print(f"\nartifact -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 26 — reflective (closed-loop) margin agent")
    ap.add_argument("--llm", action="store_true", help="use qwen for decisions + reflection")
    ap.add_argument("--no-llm", action="store_true", help="rule-reflect fallback (no Ollama)")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--spike", type=float, default=1.5, help="heavy tail: where the hedge matters")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=6)
    a = ap.parse_args()
    run(a.pool, n_jobs=16, horizon=300, seeds=list(range(a.seeds)), spike_max=a.spike,
        scale=a.scale, n_cycles=a.cycles, use_llm=not a.no_llm, model=a.model)


if __name__ == "__main__":
    main()
