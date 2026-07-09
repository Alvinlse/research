"""
Exp 41 — tabular Q-learning baseline: a LEARNED policy in the LLM's own interface.

research_plan.md's baseline row asks for "one learning-based scheduler (DRL)". A faithful
DeepRM/Decima-style DRL agent owns the whole allocator action space and is a multi-week
training project; what the thesis actually needs answered is narrower: **is the LLM's
decision quality just something a cheap learned table could match?** So the RL baseline is
scoped (scope decision, research_plan.md item 5) to tabular Q-learning over EXACTLY the
LLM's discretised interface — the same bridged states (`margin_state_key`/`reserve_state_key`)
and the same categorical actions (hedge none/some/heavy, reserve none/light/heavy). Anything
the Q-table cannot see, the LLM cannot see either; the comparison isolates the decision rule.

Training: episodic Monte-Carlo (contextual bandit — one shared return per episode, no
bootstrapping) with epsilon-greedy exploration on trace-replay windows from seeds >= 100,
DISJOINT from the eval seeds 0-7, same world as the eval tier (--time predicted). Return
= -(sla + prod_sla), the two headline metrics. States never visited in training fall back
to the deterministic rule at eval (the same rule the LLM tiers degrade to), so coverage
gaps degrade to the floor, never to garbage.

Run:  .venv/bin/python -m pins.qlearn                 # train -> pins/qlearn_table.json
      .venv/bin/python -m pins.trace_replay --baseline qlearn --time predicted   # eval
"""
from __future__ import annotations

import argparse
import json
import os
import random

from pins.llm_agent import (RESERVE_LEVELS, _rule_margin, _rule_reserve,
                            margin_state_key, reserve_amount, reserve_state_key)
from pins.negotiation_protocol import HEDGE_GPUS

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "qlearn_table.json")
HEDGE_LEVELS = list(HEDGE_GPUS)          # ["none", "some", "heavy"] — canonical action order


def load_table(path: str = TABLE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _pick(qs: dict[str, float], levels: list[str], rule_action: str,
          eps: float, rng: random.Random | None) -> str:
    """Greedy in canonical level order (deterministic ties); unseen state -> the rule;
    training-time epsilon-greedy when rng is given."""
    if rng is not None and rng.random() < eps:
        return rng.choice(levels)
    if not qs:
        return rule_action
    return max(levels, key=lambda a: qs.get(a, float("-inf")))


def make_policy_qlearn(table: dict, eps: float = 0.0, rng: random.Random | None = None,
                       visits: list | None = None):
    """Same (demand, supply_ctx, free) -> (margins, reserve, outcome) contract as the LLM
    policies in two_sided_sim; decisions come from the Q-table instead of Ollama."""
    def policy(demand, supply_ctx, free, **_):
        margins = {}
        for j in demand:
            if j.is_train:
                s = margin_state_key(j.ctx)
                a = _pick(table["margin"].get(s, {}), HEDGE_LEVELS,
                          _rule_margin(j.ctx)["hedge"], eps, rng)
                margins[j.jid] = HEDGE_GPUS[a]
                if visits is not None:
                    visits.append(("margin", s, a))
            else:
                margins[j.jid] = 0
        s = reserve_state_key(supply_ctx)
        a = _pick(table["reserve"].get(s, {}), RESERVE_LEVELS,
                  _rule_reserve(supply_ctx)["reserve"], eps, rng)
        if visits is not None:
            visits.append(("reserve", s, a))
        return margins, reserve_amount(a), None
    return policy


def train(episodes: int, pools: list[int], seed0: int = 100, alpha: float = 0.1,
          eps: float = 0.2, spike_max: float = 0.6, scale: int = 3) -> dict:
    # imported lazily: trace_replay imports this module for --baseline qlearn
    from pins.trace_replay import load_runtime_pred, load_trace, make_trace_workload
    from pins.two_sided_sim import simulate
    from pins.uncertainty_sim import assign, load_uncertainty_distribution

    trace = load_trace()
    time_pred = load_runtime_pred()      # same windows/world as the eval tier (--time predicted)
    dist = load_uncertainty_distribution()
    table: dict = {"margin": {}, "reserve": {}}
    rng = random.Random(0)
    recent: list[float] = []
    for e in range(episodes):
        seed = seed0 + e                 # >= 100: disjoint from eval seeds 0-7
        gpus = pools[e % len(pools)]
        jobs, cap_map, tcap, belief = make_trace_workload(
            trace, 16, seed, 300, None, False, None, False, time_pred, "predicted")
        cap_map = {k: min(v, gpus) for k, v in cap_map.items()}
        tcap = {k: min(v, gpus) for k, v in tcap.items()}
        u_map, spike_map = assign(jobs, seed, dist, spike_max)
        visits: list = []
        r = simulate(jobs, make_policy_qlearn(table, eps=eps, rng=rng, visits=visits),
                     gpus, 300, u_map, spike_map, scale, spike_max, cap_map,
                     true_cap_map=tcap, belief_work=belief)
        ret = -(r["sla"] + r["prod_sla"])
        for kind, s, a in set(visits):   # every state-action tried this episode shares the return
            qs = table[kind].setdefault(s, {})
            qs[a] = qs.get(a, 0.0) + alpha * (ret - qs.get(a, 0.0))
        recent.append(ret)
        if (e + 1) % 60 == 0:
            print(f"  ep {e + 1:>4}/{episodes}  mean return (last 60) = "
                  f"{sum(recent[-60:]) / len(recent[-60:]):+.3f}  "
                  f"states: {len(table['margin'])} margin / {len(table['reserve'])} reserve")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the tabular-Q baseline (Exp 41)")
    ap.add_argument("--episodes", type=int, default=900)
    ap.add_argument("--pools", default="4,6,8")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.2)
    a = ap.parse_args()
    table = train(a.episodes, [int(p) for p in a.pools.split(",")], alpha=a.alpha, eps=a.eps)
    with open(TABLE_PATH, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Q-table -> {TABLE_PATH} "
          f"({len(table['margin'])} margin states, {len(table['reserve'])} reserve states)")


if __name__ == "__main__":
    main()
