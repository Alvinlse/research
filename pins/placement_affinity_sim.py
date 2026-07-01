"""
Exp 23 — does an LLM AFFINITY hint + ILP placement beat numbers-only placement? (the gate)

The placement extension's whole justification (pins/affinity.py) is the one door Exp 21 Part C left
open: co-location slowdown is mix-dependent ONLY for heterogeneous bottlenecks, and which tasks are
complementary is a SEMANTIC judgement a numbers-only solver cannot infer. This module is the
falsifiable test of that claim, in the lab's beat-the-baseline style.

Setup. A cluster of `n_nodes x gpus_per_node`; tasks are single-node (co-located) and exactly fill
the cluster, so this is PURE placement — who shares a node with whom — not rationing. Each task has a
true bottleneck class (compute / bandwidth) and an op-profile string the agent reads to GUESS it.

Ground-truth contention (the scorer the agents do NOT see): two tasks of the SAME contending class on
one node fight over that resource — each runs at 1/(1 + alpha*k_same) speed (k_same = same-class
co-tenants). Complementary tasks (compute+bandwidth) overlap for free. So the best placement pairs
complementary tasks and separates contenders; total throughput = sum of realised rates.

Policies scored on the SAME workload:
  * pack       — ILP places with NO affinity (value-only; CBC picks an arbitrary feasible packing)
  * spread     — class-BLIND balanced placer (the strong heuristic: even out load, ignore class)
  * llm-affin  — LLM classifies each task's bottleneck -> affinity_matrix -> ILP places with it
  * oracle     — TRUE classes -> affinity_matrix -> ILP (the ceiling: perfect bottleneck labels)

Gate (the honest, Exp-21-consistent prediction): llm-affin beats spread on HETEROGENEOUS workloads
(class knowledge adds signal blind spreading cannot), and only TIES it on HOMOGENEOUS ones (all-compute
-> every pair contends -> no relational signal to exploit, exactly Exp 21's `slowdown ~ k`).

Default use_llm=False -> rule fallback, no Ollama. --llm calls qwen. Reuses pins/ilp.allocate_placement
(new affinity arg) and pins/affinity.py.

Run:  .venv/bin/python -m pins.placement_affinity_sim          # rule fallback
      .venv/bin/python -m pins.placement_affinity_sim --llm    # qwen classifier
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

from pins.affinity import affinity_matrix, llm_bottleneck
from pins.ilp import allocate_placement
from pins.llm_agent import save_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = {"compute": 0.7, "bandwidth": 0.7, "io": 0.0}     # same-class slowdown per co-tenant
VALUE = 10.0                                              # per-GPU bid (>> penalty, so all place)
PENALTY = 1.0                                             # affinity penalty per same-class pair

# Op-profiles per true class — the text the classifier reads; it must INFER the class from semantics.
PROFILES = {
    "compute":   ["ResNet-50 dense conv training", "GPT transformer matmul block",
                  "ViT attention + dense matmul", "convolutional feature extractor"],
    "bandwidth": ["BERT pretraining with large all-reduce parameter sync",
                  "DLRM huge embedding-table lookups, memory-bandwidth bound",
                  "graph-neural-net with heavy neighbour gather/scatter",
                  "model-parallel shard with cross-GPU comm each step"],
}


@dataclass
class Task:
    jid: str
    cls: str            # true bottleneck class
    profile: str        # op description the agent classifies
    demand: int         # GPUs (single node)


def make_tasks(n_nodes: int, gpus_per_node: int, demand: int, hetero: float, seed: int) -> list[Task]:
    """Fill the cluster exactly with single-node tasks; each is 'bandwidth' with prob `hetero`,
    else 'compute'. hetero=0 -> all compute (homogeneous); hetero=0.5 -> balanced (most mixable)."""
    rng = random.Random(f"aff-{seed}-{hetero}")
    n = (n_nodes * gpus_per_node) // demand
    tasks = []
    for i in range(n):
        cls = "bandwidth" if rng.random() < hetero else "compute"
        tasks.append(Task(f"t{i:02d}", cls, rng.choice(PROFILES[cls]), demand))
    return tasks


# --------------------------------------------------------------------------- #
#  Placers -> node_of (which node each task lands on)                            #
# --------------------------------------------------------------------------- #
def place_ilp(tasks, n_nodes, gpus_per_node, affinity) -> dict[str, int | None]:
    bids = {t.jid: [VALUE] * t.demand for t in tasks}     # uniform value: place all, affinity steers
    # Offline experiment, not the real-time scheduler -> give CBC a real budget so the affinity MILP
    # reaches the proven optimum (the 0.5s default truncates it and corrupts the comparison).
    res = allocate_placement(bids, n_nodes, gpus_per_node, coloc={t.jid: True for t in tasks},
                             affinity=affinity, time_limit=10.0)
    return res.detail.get("node_of", {})


def place_spread(tasks, n_nodes, gpus_per_node) -> dict[str, int | None]:
    """Class-BLIND balanced placement: put each task (in id order) on the emptiest node that fits.
    Spreads load evenly without knowing bottleneck classes — the strong numbers-only baseline."""
    free = [gpus_per_node] * n_nodes
    node_of: dict[str, int | None] = {}
    for t in tasks:
        fits = [m for m in range(n_nodes) if free[m] >= t.demand]
        m = max(fits, key=lambda m: free[m]) if fits else None
        if m is not None:
            free[m] -= t.demand
        node_of[t.jid] = m
    return node_of


# --------------------------------------------------------------------------- #
#  Ground-truth contention scorer (the agents never see this)                   #
# --------------------------------------------------------------------------- #
def score(tasks, node_of) -> dict:
    """Total realised throughput + mean slowdown + #same-class co-located pairs, under the true
    contention model: a task slows to 1/(1+alpha*k_same), k_same = same-contending-class co-tenants."""
    by_node: dict[int, list[Task]] = {}
    cls = {t.jid: t.cls for t in tasks}
    for t in tasks:
        m = node_of.get(t.jid)
        by_node.setdefault(m, []).append(t)
    throughput = 0.0
    slow_sum = 0.0
    placed = 0
    same_pairs = 0
    for m, group in by_node.items():
        if m is None:
            continue                                       # unplaced -> contributes 0 throughput
        for t in group:
            a = ALPHA[t.cls]
            k_same = sum(1 for o in group if o.jid != t.jid and o.cls == t.cls) if t.cls in ALPHA else 0
            slowdown = 1.0 + a * k_same
            throughput += 1.0 / slowdown
            slow_sum += slowdown
            placed += 1
        # count contending same-class pairs on this node (a node-level diagnostic)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i].cls == group[j].cls and group[i].cls in {"compute", "bandwidth"}:
                    same_pairs += 1
    n = len(tasks)
    return {"throughput": throughput, "mean_slowdown": slow_sum / max(placed, 1),
            "same_pairs": float(same_pairs), "placed": float(placed)}


def classify(tasks, use_llm, model, cache, trace, seen, oracle=False) -> dict[str, str]:
    if oracle:
        return {t.jid: t.cls for t in tasks}
    out = {}
    for t in tasks:
        d = llm_bottleneck({"op_profile": t.profile}, use_llm=use_llm, model=model, cache=cache)
        out[t.jid] = d["bottleneck"]
        key = t.profile
        if key not in seen:
            seen.add(key)
            trace.append({"profile": t.profile, "true": t.cls, "pred": d["bottleneck"],
                          "why": d["justification"], "_source": d["_source"]})
    return out


# --------------------------------------------------------------------------- #
#  Sweep                                                                        #
# --------------------------------------------------------------------------- #
def sweep(n_nodes, gpus_per_node, demand, heteros, seeds, use_llm, model) -> None:
    cache: dict = {}
    trace: list = []
    seen: set = set()
    tag = "rule" if not use_llm else model

    print(f"\n{'='*82}")
    print(f"EXP 23 — LLM affinity + ILP placement vs numbers-only; classifier={tag}")
    print(f"{'='*82}")
    print(f"cluster {n_nodes}x{gpus_per_node} GPUs, {demand}-GPU single-node tasks "
          f"({(n_nodes*gpus_per_node)//demand} tasks fill it), mean of {len(seeds)} seeds.")
    print("Higher throughput / lower slowdown = better. 'pairs' = same-class co-located pairs "
          "(the contention).\n")
    header = (f"{'hetero':>7}  {'policy':<10} {'throughput':>11} {'slowdown':>9} "
              f"{'pairs':>6} {'vs spread':>10}")
    n_tasks = (n_nodes * gpus_per_node) // demand
    summary = {}
    for h in heteros:
        print("-" * len(header)); print(header); print("-" * len(header))
        accs = {p: {"throughput": 0.0, "mean_slowdown": 0.0, "same_pairs": 0.0}
                for p in ("pack", "spread", "llm-affin", "oracle")}
        for s in seeds:
            tasks = make_tasks(n_nodes, gpus_per_node, demand, h, s)
            llm_cls = classify(tasks, use_llm, model, cache, trace, seen)
            true_cls = classify(tasks, use_llm, model, cache, trace, seen, oracle=True)
            placements = {
                "pack":      place_ilp(tasks, n_nodes, gpus_per_node, None),
                "spread":    place_spread(tasks, n_nodes, gpus_per_node),
                "llm-affin": place_ilp(tasks, n_nodes, gpus_per_node,
                                       affinity_matrix(llm_cls, PENALTY)),
                "oracle":    place_ilp(tasks, n_nodes, gpus_per_node,
                                       affinity_matrix(true_cls, PENALTY)),
            }
            for p, node_of in placements.items():
                r = score(tasks, node_of)
                for k in accs[p]:
                    accs[p][k] += r[k]
        spread_tp = accs["spread"]["throughput"] / len(seeds)
        for p in ("pack", "spread", "llm-affin", "oracle"):
            a = {k: v / len(seeds) for k, v in accs[p].items()}
            delta = a["throughput"] - spread_tp
            summary[f"{h}|{p}"] = a
            print(f"{h:>7.2f}  {p:<10} {a['throughput']:>11.2f} {a['mean_slowdown']:>9.3f} "
                  f"{a['same_pairs']:>6.1f} {delta:>+10.2f}")
        print()
    print(f"Ideal throughput (zero contention) = {float(n_tasks):.0f}. 'vs spread' = throughput gain "
          "over the class-blind spreader.")
    print("GATE: llm-affin should beat spread at high hetero (semantic signal) and tie it at "
          "hetero=0 (all-compute -> no relational signal, per Exp 21).")

    if use_llm:
        acc = [d for d in trace if d["_source"].startswith("llm")]
        correct = sum(1 for d in acc if d["pred"] == d["true"])
        print(f"\nLLM bottleneck classification accuracy: {correct}/{len(acc)} distinct profiles")
        save_cache(cache)
    out = os.path.join(HERE, "results_affinity.json")
    with open(out, "w") as f:
        json.dump({"classifier": tag, "use_llm": use_llm, "summary": summary,
                   "classifications": trace}, f, indent=2)
    print(f"{len(trace)} distinct classifications/decisions -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 23 — LLM affinity placement gate")
    ap.add_argument("--llm", action="store_true", help="use qwen to classify bottlenecks (needs Ollama)")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--gpn", type=int, default=4, help="GPUs per node")
    ap.add_argument("--demand", type=int, default=2, help="GPUs per single-node task")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    sweep(a.nodes, a.gpn, a.demand, heteros=[0.0, 0.25, 0.5], seeds=list(range(a.seeds)),
          use_llm=a.llm, model=a.model)


if __name__ == "__main__":
    main()
