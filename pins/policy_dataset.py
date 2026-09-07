"""Turn counterfactual rollouts into a chat dataset split by workload window.

Two properties of the label distribution shape everything here, and both were measured rather
than assumed (see `policy_labels`):

1. One action -- resize_conservative+greedy, which takes the largest legal size and then never
   releases it -- deadlocks the cluster and is the WORST of the 15 in ~91% of states. Any metric
   built on `max - min` therefore measures how bad that one action is, not how much the real
   choice matters. Such arms are FLAGGED here and handled by the evaluator; nothing is dropped,
   because "don't do the catastrophic thing" is still a thing the model must learn.

2. The real choice is a near-tie: the margin between best and second is under 0.05 in ~62% of
   states. Training on a raw argmax would hand the model contradictory targets for states that
   are, for its purposes, identical. So the target is canonicalised within an epsilon band.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from pins.elastisim_bench import POLICY_MENU, POLICY_SELECT, POLICY_UNSAFE
from pins.policy_labels import OPERATING_POINTS, reward

EPS = 0.05          # actions within this of the best are treated as equivalent


def split_windows(states: list[dict], seed: int) -> dict[str, str]:
    """Assign complete windows before fitting any dataset-level statistic.

    Target canonicalisation used to run before this split, which let held-out
    epsilon bands affect the action popularity used to construct training
    targets.  Keeping this small helper pure also gives the leakage check a
    direct seam to test.
    """
    wins = sorted({s["window"] for s in states})
    random.Random(seed).shuffle(wins)
    n_test = max(1, round(0.15 * len(wins)))
    return {**{w: "test" for w in wins[:n_test]},
            **{w: "val" for w in wins[n_test:2 * n_test]},
            **{w: "train" for w in wins[2 * n_test:]}}


def load_states(labels: Path, weights: dict) -> list[dict]:
    """One record per state with a complete 15-action reward vector."""
    by: dict = collections.defaultdict(dict)
    for f in sorted(labels.glob("rollouts.*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by[(r["window"], r["t"], int(r.get("decision_horizon", 0)))][
                    f"{r['ordering']}+{r['sizing']}"] = r
    n_actions = len(POLICY_MENU["ordering"]) * len(POLICY_MENU["sizing"])
    out = []
    for (win, t, horizon), v in sorted(by.items()):
        if len(v) < n_actions:            # a chunk was interrupted part-way through this state
            continue
        sc = {a: reward(r, weights) for a, r in v.items()}
        out.append({"window": win, "t": t, "decision_horizon": horizon, "rewards": sc})
    return out


def canonical_targets(states: list[dict]) -> dict:
    """Pick one representative per epsilon-band, globally consistent.

    Within a band every action is worth the same to within EPS, so which one is "correct" is
    arbitrary -- but it must not be arbitrary PER STATE, or near-identical packets get different
    targets and the model is taught noise. Preferring the globally most frequent band member makes
    the mapping a function of the band, not of sampling luck.
    """
    popularity: collections.Counter = collections.Counter()
    for s in states:
        best = max(s["rewards"].values())
        for a, r in s["rewards"].items():
            if best - r <= EPS:
                popularity[a] += 1
    return popularity


def catastrophic_arms(states: list[dict], frac: float = 0.5) -> list[str]:
    """Arms that are the worst available choice in more than `frac` of states."""
    worst: collections.Counter = collections.Counter()
    for s in states:
        worst[min(s["rewards"], key=s["rewards"].get)] += 1
    return sorted(a for a, c in worst.items() if c > frac * len(states))


def build_row(s: dict, packet: str, popularity: collections.Counter) -> dict:
    sc = s["rewards"]
    order = sorted(sc.values(), reverse=True)
    best_r = order[0]
    band = [a for a, r in sc.items() if best_r - r <= EPS]
    target = max(band, key=lambda a: (popularity[a], a))
    o, z = target.split("+")
    return {
        "messages": [
            {"role": "system", "content": POLICY_SELECT},
            {"role": "user", "content": packet},
            {"role": "assistant", "content": json.dumps(
                {"ordering": o, "sizing": z,
                 "why": "selected from the menu for this cluster state"})},
        ],
        "window": s["window"], "t": s["t"],
        "decision_horizon": int(s.get("decision_horizon", 0)),
        "rewards": {k: round(v, 5) for k, v in sc.items()},
        "argmax": max(sc, key=sc.get), "target": target,
        "margin": round(best_r - order[1], 5), "band_size": len(band),
        "best_r": round(best_r, 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("runs/labels_full"))
    ap.add_argument("--out", type=Path, default=Path("runs/dataset"))
    ap.add_argument("--weights", choices=sorted(OPERATING_POINTS), default="balanced")
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = OPERATING_POINTS[a.weights]

    states = load_states(a.labels, w)
    packets = {}
    for f in a.labels.glob("*_packets.jsonl"):
        win = f.name[:-len("_packets.jsonl")]
        for line in f.read_text().splitlines():
            if line.strip():
                j = json.loads(line)
                packets[(win, j["t"])] = j["packet"]
    states = [s for s in states if (s["window"], s["t"]) in packets]

    # Split by WINDOW *before* learning even apparently harmless corpus-level
    # quantities. Consecutive epochs share a trajectory, and held-out reward
    # bands must not influence which representative labels the training set.
    split = split_windows(states, a.seed)
    train_states = [s for s in states if split[s["window"]] == "train"]
    popularity = canonical_targets(train_states)

    # Safety is a property of the mechanism, not a statistic fitted on the
    # test set. Keep the training-only observed diagnostic so the declaration
    # remains auditable without allowing held-out outcomes to define the menu.
    catastrophic = sorted(POLICY_UNSAFE)
    observed_worst_train = catastrophic_arms(train_states)

    rows: dict = {"train": [], "val": [], "test": []}
    for s in states:
        rows[split[s["window"]]].append(build_row(s, packets[(s["window"], s["t"])], popularity))
    for name, rs in rows.items():
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))

    leak = sum(("_true_dur" in r["messages"][1]["content"]) or
               ("_real_wait" in r["messages"][1]["content"])
               for rs in rows.values() for r in rs)
    meta = {"weights": a.weights, "eps": EPS, "seed": a.seed,
            "n_states": len(states), "catastrophic_arms": catastrophic,
            "unsafe_policy_source": "mechanism_semantics",
            "observed_majority_worst_train": observed_worst_train,
            "target_popularity_source": "train_only",
            "counts": {k: len(v) for k, v in rows.items()},
            "windows": {k: sorted({r["window"] for r in v}) for k, v in rows.items()},
            "leak_rows": leak,
            "majority_target": collections.Counter(
                r["target"] for r in rows["train"]).most_common(1)[0] if rows["train"] else None,
            "near_tie_frac": round(sum(r["margin"] < EPS for rs in rows.values() for r in rs)
                                   / max(1, len(states)), 3),
            "near_tie_frac_by_split": {
                k: round(sum(r["margin"] < EPS for r in rs) / max(1, len(rs)), 3)
                for k, rs in rows.items()}}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    assert leak == 0, "ORACLE LEAK: a packet contains a true-runtime field"
    assert not (set(meta["windows"]["train"]) & set(meta["windows"]["test"])), "window leaked across splits"


if __name__ == "__main__":
    main()
