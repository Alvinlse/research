"""Counterfactual policy labels: for each sampled state, what every policy in the menu would do.

The teacher is the simulator. At a sampled epoch `t` the baseline policy has been running since
`t=0`; each candidate `(ordering, sizing)` then takes over for the rest of the window and the
outcome is scored. The reward vector over all candidates is the label, and its argmax is the
supervised target. True runtime is used freely HERE -- inside the teacher -- and never reaches the
packet the referee reads, which is the whole point of distilling a privileged teacher.

Cost is what makes this feasible: one window replays in ~2 s, so K x epochs x windows rollouts is
hours, not weeks, and no simulator snapshot/restore is needed. Because the login node reaps a
shell at ~12-15 CPU-min, every rollout is appended to a JSONL the moment it finishes and completed
keys are skipped on restart -- run it in as many chunks as it takes.

    .venv/bin/python -m pins.policy_labels --worlds runs/holdout --out runs/labels --limit 2
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pins.elastisim_bench import POLICY_MENU, run

# The state the counterfactual branches FROM. Held fixed across every rollout of a given epoch so
# the only thing that differs between candidates is the policy that takes over at t.
BASELINE = {"ordering": "market", "sizing": "as_requested"}

# Reward weights, as named operating points so the sensitivity sweep is a flag and not an edit.
# `util_useful` is deliberately absent from all of them: util_win is identical (0.725) across the
# whole library because allocated GPU-hours are fixed by the trace, so it would contribute zero
# variance to every label.
#
# `balanced` was CALIBRATED, not guessed. A first pass weighting bsd 4.0 and resize 0.5 gave bsd a
# 51% share of total cost and resize 16%, which made "never resize" win states where it waited 53%
# longer -- a degenerate policy winning on a churn term. These shares put the service metrics in
# front: wait ~35%, bsd ~25%, sla ~20%, fairness ~15%, churn ~5%.
OPERATING_POINTS = {
    "balanced":       {"sla10": 4.0, "bsd": 1.0, "wait_h": 2.5, "resize": 0.1, "unfair": 4.0},
    "throughput":     {"sla10": 1.0, "bsd": 1.0, "wait_h": 5.0, "resize": 0.1, "unfair": 1.0},
    "deadline":       {"sla10": 12.0, "bsd": 2.0, "wait_h": 1.0, "resize": 0.1, "unfair": 1.0},
    "fairness_first": {"sla10": 2.0, "bsd": 1.0, "wait_h": 1.5, "resize": 0.1, "unfair": 12.0},
}
WEIGHTS = OPERATING_POINTS["balanced"]

def reward(r: dict, w: dict | None = None) -> float:
    """Higher is better. A plain negative cost, so regret is a difference of these.

    `w` is resolved at CALL time, not bound as a default: the operating point is chosen by a flag
    after import, and a default argument would freeze the weights the module was loaded with.
    """
    w = w or WEIGHTS
    n = max(1, r.get("n", 1))
    return -(w["sla10"] * r.get("sla10_viol_pct", 0) / 100
             + w["bsd"] * (r.get("mean_bsd", 1) - 1) / 10
             + w["wait_h"] * r.get("mean_wait_s", 0) / 3600 / 24
             + w["resize"] * r.get("resize_events", 0) / n
             + w["unfair"] * (1 - r.get("jain_user_wait", 1)) / 10)


def actions() -> list[tuple[str, str]]:
    return [(o, z) for o in POLICY_MENU["ordering"] for z in POLICY_MENU["sizing"]]


def epochs_for(world: Path, out: Path, every: int, keep: int,
               lo: float = 0.05, hi: float = 0.60) -> list[dict]:
    """Dump the referee-visible packets along the baseline trajectory, then thin them.

    Evenly spaced rather than densest-queue: over-sampling the busiest moments would train on a
    state distribution the referee will not meet online, which is the distribution-shift trap.

    Restricted to the [lo, hi] fraction of the window, because the choice's importance decays hard
    with switch time -- measured on d104h18, the reward gap runs 17.27 at t=198, 2.12 at t=45043,
    0.78 at t=73800 and 0.41 at t=99617. The first packet is a WINDOW-level policy choice, not a
    per-epoch one, and a late switch leaves the candidate too little runway to matter, so its
    argmax is decided by noise. The label this yields is "the best policy to HOLD for the rest of
    the window", which is a different question from "the best policy for the next instant" -- name
    it that way in the write-up.
    """
    pk = out / f"{world.name}_packets.jsonl"
    if not pk.exists():
        sched = out / f"{world.name}_baseline.json"
        sched.write_text(json.dumps([{"t": 0, **BASELINE}]))
        run(world, "scripted", tag="lbl_probe", quiet=True, policy_schedule=sched,
            packet_every=every, packet_out=pk)
    rows = [json.loads(l) for l in pk.read_text().splitlines() if l.strip()]
    if not rows:
        return []
    end = max(r["t"] for r in rows)
    rows = [r for r in rows if lo * end <= r["t"] <= hi * end]
    if len(rows) <= keep:
        return rows
    step = len(rows) / keep
    return [rows[int(i * step)] for i in range(keep)]


def label_world(world: Path, out: Path, every: int, keep: int, done: set,
                lo: float, hi: float, shard: int = 0, budget: list | None = None) -> int:
    """`budget` is a one-element list acting as a mutable counter: this invocation stops once it
    hits zero, so a chunk always ends well inside the login node's ~12-15 CPU-min reaper window
    instead of being killed mid-rollout."""
    sink = out / f"rollouts.{shard}.jsonl"
    acts, n_new = actions(), 0
    for ep in epochs_for(world, out, every, keep, lo, hi):
        for o, z in acts:
            key = f"{world.name}|{ep['t']}|{o}|{z}"
            if key in done:
                continue
            sched = out / "_sched.json"
            sched.write_text(json.dumps([{"t": 0, **BASELINE},
                                         {"t": ep["t"], "ordering": o, "sizing": z}]))
            r = run(world, "scripted", tag="lbl", quiet=True, policy_schedule=sched)
            with open(sink, "a") as f:
                # Raw metric names are kept VERBATIM so `reward()` can re-score a stored row
                # directly; renaming them here is what would silently make collate() score zeros.
                f.write(json.dumps({"key": key, "window": world.name, "t": ep["t"],
                                    "ordering": o, "sizing": z, "R": round(reward(r), 6),
                                    **{k: r.get(k) for k in
                                       ("n", "sla10_viol_pct", "mean_bsd", "mean_wait_s",
                                        "jain_user_wait", "jain_user_gpu", "resize_events",
                                        "util_win", "p90_wait_s")}}) + "\n")
            done.add(key); n_new += 1
            if budget is not None:
                budget[0] -= 1
                if budget[0] <= 0:
                    return n_new
    return n_new


def _all_rollout_lines(out: Path) -> list[str]:
    """Every shard's rollouts. Workers take disjoint WINDOWS and each appends to its own file, so
    there is never a concurrent append to one file -- which on NFS is not reliably atomic."""
    lines: list[str] = []
    for f in sorted(out.glob("rollouts.*.jsonl")):
        lines += f.read_text().splitlines()
    return lines


def collate(out: Path) -> list[dict]:
    """One row per state: the full reward vector, its argmax, and how much the choice matters."""
    # Re-score from the stored RAW metrics rather than the stored R. Every rollout keeps the
    # quantities the reward is built from, so trying another operating point is a re-collate over
    # existing JSONL -- no simulator time at all. The weight sensitivity study is therefore free.
    by: dict = {}
    for line in _all_rollout_lines(out):
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault((r["window"], r["t"]), {})[f"{r['ordering']}+{r['sizing']}"] = reward(r)
    states = []
    for (win, t), vec in sorted(by.items()):
        if len(vec) < len(actions()):          # a chunk was interrupted mid-state
            continue
        best = max(vec, key=vec.get)
        states.append({"window": win, "t": t, "rewards": vec, "best": best,
                       "gap": round(max(vec.values()) - min(vec.values()), 6)})
    return states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=Path, required=True, help="directory of built world dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=int, default=1800, help="packet cadence, simulated seconds")
    ap.add_argument("--epochs", type=int, default=12, help="states kept per window")
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows (smoke runs)")
    ap.add_argument("--lo", type=float, default=0.05, help="earliest switch point, window fraction")
    ap.add_argument("--hi", type=float, default=0.60, help="latest switch point, window fraction")
    ap.add_argument("--weights", choices=sorted(OPERATING_POINTS), default="balanced")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--shards", type=int, default=1, help="number of workers; windows are split N-ways")
    ap.add_argument("--max-rollouts", type=int, default=0,
                    help="stop after N new rollouts this invocation (0 = unbounded)")
    ap.add_argument("--collate", action="store_true", help="write states.json and the summary")
    a = ap.parse_args()
    globals()["WEIGHTS"] = OPERATING_POINTS[a.weights]
    print(f"operating point {a.weights}: {WEIGHTS}")
    a.out.mkdir(parents=True, exist_ok=True)

    done = {json.loads(l)["key"] for l in _all_rollout_lines(a.out) if l.strip()}
    print(f"resuming with {len(done)} rollouts already done")

    worlds = sorted(w for w in a.worlds.iterdir() if (w / "in/jobs.json").exists())
    if a.limit:
        worlds = worlds[:a.limit]
    mine = worlds[a.shard::a.shards]
    budget = [a.max_rollouts] if a.max_rollouts else None
    for i, w in enumerate(mine, 1):
        n = label_world(w, a.out, a.every, a.epochs, done, a.lo, a.hi, a.shard, budget)
        print(f"[shard {a.shard}] [{i}/{len(mine)}] {w.name}: +{n} rollouts", flush=True)
        if budget is not None and budget[0] <= 0:
            print("chunk budget spent; exiting for the next tick to resume", flush=True)
            return
    # Only a shard that walked its whole list without spending its budget is finished.
    (a.out / f"done.{a.shard}").write_text("1")
    print(f"shard {a.shard} COMPLETE", flush=True)

    if not a.collate:       # a plain worker tick must NOT touch states.json: writing the empty
        return              # list here would clobber a finished collation with "[]"
    states = collate(a.out)
    (a.out / "states.json").write_text(json.dumps(states, indent=1))
    if states:
        gaps = sorted(s["gap"] for s in states)
        tau = gaps[int(0.7 * (len(gaps) - 1))]
        wins = {}
        for s in states:
            wins[s["best"]] = wins.get(s["best"], 0) + 1
        print(f"\n{len(states)} states | gap median {statistics.median(gaps):.4f} "
              f"max {gaps[-1]:.4f} | decisive threshold (p70) {tau:.4f}")
        print(f"decisive states: {sum(g > tau for g in gaps)}/{len(gaps)} "
              f"({100 * sum(g > tau for g in gaps) / len(gaps):.0f}%)")
        print("argmax distribution:", dict(sorted(wins.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
