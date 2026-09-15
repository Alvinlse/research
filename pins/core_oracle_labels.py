"""Oracle labels for LoRA tuning of the frozen core-2x2 policy Referee.

The online model never sees outcomes.  Offline, this teacher replays each candidate policy from
the same observable state and labels the action with the lowest deadline-violation rate.  Runs are
append-only and resumable because generating the counterfactual library takes much longer than one
login-node process lifetime.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from pins.elastisim_bench import POLICY_FAMILY, POLICY_SELECT, SELECTOR_MENU, run


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
BASELINE = {"nm": ("fairness", "adaptive"),
            "mkt": ("auction_deadline", "adaptive")}
EPS_PP = 0.1  # deadline-rate differences within 0.1 percentage point are treated as ties


def validation_selected_baselines(rows_path: Path) -> dict[str, tuple[str, str]]:
    """Select each fixed family on validation only; never inspect test outcomes."""
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    fixed = [row for row in rows
             if row.get("structure") == "fixed" and row.get("split") == "validation"]
    selected = {}
    for family in POLICY_FAMILY:
        grouped: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        for row in fixed:
            if row.get("family") == family:
                grouped[(row["ordering"], row["sizing"])].append(row)
        expected = set(actions(family))
        if set(grouped) != expected:
            missing = sorted(expected - set(grouped))
            raise ValueError(f"incomplete validation fixed-policy menu for {family}: {missing}")
        window_counts = {action: len({row["window"] for row in candidates})
                         for action, candidates in grouped.items()}
        if len(set(window_counts.values())) != 1 or next(iter(window_counts.values())) == 0:
            raise ValueError(f"unbalanced validation fixed-policy rows for {family}")
        selected[family] = min(
            grouped,
            key=lambda action: (statistics.mean(
                float(row["deadline_viol_pct"]) for row in grouped[action]), action))
    return selected


def policy_schedule(timestamp: int, candidate: tuple[str, str],
                    baseline: tuple[str, str], horizon_s: int) -> list[dict]:
    """Apply one candidate intervention, then restore the validation-selected floor."""
    if horizon_s <= 0:
        raise ValueError("decision horizon must be positive")
    return [
        {"t": 0, "ordering": baseline[0], "sizing": baseline[1]},
        {"t": timestamp, "ordering": candidate[0], "sizing": candidate[1]},
        {"t": timestamp + horizon_s, "ordering": baseline[0], "sizing": baseline[1]},
    ]


def state_flags(packet: str) -> dict[str, float | bool]:
    """Derive preregisterable pressure strata from observable pre-decision fields only."""
    fields = {name: float(value) for name, value in
              re.findall(r"(?:^|\s)([a-zA-Z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)", packet)}
    free = fields.get("free", 0.0)
    queue_depth = fields.get("queue_depth", 0.0)
    waiting_demand = fields.get("waiting_demand", 0.0)
    negative_laxity = fields.get("negative_laxity_waiting", 0.0)
    return {
        "free_gpus": free,
        "queue_depth": queue_depth,
        "waiting_demand": waiting_demand,
        "negative_laxity_waiting": negative_laxity,
        # With the frozen bounds every pending job has g_min=1.
        "minimum_contended": queue_depth > free,
        "requested_contended": waiting_demand > free,
        "deadline_pressure": negative_laxity > 0,
    }


def select_packets(rows: list[dict], keep: int, sampling: str) -> list[dict]:
    if len(rows) <= keep:
        return rows
    if sampling == "timeline":
        step = len(rows) / keep
        return [rows[int(i * step)] for i in range(keep)]

    def difficult(row: dict) -> bool:
        flags = state_flags(row["packet"])
        return bool(flags["requested_contended"] or flags["deadline_pressure"])

    def pressure(row: dict) -> tuple:
        flags = state_flags(row["packet"])
        free = max(1.0, float(flags["free_gpus"]))
        return (bool(flags["requested_contended"]), bool(flags["deadline_pressure"]),
                float(flags["waiting_demand"]) / free,
                float(flags["negative_laxity_waiting"]), row["t"])

    hard = [row for row in rows if difficult(row)]
    routine = [row for row in rows if not difficult(row)]
    chosen = []
    if hard:
        chosen.append(max(hard, key=pressure))
    if routine and len(chosen) < keep:
        chosen.append(routine[len(routine) // 2])
    remaining = [row for row in rows if row not in chosen]
    while len(chosen) < keep and remaining:
        # Fill deterministically across the timeline without consulting any outcome.
        index = int((len(chosen) / keep) * len(remaining))
        chosen.append(remaining.pop(min(index, len(remaining) - 1)))
    return sorted(chosen, key=lambda row: row["t"])


def actions(family: str) -> list[tuple[str, str]]:
    return [(ordering, sizing) for ordering in POLICY_FAMILY[family]
            for sizing in SELECTOR_MENU["sizing"]]


def rollout_files(out: Path) -> list[Path]:
    return sorted(out.glob("rollouts.*.jsonl"))


def load_rollouts(out: Path) -> list[dict]:
    rows = []
    for path in rollout_files(out):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def packet_path(out: Path, family: str, window: str) -> Path:
    return out / "packets" / family / f"{window}.jsonl"


def sampled_packets(world: Path, out: Path, family: str, every: int, keep: int,
                    lo: float, hi: float, baseline: tuple[str, str],
                    sampling: str) -> list[dict]:
    path = packet_path(out, family, world.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        ordering, sizing = baseline
        schedule = out / f"_probe_{family}.json"
        schedule.write_text(json.dumps([{"t": 0, "ordering": ordering, "sizing": sizing}]))
        run(world, "scripted", tag=f"core_oracle_probe_{family}", quiet=True,
            family=family, policy_schedule=schedule, packet_every=every, packet_out=path,
            rcon_threshold_s=300.0)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return []
    end = max(row["t"] for row in rows)
    rows = [row for row in rows if lo * end <= row["t"] <= hi * end]
    return select_packets(rows, keep, sampling)


def generate(args, windows: list[dict]) -> None:
    worker = args.worker_id if args.worker_id >= 0 else args.shard
    sink = args.out / f"rollouts.{worker}.jsonl"
    failure_sink = args.out / f"failures.{worker}.jsonl"
    existing = load_rollouts(args.out)
    done = {row["key"] for row in existing}
    blocked_states = set()
    for path in sorted(args.out.glob("failures.*.jsonl")):
        blocked_states.update(json.loads(line)["state"]
                              for line in path.read_text().splitlines() if line.strip())
    budget = args.max_rollouts
    baselines = validation_selected_baselines(args.fixed_rows)
    mine = windows[args.shard::args.shards]
    for item in mine:
        world = args.worlds / item["window"]
        for family in args.families:
            baseline = baselines[family]
            packets = sampled_packets(world, args.out, family, args.every, args.epochs,
                                      args.lo, args.hi, baseline, args.sampling)
            base_ordering, base_sizing = baseline
            for packet in packets:
                state = f"{item['split']}|{item['window']}|{family}|{packet['t']}"
                if state in blocked_states:
                    continue
                for ordering, sizing in actions(family):
                    key = (f"{item['split']}|{item['window']}|{family}|{packet['t']}|"
                           f"{ordering}|{sizing}")
                    if key in done:
                        continue
                    schedule = args.out / f"_schedule.{worker}.json"
                    schedule.write_text(json.dumps(policy_schedule(
                        packet["t"], (ordering, sizing), baseline,
                        args.decision_horizon_s)))
                    try:
                        result = run(world, "scripted", tag=f"core_oracle_{family}", quiet=True,
                                     family=family, policy_schedule=schedule,
                                     rcon_threshold_s=300.0)
                    except AssertionError as error:
                        # A reproducible simulator invariant failure makes the entire state an
                        # invalid teacher example. Keep the failure auditable and discard the
                        # state rather than retrying forever or teaching from an incomplete menu.
                        failure = {"state": state, "key": key, "error": str(error)}
                        with failure_sink.open("a") as handle:
                            handle.write(json.dumps(failure, sort_keys=True) + "\n")
                            handle.flush()
                        blocked_states.add(state)
                        print(f"BLOCKED STATE {state}: {error}", flush=True)
                        break
                    row = {
                        "key": key, "split": item["split"], "window": item["window"],
                        "family": family, "t": packet["t"], "ordering": ordering,
                        "sizing": sizing, "reward": -float(result["deadline_viol_pct"]),
                        "decision_horizon_s": args.decision_horizon_s,
                        "baseline_ordering": base_ordering,
                        "baseline_sizing": base_sizing,
                        "baseline_selection_split": "validation",
                        "state_flags": state_flags(packet["packet"]),
                        **{name: result.get(name) for name in
                           ("deadline_viol_pct", "mean_bsd", "mean_wait_s",
                            "useful_util_win", "resize_events")},
                    }
                    with sink.open("a") as handle:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        handle.flush()
                    done.add(key)
                    print(f"{len(done)} {key}: deadline={result['deadline_viol_pct']}%", flush=True)
                    if budget:
                        budget -= 1
                        if budget == 0:
                            return
    phase = "-".join(args.splits)
    families = "-".join(args.families)
    (args.out / f"done.{phase}.{families}.{worker}").write_text("1\n")


def collate(out: Path) -> dict:
    grouped: dict[tuple, dict] = collections.defaultdict(dict)
    for row in load_rollouts(out):
        state = (row["split"], row["window"], row["family"], row["t"])
        grouped[state][f"{row['ordering']}+{row['sizing']}"] = row

    complete = []
    for state, candidates in grouped.items():
        family = state[2]
        if len(candidates) != len(actions(family)):
            continue
        complete.append((state, candidates))

    popularity: dict[str, collections.Counter] = {
        family: collections.Counter() for family in POLICY_FAMILY}
    for (split, _window, family, _t), candidates in complete:
        if split != "train":
            continue
        best = max(row["reward"] for row in candidates.values())
        popularity[family].update(action for action, row in candidates.items()
                                  if best - row["reward"] <= EPS_PP)

    dataset = {"train": [], "validation": [], "test": []}
    for (split, window, family, timestamp), candidates in sorted(complete):
        best = max(row["reward"] for row in candidates.values())
        band = [action for action, row in candidates.items()
                if best - row["reward"] <= EPS_PP]
        target = max(band, key=lambda action: (popularity[family][action], action))
        ordering, sizing = target.split("+", 1)
        packet_doc = packet_path(out, family, window)
        packets = {row["t"]: row["packet"] for row in
                   (json.loads(line) for line in packet_doc.read_text().splitlines() if line.strip())}
        ordered_rewards = sorted((row["reward"] for row in candidates.values()), reverse=True)
        dataset[split].append({
            "messages": [
                {"role": "system", "content": POLICY_SELECT},
                {"role": "user", "content": packets[timestamp]},
                {"role": "assistant", "content": json.dumps({
                    "ordering": ordering, "default_sizing": sizing,
                    "job_sizing": {},
                    "why": "oracle-selected policy for this observable cluster state",
                })},
            ],
            "window": window, "family": family, "t": timestamp, "target": target,
            "argmax": max(candidates, key=lambda action: candidates[action]["reward"]),
            "state_flags": state_flags(packets[timestamp]),
            "decision_horizon_s": next(iter(candidates.values())).get("decision_horizon_s"),
            "baseline": (
                f"{next(iter(candidates.values())).get('baseline_ordering')}+"
                f"{next(iter(candidates.values())).get('baseline_sizing')}"
                if next(iter(candidates.values())).get("baseline_ordering") else None),
            "margin": round(ordered_rewards[0] - ordered_rewards[1], 6),
            "spread": round(ordered_rewards[0] - ordered_rewards[-1], 6),
            "rewards": {action: round(row["reward"], 6)
                        for action, row in sorted(candidates.items())},
        })

    data_dir = out / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in dataset.items():
        (data_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    leak = sum("_true_dur" in row["messages"][1]["content"]
               for rows in dataset.values() for row in rows)
    horizons = sorted({row.get("decision_horizon_s") for rows in dataset.values()
                       for row in rows if row.get("decision_horizon_s") is not None})
    baselines = {family: sorted({row.get("baseline") for rows in dataset.values()
                                 for row in rows if row["family"] == family and row.get("baseline")})
                 for family in POLICY_FAMILY}
    meta = {
        "version": "core-referee-oracle-v2" if horizons else "core-referee-oracle-v1",
        "objective": "deadline_viol_pct",
        "epsilon_percentage_points": EPS_PP,
        "decision_horizon_s": horizons,
        "baseline_selection_split": "validation" if horizons else None,
        "validation_selected_baselines": baselines,
        "contention_definition": "waiting_demand > free_gpus",
        "actions_per_family": {family: len(actions(family)) for family in POLICY_FAMILY},
        "counts": {split: len(rows) for split, rows in dataset.items()},
        "windows": {split: sorted({row["window"] for row in rows})
                    for split, rows in dataset.items()},
        "leak_rows": leak,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    assert leak == 0, "oracle-only runtime leaked into an online packet"
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        assert not (set(meta["windows"][left]) & set(meta["windows"][right]))
    print(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worlds", type=Path, default=ROOT / "runs/core_2x2_worlds")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/core_referee_oracle_v2")
    parser.add_argument("--fixed-rows", type=Path,
                        default=ROOT / "runs/core_2x2/rows.jsonl",
                        help="fixed-policy rows used for validation-only baseline selection")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                        default=("train", "validation"))
    parser.add_argument("--families", nargs="+", choices=tuple(POLICY_FAMILY),
                        default=tuple(POLICY_FAMILY))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--every", type=int, default=1800)
    parser.add_argument("--decision-horizon-s", type=int, default=1800,
                        help="candidate duration before restoring the fixed baseline")
    parser.add_argument("--sampling", choices=("timeline", "pressure_stratified"),
                        default="timeline")
    parser.add_argument("--lo", type=float, default=0.05)
    parser.add_argument("--hi", type=float, default=0.60)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=-1,
                        help="output/scratch id when redistributing one shard's remaining windows")
    parser.add_argument("--window-names", nargs="*",
                        help="optional explicit subset of manifest window names")
    parser.add_argument("--max-rollouts", type=int, default=0)
    parser.add_argument("--collate", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.collate:
        collate(args.out)
        return
    manifest = json.loads(args.manifest.read_text())
    windows = [item for item in manifest["windows"] if item["split"] in args.splits]
    if args.window_names:
        requested = set(args.window_names)
        windows = [item for item in windows if item["window"] in requested]
        missing = requested - {item["window"] for item in windows}
        if missing:
            raise SystemExit(f"requested windows not in selected splits: {sorted(missing)}")
    if args.limit:
        windows = windows[:args.limit]
    generate(args, windows)


if __name__ == "__main__":
    main()
