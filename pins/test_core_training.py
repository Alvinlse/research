"""Fast checks for the current-space oracle and LoRA preparation pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pins.core_oracle_labels import (
    actions, collate, packet_path, policy_schedule, select_packets, state_flags,
    validation_selected_baselines,
)
from pins.eval_core_referee import parse_action, score


def test_action_spaces() -> None:
    assert len(actions("nm")) == 16
    assert len(actions("mkt")) == 16
    assert set(actions("nm")).isdisjoint(actions("mkt"))


def test_collate_and_no_oracle_leak() -> None:
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        rollout = out / "rollouts.0.jsonl"
        rows = []
        for split, window in (("train", "train0"), ("validation", "validation0"),
                              ("test", "test0")):
            for family in ("nm", "mkt"):
                path = packet_path(out, family, window)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"t": 100, "packet": f"visible {family} state"}) + "\n")
                for index, (ordering, sizing) in enumerate(actions(family)):
                    rows.append({
                        "key": f"{split}|{window}|{family}|100|{ordering}|{sizing}",
                        "split": split, "window": window, "family": family, "t": 100,
                        "ordering": ordering, "sizing": sizing,
                        "reward": -float(index), "deadline_viol_pct": float(index),
                    })
        rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
        meta = collate(out)
        assert meta["counts"] == {"train": 2, "validation": 2, "test": 2}
        assert meta["leak_rows"] == 0
        for split in meta["counts"]:
            dataset = [json.loads(line) for line in
                       (out / "dataset" / f"{split}.jsonl").read_text().splitlines()]
            assert {row["family"] for row in dataset} == {"nm", "mkt"}
            assert all("default_sizing" in row["messages"][-1]["content"] for row in dataset)


def test_parse_current_schema() -> None:
    answer = '{"ordering":"fairness","default_sizing":"adaptive","job_sizing":{}}'
    assert parse_action(answer) == "fairness+adaptive"
    assert parse_action("not json") == "UNPARSEABLE"


def test_invalid_action_uses_deployment_fallback() -> None:
    rows = [{
        "family": "nm",
        "target": "fairness+adaptive",
        "argmax": "fairness+adaptive",
        "rewards": {"fairness+adaptive": -2.0, "fcfs+greedy": -20.0},
    }]
    strict = score(rows, ["UNPARSEABLE"])
    deployed = score(rows, ["UNPARSEABLE"], {"nm": "fairness+adaptive"})
    assert strict["mean_regret"] == 18.0
    assert deployed["mean_regret"] == 0.0
    assert strict["invalid"] == deployed["invalid"] == 1
    assert strict["raw_target_accuracy"] == 0.0
    assert deployed["applied_target_accuracy"] == 1.0


def test_fixed_selection_never_reads_test_outcomes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        rows = []
        preferred = {"nm": ("fairness", "adaptive"),
                     "mkt": ("auction_deadline", "adaptive")}
        for family in ("nm", "mkt"):
            for ordering, sizing in actions(family):
                for window in ("validation0", "validation1"):
                    rows.append({
                        "structure": "fixed", "split": "validation", "family": family,
                        "window": window, "ordering": ordering, "sizing": sizing,
                        "deadline_viol_pct": 1.0 if (ordering, sizing) == preferred[family] else 5.0,
                    })
                # Make a different action look perfect on test; selection must ignore it.
                rows.append({
                    "structure": "fixed", "split": "test", "family": family,
                    "window": "test0", "ordering": ordering, "sizing": sizing,
                    "deadline_viol_pct": 0.0 if sizing == "greedy" else 10.0,
                })
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        assert validation_selected_baselines(path) == preferred


def test_one_interval_schedule_and_observable_strata() -> None:
    schedule = policy_schedule(3600, ("fcfs", "rcon"),
                               ("fairness", "adaptive"), 1800)
    assert schedule == [
        {"t": 0, "ordering": "fairness", "sizing": "adaptive"},
        {"t": 3600, "ordering": "fcfs", "sizing": "rcon"},
        {"t": 5400, "ordering": "fairness", "sizing": "adaptive"},
    ]
    hard = {"t": 100, "packet": (
        "pool=80 free=2 queue_depth=3 waiting_demand=8 "
        "negative_laxity_waiting=1")}
    routine = {"t": 200, "packet": (
        "pool=80 free=20 queue_depth=1 waiting_demand=8 "
        "negative_laxity_waiting=0")}
    assert state_flags(hard["packet"])["requested_contended"]
    assert state_flags(hard["packet"])["minimum_contended"]
    assert state_flags(hard["packet"])["deadline_pressure"]
    assert not state_flags(routine["packet"])["requested_contended"]
    assert select_packets([hard, routine, {"t": 300, "packet": routine["packet"]}], 2,
                          "pressure_stratified")[0] == hard


if __name__ == "__main__":
    test_action_spaces()
    test_collate_and_no_oracle_leak()
    test_parse_current_schema()
    test_invalid_action_uses_deployment_fallback()
    test_fixed_selection_never_reads_test_outcomes()
    test_one_interval_schedule_and_observable_strata()
    print("6 core-training tests passed")
