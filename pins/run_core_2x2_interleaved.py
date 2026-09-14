"""Run the one-seed held-out selector follow-up in paired, window-interleaved order.

The policy implementation is the frozen core implementation.  This driver changes only execution
order and supplies each family with the fixed action selected strictly on validation.  A malformed
or off-family LLM action therefore executes that independent floor and remains counted.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
from pathlib import Path

from pins.elastisim_bench import POLICY_FAMILY, SELECTOR_MENU, run
from pins.run_core_2x2 import METRICS, sha256
from pins.verify_core_2x2 import verify


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
WORLDS = ROOT / "runs/core_2x2_worlds"
VALIDATION_ROWS = ROOT / "runs/core_2x2/rows.jsonl"
OUT = ROOT / "runs/core_2x2_interleaved_test"


def validation_floors(path: Path) -> dict[str, tuple[str, str]]:
    """Select fixed actions on validation only and require the complete balanced menu."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    fixed = [row for row in rows
             if row.get("structure") == "fixed" and row.get("split") == "validation"]
    floors = {}
    for family, orderings in POLICY_FAMILY.items():
        grouped: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        for row in fixed:
            if row.get("family") == family:
                grouped[(row["ordering"], row["sizing"])].append(row)
        expected = {(ordering, sizing) for ordering in orderings
                    for sizing in SELECTOR_MENU["sizing"]}
        if set(grouped) != expected:
            raise ValueError(f"incomplete validation menu for {family}")
        counts = {action: len({row["window"] for row in candidates})
                  for action, candidates in grouped.items()}
        if set(counts.values()) != {12}:
            raise ValueError(f"unbalanced validation menu for {family}: {counts}")
        floors[family] = min(
            grouped,
            key=lambda action: (
                statistics.mean(float(row["deadline_viol_pct"])
                                for row in grouped[action]), action))
    return floors


def key(row: dict) -> tuple:
    return (row["window"], row["structure"], row["family"], row["seed"])


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worlds", type=Path, default=WORLDS)
    parser.add_argument("--validation-rows", type=Path, default=VALIDATION_ROWS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    verify(args.manifest, args.worlds, check_results=False)
    floors = validation_floors(args.validation_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    sink = args.out / "rows.jsonl"
    rows = ([json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
            if sink.exists() else [])
    windows = [window for window in manifest["windows"] if window["split"] == "test"]
    seed = manifest["inference"]["paired_seeds"]
    if len(seed) != 1:
        raise ValueError(f"interleaved follow-up requires the frozen one-seed amendment: {seed}")
    seed = seed[0]
    cells = [("single", "nm", "policy_select"),
             ("multi", "nm", "policy_negotiate"),
             ("single", "mkt", "policy_select"),
             ("multi", "mkt", "policy_negotiate")]
    expected = len(windows) * len(cells)
    present = {key(row) for row in rows}
    if args.status:
        print(json.dumps({"present": len(present), "expected": expected,
                          "complete": len(present) == expected,
                          "validation_selected_floors": {
                              family: "+".join(action) for family, action in floors.items()},
                          "next": next((f"{window['window']}:{structure}-{family}"
                                        for window in windows
                                        for structure, family, _arm in cells
                                        if (window["window"], structure, family, seed) not in present),
                                       None)}, indent=2))
        return

    implementation_hash = manifest["implementation"]["sha256"]
    manifest_hash = sha256(args.manifest)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True).stdout.strip()
    cfg, inference, selector = (manifest["execution"], manifest["inference"],
                                manifest["selector"])
    completed = 0
    for window in windows:
        for structure, family, arm in cells:
            key_row = (window["window"], structure, family, seed)
            if key_row in present:
                continue
            fallback_ordering, fallback_sizing = floors[family]
            tag = f"c2t_{structure}_{family}_seed{seed}"
            result = run(
                args.worlds / window["window"], arm, inference["model"],
                interval=cfg["simulation_interval_s"], tag=tag, quiet=True,
                sizer="as_requested", family=family,
                sel_every=selector["selection_interval_s"],
                temperature=inference["temperature"], llm_seed=seed,
                num_predict=inference["num_predict"],
                rcon_threshold_s=selector["rcon_threshold_s"],
                packet_v2=False, demand_v2=False,
                fallback_ordering=fallback_ordering,
                fallback_sizing=fallback_sizing)
            if result["n"] != window["n_jobs"]:
                raise AssertionError((window["window"], result["n"], window["n_jobs"]))
            row = {
                "experiment_id": manifest["experiment_id"],
                "followup": "interleaved-heldout-one-seed",
                "split": "test", "window": window["window"],
                "structure": structure, "family": family,
                "ordering": "selected", "sizing": "per_job", "arm": arm,
                "seed": seed, "artifact_tag": tag,
                "fallback_policy": f"{fallback_ordering}+{fallback_sizing}",
                "fallback_selection_split": "validation",
                "packet_v2": False, "demand_v2": False,
                "json_contract": "ollama-format-json plus strict family/action validator",
                "manifest_sha256": manifest_hash,
                "processed_trace_sha256": manifest["processed_trace"]["sha256"],
                "implementation_sha256": implementation_hash,
                "git_head": git_head, "model": inference["model"],
                "temperature": inference["temperature"],
                "num_predict": inference["num_predict"],
                **{metric: result.get(metric) for metric in METRICS},
            }
            append(sink, row)
            rows.append(row)
            present.add(key_row)
            completed += 1
            print(f"{window['window']} {structure}-{family}: "
                  f"deadline={row['deadline_viol_pct']}% wait={row['mean_wait_s']}s "
                  f"calls={row['llm_calls']} invalid={row['sel_invalid']}", flush=True)
            if args.max_runs and completed >= args.max_runs:
                return


if __name__ == "__main__":
    main()
