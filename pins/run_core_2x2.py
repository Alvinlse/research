"""Run the frozen, paired core 2x2 and its deterministic fixed-policy baselines.

Every result row carries enough provenance to reject accidental mixtures of traces, manifests,
code, models, seeds, or splits. The four factorial cells are the only inferential cells; fixed
policies are selected on validation and evaluated once on test by ``analyse_core_2x2``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from pins.elastisim_bench import run
from pins.verify_core_2x2 import verify

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
DEFAULT_WORLDS = ROOT / "runs/core_2x2_worlds"
DEFAULT_OUT = ROOT / "runs/core_2x2"

METRICS = (
    "deadline_viol_pct", "mean_bsd", "mean_wait_s", "p50_wait_s", "p90_wait_s",
    "useful_util_win", "util_win", "completed", "n", "killed_pct", "jain_user_wait",
    "resize_events", "resized_gpus", "resize_overhead_s", "rcon_blocks",
    "llm_calls", "tok_calls", "tok_prompt", "tok_completion", "tok_wall", "wall_s",
    "sel_invalid", "fallbacks", "llm_malformed", "llm_errors", "sel_counts", "market_clearings",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_cells(manifest: dict) -> list[dict]:
    return [
        {"structure": "fixed", "family": family, "ordering": ordering, "sizing": sizing,
         "arm": ordering, "seed": 0}
        for family, orderings in manifest["selector"]["families"].items()
        for ordering in orderings
        for sizing in manifest["selector"]["sizing_actions"]
    ]


def core_cells(manifest: dict) -> list[dict]:
    arms = {"single": "policy_select", "multi": "policy_negotiate"}
    return [
        {"structure": structure, "family": family, "ordering": "selected",
         "sizing": "per_job", "arm": arm, "seed": seed}
        for structure, arm in arms.items()
        for family in ("nm", "mkt")
        for seed in manifest["inference"]["paired_seeds"]
    ]


def row_key(row: dict) -> tuple:
    return (row["experiment_id"], row["split"], row["window"], row["structure"],
            row["family"], row["ordering"], row["sizing"], row["seed"])


def tag(cell: dict) -> str:
    if cell["structure"] == "fixed":
        return f"c2_fixed_{cell['ordering']}_{cell['sizing']}"
    return f"c2_{cell['structure']}_{cell['family']}_seed{cell['seed']}"


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--worlds", type=Path, default=DEFAULT_WORLDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phase", choices=("all", "baselines", "core"), default="all")
    ap.add_argument("--split", choices=("all", "validation", "test"), default="all")
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    verify(args.manifest, args.worlds, check_results=False)
    args.out.mkdir(parents=True, exist_ok=True)
    sink = args.out / "rows.jsonl"
    rows = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()] \
        if sink.exists() else []
    splits = manifest["execution"]["evaluation_splits"]
    if args.split != "all":
        splits = [args.split]
    windows = [w for w in manifest["windows"] if w["split"] in splits]
    cells = []
    if args.phase in ("all", "baselines"):
        cells += fixed_cells(manifest)
    if args.phase in ("all", "core"):
        cells += core_cells(manifest)
    expected = len(windows) * len(cells)
    if args.status:
        present = sum(row.get("experiment_id") == manifest["experiment_id"] and
                      row.get("split") in splits for row in rows)
        print(json.dumps({"present": present, "expected": expected, "complete": present >= expected}))
        return

    manifest_hash = sha256(args.manifest)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    done = {row_key(r) for r in rows}
    started = time.monotonic()
    completed_now = 0
    cfg = manifest["execution"]
    inference = manifest["inference"]
    selector = manifest["selector"]
    # Finish validation in full before touching test. This preserves a clean fixed-baseline
    # selection boundary even when the sweep is resumed in hundreds of short cron chunks.
    for split in splits:
        for cell in cells:
            for w in (item for item in windows if item["split"] == split):
                key_row = {
                    "experiment_id": manifest["experiment_id"], "split": w["split"],
                    "window": w["window"], **cell,
                }
                if row_key(key_row) in done:
                    continue
                result = run(
                    args.worlds / w["window"], cell["arm"], inference["model"],
                    interval=cfg["simulation_interval_s"], tag=tag(cell), quiet=True,
                    sizer=cell["sizing"] if cell["structure"] == "fixed" else "as_requested",
                    family=cell["family"] if cell["structure"] != "fixed" else "",
                    sel_every=selector["selection_interval_s"],
                    temperature=inference["temperature"], llm_seed=cell["seed"],
                    num_predict=inference["num_predict"],
                    rcon_threshold_s=selector["rcon_threshold_s"],
                )
                assert result["n"] == w["n_jobs"], (w["window"], result["n"], w["n_jobs"])
                assert all(float(result[name]) >= 0 for name in
                           ("deadline_viol_pct", "mean_bsd", "mean_wait_s", "p50_wait_s",
                            "p90_wait_s", "resize_events", "resized_gpus"))
                row = {
                    **key_row, "artifact_tag": tag(cell), "manifest_sha256": manifest_hash,
                    "processed_trace_sha256": manifest["processed_trace"]["sha256"],
                    "implementation_sha256": manifest["implementation"]["sha256"],
                    "git_head": git_head,
                    "model": inference["model"] if cell["structure"] != "fixed" else None,
                    "temperature": inference["temperature"] if cell["structure"] != "fixed" else None,
                    "num_predict": inference["num_predict"] if cell["structure"] != "fixed" else None,
                    **{metric: result.get(metric) for metric in METRICS},
                }
                append_row(sink, row)
                rows.append(row)
                done.add(row_key(row))
                completed_now += 1
                print(f"{w['split']} {w['window']} {cell['structure']}/{cell['family']} "
                      f"{cell['ordering']}+{cell['sizing']} seed={cell['seed']}: "
                      f"deadline={row['deadline_viol_pct']}% useful={row['useful_util_win']}",
                      flush=True)
                if args.max_runs and completed_now >= args.max_runs:
                    return
                if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                    return


if __name__ == "__main__":
    main()
