"""Shard the frozen deterministic fixed-policy test sweep without changing its design."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from pins.elastisim_bench import run
from pins.freeze_core_2x2 import IMPLEMENTATION_FILES, combined_sha
from pins.run_core_2x2 import METRICS, fixed_cells, row_key, sha256, tag
from pins.verify_core_2x2 import verify


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
WORLDS = ROOT / "runs/core_2x2_worlds"
CORE_ROWS = ROOT / "runs/core_2x2/rows.jsonl"


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def append(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def collate(out: Path, manifest: dict) -> None:
    rows = read_rows([CORE_ROWS, *sorted(out.glob("rows.*.jsonl"))])
    selected = {}
    for row in rows:
        if row.get("structure") == "fixed" and row.get("split") == "test":
            selected[row_key(row)] = row
    expected = len([row for row in manifest["windows"] if row["split"] == "test"]) * \
        len(fixed_cells(manifest))
    sink = out / "rows.jsonl"
    sink.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                            for _, row in sorted(selected.items())))
    invalid = sum(row.get("status") == "invalid" for row in selected.values())
    status = {
        "attempted": len(selected), "valid": len(selected) - invalid,
        "invalid": invalid, "expected": expected,
        "complete": len(selected) == expected,
        "all_valid": len(selected) == expected and invalid == 0,
    }
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worlds", type=Path, default=WORLDS)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/core_2x2_fixed_test")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--allow-terminal-invariant-amendment", action="store_true",
                        help="allow the audited COMPLETED/KILLED mirror-state correction")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.collate:
        collate(args.out, manifest)
        return
    amended = False
    try:
        verify(args.manifest, args.worlds, check_results=False)
    except AssertionError as exc:
        if (not args.allow_terminal_invariant_amendment or
                "implementation differs from the frozen manifest" not in str(exc)):
            raise
        amended = True
        print("using audited terminal-state invariant amendment", flush=True)
    runtime_implementation_sha = combined_sha([ROOT / path for path in IMPLEMENTATION_FILES])
    windows = [row for row in manifest["windows"] if row["split"] == "test"]
    work = [(cell, window) for cell in fixed_cells(manifest) for window in windows]
    mine = work[args.shard::args.shards]
    existing = read_rows([CORE_ROWS, *sorted(args.out.glob("rows.*.jsonl"))])
    # Invalid attempts are auditable but are not completed work: after an invariant correction,
    # rerunning the shard must replace them with valid rows rather than silently skipping them.
    done = {row_key(row) for row in existing if row.get("status") != "invalid"}
    sink = args.out / f"rows.{args.shard}.jsonl"
    manifest_hash = sha256(args.manifest)
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True).stdout.strip()
    cfg = manifest["execution"]
    selector = manifest["selector"]
    for cell, window in mine:
        key_row = {
            "experiment_id": manifest["experiment_id"], "split": "test",
            "window": window["window"], **cell,
        }
        if row_key(key_row) in done:
            continue
        try:
            result = run(
                args.worlds / window["window"], cell["arm"],
                interval=cfg["simulation_interval_s"], tag=tag(cell), quiet=True,
                sizer=cell["sizing"],
                rcon_threshold_s=selector["rcon_threshold_s"])
        except AssertionError as exc:
            row = {
                **key_row, "artifact_tag": tag(cell),
                "manifest_sha256": manifest_hash,
                "processed_trace_sha256": manifest["processed_trace"]["sha256"],
                "implementation_sha256": manifest["implementation"]["sha256"],
                "runtime_implementation_sha256": runtime_implementation_sha,
                "implementation_amendment": (
                    "exclude terminal KILLED mirror assignments, as already done for COMPLETED"
                    if amended else None),
                "git_head": git_head, "model": None, "temperature": None,
                "num_predict": None, "status": "invalid", "error": str(exc),
                **{metric: None for metric in METRICS},
            }
            append(sink, row)
            done.add(row_key(row))
            print(f"shard={args.shard} INVALID test {window['window']} "
                  f"{cell['family']} {cell['ordering']}+{cell['sizing']}: {exc}",
                  flush=True)
            continue
        assert result["n"] == window["n_jobs"]
        row = {
            **key_row, "artifact_tag": tag(cell), "manifest_sha256": manifest_hash,
            "processed_trace_sha256": manifest["processed_trace"]["sha256"],
            "implementation_sha256": manifest["implementation"]["sha256"],
            "runtime_implementation_sha256": runtime_implementation_sha,
            "implementation_amendment": (
                "exclude terminal KILLED mirror assignments, as already done for COMPLETED"
                if amended else None),
            "git_head": git_head, "model": None, "temperature": None, "num_predict": None,
            "status": "valid", **{metric: result.get(metric) for metric in METRICS},
        }
        append(sink, row)
        done.add(row_key(row))
        print(f"shard={args.shard} test {window['window']} {cell['family']} "
              f"{cell['ordering']}+{cell['sizing']}: {row['deadline_viol_pct']}%", flush=True)
    (args.out / f"done.{args.shard}").write_text("1\n")


if __name__ == "__main__":
    main()
