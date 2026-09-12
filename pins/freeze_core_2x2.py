"""Freeze the zero-shot core 2x2 workload before any scheduler outcomes are inspected.

This command is the only supported way to create ``core_2x2_manifest.json`` and its worlds.
Selection uses workload characteristics only, excludes every legacy frozen-2x2 interval, and
keeps complete warm-up + measurement intervals disjoint. Generated data live under ignored
``data/`` and ``runs/``; the small manifest is committed.

    .venv/bin/python -m pins.freeze_core_2x2
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pins.elastisim_bench import GPU_TRES, SLURM_LOG, TIMELIMIT_SENTINELS, _trace_rows, build

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
WORLDS = ROOT / "runs/core_2x2_worlds"
CANONICAL = ROOT / "data/supercloud_gpu_jobs_v1.jsonl"
PREPROCESS = ROOT / "data/supercloud_gpu_jobs_v1_preprocess.json"
LEGACY = Path(__file__).with_name("frozen_2x2.json")

POOL = 80
HOURS = 24.0
WARMUP_H = 12.0
N_WINDOWS = 48
SPLIT_COUNTS = {"train": 24, "validation": 12, "test": 12}
SELECTION_SEED = 20260912
DEADLINE_ALPHA = 1.0
DEADLINE_DEFAULT_S = 86400
BUILD_CONFIG = {
    "pool": POOL, "hours": HOURS, "warmup_h": WARMUP_H, "elastic_frac": 1.0,
    "par_frac": 0.7, "max_scale": 4.0, "elastic_seed": 0, "resize_points": 10,
    "deadline_alpha": DEADLINE_ALPHA, "deadline_default_s": DEADLINE_DEFAULT_S,
}
IMPLEMENTATION_FILES = [
    "pins/elastisim_bench.py", "pins/correction.py", "pins/freeze_core_2x2.py",
    "pins/run_core_2x2.py", "pins/sweep_2x2_tick.sh",
    "pins/analyse_core_2x2.py", "pins/verify_core_2x2.py", "pins/core_2x2_prereg.md",
    "pins/test_policy_selector.py", "pins/test_core_2x2.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_sha(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def audit_raw_trace() -> dict:
    """Count mutually exclusive preprocessing outcomes against the source CSV."""
    counts: Counter[str] = Counter()
    with SLURM_LOG.open() as f:
        for row in csv.DictReader(f):
            counts["source_rows"] += 1
            if row.get("state") != "3":
                counts["removed_not_completed"] += 1
                continue
            if GPU_TRES + "=" not in row.get("tres_alloc", ""):
                counts["removed_no_gpu_allocation"] += 1
                continue
            try:
                tres = dict(x.split("=", 1) for x in row["tres_alloc"].split(",") if "=" in x)
                submit = int(row["time_submit"])
                start = int(row["time_start"])
                end = int(row["time_end"])
                int(row["timelimit"])
                gpus = int(tres[GPU_TRES])
            except (KeyError, TypeError, ValueError):
                counts["removed_invalid_numeric_field"] += 1
                continue
            if submit <= 0 or start <= 0 or end <= start or gpus <= 0:
                counts["removed_invalid_time_or_gpu_count"] += 1
                continue
            counts["canonical_rows"] += 1
            if gpus > POOL:
                counts["capacity_ineligible_rows"] += 1
    for key in ("removed_not_completed", "removed_no_gpu_allocation",
                "removed_invalid_numeric_field", "removed_invalid_time_or_gpu_count",
                "canonical_rows", "capacity_ineligible_rows"):
        counts.setdefault(key, 0)
    assert counts["canonical_rows"] == len(_trace_rows())
    return dict(sorted(counts.items()))


def write_canonical() -> dict:
    audit = audit_raw_trace()
    if not CANONICAL.exists():
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        with CANONICAL.open("w") as f:
            for row in _trace_rows():
                f.write(json.dumps(row, sort_keys=True) + "\n")
    PREPROCESS.write_text(json.dumps({
        "version": 1,
        "source": str(SLURM_LOG),
        "source_sha256": sha256(SLURM_LOG),
        "canonical": str(CANONICAL),
        "canonical_sha256": sha256(CANONICAL),
        "rules": [
            "state must equal completed code 3",
            f"allocated TRES must contain GPU id {GPU_TRES}",
            "submit/start/end/timelimit/GPU fields must parse as integers",
            "submit and start must be positive, end > start, GPU count > 0",
            f"jobs requesting more than the fixed {POOL}-GPU pool remain in the canonical trace "
            "but are excluded from every experimental world",
        ],
        "counts": audit,
    }, indent=2) + "\n")
    return audit


def legacy_hours() -> list[int]:
    if not LEGACY.exists():
        return []
    windows = json.loads(LEGACY.read_text()).get("windows", [])
    return [int(w["window"].split("h")[0][1:]) * 24 + int(w["window"].split("h")[1])
            for w in windows]


def select_windows() -> list[tuple[int, int]]:
    rows = [r for r in _trace_rows() if r["gpus"] <= POOL]
    timestamps = [r["submit"] for r in rows]
    base = timestamps[0]
    span_h = int((timestamps[-1] - base) / 3600)
    occupied = legacy_hours()
    extent_h = int(HOURS + WARMUP_H)
    candidates = []
    for hour in range(int(WARMUP_H), span_h - int(HOURS)):
        start = base + hour * 3600
        lo = bisect.bisect_left(timestamps, start)
        hi = bisect.bisect_left(timestamps, start + HOURS * 3600)
        n_jobs = hi - lo
        if n_jobs >= 100 and all(abs(hour - old) >= extent_h for old in occupied):
            candidates.append((hour, n_jobs))
    random.Random(SELECTION_SEED).shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for candidate in candidates:
        if all(abs(candidate[0] - prior[0]) >= extent_h for prior in chosen):
            chosen.append(candidate)
            if len(chosen) == N_WINDOWS:
                break
    if len(chosen) != N_WINDOWS:
        raise RuntimeError(f"only {len(chosen)} eligible non-overlapping windows")
    return chosen


def world_sha(world: Path) -> str:
    return combined_sha(sorted((world / "in").glob("*")) + [world / "meta.json"])


def main() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {MANIFEST}")
    audit = write_canonical()
    chosen = select_windows()
    splits = (["train"] * SPLIT_COUNTS["train"] +
              ["validation"] * SPLIT_COUNTS["validation"] +
              ["test"] * SPLIT_COUNTS["test"])
    WORLDS.mkdir(parents=True, exist_ok=False)
    windows = []
    for (hour, _), split in zip(chosen, splits):
        day, offset = divmod(hour, 24)
        name = f"d{day}h{offset}"
        world = WORLDS / name
        meta = build(day=day, offset_h=offset, out=world, **BUILD_CONFIG)
        jobs = json.loads((world / "in/jobs.json").read_text())["jobs"]
        scored = [j for j in jobs if not j["attributes"]["_warmup"]]
        windows.append({
            "window": name, "start_hour": hour, "split": split,
            "n_jobs": meta["n_jobs"], "n_warmup": meta["n_warmup"],
            "offered_load": meta["offered_load"],
            "mean_gpu_demand": round(sum(j["attributes"]["req_nodes"] for j in scored) /
                                     len(scored), 3),
            "priority_fraction": round(sum(j["attributes"]["tier"] == "prod" for j in scored) /
                                       len(scored), 3),
            "arrival_rate_per_h": round(len(scored) / HOURS, 3),
            "world_sha256": world_sha(world),
        })
    impl_paths = [ROOT / p for p in IMPLEMENTATION_FILES]
    missing = [str(p) for p in impl_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"implementation files must exist before freezing: {missing}")
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    manifest = {
        "experiment_id": "core-2x2-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "seed": SELECTION_SEED, "rule": "workload-only random candidate order",
            "minimum_scored_jobs": 100, "number_of_windows": N_WINDOWS,
            "non_overlap_extent_h": HOURS + WARMUP_H,
            "legacy_exclusion": str(LEGACY.relative_to(ROOT)),
            "split_counts": SPLIT_COUNTS,
        },
        "processed_trace": {
            "path": str(CANONICAL.relative_to(ROOT)), "sha256": sha256(CANONICAL),
            "source_path": str(SLURM_LOG.relative_to(ROOT)), "source_sha256": sha256(SLURM_LOG),
            "audit_path": str(PREPROCESS.relative_to(ROOT)), "counts": audit,
        },
        "build": BUILD_CONFIG,
        "deadlines": {
            "formula": "D_i = arrival_i + alpha * W_i_req_est",
            "alpha": DEADLINE_ALPHA, "missing_walltime_estimate_s": DEADLINE_DEFAULT_S,
            "generated_once_in_jobs_json": True, "primary_metric": "deadline_viol_pct",
        },
        "selector": {
            "families": {
                "nm": ["fcfs", "least_laxity", "fairness", "tier_fcfs"],
                "mkt": ["auction_wait", "auction_deadline", "auction_fairness", "auction_priority"],
            },
            "sizing_actions": ["as_requested", "adaptive", "greedy", "rcon"],
            "rcon_threshold_s": 300.0, "selection_interval_s": 1800,
        },
        "inference": {
            "model": "qwen2.5:14b", "temperature": 0.1, "num_predict": 400,
            "paired_seeds": [17, 29, 43],
        },
        "execution": {
            "simulation_interval_s": 300, "evaluation_splits": ["validation", "test"],
            "resize_overhead": "not modelled; reported as 0 seconds",
        },
        "analysis": {
            "unit": "window after averaging repeated LLM seeds",
            "primary_metric": "deadline_viol_pct", "confidence": 0.95,
            "tests": "exact paired sign-flip randomization",
            "multiplicity": "Holm correction over the three preregistered factorial contrasts",
            "contrasts": ["market main effect", "multi-agent main effect", "interaction"],
            "fixed_baseline_selection": "lowest validation deadline_viol_pct; test reported once",
        },
        "implementation": {
            "files": IMPLEMENTATION_FILES, "sha256": combined_sha(impl_paths),
            "git_head_at_freeze": git_head,
        },
        "windows": windows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"froze {len(windows)} windows at {MANIFEST}")


if __name__ == "__main__":
    main()
