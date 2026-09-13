"""Fail-closed verification gate for the frozen core 2x2 experiment."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from pins import elastisim_bench as bench
from pins.freeze_core_2x2 import IMPLEMENTATION_FILES, ROOT, combined_sha, world_sha


def verify(manifest_path: Path, worlds: Path, check_results: bool = False,
           results: Path | None = None) -> dict:
    manifest_path = manifest_path.resolve()
    worlds = worlds.resolve()
    doc = json.loads(manifest_path.read_text())
    assert doc["experiment_id"] == "core-2x2-v1"
    assert doc["selector"]["families"] == bench.POLICY_FAMILY
    assert doc["selector"]["sizing_actions"] == list(bench.SELECTOR_MENU["sizing"])
    assert len({len(v) for v in bench.POLICY_FAMILY.values()}) == 1
    assert set(bench.POLICY_FAMILY["nm"]).isdisjoint(bench.POLICY_FAMILY["mkt"])

    impl_paths = [ROOT / path for path in IMPLEMENTATION_FILES]
    actual_impl = combined_sha(impl_paths)
    assert actual_impl == doc["implementation"]["sha256"], \
        "implementation differs from the frozen manifest; re-freeze before running"
    trace = ROOT / doc["processed_trace"]["path"]
    source = ROOT / doc["processed_trace"]["source_path"]
    audit_path = ROOT / doc["processed_trace"]["audit_path"]
    import hashlib
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == doc["processed_trace"]["sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == doc["processed_trace"]["source_sha256"]
    audit = json.loads(audit_path.read_text())
    assert audit["counts"] == doc["processed_trace"]["counts"]

    windows = doc["windows"]
    assert len(windows) == doc["selection"]["number_of_windows"]
    assert len({w["window"] for w in windows}) == len(windows)
    assert Counter(w["split"] for w in windows) == Counter(doc["selection"]["split_counts"])
    extent = doc["selection"]["non_overlap_extent_h"]
    for i, left in enumerate(windows):
        for right in windows[i + 1:]:
            assert abs(left["start_hour"] - right["start_hour"]) >= extent, \
                f"overlapping windows: {left['window']} and {right['window']}"
    legacy_path = ROOT / doc["selection"]["legacy_exclusion"]
    legacy = json.loads(legacy_path.read_text()).get("windows", [])
    legacy_hours = [int(x["window"].split("h")[0][1:]) * 24 +
                    int(x["window"].split("h")[1]) for x in legacy]
    assert all(abs(w["start_hour"] - old) >= extent for w in windows for old in legacy_hours)

    build = doc["build"]
    checked_jobs = 0
    for item in windows:
        world = worlds / item["window"]
        assert world_sha(world) == item["world_sha256"], f"world changed: {item['window']}"
        meta = json.loads((world / "meta.json").read_text())
        for key in ("pool", "hours", "warmup_h", "elastic_frac", "par_frac", "max_scale",
                    "elastic_seed", "resize_points", "deadline_alpha", "deadline_default_s"):
            assert meta[key] == build[key], (item["window"], key, meta[key], build[key])
        jobs = json.loads((world / "in/jobs.json").read_text())["jobs"]
        assert all(job["type"] == "malleable" for job in jobs)
        for job in jobs:
            attrs = job["attributes"]
            estimate = (int(attrs["req_min"]) * 60) or build["deadline_default_s"]
            expected_deadline = job["submit_time"] + build["deadline_alpha"] * estimate
            assert attrs["deadline_estimate_s"] == estimate
            assert math.isclose(attrs["deadline_s"], expected_deadline, abs_tol=1e-9)
            if job["type"] == "malleable":
                lo, hi = job["num_nodes_min"], job["num_nodes_max"]
                assert 1 <= lo <= hi <= build["pool"]
                assert lo <= int(attrs["req_nodes"]) <= hi
                args = job["arguments"]
                n0 = int(attrs["req_nodes"])
                anchored = (float(args["a_par"]) * n0 + float(args["a_ser"])) / n0 / bench.FLOPS_PER_GPU
                assert math.isclose(anchored, float(attrs["_true_dur"]), rel_tol=1e-12)
                runtimes = [(float(args["a_par"]) * n + float(args["a_ser"])) /
                            n / bench.FLOPS_PER_GPU for n in range(lo, hi + 1)]
                assert all(x >= y for x, y in zip(runtimes, runtimes[1:])), \
                    f"non-monotone speedup for {attrs['jid']}"
                speedups = [runtimes[0] / x for x in runtimes]
                gains = [y - x for x, y in zip(speedups, speedups[1:])]
                assert all(x + 1e-12 >= y for x, y in zip(gains, gains[1:])), \
                    f"non-diminishing speedup for {attrs['jid']}"
            else:
                assert 1 <= job["num_nodes"] <= build["pool"]
            checked_jobs += 1

    if check_results:
        if results is None:
            raise ValueError("results path required with check_results")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        # Rows may predate a disclosed amendment (see core_2x2_prereg.md). Every accepted hash is
        # listed in the manifest, so a superseded one still verifies while an UNlisted one -- code
        # that was never frozen -- still fails. Re-stamping old rows would hide the amendment.
        accepted = set(doc["implementation"].get("accepted_sha256", [])) | {actual_impl}
        manifest_hashes = set(doc["implementation"].get("accepted_manifest_sha256", [])) | {manifest_hash}
        for line in results.read_text().splitlines():
            row = json.loads(line)
            assert row["experiment_id"] == doc["experiment_id"]
            assert row["manifest_sha256"] in manifest_hashes
            assert row["implementation_sha256"] in accepted, \
                f"row implementation {row['implementation_sha256'][:12]} was never frozen"
            assert row["processed_trace_sha256"] == doc["processed_trace"]["sha256"]
    return {"windows": len(windows), "jobs": checked_jobs, "implementation_sha256": actual_impl}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).with_name("core_2x2_manifest.json"))
    ap.add_argument("--worlds", type=Path, default=ROOT / "runs/core_2x2_worlds")
    ap.add_argument("--results", type=Path)
    args = ap.parse_args()
    out = verify(args.manifest, args.worlds, args.results is not None, args.results)
    print(json.dumps({"ok": True, **out}, indent=2))


if __name__ == "__main__":
    main()
