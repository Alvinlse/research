"""Create reproducible synthetic text exceptions for an ElastiSim Supercloud world.

The source trace has numerical accounting data but no free-text operator notes.  This tool adds a
counterfactual semantic layer without modifying the original jobs.json.  ``text_exceptions.json``
is scheduler-visible; ``text_exception_labels.json`` is an evaluation-only answer key.

Example:
    .venv/bin/python -m pins.synth_supercloud_exceptions \
        --world runs/neg_v2_malleable_99jobs --rate 0.30 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


TEMPLATES = {
    "protect": [
        ("external_stream", "Operator note: this run is consuming a live instrument stream that "
         "cannot be replayed. Keep its current GPU allocation until the stream closes; shrinking "
         "mid-run would invalidate the observation."),
        ("non_resizable_phase", "Owner note: the job is inside a non-resizable collective phase. "
         "Do not shrink its current allocation before the next phase boundary or all progress in "
         "this phase will be lost."),
        ("expiring_lease", "Operations note: this run holds a time-limited external data lease. "
         "Protect its current allocation so it finishes before the lease expires; the input cannot "
         "be reacquired today."),
    ],
    "shrink": [
        ("checkpoint_ready", "Owner note: a verified checkpoint was written at this boundary and "
         "the remaining phase is elastic. It is safe to shrink this job to its minimum allocation "
         "if another job is waiting."),
        ("best_effort_release", "Owner note: this is exploratory best-effort work. Please release "
         "its excess GPUs under queue pressure; continuing at the minimum allocation is acceptable."),
        ("input_stall", "Runtime note: the next phase will mostly wait for staged input. Extra GPUs "
         "provide no benefit right now, so shrink to the minimum while work is queued."),
    ],
    "default": [
        ("status_only", "Status note: training is progressing normally and periodic checkpoints are "
         "enabled. No special scheduling instruction has been issued."),
        ("accounting_only", "Accounting note: this job belongs to the standard research allocation. "
         "Apply the normal cluster policy."),
        ("desire_only", "Owner note: faster completion would be convenient, but there is no external "
         "deadline, dependency, or operational constraint requiring an exception."),
    ],
}


def _stable_rank(seed: int, jid: str) -> bytes:
    return hashlib.sha256(f"{seed}:{jid}".encode()).digest()


def generate(world: Path, rate: float, seed: int, targets: list[str] | None = None) -> tuple[dict, dict]:
    if not 0 <= rate <= 1:
        raise ValueError("--rate must be between 0 and 1")
    jobs_path = world / "in/jobs.json"
    jobs = json.loads(jobs_path.read_text())["jobs"]
    eligible = [j for j in jobs if j.get("type") == "malleable"]
    eligible.sort(key=lambda j: _stable_rank(seed, str(j["attributes"]["jid"])))
    if targets:
        valid = {str(j["attributes"]["jid"]) for j in eligible}
        unknown = set(targets) - valid
        if unknown:
            raise ValueError(f"target jobs are not malleable members of this world: {sorted(unknown)}")
        classes = ("protect", "shrink", "default")
        assigned = {jid: classes[(i + seed) % len(classes)] for i, jid in enumerate(targets)}
        n_exception = sum(a != "default" for a in assigned.values())
    else:
        n_exception = round(len(eligible) * rate)
        # Half of injected exceptions require protection and half explicitly permit shrinking.
        actions = (["protect", "shrink"] * ((n_exception + 1) // 2))[:n_exception]
        random.Random(seed).shuffle(actions)
        assigned = {str(j["attributes"]["jid"]): a for j, a in zip(eligible, actions)}

    visible, labels = {}, {}
    for j in sorted(eligible, key=lambda x: (x.get("submit_time", 0), str(x["attributes"]["jid"]))):
        jid = str(j["attributes"]["jid"])
        action = assigned.get(jid, "default")
        choices = TEMPLATES[action]
        idx = int.from_bytes(_stable_rank(seed + 1, jid)[:4], "big") % len(choices)
        kind, note = choices[idx]
        visible[jid] = {
            "job_id": jid,
            "note": note,
            "source": "synthetic_counterfactual",
        }
        labels[jid] = {
            "expected_action": action,
            "exception": action != "default",
            "reason_code": kind,
        }

    manifest = {
        "schema_version": 1,
        "dataset": "MIT Supercloud trace-derived ElastiSim world",
        "disclaimer": "All notes are synthetic counterfactuals; none came from MIT Supercloud users.",
        "seed": seed,
        "requested_exception_rate": rate,
        "targeted_jobs": targets or [],
        "eligible_malleable_jobs": len(eligible),
        "counts": {a: sum(x["expected_action"] == a for x in labels.values())
                   for a in ("protect", "shrink", "default")},
        "notes": list(visible.values()),
    }
    answer_key = {
        "schema_version": 1,
        "evaluation_only": True,
        "seed": seed,
        "labels": labels,
    }
    return manifest, answer_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=0.30,
                    help="fraction of malleable jobs receiving a real semantic exception")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", default="",
                    help="comma-separated trace job IDs; rotate protect/shrink/default over them")
    ap.add_argument("--out", type=Path,
                    help="visible notes file (default: WORLD/in/text_exceptions.json)")
    ap.add_argument("--labels-out", type=Path,
                    help="answer key (default: WORLD/in/text_exception_labels.json)")
    args = ap.parse_args()
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    visible, labels = generate(args.world, args.rate, args.seed, targets or None)
    out = args.out or args.world / "in/text_exceptions.json"
    labels_out = args.labels_out or args.world / "in/text_exception_labels.json"
    out.write_text(json.dumps(visible, indent=2) + "\n")
    labels_out.write_text(json.dumps(labels, indent=2) + "\n")
    print(json.dumps({"notes": str(out), "labels": str(labels_out), **visible["counts"]}))


if __name__ == "__main__":
    main()
