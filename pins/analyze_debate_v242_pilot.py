#!/usr/bin/env python3
"""Analyze the completed v2.4.2 pilot, publish it, and conditionally launch the sweep."""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path("/import/gp-home.ciero/kimseng/Research")
ROWS = ROOT / "runs/pilot_debate_v242_two/rows.jsonl"
RESULT_JSON = ROOT / "pins/debate_v242_pilot_results.json"
RESULT_MD = ROOT / "pins/debate_v242_pilot_analysis.md"
WINDOWS = ("d111h22", "d223h11")
BASELINES = {
    "d111h22": {
        "version": "v2.4.1", "commit": "956d289",
        "deadline_viol_pct": 20.7, "mean_wait_s": 73764,
        "useful_util_win": 0.651,
    },
    "d223h11": {
        "version": "frozen v2.4", "commit": "b6d0d6c",
        "deadline_viol_pct": 25.1, "mean_wait_s": 36445,
        "useful_util_win": 0.853,
    },
}


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def trace_summary(window: str) -> dict:
    path = ROOT / f"runs/core_2x2_worlds/{window}/out/pilot_v242_s17_policy_log.json"
    if not path.exists():
        return {"available": False, "path": str(path.relative_to(ROOT))}
    epochs = json.loads(path.read_text())
    rotation_actions: Counter[str] = Counter()
    referee_actions: Counter[str] = Counter()
    ordering_counts: Counter[str] = Counter()
    openings = {"demand": Counter(), "supply": Counter()}
    trial_requests = {"demand": 0, "supply": 0}
    abstentions = {"demand": 0, "supply": 0}
    outcome_events: list[dict] = []
    seen_outcomes: set[tuple] = set()
    for epoch in epochs:
        ordering_counts[str(epoch.get("ordering"))] += 1
        ratification = epoch.get("ratification") or {}
        rotation = ratification.get("rotation") or {}
        if rotation.get("action"):
            rotation_actions[str(rotation["action"])] += 1
        if ratification.get("referee_action"):
            referee_actions[str(ratification["referee_action"])] += 1
        for role in ("demand", "supply"):
            opening = (ratification.get("openings") or {}).get(role) or {}
            candidate = opening.get("ordering")
            if candidate:
                openings[role][str(candidate)] += 1
                abstentions[role] += int(candidate == "abstain")
            trial_requests[role] += int(opening.get("request_trial") is True)
        for event in (ratification.get("settled_before_debate"), rotation):
            if not event or event.get("action") not in {
                    "trial_accept", "trial_rollback", "trial_probation"}:
                continue
            trial = event.get("trial") or {}
            key = (event.get("action"), trial.get("start_t"), event.get("reason"))
            if key in seen_outcomes:
                continue
            seen_outcomes.add(key)
            outcome_events.append({
                "action": event.get("action"),
                "reason": event.get("reason"),
                "role": trial.get("role"),
                "transition": f"{trial.get('incumbent')}->{trial.get('ordering')}",
                "phase": trial.get("phase"),
                "checks": event.get("checks"),
            })
    return {
        "available": True,
        "path": str(path.relative_to(ROOT)),
        "epochs": len(epochs),
        "ordering_counts": dict(ordering_counts),
        "rotation_actions": dict(rotation_actions),
        "referee_actions": dict(referee_actions),
        "opening_orderings": {role: dict(counts) for role, counts in openings.items()},
        "trial_requests": trial_requests,
        "abstentions": abstentions,
        "invalid_holds": sum(
            int(bool((epoch.get("ratification") or {}).get("invalid_hold_used")))
            for epoch in epochs),
        "outcome_events": outcome_events,
    }


def fmt_delta(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    by_window = {row["window"]: row for row in rows}
    missing = [window for window in WINDOWS if window not in by_window]
    if missing:
        raise SystemExit(f"pilot incomplete; missing {', '.join(missing)}")
    pilot = [by_window[window] for window in WINDOWS]
    baseline = [BASELINES[window] for window in WINDOWS]
    macro = {
        "baseline_deadline_viol_pct": mean(baseline, "deadline_viol_pct"),
        "pilot_deadline_viol_pct": mean(pilot, "deadline_viol_pct"),
        "baseline_mean_wait_s": mean(baseline, "mean_wait_s"),
        "pilot_mean_wait_s": mean(pilot, "mean_wait_s"),
        "baseline_useful_util_win": mean(baseline, "useful_util_win"),
        "pilot_useful_util_win": mean(pilot, "useful_util_win"),
    }
    completed_trials = sum(
        int(row["debate_v24_trial_accepts"]) + int(row["debate_v24_trial_rollbacks"])
        for row in pilot)
    checks = {
        "both_complete_zero_llm_errors": sum(int(row["llm_errors"]) for row in pilot) == 0,
        "revised_evaluator_exercised": completed_trials >= 1,
        "macro_deadline_improves_by_1pp": (
            macro["pilot_deadline_viol_pct"]
            <= macro["baseline_deadline_viol_pct"] - 1.0),
        "no_window_deadline_regresses_over_2pp": all(
            float(by_window[window]["deadline_viol_pct"])
            <= BASELINES[window]["deadline_viol_pct"] + 2.0 for window in WINDOWS),
        "macro_wait_within_5pct": (
            macro["pilot_mean_wait_s"] <= macro["baseline_mean_wait_s"] * 1.05),
        "macro_util_within_0_02": (
            macro["pilot_useful_util_win"]
            >= macro["baseline_useful_util_win"] - 0.02),
    }
    positive = all(checks.values())
    traces = {window: trace_summary(window) for window in WINDOWS}
    result = {
        "protocol_version": "2.4.2",
        "implementation_commit": "2d491ac",
        "gate_commit": "1cf8237",
        "tag": "pilot_v242_s17",
        "llm_seed": 17,
        "baselines": BASELINES,
        "windows": pilot,
        "macro": macro,
        "completed_trials": completed_trials,
        "gate_checks": checks,
        "positive": positive,
        "trace_analysis": traces,
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    table_rows = []
    for window in WINDOWS:
        row, base = by_window[window], BASELINES[window]
        table_rows.append(
            f"| {window} | {base['version']} | {base['deadline_viol_pct']:.1f}% | "
            f"{row['deadline_viol_pct']:.1f}% | "
            f"{fmt_delta(row['deadline_viol_pct'] - base['deadline_viol_pct'])} pp | "
            f"{base['mean_wait_s']:,} | {row['mean_wait_s']:,} | "
            f"{fmt_delta((row['mean_wait_s'] / base['mean_wait_s'] - 1) * 100)}% | "
            f"{base['useful_util_win']:.3f} | {row['useful_util_win']:.3f} | "
            f"{fmt_delta(row['useful_util_win'] - base['useful_util_win'], 3)} |")
    check_rows = [
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    ]
    trace_lines = []
    for window in WINDOWS:
        row, trace = by_window[window], traces[window]
        if not trace["available"]:
            trace_lines.append(f"- `{window}`: policy trace unavailable at `{trace['path']}`.")
            continue
        trials = (
            f"{row['debate_v24_trials']} started, "
            f"{row['debate_v24_trial_probations']} probation, "
            f"{row['debate_v24_trial_accepts']} accepted, "
            f"{row['debate_v24_trial_rollbacks']} rolled back")
        trace_lines.append(
            f"- `{window}`: {trials}; {trace['invalid_holds']} invalid debates held the incumbent. "
            f"Executed ordering epochs: `{json.dumps(trace['ordering_counts'], sort_keys=True)}`. "
            f"Rotation actions: `{json.dumps(trace['rotation_actions'], sort_keys=True)}`.")
        for event in trace["outcome_events"]:
            checks_text = json.dumps(event.get("checks"), sort_keys=True)
            trace_lines.append(
                f"  - {event['action']} for {event['transition']} ({event['role']}): "
                f"reason `{event.get('reason')}`; checks `{checks_text}`.")
    interpretation = (
        "The gate passed, so the revision improved the primary deadline metric without breaching "
        "the wait or utilization guardrails. The trace counters show whether that improvement "
        "coincided with accepted rotation, protective rollback, or incumbent holds. This is a "
        "mechanistic interpretation, not a causal isolation of rotation from deterministic sizing."
        if positive else
        "The gate failed at least one preregistered condition. Even if an individual metric "
        "improved, the evidence is not strong enough to justify a 24-window sweep under the frozen "
        "rule. Trace counters identify whether the failure occurred despite rotation, rollback, or "
        "mostly incumbent holds; they do not by themselves establish causal attribution."
    )
    markdown = f"""# Debate v2.4.2 two-window pilot analysis

Decision: **{'POSITIVE — launch the 24-window sweep' if positive else 'NOT POSITIVE — do not launch the sweep'}**.

The decision applies the gate frozen in `debate_v242_pilot_gate.md` before results were available.

## Matched-window results

| window | predecessor | base deadline | pilot deadline | delta | base wait (s) | pilot wait (s) | delta | base util | pilot util | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

Macro deadline violation changed from {macro['baseline_deadline_viol_pct']:.2f}% to
{macro['pilot_deadline_viol_pct']:.2f}% ({fmt_delta(macro['pilot_deadline_viol_pct'] - macro['baseline_deadline_viol_pct'], 2)} pp).
Macro mean wait changed from {macro['baseline_mean_wait_s']:.1f} s to
{macro['pilot_mean_wait_s']:.1f} s ({fmt_delta((macro['pilot_mean_wait_s'] / macro['baseline_mean_wait_s'] - 1) * 100, 2)}%).
Macro useful utilization changed from {macro['baseline_useful_util_win']:.3f} to
{macro['pilot_useful_util_win']:.3f} ({fmt_delta(macro['pilot_useful_util_win'] - macro['baseline_useful_util_win'], 3)}).

## Frozen gate

| condition | result |
| --- | --- |
{chr(10).join(check_rows)}

## Trace evidence

{chr(10).join(trace_lines)}

## Interpretation

{interpretation}
"""
    RESULT_MD.write_text(markdown)

    relative = [str(path.relative_to(ROOT)) for path in (RESULT_JSON, RESULT_MD)]
    subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *relative], cwd=ROOT)
    if changed.returncode:
        subprocess.run([
            "git", "commit", "--only", "-m", "results: analyze v2.4.2 pilot",
            "--", *relative,
        ], cwd=ROOT, check=True)
    git_env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
    }
    subprocess.run(
        ["git", "push", "origin", "elastisim"], cwd=ROOT, check=True,
        env=git_env)

    current = subprocess.run(
        ["crontab", "-l"], text=True, capture_output=True, check=False).stdout
    kept = [line for line in current.splitlines()
            if "debate_v242_pilot_tick" not in line
            and "debate_v242_sweep_tick" not in line]
    if positive:
        kept.append(
            "* * * * * /import/gp-home.ciero/kimseng/Research/"
            "pins/debate_v242_sweep_tick.sh")
    subprocess.run(
        ["crontab", "-"], input=("\n".join(kept) + ("\n" if kept else "")),
        text=True, check=True)
    print(
        f"pilot analysis published; positive={positive}; "
        f"24-window sweep {'scheduled' if positive else 'not scheduled'}")


if __name__ == "__main__":
    main()
