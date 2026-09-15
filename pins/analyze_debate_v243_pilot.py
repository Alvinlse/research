#!/usr/bin/env python3
"""Analyze v2.4.3, publish after analysis, and conditionally launch all 24 windows."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pins.analyze_debate_v242_pilot import fmt_delta, mean, trace_summary


ROOT = Path("/import/gp-home.ciero/kimseng/Research")
ROWS = ROOT / "runs/pilot_debate_v243_two/rows.jsonl"
RESULT_JSON = ROOT / "pins/debate_v243_pilot_results.json"
RESULT_MD = ROOT / "pins/debate_v243_pilot_analysis.md"
WINDOWS = ("d111h22", "d223h11")
BASELINES = {
    "d111h22": {
        "version": "v2.4.2", "commit": "df531df",
        "deadline_viol_pct": 8.9, "mean_wait_s": 57764,
        "useful_util_win": 0.721,
    },
    "d223h11": {
        "version": "v2.4.2", "commit": "df531df",
        "deadline_viol_pct": 28.0, "mean_wait_s": 37929,
        "useful_util_win": 0.850,
    },
}
TAG = "pilot_v243_s17"


def structural_trace_check(window: str) -> dict:
    """Verify phase-correct feedback and that no blocked transition starts."""
    path = ROOT / f"runs/core_2x2_worlds/{window}/out/{TAG}_policy_log.json"
    epochs = json.loads(path.read_text())
    phase_feedback_mismatches = []
    backoff_start_violations = []
    blocked_requests = 0
    for index, epoch in enumerate(epochs):
        ratification = epoch.get("ratification") or {}
        state = ratification.get("decision_state") or {}
        learning = (state.get("trial_learning") or {}).get("by_transition") or {}
        blocked_requests += len(ratification.get("blocked_trial_requests") or {})
        settled = ratification.get("settled_before_debate") or {}
        if settled.get("action") in {"trial_accept", "trial_rollback"}:
            trial = settled.get("trial") or {}
            key = f"{trial.get('incumbent')}->{trial.get('ordering')}"
            learned = learning.get(key) or {}
            checks = settled.get("checks") or {}
            expected_verdict = (
                "beneficial_challenger" if settled["action"] == "trial_accept"
                else "harmful_challenger")
            if (learned.get("last_phase_queue_delta") != checks.get("queue_delta")
                    or learned.get("last_phase_deadline_pressure_delta")
                    != checks.get("deadline_pressure_delta")
                    or learned.get("last_verdict") != expected_verdict):
                phase_feedback_mismatches.append({
                    "epoch": index, "transition": key,
                    "expected_queue_delta": checks.get("queue_delta"),
                    "learned_queue_delta": learned.get("last_phase_queue_delta"),
                    "expected_pressure_delta": checks.get("deadline_pressure_delta"),
                    "learned_pressure_delta": learned.get(
                        "last_phase_deadline_pressure_delta"),
                    "expected_verdict": expected_verdict,
                    "learned_verdict": learned.get("last_verdict"),
                })
        rotation = ratification.get("rotation") or {}
        if rotation.get("action") == "trial_start":
            trial = rotation.get("trial") or {}
            key = f"{trial.get('incumbent')}->{trial.get('ordering')}"
            if (learning.get(key) or {}).get("blocked_now"):
                backoff_start_violations.append({"epoch": index, "transition": key})
    return {
        "phase_feedback_mismatches": phase_feedback_mismatches,
        "backoff_start_violations": backoff_start_violations,
        "blocked_requests": blocked_requests,
        "passed": not phase_feedback_mismatches and not backoff_start_violations,
    }


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
    structural = {window: structural_trace_check(window) for window in WINDOWS}
    checks = {
        "both_complete_zero_llm_errors": sum(int(row["llm_errors"]) for row in pilot) == 0,
        "revised_evaluator_exercised": completed_trials >= 1,
        "d223_deadline_at_most_27_1": by_window["d223h11"]["deadline_viol_pct"] <= 27.1,
        "d111_deadline_at_most_10_9": by_window["d111h22"]["deadline_viol_pct"] <= 10.9,
        "macro_deadline_not_worse": (
            macro["pilot_deadline_viol_pct"] <= macro["baseline_deadline_viol_pct"]),
        "macro_wait_within_5pct": (
            macro["pilot_mean_wait_s"] <= macro["baseline_mean_wait_s"] * 1.05),
        "macro_util_within_0_02": (
            macro["pilot_useful_util_win"] >= macro["baseline_useful_util_win"] - 0.02),
        "runtime_structural_integrity": all(item["passed"] for item in structural.values()),
    }
    positive = all(checks.values())
    sweep_override = True
    traces = {window: trace_summary(window, TAG) for window in WINDOWS}
    result = {
        "protocol_version": "2.4.3",
        "tag": TAG,
        "llm_seed": 17,
        "baselines": BASELINES,
        "windows": pilot,
        "macro": macro,
        "completed_trials": completed_trials,
        "gate_checks": checks,
        "positive": positive,
        "sweep_override": sweep_override,
        "sweep_launch": "frozen_gate" if positive else "explicit_user_override",
        "structural_trace_checks": structural,
        "trace_analysis": traces,
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    table_rows = []
    for window in WINDOWS:
        row, base = by_window[window], BASELINES[window]
        table_rows.append(
            f"| {window} | {base['deadline_viol_pct']:.1f}% | {row['deadline_viol_pct']:.1f}% | "
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
        row, trace, structure = by_window[window], traces[window], structural[window]
        trace_lines.append(
            f"- `{window}`: {row['debate_v24_trials']} trials, "
            f"{row['debate_v24_trial_accepts']} accepts, "
            f"{row['debate_v24_trial_rollbacks']} rollbacks, "
            f"{row['debate_v24_transition_blocked_requests']} blocked requests; "
            f"phase-feedback mismatches={len(structure['phase_feedback_mismatches'])}, "
            f"backoff start violations={len(structure['backoff_start_violations'])}. "
            f"Executed orderings: `{json.dumps(trace.get('ordering_counts', {}), sort_keys=True)}`.")
    interpretation = (
        "All frozen conditions passed. The phase-level evidence is internally consistent and the "
        "backoff admitted no blocked trial starts, so the 24-window sweep is launched. Aggregate "
        "movement remains observational and does not isolate rotation from sizing or emergencies."
        if positive else
        "At least one frozen condition failed. The 24-window sweep nevertheless continues under "
        "the explicit presentation-deadline override; the failed gate remains reported unchanged."
    )
    RESULT_MD.write_text(f"""# Debate v2.4.3 two-window pilot analysis

Decision: **{'POSITIVE — sweep justified by frozen gate' if positive else 'GATE NOT PASSED — sweep continues by explicit user override'}**.

## Matched-window results

| window | base deadline | pilot deadline | delta | base wait (s) | pilot wait (s) | delta | base util | pilot util | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table_rows)}

Macro deadline: {macro['baseline_deadline_viol_pct']:.2f}% to {macro['pilot_deadline_viol_pct']:.2f}%.
Macro wait: {macro['baseline_mean_wait_s']:.1f} s to {macro['pilot_mean_wait_s']:.1f} s.
Macro utilization: {macro['baseline_useful_util_win']:.4f} to {macro['pilot_useful_util_win']:.4f}.

## Frozen gate

| condition | result |
| --- | --- |
{chr(10).join(check_rows)}

## Trace evidence

{chr(10).join(trace_lines)}

## Interpretation

{interpretation}
""")

    relative = [str(path.relative_to(ROOT)) for path in (RESULT_JSON, RESULT_MD)]
    subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *relative], cwd=ROOT)
    if changed.returncode:
        subprocess.run([
            "git", "commit", "--only", "-m", "results: analyze v2.4.3 pilot",
            "--", *relative,
        ], cwd=ROOT, check=True)
    git_env = {
        **os.environ, "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
    }
    subprocess.run(
        ["git", "push", "origin", "elastisim"], cwd=ROOT, check=True, env=git_env)

    current = subprocess.run(
        ["crontab", "-l"], text=True, capture_output=True, check=False).stdout
    kept = [line for line in current.splitlines()
            if "debate_v243_pilot_tick" not in line
            and "debate_v243_sweep_tick" not in line]
    if positive or sweep_override:
        kept.append(
            "* * * * * /import/gp-home.ciero/kimseng/Research/"
            "pins/debate_v243_sweep_tick.sh")
    subprocess.run(
        ["crontab", "-"], input=("\n".join(kept) + ("\n" if kept else "")),
        text=True, check=True)
    print(
        f"v2.4.3 analysis published; positive={positive}; "
        f"24-window sweep scheduled; override={sweep_override and not positive}")


if __name__ == "__main__":
    main()
