#!/usr/bin/env python3
"""Publish a presentation-ready analysis after all 24 v2.4.3 training windows finish."""
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path("/import/gp-home.ciero/kimseng/Research")
ROWS = ROOT / "runs/sweep_train_debate_v243/rows.jsonl"
BASELINE_ROWS = ROOT / "runs/sweep_train_debate/rows.jsonl"
RESULT_JSON = ROOT / "pins/debate_v243_24_window_results.json"
RESULT_MD = ROOT / "pins/debate_v243_24_window_analysis.md"
METRICS = ("deadline_viol_pct", "mean_wait_s", "p90_wait_s", "useful_util_win")


def macro(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def weighted(rows: list[dict], key: str) -> float:
    jobs = sum(int(row["n_jobs"]) for row in rows)
    return sum(float(row[key]) * int(row["n_jobs"]) for row in rows) / jobs


def paired(candidate: dict, baseline: dict, key: str, higher_better: bool = False) -> str:
    left, right = float(candidate[key]), float(baseline[key])
    if left == right:
        return "tie"
    improved = left > right if higher_better else left < right
    return "win" if improved else "loss"


def main() -> None:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    if len(rows) != 24 or len({row["window"] for row in rows}) != 24:
        raise SystemExit("24 unique v2.4.3 rows are required before sweep analysis")
    old_rows = [json.loads(line) for line in BASELINE_ROWS.read_text().splitlines()
                if line.strip()]
    baselines = {
        config: {row["window"]: row for row in old_rows if row["config"] == config}
        for config in ("single", "debate")
    }
    windows = {row["window"] for row in rows}
    if any(set(items) != windows for items in baselines.values()):
        raise SystemExit("paired single/debate baselines do not match the 24 sweep windows")

    summaries = {}
    comparisons = {}
    for name, items in (("v2.4.3", rows),
                        ("single", list(baselines["single"].values())),
                        ("debate_v1", list(baselines["debate"].values()))):
        summaries[name] = {
            "macro": {key: round(macro(items, key), 4) for key in METRICS},
            "job_weighted": {key: round(weighted(items, key), 4) for key in METRICS},
        }
    for config in ("single", "debate"):
        counts = {key: Counter() for key in METRICS}
        deltas = {key: [] for key in METRICS}
        for row in rows:
            base = baselines[config][row["window"]]
            for key in METRICS:
                outcome = paired(row, base, key, higher_better=key == "useful_util_win")
                counts[key][outcome] += 1
                deltas[key].append(float(row[key]) - float(base[key]))
        comparisons[config] = {
            key: {
                "wins": counts[key]["win"], "ties": counts[key]["tie"],
                "losses": counts[key]["loss"],
                "mean_candidate_minus_baseline": round(sum(deltas[key]) / 24, 4),
            }
            for key in METRICS
        }
    per_window = []
    for row in sorted(rows, key=lambda item: -item["offered_load"]):
        single, debate = baselines["single"][row["window"]], baselines["debate"][row["window"]]
        per_window.append({
            "window": row["window"], "offered_load": row["offered_load"],
            "source_tag": row["source_tag"],
            "v243": {key: row[key] for key in METRICS},
            "single": {key: single[key] for key in METRICS},
            "debate_v1": {key: debate[key] for key in METRICS},
            "deadline_vs_single": paired(row, single, "deadline_viol_pct"),
            "deadline_vs_debate_v1": paired(row, debate, "deadline_viol_pct"),
        })
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "llm_calls", "llm_errors", "debate_v24_trials",
            "debate_v24_trial_probations", "debate_v24_trial_accepts",
            "debate_v24_trial_rollbacks", "debate_v24_invalid_holds",
            "debate_v24_transition_blocked_requests",
            "debate_v24_transition_backoff_holds")
    }
    result = {
        "protocol_version": "2.4.3", "llm_seed": 17,
        "scope": "24 frozen training/development windows",
        "sweep_authorization": "explicit user override of the two-window gate",
        "source_counts": dict(Counter(row["source_tag"] for row in rows)),
        "summaries": summaries, "paired_comparisons": comparisons,
        "totals": totals, "per_window": per_window,
    }
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary_lines = []
    for name in ("v2.4.3", "single", "debate_v1"):
        item = summaries[name]["macro"]
        summary_lines.append(
            f"| {name} | {item['deadline_viol_pct']:.2f}% | "
            f"{item['mean_wait_s']:.0f} | {item['p90_wait_s']:.0f} | "
            f"{item['useful_util_win']:.3f} |")
    comparison_lines = []
    for name in ("single", "debate"):
        item = comparisons[name]["deadline_viol_pct"]
        comparison_lines.append(
            f"| {name} | {item['wins']} | {item['ties']} | {item['losses']} | "
            f"{item['mean_candidate_minus_baseline']:+.2f} pp |")
    window_lines = []
    for item in per_window:
        window_lines.append(
            f"| {item['window']} | {item['offered_load']:.2f} | "
            f"{item['v243']['deadline_viol_pct']:.1f}% | "
            f"{item['single']['deadline_viol_pct']:.1f}% | "
            f"{item['debate_v1']['deadline_viol_pct']:.1f}% | "
            f"{item['v243']['useful_util_win']:.3f} | {item['source_tag']} |")
    RESULT_MD.write_text(f"""# Debate v2.4.3 — 24-window sweep analysis

This is an exploratory repair sweep over the same 24 training/development windows used to revise
the protocol. It is appropriate for the presentation as development evidence, not as an unbiased
generalization result. The sweep ran by explicit user override regardless of the two-window gate.

## Macro results

| method | deadline violations | mean wait (s) | p90 wait (s) | useful utilization |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(summary_lines)}

## Paired deadline result

| baseline | wins | ties | losses | mean v2.4.3 minus baseline |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_lines)}

## Protocol behavior

- LLM calls: {totals['llm_calls']}; LLM errors: {totals['llm_errors']}.
- Trials: {totals['debate_v24_trials']} started, {totals['debate_v24_trial_probations']} reached probation,
  {totals['debate_v24_trial_accepts']} accepted, and {totals['debate_v24_trial_rollbacks']} rolled back.
- Transition-backoff blocks: {totals['debate_v24_transition_blocked_requests']} advocate requests;
  defense-layer holds: {totals['debate_v24_transition_backoff_holds']}.
- Invalid debates held the incumbent {totals['debate_v24_invalid_holds']} times without fixed fallback.

## Per-window deadline results

| window | load | v2.4.3 | single | debate v1 | v2.4.3 util | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(window_lines)}
""")

    relative = [str(path.relative_to(ROOT)) for path in (RESULT_JSON, RESULT_MD)]
    subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *relative], cwd=ROOT)
    if changed.returncode:
        subprocess.run([
            "git", "commit", "--only", "-m", "results: analyze v2.4.3 24-window sweep",
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
            if "debate_v243_sweep_tick" not in line
            and "debate_v243_pilot_tick" not in line]
    subprocess.run(
        ["crontab", "-"], input=("\n".join(kept) + ("\n" if kept else "")),
        text=True, check=True)
    print("v2.4.3 24-window analysis published; sweep cron removed")


if __name__ == "__main__":
    main()
