"""Pre-registered analysis for the frozen core 2x2.

Inferential unit: complete workload window. Repeated LLM seeds are averaged inside each window.
Primary outcome: ``deadline_viol_pct`` (lower is better). Exactly three two-sided paired contrasts
are tested on the test split: market main effect, multi-agent main effect, and their interaction.
Uncertainty is a paired t 95% CI; p-values are exact sign-flip randomization tests and are Holm
corrected across those three tests. Secondary outcomes are descriptive. Fixed policies are chosen
only on validation deadline violations, then reported once on test.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t as student_t

from pins.freeze_core_2x2 import ROOT
from pins.verify_core_2x2 import verify

DEFAULT_MANIFEST = Path(__file__).with_name("core_2x2_manifest.json")
DEFAULT_WORLDS = ROOT / "runs/core_2x2_worlds"
DEFAULT_ROWS = ROOT / "runs/core_2x2/rows.jsonl"
SECONDARY = ("mean_bsd", "mean_wait_s", "p50_wait_s", "useful_util_win", "util_win",
             "completed", "resize_events", "resize_overhead_s", "tok_prompt", "tok_completion",
             "tok_wall", "fallbacks", "sel_invalid")


def mean(xs):
    return statistics.mean(xs)


def ci95(differences: list[float]) -> tuple[float, float]:
    n = len(differences)
    centre = mean(differences)
    if n < 2:
        return centre, centre
    half = student_t.ppf(0.975, n - 1) * statistics.stdev(differences) / math.sqrt(n)
    return centre - half, centre + half


def sign_flip_p(differences: list[float]) -> float:
    observed = abs(mean(differences))
    n = len(differences)
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=n):
        candidate = abs(sum(s * x for s, x in zip(signs, differences)) / n)
        extreme += candidate >= observed - 1e-12
    return extreme / (2 ** n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    out, running, m = {}, 0.0, len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        out[name] = running
    return out


def analyse(manifest_path: Path, worlds: Path, rows_path: Path,
            allow_incomplete: bool = False) -> dict:
    verify(manifest_path, worlds, check_results=True, results=rows_path)
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    test_windows = sorted(w["window"] for w in manifest["windows"] if w["split"] == "test")
    seeds = manifest["inference"]["paired_seeds"]
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    core_rows = [r for r in rows if r["split"] == "test" and r["structure"] in ("single", "multi")]
    for row in core_rows:
        values.setdefault((row["structure"], row["family"]), {}).setdefault(row["window"], []).append(
            float(row["deadline_viol_pct"]))
    missing = []
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for structure in ("single", "multi"):
        for family in ("nm", "mkt"):
            key = (structure, family)
            cells[key] = {}
            for window in test_windows:
                got = values.get(key, {}).get(window, [])
                if len(got) != len(seeds):
                    missing.append(f"{structure}/{family}/{window}: {len(got)}/{len(seeds)} seeds")
                elif got:
                    cells[key][window] = mean(got)
    if missing and not allow_incomplete:
        raise RuntimeError("incomplete factorial: " + "; ".join(missing[:8]))
    complete = sorted(set.intersection(*(set(v) for v in cells.values()))) if all(cells.values()) else []
    if not complete:
        return {"complete": False, "missing": missing}

    diffs = {
        "market_main_effect": [
            ((cells[("single", "mkt")][w] - cells[("single", "nm")][w]) +
             (cells[("multi", "mkt")][w] - cells[("multi", "nm")][w])) / 2 for w in complete],
        "multi_agent_main_effect": [
            ((cells[("multi", "nm")][w] - cells[("single", "nm")][w]) +
             (cells[("multi", "mkt")][w] - cells[("single", "mkt")][w])) / 2 for w in complete],
        "interaction": [
            (cells[("multi", "mkt")][w] - cells[("single", "mkt")][w]) -
            (cells[("multi", "nm")][w] - cells[("single", "nm")][w]) for w in complete],
    }
    raw_p = {name: sign_flip_p(ds) for name, ds in diffs.items()}
    adjusted = holm(raw_p)
    effects = {name: {"mean_difference_pct_points": mean(ds), "ci95": list(ci95(ds)),
                      "p_exact": raw_p[name], "p_holm": adjusted[name], "n_windows": len(ds)}
               for name, ds in diffs.items()}

    fixed = [r for r in rows if r["structure"] == "fixed"]
    selected = {}
    for family in ("nm", "mkt"):
        candidates = sorted({(r["ordering"], r["sizing"]) for r in fixed if r["family"] == family})
        scores = {}
        for candidate in candidates:
            vals = [float(r["deadline_viol_pct"]) for r in fixed
                    if r["split"] == "validation" and r["family"] == family and
                    (r["ordering"], r["sizing"]) == candidate]
            if vals:
                scores[candidate] = mean(vals)
        if scores:
            winner = min(scores, key=lambda x: (scores[x], x))
            test = [float(r["deadline_viol_pct"]) for r in fixed if r["split"] == "test" and
                    r["family"] == family and (r["ordering"], r["sizing"]) == winner]
            selected[family] = {"ordering": winner[0], "sizing": winner[1],
                                "validation_mean": scores[winner],
                                "test_mean": mean(test) if test else None, "test_n": len(test)}

    cell_means = {f"{s}_{f}": mean(list(cells[(s, f)].values()))
                  for s in ("single", "multi") for f in ("nm", "mkt")}
    secondary = {}
    for structure in ("single", "multi"):
        for family in ("nm", "mkt"):
            subset = [r for r in core_rows if r["structure"] == structure and r["family"] == family]
            secondary[f"{structure}_{family}"] = {
                metric: mean([float(r[metric]) for r in subset if r.get(metric) is not None])
                for metric in SECONDARY if any(r.get(metric) is not None for r in subset)}
    return {"complete": not missing, "analysis_windows": complete, "missing": missing,
            "cell_means": cell_means, "effects": effects,
            "validation_selected_fixed_baselines": selected, "secondary_descriptive": secondary}


def markdown(result: dict) -> str:
    lines = ["# Core 2x2 analysis", ""]
    if not result.get("analysis_windows"):
        return "\n".join(lines + ["No complete paired test windows yet.", ""])
    lines += ["Primary metric: synthetic deadline violation rate (percentage points; lower is better).", "",
              "| Contrast | Mean difference | 95% CI | exact p | Holm p | n |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in result["effects"].items():
        lines.append(f"| {name} | {row['mean_difference_pct_points']:.3f} | "
                     f"[{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}] | "
                     f"{row['p_exact']:.4f} | {row['p_holm']:.4f} | {row['n_windows']} |")
    lines += ["", "Negative main effects favor market or multi-agent, respectively.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--worlds", type=Path, default=DEFAULT_WORLDS)
    ap.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/core_2x2/analysis.json")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    result = analyse(args.manifest, args.worlds, args.rows, args.allow_incomplete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
