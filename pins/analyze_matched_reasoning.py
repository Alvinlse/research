"""Paired window analysis; no states or repeated seeds are treated as independent."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics
from pathlib import Path

from pins.run_matched_reasoning import digest, read_rows


def contrast(deltas):
    rng = random.Random(20260916)
    n = len(deltas)
    centre = statistics.mean(deltas)
    if n < 2:
        return {"n_windows": n, "mean_delta_pp": centre, "ci95": None, "p": None,
                "warning": "at least two windows required for uncertainty"}
    samples = sorted(statistics.mean(rng.choices(deltas, k=n)) for _ in range(10000))
    nonzero = [x for x in deltas if abs(x) > 1e-12]
    observed = abs(sum(nonzero))
    if len(nonzero) <= 16:
        signs = itertools.product((-1, 1), repeat=len(nonzero))
        extreme = sum(abs(sum(s*x for s, x in zip(sign, nonzero))) >= observed - 1e-12 for sign in signs)
        p = extreme / 2 ** len(nonzero)
        method = "exact paired sign-flip"
    else:
        extreme = sum(abs(sum(rng.choice((-1, 1))*x for x in nonzero)) >= observed - 1e-12
                      for _ in range(100000))
        p = (extreme + 1) / 100001
        method = "100000 Monte Carlo paired sign flips; seed 20260916"
    return {"n_windows": n, "mean_delta_pp": centre, "median_delta_pp": statistics.median(deltas),
            "ci95": [samples[249], samples[9749]], "p": p, "test": method,
            "wins": sum(x < -1e-12 for x in deltas), "ties": sum(abs(x) <= 1e-12 for x in deltas),
            "losses": sum(x > 1e-12 for x in deltas), "window_deltas_pp": deltas}


def analyze(out):
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fingerprint = digest(manifest_path)
    rows = read_rows(out / "rows.jsonl")
    expected = {t["key"]: t for t in manifest["tasks"]}
    actual = {}
    for row in rows:
        if row["key"] in actual or row["key"] not in expected or row["run_manifest_sha256"] != fingerprint:
            raise ValueError("duplicate or incompatible result row")
        if any(row[k] != v for k, v in expected[row["key"]].items()):
            raise ValueError("result task metadata changed")
        actual[row["key"]] = row
    if set(actual) != set(expected):
        raise ValueError(f"incomplete paired experiment: {len(actual)}/{len(expected)} cells; finish before analysis")
    cfg = manifest["config"]
    # Average states/seeds within each family, then average families within window.
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["window"], row["structure"], row["family"])].append(float(row["deadline_viol_pct"]))
    windows = sorted({r["window"] for r in rows})
    values = {}
    for window in windows:
        for structure in cfg["structures"]:
            values[window, structure] = statistics.mean(
                statistics.mean(grouped[window, structure, family]) for family in cfg["families"])
    comparisons = [cfg["primary_contrast"], *cfg["secondary_contrasts"]]
    contrasts = {f"{left}_minus_{right}": contrast([values[w, left] - values[w, right] for w in windows])
                 for left, right in comparisons}
    ordered = sorted((name for name in contrasts if contrasts[name]["p"] is not None),
                     key=lambda name: contrasts[name]["p"])
    previous = 0.0
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1.0, contrasts[name]["p"] * (len(ordered)-rank)))
        contrasts[name]["holm_p"] = previous
    descriptive = {}
    for structure in cfg["structures"]:
        selected = [r for r in rows if r["structure"] == structure]
        descriptive[structure] = {
            "macro_deadline_viol_pct": statistics.mean(values[w, structure] for w in windows),
            "total_calls": sum(r["llm_calls"] for r in selected),
            "total_prompt_tokens": sum(r.get("tok_prompt", 0) for r in selected),
            "total_completion_tokens": sum(r.get("tok_completion", 0) for r in selected),
            "total_inference_wall_s": sum(r.get("tok_wall", 0) for r in selected),
            "total_invalid_final": sum(r.get("sel_invalid", 0) for r in selected),
            "total_invalid_reviews": sum(r.get("invalid_reviews", 0) for r in selected),
            "total_llm_errors": sum(r.get("llm_errors", 0) for r in selected),
        }
        for metric in ("regret_pp", "mean_wait_s", "p90_wait_s", "jain_user_wait", "fixed_deadline_viol_pct",
                       "oracle_deadline_viol_pct", "near_optimal"):
            if all(metric in row and row[metric] is not None for row in selected):
                family_means = []
                for window in windows:
                    for family in cfg["families"]:
                        family_means.append(statistics.mean(float(r[metric]) for r in selected
                                                           if r["window"] == window and r["family"] == family))
                descriptive[structure]["macro_" + metric] = statistics.mean(family_means)
    return {"mode": manifest["mode"], "split": manifest["split"], "interpretation": cfg["interpretation"],
            "primary": "_minus_".join(cfg["primary_contrast"]), "lower_is_better": True,
            "inference_unit": "window, averaging states and seeds within family, then families",
            "ci_method": "paired window percentile bootstrap; 10000 resamples; seed 20260916",
            "multiplicity": "Holm across the three registered contrasts",
            "structures": descriptive, "contrasts": contrasts,
            "budget_note": "Three-call arms match call count and output caps per epoch, not actual tokens. "
                           "Closed-loop arms can visit different numbers of decision epochs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.out)
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Matched reasoning results", "", report["interpretation"], "",
             f"Mode: {report['mode']}; split: {report['split']}. Negative deltas favor Multi.", "",
             "| Comparison | Mean delta (pp) | 95% CI | Holm p | Wins/ties/losses |",
             "| --- | ---: | --- | ---: | --- |"]
    for name, item in report["contrasts"].items():
        lines.append(f"| {name} | {item['mean_delta_pp']:.4f} | {item['ci95']} | "
                     f"{item.get('holm_p')} | {item.get('wins')}/{item.get('ties')}/{item.get('losses')} |")
    lines.extend(["", report["budget_note"], "", "Full costs, error counts and secondary metrics: analysis.json."])
    (args.out / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
