"""Evaluate zero-shot and LoRA-tuned Referees on held-out core oracle states."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import statistics
import time
from pathlib import Path

from pins.finetune_core_referee import BASES, render_prompt
from pins.core_oracle_labels import state_flags
from pins.elastisim_bench import (
    POLICY_DEMAND_REVIEW, POLICY_REVIEW_REFEREE, POLICY_SUPPLY_REVIEW,
)


T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
MAX_INPUT_TOKENS = 8192


def ci95(differences: list[float]) -> tuple[float, float]:
    centre = statistics.mean(differences)
    if len(differences) < 2:
        return centre, centre
    critical = T975.get(len(differences) - 1, 1.96)
    half = critical * statistics.stdev(differences) / math.sqrt(len(differences))
    return centre - half, centre + half


def sign_flip_p(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, differences)) /
            len(differences)) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(differences)))
    return extreme / (2 ** len(differences))


def parse_action(text: str) -> str:
    ordering = re.search(r'"ordering"\s*:\s*"([^"]+)"', text)
    sizing = re.search(r'"(?:default_sizing|sizing)"\s*:\s*"([^"]+)"', text)
    return f"{ordering.group(1)}+{sizing.group(1)}" if ordering and sizing else "UNPARSEABLE"


def score(rows: list[dict], picks: list[str],
          fallback_by_family: dict[str, str] | None = None) -> dict:
    regrets, violations = [], []
    invalid, raw_target, applied_target, raw_argmax, applied_argmax = 0, 0, 0, 0, 0
    for row, pick in zip(rows, picks):
        rewards = row["rewards"]
        raw_target += pick == row.get("target")
        raw_argmax += pick == row.get("argmax")
        if pick not in rewards:
            invalid += 1
            if fallback_by_family is None:
                # Keep a conservative strict-output diagnostic alongside deployment scoring.
                picked_reward = min(rewards.values())
            else:
                fallback = fallback_by_family[row["family"]]
                if fallback not in rewards:
                    raise ValueError(f"fallback {fallback!r} is outside {row['family']} menu")
                picked_reward = rewards[fallback]
                applied_pick = fallback
        else:
            picked_reward = rewards[pick]
            applied_pick = pick
        if pick not in rewards and fallback_by_family is None:
            applied_pick = pick
        applied_target += applied_pick == row.get("target")
        applied_argmax += applied_pick == row.get("argmax")
        regrets.append(max(rewards.values()) - picked_reward)
        violations.append(-picked_reward)
    regrets.sort()
    return {
        "n": len(regrets),
        "mean_deadline_viol_pct": round(statistics.mean(violations), 6),
        "mean_regret": round(statistics.mean(regrets), 6),
        "median_regret": round(statistics.median(regrets), 6),
        "p90_regret": round(regrets[int(0.9 * (len(regrets) - 1))], 6),
        "within_0.1pp": round(sum(value <= 0.1 for value in regrets) / len(regrets), 4),
        "raw_target_accuracy": round(raw_target / len(regrets), 4),
        "applied_target_accuracy": round(applied_target / len(regrets), 4),
        "raw_argmax_accuracy": round(raw_argmax / len(regrets), 4),
        "applied_argmax_accuracy": round(applied_argmax / len(regrets), 4),
        "invalid": invalid,
    }


def generate(rows: list[dict], base: str, adapter: Path | None,
             multi: bool = False) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0})
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    picks, outputs, reviews = [], [], []
    prompt_tokens = completion_tokens = calls = truncated_calls = 0
    started = time.monotonic()

    def ask(system: str, user: str) -> tuple[str, int, int]:
        nonlocal calls, truncated_calls
        prompt = render_prompt(tokenizer, [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            # Preserve the system/action menu at the beginning and the most recent role reviews at
            # the end.  The current frozen packets fit, but count any future truncation explicitly.
            keep_left = MAX_INPUT_TOKENS // 2
            keep_right = MAX_INPUT_TOKENS - keep_left
            for key in ("input_ids", "attention_mask"):
                encoded[key] = torch.cat(
                    (encoded[key][:, :keep_left], encoded[key][:, -keep_right:]), dim=1)
            truncated_calls += 1
        encoded = encoded.to("cuda")
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=100, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        new_ids = generated[0][encoded["input_ids"].shape[1]:]
        calls += 1
        return (tokenizer.decode(new_ids, skip_special_tokens=True),
                int(encoded["input_ids"].numel()), int(new_ids.numel()))

    for index, row in enumerate(rows, 1):
        packet = row["messages"][1]["content"]
        if multi:
            demand, p, c = ask(POLICY_DEMAND_REVIEW, packet)
            prompt_tokens += p
            completion_tokens += c
            supply, p, c = ask(POLICY_SUPPLY_REVIEW, packet)
            prompt_tokens += p
            completion_tokens += c
            referee_packet = (packet + "\n\nDEMAND STATEMENT:\n" + demand +
                              "\n\nSUPPLY STATEMENT:\n" + supply)
            text, p, c = ask(POLICY_REVIEW_REFEREE, referee_packet)
            reviews.append({"demand": demand, "supply": supply})
        else:
            text, p, c = ask(row["messages"][0]["content"], packet)
            reviews.append(None)
        prompt_tokens += p
        completion_tokens += c
        picks.append(parse_action(text))
        outputs.append(text)
        if index % 20 == 0:
            label = "multi" if multi else "single"
            print(f"evaluated {label} {index}/{len(rows)}", flush=True)
    return {
        "picks": picks, "outputs": outputs, "reviews": reviews, "calls": calls,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "truncated_calls": truncated_calls,
        "wall_s": round(time.monotonic() - started, 3),
    }


def applied_window_means(rows: list[dict], picks: list[str],
                         fallback_by_family: dict[str, str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row, pick in zip(rows, picks):
        action = pick if pick in row["rewards"] else fallback_by_family[row["family"]]
        grouped[row["window"]].append(-float(row["rewards"][action]))
    return {window: statistics.mean(values) for window, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("runs/core_referee_oracle_v1/dataset"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", choices=tuple(BASES), default="qwen3-8b")
    parser.add_argument("--base")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--include-multi", action="store_true",
                        help="run Demand + Supply + Referee on the identical held-out states")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/core_referee_eval.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            (args.data / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"no {args.split} rows")

    majority = {}
    train_path = args.data / "train.jsonl"
    train = ([json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
             if train_path.exists() else [])
    embedded = collections.defaultdict(set)
    for row in rows:
        if row.get("baseline"):
            embedded[row["family"]].add(row["baseline"])
    families = {row["family"] for row in rows}
    if set(embedded) == families and all(len(values) == 1 for values in embedded.values()):
        best_fixed = {family: next(iter(values)) for family, values in embedded.items()}
        fixed_source = "validation-selected fixed action embedded by oracle generation"
    elif train:
        best_fixed = {}
        for family in {row["family"] for row in train}:
            family_rows = [row for row in train if row["family"] == family]
            menu = family_rows[0]["rewards"]
            best_fixed[family] = max(menu, key=lambda action: statistics.mean(
                row["rewards"][action] for row in family_rows))
        fixed_source = "oracle training split"
    else:
        raise SystemExit("no independent fixed-policy selection source")

    strategies = {
        "fixed_by_family": {
            "picks": [best_fixed[row["family"]] for row in rows],
            "outputs": [], "reviews": [], "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "truncated_calls": 0, "wall_s": 0.0,
        },
        "oracle": {
            "picks": [max(row["rewards"], key=row["rewards"].get) for row in rows],
            "outputs": [], "reviews": [], "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "truncated_calls": 0, "wall_s": 0.0,
        },
    }
    if train:
        for family in {row["family"] for row in train}:
            family_rows = [row for row in train if row["family"] == family]
            majority[family] = collections.Counter(
                row["target"] for row in family_rows).most_common(1)[0][0]
        strategies["majority_by_family"] = {
            "picks": [majority[row["family"]] for row in rows],
            "outputs": [], "reviews": [], "calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "truncated_calls": 0, "wall_s": 0.0,
        }
    base = args.base or BASES[args.model]
    strategies["zero_shot"] = generate(rows, base, None)
    if args.include_multi:
        strategies["demand_supply_referee"] = generate(rows, base, None, multi=True)
    if args.adapter:
        strategies["fine_tuned"] = generate(rows, base, args.adapter)

    report = {
        "split": args.split, "n": len(rows), "base": base,
        "majority_by_family": majority, "fixed_by_family": best_fixed,
        "fixed_selection_source": fixed_source,
        "state_sample": "pressure-stratified" if any(row.get("state_flags") for row in rows)
                        else "unspecified",
        "arms": {},
    }
    flag_sets = {
        "all": lambda flags: True,
        "requested_contended": lambda flags: bool(flags["requested_contended"]),
        "requested_not_contended": lambda flags: not bool(flags["requested_contended"]),
        "deadline_pressure": lambda flags: bool(flags["deadline_pressure"]),
        "contention_or_deadline_pressure": lambda flags: bool(
            flags["requested_contended"] or flags["deadline_pressure"]),
    }
    for name, generated in strategies.items():
        picks = generated["picks"]
        deployed = score(rows, picks, best_fixed)
        cost = {key: generated[key] for key in
                ("calls", "prompt_tokens", "completion_tokens", "truncated_calls", "wall_s")}
        cost["mean_wall_s_per_call"] = (
            round(float(generated["wall_s"]) / int(generated["calls"]), 6)
            if generated["calls"] else 0.0)
        by_stratum = {}
        for stratum, predicate in flag_sets.items():
            selected = [(row, pick) for row, pick in zip(rows, picks)
                        if predicate(row.get("state_flags") or
                                     state_flags(row["messages"][1]["content"]))]
            if selected:
                by_stratum[stratum] = score(
                    [item[0] for item in selected], [item[1] for item in selected], best_fixed)
        report["arms"][name] = {
            "all": deployed,
            "by_family": {
                family: score(
                    [row for row in rows if row["family"] == family],
                    [pick for row, pick in zip(rows, picks) if row["family"] == family],
                    best_fixed)
                for family in sorted({row["family"] for row in rows})
            },
            "by_stratum": by_stratum,
            "strict_worst_case": score(rows, picks),
            "pick_counts": dict(collections.Counter(picks)),
            "cost": cost,
        }
    report["fixed_candidate_mean_deadline_viol_pct"] = {
        family: {
            action: round(statistics.mean(-row["rewards"][action]
                                          for row in rows if row["family"] == family), 6)
            for action in next(row for row in rows if row["family"] == family)["rewards"]
        }
        for family in sorted({row["family"] for row in rows})
    }
    window_means = {
        name: applied_window_means(rows, generated["picks"], best_fixed)
        for name, generated in strategies.items()
    }
    contrasts = {}
    for name, left, right in (
        ("oracle_minus_fixed", "oracle", "fixed_by_family"),
        ("single_minus_fixed", "zero_shot", "fixed_by_family"),
        ("multi_minus_single", "demand_supply_referee", "zero_shot"),
    ):
        if left not in window_means or right not in window_means:
            continue
        windows = sorted(set(window_means[left]) & set(window_means[right]))
        differences = [window_means[left][window] - window_means[right][window]
                       for window in windows]
        contrasts[name] = {
            "mean_difference_pct_points": round(statistics.mean(differences), 6),
            "ci95": [round(value, 6) for value in ci95(differences)],
            "p_exact_sign_flip": round(sign_flip_p(differences), 6),
            "n_windows": len(windows),
            "direction": "negative favors the left-hand method",
        }
    report["paired_window_contrasts"] = contrasts
    report["invalid_action_semantics"] = {
        "deployed": "use independently selected fixed action in the same family",
        "strict_worst_case": "assign the worst candidate reward for diagnostics only",
        "invalid_outputs_remain_counted": True,
    }
    report["predictions"] = {
        name: [
            {"window": row["window"], "family": row["family"], "t": row["t"],
             "raw_pick": pick,
             "applied_pick": pick if pick in row["rewards"] else best_fixed[row["family"]],
             "fallback": pick not in row["rewards"],
             "raw_output": generated["outputs"][index]
                           if index < len(generated["outputs"]) else None,
             "reviews": generated["reviews"][index]
                        if index < len(generated["reviews"]) else None}
            for index, (row, pick) in enumerate(zip(rows, picks))
        ]
        for name, generated in strategies.items()
        for picks in [generated["picks"]]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
