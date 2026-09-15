"""Resumable, window-interleaved driver; see matched_reasoning_prereg.md."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from pins.matched_reasoning import STRUCTURES, decide, run_closed_loop

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "pins/matched_reasoning_config.json"
MANIFEST = ROOT / "pins/core_2x2_manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_config(cfg):
    if cfg["structures"] != list(STRUCTURES) or cfg["families"] != ["nm", "mkt"]:
        raise ValueError("v1 requires all four structures and both families")
    if cfg["selection_interval_s"] != 1800 or cfg["simulation_interval_s"] != 300:
        raise ValueError("v1 clocks are fixed at 1800/300 seconds")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if cfg["num_predict"] <= 0 or not 0 <= cfg["temperature"] <= 2:
        raise ValueError("invalid decoding budget")


def load_states(data, split, manifest, cfg):
    """Read only observable user packets into inference; rewards stay in the scorer."""
    from pins.elastisim_bench import POLICY_FAMILY, SELECTOR_MENU
    by_split = {name: read_rows(data / f"{name}.jsonl") for name in ("train", "validation", "test")}
    seen_windows = {}
    for name, rows in by_split.items():
        for row in rows:
            previous = seen_windows.setdefault(row["window"], name)
            if previous != name:
                raise ValueError("dataset window leaks across splits")
    allowed = {w["window"] for w in manifest["windows"] if w["split"] == split}
    states, seen = [], set()
    for row in by_split[split]:
        family = row["family"]
        if family not in cfg["families"] or row["window"] not in allowed:
            raise ValueError("dataset family/window disagrees with selected manifest split")
        key = (row["window"], family, row["t"])
        if key in seen:
            raise ValueError("duplicate offline state")
        seen.add(key)
        menu = {f"{o}+{s}" for o in POLICY_FAMILY[family] for s in SELECTOR_MENU["sizing"]}
        if set(row["rewards"]) != menu or not all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in row["rewards"].values()):
            raise ValueError("oracle requires a complete finite reward vector")
        if row.get("decision_horizon_s") != cfg["selection_interval_s"]:
            raise ValueError("oracle horizon mismatch: regenerate one-interval labels")
        if row.get("baseline") != "+".join(cfg["fallbacks"][family]):
            raise ValueError("oracle continuation differs from registered fallback")
        packets = [m["content"] for m in row["messages"] if m["role"] == "user"]
        if len(packets) != 1 or not isinstance(packets[0], str):
            raise ValueError("expected exactly one observable user packet")
        states.append({"window": row["window"], "family": family, "t": row["t"],
                       "packet": packets[0], "rewards": row["rewards"]})
    if not states:
        raise ValueError(f"no offline states in {data}/{split}.jsonl")
    # Every retained window must support a paired family average.
    for window in {s["window"] for s in states}:
        if {s["family"] for s in states if s["window"] == window} != set(cfg["families"]):
            raise ValueError(f"incomplete family coverage: {window}")
    return states


def build_tasks(cfg, manifest, split, mode, states=()):
    windows = [w for w in manifest["windows"] if w["split"] == split]
    tasks = []
    rng = random.Random(20260916)
    for window in windows:
        cells = []
        units = ([s for s in states if s["window"] == window["window"]] if mode == "offline"
                 else [{"window": window["window"], "family": f, "t": None} for f in cfg["families"]])
        for unit in units:
            for seed in cfg["seeds"]:
                for structure in cfg["structures"]:
                    cells.append({"window": unit["window"], "family": unit["family"],
                                  "t": unit["t"], "seed": seed, "structure": structure})
        rng.shuffle(cells)
        tasks.extend(cells)
    for task in tasks:
        task["key"] = "|".join(str(task[k]) for k in ("window", "family", "t", "seed", "structure"))
    return tasks


def model_identity(model):
    from pins.llm_agent import HOST
    with urllib.request.urlopen(HOST.rstrip("/") + "/api/tags", timeout=15) as response:
        models = json.load(response)["models"]
    match = next((m for m in models if m.get("name") in (model, model + ":latest")), None)
    if not match or not match.get("digest"):
        raise ValueError(f"Ollama model {model} must be installed before running")
    return {"name": model, "digest": match["digest"], "host": HOST}


def freeze(out, identity):
    path = out / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("resume rejected: code, data, model, config, or task set changed; use a new output directory")
    else:
        path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return digest(path)


def verify_worlds(manifest_path, worlds, cfg):
    """Freeze the inherited b97935f implementation without restamping old results.

    Upstream added post-core code after its old manifest hash was frozen. Pin that
    exact inherited source in our config, then use the original full world checker
    with a temporary derived contract. No historical manifest/result is mutated.
    """
    from pins.freeze_core_2x2 import IMPLEMENTATION_FILES, ROOT as core_root, combined_sha
    from pins.verify_core_2x2 import verify
    actual = combined_sha([core_root / p for p in IMPLEMENTATION_FILES])
    if actual != cfg["inherited_implementation_sha256"]:
        raise ValueError("inherited simulator implementation changed since next-experiment freeze")
    contract = json.loads(manifest_path.read_text())
    contract["implementation"]["sha256"] = actual
    with tempfile.TemporaryDirectory(prefix="matched-world-contract-") as directory:
        path = Path(directory) / "world_contract.json"
        path.write_text(json.dumps(contract))
        return verify(path, worlds, check_results=False)


def offline_cell(state, task, cfg):
    from pins.correction import _ask
    from pins.elastisim_bench import POLICY_FAMILY, SELECTOR_MENU
    from pins.llm_agent import HOST, take_tokens
    take_tokens()
    audit = []
    start = time.monotonic()

    def ask(system, user, stage, seed):
        return _ask(system, user, cfg["model"], HOST, {},
                    f"matched-v1-{task['structure']}-{stage}", seed=seed, think=False,
                    temperature=cfg["temperature"], num_predict=cfg["num_predict"], audit=audit)

    result = decide(state["packet"], task["structure"], POLICY_FAMILY[task["family"]],
                    list(SELECTOR_MENU["sizing"]), cfg["fallbacks"][task["family"]],
                    ask=ask, seed=task["seed"])
    answer = result["answer"]
    action = answer["ordering"] + "+" + answer["default_sizing"]
    reward = state["rewards"][action]
    best = max(state["rewards"].values())
    meter = take_tokens()
    return {"deadline_viol_pct": -reward, "regret_pp": best - reward,
            "oracle_deadline_viol_pct": -best,
            "fixed_deadline_viol_pct": -state["rewards"]["+".join(cfg["fallbacks"][task["family"]])],
            "near_optimal": best - reward <= 0.1, "action": action,
            "sel_invalid": int(not result["valid"]), "invalid_reviews": result["invalid_reviews"],
            "llm_calls": result["calls"], "llm_errors": sum(bool(a.get("error")) for a in audit),
            "tok_prompt": meter["prompt"], "tok_completion": meter["completion"],
            "tok_calls": meter["calls"], "tok_wall": meter["wall"],
            "wall_s": time.monotonic() - start, "trace": result["trace"]}, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "closed-loop"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worlds", type=Path, default=ROOT / "runs/core_2x2_worlds")
    parser.add_argument("--data", type=Path, help="one-interval oracle dataset directory (offline)")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-runs", type=int, default=1, help="bounded cells per tick; 0 runs all")
    parser.add_argument("--dry-run", action="store_true", help="list plan without inference or simulator")
    args = parser.parse_args()
    cfg, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text())
    validate_config(cfg)
    if args.max_runs < 0:
        parser.error("--max-runs must be nonnegative")
    if args.mode == "offline" and args.data is None:
        parser.error("offline mode requires --data")
    states = load_states(args.data, args.split, manifest, cfg) if args.mode == "offline" else []
    tasks = build_tasks(cfg, manifest, args.split, args.mode, states)
    if not tasks:
        raise ValueError("empty task plan")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "split": args.split, "cells": len(tasks),
                          "windows": len({t['window'] for t in tasks}),
                          "interpretation": cfg["interpretation"], "first": tasks[0],
                          "note": "plan only; runtime dependencies and world provenance not checked"}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Preserve the existing frozen world and implementation verification.
        verify_worlds(args.manifest, args.worlds, cfg)
        from pins.elastisim_bench import POLICY_FAMILY, SELECTOR_MENU
        for family in cfg["families"]:
            fallback = cfg["fallbacks"][family]
            if fallback[0] not in POLICY_FAMILY[family] or fallback[1] not in SELECTOR_MENU["sizing"]:
                raise ValueError("fallback outside family")
        identity = {"config": cfg, "mode": args.mode, "split": args.split, "tasks": tasks,
                    "core_manifest_sha256": digest(args.manifest),
                    "model": model_identity(cfg["model"]),
                    "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                                      for p in sorted((ROOT / "pins").glob("*.py"))},
                    "protocol_sha256": digest(ROOT / "pins/matched_reasoning_prereg.md"),
                    "dataset_sha256": ({p.name: digest(p) for p in sorted(args.data.glob("*.json*"))}
                                       if args.mode == "offline" else {}),
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        manifest_hash = freeze(args.out, identity)
        rows = read_rows(args.out / "rows.jsonl")
        expected = {t["key"] for t in tasks}
        present = set()
        for row in rows:
            if row["key"] not in expected or row["key"] in present or row["run_manifest_sha256"] != manifest_hash:
                raise ValueError("duplicate, foreign, or incompatible result row")
            present.add(row["key"])
        by_state = {(s["window"], s["family"], s["t"]): s for s in states}
        completed = 0
        for task in tasks:
            if task["key"] in present:
                continue
            tag = "mr1_" + manifest_hash[:10] + "_" + hashlib.sha256(task["key"].encode()).hexdigest()[:12]
            if args.mode == "offline":
                result, audit = offline_cell(by_state[(task["window"], task["family"], task["t"])], task, cfg)
                (args.out / f"{tag}_audit.json").write_text(json.dumps(audit, indent=1))
            else:
                fallback = cfg["fallbacks"][task["family"]]
                result = run_closed_loop(
                    args.worlds / task["window"], task["structure"], model=cfg["model"],
                    interval=cfg["simulation_interval_s"], tag=tag, quiet=True,
                    family=task["family"], sizer=fallback[1],
                    sel_every=cfg["selection_interval_s"], temperature=cfg["temperature"],
                    llm_seed=task["seed"], num_predict=cfg["num_predict"],
                    fallback_ordering=fallback[0], fallback_sizing=fallback[1],
                    packet_v2=False, demand_v2=False, rcon_threshold_s=300.0)
                item = next(w for w in manifest["windows"] if w["window"] == task["window"])
                if result["n"] != item["n_jobs"]:
                    raise ValueError("scored job count changed")
                policy_log = read_rows_policy(args.worlds / task["window"] / "out" / f"{tag}_policy_log.json")
                expected_calls = len(policy_log) * (1 if task["structure"] == "single" else 3)
                if result["llm_calls"] != expected_calls:
                    raise ValueError("decision/call accounting mismatch")
                result["decision_epochs"] = len(policy_log)
                result["invalid_reviews"] = sum(r.get("invalid_reviews", 0) for r in policy_log)
            row = {**result, **task, "split": args.split, "mode": args.mode,
                   "run_manifest_sha256": manifest_hash, "artifact_tag": tag}
            append(args.out / "rows.jsonl", row)
            print(f"{task['key']}: deadline={row['deadline_viol_pct']} calls={row['llm_calls']}", flush=True)
            completed += 1
            if args.max_runs and completed >= args.max_runs:
                break


def read_rows_policy(path):
    return json.loads(path.read_text()) if path.exists() else []


if __name__ == "__main__":
    main()
