"""Small paired local-model Multi-Agent follow-up using the frozen core-2x2 worlds.

This intentionally does not modify the preregistered manifest or its implementation files.
Qwen3 thinking is disabled because the experiment expects strict JSON within a 400-token budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pins.correction as correction
from pins.elastisim_bench import run


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="d178h3")
    parser.add_argument("--family", choices=("nm", "mkt"), default="nm")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/qwen3_multi_smoke/rows.jsonl")
    args = parser.parse_args()

    original_ask = correction._ask
    model_slug = args.model.replace(":", "_").replace("/", "_")

    def no_think_ask(*ask_args, **ask_kwargs):
        ask_kwargs.setdefault("think", False)
        return original_ask(*ask_args, **ask_kwargs)

    correction._ask = no_think_ask
    result = run(
        ROOT / "runs/core_2x2_worlds" / args.window,
        "policy_negotiate",
        args.model,
        interval=300,
        tag=f"{model_slug}_smoke_multi_{args.family}_seed{args.seed}",
        quiet=True,
        sizer="as_requested",
        family=args.family,
        sel_every=1800,
        temperature=0.1,
        llm_seed=args.seed,
        num_predict=400,
        rcon_threshold_s=300.0,
    )
    row = {
        "follow_up": f"{model_slug}-multi-smoke-v1",
        "window": args.window,
        "structure": "multi",
        "family": args.family,
        "seed": args.seed,
        "model": args.model,
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        sink.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
