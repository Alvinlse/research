"""Is there anything to distill? — r1 referee vs vanilla 14b referee, on the STORED windows.

Distillation (TinyLLM, arXiv 2402.04616) needs the teacher to be better than the student at
the task. Exp 51-54 kept finding null transfer; this checks the premise those experiments
assume — that r1's refereeing is worth transferring at all.

Pairing is verified, not assumed: the two tiers are only compared where their no-llm floor
arrays match seed-for-seed (same workload, same spikes).
"""
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
A, B = "deepseek-r1:32b+referee", "qwen2.5:14b+referee"
METRICS = ("prod_sla", "sla", "util", "slowdown")


def ci95(xs):
    if len(xs) < 2:
        return float("nan"), float("nan")
    return st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))


def main():
    tiers = json.load(open(os.path.join(HERE, "results_trace_replay.json")))["tiers"]
    for pool in sorted(set(tiers[A]["per_seed"]) & set(tiers[B]["per_seed"]), key=int):
        pa, pb = tiers[A]["per_seed"][pool], tiers[B]["per_seed"][pool]
        fa, fb = pa.get("no-llm", []), pb.get("no-llm", [])
        n = min(len(pa.get("referee", [])), len(pb.get("referee", [])))
        # paired only where the same seed produced the same floor outcome
        idx = [i for i in range(n)
               if i < len(fa) and i < len(fb)
               and all(abs(fa[i][m] - fb[i][m]) < 1e-9 for m in ("sla", "prod_sla", "util"))]
        print(f"\n=== pool {pool}: {len(pa.get('referee', []))} r1 seeds, "
              f"{len(pb.get('referee', []))} 14b seeds, {len(idx)} PAIRED (same floor)")
        if not idx:
            print("  no shared windows — not comparable")
            continue
        for m in METRICS:
            d = [pa["referee"][i][m] - pb["referee"][i][m] for i in idx]
            mean, half = ci95(d)
            sig = "*" if abs(mean) > half else " "
            unit = 100 if m in ("prod_sla", "sla", "util") else 1
            print(f"  d{m:9s} (r1 - 14b): {mean * unit:+6.2f} +/- {half * unit:5.2f}{sig}")
        for tier, ps in ((A, pa), (B, pb)):
            v = [ps["referee"][i]["prod_sla"] for i in idx]
            f = [ps["no-llm"][i]["prod_sla"] for i in idx]
            delta = (st.mean(v) - st.mean(f)) * 100
            print(f"  {tier:26s} prodSLA {st.mean(v) * 100:5.1f}%  "
                  f"(floor {st.mean(f) * 100:5.1f}%, d={delta:+5.1f})")
    print("\n* = 95% CI excludes 0. Positive d => r1 better.")


if __name__ == "__main__":
    main()
