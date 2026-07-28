"""One-shot: merge the seeds-8-31 thinking side-run into the live 35b referee tier,
then write a paired-stats report. Safe to run only after BOTH 35b sweeps have exited
(no concurrent writer on the live results file). Idempotent-ish: re-merging detects a
tier already at 32 seeds and skips the concat."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pins.trace_replay import paired_ci

LIVE = "pins/results_trace_replay.json"
SIDE = "pins/results_35b_think_reseed.json"
THINK = "qwen3.5:35b+referee"
NOTHINK = "qwen3.5:35b+referee+nothink"
POOL = "8"
METRICS = (("sla", "dSLA", 1), ("prod_sla", "dprodSLA", 1),
           ("util", "dutil", 1), ("slowdown", "dslow", 0))
out = []


def line(s=""):
    out.append(s)


def vs_floor(ps, pols):
    floor = ps["no-llm"]
    rows = []
    for pol in pols:
        if pol not in ps:
            continue
        parts = []
        for m, lbl, pct in METRICS:
            diffs = [a[m] - b[m] for a, b in zip(ps[pol], floor)]
            mean, h = paired_ci(diffs)
            u = 100.0 if pct else 1.0
            sig = "*" if h < abs(mean) else " "
            parts.append(f"{lbl} {mean*u:+6.1f} ±{h*u:4.1f}{sig}")
        fb = sum(r.get("fallback_rate", 0) for r in ps[pol]) / len(ps[pol])
        rows.append(f"  {pol:<11} " + "  ".join(parts) + f"  fb {fb:.0%}")
    return "\n".join(rows), len(floor)


def head_to_head(a_ps, b_ps, pol, alabel, blabel):
    ra, rb = a_ps[pol], b_ps[pol]
    n = min(len(ra), len(rb))
    parts = []
    for m, lbl, pct in METRICS:
        diffs = [x[m] - y[m] for x, y in zip(ra[:n], rb[:n])]
        mean, h = paired_ci(diffs)
        u = 100.0 if pct else 1.0
        sig = "*" if h < abs(mean) else " "
        parts.append(f"{lbl} {mean*u:+6.1f} ±{h*u:4.1f}{sig}")
    floor_match = all(abs(a_ps["no-llm"][i]["sla"] - b_ps["no-llm"][i]["sla"]) < 1e-9
                      for i in range(n))
    return f"  {alabel} MINUS {blabel} ({pol}, n={n}, floors match={floor_match}):\n    " \
           + "  ".join(parts)


live = json.load(open(LIVE))
tiers = live["tiers"]

# --- merge: seeds 0-7 (live) + seeds 8-31 (side) into the live thinking tier ---
merged_note = ""
tps = tiers.get(THINK, {}).get("per_seed", {}).get(POOL)
if tps and len(tps.get("no-llm", [])) >= 32:
    merged_note = f"thinking tier already at n={len(tps['no-llm'])}; skipped merge"
elif tps and os.path.exists(SIDE):
    side = json.load(open(SIDE))
    sps = side["tiers"][THINK]["per_seed"][POOL]
    for pol in tps:
        if pol in sps:
            tps[pol] = tps[pol][:8] + sps[pol]        # 0-7 then 8-31, seed-ordered
    tiers[THINK]["n_seeds"] = len(tps["no-llm"])
    tiers[THINK]["decisions"] = (tiers[THINK].get("decisions", [])
                                 + side["tiers"][THINK].get("decisions", []))
    json.dump(live, open(LIVE, "w"), indent=2)
    merged_note = f"merged -> thinking tier now n={len(tps['no-llm'])}"
else:
    merged_note = "merge skipped: side file or live thinking tier missing"

# --- report ---
line("# Exp 49 — qwen3.5:35b referee, thinking vs no-think (pool 8)")
line(f"\n_{merged_note}_\n")

for tier, title in ((NOTHINK, "NO-THINK (think=False)"), (THINK, "THINKING (default)")):
    line(f"\n## {tier}  —  {title}")
    ps = tiers.get(tier, {}).get("per_seed", {}).get(POOL)
    if not ps:
        line("  (missing — run did not finish / not saved)")
        continue
    body, n = vs_floor(ps, ("referee", "negotiated"))
    line(f"vs own floor, pool 8 (n={n}), paired by seed, 95% CI:")
    line(body)

line("\n## Head-to-head (paired by seed)")
nps = tiers.get(NOTHINK, {}).get("per_seed", {}).get(POOL)
tps2 = tiers.get(THINK, {}).get("per_seed", {}).get(POOL)
if nps and tps2:
    line(head_to_head(nps, tps2, "referee", "no-think", "thinking"))
for other, olabel in (("qwen2.5:14b+referee", "14b"), ("deepseek-r1:32b+referee", "r1:32b")):
    ops = tiers.get(other, {}).get("per_seed", {}).get(POOL)
    if tps2 and ops:
        line(head_to_head(tps2, ops, "referee", "thinking-35b", olabel))

report = "\n".join(out)
open("pins/exp49_35b_report.md", "w").write(report + "\n")
print(report)
