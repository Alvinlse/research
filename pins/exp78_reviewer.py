"""Exp 78 — does the demand reviewer's judgement track MEASURED usage on real jobs?

  .venv/bin/python -m pins.exp78_reviewer --model qwen2.5:14b --n 240

The correction layer's whole premise is that free text carries a fact the bid does not: "this
job needs more GPU than its numerical bid suggests". On the Alibaba v2020 trace that premise
is now checkable against measurement rather than against predicates we authored:

    input  (what an operator would have BEFORE the job runs): job_context.csv — task roles,
           instance count, GPU class. The bid never sees these.
    truth  (what we check AFTERWARDS):                        job_usage.csv  — measured
           gpu_wrk_util. Never shown to the model.

DESIGN — the numeric state is held NEUTRAL and IDENTICAL across jobs (same tier, same deadline
bucket, bid_allocation = requested = the job's declared quanta). So the only thing that varies
between prompts is the note. Any structure in the reviewer's answers is therefore attributable
to the text channel, not to numbers it could have read off the bid.

SCORING. Two pre-registered classes from the measurement:
    IDLE   gpu_util_mean < 1%   — 36.2% of real jobs; they hold quanta and use nothing.
                                  Asking for MORE here is a false positive.
    BUSY   gpu_util_mean > 30%  — jobs that demonstrably convert GPU into work.
The headline is DISCRIMINATION = P(ask | BUSY) - P(ask | IDLE). Zero means the note carries no
usable signal for this reviewer, however eloquent its evidence string. Accuracy is NOT the
headline: a reviewer that never asks scores 64% "correct" by staying silent, which is exactly
the failure H2 found.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random

from pins.correction import SYSTEM_DEMAND_CTX, _ask
from pins.llm_agent import take_tokens

# PRE-REGISTERED 2026-07-22, before any run. The `policy` prompt (correction.SYSTEM_DEMAND_CTX)
# defines evidence as SOCIAL exception text — starvation history, an operator instruction, an
# external deadline. Exp 78 run 1 fed it TECHNICAL context (task roles, instance counts) and it
# correctly said no to 239/240 jobs: discrimination -0.8%, a clean null and a mis-pairing of
# prompt to channel, not a model failure.
# This prompt asks the question the trace can actually answer. It is scored on the same jobs,
# the same measurement, and against a deterministic role-lookup baseline, because a lookup
# table already reaches rho=0.33 on this signal and an LLM that merely ties it adds nothing.
SYSTEM_WORKLOAD_CTX = (
    "You are a GPU capacity reviewer. You are given one job's WORKLOAD PROFILE: the roles of "
    "its tasks, how many instances it runs, the GPU share it planned per instance, and the "
    "number of GPU units it was allocated.\n"
    "Answer ONE question: if this job were given an additional GPU unit, would it convert that "
    "unit into real work?\n"
    "Reason from what the roles imply about how the job uses a GPU. Some roles coordinate, "
    "serve parameters, evaluate, visualise or host an interactive session, and add little GPU "
    "computation however much capacity they hold; others run dense training or compute kernels "
    "and saturate what they are given. The allocation itself tells you what was REQUESTED, not "
    "what the job can use — a large request is not evidence of capacity to use it.\n"
    "Respond with ONLY this JSON object:\n"
    '{"converts": true or false, "evidence": "<the part of the profile that decides it>"}'
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "alibaba-gpu-v2020")
OUT = os.environ.get("PINS_RESULTS", os.path.join(HERE, "results_exp78.json"))
IDLE_MAX, BUSY_MIN = 1.0, 30.0


def load(n: int, seed: int) -> list[dict]:
    ctxf = {r["job_name"]: r for r in csv.DictReader(open(os.path.join(DATA, "job_context.csv")))}
    ctx = ctxf
    rep = {r["job_name"]: r for r in csv.DictReader(open(os.path.join(DATA, "replay_jobs.csv")))}
    jobs = []
    with open(os.path.join(DATA, "job_usage.csv")) as f:
        for r in csv.DictReader(f):
            jid = r["job_name"]
            if jid not in ctx or jid not in rep or not r["gpu_util_mean"]:
                continue
            u = float(r["gpu_util_mean"])
            cls = "IDLE" if u < IDLE_MAX else ("BUSY" if u > BUSY_MIN else None)
            if cls is None:
                continue                      # the middle is not scored: it has no clear answer
            jobs.append({"jid": jid, "roles": ctx[jid]["roles"], "note": ctx[jid]["note"],
                         "inst": ctx[jid]["inst_total"],
                         "quanta": int(rep[jid]["quanta"]), "util": u, "class": cls})
    rng = random.Random(seed)
    rng.shuffle(jobs)
    # stratify by (class, role signature): the tail roles are the point, so they must not be
    # swamped by the 53% of jobs that are plain `tensorflow`
    per, out = collections.Counter(), []
    cap = max(1, n // 12)
    for j in jobs:
        k = (j["class"], j["roles"])
        if per[k] >= cap:
            continue
        per[k] += 1
        out.append(j)
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", default="full",
                    choices=("full", "no-numbers", "no-role", "shuffled-role"),
                    help="role-controlled ablation grid. 'full' = role+instances+plan_gpu; "
                         "'no-numbers' drops plan_gpu/quanta (the declaration the bid already "
                         "prices); 'no-role' drops the role label; 'shuffled-role' pairs real "
                         "numbers with another job's role. Measured utilisation is never an "
                         "input in any condition — it is the scoring truth.")
    ap.add_argument("--prompt", default="workload", choices=("policy", "workload"),
                    help="'policy' = the correction layer's social-exception reviewer "
                         "(Exp 78 run 1, a null); 'workload' = the pre-registered capacity "
                         "question this trace can answer.")
    a = ap.parse_args()

    jobs = load(a.n, a.seed)
    _rng = random.Random(a.seed + 1)           # role labels shuffled ACROSS jobs, same multiset
    _perm = [j["roles"].replace("|", ", ") for j in jobs]
    _rng.shuffle(_perm)
    for _j, _r in zip(jobs, _perm):
        _j["shuffled_role"] = _r
    print(f"{len(jobs)} real jobs: " + ", ".join(
        f"{k}={v}" for k, v in collections.Counter(j["class"] for j in jobs).items()))
    cache: dict = {}
    res = []
    for i, j in enumerate(jobs, 1):
        # numeric state deliberately neutral and identical: only the note varies
        user = (f"job {j['jid'][:12]}: tier=besteffort deadline=ontrack "
                f"bid_allocation={j['quanta']} requested={j['quanta']}\n"
                f"note: \"{j['note']}\"")
        if a.prompt == "workload":
            roles, inst = j["roles"].replace("|", ", "), j.get("inst") or "?"
            if a.ablate == "no-numbers":       # role + instances, no declared request at all
                prof = f"task roles: {roles}; {inst} instances in total."
                user = f"job {j['jid'][:12]}\nworkload profile: {prof}"
            elif a.ablate == "no-role":        # numbers only, role hidden
                prof = j["note"].split(";", 1)[1].strip() if ";" in j["note"] else j["note"]
                user = (f"job {j['jid'][:12]}: allocated {j['quanta']} GPU units\n"
                        f"workload profile: {prof}")
            elif a.ablate == "shuffled-role":  # real numbers, a DIFFERENT job's role label
                prof = j["note"].split(";", 1)
                tail = prof[1].strip() if len(prof) > 1 else ""
                user = (f"job {j['jid'][:12]}: allocated {j['quanta']} GPU units\n"
                        f"workload profile: task roles: {j['shuffled_role']}; {tail}")
            else:
                user = (f"job {j['jid'][:12]}: allocated {j['quanta']} GPU units\n"
                        f"workload profile: {j['note']}")
        sysp = SYSTEM_WORKLOAD_CTX if a.prompt == "workload" else SYSTEM_DEMAND_CTX
        d = _ask(sysp, user, a.model, os.environ.get("OLLAMA_HOST",
                 "http://localhost:11434"), cache, "demand-" + a.prompt) or {}
        if a.prompt == "workload":
            extra = 1 if d.get("converts") is True else 0
        else:
            try:
                extra = int(d.get("extra_gpus") or 0)
            except (TypeError, ValueError):
                extra = 0
        res.append(j | {"extra": extra, "evidence": str(d.get("evidence", ""))[:120]})
        print(f"\r  {i}/{len(jobs)}", end="", flush=True)
    print()

    tok = take_tokens()
    ask = collections.Counter((r["class"], r["extra"] > 0) for r in res)
    n_busy = sum(1 for r in res if r["class"] == "BUSY")
    n_idle = sum(1 for r in res if r["class"] == "IDLE")
    p_busy = ask[("BUSY", True)] / max(n_busy, 1)
    p_idle = ask[("IDLE", True)] / max(n_idle, 1)
    print(f"\n  P(ask more | BUSY, util>{BUSY_MIN:.0f}%) = {p_busy:.1%}  ({ask[('BUSY', True)]}/{n_busy})")
    print(f"  P(ask more | IDLE, util<{IDLE_MAX:.0f}%)  = {p_idle:.1%}  ({ask[('IDLE', True)]}/{n_idle})")
    print(f"  DISCRIMINATION = {p_busy - p_idle:+.1%}")
    print(f"  overall ask rate {sum(1 for r in res if r['extra'] > 0) / len(res):.1%}, "
          f"{tok['calls']} calls, {tok['prompt'] + tok['completion']} tokens")

    # deterministic control: a role-lookup table built on the jobs NOT sampled here. If the
    # LLM only ties this, the language model is not earning its place.
    sampled = {r["jid"] for r in res}
    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(DATA, "job_usage.csv"))):
        if r["job_name"] in sampled or not r["gpu_util_mean"]:
            continue
        pool[r["job_name"]] = float(r["gpu_util_mean"])
    ctx = {r["job_name"]: r["roles"] for r in
           csv.DictReader(open(os.path.join(DATA, "job_context.csv")))}
    by_role = collections.defaultdict(list)
    for jid, u in pool.items():
        if jid in ctx:
            by_role[ctx[jid]].append(u)
    med = {k: sorted(v)[len(v) // 2] for k, v in by_role.items() if v}
    b_busy = sum(1 for r in res if r["class"] == "BUSY" and med.get(r["roles"], 0) > 15.0)
    b_idle = sum(1 for r in res if r["class"] == "IDLE" and med.get(r["roles"], 0) > 15.0)
    bp, bi = b_busy / max(n_busy, 1), b_idle / max(n_idle, 1)
    print(f"\n  BASELINE (held-out role median > 15%): P(yes|BUSY) {bp:.1%}, "
          f"P(yes|IDLE) {bi:.1%}, DISCRIMINATION {bp - bi:+.1%}")

    per_role = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in res:
        s = per_role[r["roles"]]
        s[0] += 1
        s[1] += r["extra"] > 0
        s[2] += r["util"]
    print(f"\n  {'role':<26}{'n':>4}{'asked':>7}{'med util':>10}")
    for role, (n, asked, us) in sorted(per_role.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  {role[:26]:<26}{n:>4}{asked:>7}{us / n:>9.1f}%")

    with open(OUT, "w") as f:
        json.dump({"model": a.model, "ablate": a.ablate, "n": len(jobs),
                   "p_busy": p_busy, "p_idle": p_idle,
                   "discrimination": p_busy - p_idle, "results": res}, f, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
