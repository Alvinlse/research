"""
The LLM 'strategist' that sits ON TOP of the deterministic bid (Stage-2, Exp 10).

Per the project hinge (CLAUDE.md: "the LLM reasons/explains; deterministic code decides") and the
hard-won Stage-1 lesson (Exp 1-7: an LLM cannot calibrate a magnitude), the LLM here NEVER emits a
GB, a %, or a price. The calibrated bid value comes from the deterministic predictor
(`pins/predictor.marginal_values`, standing in for the Stage-1 forecaster). This module asks a
local LLM only a *strategic, categorical* question:

    given my predicted workload + how far behind my deadline I am + my priority tier + how
    contended the pool is right now -> how hard should I fight, and how many GPUs should I
    concentrate on?  (+ a one-sentence justification)

The LLM returns `{stance in {aggressive,balanced,concede}, focus_gpus in [1,capacity], why}`.
`stance` is a discrete choice and `focus_gpus` is a small COUNT/selection (<= capacity) — the kind
of thing Exp 3 showed the LLM does reliably — not a magnitude. Deterministic `apply_strategy` then
turns that into the final marginal-value curve: a fixed stance multiplier on the calibrated
baseline, plus an all-or-nothing CONCENTRATION at `focus_gpus` (the lever Exp 9 showed SLA needs).

The LLM is NEVER in the hot loop: like `pins/forecast/llm_facts.get_facts` (cached per label) and
`pins/job_agent._justify` (once per round), `llm_strategy` is queried once per DISCRETISED state
and cached to JSON, so a 300-step x multi-job x multi-pool sweep costs only ~tens of Ollama calls.
Degrades gracefully: Ollama down / bad JSON / `--no-llm` -> a deterministic rule strategy.

Run:  .venv/bin/python -m pins.llm_agent            # smoke: strategy + justification per context
      .venv/bin/python -m pins.llm_agent --no-llm   # rule fallback only
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "llm_agent_cache.json")
DEFAULT_MODEL = "qwen2.5:3b"          # project default; --model swaps it (e.g. a llama tag)
HOST = "http://localhost:11434"

STANCES = ["aggressive", "balanced", "concede"]
STANCE_MULT = {"aggressive": 1.5, "balanced": 1.0, "concede": 0.6}   # fixed code constants
BEYOND_FOCUS = 0.05                   # near-zero weight for GPUs past the concentration point

DEADLINE_BUCKETS = ["behind", "ontrack", "ahead"]
CONTENTION_BUCKETS = ["low", "high"]
TIERS = ["prod", "besteffort"]

SYSTEM = (
    "You are the negotiation strategist for ONE HPC training job competing for GPUs in a shared "
    "cluster. A sealed-bid auction clears each round; a separate system already computed the "
    "MONETARY value of each GPU to you. Your ONLY job is to choose a bidding STANCE and how many "
    "GPUs to concentrate your bid on. You do NOT output any GB, %, price, or value number.\n"
    "Respond with ONLY this JSON object, using EXACTLY the allowed values:\n"
    '{"stance": "aggressive|balanced|concede", "focus_gpus": <integer>, '
    '"justification": "<one short sentence>"}\n'
    "Guidance: stance = how hard to fight this round. 'aggressive' when you are BEHIND your "
    "deadline and the pool is CONTENDED or you are a 'prod' (production) job; 'concede' when you "
    "are AHEAD of schedule or low priority and others may need the GPUs more; 'balanced' "
    "otherwise. focus_gpus = how many GPUs you truly need THIS phase to stay on track (an integer "
    "from 1 up to your max useful GPUs); concentrate on fewer GPUs when behind so you actually "
    "finish, rather than spreading thin. Justify in one sentence referring to your deadline, "
    "tier, and the contention."
)


# --------------------------------------------------------------------------- #
#  State -> cache key + prompt (kept qualitative so a cached answer is valid    #
#  for every concrete state that maps to the same discretised bucket).          #
# --------------------------------------------------------------------------- #
def state_key(ctx: dict) -> str:
    return (f"{ctx['phase']}|cap{ctx['capacity']}|{ctx['deadline']}|"
            f"{ctx['contention']}|{ctx['tier']}")


def _user_prompt(ctx: dict) -> str:
    return (
        f"Your job is in its '{ctx['phase']}' phase and can usefully use up to "
        f"{ctx['capacity']} GPU(s) (your predicted workload for the next few minutes). "
        f"Priority tier: {ctx['tier']}. Deadline status: you are {ctx['deadline']} schedule. "
        f"Cluster contention right now: {ctx['contention']}. "
        f"Choose your stance and focus_gpus (1..{ctx['capacity']})."
    )


# --------------------------------------------------------------------------- #
#  Robust parse + coerce/clip (mirror pins/forecast/llm_facts)                  #
# --------------------------------------------------------------------------- #
def _parse(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _coerce(obj: dict, capacity: int) -> dict:
    """Snap stance to an allowed value and clip focus_gpus into [1, capacity]."""
    stance = str(obj.get("stance", "")).strip().lower()
    if stance not in STANCES:
        stance = "balanced"
    try:
        focus = int(round(float(obj.get("focus_gpus", capacity))))
    except Exception:
        focus = capacity
    focus = max(1, min(capacity, focus))
    why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
    return {"stance": stance, "focus_gpus": focus, "justification": why}


def _rule_strategy(ctx: dict) -> dict:
    """Deterministic fallback (also the Ollama-down path): behind+contended/prod -> fight and
    concentrate; ahead/low-priority -> concede; else balanced. Mirrors the SYSTEM guidance."""
    cap = ctx["capacity"]
    behind = ctx["deadline"] == "behind"
    ahead = ctx["deadline"] == "ahead"
    contended = ctx["contention"] == "high"
    prod = ctx["tier"] == "prod"
    if behind and (contended or prod):
        stance, focus = "aggressive", max(1, cap - (0 if contended else 0))   # fight for all useful
    elif ahead or (not prod and contended):
        stance, focus = "concede", max(1, cap // 2)
    else:
        stance, focus = "balanced", cap
    return {"stance": stance, "focus_gpus": focus,
            "justification": f"rule: {stance} ({ctx['deadline']}, {ctx['contention']} contention, {ctx['tier']})",
            "_source": "rule"}


# --------------------------------------------------------------------------- #
#  Cache (one query per distinct discretised state)                            #
# --------------------------------------------------------------------------- #
def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, CACHE_PATH)


def llm_strategy(ctx: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
                 host: str = HOST, cache: dict | None = None) -> dict:
    """Return `{stance, focus_gpus, justification, _source}` for a discretised state, cached."""
    cache = load_cache() if cache is None else cache
    key = f"{state_key(ctx)}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    strat = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 150},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": _user_prompt(ctx)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None:
                strat = _coerce(obj, ctx["capacity"])
                strat["_source"] = f"llm:{model}"
        except Exception as e:                       # Ollama down / network / bad JSON
            print(f"  ! llm_agent fallback for [{state_key(ctx)}]: {type(e).__name__}: {e}")
    if strat is None:
        strat = _rule_strategy(ctx)

    cache[key] = strat
    return strat


# --------------------------------------------------------------------------- #
#  Deterministic application: strategy -> final marginal-value curve            #
# --------------------------------------------------------------------------- #
def apply_strategy(baseline: list[float], stance: str, focus_gpus: int) -> list[float]:
    """Turn a (stance, focus) decision into a bid curve over the CALIBRATED baseline.

    Stance scales the whole curve by a fixed constant; focus CONCENTRATES — GPUs up to `focus`
    keep full (scaled) value, GPUs beyond it collapse to a near-zero weight, so the job
    voluntarily caps itself and yields surplus GPUs to others (the EDF-like lever from Exp 9).
    Pure code: the LLM supplied no number here, only the categorical knobs."""
    mult = STANCE_MULT.get(stance, 1.0)
    focus = max(1, min(len(baseline), int(focus_gpus)))
    return [round(v * mult * (1.0 if k < focus else BEYOND_FOCUS), 4)
            for k, v in enumerate(baseline)]


# --------------------------------------------------------------------------- #
#  Priority setting for the COMMITTED auction (Exp 12)                          #
#                                                                              #
#  The Exp-11 winner serialises jobs by a FROZEN priority. Here the LLM SETS    #
#  that priority once, on arrival, from the job's intrinsic profile — but as an #
#  ORDINAL CLASS, never a number. Code maps class -> weight and does all the    #
#  ordering. This is the interpretable, AI-agent version of committed-auction:  #
#  every serialisation order comes with a justification (the edge vs RL).       #
# --------------------------------------------------------------------------- #
PRIORITY_CLASSES = ["critical", "high", "normal", "low"]
PRIORITY_WEIGHT = {"critical": 4.0, "high": 3.0, "normal": 2.0, "low": 1.0}

SYSTEM_PRIORITY = (
    "You are the admission controller for a shared GPU cluster. Given ONE job's intrinsic "
    "profile, assign it a scheduling PRIORITY CLASS that decides whose deadline is protected "
    "first when GPUs are scarce. You output NO numbers (no GB, %, or value).\n"
    "Respond with ONLY this JSON, using EXACTLY the allowed values:\n"
    '{"priority": "critical|high|normal|low", "justification": "<one short sentence>"}\n'
    "Guidance: 'prod' (production) jobs and jobs with a TIGHT deadline deserve higher priority; "
    "'besteffort' jobs and jobs with a LOOSE deadline can yield. A large job with a tight "
    "deadline is the most at risk. Justify in one sentence referring to tier and deadline."
)


def priority_weight(cls: str) -> float:
    return PRIORITY_WEIGHT.get(cls, 2.0)


def priority_state_key(ctx: dict) -> str:
    return f"{ctx['tier']}|{ctx['deadline']}|{ctx['size']}"


def _priority_prompt(ctx: dict) -> str:
    return (f"Job profile — tier: {ctx['tier']}; deadline: {ctx['deadline']}; "
            f"size: {ctx['size']} workload. Assign its priority class.")


def _rule_priority(ctx: dict) -> dict:
    """Deterministic fallback: tier dominates, tight deadline bumps it up a notch."""
    tight = ctx["deadline"] == "tight"
    if ctx["tier"] == "prod":
        cls = "critical" if tight else "high"
    else:
        cls = "normal" if tight else "low"
    return {"priority": cls, "justification": f"rule: {ctx['tier']}, {ctx['deadline']} deadline",
            "_source": "rule"}


def llm_priority(ctx: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
                 host: str = HOST, cache: dict | None = None) -> dict:
    """Return `{priority, justification, _source}` for a job's profile, cached per state."""
    cache = load_cache() if cache is None else cache
    key = f"prio|{priority_state_key(ctx)}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    out = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[{"role": "system", "content": SYSTEM_PRIORITY},
                          {"role": "user", "content": _priority_prompt(ctx)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None:
                cls = str(obj.get("priority", "")).strip().lower()
                if cls not in PRIORITY_CLASSES:
                    cls = "normal"
                why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
                out = {"priority": cls, "justification": why, "_source": f"llm:{model}"}
        except Exception as e:
            print(f"  ! llm_priority fallback for [{priority_state_key(ctx)}]: "
                  f"{type(e).__name__}: {e}")
    if out is None:
        out = _rule_priority(ctx)

    cache[key] = out
    return out


# --------------------------------------------------------------------------- #
#  Supply-side reservation level (Exp 14 — the SUPPLY agent)                    #
#                                                                              #
#  The two-sided thesis needs a resource/supply agent with an asymmetric job.  #
#  Under RIGID (non-preemptable) incumbents, a late prod job can only run on    #
#  FREE GPUs, so the supply agent decides how much idle HEADROOM to reserve     #
#  for incoming prod load. Categorical (none/light/heavy), never a number —     #
#  code maps level -> GPUs. The hard part is JUDGEMENT, and it has real stakes: #
#  reserve when prod is incoming AND the pool is not already scarce; reserve    #
#  NOTHING when contention is high (idle GPUs starve everyone -> worse SLA) or  #
#  no prod is coming. A wrong call here actively hurts SLA (Exp 14 deterministic#
#  sweep: light reservation helped prodSLA at moderate contention, hurt at high)#
# --------------------------------------------------------------------------- #
RESERVE_LEVELS = ["none", "light", "heavy"]
RESERVE_AMOUNT = {"none": 0, "light": 1, "heavy": 2}     # level -> GPUs held idle (code owns this)

SYSTEM_RESERVE = (
    "You are the SUPPLY agent for a shared GPU cluster where running jobs are RIGID — once a job "
    "starts it CANNOT be preempted, so a late-arriving high-priority ('prod') job can only run on "
    "GPUs that are FREE. Your job: decide how much idle headroom to RESERVE for incoming prod jobs. "
    "You output NO numbers (no GB, count, or %).\n"
    "Respond with ONLY this JSON, using EXACTLY the allowed values:\n"
    '{"reserve": "none|light|heavy", "justification": "<one short sentence>"}\n'
    "Guidance: reserving keeps GPUs IDLE now to protect a future prod job — worth it ONLY when "
    "prod jobs are still INCOMING and the pool is NOT already scarce. Reserve 'none' when no prod "
    "is incoming (nothing to protect) OR when contention is HIGH (idle GPUs would starve the jobs "
    "already waiting and make SLA worse). Reserve 'light' in the moderate case; 'heavy' only when "
    "much prod is incoming and there is slack. Justify in one sentence citing contention and load."
)


def reserve_amount(level: str) -> int:
    return RESERVE_AMOUNT.get(level, 0)


def reserve_state_key(ctx: dict) -> str:
    return f"{ctx['contention']}|{ctx['incoming_prod']}"


def _reserve_prompt(ctx: dict) -> str:
    return (f"Cluster contention right now: {ctx['contention']}. "
            f"High-priority (prod) jobs still INCOMING (not yet started): {ctx['incoming_prod']}. "
            f"How much headroom should you reserve?")


def _rule_reserve(ctx: dict) -> dict:
    """Deterministic fallback encoding the Exp-14 finding: reserve only at moderate contention with
    prod incoming; never when scarce or when nothing is coming."""
    inc, con = ctx["incoming_prod"], ctx["contention"]
    if inc == "none" or con == "scarce":
        lvl = "none"
    elif con == "moderate":
        lvl = "light"
    else:                                                # ample slack + prod incoming
        lvl = "light" if inc == "few" else "heavy"
    return {"reserve": lvl, "justification": f"rule: {con} contention, {inc} prod incoming",
            "_source": "rule"}


def llm_reserve(ctx: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
                host: str = HOST, cache: dict | None = None) -> dict:
    """Return `{reserve, justification, _source}` for a discretised supply-side state, cached."""
    cache = load_cache() if cache is None else cache
    key = f"resv|{reserve_state_key(ctx)}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    out = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[{"role": "system", "content": SYSTEM_RESERVE},
                          {"role": "user", "content": _reserve_prompt(ctx)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None:
                lvl = str(obj.get("reserve", "")).strip().lower()
                if lvl not in RESERVE_LEVELS:
                    lvl = "none"
                why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
                out = {"reserve": lvl, "justification": why, "_source": f"llm:{model}"}
        except Exception as e:
            print(f"  ! llm_reserve fallback for [{reserve_state_key(ctx)}]: "
                  f"{type(e).__name__}: {e}")
    if out is None:
        out = _rule_reserve(ctx)

    cache[key] = out
    return out


# --------------------------------------------------------------------------- #
#  Demand-side safety MARGIN from forecast uncertainty (Exp 17)                 #
#                                                                              #
#  Stage-1 now emits a quantile UNCERTAINTY per job (pins/forecast/            #
#  model_quantile). Exp 16 showed a margin sized by that uncertainty is        #
#  insurance whose value GROWS with the demand tail — but only where there is  #
#  SPARE capacity (a blanket margin backfires by over-subscribing). Here the   #
#  LLM demand agent makes that JUDGEMENT: given its uncertainty + deadline +    #
#  contention, decide HOW MUCH to hedge (none/some/heavy) — categorical, never  #
#  a GPU count. Code maps the hedge to an effective uncertainty fed to          #
#  predictor.marginal_values, which owns the number. The justification makes    #
#  the hedge auditable (the edge vs RL), and reasoning encodes the Exp-16       #
#  regime lesson: hedge only when uncertain AND at-risk AND capacity is spare.  #
# --------------------------------------------------------------------------- #
MARGIN_HEDGES = ["none", "some", "heavy"]
UNCERTAINTY_BUCKETS = ["low", "medium", "high"]
SPIKE_RISK_BUCKETS = ["low", "medium", "high"]

SYSTEM_MARGIN = (
    "You are the demand agent for ONE HPC training job. A forecaster gives you (1) how UNCERTAIN your "
    "near-future GPU demand is and (2) your SPIKE RISK — how LARGE a demand spike could plausibly be "
    "(the heaviness of your demand's upper tail). Decide how much SAFETY MARGIN (extra GPUs beyond "
    "your forecast) to bid for, so a spike does not make you miss your deadline. You output NO "
    "numbers (no GB, count, or %).\n"
    "Respond with ONLY this JSON, using EXACTLY the allowed values:\n"
    '{"hedge": "none|some|heavy", "justification": "<one short sentence>"}\n'
    "Guidance: a margin is INSURANCE against a spike. Hedge 'none' when uncertainty is LOW (demand is "
    "predictable) or you are AHEAD of schedule (slack absorbs any spike). When your SPIKE RISK is "
    "HIGH a spike would likely make you miss your deadline, so hedge to protect it EVEN IF contention "
    "is high (use 'heavy' if you are also behind schedule, else 'some'). When spike risk is only mild, "
    "the contention rule applies: hedge 'none' under HIGH contention (extra GPUs would be wasted and "
    "starve others) and 'some' when there is spare capacity. Justify in one sentence citing your "
    "spike risk, deadline, and the contention."
)


def uncertainty_bucket(u: float) -> str:
    """Discretise the Stage-1 uncertainty scalar into low/medium/high (calibrated u: median ~0.16,
    max ~0.9), so a cached LLM answer is valid for every job in the same bucket."""
    return "low" if u < 0.1 else "high" if u >= 0.33 else "medium"


def spike_risk_bucket(severity: float) -> str:
    """Discretise SPIKE RISK = the plausible relative over-run of demand above the forecast (the
    upper-tail magnitude the forecaster sees, e.g. (P90−P50)/P50). Unlike `uncertainty` (interval
    WIDTH), this is the SEVERITY of a miss — the signal Exp 17 found was missing: under a heavy tail
    a job must hedge even when contended, because an unhedged spike means a deadline violation."""
    return "low" if severity < 0.15 else "high" if severity >= 0.35 else "medium"


def margin_uncertainty(hedge: str, u: float, scale: int) -> float:
    """Map the LLM's categorical hedge to the EFFECTIVE uncertainty fed to marginal_values (which
    turns it into round(u*scale) margin GPUs). none -> no margin; some -> the forecast uncertainty
    (= the Exp-16 'uncertainty-sized' policy); heavy -> one extra GPU's worth. Code owns the count."""
    if hedge == "none":
        return 0.0
    if hedge == "heavy":
        return min(1.0, u + 1.0 / max(scale, 1))
    return u                                              # "some"


def margin_state_key(ctx: dict) -> str:
    return (f"{ctx['uncertainty']}|{ctx['spike_risk']}|{ctx['deadline']}|"
            f"{ctx['contention']}|{ctx['tier']}")


def _margin_prompt(ctx: dict) -> str:
    return (f"Your forecast uncertainty is {ctx['uncertainty']} and your spike risk (how large a "
            f"demand spike could be) is {ctx['spike_risk']}. Deadline status: you are "
            f"{ctx['deadline']} schedule. Priority tier: {ctx['tier']}. Cluster contention right "
            f"now: {ctx['contention']}. How much safety margin should you hedge for?")


def _rule_margin(ctx: dict) -> dict:
    """Deterministic fallback (Exp 17 fix): spike RISK can override the contention-gate. Never hedge
    when demand is predictable or there is deadline slack; under a HIGH spike risk hedge to protect
    the deadline even when contended; otherwise apply the Exp-16 lesson (hedge only with spare
    capacity)."""
    unc, sr, db, con = ctx["uncertainty"], ctx["spike_risk"], ctx["deadline"], ctx["contention"]
    if unc == "low" or db == "ahead":
        h = "none"                                  # predictable, or slack absorbs any spike
    elif sr == "high":
        h = "heavy" if db == "behind" else "some"   # severe over-run: protect deadline even if contended
    elif con == "high":
        h = "none"                                  # mild risk + no spare capacity -> don't waste
    else:
        h = "heavy" if (db == "behind" and unc == "high") else "some"
    return {"hedge": h,
            "justification": f"rule: {unc} uncertainty, {sr} spike-risk, {db}, {con} contention",
            "_source": "rule"}


def llm_margin(ctx: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
               host: str = HOST, cache: dict | None = None) -> dict:
    """Return `{hedge, justification, _source}` for a discretised demand-side state, cached."""
    cache = load_cache() if cache is None else cache
    key = f"marg|{margin_state_key(ctx)}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    out = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 120},
                messages=[{"role": "system", "content": SYSTEM_MARGIN},
                          {"role": "user", "content": _margin_prompt(ctx)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None:
                h = str(obj.get("hedge", "")).strip().lower()
                if h not in MARGIN_HEDGES:
                    h = "some"
                why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
                out = {"hedge": h, "justification": why, "_source": f"llm:{model}"}
        except Exception as e:
            print(f"  ! llm_margin fallback for [{margin_state_key(ctx)}]: "
                  f"{type(e).__name__}: {e}")
    if out is None:
        out = _rule_margin(ctx)

    cache[key] = out
    return out


if __name__ == "__main__":
    import argparse
    from pins.predictor import marginal_values

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="use the rule fallback only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # A few hand-built contexts spanning the decision space.
    probes = [
        {"phase": "train", "capacity": 8, "deadline": "behind", "contention": "high", "tier": "prod"},
        {"phase": "train", "capacity": 8, "deadline": "ahead", "contention": "high", "tier": "besteffort"},
        {"phase": "eval", "capacity": 2, "deadline": "ontrack", "contention": "low", "tier": "prod"},
        {"phase": "preprocess", "capacity": 1, "deadline": "behind", "contention": "low", "tier": "besteffort"},
    ]
    cache = load_cache()
    src = "rule" if args.no_llm else args.model
    print(f"strategy source: {src}\n")
    for ctx in probes:
        s = llm_strategy(ctx, use_llm=not args.no_llm, model=args.model, cache=cache)
        base = marginal_values(ctx["phase"], urgency=1.5)
        curve = apply_strategy(base, s["stance"], s["focus_gpus"])
        print(f"[{state_key(ctx)}]")
        print(f"   -> {s['stance']:10s} focus={s['focus_gpus']} ({s.get('_source','?')})")
        print(f"      why: {s['justification']}")
        print(f"      baseline={base}")
        print(f"      bid     ={curve}\n")

    print("=== committed-auction priority (Exp 12) ===\n")
    prio_probes = [
        {"tier": "prod", "deadline": "tight", "size": "large"},
        {"tier": "prod", "deadline": "loose", "size": "small"},
        {"tier": "besteffort", "deadline": "tight", "size": "large"},
        {"tier": "besteffort", "deadline": "loose", "size": "small"},
    ]
    for ctx in prio_probes:
        p = llm_priority(ctx, use_llm=not args.no_llm, model=args.model, cache=cache)
        print(f"[{priority_state_key(ctx)}] -> {p['priority']:8s} "
              f"(w={priority_weight(p['priority'])}, {p.get('_source','?')})")
        print(f"      why: {p['justification']}\n")

    print("=== demand-side safety margin from uncertainty (Exp 17) ===\n")
    margin_probes = [
        {"uncertainty": "high", "spike_risk": "high", "deadline": "behind", "contention": "high", "tier": "prod"},
        {"uncertainty": "high", "spike_risk": "low", "deadline": "behind", "contention": "high", "tier": "prod"},
        {"uncertainty": "high", "spike_risk": "high", "deadline": "behind", "contention": "low", "tier": "prod"},
        {"uncertainty": "low", "spike_risk": "low", "deadline": "behind", "contention": "low", "tier": "prod"},
        {"uncertainty": "medium", "spike_risk": "medium", "deadline": "ahead", "contention": "low", "tier": "besteffort"},
    ]
    for ctx in margin_probes:
        m = llm_margin(ctx, use_llm=not args.no_llm, model=args.model, cache=cache)
        print(f"[{margin_state_key(ctx)}] -> hedge={m['hedge']:6s} ({m.get('_source','?')})")
        print(f"      why: {m['justification']}\n")
    save_cache(cache)
