"""Bid-first allocation with context-triggered multi-agent CORRECTION (Exp 77).

The architecture the Exp 68-76 results point to. The market decides; the agents may only
propose a correction to what it decided; the referee may only accept or shrink that
correction. Nobody generates an allocation from scratch.

    A_bid        = market.clear_market(...)          deterministic, the anchor (Exp 72/74)
    demand agent : "does this job need more GPU than its numerical bid suggests?"
    supply agent : "would granting that create a future capacity/fragmentation problem?"
    referee      : "is there enough evidence to override the market?"  -> emits dA only
    A_candidate  = A_bid + dA,   validated, never repaired
    invalid dA   -> execute A_bid                     (fallback = the strongest baseline)

H1: bid-first + context-triggered correction has lower allocation regret than direct LLM
    allocation, while handling exceptional states better than the bare market.
H2: restricting the referee to MODIFYING A_bid rather than GENERATING an allocation reduces
    harmful changes and invalid decisions.

Two design decisions worth stating, because they are where this could go wrong:

1. THE DEVIATION PENALTY IS A BUDGET, NOT A J-GATE. The natural reading of
   J(A) = J_schedule(A) - lambda*||A - A_bid||_1 is to accept a correction only when it
   improves J. That would make the exception channel useless BY CONSTRUCTION: J is computed
   from the modelled quantities, and the whole point of the correction is to carry facts the
   model does not contain ("starved three windows", "the 15:00 demo"). A deterministic J-gate
   would reject exactly the cases the layer exists for. So lambda enters two other ways:
   as an explicit change BUDGET (||dA||_1 <= budget, code-enforced), and as a stated cost in
   the referee's prompt. J - lambda*||dA||_1 is then used to SCORE overrides after the fact —
   which is the measurement H1 asks for, not a gate.

2. EXTRACTION IS TRIGGERED BY TEXT, NOT BY TICKS. A job with no free-text note never triggers
   a demand call. On the trace workload (no notes anywhere) this arm is byte-identical to
   `market` at zero tokens; it only wakes up where there is context to carry.
"""
from __future__ import annotations

from pins.llm_agent import CTX_OPT, DEFAULT_MODEL, HOST, _parse, metered_client

SYSTEM_DEMAND_CTX = (
    "You are the DEMAND-side reviewer for one job in a GPU scheduler. A market has already "
    "allocated the pool using each job's numerical bid. Your ONLY question is:\n"
    "  does this job need more GPU than its numerical bid suggests?\n"
    "Answer YES only when the job's free-text note carries a fact the numbers cannot express "
    "— starvation history, an external or contractual deadline, a standing operator "
    "instruction, a dependency nothing else records. Ordinary urgency, a tight deadline or a "
    "large request are ALREADY in the bid: they are not evidence, and asking for more on their "
    "basis is double-counting.\n"
    "Respond with ONLY this JSON object:\n"
    '{"extra_gpus": <0 or a small positive integer>, "evidence": "<the exact phrase from the '
    'note that justifies it, or empty>"}'
)

SYSTEM_SUPPLY_CTX = (
    "You are the SUPPLY-side reviewer for a GPU cluster. A market has already allocated the "
    "free pool using each job's numerical bid. Your ONLY question is:\n"
    "  would granting these allocations create a future capacity or fragmentation problem?\n"
    "Answer YES only when the notes or the cluster state carry a fact the market's price does "
    "not already contain — an announced arrival, a maintenance window, a job that will need "
    "the whole pool contiguously. Ordinary scarcity is ALREADY in the ask price.\n"
    "Respond with ONLY this JSON object:\n"
    '{"hold_back_gpus": <0 or a small positive integer>, "evidence": "<the phrase that '
    'justifies it, or empty>"}'
)

SYSTEM_REFEREE_DELTA = (
    "You are the REFEREE. A market has produced an allocation from the jobs' numerical bids. "
    "Two reviewers have proposed CHANGES to it, each with evidence. You do NOT produce an "
    "allocation. You decide only how much of the proposed change is justified.\n"
    "Rules, in order:\n"
    "1. THE MARKET IS THE DEFAULT. Override it only where a reviewer cites a fact the bid "
    "could not contain. No evidence, no change.\n"
    "2. EVERY CHANGE COSTS. A job whose allocation moves pays a reconfiguration penalty, so a "
    "change must be worth more than the disruption. Prefer the smallest correction that "
    "addresses the evidence.\n"
    "3. YOU DO NOT DO ARITHMETIC. Say only WHICH job deserves HOW MANY extra GPUs on the "
    "evidence. You never name a donor and never balance the books: the market funds every "
    "accepted correction by revoking the GPUs it valued least, and rejects it outright if it "
    "cannot. Feasibility is not your problem and must not shape your judgement.\n"
    "4. NO DOUBLE-COUNTING. Urgency, tier and deadline are already priced into the bid. "
    "Reject any proposal whose only evidence is one of those.\n"
    "Respond with ONLY this JSON object:\n"
    '{"accept": {"<job_id>": <extra GPUs justified by the evidence>}, '
    '"justification": "<one sentence naming the evidence you accepted or why you accepted '
    'none>"}\n'
    "An empty accept object is the correct answer for an ordinary scene."
)

PROMPT_VERSION = "corr-v1"
DEFAULT_BUDGET = 4          # code-enforced ||dA||_1 ceiling per decision (the lambda knob)


def _has_text(s) -> bool:
    """The trigger. No note, no call — so this arm is free on a workload without notes."""
    return bool((s or "").strip())


def _ask_api(system: str, user: str, model: str, max_tokens: int, temperature: float) -> str:
    """Frontier-model route (submission_plan.md item 4): claude-* -> Anthropic, gpt-* -> OpenAI.

    Same prompts, same cache (the key already includes the model), same output budget as the
    local arms, so an API model drops into every existing arm unchanged. Keys come from the
    environment (ANTHROPIC_API_KEY / OPENAI_API_KEY); a missing key surfaces as the standard
    '! tag fallback' error and the arm scores as 'proposed nothing', it never crashes a run.
    format='json' has no API equivalent; the prompts already demand a JSON object and _parse
    extracts one from surrounding prose."""
    if model.startswith("claude-"):
        import anthropic
        # SDK 1.x dropped the temperature kwarg (sampling knobs left the typed surface).
        # Send it via extra_body ONLY for the boN arm (temperature>0); if the server also
        # rejects it, boN degrades to the API's default sampling — document, don't hide.
        # No output_config json_schema: it requires a schema, and the local arms run
        # schemaless format='json'; a per-provider interface change is exactly the confound
        # Exp 80-82 warned about.
        extra = {"temperature": temperature} if temperature else {}
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}], extra_body=extra or None)
        return r.content[0].text
    import openai
    r = openai.OpenAI().chat.completions.create(
        model=model, max_completion_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content


def _ask(system: str, user: str, model: str, host: str, cache: dict, tag: str,
         num_predict: int = 200, think: bool | None = None,
         temperature: float = 0) -> dict | None:
    """Defaults are Exp 77's exactly, so those tiers keep replaying byte-identically.

    `num_predict`/`think` exist for hybrid reasoners: with thinking on, deepseek-r1 and qwen3
    spend the whole budget in the thinking channel and emit no JSON at all, which would read as
    "the model proposed nothing" when in fact it never got to answer. Callers testing those
    models must raise the budget (see pins/correction_signed.py).

    `temperature` defaults to 0 (every existing tier is unaffected). Exp 88 raises it for the
    self-consistency control ONLY: at temperature 0 the sampler is deterministic, so drawing K
    samples of one prompt returns K identical answers and best-of-N is impossible to build.
    Note the cache key does NOT include temperature -- callers drawing multiple samples must
    vary `tag` per sample, which Exp 88 does.
    """
    key = f"{PROMPT_VERSION}|{tag}|{user}|{model}"
    if key in cache:
        return cache[key]
    try:
        if model.startswith(("claude-", "gpt-")):
            out = _parse(_ask_api(system, user, model, num_predict, temperature))
        else:
            from pins.referee import _HYBRID
            client = metered_client(host)
            kw = {"think": think} if (think is not None and _HYBRID(model)) else {}
            resp = client.chat(model=model, format="json", **kw,
                               options={"temperature": temperature, "num_predict": num_predict,
                                        **CTX_OPT},
                               messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])
            out = _parse(resp.message.content)
    except Exception as e:
        print(f"  ! {tag} fallback: {type(e).__name__}: {e}")
        out = None
    cache[key] = out
    return out


def gather_corrections(jobs: list[dict], supply_note: str, free: int, alloc: dict[str, int],
                       use_llm: bool = True, model: str = DEFAULT_MODEL,
                       cache: dict | None = None, host: str = HOST) -> dict:
    """Run the two reviewers ONLY over the jobs that carry text. Returns proposals, not moves."""
    cache = {} if cache is None else cache
    props: list[dict] = []
    if not use_llm:
        return {"demand": props, "supply": None}
    for j in jobs:
        if not _has_text(j.get("note")):
            continue                      # no context -> no call -> no proposal
        user = (f"job {j['jid']}: tier={j.get('tier')} deadline={j.get('deadline')} "
                f"bid_allocation={alloc.get(j['jid'], 0)} requested={j.get('requested')}\n"
                f"note: \"{j['note'].strip()}\"")
        d = _ask(SYSTEM_DEMAND_CTX, user, model, host, cache, "demand-ctx")
        if d and int(d.get("extra_gpus") or 0) > 0:
            props.append({"jid": j["jid"], "extra": int(d["extra_gpus"]),
                          "evidence": str(d.get("evidence", ""))[:160]})
    sup = None
    if _has_text(supply_note):
        user = (f"free_gpus={free} allocated={sum(alloc.values())} "
                f"jobs={len(jobs)}\nsupply note: \"{supply_note.strip()}\"")
        s = _ask(SYSTEM_SUPPLY_CTX, user, model, host, cache, "supply-ctx")
        if s and int(s.get("hold_back_gpus") or 0) > 0:
            sup = {"hold_back": int(s["hold_back_gpus"]),
                   "evidence": str(s.get("evidence", ""))[:160]}
    return {"demand": props, "supply": sup}


def referee_delta(alloc: dict[str, int], props: dict, free: int, use_llm: bool = True,
                  model: str = DEFAULT_MODEL, cache: dict | None = None,
                  host: str = HOST) -> dict:
    """The referee sees A_bid and the proposals, and returns dA only. No proposals -> no call."""
    cache = {} if cache is None else cache
    if not props["demand"] and not props["supply"]:
        return {"delta": {}, "justification": "no contextual evidence raised", "_source": "skip"}
    if not use_llm:
        return {"delta": {}, "justification": "rule: market stands", "_source": "rule"}
    unsold = free - sum(alloc.values())
    lines = [f"free_gpus: {free}", f"market_allocation: {dict(sorted(alloc.items()))}",
             f"unsold_in_pool: {unsold}", "proposed changes:"]
    for p in props["demand"]:
        lines.append(f"  - demand {p['jid']}: +{p['extra']} GPU — evidence: \"{p['evidence']}\"")
    if props["supply"]:
        lines.append(f"  - supply: hold back {props['supply']['hold_back']} GPU — "
                     f"evidence: \"{props['supply']['evidence']}\"")
    out = _ask(SYSTEM_REFEREE_DELTA, "\n".join(lines), model, host, cache, "referee-delta")
    if not out:
        return {"delta": {}, "justification": "fallback: market stands", "_source": "rule"}
    # The referee's judgement is {jid: extra}; CODE funds it. Exp 77 found 14b could identify
    # the right evidence and then break its own zero-sum rule doing the bookkeeping (it
    # transferred from "_pool" when the prompt said unsold=0) — the Exp 1-7 lesson again.
    # Removing arithmetic from the LLM removes that failure mode by construction.
    accept: dict[str, int] = {}
    for jid, v in (out.get("accept") or {}).items():
        try:
            n = int(v)
        except (TypeError, ValueError):
            continue
        if n > 0:
            accept[str(jid)] = n
    delta = fund(alloc, accept, free, ranking=props.get("ranking"),
                 floors=props.get("floors"))
    return {"delta": delta, "justification": str(out.get("justification", ""))[:200],
            "_source": f"llm:{model}"}


def fund(alloc: dict[str, int], accept: dict[str, int], free: int,
         ranking: list[tuple[float, str]] | None = None,
         floors: dict[str, int] | None = None) -> dict[str, int]:
    """Turn the referee's judgement into a feasible dA, or return {} if it cannot be funded.

    Spend the market's unsold remainder first, then revoke the units the MARKET ITSELF valued
    least — `ranking` is its ascending (value, jid) list, so funding a correction is exactly
    'the beneficiary's bid was higher than we thought'. That keeps the mechanism the decider
    of who pays, and the LLM the decider of who deserves. Never revokes from a beneficiary.

    `floors` is each job's guaranteed MINIMUM (its base). Only the MARGIN above that floor was
    ever contested — a job's minimum is not a fundable resource, so a correction that could
    only be paid for out of someone's base cannot be funded at all."""
    delta: dict[str, int] = {j: n for j, n in accept.items()}
    need = sum(accept.values()) - max(0, free - sum(alloc.values()))
    if need <= 0:
        return delta
    donors = list(ranking or sorted(((1.0, j) for j in alloc), key=lambda t: t[1]))
    for _v, jid in donors:
        if need <= 0:
            break
        if jid in accept:
            continue
        floor = (floors or {}).get(jid, 0)
        can = alloc.get(jid, 0) + delta.get(jid, 0) - floor      # margin only, never the base
        take = min(can, need)
        if take > 0:
            delta[jid] = delta.get(jid, 0) - take
            need -= take
    return {} if need > 0 else delta          # cannot fund -> no correction, market stands


def validate_delta(alloc: dict[str, int], delta: dict[str, int], free: int,
                   caps: dict[str, int] | None = None,
                   budget: int = DEFAULT_BUDGET,
                   floors: dict[str, int] | None = None) -> list[str]:
    """Non-repairing check on A_bid + dA. Any violation drops the WHOLE correction, and the
    market's allocation executes — so the failure mode of this layer is 'no worse than the
    best deterministic baseline', not 'fall back to the floor'."""
    bad = []
    for jid in delta:
        if jid not in alloc:
            bad.append(f"unknown job {jid}")
    cand = {j: alloc.get(j, 0) + delta.get(j, 0) for j in set(alloc) | set(delta)}
    if any(v < 0 for v in cand.values()):
        bad.append("negative allocation")
    if sum(cand.values()) > free:
        bad.append(f"total {sum(cand.values())} exceeds free pool {free}")
    if caps:
        for jid, v in cand.items():
            if jid in caps and v > caps[jid]:
                bad.append(f"{jid}: {v} exceeds its usable ceiling {caps[jid]}")
    for jid, fl in (floors or {}).items():          # the base is guaranteed, never fundable
        if cand.get(jid, 0) < fl:
            bad.append(f"{jid}: {cand.get(jid, 0)} below its guaranteed base {fl}")
    l1 = sum(abs(v) for v in delta.values())
    if l1 > budget:
        bad.append(f"change budget: ||dA||_1 = {l1} > {budget}")
    return bad


def apply_correction(alloc: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    return {j: alloc.get(j, 0) + delta.get(j, 0) for j in set(alloc) | set(delta)}
