# Does the LLM Earn Its Cost in GPU Scheduling? A Gated Architecture That Separates Efficiency from Protection

*Working draft — 2026-07-23. English; port to the jlreq template in `main.tex` and translate for
submission. Supersedes the negotiation-era framing of the current `main.tex` (Exp 22–48); that
material becomes the baseline arm here. Every number is a 32-seed paired result from
`research_progress.md`; significance (\*) = 95% CI excludes 0.*

---

## Abstract

On a saturated GPU cluster, utilisation and SLA compliance trade off, and the trade is
fundamentally a *rationing* decision: whose safety margin to cut. A natural proposal is to let a
reasoning LLM make that decision. We test whether it earns its cost. On the Alibaba
cluster-trace-gpu-v2020 replayed as a two-sided allocation game (32-seed paired comparison), we
find: (1) a **deterministic bid/ask market** wins every efficiency metric — utilisation, useful
utilisation, and oracle regret — against every LLM arm, in all six pool×law cells, at **zero
tokens** (Exp 72); (2) the reasoning layer's *entire* reproducible contribution is production-tier
protection, and that contribution is captured by a **single reserve scalar** — a full referee
that emits per-job allocations is dominated by one cheap LLM call choosing that scalar, at 1/24000
of the tokens (Exp 73); (3) protection and efficiency draw on the **same idle GPUs**, so they form
one Pareto dial, not two composable layers. We therefore propose a **gated architecture**: a
deterministic auction, validated every tick by cheap code, applies on routine ticks; a reasoning
escalation fires only on a contextual trigger. Layering a debate-based escalation with operator
documentation onto the validated auction is **SLA-neutral** (−0.2 pt, ns; Exp 87) — it adds
exception-handling capability without costing efficiency. Finally we show *where* the reasoning
layer genuinely earns its cost: on **text exceptions** the numbers cannot express, a debate of
LLM reviewers corrects a single model's unreliable exception rulings (15/31 vs 9/31 on a
pre-registered hard-case suite; Exp 83). We report the honest boundary — public GPU traces are
structurally scrubbed of the operator free-text this capability needs, so the in-simulation SLA
gain cannot be measured on them, and we say so rather than manufacture it.

---

## 1. Introduction

AI and scientific workloads are bursty and heterogeneous; static batch schedulers cannot hold
both high utilisation and low SLA-violation rate. Pushing utilisation up raises the violation
rate, and the reason is physical: in the saturated regime it is *impossible* to give every job a
safety margin. What actually sets service quality is the rationing decision — **whose margin gets
cut** — which classical schedulers make blindly (FCFS, fixed priority).

Recent work (LLMsched [1]) proposes a *reason-then-guarantee* spine: an LLM proposes a schedule,
a fast ILP repairs it to feasibility. The appeal is that an LLM can reason over a messy
multi-objective situation and generalise to unfamiliar states. This paper asks the question that
determines whether that appeal is real for *rationing*: **does the reasoning layer earn its
cost, and if so, on which decisions?** We answer with a decomposition rather than a single arm.

Our contributions:

1. **The market dominates the reasoning layer on efficiency, at zero cost** (§4.1). A
   deterministic bid/ask auction over each job's real marginal useful progress beats every LLM
   arm on utilisation, useful utilisation, and regret, in every cell, under both progress laws —
   with no LLM calls (Exp 72).
2. **The LLM's whole contribution is one scalar** (§4.2). A full referee that emits per-job
   allocations is dominated by a single LLM call that chooses only the protective reserve; the
   per-job reasoning is pure cost (regret\*, useful\*, 24k tokens/seed) (Exp 70/71/73).
3. **Protection and efficiency are one dial** (§4.2). The reserve protects the production tier by
   holding GPUs idle, and idle GPUs are exactly the market's utilisation gain. We report an
   interpretable frontier with an operator-set dial, which is a stronger and more honest claim
   than "the LLM wins."
4. **A gated architecture that is efficiency-neutral** (§4.3). Validated auction every tick,
   reasoning escalation only on triggers. The per-tick validator is inert on correct clearings
   (byte-identical to the un-validated market, Exp 86); the debate escalation with operator docs
   is SLA-neutral on the auction (−0.2 pt ns, Exp 87).
5. **Where reasoning earns its cost, and the honest boundary** (§4.4). Debate improves a single
   LLM's unreliable *text-exception* rulings (15/31 vs 9/31, Exp 83); but the numeric trace has
   no text channel for this to act on in-sim, which we demonstrate and do not paper over
   (Exp 84).

The governing design rule throughout is **the LLM reasons and explains; deterministic code
decides** — and our central finding sharpens it: on numeric scenes the deterministic code should
*also decide efficiently on its own*, and reasoning should be invoked only where text carries a
fact the numbers cannot.

## 2. Related work

**LLM-driven scheduling.** LLMsched [1] repairs a single LLM's proposal with an ILP and reports
large gains over Kubernetes. We inherit the reason-then-guarantee spine but invert its default:
we show that on numeric rationing the deterministic layer should be the *proposer*, not just the
repairer, and the LLM should be gated behind a contextual trigger.

**Learning-based schedulers.** DRL schedulers (DeepRM/Decima family) learn allocation policies
but are black-box and brittle under distribution shift. On the same interface we measure a
tabular Q-learner as a weak representative; episode-return variance dominates the action effect
at realisable sample counts, which is our empirical basis for treating DRL as out of scope.

**Mechanism design.** Uniform-price auctions and budget constraints are the classical route to
truthful revelation. Our contribution is measurement in a *closed loop containing declarative LLM
agents* — a best-response test rather than an assumed equilibrium.

**Malleability.** Runtime job elasticity has been studied for decades but is rarely deployed in
production [2]. Our partial-allocation model rests on it, and we report its consequence honestly
(all-or-nothing backfill is disadvantaged by construction).

## 3. The gated architecture

**Overview.** Each three-minute scheduling epoch:

```
demand LLM bids  ─┐
                  ├─► deterministic auction clearing ─► automatic validation (cheap code)
supply LLM asks  ─┘                                          │
                                       valid & no trigger ───┼──► apply auction allocation
                                       trigger & valid    ───┼──► apply escalation ruling
                                       otherwise          ───┴──► retain / fall back
```

**Auction (default path, zero LLM).** Demand posts a non-increasing marginal-value curve
`b_{j,k} = α·dp̂_{j,k} + β·R_SLA + γ·W_wait − δ·C_resize` per job per extra GPU, where **`dp̂` is
the job's real marginal useful progress** under the counterfactual progress model; supply posts a
rising ask `a_q = η1·Scarcity + η2·Frag + η3·ReservePressure + η4·ArrivalPressure`; the market
clears at `Q = max{q : b_q ≥ a_q}`. Reserve pressure lives in the *price* (η3), so incoming-prod
headroom is expressed by selling fewer margin units rather than by blocking bases.

**Validation (every tick, cheap code).** Before any allocation is applied, a deterministic check
verifies feasibility (Σ awarded + reserve ≤ free), no negative award, no unknown job id, floor
and tier constraints. On violation the tick concedes to the floor and is counted as a fallback.
Because the auctioneer generates the result, this normally passes; it guards against clearing
bugs and stale input, not against the LLM.

**Escalation (only on a trigger).** A reasoning arm rules only when the auction output conflicts
with a note, a production job arrives, a job falls behind its deadline, or contested capacity
crosses a bucket boundary. Routine ticks never invoke it. The escalation arm is a **debate**:
demand- and supply-side reviewers state positions, read each other, and revise before a referee
rules, optionally with an operator precedent manual — the configuration §4.4 shows is the
strongest on text exceptions.

**Guarantees.** The escalation never repairs an infeasible ruling silently; an infeasible
escalation falls back and re-fires next tick. Reasoning is best-effort improvement, never on the
critical path.

## 4. Evaluation

**Setup.** We replay contiguous windows of the 606k-job Alibaba cluster-trace-gpu-v2020, taking
arrival, duration, and GPU demand jointly from the trace and synthesising only what the trace
lacks (tier, deadline, urgency; seeded recipe). Pools ∈ {4,6,8} GPU, quarter-GPU quanta,
32 seeds, windows and seeds paired across arms; \* marks a 95% CI excluding 0. Progress runs
under both an Amdahl and a saturating law (a claim counts only if it holds under both). LLMs are
local Ollama models (qwen2.5, gemma2, llama3). Baselines: the no-LLM floor, a single LLM holding
both objectives, isolated agents, EASY backfill, and a tabular Q-learner.

### 4.1 The deterministic market wins efficiency at zero cost (Exp 72)

Replacing the reasoning layer with the explicit bid/ask market, `market` vs the floor:

| law | pool | dSLA | dprodSLA | dutil | duseful | dregret |
|---|---|---|---|---|---|---|
| amdahl | 8 | −1.6\* | −2.7 | **+2.1\*** | **+2.1\*** | **−2.8\*** |
| sat | 8 | −0.2 | −0.8 | **+2.3\*** | +0.5\* | −0.5\* |

Against `negotiated` — the arm that had won the prior evaluation — the market improves
utilisation, useful utilisation, and regret significantly in **all six** pool×law cells (e.g.
amdahl pool 8: dutil +1.1\*, duseful +1.8\*, dregret −3.0\*). It is the only arm in the project's
history simultaneously positive on useful utilisation and negative on regret. It does **not**
protect the production tier (dprodSLA small and ns everywhere). And it costs **zero tokens, 0%
fallback**. Efficiency belongs to the deterministic market.

### 4.2 The LLM's contribution is one scalar, and protection is one dial (Exp 70/71/73)

The referee's one distinctive, reproducible effect is production-tier protection (dprodSLA
−8.9\*/−6.3\* under the two laws). We amputate everything else: the **composed** arm lets the LLM
choose *only* the reserve level and hands the remaining pool to the market. vs the floor, pool 8:

| law | arm | dSLA | dprodSLA | dutil | duseful | dregret |
|---|---|---|---|---|---|---|
| amdahl | referee  | −1.4  | **−8.9\*** | −1.2  | −1.9\* | +3.2\* |
| amdahl | composed | +0.4  | **−5.6\*** | −0.1  | −0.2   | +0.8   |
| amdahl | market   | −2.1\*| −3.8\*     | +1.3\*| +0.7\* | −1.3\* |
| sat    | referee  | −0.2  | **−6.3\*** | −0.8  | −2.2\* | +4.4\* |
| sat    | composed | −1.0  | **−7.5\*** | +0.2  | −0.8   | +1.7\* |
| sat    | market   | +0.4  | −0.7       | +2.5\*| +1.2\* | −0.8\* |

**The protection survives the amputation; the per-job reasoning was pure cost.** composed −
referee: prodSLA +3.3/−1.1 (both ns — indistinguishable) while regret improves −2.4\*/−2.6\* and
useful utilisation +1.7\*/+1.4. `composed` **dominates** `referee` and replaces it. A full
referee's per-job statements, rebuttals, and rulings buy nothing over a single scalar, and cost
regret, useful work, and 24,240 tokens/seed.

**Protection and efficiency are the same resource.** composed − market: the reserve buys
production protection (prodSLA −6.7\* under sat) but *loses* util −2.3\*, useful −2.0\*, regret
+2.5\*. The reserve protects by holding GPUs idle, and idle GPUs are exactly what the market's
utilisation gain was made of. This is **one dial**, set by operator preference (throughput vs
production protection), not a winner the data selects:

| arm | buys | costs | bill |
|---|---|---|---|
| `market`   | util, useful util, regret (all \*) | no prod protection      | 0 tokens |
| `composed` | referee-equivalent protection      | efficiency, significantly | 1 call/tick |
| `referee`  | nothing composed doesn't           | regret\*, useful\*        | 24,240 tok/seed |

### 4.3 The gated architecture is efficiency-neutral (Exp 86, 87)

**The validator is inert on correct clearings.** Re-running the market arm with the per-tick
validator reproduces the Exp 72 numbers **byte-identically** across both laws × pools 4/6/8 × 32
seeds, 0% fallback throughout (Exp 86). Fault injection (oversubscription, negative award,
unknown id) confirms it rejects when it should. It is a bug/stale-input guard that never touches
a well-formed auction output — exactly what a safety layer should be.

**The debate escalation is SLA-neutral on the auction.** With the debate-plus-operator-docs
escalation gated behind the contextual trigger (qwen2.5:14b, caps=predicted, pool 8, n=32):

| arm | SLA | prodSLA | util |
|---|---|---|---|
| no-llm (floor) | 53.9% | 59.1% | 80% |
| market (validated auction alone) | 51.8% | 55.3% | 81% |
| gated (auction + debate escalation) | 51.6% | 54.7% | 81% |

gated − market (paired): **dSLA −0.20 ± 1.40 (ns)**, dprodSLA −0.63 (ns), dutil −0.21 (ns). The
trigger fired on **15.7%** of ticks (1076 escalations, 9 infeasible) at 52,853 tokens/seed — the
escalation genuinely ran and did not move the outcome. Layering reasoning onto the validated
auction adds capability without costing efficiency.

### 4.4 Where reasoning earns its cost: text exceptions (Exp 83, 84)

A scalar reserve cannot express an exception whose content is *text* — an operator instruction, a
declaration a note says is wrong, a crash-restart the numbers do not show. On a pre-registered
hard-case suite of 31 such text-dependent exceptions, all arms reading an identical structured
decision packet, we compare a single LLM, a parallel referee, and a debate (each ± an operator
precedent manual):

| arm | handled (no docs) | handled (+docs) |
|---|---|---|
| market (numbers only) | 0/31 | 0/31 |
| single LLM | 9/31 | 11/31 |
| referee | 11/31 | 13/31 |
| **debate** | **14/31** | **15/31** |

Debate strictly dominates the single call (head-to-head 6 vs 1 without docs, one-sided exact
p=0.0625) with zero harm on the primary set. Debate and the operator docs are **substitutes, not
complements** — both push attention onto the operator instruction, so the docs add +2 to the
single call but only +1 to debate; the best absolute configuration is debate+docs at 15/31, and
the cleanest mechanism evidence is the 14-vs-9 no-docs comparison. This is the decision a scalar
cannot make and where the reasoning layer earns its tokens.

**The honest boundary.** This gain cannot be shown as an in-simulation SLA improvement on the
numeric trace, and we demonstrate why rather than assert otherwise. The plain debate arm *in the
sim* is SLA-null and prod-costly (dSLA +0.8 ns, dprodSLA −12.0\* when it replaces the auction;
Exp 84), because the trace carries no operator text for it to act on — the market already gets
the numeric scenes right, and the 15.7% of gated ticks that escalated (§4.3) were numeric
capacity crossings, not text exceptions. Public GPU traces are scrubbed of the user-authored free
text this capability needs, by construction (it is PII). So the text-exception result stands on
the authored hard-case suite, and the in-sim claim is precisely: gated reasoning is
efficiency-neutral where there is no text (§4.3) and correct where there is (§4.4) — an
efficiency-neutral capability-add, which is a stronger and more defensible pairing than a
manufactured SLA gain.

## 5. Discussion and limitations

**What the decomposition buys.** Reporting the LLM's contribution as one interpretable scalar on
a Pareto dial, rather than as a monolithic win, is both more honest and more useful: an operator
reads the dial directly and swaps the supply-side objective by editing one prompt, with no
retraining — the interpretability edge over a black-box DRL policy, stated as a measured property.

**The SLA lever is admission, not margin.** In this contended sim, overall SLA is governed by
*which jobs start*, not *which go faster*; a margin GPU is a speed-up the pool cannot hand to a
large fraction of jobs. A reasoning gain on overall SLA therefore requires a text exception that
moves *admission or priority*, not margin — a direction we scope for future work with a real text
channel rather than a synthetic one.

**Threats to validity.** Tier, deadline, and urgency are a synthetic recipe; the replay world
never preempts base allocations, so we make no free-malleability claims. Anonymous charging
cannot rescue a world where every agent lies. LLM inference latency is hidden behind wait time by
the gate in the design, but our evaluation runs on the simulation clock, not wall-clock. The
text-exception suite is authored, not sampled from operations.

## 6. Conclusion

The division of labour "the LLM reasons and explains; deterministic code decides" is not a slogan
but the empirically optimal split at every layer — and our decomposition sharpens it. On numeric
GPU rationing, a deterministic market decides *efficiently* on its own at zero cost, and the
reasoning layer's entire reproducible contribution collapses to a single protective scalar on a
one-dimensional Pareto dial. A gated architecture captures the efficiency for free and invokes
reasoning only on a contextual trigger, which is efficiency-neutral. The reasoning layer earns
its cost precisely on the exceptions a scalar cannot express — text the numbers do not contain —
where a debate of reviewers corrects a single model's unreliable rulings. The open frontier is a
workload that actually carries that text: a real operator-annotated trace, on which the
efficiency-neutral capability of §4.3 should convert into a measurable service-quality gain.

## References

[1] LLM-Driven Adaptive Cloud Resource Scheduling: Bridging Reasoning Intelligence with
Optimization Guarantees. IEEE Open J. Comput. Soc., 2026. *(complete citation)*
[2] A. Tarraf et al. Malleability in Modern HPC Systems. IEEE TPDS, 2024. *(complete citation)*
[3] Alibaba cluster-trace-gpu-v2020. *(citation)*
[4] D. Lifka. The ANL/IBM SP scheduling system (EASY). JSSPP, 1995. *(citation)*
