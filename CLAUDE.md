# Research discussion log — LLM+ILP scheduling → two-LLM negotiated scheduler

> Captures a design discussion (2026-06-19) sparked by reading
> *"LLM-Driven Adaptive Cloud Resource Scheduling: Bridging Reasoning Intelligence with
> Optimization Guarantees"* (IEEE OJ-CS 2026, doc #11409427). Full paper notes live in
> [`LLMsched_restore.md`](./LLMsched_restore.md). This file is the *idea* that grew out of it
> — a candidate direction for the PINS thesis.

---

## 1. The reference paper, in one breath

A cloud scheduler with a **reason-then-guarantee** spine:
- An **LLM proposes** scheduling decisions by reasoning over cluster state written as text.
- A **lightweight ILP** (Gurobi, 50 ms budget) then **repairs** the proposal so it provably
  satisfies hard constraints (capacity, precedence, affinity) — the "guarantee."
- To stay cheap, the LLM fires for only ~3.2% of decisions; ~97% are served from a strategy
  cache in ~4 ms.
- Results vs Kubernetes: −23.7% JCT, +18.4% utilization, −31.2% SLA violations; <8%
  degradation under distribution shift (RL baselines drop 13–17%).

It **does** consume task metadata (resource demand, priority/weight, deadlines, DAG
structure, data-transfer volume, affinity) — both in the LLM prompt and as ILP
variables/constraints. What it does *not* assume is an accurate duration estimate (that's
the stochastic part).

## 2. Roles of LLM vs ILP (the core mental model)

The whole design rests on a complementarity: **the LLM is smart but unreliable; the ILP is
reliable but not smart.** Each covers the other's weakness.

- **LLM = creative proposer.** Reasons over the messy multi-objective situation, generates
  several candidate plans, generalizes to unfamiliar states. *Flaw:* hallucinates illegal
  plans (LLM-only ablation broke hard constraints ~78% of the time).
- **ILP = rule-enforcing fixer.** Takes the candidate and does the **minimum edit** needed to
  make it feasible and low-cost, via a soft penalty `λ` that rewards keeping the LLM's
  choices. *Flaw:* no judgment, doesn't scale from scratch (ILP-only gave slow, naive plans).

Ablation proves both are load-bearing: remove ILP → SLA violations jump 11.9% → ~48%; remove
LLM → slow, semantically naive schedules.

**One-liner:** *the LLM decides what looks smart; the ILP decides what's actually allowed
and optimal.* This is exactly the PINS rule — **"the LLM reasons/explains; deterministic
code decides."** Here the deterministic decider is the ILP; in PINS it's the auction.

### Working analogy (refined)
- **LLM = brilliant scientist** — floods the room with creative proposals, some impractical.
- **ILP = strict structural engineer / code inspector** — takes the favorite proposal and
  does minimal surgery to make it provably safe and well-built, but invents nothing.
- Key asymmetry to preserve: the scientist supplies the **intelligence**; the engineer
  supplies the **guarantees**. The engineer is *reliable, not smart*. (Correction to an
  earlier framing: the ILP does **not** "understand real-world demand" — the judgment about
  what's a *good* schedule lives in the LLM and in the human-set objective weights.)

## 3. The proposed idea — two-LLM negotiated scheduler

Split the single centralized LLM proposer into a **two-sided negotiation**, keep the ILP as
the guarantee layer. This fuses *this paper* (LLM+ILP) with *PINS* (negotiation/auction).

- **Job-queue agent** = the *demand* side. Objective: minimize JCT / hit deadlines / get jobs
  placed. Private info: each job's urgency, true value, flexibility.
- **Resource agent** = the *supply* side. Objective: maximize utilization, protect fairness,
  keep headroom for future arrivals. Private info: real capacity, predicted load, preemption
  cost.
- They **negotiate over MCP**, reach a final plan, submit it to the ILP.

### The decision that makes or breaks it: **asymmetry**
Two agents negotiating is only meaningful if they want different things or know different
things. With symmetric objectives the "discussion" is theater. Define the demand/supply
asymmetry deliberately so the back-and-forth genuinely resolves the urgency-vs-utilization-
vs-fairness tradeoff. A single LLM with both objectives collapses that tension *invisibly*;
two agents make it **explicit and auditable** — that transcript is the interpretability edge
over RL.

### Architecture trap: don't create three deciders
With job-LLM + resource-LLM + ILP, if all three "decide," nothing is guaranteed. Keep the
principle **LLMs reason; a deterministic mechanism decides**:

```
Job-LLM  ─┐
          ├─ negotiate (reasons, proposals, justifications) ─► structured mechanism ─► ILP
Res-LLM  ─┘                                                    (auction clears)       (feasibility/repair)
```

The LLMs output **valuations + justifications + bids/offers**, NOT a final free-form plan. A
**structured mechanism (the PINS auction)** clears it; the ILP reconciles to live state and
guarantees feasibility. Free-form chat as the final decider would throw away the guarantees.

### Three risks the design must answer
1. **Termination/convergence.** Free-form LLM chat can loop or one side sycophantically
   caves. → Use a **bounded protocol** (rounds of offers), not open chat. This is exactly why
   PINS uses an auction: structured negotiation with provable termination.
2. **Incentive compatibility.** A job-agent could exaggerate urgency to grab resources. A
   well-designed (e.g., uniform-price) auction can make honesty optimal; ad-hoc LLM
   negotiation cannot. → Both a risk *and* a differentiator.
3. **Cost vs the obvious baseline.** Two LLMs over multiple rounds multiplies latency/tokens.
   Must beat a **single-LLM-with-both-objectives** baseline. The justification is likely
   interpretability + modularity + incentive properties, and those must be *measured*, not
   asserted.

### What it buys you over the paper
- A **human-readable negotiation transcript** explaining *why* an allocation happened.
- **Modularity:** swap the resource-side policy (greener/fairer operator) by changing one
  agent's objective prompt — no retraining.
- A principled **uncertainty story**: demand agent bids *with margin* under prediction
  uncertainty ("uncertainty sizes the margin"); the mechanism rations scarce capacity
  ("auction rations it").

## 4. The latency objection — and the gating answer

Concern: LLMs are slow (~343 ms each; two agents × multiple rounds is worse). Answer:
**selective, amortized, fallback-protected invocation.**

### The gate (latency amortization + difficulty signal)
Only run the negotiation for a queued job once it has **already waited longer than the
negotiation takes** — so the cost is hidden behind wait time already being incurred.
Stronger framing: a long wait is a **signal the cheap heuristic failed for this job**, which
is precisely when extra reasoning pays off.

> Escalate to negotiation when the fast path has demonstrably failed this job — and by then
> the negotiation cost is already hidden behind the wait.

### Nuance: waiting ≠ valuable
Long-waiting jobs are often *low-priority* (that's why they wait). Don't spend two premium
LLMs on a job nobody cares about. Make the gate two-dimensional:
```
escalate if   (wait_time > T_negotiation)   AND   (job is worth it: high value / near deadline / large)
```
Waiting time → it's *affordable*; value/urgency → it's *worthwhile*. Need both.

### Two safety requirements of the time-boxed discussion
1. **Mandatory fallback:** if the agents don't converge within the time budget, drop back to
   the cheap heuristic — never block the job. → Negotiation is strictly *best-effort
   improvement, never on the critical path.* Property: "negotiation can only help or be
   neutral, never stall."
2. **Re-validate against live state:** while the LLMs negotiate (seconds), the cluster moves,
   so the negotiated plan may be **stale**. The **ILP already fixes this** by repairing
   against current state at submission. Property: *negotiation proposes on a slightly-stale
   snapshot; the ILP reconciles to reality.*

## 5. Resulting system shape

- **Fast path (~97%):** cheap heuristic / cached strategy, milliseconds.
- **Escalation gate:** `waited long enough AND valuable enough`.
- **Two-LLM negotiation:** time-boxed, best-effort, emits a plan + justification transcript.
- **Mechanism (auction):** clears the negotiated bids deterministically.
- **ILP:** reconciles to live state, guarantees feasibility.
- **Fallback:** timeout → heuristic, so it never blocks.

Thesis sentence: *LLMs reason about value under uncertainty; the auction decides the
allocation; the ILP guarantees feasibility.* Each layer does a job no other layer can.

## 6. Open questions (next dominoes)

1. **Where does the allocation actually get made?** Two architectures:
   (a) negotiation *produces* the allocation, ILP only feasibility-checks (more radical, more
   "you"); vs (b) ILP still does heavy optimization, negotiation just sets its
   objective/priorities (safer, closer to the paper). *Which one?*
2. **Setting `T_negotiation`:** fixed budget vs adaptive to cluster load (tighter under load,
   since waiting jobs pile up faster)?
3. **Gate threshold:** learned from traces vs a simple fixed multiple of average wait?
4. **Negotiation protocol:** how many rounds, what's exchanged (bids? prices? NL
   justifications?), and how is incentive-compatibility enforced by the clearing rule?
5. **The must-have baseline:** single-LLM-with-both-objectives + ILP, to prove the
   negotiation earns its extra cost.

## 7. Related memory
See `[[research-thesis-refocus-2026-06]]` (negotiation is the star; uncertainty sizes the
margin, auction rations it; interpretability = edge vs RL) and `[[pins-negotiation-prototype]]`.
