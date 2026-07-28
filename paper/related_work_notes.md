# Related-work notes

Candidate citations for the 関連研究 section of `main.tex` (its `\todo` asks for 2–3 refs
per paragraph). Each entry: what the paper does, why it is NOT us, and the one-sentence
delta to state.

## Croitoru, Croitoru & Ganesh (ICAART 2025) — prediction-guided selective negotiation

- **Cite as:** M. Croitoru, C. Croitoru, G. Ganesh: *Prediction-Based Selective Negotiation
  for Refining Multi-Agent Resource Allocation.* ICAART 2025, Vol. 1, pp. 656–662.
  DOI: 10.5220/0013368000003890. PDF: https://www.scitepress.org/Papers/2025/133680/133680.pdf
- **What it does:** One-shot fair division of static goods. Stage 1: Borda-vote preference
  aggregation → greedy allocation. Stage 2: a classical ML classifier predicts each agent's
  preference ranking from features; agents whose predicted ≠ stated preferences are paired
  for Condorcet-style pairwise good swaps. Theorem: monotone satisfaction ⇒ swaps never
  decrease total satisfaction. Worked example only (10 children, 5 goods); no experiments.
- **Genuine overlap:** *selectivity* — predictions decide WHICH agents are worth
  negotiating, structurally similar to our escalation gate (negotiate only when wait time +
  value justify it).
- **Not us:** no LLMs, no temporal/scheduling dimension (one-shot goods, not GPU-hours under
  SLA), "negotiation" is a deterministic swap search, prediction targets preferences (not
  job resource needs), no mechanism/incentive layer, no trace evaluation.
- **Delta sentence:** prediction-guided selective negotiation has been explored in one-shot
  fair division; we bring the idea to temporal HPC scheduling with LLM agents, an auction
  that decides, and an ILP that guarantees feasibility.
- **Where in main.tex:** メカニズムデザイン paragraph, or a new 交渉 paragraph together
  with Bo An (below).

```latex
\bibitem{croitoru2025} M.~Croitoru, C.~Croitoru, G.~Ganesh: Prediction-Based Selective
Negotiation for Refining Multi-Agent Resource Allocation. Proc.\ ICAART 2025, Vol.~1,
pp.~656--662, 2025.
```

## Bo An (PhD dissertation, UMass Amherst 2011) — game-theoretic automated negotiation

- **Cite as:** B. An: *Automated Negotiation for Complex Multi-Agent Resource Allocation.*
  Ph.D. dissertation, University of Massachusetts Amherst, 2011 (advisor V. Lesser).
  IFAAMAS Victor Lesser Distinguished Dissertation Award 2010.
  https://scholarworks.umass.edu/open_access_dissertations/329/
- **What it does:** Classical game-theoretic negotiation for resource allocation:
  alternating-offers bargaining under deadlines and incomplete information, concurrent
  one-to-many / many-to-many negotiation with competitors, commitment/decommitment
  penalties in dynamic markets; applied to networked resources incl. cloud pricing. The
  product is analytically derived equilibrium strategies.
- **Genuine overlap:** the closest classical ancestor — negotiation for computational
  resource allocation under uncertainty with self-interested agents.
- **Not us:** (1) agents execute closed-form equilibrium strategies derived offline vs. our
  LLMs reasoning in NL (the transcript is the interpretability edge — a strategy formula
  explains nothing per-decision); (2) the bargaining process IS the allocator vs. our
  negotiation-only-proposes, auction decides, ILP guarantees; (3) honesty from equilibrium
  analysis vs. our measured mechanism design (budgets/claim pricing/tariffs, Exp 32–33
  best-response tests with lying LLM agents); (4) no job-resource prediction; (5) analytical
  market models vs. real-trace replay with seed statistics.
- **Delta sentence:** PINS replaces hand-derived equilibrium bargaining strategies with LLM
  reasoning, and moves the decision out of the bargaining process into a deterministic
  auction + ILP, so the outcome stays guaranteed even when the reasoner is wrong.
- **Where in main.tex:** anchor reference for the negotiation lineage — fits a new
  自動交渉 paragraph in 関連研究, or the メカニズムデザイン paragraph's opening claim
  that structured negotiation is a classical subject.

```latex
\bibitem{boan2011} B.~An: Automated Negotiation for Complex Multi-Agent Resource
Allocation. Ph.D.\ dissertation, University of Massachusetts Amherst, 2011.
```
