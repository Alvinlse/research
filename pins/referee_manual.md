# Referee manual — precedents distilled from experiments

Appended to the referee's system prompt when the manual arm is on. The five rules
(feasibility, individual rationality, priority, envy-freeness, incentive skepticism)
live in `SYSTEM_REFEREE` in `referee.py`; this file holds only what the rules don't
say — **rulings that were solved the expensive way once** (reasoning-model calls,
lost seeds, closed experiments), each conditioned on the cluster state that justified
it. Exp 50's transcript study showed the same request correctly gets *opposite*
rulings depending on state, so a precedent without its WHEN clause is a bug.

Keep it short: every line costs prompt tokens (sweeps run at num_ctx 8192). The
cache key includes this file's hash — editing it starts a fresh experimental arm,
it never mixes with cached rulings from an older manual.

Adding a precedent: when an experiment closes, distill the win/loss into one entry —
`WHEN <state> → <ruling>` plus the one-line source. Prefer editing an existing entry
over adding a near-duplicate.

<!-- PROMPT-START: everything below this line is sent to the referee -->
PRECEDENTS from past allocations (apply when the WHEN clause matches; cite the
precedent id in your justification):

P1. WHEN a prod job requests margin and the free pool covers only part of it:
    grant a PARTIAL margin (asked 2 -> give 1) instead of all-or-nothing.
    [source: Exp-50 seed 3 win — partial grant + reserve beat the rigid rule]

P2. WHEN incoming_prod is 'few' or 'many' AND the pool is tight:
    do NOT spend GPUs hedging besteffort jobs that are 'ahead' of schedule;
    hold that headroom for the incoming prod work.
    [source: Exp-50 seed 2 loss — hedged ahead-of-schedule besteffort, pool was
    empty when the prod job arrived]

P3. WHEN the free pool cannot cover every base (shortfall):
    someone must be refused — prefer partial coverage of prod bases over serving
    everyone; NEVER exceed the pool to avoid saying no.
    [source: Exp-49 — every chat model overcommitted under scarcity rather than refuse]
