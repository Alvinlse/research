# Referee packet v2 — pre-declaration

Written before any v2 run, because every change below is a prompt intervention on a structure whose
result is already known. The purpose of this file is to bound the search so the outcome cannot be
reached by tuning until the sign flips.

## What prompted it

On the highest-load training window (`d48h23`, offered load 4.99) the role-separated referee chose
least-laxity ordering with queue-proportional sizing in 67 of 74 decision epochs, while the single
referee on the same window moved between four policies. The two arms' outcomes, 31.70% and 24.10%
deadline violations, are close to the fixed-policy scores of the policies each kept choosing
(least-laxity 32.00%, fair-share 22.50%). The role-separated arm did not fail to decide; it settled on
the worse policy and held it.

Two candidate causes were identified in the packet, and one was measured away:

- The demand analyst flagged urgent jobs in 74 of 74 epochs and had no way to report their absence.
  **Fixed and under test separately** (`runs/pilot_anchor`), by defining urgency against the median
  laxity the packet already shows and licensing the empty list.
- Statement **verbosity** was hypothesised and refuted by measurement: the two statements are 307 and
  248 characters against 1,277 for the static menu and rules, about 4% of a 3,400-token prompt.

## The intervention

One combined change, not four measured separately. We are not estimating individual effects; we are
asking whether a better-constructed packet removes the collapse. Decomposition happens only if the
combined change works.

1. **Statements move before the state.** They currently sit last, immediately before the instruction to
   decide. As inputs to reading the state they belong first.
2. **The history block gains a comparison.** It reports that the queue grew after a choice, which is
   uninterpretable at 4.99 offered load where the queue grows under every policy, and closes with an
   instruction that therefore fires at every epoch. It will report the queue delta against the previous
   interval instead.
3. **Stickiness becomes visible.** `current_policy` gains the number of consecutive intervals it has
   been held.
4. **The static menu keeps action names and drops the rule text.** It is the largest block in the packet
   and identical at every epoch, roughly 17k repeated tokens per window for one call per decision and
   51k for three. Tested rather than assumed: an earlier experiment found the decision packet is what
   makes rulings feasible and that the effect was capability-gated, so invalid-ruling counts are a
   primary readout here, not a footnote.

Implemented behind a flag, default off, so the frozen protocol path is unchanged.

The three runs form an incremental chain on the same window, seed and arm, so each step adds exactly
one intervention: (1) the original packet and analyst prompt, already measured at 31.70% violations
with 67 of 74 epochs on one policy; (2) the analyst anchor fix alone; (3) the anchor fix plus the four
packet changes above. Step 3 is therefore not attributable on its own, and if it succeeds the
decomposition is between steps 2 and 3, not within step 3.

## Bounds on the search

- **Training windows only.** No v2 run touches validation or the reserved split.
- **At most three attempts.** If the third does not remove the collapse, the packet is not the cause and
  the negative result stands as measured.
- **One reserved-split run for the winner.** Whichever configuration is carried forward is run once on
  the reserved split, as an amendment recorded with the protocol, and reported whatever it shows.
- Every v2 run changes the implementation hash, so the gate will reject protocol runs until the
  configuration is either recorded as an amendment or reverted.

## Readouts, fixed in advance

Primary: the share of epochs spent on the most-chosen policy, against 67/74 before. A packet that has
stopped anchoring produces a distribution closer to the single referee's 38/76.

Secondary, in this order: deadline violations against 31.70% on the same window; the urgency flag rate
against 74/74; invalid rulings, which were 0 on this window and 3 across the twelve validation windows;
and prompt tokens per window, against 551k.

Not a readout: whether role separation beats the single referee on this window. One training window
chosen for contention cannot settle that, and it is not what these runs are for.
