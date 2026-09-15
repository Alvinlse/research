# Corrected one-interval oracle follow-up

Frozen before inspecting any `core-referee-oracle-v2` test outcome. This is a prospective
corrective follow-up to the validation-only pilot, not the original `core-2x2-v1` confirmatory
analysis.

## Split and fixed-policy discipline

- Evaluate only the 12 untouched `core-2x2-v1` test windows.
- Select one fixed action per family using the already completed fixed-policy validation rows in
  `runs/core_2x2/rows.jsonl`; never inspect a test outcome during selection.
- The expected frozen actions are FAIR+ADAPT for non-market and Auction-Deadline+ADAPT for market,
  but generation must recompute and verify these from validation rows.

## State selection

- Generate scheduler-visible packets every 1,800 seconds under the validation-selected fixed
  policy and retain states between 5% and 60% of the simulated trajectory.
- Retain two states per window and family without using outcomes: one state with the greatest
  observable pressure among states satisfying requested contention or deadline pressure, and one
  midpoint routine state. If a stratum is unavailable, fill deterministically across the timeline.
- Requested contention is frozen as `waiting_demand > free_gpus`.
- Minimum contention is `queue_depth > free_gpus`, because every frozen job has `g_min=1`.
- Deadline pressure is `negative_laxity_waiting > 0`.
- Because difficult states are deliberately oversampled, the aggregate over all retained states is
  a balanced diagnostic sample, not an estimate of natural event prevalence. Report strata
  separately.

## Counterfactual intervention

For every retained state and each of the 16 ordering/sizing actions in its family:

1. replay the validation-selected fixed policy before the saved state;
2. apply the candidate action at the saved timestamp;
3. restore the same fixed policy exactly 1,800 simulated seconds later; and
4. continue the identical world, arrivals, and seed to the end.

Reward is negative final deadline-violation percentage. Thus the oracle label measures a one-step
policy intervention followed by fixed-policy continuation, not a policy chosen in hindsight for
the remainder of the window. Discard an entire state if any candidate violates a simulator
invariant.

## Selector comparison

- Evaluate Qwen3-8B Single Referee and Demand + Supply + Referee on identical saved packets using
  deterministic, non-thinking decoding and the same 100-token limit per call.
- Demand and Supply emit analysis only; only the Referee selects an action.
- A malformed or out-of-family result is counted and executes the validation-selected fixed action
  in the same family. Also retain worst-case strict-output scoring as a diagnostic.
- Do not fine-tune again for this follow-up.

## Outcomes

- Fixed and oracle final deadline-violation percentage.
- Oracle headroom: oracle reward minus fixed reward.
- Single and Multi mean/median/p90 selection regret.
- Exact target accuracy, exact argmax accuracy, and fraction within 0.1 percentage point of oracle.
- Invalid outputs and deterministic fallbacks.
- Calls, prompt/completion tokens, and wall-clock inference latency.
- Report all metrics overall, by family, under requested contention, outside requested contention,
  under deadline pressure, and under their union.
- Average the two sampled states and two families within each workload window before paired
  window-level uncertainty or tests. This follow-up is reported separately from the original
  preregistered factorial hypotheses.

## Runtime audit amendment (2026-09-13)

The first deterministic test sweep raised `allocated 81 GPUs from pool 80` during a killed-job
callback. Inspection of the ElastiSim interface showed that, for that callback only, the terminal
`KILLED` job retains its old `assigned_nodes` mirror after the simulator has freed and re-offered
those nodes. The invariant already excluded the identical `COMPLETED` transition but not
`KILLED`. We amended only the ownership assertion to exclude both terminal states; all active
states remain checked for bounds, capacity, and duplicate ownership. The trigger, failed attempts,
runtime implementation hash, and reruns are retained. This correction was failure-triggered and
does not alter scheduling, sampling, policy selection, rewards, or any analysis rule above.
