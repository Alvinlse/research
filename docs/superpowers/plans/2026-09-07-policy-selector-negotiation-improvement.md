# Demand--Supply Policy-Selector Improvement Plan

Date: 2026-09-07

## Objective

Test the primary research question directly: whether structured demand--supply
multi-LLM negotiation improves GPU scheduling decisions over a single LLM at a
matched inference budget.  Every learned arm must select only a named
ordering--sizing policy; deterministic scheduler code remains responsible for
job admission, GPU counts, and feasibility.

## Primary hypothesis and controls

The primary comparison is `policy_negotiate` versus a budget-matched neutral
three-pass reviewer.  Both arms receive the same union of online information,
make three model calls per selection epoch, use the same model and decoding
settings, select from the same safe policy menu, and fall back to the same
fixed policy on an invalid response.  Secondary arms are the one-call
`policy_select`, the training-window best fixed policy, and a tabular reward
regressor.  Demand-only, supply-only, symmetric-review, and shuffled-statement
ablations are required before attributing a gain specifically to opposed roles.

The primary endpoint is paired, window-level end-to-end cost on held-out
closed-loop simulations.  Offline safe-menu regret is a diagnostic endpoint,
not the headline.  Report the paired mean difference with a window bootstrap
95% confidence interval, the win count, parse failures, unsafe selections,
calls, tokens, and latency.  Pre-register one reward operating point; report
the other operating points as sensitivity analyses.

## Phase 1 -- repair the current offline experiment

1. Split by workload window before deriving any dataset-wide statistic.
2. Derive epsilon-band popularity on training windows only and freeze it for
   validation and test construction.
3. Define forbidden actions from mechanism semantics, not held-out outcomes;
   preserve training-only worst-action diagnostics separately.
4. Add the same best-fixed-on-training baseline to every report and distinguish
   it from the raw-argmax majority label.
5. Persist per-state model choices so paired window analyses are reproducible.

Acceptance checks: train/validation/test windows are disjoint; changing a test
reward vector cannot change a training target or the safe menu; the tree report
contains a fold-specific best-fixed comparison; all deterministic tests pass.

## Phase 2 -- implement the negotiated selector

At each selection epoch, construct two asymmetric but non-oracle views.

- Demand view: queue depth and GPU demand, waiting-time quantiles, requested
  size and walltime summaries, malleability, production jobs, and laxity.
- Supply view: free/running capacity, recent arrival load, running-job age and
  declared-limit summaries, current policy, and the policy menu.

The demand and supply models emit structured statements, not allocations.  A
referee receives the base state, both statements, and the policy menu, then
selects one ordering and one sizing rule.  Invalid output retains the previous
safe policy.  A neutral three-pass arm uses identical packets and call count so
role specialization is not confounded with inference spend.

Acceptance checks: the referee prompt contains both statements; advocates
cannot name job allocations; output validation rejects unknown policies;
invalid answers preserve the prior policy; call accounting is three per
selection for both matched arms.

## Phase 3 -- align counterfactual supervision with online control

The current teacher asks which policy should be held for the rest of the
window, while the runtime selector says "next interval." Replace this with a
fixed decision horizon: branch each candidate for one selection interval, then
continue with one frozen continuation policy.  Collect states from several
behaviour policies and repeat collection after the learned policy is deployed
(DAgger-style) to cover states it actually visits.

Do not convert an epsilon-optimal set into one globally popular hard label.
Train either the complete centred reward vector or a soft distribution derived
from regret.  For pairwise training, include a pair only when its reward gap
exceeds the pre-registered practical threshold.  On near-tie states, execute
the best fixed safe policy without an LLM call.

## Phase 4 -- evaluation protocol

Use at least 30 independent windows, stratified by offered load, production
mix, malleability, and requested-size distribution.  Keep all states from one
window in one fold.  Tune only on grouped train/validation folds and run the
held-out test once.  The final closed-loop arms are:

1. best fixed safe policy selected on training windows;
2. deterministic least-laxity and market policies;
3. one-call single referee;
4. three-pass neutral/self-review referee;
5. symmetric two-reviewer referee;
6. demand--supply two-reviewer referee;
7. demand-only, supply-only, and shuffled-statement ablations.

For the text-exception study, retain the registered 81-case result but add
independent annotation, inter-rater agreement, and cross-family replication.
State the present conclusion narrowly: structured review improves Qwen2.5-14B
on authored textual exceptions; whether opposed roles are the cause remains
unresolved.

## Execution commands

The deterministic repair can run on a login node:

```bash
cd /import/gp-home.ciero/kimseng/Research
.venv/bin/python -m pins.test_policy_selector
.venv/bin/python -m pins.policy_dataset --labels runs/labels_full --out runs/dataset_v2
.venv/bin/python -m pins.policy_tree --data runs/dataset_v2 --out runs/tree_report_v2.json
```

The model stages require a GPU and a running model service:

```bash
.venv-forecast/bin/python -m pins.finetune_referee \
  --data runs/dataset_v2 --out runs/referee_lora_v2 --epochs 3
.venv-forecast/bin/python -m pins.policy_eval \
  --data runs/dataset_v2 --lora runs/referee_lora_v2/final \
  --split test --out runs/eval_report_v2.json

.venv/bin/python -m pins.elastisim_bench run \
  --world runs/holdout/<window> --arm policy_negotiate \
  --model qwen2.5:14b --sel-every 1800 --tag policy_negotiate_v1
```

Do not replace the registered test artifacts.  All corrected outputs use the
`_v2` suffix until the revised protocol is frozen.

## Execution record

Run on 2026-09-07. Phase 1 and the executable core of Phases 2--3 are complete;
the full pre-registered closed-loop experiment in Phase 4 has not been run.

- Rebuilt `runs/dataset_v2` from 589 states in 54 windows after splitting by
  window. The split is 407/88/94 states, `leak_rows` is zero, epsilon-band
  popularity is fit on training only, and the forbidden pair is declared from
  mechanism semantics. Relative to the legacy pre-split canonicalisation, the
  correction changes 9 training, 2 validation, and 15 test targets.
- Added fold-specific best-fixed and held-out paired baselines. Best fixed has
  grouped-CV safe regret 0.1185; the tree has 0.2629. On the held-out split,
  best fixed is 0.4208, tree is 0.4429, and ridge is 0.4460. The paired
  tree-minus-fixed delta is 0.0220 with window-bootstrap 95% CI
  [-0.0031, 0.0485]; the ridge delta is 0.0252 with CI [-0.0056, 0.0648].
  Thus these learned tabular selectors have not established an improvement.
- Implemented a one-call selector, three-sample best-of-three control,
  symmetric two-reviewer/referee control, and the intended demand-statement +
  supply-statement -> referee selector. The referee only chooses a policy;
  deterministic scheduler code performs allocation. Unknown or forbidden
  answers retain the prior safe policy.
- Ran one real Qwen2.5-3B/Ollama decision on the same simulator world for all
  four arms. Call counts were 1, 3, 3, and 3 respectively; all responses were
  valid, no fallback fired, and all chose `least_laxity+adaptive`. This is a
  wiring smoke test, not evidence for the hypothesis.
- Added the fixed decision-horizon teacher and ran one complete 1,800-second
  interval state end to end (15 counterfactual policies). It returned to the
  same baseline continuation for every arm and produced a complete state with
  reward range 0.012406. This small spread is direct evidence that the full
  corpus needs tie-aware targets. The run also exposed and fixed an interrupted
  probe-cache bug that could otherwise turn an empty packet file into a silent
  empty dataset.
- All ten deterministic test modules pass, including eight policy-selector
  regression tests; every Python module compiles and `git diff --check` is
  clean.

Not yet executed: full interval-label regeneration, soft reward-vector
training, DAgger-style recollection, LoRA retraining, and the at-least-30-window
matched closed-loop evaluation. The shell has no usable CUDA device, so the
LoRA stage cannot be run responsibly in this environment. These remaining
stages are required before answering the primary research question.
