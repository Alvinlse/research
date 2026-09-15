# Next experiment: does multi-LLM reasoning improve a single LLM's decisions?

Protocol v1, 2026-09-16. Status: implemented; no model/simulator results claimed.

## Research question and scope

Primary contrast: Demand–Supply–Referee minus single-model self-review on final
deadline-violation percentage points, lower is better. Secondary contrasts are
Multi minus one-call Single and Multi minus symmetric independent reviewers.
Use the same base model for every role. This tests reasoning organization, not
heterogeneous model quality or negotiation between independently trained agents.

The existing core workload windows, including the former test split, have already
informed this project. ALL results from this follow-up are exploratory. Freeze a
new independent workload evaluation before making a new confirmatory claim; do not
rename old test windows as untouched. Default execution uses the 24 training windows.
Do not tune prompts between windows or select a winning subset after seeing outcomes.

## Four arms

1. `single`: one final decision, one call.
2. `self_review`: neutral proposal, critique/revised proposal that sees the first,
   final decision that sees both. Three calls to the same model.
3. `symmetric`: two independent neutral proposals, then the same final referee.
4. `multi`: independent Demand and Supply proposals, then the same final referee.

All reviewers receive the complete same observable packet. Role specialization
changes the focus of attention, not information access, action permissions, or the
primary objective. All stages propose global ordering/sizing pairs. Final prompts,
schemas, parsers and deterministic executor are identical. No per-job overrides,
trial/probation manager, learned checker, backoff, fine-tuning, or outcome memory is
added. Existing recent-state/decision fields in the common packet remain shared.
No free-form chain-of-thought is required: a short evidence/tradeoff statement suffices.

Three-call arms match the per-stage model, temperature, maximum output tokens and
seeds. Stage seeds are repetition seed + 0/1/2; Single uses +2. Calls run serially,
including independent reviewers, so concurrency does not confound measured cost.
Cache reuse and automatic retries are disabled. Invalid reviews are marked missing;
the remaining scheduled calls still run. Invalid finals use the same fixed fallback
and are counted. Full prompts, raw outputs and parse errors are audited.

This matches three calls and 1,200 maximum output tokens per decision, NOT actual
input/output tokens or exact compute. Self-review consumes its earlier proposal in
the second call. Report actual prompt/completion tokens and inference wall time.
Closed-loop arms may visit different numbers of decision epochs; report total cost
as well as invalid decisions and number of epochs. Do not describe one versus three
calls as a budget-controlled comparison.

## Frozen settings and action space

`matched_reasoning_config.json` fixes Qwen2.5-14B, temperature 0.1, 400 tokens/call,
non-thinking mode, repetition seeds 17/29/43, scheduler interval 300 s and selection
interval 1,800 s. Use both existing policy families separately, each with four
ordering/pricing choices crossed with four sizing choices. These are global pairs;
the same-state oracle therefore measures exactly the available action space.

Fallback/initial policies are fixed before this experiment: `fairness+adaptive`
for non-market, `auction_deadline+adaptive` for market. They are references, not
asserted optimal baselines. Fixed-policy outcomes from the existing core study may
be shown descriptively; do not compare incompatible workloads or select on test.
FCFS is already a non-market choice. SJF is outside this frozen family and is not
silently added. The main question is Multi versus Single, not beating FCFS.

The selector retains the frozen bench's eligibility: decisions occur on an eligible
normal callback with pending jobs AND free resources, at least 1,800 s after the
previous decision. Thus the interval is a minimum, not guaranteed wall-clock
periodicity. All structures have the identical clock. Work-boundary resizing remains
deterministic and uses the currently chosen sizing policy, common feasibility,
cooldown and resize-budget checks. Frozen world assumptions, including zero modeled
resize overhead, remain limitations. Model latency is measured but not injected into
simulated time; service metrics alone are not a net operational benefit claim.

## Two evaluation modes

### Same-state diagnostic

Use complete one-interval oracle datasets from `pins.core_oracle_labels`: candidate
for 1,800 seconds followed by the same fixed continuation, identical arrivals.
The runner rejects missing actions, incompatible horizons/continuations, duplicate
states, missing family coverage and windows crossing train/validation/test splits.
Legacy rest-of-window labels MUST NOT be used. Each structure receives only the
user packet, never assistant targets or reward vectors. Rewards are used after the
decision for regret and realized counterfactual deadline scoring.

This samples the baseline's states; it does not establish performance on states
visited by the learned controller. Use it to diagnose decision quality, output
failures and potential headroom. Do not treat the per-state oracle as a realizable
closed-loop scheduler. Do not condition completion of the full-run evaluation on
which arm wins this diagnostic.

### Closed-loop primary evaluation

Each structure runs the whole world with the same warm-up and scoring population.
Within a window all families, seeds and structures run in a fixed shuffled order
(seed 20260916); windows are completed in manifest order. One invocation defaults
to one cell, with append-only results and a process lock. No arm output is read by
another run. Decision records preserve all intermediate proposals and final validity.

The original core source and manifest remain unchanged. A scoped registration uses
the frozen `policy_select` clock/executor for all four new structures; rows carry
the actual `structure` and a separate experiment manifest. The original runtime
world verification is retained. The new run manifest additionally binds every Python
source, this protocol, configuration, data hashes, model digest, commit and exact task
set. Resume fails on mismatches. Never overwrite an existing output directory to
hide a failed or changed run. Audit malformed/errors, even if fallback performs well.

Inherited provenance repair: at upstream `b97935f`, the core manifest's implementation
hash no longer matches its listed files because subsequent development changed them.
This experiment explicitly pins the inherited combined hash
`1925a0b38ebd73987ea94644e6a87bfa52ef45d2b7507ba699cfd93fb73e7df3` in its own config.
The original verifier receives a temporary derived world contract using that pinned
hash; all world, trace, split and job checks still execute. The old core manifest
and historical result hashes remain untouched. New code changes fail this pin or
the additional run-manifest source hashes; they are not silently re-frozen on resume.

## Analysis frozen before execution

Primary endpoint is full-run deadline violations. Same-state regret is diagnostic.
Average repetitions/states within each window and family, then average the two
families equally. The inferential unit is the workload window, never a state, job,
family or model seed. Report paired mean/median differences, wins/ties/losses,
10,000-resample paired-window bootstrap 95% intervals (seed 20260916), and two-sided
paired sign-flip p-values. Enumerate up to 16 nonzero differences; otherwise use
100,000 sampled signs with the +1 correction. Holm-adjust all three contrasts.
Sign-flip inference assumes sign exchangeability; bootstrap precision is limited
with few windows. Confidence intervals and p-values can disagree; report both.

Waiting, tail waiting, fairness, fixed/oracle references, parsing failures, calls,
tokens and latency are secondary/descriptive. Analyzer refuses incomplete or mixed
experiments. Keep exploratory status regardless of significance. A win over Single
alone does not establish benefit beyond extra inference; Multi versus self-review
tests that question; Multi versus symmetric tests whether role specialization helps.

## Run on the ElastiSim/Ollama host

From repository root, with the existing `.venv`, model server and frozen worlds:

```bash
python3 -m unittest pins.test_matched_reasoning
python3 -m pins.run_matched_reasoning --mode closed-loop --split train \
  --out runs/matched_reasoning_v1_train --dry-run

# One complete simulation per invocation; repeat until all 576 cells complete.
bash pins/matched_reasoning_tick.sh --mode closed-loop --split train \
  --out runs/matched_reasoning_v1_train --max-runs 1

# Or run continuously on a suitably allocated compute node.
bash pins/matched_reasoning_tick.sh --mode closed-loop --split train \
  --out runs/matched_reasoning_v1_train --max-runs 0

.venv/bin/python -m pins.analyze_matched_reasoning \
  --out runs/matched_reasoning_v1_train
```

For the diagnostic, use the matching one-interval dataset path:

```bash
bash pins/matched_reasoning_tick.sh --mode offline --split validation \
  --data runs/core_referee_oracle_v2/dataset \
  --out runs/matched_reasoning_v1_offline_validation --max-runs 1
```

The dataset path above is an example, not an assertion that it exists. Generate
matching labels using `pins.core_oracle_labels --decision-horizon-s 1800` and
the existing oracle-v2 protocol first if necessary. `--dry-run` on offline mode
validates the dataset. `PINS_PYTHON` can select a different Python environment;
`--worlds` selects an existing verified world directory. A full 576-cell run is
expensive; the one-cell default is intentional. Never launch a full sweep on a
login node. Commands stage the next experiment; installation does not start it.
