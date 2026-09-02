# Migration plan — ElastiSim as the HPC AI-cluster world (post-IEEE)

**Date:** 2026-09-03  **Branch:** `referee_allocator`  **Owner:** Lay Kim Seng
**Status:** plan only. No code written, no trace converted, no run made.
**Gate:** nothing here starts before the IEEE workshop submission on 2026-09-21.

## 1. The decision

Port the replay world from `pins/two_sided_sim.py` + Alibaba v2020 to **ElastiSim**, targeting an
**HPC AI cluster** (elastic DL training / inference on a batch system), not MPI, and not cloud.

The deployment target is an AOBA-class university HPC AI cluster. AOBA job data is not available
to this project, so the world has to be reconstructed from a public trace with HPC batch
semantics.

## 2. Why the current world is the wrong one for that target

Not a defect of the simulator — a scope statement that has been in the plan since before the
experiments ran (`research_plan.md:317`):

> the trace-replay world models jobs as *moldable and grow-only* ... This matches **elastic
> GPU/ML workloads** (TorchElastic-style rescaling; the containerized ML tasks of the Alibaba
> trace it replays), which is the scoped target. It does **not** cover totally rigid
> fixed-communicator MPI jobs.

Restricting to **AI train/inference only** keeps the job model defensible — it is a poor model of
MPI CFD and a reasonable model of elastic DL. What is missing is the *operational* layer, not the
workload model:

| dimension | v2020 world | HPC AI cluster |
|---|---|---|
| queue wait | **absent** — trace has p99 = 0 s (Exp 98) | hours; the dominant QoS metric |
| deadline | synthetic `--slack-mult` ρ, *"no empirical referent"* (Exp 99b) | walltime, enforced, real |
| allocation unit | quarter-GPU quanta | whole GPU / whole node |
| rigid fraction | not modelled; ablation open since Exp 15 | most production jobs |
| comparator | no-LLM floor, EASY | FCFS / SJF / SRTF / DRF + backfill |

Note v2020 **is** an AI trace (Alibaba PAI = Platform for AI). The axis changing is the
*scheduling regime* — cloud-elastic → HPC-batch — not the workload domain. Do not describe this
as "moving to an AI workload"; a reviewer who knows PAI will catch it.

Prior empirical warning: **Exp 47** ran the mechanism on MIT Supercloud (Slurm, whole V100s) and
came back *"INERT, not harmful"* — all deltas ~0 at whole-GPU slots with **thinned load**. Read
that as an artifact of thinned load rather than a granularity verdict (Exp 99d found the market
helps *more* at coarse quantum, `dutil +1.9*` vs `+0.9*`), but treat it as a live risk, not a
solved one. §5 phase 1 exists to settle it before any mechanism is ported.

## 3. Why ElastiSim (verified 2026-09-03)

| requirement | ElastiSim |
|---|---|
| mid-run reallocation | **malleable** job type; the only batch simulator supporting it |
| rigid/malleable mix (the φ axis) | **a field on the job** — rigid/moldable/malleable/evolving/adaptive |
| tick-based clearing | scheduler invoked on a **user-defined interval** or on events |
| Python policy | external scheduler process over **ZeroMQ**; Python supported |
| GPU | `num_gpus_per_node_min` / `_max`; explicit GPU model aimed at DL |
| literature baselines | FCFS, SJF, SRTF, DRF ship with it |
| LLM latency | *"the entire simulation stops at invocation time and continues after the
  scheduling algorithm returns"* — **decision time consumes simulated time** |

Two consequences worth stating plainly:

- **The φ ablation stops being a build.** Open since Exp 15 (`research_progress.md:85`,
  *"malleability ablation remains unstarted"*), it becomes a workload config field. It is the
  experiment that decides whether the mechanism survives at HPC rigidity.
- **The malleability premise becomes a citation.** The paper currently concedes that runtime
  elasticity is rarely deployed (`tarraf2024`); running on the simulator built for malleable
  workloads converts an apology into a reference.

Docs: <https://elastisim.github.io> · repo <https://github.com/elastisim> ·
paper: ElastiSim, ICPP 2022, doi 10.1145/3545008.3545046

## 4. What it costs

1. **Deadlines do not exist — walltime only.** The whole SLA framing must be remapped. This is the
   *good* kind of problem: walltime is real and enforced; ρ was invented and has been apologised
   for since Exp 99b. Derive laxity from walltime, or carry a deadline in `attributes`.
2. **`application_model` per job.** The Amdahl / saturating counterfactual progress laws move into
   a performance-model file. Right home for them, but real work. **Prototype this first** — it is
   the piece most likely to not map.
3. **Trace conversion** to ElastiSim JSON. Third adapter of its kind; the least risky item.
4. **Everything bespoke re-expressed**: bid/ask clearing, tick validator, oracle regret, decision
   packet, trigger. Budget 1–3 months and expect it to reproduce existing findings on a new
   substrate before it produces anything new.

## 5. Phases, each with a go/no-go

### Phase 0 — the one-column check (hours)
`head -1 data/slurm-log.csv | tr ',' '\n' | nl` on the cluster. The adapter reads `id_job`,
`time_start`, `time_end`, `tres_alloc`, `state` and compares state to numeric `"3"` — that is the
slurmdbd `*_job_table` schema, so these are likely present and currently discarded:

- **`time_submit`** → real queue wait. *Decides whether the HPC world is viable at all.*
- **`timelimit`** → real deadline referent. Retires ρ.
- **`job_name` / `submit_line` / `admin_comment`** → possible real, non-synthetic text channel.
  Note `build_supercloud_replay.py` currently writes `id_job` into the `job_name` column, so a
  real `job_name` field, if present, is being thrown away.

**No-go if `time_submit` is absent:** fall back to a DL-cluster trace that has it — Philly
(Jeon et al., ATC'19), Helios (Hu et al., SC'21), or Acme (Hu et al., NSDI'24).

### Phase 1 — characterise the floor. THE GO/NO-GO. (days)
Convert Supercloud → ElastiSim JSON. Run **FCFS and SRTF only**. No market, no LLM, no gate.
Report: utilisation, queue-wait distribution, timeout rate, contention ratio.

**This phase exists because the project has been burned twice by porting a mechanism into a world
with no rationing problem:** Exp 97 (the deadline recipe was infeasible — every SLA number in nine
experiments died) and Exp 98 (real tiers → floor already at 0.0% prod violations, no headroom).
Exp 47's inert result on this very trace is the third warning.

**No-go if the world is not contended.** Stop here; the cost is days, not months.

### Phase 2 — the φ ablation (days, once phase 1 passes)
Sweep the rigid/malleable mix as a workload field. Deterministic arms only.
Tests Exp 14–15's standing prediction that *"the supply agent's value scales with the rigid
fraction — rigidity is where the contribution lives"*, never tested on a real HPC trace.
**Publishable either way**, and it decides whether phases 3–4 are worth doing.

### Phase 3 — port the deterministic layer
Floor → market → least-laxity ordering → validator. Re-derive the operating point from scratch;
do **not** assume v2020's transfers (that assumption is exactly what Exp 97 punished).
Baselines are now FCFS/SJF/SRTF/DRF, not just the in-house floor — this closes the
"no baselines from the literature" gap, which is a bigger reviewer risk than the simulator itself.

### Phase 4 — the gate, and the latency result
Port trigger + validator + floor fallback. **Do not block on live LLM calls** — 5,494 escalations
× ~60 s never finishes in wall-clock. Charge latency as a modelled constant and serve rulings from
the existing scene cache (`pins/referee.py` is already scene-cached). That yields the measurement
the IEEE draft currently concedes it cannot make (*"evaluation runs on simulation clock"*).

## 6. Explicitly out of scope

**Do not synthesise descriptive operator text.** Exp 94 authored notes into the trace: the
allocation moved on **4 of ~842 escalated ticks and flipped zero outcomes**, SLA identical
seed-for-seed. A note that duplicates observable state cannot change an allocation, and text
generated by an LLM then read by an LLM is circular. It would also cost the paper its most
credible sentence — that the channel's absence was verified rather than papered over.

The admissible version, if the text channel is ever revisited, is a **latent world state the
simulator enforces**: true capacity ≠ declared capacity, with the note as the sole carrier of the
truth, and ground truth = the sampled latent parameter rather than an annotator's opinion. That
also closes the outcome gap in §7. Out of scope for this migration; recorded so it is not
re-derived.

## 7. Open issues this migration does *not* close

- **Predicate ≠ outcome.** `HardCase.predicate` scores "did it do the defensible thing"; there is
  no SLA/util/regret anywhere in the suite harness. The suite has rulings without outcomes; the sim
  has outcomes without rulings (Exp 92: 0 calls). **They have never met.** The IEEE draft must
  state this as a limitation.
- **Exp 84 and Exp 96 SLA/prodSLA numbers are dead** — on the Exp 97 invalidation list
  (`research_progress.md:3152`) and never re-run. `dprodSLA −12.0*` and the tight-tercile
  `+6.9 ± 5.1*` are **not citable**. Exp 84's efficiency metrics (`duseful −2.9 ± 1.7*`,
  `dregret +5.0 ± 2.1*`) survive — the invalidation note exempts util/useful/regret explicitly.
- **The numeric claim must be economic, not capability-based.** "LLMs are bad at numerics" is
  refuted by this project's own data: Exp 90/91 (0/400 false suggestions on real numeric scenes),
  Exp 49 (*"constraint enforcement, not arithmetic, is the failure"*; r1:32b and qwen3.5:35b 100%
  feasible under scarcity), Exp 34 (MAE 572→43→22 GB as 3b→7b→14b). The defensible claim is
  **dominance**: the numeric decision has a computable optimum, so the reasoning layer's best case
  is a tie, and it costs tokens and latency for that tie. That claim survives whatever the
  frontier-model arm returns.
- **The finding is three-level, not two.** Text carrying a decision-relevant fact → debate wins
  (43/81 vs rigid 0/81). No text → ties safely (0/400; in-sim no-op). Text present but decorative
  or expired → **loses** (12/17 vs market 16/17). The hazard is irrelevant prose, not numbers —
  so the gate must discriminate *decision-relevant* text, not merely the presence of text.

## 8. Reading before phase 0

- Evaluating Malleable Job Scheduling in HPC Clusters using Real-World Workloads —
  <https://arxiv.org/html/2602.17318> (2026; closest published version of the phase-0/1 conversion)
- Probabilistic Job History Conversion and Performance Model Generation —
  doi 10.1007/978-3-031-40843-4_7
- ElastiSim scheduling protocol — <https://elastisim.github.io/get-started/scheduling-protocol/>
- ElastiSim job spec — <https://elastisim.github.io/workload/job/>
