# ElastiSim bench — stratified window design (PRE-REGISTRATION)

**Date:** 2026-09-03  **Branch:** `elastisim`  **Bench:** `pins/elastisim_bench.py`
**Status:** written **before any measured run**. Selection of the 12 windows is frozen in this
document (§3) and was made from workload properties alone, with no scheduler executed. The
motivating diagnostics in §1 were run on throwaway windows in a scratch directory and are not part
of the measured set.

## 1. Why the existing setup cannot support a result

Three findings from the 2026-09-03 diagnostic pass. All are properties of the Supercloud trace, not
bugs in the harness.

**EASY backfilling is inert.** Across five replayed windows: 1351 blocked scheduling points, 48
backfills (3.6%); 523/1 on day 157 and 61/0 on day 171, with EASY and FCFS metrics identical to the
digit. The mechanism is `now + est(j) <= shadow` (`elastisim_bench.py:214`): requested walltimes on
this trace are 100–600× larger than true runtimes, so the shadow time is computed from garbage and
nothing can be shown to fit before it. The estimate-independent route (`j.num_nodes <= extra`,
`:216`) still fires occasionally — one window got 47 backfills that way — so the claim is "EASY
degenerates to FCFS on most windows", not "EASY cannot backfill".

**Requested walltime carries almost no runtime signal.** Spearman(requested, true) over jobs with a
declared limit, tie-corrected: d40h5 +0.156 (p=0.104), d161h4 −0.111 (p=0.088), d169h2 −0.726
(p<0.001), d157 −0.006 (p=0.901). The one significant window is *negative*. The cause is visible in
the cardinality: 487 jobs share **9 distinct** requested walltimes; 110 jobs share **3**. Requested
walltime is a partition label with about three settings, not a duration estimate.

Consequently `arm_sjf` is not shortest-job-first. Its ordering key
`(int(req_min) or 10**9, submit_time)` (`:182`) sorts undeclared jobs last, and undeclared jobs are
usually much longer (sentinel vs declared median true runtime: 47830 vs 2228, 32028 vs 14441, 50913
vs 2805 — inverting in one window, 2070 vs 2872). Whatever it wins, it wins from a one-bit
"did you declare a limit" rule, not from SPT.

**Ordering is confounded with packing.** `arm_sjf` delegates to `arm_firstfit`, which skips a
non-fitting job and keeps scanning; `arm_fcfs` stops at the first non-fitting job (`:169`, `:176`,
`:184`). Any `sjf` − `fcfs` gap therefore mixes ordering with head-of-line bypass. `easy` also stops
at the head, so switching the baseline from `fcfs` to `easy` did not remove this confound.

**No true-runtime leak.** `_true_dur` is written into job attributes (`:126`) and read by no arm.
It reaches the simulation only through `flops = duration × FLOPS_PER_GPU` (`:120`), which is the
intended path.

**The blocking design fault.** Offered load is currently an *output*: `pool = GPU-hours / hours /
load` (`:103`, `:111`). Every window is 3.93× by construction, so load cannot be a stratum. This
design inverts it — the pool is fixed and load becomes a measured property of each window.

## 2. Cluster and timeline

**Pool: 80 GPUs, fixed across every window.** One ElastiSim node = one GPU (unchanged). 80 is
chosen so the trace exercises all four load bands: it is the tested size at which all twelve
(load × tier) cells hold ≥11 non-overlapping candidate windows. At 121 GPUs — the median window's
demand — the stress row collapses to 5/0/3 and one cell is empty. This justification is
stratum coverage and is stated as such; it is **not** a reconstruction of Supercloud's real
hardware, which the repository has never calibrated (`data/build_supercloud_replay.py:16` notes
TX-GAIA nodes carry two V100s while the bench models one node as one GPU).

**Timeline per window:**

```
[0, 12h)     warm-up    prior arrivals; simulated, occupy nodes, NOT scored
[12h, 36h)   measured   the 24h window; the only jobs scored
[36h, …)     drain      measured jobs run to completion
```

Warm-up exists so no window begins on an empty cluster. All submit times shift by +12h so
ElastiSim never receives a negative submit time. Warm-up jobs carry `"_warmup": true` in
`attributes`.

**Warm-up jobs are scheduled by the arm under test.** There is one scheduler in the simulation and
the tag is invisible to it — warm-up jobs are ordinary pending work, and only scoring (§4) treats
them differently. This is deliberate: a real scheduler inherits the backlog its own earlier
decisions produced, and an arm should live with that.

The consequence is that the cluster state at `t = 12h` is **arm-dependent** — an arm that clears
the warm-up efficiently enters the measured window less congested than one that does not. Arms
therefore receive identical *inputs* but not an identical starting state, and a measured-window
difference includes the effect of how the arm handled the run-up. Two things follow, and both are
reported: the pairing in §6 is on the window, not on the state at measurement start; and any arm
whose advantage disappears when the warm-up is scheduled by a common policy is winning on run-up
management rather than on the measured window. The fixed-warm-up variant (one reference policy
through `t = 12h`, handover to the arm under test after) is **not** part of this design and is
named here only as the follow-up that would separate the two effects.

12h does not cover every job: p90 runtime reaches 129772s (36h) in `stress/hi` and 115087s in
`stress/md`. Those two cells begin marginally emptier than reality. This is documented rather than
fixed, because extending the warm-up shrinks how many disjoint windows each cell can supply.

**Window length 24h** is the default (day/night variation plus queue build-up). 12h and 36h are
sensitivity tests only, re-selected through the same procedure.

## 3. The frozen window set

Selection procedure, executed once: census every hourly 24h start across the 234.8-day trace
(5613 starts, 4312 with ≥100 jobs); compute per candidate the jobs, GPU-hours, demand
(GPU-hours/24 = mean GPUs busy), prod fraction, median and p90 runtime, and GPU-count
distribution; assign each to a (load, tier) cell; then for each cell in fixed order, shuffle with
`random.Random(20260903)` and take the first candidate whose start is ≥36h from every window
already chosen.

Load band = demand / 80. Tier band = fraction of jobs with Slurm priority ≥ 100000.

| load | range | tier | range |
|---|---|---|---|
| low | 0.7–1.0× | lo | <2% |
| med | 1.0–1.5× | md | 2–10% |
| high | 1.5–2.5× | hi | >10% |
| stress | >2.5× | | |

Tier bands are absolute, not tertiles: tertiles over the candidate pool put the "high" cut at 6%,
which is not a contrast. Absolute bands keep all twelve cells populated (≥11 disjoint windows each).

**The 12 windows** (`day`/`off` are the window start; seed 20260903):

| cell | day | off | jobs | gpu_h | load | prod | med_s | p90_s | maxg |
|---|---|---|---|---|---|---|---|---|---|
| high/hi | 41 | 23 | 315 | 4184 | 2.18 | 0.581 | 23449 | 74715 | 2 |
| stress/hi | 48 | 22 | 429 | 9542 | 4.97 | 0.200 | 12940 | 129772 | 16 |
| med/md | 71 | 16 | 206 | 1977 | 1.03 | 0.029 | 6847 | 56298 | 2 |
| low/hi | 104 | 0 | 235 | 1493 | 0.78 | 0.145 | 8542 | 53852 | 2 |
| med/hi | 105 | 13 | 608 | 2626 | 1.37 | 0.390 | 11981 | 23411 | 2 |
| low/md | 130 | 9 | 161 | 1835 | 0.96 | 0.031 | 24716 | 73691 | 2 |
| high/lo | 173 | 18 | 627 | 3853 | 2.01 | 0.011 | 3470 | 32666 | 2 |
| high/md | 177 | 2 | 401 | 3107 | 1.62 | 0.052 | 11355 | 77239 | 4 |
| stress/md | 183 | 23 | 328 | 6174 | 3.22 | 0.021 | 5776 | 115087 | 64 |
| stress/lo | 198 | 7 | 1024 | 5454 | 2.84 | 0.013 | 3289 | 29233 | 2 |
| low/lo | 217 | 9 | 572 | 1438 | 0.75 | 0.007 | 2455 | 17504 | 2 |
| med/lo | 222 | 10 | 649 | 2150 | 1.12 | 0.012 | 1917 | 22004 | 8 |

Minimum gap 37h (requirement 36h). Days 41–222 of 234.

This table is the pre-registration. It is committed to `pins/windows12.json` and read at run time;
the sampler is **not** re-rolled per run.

**Case studies, outside the 12.** Day 157 (load 4.72×, prod 1.8%, median 3154s, jobs up to 8 GPUs)
and day 105 (load 1.58×, prod 28.2%, median 9740s, jobs up to 2 GPUs) are retained as illustrative
single windows. They differ on every axis, which is what motivated the tier stratification. They
are reported as case studies and excluded from every pooled statistic.

## 4. Scoring

Two distinct filters, and conflating them is the main implementation risk:

- **Scored set** — jobs submitted in `[12h, 36h)`. All wait, turnaround, BSD, SLA-10, tier and
  requested-limit metrics use this set only.
- **Occupancy** — utilisation during `[12h, 36h)` counts *every* running job, warm-up included,
  because the cluster is genuinely occupied by them. `util_win` = busy GPU-seconds in that interval
  / (80 × 24h).

Changes required in `summarise()` (`:425`; `W` at `:429`, `span` at `:447`): `W` becomes the interval `[warmup, warmup+hours)`
rather than `[0, hours)`; the per-row loop skips warm-up jobs for `wait`, `bsd`, `late`, `tier`,
`n_req`/`late_req` and `gpu_s`, but not for `busy_win`; `span` is computed from the scored jobs'
min submit and max end.

Real Slurm waits for the **scored** jobs stay in `meta.json` as calibration targets. They are
recorded and never used to size the pool or select a window.

## 5. Arms

Baseline is `easy`, with §1's caveat stated wherever it is used: on this trace EASY degenerates to
FCFS on most windows.

Required before any headline ordering claim, a new arm **`declared_first`**: key
`(req_min == 0, submit_time)` fed through the existing `arm_firstfit`. It isolates the two effects
that `sjf` currently conflates:

- `firstfit → declared_first` — what the declared/undeclared bit is worth
- `declared_first → sjf` — what requested-walltime ordering adds on top

Comparisons that must hold packing constant use `firstfit`, not `fcfs`, as the reference, since
both route through `arm_firstfit`.

`sjf` is labelled **"requested-walltime ordering"** in all output and write-up, never "SJF" or
"shortest job first". An oracle true-runtime arm may be added later as an upper-bound diagnostic
only, clearly marked as not implementable.

## 6. Analysis

Each of the 12 windows is a paired block: every arm sees an identical *world* — the same jobs,
arrivals and cluster. It does not see an identical cluster *state* at the start of the measured
window, because each arm schedules its own warm-up (§2). The primary comparison
is the paired delta against the baseline, pooled across all 12, with the win-rate table already in
`report()` showing whether a mean win is consistent or carried by outliers.

**Cells are coverage, not claims.** With one window per cell, a cell-level difference is fully
confounded with that particular window's identity — nothing separates "stress windows behave like
this" from "day 48 behaves like this". Per-cell numbers are reported descriptively to show the
result is not confined to one corner of the design; any claim that a stratum *causes* a difference
requires replication (2 windows per cell, 24 total) and is out of scope here.

## 7. Work required

In `pins/elastisim_bench.py`:

1. `build`: fix `--pool`; retire load-based sizing on this path; add `--warmup-h` (default 12) —
   load `[t0 − warmup, t0 + hours)`, shift submits by +warmup, tag warm-up jobs.
2. `summarise`: the two-filter split of §4.
3. `--hours` default 12 → 24.
4. `census` subcommand: emit the §3 candidate table (properties only, no simulation).
5. `windows12.json` manifest + a `sweep --manifest` path that replays exactly those windows.
6. `declared_first` arm.

Existing `--sample`/`--seed` random-window sampling stays for exploratory use; the measured design
uses the manifest.
