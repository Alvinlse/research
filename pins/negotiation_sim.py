"""
Stage-2 simulation: which ALLOCATION MECHANISM rations scarce GPUs best?

This is the headline experiment for the negotiation contribution (research_plan.md,
thesis refocus 2026-06-17): *break the utilisation/SLA trade-off — beat value-blind
schedulers on SLA-violation rate at high utilisation.* It is also the load-bearing
ablation: "kill the auction (use a value-blind scheduler) and SLA should worsen."

What it does
------------
A seeded stream of jobs contends for a fixed GPU pool over discrete time. Each job has
a phase timeline (preprocess -> train -> ... -> eval), a private **urgency**, and a
**deadline**. Urgency does double duty — the spine of the thesis argument:

  * it scales the job's **private value** (predictor.marginal_values), so urgent jobs BID
    higher, and
  * it tightens the job's **deadline**, so urgent jobs have less slack to miss.

A job advances through a phase at rate ``min(allocated, capacity) / capacity`` per step,
so under-allocation literally slows it down. SLA is violated if the job finishes after its
deadline (or not at all within the horizon).

We run the SAME workload through several schedulers and compare. The PINS auction
(mechanism.clear) is the only value-aware one; the rest are classical value-blind foils.
The contention knob is the pool size — sweeping it traces SLA-vs-utilisation.

Design hinge preserved: the auctioneer is the pure decider (pins/mechanism.py); here we
only wrap baselines with the SAME signature and a deterministic simulator around them. No
LLM, no MCP, no network — runs in the base .venv, instantly and reproducibly.

Run:  .venv/bin/python -m pins.negotiation_sim
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

from pins.ilp import allocate as ilp_allocate
from pins.mechanism import clear, welfare
from pins.predictor import PHASE_PROFILES, marginal_values

# A scheduler is: (bids, total_gpus, current_alloc) -> new_alloc.
Bids = dict[str, list[float]]
Alloc = dict[str, int]


# --------------------------------------------------------------------------- #
#  Workload                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    """One simulated job with a phase timeline and a deadline."""
    jid: str
    arrival: int
    phases: list[str]            # e.g. ["preprocess", "train", "train", "eval"]
    need: list[float]            # full-speed steps each phase needs (work, in time units)
    urgency: float               # private valuation multiplier AND deadline pressure
    deadline: int                # absolute step by which the job must finish
    tier: str = "besteffort"     # priority tier {prod, besteffort} for the tiered-SLA metric

    # runtime state ----------------------------------------------------------
    phase_idx: int = 0
    progress: float = 0.0        # full-speed-steps accumulated in the current phase
    done_at: int | None = None

    def active(self, t: int) -> bool:
        return self.arrival <= t and self.done_at is None

    def remaining(self) -> float:
        """Full-speed work still to do across the rest of the timeline."""
        return sum(self.need[self.phase_idx:]) - self.progress

    def deadline_bucket(self, t: int) -> str:
        """How the job is tracking vs its deadline: behind / ontrack / ahead."""
        ratio = self.remaining() / max(1, self.deadline - t)
        return "behind" if ratio > 1.0 else "ahead" if ratio < 0.6 else "ontrack"

    @property
    def nominal(self) -> float:
        """Duration at full allocation (the fastest the job could possibly run)."""
        return float(sum(self.need))

    def phase(self) -> str:
        return self.phases[self.phase_idx]

    def bid(self) -> list[float]:
        """The job's marginal-value curve for its current phase (urgency-scaled)."""
        return marginal_values(self.phase(), self.urgency)

    def capacity(self) -> int:
        """Max useful GPUs this phase (= bid-curve length = phase profile capacity)."""
        return PHASE_PROFILES[self.phase()][0]

    def step(self, gpus: int, t: int) -> None:
        """Advance the job one tick given `gpus` allocated this step."""
        cap = self.capacity()
        rate = 1.0 if cap == 0 else min(gpus, cap) / cap     # 0..1 fraction of full speed
        self.progress += rate
        while self.done_at is None and self.progress >= self.need[self.phase_idx] - 1e-9:
            self.progress -= self.need[self.phase_idx]
            self.phase_idx += 1
            if self.phase_idx >= len(self.phases):
                self.done_at = t
                break


def make_workload(n_jobs: int, seed: int, horizon: int) -> list[Job]:
    """A reproducible mix of urgent and relaxed jobs arriving over the first ~60% of time.

    Urgency in [0.6, 2.2]; train-heavy timelines (the contended phase). Deadline slack
    shrinks with urgency: urgent jobs get ~1.2x nominal, relaxed jobs ~2.3x — so the
    deadline genuinely encodes priority, which is exactly what a value-aware scheduler
    can exploit and a value-blind one cannot.
    """
    rng = random.Random(seed)
    jobs: list[Job] = []
    last_arrival = int(horizon * 0.6)
    for i in range(n_jobs):
        urgency = round(rng.uniform(0.6, 2.2), 3)
        n_train = rng.randint(2, 5)                          # train-heavy: the scarce phase
        phases = ["preprocess"] + ["train"] * n_train + ["eval"]
        need = [float(rng.randint(1, 2))]                    # preprocess: short
        need += [float(rng.randint(2, 4)) for _ in range(n_train)]   # train phases
        need += [float(rng.randint(1, 3))]                   # eval
        arrival = rng.randint(0, last_arrival)
        nominal = sum(need)
        slack = 2.5 - 0.65 * urgency                         # urgent -> tight, relaxed -> loose
        slack = max(1.15, min(2.4, slack))
        deadline = arrival + int(round(nominal * slack))
        tier = "prod" if urgency >= 1.667 else "besteffort"  # top third of [0.6,2.2] = production
        jobs.append(Job(f"j{i:02d}", arrival, phases, need, urgency, deadline, tier))
    return jobs


# --------------------------------------------------------------------------- #
#  Bid builders — how a job turns its state into a marginal-value curve         #
# --------------------------------------------------------------------------- #
# Bid builders take (job, t, market); `market` = {"free_gpus","total_gpus","contention"}.
# The deterministic builders ignore `market`; only the LLM strategist reads it.
def static_bid(job: Job, t: int, market: dict) -> list[float]:
    """Urgency-only bid: the predictor curve, scaled by the job's fixed urgency. The bid
    does NOT change as the deadline nears — the naive value-maximising case."""
    return job.bid()


def deadline_bid(job: Job, t: int, market: dict) -> list[float]:
    """Deadline-aware bid: scale the static curve by how far BEHIND SCHEDULE the job is.

    pressure = (full-speed work still to do) / (time left to deadline). pressure > 1 means
    the job cannot finish even at full speed unless it gets GPUs NOW, so it bids up; a job
    with comfortable slack bids down and yields. This makes the same auction ration by
    deadline risk (a value-weighted earliest-deadline-first), which is what SLA actually
    rewards — vs static_bid, which only chases raw value."""
    ttl = max(1, job.deadline - t)
    mult = min(10.0, max(0.5, job.remaining() / ttl))
    return [round(v * mult, 4) for v in job.bid()]


def make_llm_bidder(cache: dict, trace: list, use_llm: bool, model: str):
    """Build an LLM-strategist bid builder (Exp 10). The LLM reads the discretised state
    (predicted workload = phase capacity, deadline bucket, tier, market contention) and returns
    a {stance, focus_gpus} decision + justification, cached per state; deterministic code applies
    it to the CALIBRATED baseline curve. The LLM never emits a number — see pins/llm_agent.py."""
    from pins.llm_agent import apply_strategy, llm_strategy, state_key

    seen: set[str] = set()

    def llm_bid(job: Job, t: int, market: dict) -> list[float]:
        ctx = {"phase": job.phase(), "capacity": job.capacity(),
               "deadline": job.deadline_bucket(t), "contention": market["contention"],
               "tier": job.tier}
        if job.capacity() == 0:
            return job.bid()
        strat = llm_strategy(ctx, use_llm=use_llm, model=model, cache=cache)
        key = state_key(ctx)
        if key not in seen:                              # keep one justification per distinct state
            seen.add(key)
            trace.append({"state": key, **{k: strat[k] for k in
                          ("stance", "focus_gpus", "justification", "_source")}})
        return apply_strategy(job.bid(), strat["stance"], strat["focus_gpus"])

    return llm_bid


# --------------------------------------------------------------------------- #
#  Schedulers (all share the signature; only `auction` is value-aware)         #
# --------------------------------------------------------------------------- #
# Anti-thrashing cost charged per GPU that changes hands (mechanism.py step 4). Non-zero so
# the auction does not churn allocations every round — stability is what lets a job actually
# RUN to completion, which is exactly what SLA rewards. This is a built-in part of PINS, not a
# tuning knob added for this sim; it is set comparable to a train-phase marginal value.
RESCALE_COST = 2.0


def sched_auction(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
    """PINS: sealed-bid uniform-price auction. Rations by marginal VALUE (=urgency), with the
    anti-thrashing gate ON so it does not thrash GPUs between jobs every round."""
    return clear(bids, total_gpus, current=current, rescale_cost=RESCALE_COST).allocation


def sched_ilp(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
    """LLMSched-style ILP decider (Open-Question #1, arm b). Consumes the SAME negotiated bids
    as sched_auction and maximises welfare under the SAME per-GPU rescale penalty — but solves
    it as a MILP, so it can do FINE-GRAINED partial preemption instead of the auction's
    all-or-nothing anti-thrash gate. See pins/ilp.py."""
    return ilp_allocate(bids, total_gpus, current=current, rescale_cost=RESCALE_COST).allocation


def sched_greedy(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
    """Greedy-by-demand, FIFO. Serve jobs in id order; each takes its full capacity
    until the pool is empty. Classic backfill — value-blind, order = arrival/queue."""
    alloc = {a: 0 for a in bids}
    left = total_gpus
    for a in sorted(bids):                                   # stable queue order
        want = min(len(bids[a]), left)
        alloc[a] = want
        left -= want
        if left <= 0:
            break
    return alloc


def sched_equal(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
    """Equal/fair share. Split the pool evenly across active jobs (capped at each job's
    capacity); hand any remainder out round-robin. Fairness-based, value-blind."""
    agents = [a for a in sorted(bids) if len(bids[a]) > 0]
    alloc = {a: 0 for a in bids}
    if not agents:
        return alloc
    left = total_gpus
    share = max(1, total_gpus // len(agents))
    for a in agents:
        give = min(share, len(bids[a]), left)
        alloc[a] = give
        left -= give
    for a in agents:                                         # distribute remainder
        if left <= 0:
            break
        if alloc[a] < len(bids[a]):
            alloc[a] += 1
            left -= 1
    return alloc


def sched_static(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
    """Static / sticky FIFO. A job KEEPS what it already holds (no preemption); only
    free GPUs go to jobs holding none, in queue order. Models a non-elastic scheduler —
    the foil for PINS's live rescaling."""
    alloc = {a: min(current.get(a, 0), len(bids[a])) for a in bids}
    left = total_gpus - sum(alloc.values())
    for a in sorted(bids):                                   # give spare GPUs to the unserved
        if left <= 0:
            break
        if alloc[a] == 0 and len(bids[a]) > 0:
            give = min(len(bids[a]), left)
            alloc[a] = give
            left -= give
    return alloc


def make_stable_auction(bonus: float):
    """Auction with INCUMBENCY (the Exp-11 stability lever). GPUs a job currently holds get
    `+bonus` added to their marginal value when clearing, so a challenger must outbid
    `incumbent_value + bonus` to displace a running job. bonus = switching/disruption cost.

    bonus -> 0 recovers the thrashing welfare-max auction; bonus -> large makes allocations
    fully sticky (greedy-like run-to-completion). The boost only steers the SORT; welfare/SLA
    are still scored on the true bids by the simulator. mechanism.clear stays untouched."""
    def sched(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
        boosted = {a: [v + (bonus if k < current.get(a, 0) else 0.0)
                       for k, v in enumerate(curve)]
                   for a, curve in bids.items()}
        return clear(boosted, total_gpus, current=current, rescale_cost=0.0).allocation
    return sched


def _serialise(order: list[str], bids: Bids, total_gpus: int) -> Alloc:
    """Run-to-completion block allocation: in `order`, give each job its full GPU block until
    the pool is empty. Concentration (full capacity per job) + a stable `order` is what beats
    the spreading per-round auction on deadlines (Exp 11)."""
    alloc = {a: 0 for a in bids}
    left = total_gpus
    for a in order:
        want = min(len(bids[a]), left)
        alloc[a] = want
        left -= want
        if left <= 0:
            break
    return alloc


def make_committed_auction():
    """COMMITTED auction — the Exp-11 winner. Each agent's priority is fixed by its FIRST bid
    (bid-once, on arrival), then the orchestrator SERIALISES by that frozen priority.

    Why this and not the per-round marginal auction: deadline-meeting wants (a) concentration —
    full capacity to one job so it finishes, not GPUs spread thin so everyone runs slow — and
    (b) a STABLE order, because re-pricing every round flips who is at the front and re-thrashes
    (Exp 11 value-block). Freezing priority gives both. Priority = the job's initial declared
    value (urgency-scaled), so urgent/prod jobs are served first. The single bid is the
    'negotiation'; `make_llm_committed` swaps this for an LLM-set, justified priority (Exp 12)."""
    frozen: dict[str, float] = {}

    def sched(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
        for a, curve in bids.items():
            frozen.setdefault(a, sum(curve))          # bid-once: priority set on first appearance
        order = sorted(bids, key=lambda a: (-frozen.get(a, 0.0), a))
        return _serialise(order, bids, total_gpus)

    return sched


def make_llm_committed(cache: dict, trace: list, use_llm: bool, model: str):
    """The INTERPRETABLE committed auction (Exp 12): an LLM sets each job's serialisation
    PRIORITY once, on arrival, from its intrinsic profile (tier / deadline tightness / size) —
    as an ordinal CLASS, never a number. Returns the (bid_builder, allocator_factory) pair that
    share one frozen-priority map: the bid builder freezes priority via the LLM the first time
    it sees a job and emits the plain calibrated bid; the allocator serialises by that priority.
    The LLM touches only the ORDER (+ a justification); code owns every magnitude."""
    from pins.llm_agent import llm_priority, priority_state_key, priority_weight

    frozen: dict[str, float] = {}
    seen: set[str] = set()

    def bid_builder(job: Job, t: int, market: dict) -> list[float]:
        if job.jid not in frozen:                     # bid-once: set priority on arrival
            slack = (job.deadline - job.arrival) / max(job.nominal, 1e-9)
            ctx = {"tier": job.tier,
                   "deadline": "tight" if slack < 1.5 else "loose",
                   "size": "small" if job.nominal <= 10 else "large"}
            p = llm_priority(ctx, use_llm=use_llm, model=model, cache=cache)
            frozen[job.jid] = priority_weight(p["priority"])
            key = priority_state_key(ctx)
            if key not in seen:
                seen.add(key)
                trace.append({"state": key, "priority": p["priority"],
                              "justification": p["justification"], "_source": p["_source"]})
        return job.bid()

    def alloc_factory():
        def sched(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
            order = sorted(bids, key=lambda a: (-frozen.get(a, 0.0), a))
            return _serialise(order, bids, total_gpus)
        return sched

    return bid_builder, alloc_factory


# --------------------------------------------------------------------------- #
#  Incentives (Exp 13): what if jobs LIE about their priority?                  #
# --------------------------------------------------------------------------- #
# The committed-auction trusts each job's DECLARED priority. These declare-fns let a job
# report something other than the truth; the metric still scores TRUE prod jobs.
PRIO_CLASS_COST = {"critical": 4.0, "high": 2.0, "normal": 1.0, "low": 0.0}


def _truthful_class(job: Job) -> str:
    tight = (job.deadline - job.arrival) / max(job.nominal, 1e-9) < 1.5
    if job.tier == "prod":
        return "critical" if tight else "high"
    return "normal" if tight else "low"


def declare_truthful(job: Job) -> str:
    return _truthful_class(job)


def declare_inflate_all(job: Job) -> str:
    """Every job lies its way to the top."""
    return "critical"


def declare_inflate_besteffort(job: Job) -> str:
    """Selfish best-effort jobs claim 'critical'; prod jobs stay honest."""
    return "critical" if job.tier == "besteffort" else _truthful_class(job)


def make_declared_committed(declare_fn, budget: float | None = None):
    """Committed auction whose priority comes from each job's DECLARED class (possibly a lie).

    With `budget` set, a job pays `PRIO_CLASS_COST[class]` from an equal budget EVERY tick it is
    active (it pays for the priority it CLAIMS, served or not). When the budget runs dry the job
    is demoted to the bottom. An honest urgent job finishes fast -> short residence -> affordable;
    a job that lies 'critical' on a long, loose workload pays tick after tick -> runs out -> loses
    the false priority. This is the incentive layer that makes over-claiming self-defeating."""
    from pins.llm_agent import priority_weight

    cls: dict[str, str] = {}
    weight: dict[str, float] = {}
    bud: dict[str, float] = {}

    def bid_builder(job: Job, t: int, market: dict) -> list[float]:
        if job.jid not in cls:
            c = declare_fn(job)
            cls[job.jid], weight[job.jid] = c, priority_weight(c)
            if budget is not None:
                bud[job.jid] = budget
        return job.bid()

    def alloc_factory():
        def sched(bids: Bids, total_gpus: int, current: Alloc) -> Alloc:
            def eff(a: str) -> float:
                if budget is not None and bud.get(a, 1.0) <= 0:
                    return 0.0                              # insolvent -> demoted to the bottom
                return weight.get(a, 2.0)
            order = sorted(bids, key=lambda a: (-eff(a), a))
            alloc = _serialise(order, bids, total_gpus)
            if budget is not None:                          # charge every active job for its CLAIM
                for a in bids:
                    bud[a] = bud.get(a, 0.0) - PRIO_CLASS_COST.get(cls.get(a, "normal"), 1.0)
            return alloc
        return sched

    return bid_builder, alloc_factory


# A strategy pairs a BID BUILDER (how a job values GPUs) with an ALLOCATOR FACTORY (how the pool
# is split). The factory is called fresh per run so stateful allocators (committed-auction freezes
# per-job priority) start clean. The bid builder only matters for the value-aware allocators; the
# value-blind baselines read the curve for capacity/active-set only.
#   PINS-auction      — per-round marginal auction, static urgency (thrashes/spreads; Exp 9)
#   PINS-auct-DL      — per-round marginal auction, deadline-pressure bids (Exp 9)
#   committed-auction — bid-once priority + serialised run-to-completion (Exp 11 winner on prodSLA)
STRATEGIES = [
    ("PINS-auction",      static_bid,   lambda: sched_auction),
    ("PINS-auct-DL",      deadline_bid, lambda: sched_auction),
    ("ILP-welfare",       static_bid,   lambda: sched_ilp),
    ("ILP-DL",            deadline_bid, lambda: sched_ilp),
    ("committed-auction", static_bid,   make_committed_auction),
    ("greedy-FIFO",       static_bid,   lambda: sched_greedy),
    ("equal-share",       static_bid,   lambda: sched_equal),
    ("static-sticky",     static_bid,   lambda: sched_static),
]


# --------------------------------------------------------------------------- #
#  Simulator + metrics                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    sla_violation_rate: float    # fraction of ALL jobs finishing after their deadline / never
    prod_sla_rate: float         # violation rate among 'prod'-tier jobs (the value-weighted view)
    utilisation: float           # mean busy-GPU fraction while any job is active
    welfare: float               # total value realised over the run (auction's objective)
    mean_slowdown: float         # finished jobs: actual / nominal duration (1.0 = ideal)
    finished: int
    n_jobs: int


def simulate(jobs_proto: list[Job], bid_builder, allocator, total_gpus: int,
             horizon: int) -> Result:
    """Run one strategy (bid_builder + allocator) on a fresh copy of the workload.

    `bid_builder(job, t, market)` produces the curve used for ALLOCATION (deadline/LLM builders
    reshape it); the welfare METRIC is always scored on the static base curve `job.bid()` so
    welfare is comparable across strategies and not gamed by a bid multiplier."""
    jobs = [Job(j.jid, j.arrival, list(j.phases), list(j.need), j.urgency, j.deadline, j.tier)
            for j in jobs_proto]                             # deep-enough copy of mutable state
    by_id = {j.jid: j for j in jobs}
    current: Alloc = {}
    busy_sum = 0.0
    busy_steps = 0
    total_welfare = 0.0

    for t in range(horizon):
        active = [j for j in jobs if j.active(t)]
        if not active:
            current = {}
            if all(j.done_at is not None for j in jobs) and any(j.arrival <= t for j in jobs):
                break
            continue
        # Market snapshot for the strategist: is aggregate demand above supply right now?
        demand = sum(j.capacity() for j in active)
        used = sum(current.get(j.jid, 0) for j in active)
        market = {"total_gpus": total_gpus, "free_gpus": total_gpus - used,
                  "contention": "high" if demand > total_gpus else "low"}
        bids = {j.jid: bid_builder(j, t, market) for j in active}   # allocation bids
        base = {j.jid: j.bid() for j in active}              # static bids for the welfare metric
        cur = {jid: current.get(jid, 0) for jid in bids}
        alloc = allocator(bids, total_gpus, cur)
        total_welfare += welfare(base, alloc)
        busy_sum += sum(alloc.values()) / total_gpus
        busy_steps += 1
        for j in active:
            j.step(alloc.get(j.jid, 0), t)
        current = {jid: alloc.get(jid, 0) for jid in by_id}  # carry alloc for sticky/anti-thrash

    finished = [j for j in jobs if j.done_at is not None]
    def violated(j: Job) -> bool:
        return j.done_at is None or j.done_at > j.deadline
    violations = sum(1 for j in jobs if violated(j))
    prod = [j for j in jobs if j.tier == "prod"]
    prod_viol = sum(1 for j in prod if violated(j))
    slow = [(j.done_at - j.arrival) / j.nominal for j in finished if j.nominal > 0]
    return Result(
        sla_violation_rate=violations / len(jobs),
        prod_sla_rate=prod_viol / max(len(prod), 1),
        utilisation=busy_sum / max(busy_steps, 1),
        welfare=total_welfare,
        mean_slowdown=sum(slow) / max(len(slow), 1),
        finished=len(finished),
        n_jobs=len(jobs),
    )


def sweep(n_jobs: int, horizon: int, pools: list[int], seed: int, strategies=None) -> None:
    """Run every strategy at every pool size on the same workload; print the table."""
    strategies = strategies or STRATEGIES
    jobs = make_workload(n_jobs, seed, horizon)
    n_prod = sum(1 for j in jobs if j.tier == "prod")
    print(f"workload: {n_jobs} jobs ({n_prod} prod / {n_jobs - n_prod} besteffort), "
          f"horizon={horizon} steps, seed={seed} | urgency in [0.6,2.2], train-heavy\n")
    print("Smaller pool = higher utilisation = more contention. "
          "SLA = all jobs; prodSLA = prod-tier only.\n")
    header = (f"{'pool':>4}  {'strategy':<22} {'SLA-viol':>9} {'prodSLA':>8} "
              f"{'util':>6} {'welfare':>9} {'slowdown':>9} {'done':>7}")
    for gpus in pools:
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        rows = [(name, simulate(jobs, bb, alf(), gpus, horizon)) for name, bb, alf in strategies]
        best_sla = min(r.sla_violation_rate for _, r in rows)
        best_prod = min(r.prod_sla_rate for _, r in rows)
        for name, r in rows:
            star = "*" if r.sla_violation_rate == best_sla else " "
            pstar = "*" if r.prod_sla_rate == best_prod else " "
            print(f"{gpus:>4}  {name:<22} {r.sla_violation_rate:>8.1%}{star}"
                  f"{r.prod_sla_rate:>7.1%}{pstar}{r.utilisation:>6.0%} {r.welfare:>9.0f} "
                  f"{r.mean_slowdown:>9.2f} {r.finished:>3}/{r.n_jobs:<3}")
        print()


def main() -> None:
    import argparse
    from pins.llm_agent import DEFAULT_MODEL, save_cache

    ap = argparse.ArgumentParser(description="Stage-2 allocation-strategy sweep")
    ap.add_argument("--llm", action="store_true",
                    help="add an LLM-strategist bidding row (Exp 10); needs Ollama")
    ap.add_argument("--no-llm", action="store_true",
                    help="with --llm: use the rule fallback instead of calling Ollama")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for the LLM strategist")
    a = ap.parse_args()

    strategies = list(STRATEGIES)
    cache, trace = {}, []
    prio_trace = []
    if a.llm:
        use_llm = not a.no_llm
        tag = "rule" if not use_llm else a.model
        bidder = make_llm_bidder(cache, trace, use_llm=use_llm, model=a.model)
        # Per-round LLM strategist (Exp 10) right after the two formula auctions.
        strategies.insert(2, (f"llm-strategic({tag})", bidder, lambda: sched_auction))
        # LLM-set committed priority (Exp 12) right after the deterministic committed-auction.
        llm_bb, llm_alf = make_llm_committed(cache, prio_trace, use_llm=use_llm, model=a.model)
        idx = next(i for i, s in enumerate(strategies) if s[0] == "committed-auction") + 1
        strategies.insert(idx, (f"llm-committed({tag})", llm_bb, llm_alf))

    # Right-sized so the system can CLEAR at large pools (SLA -> low) but is genuinely
    # contended at small ones (SLA -> high): a train phase wants up to 8 GPUs, so pools
    # straddling 8 with a few concurrent jobs trace the SLA-vs-utilisation curve cleanly.
    sweep(n_jobs=16, horizon=300, pools=[4, 6, 8, 12, 20], seed=0, strategies=strategies)
    print("'*' = best (lowest) violation rate at that pool size (SLA = all, prodSLA = prod tier).")
    print("PINS-auction = static urgency; PINS-auct-DL = deadline-pressure bids.")

    if a.llm:
        from pins.llm_agent import CACHE_PATH
        save_cache(cache)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_llm_negotiation.json")
        with open(out, "w") as f:
            json.dump({"model": a.model, "use_llm": not a.no_llm,
                       "strategist_states": trace, "committed_priorities": prio_trace},
                      f, indent=2)
        print(f"\nLLM: {len(trace)} strategist states + {len(prio_trace)} committed priorities "
              f"decided (cache {CACHE_PATH}); justification trace -> {out}")


if __name__ == "__main__":
    main()
