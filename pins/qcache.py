"""Quality-aware similarity cache (elevated research plan §12).

The referee's existing cache is EXACT-MATCH on a discretised scene key: a scene either was
seen verbatim or it pays for a fresh ruling. §12 asks for something stronger — retrieve a
*similar* past decision, weight it by how GOOD that decision was and how OLD it is, adapt it
safely to the current jobs, and re-validate before executing. And it asks for the honest
counter-metric: not the hit rate (which any loose threshold inflates) but the FALSE-REUSE
rate.

  z_t     = [load, active_jobs, urgent_ratio, mean_util, util_std, mean_slack,
             fragmentation, churn]                                        (§12.1)
  sim     = cosine(z_t, z_i)                                              (§12.2)
  R_i     = sim * Q_i * exp(-lambda * age)                                (§12.2)

DECISION-TIME QUALITY (the design choice, made 2026-07-22). The plan defines
`Q_i = a*U_useful - b*SVR - c*C_resize - d*R_invalid`, i.e. on the OUTCOME of a decision. In
a 300-tick simulation an outcome is only attributable at the end of the run, so scoring on it
would make the cache un-deployable (a real scheduler cannot wait for the job to finish before
deciding whether to trust a cached ruling). We therefore score every entry on quantities that
are known the moment the ruling is made:

  fill     — the share of the free pool the ruling actually put to work, capped at each job's
             usable parallelism. The decision-time stand-in for U_useful: GPUs awarded beyond
             what a job can use are visible immediately, no outcome needed.
  invalid  — did the deterministic validator reject the ruling (rule 1-4 violation)? Known
             immediately, and the plan's own R_invalid term.
  churn    — share of jobs whose award moved vs the last executed allocation. The plan's
             C_resize term, also immediate.

  Q = W_FILL*fill - W_INVALID*invalid - W_CHURN*churn,  clipped to [0, 1]

This is weaker than outcome scoring and we say so: it cannot see an allocation that was
locally sensible and globally wrong. It is what a deployed scheduler could compute.

FALSE REUSE is measured on the same footing: a reuse is FALSE when the adapted ruling's
quality ON THE CURRENT SCENE falls more than FALSE_MARGIN below the quality the entry was
stored with — i.e. the decision travelled badly. Rejections by the validator count as false
reuses too (they were retrieved, then thrown away).

Nothing here runs unless a threshold is passed: `--qcache THR` in trace_replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

W_FILL, W_INVALID, W_CHURN = 1.0, 1.0, 0.3
AGE_LAMBDA = 0.01          # exp(-0.01*age_ticks): a 100-tick-old ruling is worth ~37% of fresh
FALSE_MARGIN = 0.15        # quality drop that makes a reuse "false"


@dataclass
class Entry:
    z: list[float]
    alloc: dict[str, int]              # jid -> margin GPUs awarded
    reserve: int
    q: float
    tick: int
    cat: dict[str, str]                # jid -> category (tier|deadline), for safe remapping


@dataclass
class QualityCache:
    """Similarity retrieval over validated rulings, with the plan's §12.3 safe adaptation."""
    threshold: float = 0.8
    entries: list[Entry] = field(default_factory=list)
    reuses: int = 0
    false_reuses: int = 0
    misses: int = 0

    # -- §12.1 state vector ------------------------------------------------------------- #
    @staticmethod
    def state(demand, free_gpus: int, total_want: int, prev: dict | None) -> list[float]:
        n = max(len(demand), 1)
        urgent = sum(1 for j in demand if j.ctx.get("deadline") == "behind") / n
        unc = [{"high": 1.0, "medium": 0.5}.get(j.ctx.get("uncertainty"), 0.0) for j in demand]
        mean_u = sum(unc) / n
        var = sum((x - mean_u) ** 2 for x in unc) / n
        slack = max(0.0, free_gpus - total_want) / max(free_gpus, 1)
        frag = 1.0 / (1.0 + free_gpus)          # a pool that is free in ones is fragmented
        churn = (sum(1 for j in demand if j.jid not in prev) / n) if prev else 0.0
        return [total_want / max(free_gpus, 1), n / 16.0, urgent, mean_u,
                var ** 0.5, slack, frag, churn]

    # -- decision-time quality ---------------------------------------------------------- #
    @staticmethod
    def quality(alloc: dict[str, int], reserve: int, demand, free_gpus: int,
                invalid: bool, prev: dict | None) -> float:
        n = max(len(demand), 1)
        usable = {j.jid: max(0, getattr(j, "forecast_cap", 0) or 1) for j in demand}
        used = sum(min(alloc.get(j.jid, 0), usable[j.jid] + 1) for j in demand)
        fill = min(1.0, used / max(free_gpus, 1))
        churn = (sum(1 for j in demand if alloc.get(j.jid, 0) != prev.get(j.jid, 0))
                 / n) if prev else 0.0
        return max(0.0, min(1.0, W_FILL * fill - W_INVALID * float(invalid) - W_CHURN * churn))

    # -- §12.2 retrieval ---------------------------------------------------------------- #
    def retrieve(self, z: list[float], tick: int) -> tuple[Entry, float] | None:
        best, best_r = None, 0.0
        for e in self.entries:
            s = _cos(z, e.z)
            r = s * e.q * math.exp(-AGE_LAMBDA * max(0, tick - e.tick))
            if r > best_r:
                best, best_r = e, r
        if best is None or best_r < self.threshold:
            self.misses += 1
            return None
        return best, best_r

    # -- §12.3 safe adaptation ---------------------------------------------------------- #
    @staticmethod
    def adapt(entry: Entry, demand, free_gpus: int) -> dict[str, int]:
        """Map the cached award by job CATEGORY, never by raw id: same tier and same deadline
        bucket inherits the same margin. Jobs whose category is absent from the entry get 0 —
        the conservative direction, since a margin is the surplus and the base is untouched by
        this layer. The award is then clipped to what is actually free."""
        by_cat: dict[str, list[int]] = {}
        for jid, g in entry.alloc.items():
            by_cat.setdefault(entry.cat.get(jid, "?"), []).append(g)
        mean = {c: round(sum(v) / len(v)) for c, v in by_cat.items()}
        out, budget = {}, max(0, free_gpus - entry.reserve)
        for j in sorted(demand, key=lambda j: 0 if j.ctx.get("tier") == "prod" else 1):
            g = min(mean.get(category(j), 0), budget)
            out[j.jid] = g
            budget -= g
        return out

    def store(self, z, alloc, reserve, q, tick, demand) -> None:
        self.entries.append(Entry(list(z), dict(alloc), int(reserve), float(q), int(tick),
                                  {j.jid: category(j) for j in demand}))

    def note_reuse(self, stored_q: float, live_q: float) -> None:
        self.reuses += 1
        if live_q < stored_q - FALSE_MARGIN:
            self.false_reuses += 1

    def stats(self) -> dict:
        tot = self.reuses + self.misses
        return {"qcache_reuse": self.reuses / max(tot, 1),
                "qcache_false": self.false_reuses / max(self.reuses, 1),
                "qcache_entries": float(len(self.entries))}


def category(j) -> str:
    return f"{j.ctx.get('tier','?')}|{j.ctx.get('deadline','?')}"


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0
