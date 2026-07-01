"""
Stage-1 -> Stage-2 TEXT BRIDGE (build task #1).

The Stage-1 evals emit calibrated NUMBERS per job: a runtime interval (P10/P50/P90 minutes,
the retrieval quantiles of Exp 19), a peak GPU footprint (the level-sum rule of Exp 21), and a
forecast uncertainty scalar (the conformal quantile width of model_quantile.py). The Stage-2 LLM
agents (pins/llm_agent.py) do NOT consume numbers — by the project hinge (Exp 1-7: an LLM cannot
calibrate a magnitude) they reason over QUALITATIVE buckets. This module is the missing connective
tissue: it turns the Stage-1 numbers into (a) the bucketed `ctx` dicts each agent already expects,
and (b) a one-line natural-language FACT STRING for the prompt — the literal "text bridge".

No LLM, no magnitude invented here: every threshold is deterministic code (the "code decides"
half). Reuses the discretisers already shipped in llm_agent so a cached LLM answer stays valid.

Run:  .venv/bin/python -m pins.bridge        # smoke: a few Stage-1 fact sets -> ctx + text
"""
from __future__ import annotations

from dataclasses import dataclass

from pins.llm_agent import spike_risk_bucket, uncertainty_bucket

# Runtime magnitude that counts as a "large" job (min). Supercloud median runtime ~163 min
# (Exp 19), so a job materially longer than the median is large for priority purposes.
LARGE_RUNTIME_MIN = 180.0
# Deadline-tracking thresholds (mirror negotiation_sim.Job.deadline_bucket so the bridged bucket
# matches the simulator's own definition exactly).
BEHIND_RATIO = 1.0
AHEAD_RATIO = 0.6
# Deadline-slack threshold for the committed-priority "tight/loose" axis (mirror make_llm_committed).
TIGHT_SLACK = 1.5


@dataclass
class Stage1Facts:
    """The calibrated Stage-1 outputs for ONE job (the bridge's input contract).

    runtime_p10/p50/p90: predicted runtime quantiles in minutes (Exp 19 retrieval).
    peak_gpu_gb:         predicted peak-concurrent GPU memory (Exp 21 level-sum rule).
    uncertainty:         per-job forecast uncertainty in [0,1] (model_quantile, conformal width).
                         If None it is derived from the runtime interval below.
    tier:                priority tier {prod, besteffort} (job metadata, not predicted).
    """
    jid: str
    runtime_p50: float
    runtime_p10: float | None = None
    runtime_p90: float | None = None
    peak_gpu_gb: float | None = None
    uncertainty: float | None = None
    tier: str = "besteffort"
    req_gpu: int | None = None      # Stage-1 predicted requested GPU count (predict_gpu P50)

    def severity(self) -> float:
        """Spike-risk SEVERITY = plausible relative over-run of runtime above the median,
        (P90-P50)/P50 — the heavy-tail signal Exp 17 found the agent needed. 0 if no P90."""
        if self.runtime_p90 is None or self.runtime_p50 <= 0:
            return 0.0
        return max(0.0, (self.runtime_p90 - self.runtime_p50) / self.runtime_p50)

    def u(self) -> float:
        """The uncertainty scalar: the model_quantile value if given, else the normalised
        runtime interval half-width (P90-P10)/(2*P50) clipped to [0,1]."""
        if self.uncertainty is not None:
            return float(max(0.0, min(1.0, self.uncertainty)))
        if self.runtime_p10 is None or self.runtime_p90 is None or self.runtime_p50 <= 0:
            return 0.0
        return float(max(0.0, min(1.0, (self.runtime_p90 - self.runtime_p10) / (2 * self.runtime_p50))))


# --------------------------------------------------------------------------- #
#  Numeric facts -> qualitative buckets (deterministic)                         #
# --------------------------------------------------------------------------- #
def deadline_bucket(remaining_min: float, minutes_to_deadline: float) -> str:
    """behind / ontrack / ahead, from PREDICTED remaining runtime vs time left to the deadline.
    This is the bridged version of Job.deadline_bucket — there it used true remaining work; here
    it uses the Stage-1 P50 estimate, so the agent reasons over a forecast, not an oracle."""
    ratio = remaining_min / max(1.0, minutes_to_deadline)
    return "behind" if ratio > BEHIND_RATIO else "ahead" if ratio < AHEAD_RATIO else "ontrack"


def slack_bucket(runtime_p50: float, minutes_to_deadline: float) -> str:
    """tight / loose for the committed-priority axis: how much deadline head-room vs P50 runtime."""
    return "tight" if minutes_to_deadline < TIGHT_SLACK * max(runtime_p50, 1e-9) else "loose"


def size_bucket(facts: Stage1Facts) -> str:
    """small / large workload, from predicted P50 runtime (the dominant cost signal)."""
    return "large" if facts.runtime_p50 >= LARGE_RUNTIME_MIN else "small"


def gpu_bucket(req_gpu: int | None) -> str:
    """small / medium / large from the PREDICTED requested GPU count (predict_gpu P50). A larger
    ask is harder to satisfy AND harder to add margin to under contention — the demand agent uses
    this to temper how aggressively it hedges. None (no prediction) -> 'medium' (neutral)."""
    if req_gpu is None:
        return "medium"
    return "small" if req_gpu <= 1 else "large" if req_gpu >= 4 else "medium"


def incoming_prod_bucket(n_incoming_prod: int) -> str:
    """none / few / many — the supply agent's predicted-load signal (matches llm_reserve)."""
    return "none" if n_incoming_prod == 0 else "few" if n_incoming_prod <= 2 else "many"


def contention_bucket(demand_gpus: float, total_gpus: int) -> str:
    """ample / moderate / scarce from aggregate demand vs pool (matches supply_sim's reserve gate)."""
    ratio = demand_gpus / max(total_gpus, 1)
    return "scarce" if ratio >= 1.8 else "moderate" if ratio >= 1.0 else "ample"


# --------------------------------------------------------------------------- #
#  ctx builders — assemble the exact dict each llm_agent function expects        #
# --------------------------------------------------------------------------- #
def margin_ctx(facts: Stage1Facts, deadline: str, contention: str) -> dict:
    """ctx for llm_margin / _rule_margin (demand-side hedge). `contention` here is the
    coarse high/low the demand agent sees (spare capacity or not)."""
    return {"uncertainty": uncertainty_bucket(facts.u()),
            "spike_risk": spike_risk_bucket(facts.severity()),
            "req_gpu": gpu_bucket(facts.req_gpu),
            "deadline": deadline, "contention": contention, "tier": facts.tier}


def priority_ctx(facts: Stage1Facts, minutes_to_deadline: float) -> dict:
    """ctx for llm_priority (committed-auction serialisation class)."""
    return {"tier": facts.tier,
            "deadline": slack_bucket(facts.runtime_p50, minutes_to_deadline),
            "size": size_bucket(facts)}


def reserve_ctx(contention: str, n_incoming_prod: int) -> dict:
    """ctx for llm_reserve (supply-side headroom). Supply-side facts are cluster-level, not
    per-job, so the bridge forwards the live contention + the predicted incoming-prod count."""
    return {"contention": contention, "incoming_prod": incoming_prod_bucket(n_incoming_prod)}


# --------------------------------------------------------------------------- #
#  The literal text bridge: numbers -> one NL sentence for a prompt             #
# --------------------------------------------------------------------------- #
def facts_to_text(facts: Stage1Facts) -> str:
    """A one-line natural-language summary of the Stage-1 facts, for an LLM prompt or a transcript.
    States the numbers AND their qualitative reading so the (number-blind) LLM has both."""
    parts = [f"Job {facts.jid} ({facts.tier})"]
    rt = f"predicted runtime ~{facts.runtime_p50:.0f} min"
    if facts.runtime_p10 is not None and facts.runtime_p90 is not None:
        rt += (f" (P10 {facts.runtime_p10:.0f}–P90 {facts.runtime_p90:.0f}, "
               f"up to ~{1 + facts.severity():.1f}x the median — {spike_risk_bucket(facts.severity())} spike risk)")
    parts.append(rt)
    if facts.req_gpu is not None:
        parts.append(f"predicted request ~{facts.req_gpu} GPU ({gpu_bucket(facts.req_gpu)})")
    if facts.peak_gpu_gb is not None:
        parts.append(f"peak GPU ~{facts.peak_gpu_gb:.1f} GB")
    parts.append(f"forecast uncertainty {uncertainty_bucket(facts.u())}")
    return "; ".join(parts) + "."


def main() -> None:
    probes = [
        Stage1Facts("j01", runtime_p50=420, runtime_p10=300, runtime_p90=900,
                    peak_gpu_gb=18.0, uncertainty=0.41, tier="prod"),
        Stage1Facts("j02", runtime_p50=120, runtime_p10=110, runtime_p90=140,
                    peak_gpu_gb=4.0, uncertainty=0.05, tier="besteffort"),
        Stage1Facts("j03", runtime_p50=163, runtime_p10=80, runtime_p90=410,
                    peak_gpu_gb=6.0, tier="besteffort"),   # uncertainty derived from the interval
    ]
    print("=== Stage-1 facts -> text + bridged ctx ===\n")
    for f in probes:
        print(facts_to_text(f))
        # assume the job has used ~30% of its budget and a deadline at 2x its P50 runtime
        ttl = 2 * f.runtime_p50 - 0.3 * f.runtime_p50
        print(f"   margin_ctx   -> {margin_ctx(f, deadline_bucket(0.7 * f.runtime_p50, ttl), 'high')}")
        print(f"   priority_ctx -> {priority_ctx(f, ttl)}")
        print(f"   u={f.u():.3f}  severity={f.severity():.3f}\n")
    print(f"reserve_ctx(scarce, 3 incoming) -> {reserve_ctx('scarce', 3)}")


if __name__ == "__main__":
    main()
