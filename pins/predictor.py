"""
Stage-1 predictor stub (research_plan.md:47-51).

The real Stage-1 is a hybrid: a numeric forecaster for warm jobs + an LLM
cold-start profiler that reads the submission script. For the negotiation
prototype we only need the *output contract*: given a job's current phase, emit a
non-increasing marginal-value curve (value of the 1st, 2nd, ... GPU).

This stub encodes the premise of the whole project (research_plan.md:10): GPU
demand VARIES BY PHASE. A job barely wants a GPU during preprocess, wants many
during train, and something in between during eval. Swap this file for the real
hybrid predictor later — the negotiation layer above it does not change.
"""
from __future__ import annotations

# phase -> (capacity = max useful GPUs, base value of 1st GPU, diminishing-returns factor)
PHASE_PROFILES: dict[str, tuple[int, float, float]] = {
    "preprocess": (1, 1.0, 0.50),   # I/O bound: one GPU is plenty
    "train":      (8, 10.0, 0.82),  # compute bound: wants many, diminishing
    "eval":       (2, 4.0, 0.65),   # moderate
    "idle":       (0, 0.0, 0.0),    # holds nothing
}


# Max extra "safety-margin" GPUs a fully-uncertain job (uncertainty=1.0) requests beyond its
# forecast capacity. The margin is sized by the Stage-1 quantile width and rationed by the
# auction — "uncertainty sizes the margin; the auction rations it" (research_plan.md:167).
MARGIN_GPU_SCALE = 3


def marginal_values(phase: str, urgency: float = 1.0, uncertainty: float = 0.0,
                    margin_scale: int = MARGIN_GPU_SCALE) -> list[float]:
    """Return the non-increasing marginal-value curve for a job in `phase`.

    Args:
        phase: one of PHASE_PROFILES.
        urgency: per-job multiplier (deadline pressure / priority). Lets two jobs
            in the same phase value GPUs differently — this is the *private
            valuation* that makes negotiation meaningful (research_plan.md:81).
        uncertainty: Stage-1 forecast uncertainty in [0,1] (the quantile interval width from
            pins/forecast/model_quantile). It SIZES a safety margin: the job requests up to
            `round(uncertainty * margin_scale)` extra GPUs beyond its forecast capacity to
            absorb under-forecast (high-quantile) demand without missing its deadline. The
            margin GPUs continue the diminishing curve, scaled by an insurance premium
            `(1+uncertainty)` — you pay more to hedge when you are less sure. uncertainty=0
            reproduces the original curve exactly (backward compatible).
        margin_scale: GPUs of margin at full uncertainty.
    """
    cap, base, decay = PHASE_PROFILES.get(phase, PHASE_PROFILES["idle"])
    curve = [urgency * base * (decay ** k) for k in range(cap)]
    n_margin = round(uncertainty * margin_scale) if cap > 0 else 0
    for k in range(n_margin):                       # uncertainty-sized safety margin
        curve.append(urgency * base * (decay ** (cap + k)) * (1.0 + uncertainty))
    return [round(v, 4) for v in curve]
