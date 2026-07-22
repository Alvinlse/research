"""The explicit GPU market (elevated research plan §6) — a new allocator arm.

Until now the live simulator (`two_sided_sim`) never ran a market. Its bid entered exactly
once, at `frozen = {j.jid: sum(j.bid()) for j in active}`, i.e. the marginal-value CURVE was
collapsed to one scalar and used as a tie-break for greedy fill. There was no supply ask and
no clearing condition. `pins/mechanism.py` has a real uniform-price auction, but only the
older sims call it.

This module builds the plan's market properly, as an arm that can be compared paired-by-seed
against `negotiated`, `referee` and the floor:

  demand, per job j and per extra GPU k (§6.1):
      b_(j,k) = alpha*dp_hat_(j,k) + beta*R_SLA_j + gamma*W_wait_j - delta*C_resize_(j,k)
  supply, per unit sold (§6.2):
      a_k     = eta1*Scarcity + eta2*Fragmentation + eta3*Reserve + eta4*ArrivalPressure,
                rising as more of the pool is sold
  clearing (§6.3):
      Q_t = max { q : b_(q) >= a_(q) }        b sorted descending, a ascending

`dp_hat` is the term that made this buildable only after Exp 68: it is the job's ACTUAL
marginal useful progress from the counterfactual progress model (`two_sided_sim._rate`), not
a phase-profile lookup. Every term is normalised to [0,1] and the curve is enforced
non-increasing, per the plan.

HEADROOM LIVES IN THE PRICE, NOT IN AN IDLE BLOCK. The first build of this arm returned
`reserve = free - sold`, treating every unsold unit as held-idle headroom. That collapsed the
cluster (util 30%, 8.6/16 jobs finished): in `two_sided_sim` the reserve blocks best-effort
jobs from reaching their BASE, not just their margin, so an unsold margin unit was starving
queued work. The correct mapping is that the supply side expresses scarcity through the ask
(eta3 raises the price when prod arrivals are pending, so fewer margin units sell) and the
units it declines to sell stay AVAILABLE for other jobs' bases — which is what actually helps
an incoming prod job. `hold_unsold=True` restores the idle-block semantics for comparison.
"""
from __future__ import annotations

from pins.negotiation_protocol import NegotiationOutcome
from pins.two_sided_sim import STARVE_TICKS, _rate, _useful_units

# alpha, beta, gamma, delta — demand weights (plan §6.1)
BID_W = (1.0, 0.5, 0.3, 0.5)
# eta1..eta4 — supply weights (plan §6.2)
ASK_W = (0.5, 0.2, 0.3, 0.2)
SLA_RISK = {"behind": 1.0, "ontrack": 0.4, "ahead": 0.0}


def bid_curve(facts: dict, ctx: dict, env: dict, w=BID_W) -> list[float]:
    """The job's non-increasing marginal-value curve over EXTRA (margin) GPUs."""
    a, b, g, d = w
    base, usable = facts.get("base", 0), facts.get("usable", 0)
    if base <= 0 or usable <= 0:
        return []
    law, kappa = env.get("law", "amdahl"), env.get("kappa", 2.0)
    alpha, norm = env.get("alpha", 0.0), env.get("alpha_norm", "c0")
    ceil_use = base + usable

    def useful(g_):
        return _useful_units(g_, base, _rate(g_, base, ceil_use, alpha, norm, law, kappa))

    gains = [useful(base + k) - useful(base + k - 1) for k in range(1, usable + 1)]
    top = max(gains) if gains else 0.0
    if top <= 0:
        return []
    r_sla = SLA_RISK.get(ctx.get("deadline"), 0.4)
    w_wait = min(1.0, facts.get("waited", 0) / STARVE_TICKS)
    # each extra GPU is one more rank to reconfigure on the next resize (plan §15's c1 term)
    c_res = env.get("resize_c1", 0.0) + env.get("realloc_cost", 0.0)
    curve, prev = [], float("inf")
    for k, gain in enumerate(gains, start=1):
        v = a * (gain / top) + b * r_sla + g * w_wait - d * min(1.0, c_res * k)
        v = min(max(v, 0.0), prev)            # plan §6.1: bids must be non-increasing
        curve.append(v)
        prev = v
    return curve


def ask_curve(free: int, env: dict, w=ASK_W) -> list[float]:
    """The supply side's price for the q-th GPU sold — rising, so the pool's last units are
    dearest and the RESERVE emerges as what nobody outbid (plan §6.2)."""
    e1, e2, e3, e4 = w
    total = max(env.get("total_gpus", free), 1)
    scarcity = 1.0 - free / total
    frag = 1.0 / (1.0 + free)
    reserve_p = min(1.0, env.get("incoming_prod", 0) / 4.0)
    arrival_p = min(1.0, env.get("n_waiting", 0) / max(env.get("n_active", 1), 1))
    base = e1 * scarcity + e2 * frag + e3 * reserve_p + e4 * arrival_p
    return [base * (q / max(free, 1)) for q in range(1, free + 1)]


def clear_market(demand, free: int, env: dict, bid_w=BID_W, ask_w=ASK_W):
    """Plan §6.3. Returns (margins, reserve, clearing_price, units_sold)."""
    units: list[tuple[float, str, int]] = []
    for j in demand:
        for k, v in enumerate(bid_curve(getattr(j, "facts", None) or {}, j.ctx, env, bid_w),
                              start=1):
            units.append((v, j.jid, k))
    units.sort(key=lambda u: (-u[0], u[1], u[2]))       # descending value, deterministic ties
    asks = ask_curve(free, env, ask_w)
    margins = {j.jid: 0 for j in demand}
    price, sold = 0.0, 0
    for q, (v, jid, _k) in enumerate(units[:free]):
        if v < asks[q]:                                 # single crossing: b descending, a rising
            break
        margins[jid] += 1
        price, sold = asks[q], sold + 1
    return margins, max(0, free - sold), price, sold


def make_policy_market(trace=None, seen=None, bid_w=BID_W, ask_w=ASK_W,
                       hold_unsold: bool = False):
    """A `two_sided_sim` policy: the market decides margins AND (implicitly) the reserve."""
    def policy(demand, supply_ctx, free, waiting=None, env=None, **_):
        env = dict(env or {})
        env.setdefault("n_waiting", len(waiting or []))
        env.setdefault("n_active", max(len(demand), 1))
        margins, unsold, price, sold = clear_market(demand, free, env, bid_w, ask_w)
        reserve = unsold if hold_unsold else 0      # see module docstring
        out = NegotiationOutcome(
            margins=margins, reserve=reserve, rounds=0, agreed=True,
            transcript=[{"round": 0, "actor": "market", "price": round(price, 4),
                         "units_sold": sold, "free": free,
                         "unsold": unsold, "reserve": reserve,
                         "why": f"cleared {sold}/{free} GPUs at ask {price:.3f}; "
                                f"{unsold} unsold, {reserve} held idle"}])
        if trace is not None and seen is not None:
            sig = f"market|free={free}|sold={sold}|m={sorted(margins.items())}"
            if sig not in seen:
                seen.add(sig)
                trace.append({"policy": "market", "free_gpus": free, "price": price,
                              "units_sold": sold, "reserve": reserve, "margins": margins})
        return margins, reserve, out
    return policy
