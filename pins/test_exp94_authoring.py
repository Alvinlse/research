"""Exp 94 harness checks — the placebo only works if the two modes are budget-identical.

Run: .venv/bin/python -m pins.test_exp94_authoring   (no network, no Ollama)

What is actually load-bearing here:
  1. `narrated` and `attributed` must author on the SAME jobs and make the SAME number of calls,
     or the contrast measures spend instead of causal text.
  2. Their cache tags must differ. correction._ask keys on (tag, user, model) and NOT on the
     system prompt, so a shared tag would silently serve narrated notes to the attributed arm.
  3. `corrected` with authored=None must still be the Exp-92 arm: no note calls at all.
"""
from pins.market import (author_notes, clear_market, make_policy_corrected, make_policy_market,
                         _speaks)
from pins.negotiation_protocol import DemandJob


def _jobs():
    return [DemandJob(jid="j0", ctx={"tier": "prod", "deadline": "behind"}, forecast_cap=0,
                      facts={"base": 1, "usable": 2, "held": 2, "waited": 0}),
            DemandJob(jid="j1", ctx={"tier": "besteffort", "deadline": "ontrack"}, forecast_cap=0,
                      facts={"base": 1, "usable": 2, "held": 1, "waited": 3})]


def _stub(calls):
    def _ask(system, user, model, host, cache, tag, **kw):
        calls.append({"tag": tag, "user": user, "system": system})
        return {"note": "stub note"}
    return _ask


def _run_mode(mode, prev):
    import pins.correction_signed as cs
    calls, real = [], cs._ask
    cs._ask = _stub(calls)
    try:
        notes, sup = author_notes(_jobs(), {"j0": 1, "j1": 0}, 4, prev, mode, cache={})
    finally:
        cs._ask = real
    return notes, sup, calls


def main() -> None:
    prev = {"jobs": {"j0": {"held_margin": 0, "deadline": "behind", "requested": 0}},
            "free": 6}          # j0's margin moved, j1 is new -> both speak
    n_a, s_a, c_a = _run_mode("narrated", prev)
    n_b, s_b, c_b = _run_mode("attributed", prev)

    assert sorted(n_a) == sorted(n_b) == ["j0", "j1"], (n_a, n_b)
    assert len(c_a) == len(c_b) == 3, (len(c_a), len(c_b))      # 2 demand + 1 supply
    assert s_a and s_b, "supply always speaks on a fired tick"
    assert [c["user"] for c in c_a] == [c["user"] for c in c_b], "payloads must be identical"
    assert {c["tag"] for c in c_a}.isdisjoint({c["tag"] for c in c_b}), "cache tags would collide"
    assert all(c_a[i]["system"] != c_b[i]["system"] for i in range(3)), "prompts must differ"
    print(f"budget match: {len(c_a)} calls both modes, tags "
          f"{sorted({c['tag'] for c in c_a})} vs {sorted({c['tag'] for c in c_b})}")

    # a job whose margin, deadline and identity are all unchanged stays silent
    same = {"held_margin": 1, "deadline": "behind", "requested": 1}
    assert not _speaks(same, dict(same))
    assert _speaks(same, dict(same, held_margin=0))
    assert _speaks(same, dict(same, deadline="ontrack"))
    assert _speaks(same, None)
    print("authoring condition: silent when nothing moved, speaks on margin/deadline/arrival")

    # Exp 92 arm untouched: no authoring, and the allocation is still the market's
    jobs, env = _jobs(), {"total_gpus": 8, "n_waiting": 0, "n_active": 2}
    ref, _, _, _ = clear_market(jobs, 4, env)
    calls = []
    import pins.correction_signed as cs
    real, cs._ask = cs._ask, _stub(calls)
    try:
        m, r, out = make_policy_corrected(use_llm=False)(jobs, {}, 4, env=env)
    finally:
        cs._ask = real
    assert m == ref and r == 0 and not calls, (m, ref, calls)
    print(f"corrected(authored=None): 0 calls, margins {m} == market {ref}")
    print("OK")


if __name__ == "__main__":
    main()
