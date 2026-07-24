"""Offline tests for the Exp 91 hard-specificity sampler bait.
Run: .venv/bin/python -m pins.test_exp91_scenes"""
from pins.no_exception_scenes import sample_scenes, CAP_CLIP


def test_defaults_reproduce_exp90():
    # default params == Exp 90; explicit Exp 90 values must give identical scenes
    a, _ = sample_scenes(25, seed=0)
    b, _ = sample_scenes(25, seed=0, spread_min=0, slack_lo=0.15, slack_hi=0.6)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.free_gpus for c in a] == [c.free_gpus for c in b]


def test_imbalance_gate_enforced():
    scenes, meta = sample_scenes(30, seed=1, spread_min=4)
    for c, m in zip(scenes, meta):
        bases = [x["base_gpus"] for x in c.stmts if x["side"] == "demand"]
        assert max(bases) - min(bases) >= 4          # lopsided by construction
        assert m["spread"] == max(bases) - min(bases)
        assert m["spread"] <= CAP_CLIP - 1


def test_tight_slack_reduces_headroom():
    # tight slack must leave strictly less headroom than Exp 90's generous default, on average
    tight, mt = sample_scenes(40, seed=2, spread_min=4, slack_lo=0.05, slack_hi=0.30)
    loose, ml = sample_scenes(40, seed=2, spread_min=4, slack_lo=0.15, slack_hi=0.60)
    hr_t = sum(m["free"] - m["base_sum"] for m in mt) / len(mt)
    hr_l = sum(m["free"] - m["base_sum"] for m in ml) / len(ml)
    assert hr_t < hr_l
    for m in mt:                                     # invariant: headroom never zero
        assert m["free"] > m["base_sum"]


def test_invariant_still_holds_under_bait():
    scenes, _ = sample_scenes(30, seed=3, spread_min=4, slack_lo=0.05, slack_hi=0.30)
    for c in scenes:
        dem = [x for x in c.stmts if x["side"] == "demand"]
        assert all(x["requested_margin_gpus"] == 0 for x in dem)   # inelastic
        assert all(x["justification"] == "" for x in dem)          # no text
        assert all(x["side"] != "supply" or x["requested_reserve_gpus"] == 0 for x in c.stmts)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} exp91 scene tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
