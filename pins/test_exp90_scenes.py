"""Offline tests for the Exp 90 scene sampler — run: python -m pins.test_exp90_scenes"""
from pins.no_exception_scenes import sample_scenes, CAP_CLIP
from pins.h2_eval import build_anchor
from pins.hardcases import total


def test_deterministic_per_seed():
    a, _ = sample_scenes(20, seed=0)
    b, _ = sample_scenes(20, seed=0)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.free_gpus for c in a] == [c.free_gpus for c in b]


def test_no_text_and_inelastic():
    scenes, _ = sample_scenes(30, seed=1)
    for c in scenes:
        dem = [x for x in c.stmts if x["side"] == "demand"]
        assert 3 <= len(dem) <= 5
        assert all(x["requested_margin_gpus"] == 0 for x in dem)
        assert all(1 <= x["base_gpus"] <= CAP_CLIP for x in dem)


def test_correct_action_is_retain_market():
    # every base seats (free >= sum bases) and idle headroom is left over
    scenes, meta = sample_scenes(30, seed=2)
    for c, m in zip(scenes, meta):
        floors, alloc, ranking, _ = build_anchor(c)
        base_sum = sum(x["base_gpus"] for x in c.stmts if x["side"] == "demand")
        assert total(alloc) == base_sum          # all bases seated, no margin sold
        assert c.free_gpus > base_sum             # genuine idle headroom to meddle with


def test_free_exceeds_base_at_minimum():
    # smallest possible bases (all 1) with the lowest slack must still leave idle headroom
    from pins.no_exception_scenes import _scene
    case, m = _scene(0, [1, 1, 1], 0.15)
    assert case.free_gpus > m["base_sum"]


def test_non_vacuous_menu():
    from pins.packet import candidate_actions
    scenes, meta = sample_scenes(30, seed=3)
    for c, m in zip(scenes, meta):
        floors, alloc, ranking, _ = build_anchor(c)
        menu = candidate_actions(alloc, c.free_gpus, ranking, floors, 6)
        assert len(menu) > 1                      # more than just retain_market (id 0)
        assert m["menu_size"] == len(menu)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} scene tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
