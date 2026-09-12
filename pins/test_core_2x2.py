"""Fast tests for the frozen factorial driver and pre-registered statistics."""
from __future__ import annotations

from pins.analyse_core_2x2 import holm, sign_flip_p
from pins.run_core_2x2 import core_cells, fixed_cells


def manifest() -> dict:
    return {
        "selector": {
            "families": {"nm": ["a", "b", "c", "d"], "mkt": ["e", "f", "g", "h"]},
            "sizing_actions": ["req", "adapt", "greedy", "rcon"],
        },
        "inference": {"paired_seeds": [17, 29, 43]},
    }


def test_cell_counts_and_paired_seeds() -> None:
    doc = manifest()
    fixed, core = fixed_cells(doc), core_cells(doc)
    assert len(fixed) == 32
    assert len(core) == 12
    for structure in ("single", "multi"):
        for family in ("nm", "mkt"):
            assert {x["seed"] for x in core if x["structure"] == structure and
                    x["family"] == family} == {17, 29, 43}


def test_exact_sign_flip_is_two_sided() -> None:
    assert sign_flip_p([1.0, 1.0]) == 0.5
    assert sign_flip_p([0.0, 0.0]) == 1.0


def test_holm_is_monotone_and_bounded() -> None:
    adjusted = holm({"a": 0.01, "b": 0.03, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.5}


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} core-2x2 tests")
    for test in tests:
        test()
        print(f"  {test.__name__}: OK")
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
