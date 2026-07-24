"""Offline tests for the Exp 90 driver — run: python -m pins.test_exp90_analysis"""
from pins.exp90_specificity import single_no_packet_fired, fired_discordant
from pins.no_exception_scenes import sample_scenes


def test_single_no_packet_is_silent_on_no_text():
    # correction.py is text-gated: empty notes -> no LLM call -> never fires
    scenes, _ = sample_scenes(5, seed=0)
    for c in scenes:
        assert single_no_packet_fired(c, "qwen2.5:14b", use_llm=True) is False


def test_fired_discordant_counts():
    res = {
        "s1": {"arms": {"A": {"fired": True},  "B": {"fired": False}}},  # A only
        "s2": {"arms": {"A": {"fired": False}, "B": {"fired": True}}},   # B only
        "s3": {"arms": {"A": {"fired": True},  "B": {"fired": True}}},   # both
        "s4": {"arms": {"A": {"fired": False}, "B": {"fired": False}}},  # neither
    }
    assert fired_discordant(res, "A", "B") == (1, 1)   # (A-only, B-only)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} analysis tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
