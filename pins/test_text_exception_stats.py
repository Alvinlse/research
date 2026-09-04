"""Self-check for the outcome-based exception scorer.

The bug this pins down: the old scorer divided by the number of times the scheduler chose to ask
the LLM, so an arm that escalated once and answered it correctly reported 100% recall while
ignoring every other planted exception. Run: .venv/bin/python -m pins.test_text_exception_stats
"""
import json
import tempfile
from pathlib import Path

from pins.elastisim_bench import text_exception_stats


def _world(tmp: Path):
    """Two protect cases and two shrink cases; the run shrinks one job of each kind."""
    jobs = [{"attributes": {"jid": f"J{i}"}} for i in range(4)]
    labels = {"J0": {"expected_action": "protect"}, "J1": {"expected_action": "protect"},
              "J2": {"expected_action": "shrink"}, "J3": {"expected_action": "shrink"}}
    (tmp / "in").mkdir(parents=True)
    (tmp / "out").mkdir()
    (tmp / "in/jobs.json").write_text(json.dumps({"jobs": jobs}))
    (tmp / "in/text_exception_labels.json").write_text(json.dumps({"labels": labels}))
    # J1 is shrunk (a protect violation); J2 is shrunk (correct); J0 and J3 hold their size.
    (tmp / "out/a_size_history.json").write_text(json.dumps({
        "0": [[0, 4], [10, 4]], "1": [[0, 4], [10, 1]],
        "2": [[0, 4], [10, 1]], "3": [[0, 4], [10, 4]]}))
    return tmp


def main():
    with tempfile.TemporaryDirectory() as d:
        w = _world(Path(d))
        tr = w / "out/a_transcript.jsonl"

        # The arm was asked about exactly one case and got it right -- the old metric's 100%.
        tr.write_text(json.dumps({"trace_job_id": "J0", "model_action": "protect"}) + "\n")
        s = text_exception_stats(tr, w / "in/text_exception_labels.json", w)
        assert s["exceptions_escalated"] == 1, s
        assert s["asked_accuracy_pct"] == 100.0, s          # right on the one it saw ...
        assert s["exception_compliance_pct"] == 50.0, s     # ... but the run scores 50%
        assert s["protect_violations"] == 1, s              # J1 was shrunk anyway
        assert s["shrink_taken"] == 1, s                    # only J2 of the two shrink cases
        assert s["exceptions_escalated_pct"] == 25.0, s

        # An arm that never calls the model is scored on the same footing, not skipped.
        tr.write_text("")
        z = text_exception_stats(tr, w / "in/text_exception_labels.json", w)
        assert z["exceptions_escalated"] == 0, z
        assert z["asked_accuracy_pct"] is None, z
        assert z["exception_compliance_pct"] == 50.0, z     # outcome is identical, as it must be
        assert z["planted_protect"] == 2 and z["planted_shrink"] == 2, z

        # A job pinned at its minimum all run cannot be shrunk, so its note is unscoreable: it
        # must leave the actionable denominator rather than count as a miss.
        (w / "out/a_size_history.json").write_text(json.dumps({
            "0": [[0, 4], [10, 4]], "1": [[0, 4], [10, 1]],
            "2": [[0, 4], [10, 1]], "3": [[0, 1], [10, 1]]}))   # J3 never above its min
        m = text_exception_stats(tr, w / "in/text_exception_labels.json", w)
        assert m["planted_shrink"] == 2, m                  # still two shrink notes planted ...
        assert m["shrink_actionable"] == 1, m               # ... but only J2 could ever comply
        assert m["shrink_actionable_pct"] == 100.0, m       # and it did
        assert m["shrink_compliance_pct"] == 50.0, m        # the raw figure still shows 1 of 2
    print("text_exception_stats: OK (denominator is planted exceptions, not model calls)")


if __name__ == "__main__":
    main()
