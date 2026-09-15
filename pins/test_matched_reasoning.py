"""Offline regression tests: no model server or simulator process required."""
from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pins import matched_reasoning as protocol
from pins.analyze_matched_reasoning import analyze, contrast
from pins.run_matched_reasoning import (CONFIG, build_tasks, digest, freeze, load_states,
                                       main as run_main, offline_cell, read_rows,
                                       validate_config, verify_worlds)


def answer(ordering="fcfs", why="observable evidence"):
    return {"ordering": ordering, "default_sizing": "adaptive", "why": why}


class ProtocolTests(unittest.TestCase):
    def run_protocol(self, structure, replies=None):
        calls = []
        def ask(system, user, stage, seed):
            calls.append((system, user, stage, seed))
            return replies[stage] if replies is not None else answer(why=f"unique-stage-{stage}")
        result = protocol.decide("OBSERVABLE STATE", structure, ["fcfs", "fairness"],
                                 ["adaptive"], ("fairness", "adaptive"), ask=ask, seed=17)
        return result, calls

    def test_call_budget_and_final_seed(self):
        for structure in protocol.STRUCTURES:
            result, calls = self.run_protocol(structure)
            self.assertEqual(result["calls"], 1 if structure == "single" else 3)
            self.assertEqual(calls[-1][2:], (2, 19))
            self.assertEqual(calls[-1][0], protocol.FINAL)
            self.assertTrue(all("OBSERVABLE STATE" in call[1] for call in calls))

    def test_independent_openings_and_self_review_dependency(self):
        for structure in ("symmetric", "multi"):
            _, calls = self.run_protocol(structure)
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertNotIn("unique-stage-0", calls[1][1])
            self.assertIn("unique-stage-0", calls[2][1])
            self.assertIn("unique-stage-1", calls[2][1])
        _, calls = self.run_protocol("self_review")
        self.assertIn("unique-stage-0", calls[1][1])

    def test_invalid_review_does_not_shorten_budget_or_poison_referee(self):
        result, calls = self.run_protocol("multi", {0: {"why": "BAD_SENTINEL"}, 1: None, 2: answer()})
        self.assertEqual(result["calls"], 3)
        self.assertEqual(result["invalid_reviews"], 2)
        self.assertTrue(result["valid"])
        self.assertNotIn("BAD_SENTINEL", calls[-1][1])

    def test_final_rejects_off_menu_and_overrides(self):
        bad_answers = [None, [], answer("invented"), {**answer(), "job_sizing": {}},
                       {**answer(), "default_sizing": []}, {**answer(), "why": ""}]
        for bad in bad_answers:
            result, _ = self.run_protocol("single", {2: bad})
            self.assertFalse(result["valid"])
            self.assertEqual(result["answer"]["ordering"], "fairness")

    def test_scoped_registration_restored_even_after_failure(self):
        from pins import elastisim_bench as bench
        original = bench.ARMS["policy_select"]
        def fake_run(*args, **kwargs):
            self.assertIsNot(bench.ARMS["policy_select"], original)
            raise RuntimeError("simulator failure")
        with patch.object(bench, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError):
                protocol.run_closed_loop(Path("unused"), "multi")
        self.assertIs(bench.ARMS["policy_select"], original)

    def test_real_selector_clock_record_and_fallback(self):
        from pins import elastisim_bench as bench
        pending = [SimpleNamespace(identifier=1, attributes={"jid": "j1", "req_nodes": 2})]
        for structure in protocol.STRUCTURES:
            ctx = {"now": 0, "sel_every": 1800, "family": "nm", "pool_n": 80,
                   "sel_ordering": "fcfs", "sel_sizing": "adaptive", "calls": 0,
                   "fallback_ordering": "fairness", "fallback_sizing": "adaptive",
                   "model": "mock", "host": "mock", "temperature": .1,
                   "num_predict": 400, "llm_seed": 17}
            with patch.object(bench, "_packet_policy", return_value="state"), \
                    patch.object(bench, "_policy_execute") as execute, \
                    patch("pins.correction._ask", return_value=None) as ask:
                protocol.arm(pending, [], ctx, structure=structure)
                budget = 1 if structure == "single" else 3
                self.assertEqual(ctx["calls"], budget)
                self.assertEqual(ctx["sel_invalid"], 1)
                self.assertEqual(ctx["sel_ordering"], "fairness")
                self.assertEqual(ctx["job_sizing"], {"j1": "adaptive"})
                ctx["now"] = 300
                protocol.arm(pending, [], ctx, structure=structure)
                self.assertEqual(ask.call_count, budget)
                self.assertEqual(execute.call_count, 2)
                ctx["now"] = 1800
                protocol.arm(pending, [], ctx, structure=structure)
                self.assertEqual(ask.call_count, 2*budget)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.cfg = json.loads(CONFIG.read_text())
        self.manifest = {"windows": [{"window": "w1", "split": "train"}]}

    def write_dataset(self):
        from pins.elastisim_bench import POLICY_FAMILY, SELECTOR_MENU
        rows = []
        for family in self.cfg["families"]:
            rows.append({"window": "w1", "family": family, "t": 1800,
                         "baseline": "+".join(self.cfg["fallbacks"][family]),
                         "decision_horizon_s": 1800,
                         "messages": [{"role": "user", "content": "OBSERVABLE"},
                                      {"role": "assistant", "content": "SECRET_TARGET"}],
                         "rewards": {o+"+"+s: -10 for o in POLICY_FAMILY[family]
                                     for s in SELECTOR_MENU["sizing"]}})
        self.write_rows(rows)
        return rows

    def write_rows(self, rows):
        (self.path / "train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))

    def test_oracle_complete_menu_horizon_and_split_validation(self):
        original = self.write_dataset()
        self.assertEqual(len(load_states(self.path, "train", self.manifest, self.cfg)), 2)
        for mutation in ("horizon", "missing_action", "baseline", "duplicate", "nonfinite"):
            rows = copy.deepcopy(original)
            if mutation == "horizon": rows[0]["decision_horizon_s"] = 0
            if mutation == "missing_action": rows[0]["rewards"].pop(next(iter(rows[0]["rewards"])))
            if mutation == "baseline": rows[0]["baseline"] = "fcfs+adaptive"
            if mutation == "duplicate": rows.append(rows[0])
            if mutation == "nonfinite": rows[0]["rewards"][next(iter(rows[0]["rewards"]))] = float("nan")
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                load_states(self.path, "train", self.manifest, self.cfg)
        self.write_rows(original)
        (self.path / "test.jsonl").write_text(json.dumps(original[0])+"\n")
        with self.assertRaises(ValueError):
            load_states(self.path, "train", self.manifest, self.cfg)

    def test_offline_labels_never_enter_model_prompt(self):
        self.write_dataset()
        state = load_states(self.path, "train", self.manifest, self.cfg)[0]
        task = {"family": "nm", "structure": "multi", "seed": 17}
        with patch("pins.correction._ask", return_value=answer()) as ask:
            result, _ = offline_cell(state, task, self.cfg)
        self.assertEqual(result["regret_pp"], 0)
        self.assertEqual(result["llm_calls"], 3)
        for call in ask.call_args_list:
            self.assertIn("OBSERVABLE", call.args[1])
            self.assertNotIn("SECRET_TARGET", call.args[1])
            self.assertNotIn("rewards", call.args[1])

    def test_resumption_rejects_changed_identity(self):
        first = {"model": "digest-a", "tasks": [1]}
        freeze(self.path, first)
        freeze(self.path, first)
        with self.assertRaises(ValueError):
            freeze(self.path, {"model": "digest-b", "tasks": [1]})

    def test_balanced_deterministic_task_plan(self):
        validate_config(self.cfg)
        tasks = build_tasks(self.cfg, self.manifest, "train", "closed-loop")
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({t["key"] for t in tasks}), 24)
        self.assertEqual(tasks, build_tasks(self.cfg, self.manifest, "train", "closed-loop"))

    def test_inherited_source_pin_and_world_verifier(self):
        from pins.freeze_core_2x2 import IMPLEMENTATION_FILES, ROOT, combined_sha
        self.assertEqual(combined_sha([ROOT/p for p in IMPLEMENTATION_FILES]),
                         self.cfg["inherited_implementation_sha256"])
        with patch("pins.verify_core_2x2.verify", return_value={"checked": True}) as check:
            self.assertEqual(verify_worlds(ROOT / "pins/core_2x2_manifest.json", self.path, self.cfg),
                             {"checked": True})
            self.assertEqual(check.call_count, 1)
        bad = {**self.cfg, "inherited_implementation_sha256": "changed"}
        with self.assertRaises(ValueError):
            verify_worlds(ROOT / "pins/core_2x2_manifest.json", self.path, bad)

    def test_bounded_driver_resumes_without_repeating_cells(self):
        cfg = {**self.cfg, "seeds": [17]}
        config_path = self.path / "config.json"
        config_path.write_text(json.dumps(cfg))
        manifest_path = self.path / "worlds.json"
        manifest_path.write_text(json.dumps({"windows": [
            {"window": "w1", "split": "train", "n_jobs": 3}]}))
        out = self.path / "results"
        worlds = self.path / "worlds"
        (worlds / "w1/out").mkdir(parents=True)
        calls = []
        def fake_run(world, structure, **kwargs):
            calls.append((structure, kwargs["family"]))
            (world / "out" / (kwargs["tag"] + "_policy_log.json")).write_text(
                json.dumps([{"invalid_reviews": 0}]))
            return {"n": 3, "llm_calls": 1 if structure == "single" else 3,
                    "deadline_viol_pct": 10, "sel_invalid": 0}
        argv = ["runner", "--mode", "closed-loop", "--config", str(config_path),
                "--manifest", str(manifest_path), "--worlds", str(worlds), "--out", str(out)]
        with patch("pins.run_matched_reasoning.verify_worlds"), \
                patch("pins.run_matched_reasoning.model_identity", return_value={"digest": "fake"}), \
                patch("pins.run_matched_reasoning.run_closed_loop", side_effect=fake_run), \
                contextlib.redirect_stdout(io.StringIO()):
            with patch("sys.argv", argv): run_main()
            self.assertEqual(len(read_rows(out / "rows.jsonl")), 1)
            with patch("sys.argv", argv + ["--max-runs", "0"]): run_main()
            self.assertEqual(len(read_rows(out / "rows.jsonl")), 8)
            with patch("sys.argv", argv): run_main()
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(set(calls)), 8)
        self.assertEqual(analyze(out)["contrasts"]["multi_minus_self_review"]["mean_delta_pp"], 0)


class AnalysisTests(unittest.TestCase):
    def test_zero_difference_is_not_a_win(self):
        report = contrast([0, 0, 0])
        self.assertEqual(report["p"], 1)
        self.assertEqual(report["ci95"], [0, 0])
        self.assertEqual(report["wins"], 0)

    def test_window_aggregation_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            cfg = json.loads(CONFIG.read_text())
            manifest = {"windows": [{"window": f"w{i}", "split": "train"} for i in range(3)]}
            tasks = build_tasks(cfg, manifest, "train", "closed-loop")
            freeze(path, {"config": cfg, "tasks": tasks, "mode": "closed-loop", "split": "train"})
            fingerprint = digest(path / "run_manifest.json")
            rows = [{**t, "deadline_viol_pct": 9 if t["structure"] == "multi" else 10,
                     "llm_calls": 3, "run_manifest_sha256": fingerprint} for t in tasks]
            sink = path / "rows.jsonl"
            sink.write_text("".join(json.dumps(r)+"\n" for r in rows))
            report = analyze(path)
            primary = report["contrasts"]["multi_minus_self_review"]
            self.assertEqual(primary["n_windows"], 3)  # not 72 cells
            self.assertEqual(primary["mean_delta_pp"], -1)
            sink.write_text("".join(json.dumps(r)+"\n" for r in rows[:-1]))
            with self.assertRaises(ValueError): analyze(path)
            sink.write_text("".join(json.dumps(r)+"\n" for r in rows + rows[:1]))
            with self.assertRaises(ValueError): analyze(path)


if __name__ == "__main__":
    unittest.main()
