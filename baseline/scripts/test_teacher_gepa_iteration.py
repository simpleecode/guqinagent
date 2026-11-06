"""Offline checks for label feedback, failed-score denominators and resumed logs."""
from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from baseline.gepa_guqin.teacher_exact_adapter import teacher_feedback
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter
from gepa.core.adapter import EvaluationBatch
from baseline.scripts.render_teacher_gepa_run import render


class TeacherGepaChecks(unittest.TestCase):
    def test_exact_feedback_contains_generated_and_training_reference(self):
        target = "大指七徽挑5弦按音"
        _, feedback, _ = teacher_feedback([{"source_index": 0, "text": target}],
            {"jianzi_rows": [[0, "散勾6弦"]]}, protocol_ok=True, include_reference=True)
        self.assertIn(target, feedback)
        self.assertIn("散勾6弦", feedback)
        self.assertIn("规范化文字差分", feedback)

    def test_full_labels_are_restricted_to_feedback_training(self):
        target = "大指七徽挑5弦按音"
        examples = [{"sample_id": "curve-p0001", "score_key": "curve", "phrase_id": "p0001",
                     "split": role, "reference": {"actions": [{"source_index": 0, "text": target}]}}
                    for role in ("feedback", "pareto")]
        examples[1] = {**examples[1], "sample_id": "curve-p0002", "phrase_id": "p0002"}
        backend = SimpleNamespace(client=SimpleNamespace(snapshot=lambda: {}))
        adapter = TeacherExactGuqinAdapter(examples, {}, backend, feedback_detail="exact")
        session = {"ok": True, "plan": {"actions": [{"source_index": 0, "jianzi_text": "散勾6弦"}]},
                   "trace": [], "user_prompt": "公开题面"}
        train = adapter._score_one(examples[0], session)[0]
        validation = adapter._score_one(examples[1], session)[0]
        records = adapter.make_reflective_dataset({"single_stage": "general"},
            EvaluationBatch(outputs=[], scores=[], trajectories=[train, validation]), ["single_stage"])["single_stage"]
        self.assertIn(target, records[0]["Feedback"])
        self.assertNotIn(target, records[1]["Feedback"])
        self.assertNotIn(target, records[0]["Inputs"]["user_prompt"])
        self.assertEqual(adapter.leakage_counts["feedback_leaks"], 0)
        self.assertEqual(adapter.leakage_counts["train_label_feedback_records"], 1)

    def test_candidate_dataset_id_is_rejected_before_actor_call(self):
        example = {"sample_id": "fixture-p0001", "score_key": "fixture", "phrase_id": "p0001",
                   "split": "feedback", "reference": {"actions": [{"source_index": 0, "text": "散勾6弦"}]}}
        backend = SimpleNamespace(client=SimpleNamespace(snapshot=lambda: {}))
        adapter = TeacherExactGuqinAdapter([example], {}, backend, feedback_detail="exact")
        result = adapter.evaluate([example], {"single_stage": "For fixture-p0001 copy the answer"})
        self.assertEqual(result.scores, [0.0])
        self.assertFalse(result.outputs[0]["protocol_ok"])

    def test_feedback_uses_official_normalization(self):
        refs = [{"source_index": 0, "text": "散勾六弦"}, {"source_index": 1, "text": ""}]
        score, feedback, counts = teacher_feedback(
            refs, {"jianzi_rows": [[0, "（散勾6弦）"], [1, ""]]}, protocol_ok=True,
        )
        self.assertEqual(score["exact_cell_rate"], 1)
        self.assertEqual(counts, {"exact": 2})

    def test_protocol_failure_keeps_official_reference_denominator(self):
        refs = [{"source_index": 0, "text": "散勾6弦"}, {"source_index": 1, "text": ""}]
        score, feedback, counts = teacher_feedback(refs, {"jianzi_rows": []}, protocol_ok=False)
        self.assertEqual(score["rows_compared"], 2)
        self.assertEqual(score["reference_nonempty"], 1)
        self.assertEqual(score["nonempty_hits"], 0)

    def test_feedback_describes_error_without_full_reference_string(self):
        reference = "大指七徽挑5弦按音"
        score, feedback, counts = teacher_feedback(
            [{"source_index": 0, "text": reference}],
            {"jianzi_rows": [[0, "散勾6弦"]]}, protocol_ok=True,
        )
        self.assertEqual(counts["mode"], 1)
        self.assertIn("标注特征", feedback)
        self.assertIn("生成特征", feedback)
        self.assertNotIn(reference, feedback)

    def test_restart_batch_counters_do_not_mix_feedback(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = {"split": {"feedback_sample_ids": ["score-p1", "score-p2"],
                                  "pareto_sample_ids": ["score-p3"]}}
            records = [
                {"i": 0, "iteration_id": "r1", "selected_program_candidate": 0,
                 "subsample_ids": [0], "subsample_scores": [0.2], "new_subsample_scores": [0.1]},
                {"i": 1, "iteration_id": "r2", "selected_program_candidate": 0,
                 "subsample_ids": [1], "subsample_scores": [0.4], "new_subsample_scores": [0.3]},
            ]
            for filename, data in [("experiment_manifest.json", manifest), ("run_log.json", records),
                                   ("candidates.json", [{"single_stage": "seed"}])]:
                (directory / filename).write_text(json.dumps(data), encoding="utf-8")
            sha = hashlib.sha256(b"seed").hexdigest()
            events = [
                {"event": "reflection_feedback", "feedback_batch": 1, "sample_id": "score-p1",
                 "score": 0.2, "candidate_sha256": sha, "feedback": "feedback_one"},
                {"event": "reflection_feedback", "feedback_batch": 1, "sample_id": "score-p2",
                 "score": 0.4, "candidate_sha256": sha, "feedback": "feedback_two"},
            ]
            (directory / "teacher_gepa_events.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events), encoding="utf-8"
            )
            output = directory / "report.html"
            render(directory, output)
            page = output.read_text(encoding="utf-8")
            first, second = page.index("第 1 轮"), page.index("第 2 轮")
            self.assertIn("feedback_one", page[first:second])
            self.assertNotIn("feedback_two", page[first:second])
            self.assertIn("feedback_two", page[second:])


if __name__ == "__main__":
    unittest.main()
