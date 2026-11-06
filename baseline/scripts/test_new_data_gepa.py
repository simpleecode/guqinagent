"""Offline checks for the new family split and native GEPA callback lifecycle."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
from baseline.scripts.prompt_data_sources import TRAIN, training_identity
from baseline.scripts.teacher_gepa_audit import TeacherGepaAudit
from baseline.gepa_guqin.teacher_exact_adapter import teacher_feedback
from baseline.scripts.run_teacher_gepa_iteration import aggregate_official
from train.scripts.score_agent_predictions import score as official_score
from baseline.scripts.prompt_series import candidate_versions, export_series


class NewDataChecks(unittest.TestCase):
    def test_series_names_follow_proposals_not_accepted_indices(self):
        rounds = [{"i": 0, "selected_program_candidate": 0},
                  {"i": 1, "selected_program_candidate": 0, "new_program_indices": [1]}]
        self.assertEqual(candidate_versions("N", rounds), {0: "N0", 1: "N2"})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_series(root, "N", [{"single_stage": "seed"}, {"single_stage": "accepted"}], rounds,
                          [{"event": "gepa_proposal", "iteration": 1, "new_instructions": {"single_stage": "rejected"}},
                           {"event": "gepa_proposal", "iteration": 2, "new_instructions": {"single_stage": "accepted"}}])
            registry = json.loads((root / "version_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry[1]["status"], "rejected_archived")
            self.assertEqual(registry[2]["version"], "N2")
            self.assertEqual((root / "prompts/N0.txt").read_text().strip(), "seed")

    def test_all_three_aggregate_metrics_match_senior_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = [{"sample_id": "a", "reference": {"actions": [
                {"source_index": 0, "text": "散勾六弦"}, {"source_index": 1, "text": ""}]}},
                {"sample_id": "b", "reference": {"actions": [
                    {"source_index": 0, "text": "大指七徽挑5弦按音"}]}}]
            predictions = [{"sample_id": "a", "protocol_valid": True,
                            "jianzi_rows": [[0, "(散勾6弦)"], [1, "多填"], [3, "额外索引"]]},
                           {"sample_id": "b", "protocol_valid": False, "jianzi_rows": []}]
            outputs = []
            for ref, pred in zip(refs, predictions):
                metrics, _, _ = teacher_feedback(ref["reference"]["actions"], pred,
                                                 protocol_ok=pred["protocol_valid"])
                outputs.append({"protocol_ok": pred["protocol_valid"], "teacher_score": metrics})
            for filename, records in (("refs.jsonl", refs), ("preds.jsonl", predictions)):
                (root / filename).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records), encoding="utf-8")
            official_score(SimpleNamespace(reference=root / "refs.jsonl", predictions=root / "preds.jsonl", output=root / "score.json"))
            expected = json.loads((root / "score.json").read_text(encoding="utf-8"))["aggregate"]
            actual = aggregate_official(outputs)
            for key in ("exact_cell_rate", "nonempty_recall", "protocol_valid_rate"):
                self.assertEqual(actual[key], expected[key])

    def test_split_hash_partition_and_family_disjointness(self):
        manifest = json.loads((TRAIN.parent / "new_dataset_gepa_split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_sha256"], training_identity()["source_sha256"])
        roles = list(manifest["splits"].values())
        self.assertEqual(sum(role["rows"] for role in roles), 3976)
        self.assertEqual(sum(len(role["score_keys"]) for role in roles), 153)
        for i, role in enumerate(roles):
            for other in roles[i + 1:]:
                self.assertFalse(set(role["families"]) & set(other["families"]))
                self.assertFalse(set(role["score_keys"]) & set(other["score_keys"]))

    def test_pilot_is_10_families_and_complete_whole_scores(self):
        pilot = json.loads((TRAIN.parent / "gepa_pilot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(pilot["score_count"], 10)
        self.assertEqual(pilot["prefix_phrases"], 0)
        with TRAIN.open(encoding="utf-8") as handle:
            source_ids = [json.loads(line)["trajectory_id"] for line in handle if line.strip()]
        all_families, all_ids = set(), set()
        for role in pilot["splits"].values():
            self.assertFalse(all_families & set(role["families"]))
            self.assertFalse(all_ids & set(role["sample_ids"]))
            all_families.update(role["families"])
            all_ids.update(role["sample_ids"])
            for key in role["score_keys"]:
                chosen = [sid for sid in role["sample_ids"] if sid.startswith(key + "-")]
                complete = [sid for sid in source_ids if sid.startswith(key + "-")]
                self.assertEqual(chosen, complete)
                self.assertEqual(chosen, [f"{key}-p{i:04d}" for i in range(1, len(chosen) + 1)])
        self.assertEqual(len(all_ids), 172)
        self.assertEqual(len(all_families), 10)

    def test_native_gepa_callback_and_memory_without_model_api(self):
        class LocalAdapter:
            propose_new_texts = None
            audit_context = {}
            evolution_memory = []

            def evaluate(self, batch, candidate, capture_traces=False):
                value = 0.75 if candidate["single_stage"] == "improved" else 0.25
                return EvaluationBatch(
                    outputs=[{"value": value} for _ in batch], scores=[value for _ in batch],
                    trajectories=[{"value": value} for _ in batch] if capture_traces else None,
                    objective_scores=[{"exact_cell_rate": value} for _ in batch],
                )

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                return {"single_stage": [{"Feedback": "local deterministic fixture"}]}

        with tempfile.TemporaryDirectory() as temp:
            adapter, events = LocalAdapter(), []
            audit = TeacherGepaAudit(events.append, adapter, Path(temp), lambda *args: None)
            result = optimize(adapter=adapter, seed_candidate={"single_stage": "seed"},
                              trainset=[{"id": 0}, {"id": 1}], valset=[{"id": 2}, {"id": 3}],
                              reflection_lm=lambda prompt: "improved", reflection_minibatch_size=1,
                              max_metric_calls=12, stop_callbacks=[MaxCandidateProposalsStopper(2)],
                              run_dir=temp, callbacks=[audit], seed=1)
            self.assertGreater(result.best_score, 0.25)
            self.assertTrue(any(event["event"] == "gepa_proposal" for event in events))
            self.assertTrue(adapter.evolution_memory)


if __name__ == "__main__":
    unittest.main()
