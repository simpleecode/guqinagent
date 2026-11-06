"""Offline safeguards for continuous search; no API requests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from baseline.gepa_guqin.handbook import compose_system
from baseline.scripts.continue_new_data_gepa import confirmed_improvement, confirmation_needed


class ContinuousSearchTests(unittest.TestCase):
    def fixture(self, root, candidate="new prompt"):
        def save(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        save(root / "selection.json", {"safe_best_is_seed": False})
        save(root / "best_candidate.json", {"safe_best_candidate": {"single_stage": candidate}})
        result = {"status": "complete", "configuration": {"prompt_hashes": {
            "candidate": hashlib.sha256(compose_system(candidate).encode()).hexdigest()}},
            "repeat_mean_scores": {
                "baseline": {"exact_cell_rate": .2, "nonempty_recall": .95, "protocol_valid_rate": 1},
                "candidate": {"exact_cell_rate": .3, "nonempty_recall": .96, "protocol_valid_rate": 1}}}
        save(root / "confirmation/paired_result.json", result)
        return result, save

    def test_success_requires_recall_not_to_regress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, save = self.fixture(root)
            self.assertTrue(confirmed_improvement(root))
            result["repeat_mean_scores"]["candidate"]["nonempty_recall"] = .94
            save(root / "confirmation/paired_result.json", result)
            self.assertFalse(confirmed_improvement(root))
            self.assertFalse(confirmation_needed(root))

    def test_stale_candidate_confirmation_is_not_a_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, save = self.fixture(root)
            save(root / "best_candidate.json", {"safe_best_candidate": {"single_stage": "different prompt"}})
            self.assertFalse(confirmed_improvement(root))
            self.assertTrue(confirmation_needed(root))
            self.assertFalse((root / "confirmation").exists())
            self.assertEqual(len(list((root / "confirmation_history").glob("*/paired_result.json"))), 1)
            save(root / "best_candidate.json", {"safe_best_candidate": {"single_stage": "new prompt"}})
            self.assertFalse(confirmation_needed(root))


if __name__ == "__main__":
    unittest.main()
