"""Offline checks of the single-entry launcher; no credentials or API calls."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline.scripts.run_auto_prompt_training import RunLock, redact, run_path, update
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter
from baseline.scripts.run_teacher_gepa_iteration import JsonlLogger


class AutomationChecks(unittest.TestCase):
    def test_event_log_redacts_echoed_secret(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"GLM_API_KEY": "fixture_secret_123"}):
            path = Path(temp) / "events.jsonl"
            JsonlLogger(path)({"event": "provider_error", "message": "fixture_secret_123"})
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("fixture_secret_123", text)
            self.assertIn("API_KEY_REDACTED", text)

    def test_run_name_cannot_escape_runs_directory(self):
        for name in ("..", "../escape", "D:/escape", "a/b", "a\\b"):
            with self.assertRaises(ValueError):
                run_path(name)

    def test_lock_rejects_duplicate_and_is_cleaned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with RunLock(root):
                with self.assertRaises(RuntimeError):
                    with RunLock(root):
                        self.fail("duplicate lock accepted")
            self.assertFalse((root / "auto_training.lock").exists())

    def test_progress_page_redacts_secret(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"GLM_API_KEY": "fixture_secret_123"}):
            root = Path(temp)
            update(root, {"status": "failed", "stage": "local fixture", "message": "fixture_secret_123"})
            for filename in ("auto_training_status.json", "auto_training.html"):
                text = (root / filename).read_text(encoding="utf-8")
                self.assertNotIn("fixture_secret_123", text)
                self.assertIn("API_KEY_REDACTED", text)

    def test_network_failure_does_not_become_prompt_score(self):
        adapter = TeacherExactGuqinAdapter.__new__(TeacherExactGuqinAdapter)
        adapter._session_cache = {"fixture": {"x": {"ok": False, "error": "APIConnectionError: local fixture"}}}
        with patch("baseline.gepa_guqin.adapter.GuqinAgentAdapter.evaluate", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "基础设施"):
                adapter.evaluate([], {"single_stage": "fixture"})


if __name__ == "__main__":
    unittest.main()
