from __future__ import annotations

import unittest

from agents.abc_to_jianzipu.plan_editor import apply_plan_edits


class PlanEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {"actions": [{
            "action_id": "a00001", "mode": "stopped", "string": 4, "hui": 7.6,
            "left_finger": "大指", "right_finger": "勾", "attack": True,
            "techniques": [],
        }]}

    def test_local_edit_creates_new_revision_without_mutating_input(self) -> None:
        result = apply_plan_edits(
            self.plan, current_revision=3, base_revision=3,
            operations=[{"op": "add_technique", "action_id": "a00001",
                         "expected": {"string": 4, "hui": 7.6}, "technique": "吟"}],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["revision"], 4)
        self.assertEqual(result["plan"]["actions"][0]["techniques"], ["吟"])
        self.assertEqual(self.plan["actions"][0]["techniques"], [])

    def test_stale_revision_is_rejected(self) -> None:
        result = apply_plan_edits(self.plan, current_revision=3, base_revision=2, operations=[])
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["code"], "STALE_REVISION")

    def test_precondition_failure_is_atomic(self) -> None:
        result = apply_plan_edits(
            self.plan, current_revision=0, base_revision=0,
            operations=[{"op": "set_fields", "action_id": "a00001",
                         "expected": {"hui": 9.0}, "fields": {"hui": 8.0}}],
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["plan"], self.plan)


if __name__ == "__main__":
    unittest.main()
