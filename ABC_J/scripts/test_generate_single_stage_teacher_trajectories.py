"""Focused contracts for the direct single-stage teacher surface."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "ABC_J" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_teacher_tool_trajectories import public_system_for, public_tools_for


class SingleStagePromptTest(unittest.TestCase):
    def test_direct_final_prompt_has_full_scope_and_candidates(self) -> None:
        prompt = public_system_for("single_stage", basic=True)
        self.assertIn("一次完成可演奏的最终减字谱", prompt)
        self.assertIn("复杂技法", prompt)
        self.assertNotIn("基础初稿只使用", prompt)
        self.assertIn("get_pitch_candidates", {tool["name"] for tool in public_tools_for("single_stage", basic=True)})


if __name__ == "__main__":
    unittest.main()
