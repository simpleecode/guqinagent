from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from ABC_J.scripts.generate_teacher_tool_trajectories import parse_teacher_envelope
from ABC_J.scripts.generate_teacher_tool_trajectories import response_content_blocks


class TeacherToolProtocolTests(unittest.TestCase):
    def test_tool_envelope_accepts_batch_calls(self) -> None:
        payload = parse_teacher_envelope(
            '{"decision_summary":"查询两个起音。","tool_calls":['
            '{"name":"get_pitch_candidates","arguments":{"source_indices":[1,2]}}]}'
        )
        self.assertEqual(payload["tool_calls"][0]["name"], "get_pitch_candidates")
        self.assertEqual(payload["tool_calls"][0]["arguments"]["source_indices"], [1, 2])

    def test_tool_turn_wrapper_is_unwrapped(self) -> None:
        payload = parse_teacher_envelope(
            '{"tool_turn":{"decision_summary":"查询。","tool_calls":['
            '{"name":"get_pitch_candidates","arguments":{"source_indices":[1]}}]}}'
        )
        self.assertEqual(payload["tool_calls"][0]["name"], "get_pitch_candidates")

    def test_prefixed_reasoning_and_calls_only_json_is_recovered(self) -> None:
        payload = parse_teacher_envelope(
            '【公开思考】\n先核对当前音与前段的衔接。\n【工具调用】\n'
            '{"tool_calls":[{"name":"edit_plan","arguments":{"jianzi_rows":[[1,"勾三弦"]]}}]}'
        )
        self.assertEqual(payload["decision_summary"], "先核对当前音与前段的衔接。")
        self.assertEqual(payload["tool_calls"][0]["name"], "edit_plan")

    def test_chinese_labelled_envelope_is_recovered_losslessly(self) -> None:
        payload = parse_teacher_envelope(
            'decision_summary：逐音分析后，需要清除复合动作覆盖的后续行。\n\n'
            'tool_calls：[{' 
            '"name":"edit_plan","arguments":{"jianzi_rows":[[148,""]]}}]'
        )
        self.assertEqual(
            payload["decision_summary"],
            "逐音分析后，需要清除复合动作覆盖的后续行。",
        )
        self.assertEqual(payload["tool_calls"][0]["arguments"]["jianzi_rows"], [[148, ""]])

    def test_chinese_labelled_envelope_survives_compacted_whitespace(self) -> None:
        payload = parse_teacher_envelope(
            'decision_summary：逐音分析后提交修改。 tool_calls：'
            '[{"name":"edit_plan","arguments":{"jianzi_rows":[[149,""] ]}}]'
        )
        self.assertEqual(payload["decision_summary"], "逐音分析后提交修改。")
        self.assertEqual(payload["tool_calls"][0]["arguments"]["jianzi_rows"], [[149, ""]])

    def test_stringified_arguments_object_is_normalized(self) -> None:
        payload = parse_teacher_envelope(
            '{"decision_summary":"提交。","tool_calls":[{"name":"edit_plan",'
            '"arguments":"{\\"jianzi_rows\\":[[150,\\"勾三弦\\"]]}"}]}'
        )
        self.assertEqual(payload["tool_calls"][0]["arguments"], {"jianzi_rows": [[150, "勾三弦"]]})

    def test_calls_only_json_without_reasoning_is_not_recovered(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"tool_calls":[{"name":"edit_plan","arguments":{"jianzi_rows":[]}}]}'
            )

    def test_empty_tool_calls_are_a_valid_noop_envelope(self) -> None:
        payload = parse_teacher_envelope(
            '{"decision_summary":"本段无需修改。","tool_calls":[]}'
        )
        self.assertEqual(payload["tool_calls"], [])

    def test_tool_envelope_requires_summary(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"tool_calls":[{"name":"get_pitch_candidates",'
                '"arguments":{"source_indices":[1]}}]}'
            )

    def test_private_gqs_must_not_appear_in_public_summary(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"GQS提示此处使用注下。","tool_calls":['
                '{"name":"edit_plan","arguments":{"jianzi_rows":[[1,"注下"]]}}]}'
            )

    def test_teacher_prompt_must_not_appear_in_public_summary(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"教师提示要求此处写泛止。","tool_calls":['
                '{"name":"edit_plan","arguments":{"jianzi_rows":[[1,"泛止"]]}}]}'
            )

    def test_private_reference_indirection_must_not_appear_in_public_summary(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"第16行与参考一致，保持注下。","tool_calls":['
                '{"name":"edit_plan","arguments":{"jianzi_rows":[[16,"注下七徽九分"]]}}]}'
            )

    def test_dynamic_private_reference_text_can_be_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"根据注下七徽九分，修改第16行。","tool_calls":['
                '{"name":"edit_plan","arguments":{"jianzi_rows":[[16,"注下七徽九分"]]}}]}',
                forbidden_phrases=("注下七徽九分",),
            )

    def test_decision_summary_has_no_arbitrary_length_cap(self) -> None:
        summary = "逐音核对。" + ("说明当前公开谱面与音高关系。" * 120)
        payload = parse_teacher_envelope(
            '{"decision_summary":' + json.dumps(summary, ensure_ascii=False)
            + ',"tool_calls":[{"name":"edit_plan","arguments":'
            '{"jianzi_rows":[]}}]}'
        )
        self.assertEqual(payload["decision_summary"], summary)

    def test_full_response_content_preserves_thinking_for_next_turn(self) -> None:
        thinking = SimpleNamespace(model_dump=lambda exclude_none=True: {
            "type": "thinking", "thinking": "private reasoning",
        })
        text = SimpleNamespace(model_dump=lambda exclude_none=True: {
            "type": "text", "text": "public envelope",
        })
        blocks = response_content_blocks(SimpleNamespace(content=[thinking, text]))
        self.assertEqual([block["type"] for block in blocks], ["thinking", "text"])
        self.assertEqual(blocks[0]["thinking"], "private reasoning")

    def test_final_envelope_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"编辑完成。","final":{"patches":[]}}'
            )

    def test_prose_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope("Let me inspect the phrase first.")

    def test_explicit_glm_noop_prose_is_normalized(self) -> None:
        payload = parse_teacher_envelope(
            "逐音核对后无需改写当前减字，不调用 edit_plan。"
        )
        self.assertEqual(payload["tool_calls"], [])

    def test_only_missing_trailing_delimiters_are_repaired(self) -> None:
        payload = parse_teacher_envelope(
            '{"decision_summary":"查询。","tool_calls":['
            '{"name":"get_pitch_candidates","arguments":{"source_indices":[1,2]}}'
        )
        self.assertEqual(payload["tool_calls"][0]["arguments"]["source_indices"], [1, 2])

    def test_json_with_prose_and_fence_is_extracted(self) -> None:
        payload = parse_teacher_envelope(
            '说明如下：\n```json\n{"decision_summary":"查询。",'
            '"tool_calls":[{"name":"list_context","arguments":{}}]}\n```\n'
        )
        self.assertEqual(payload["tool_calls"][0]["name"], "list_context")

    def test_malformed_middle_is_not_repaired(self) -> None:
        with self.assertRaises(ValueError):
            parse_teacher_envelope(
                '{"decision_summary":"查询。" "tool_calls":[]}'
            )


if __name__ == "__main__":
    unittest.main()
