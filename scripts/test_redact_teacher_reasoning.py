from __future__ import annotations

import unittest

from scripts.redact_teacher_reasoning import (
    BANNED_PUBLIC_TERMS,
    normalize_summary,
    parse_json_object,
    private_reference_texts,
    factual_tokens,
    rewrite_prompt,
    validate_rewritten_summary,
)


class ReasoningRedactionTests(unittest.TestCase):
    def test_parse_json_object_accepts_fenced_response(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"summaries":["按音连接"]}\n```'),
                         {"summaries": ["按音连接"]})

    def test_empty_or_private_terms_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_summary("")
        with self.assertRaises(ValueError):
            normalize_summary("参考谱面建议改为注下")

    def test_private_reference_text_is_allowed_without_source_marker(self) -> None:
        private = {"teacher_private": {"annotation_gqs":
            '音｜[718,"6̣","名指徽外挑4弦"]'}}
        self.assertIn("名指徽外挑4弦", private_reference_texts(private))
        self.assertEqual(
            validate_rewritten_summary(
                "改为名指徽外挑4弦以增强连贯性",
                private,
                "该处原为名指徽外挑4弦",
            ),
            "改为名指徽外挑4弦以增强连贯性",
        )

    def test_rewrite_cannot_introduce_a_new_string_or_hui_number(self) -> None:
        private = {"teacher_private": {"annotation_gqs": ""}}
        self.assertEqual(factual_tokens("取一弦十徽"), {"1弦", "10徽"})
        with self.assertRaises(ValueError):
            validate_rewritten_summary("改为二弦十一徽", private, "取一弦十徽")

    def test_banned_terms_include_reference_surface_variants(self) -> None:
        self.assertIn("参考谱面", BANNED_PUBLIC_TERMS)
        self.assertIn("参考谱", BANNED_PUBLIC_TERMS)
        self.assertIn("参考建议", BANNED_PUBLIC_TERMS)

    def test_rewrite_prompt_requires_one_summary_per_call(self) -> None:
        prompt = rewrite_prompt({"agent_stage": "guqinization"}, [{"original_summary": "原文"}])
        self.assertIn("summary", prompt)
        self.assertIn("不要按音符拆成多条", prompt)


if __name__ == "__main__":
    unittest.main()
