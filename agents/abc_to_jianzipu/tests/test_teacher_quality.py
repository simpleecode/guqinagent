from __future__ import annotations

import unittest
from pathlib import Path

from agents.abc_to_jianzipu.jianzi_renderer import render_jianzi_surface
from agents.abc_to_jianzipu.teacher_trajectory import _render_phrase_lines
from ABC_J.scripts.generate_teacher_tool_trajectories import (
    RealToolRuntime,
    TOOLS,
    blank_plan_from_item,
    harmonic_region_at_phrase_start,
    public_tools_for,
    render_teacher_reference_gqs,
    render_annotation_gqs,
    render_teacher_system,
    expand_jianzi_rows,
    canonical_jianzi_text,
    infer_jianzi_text_patches,
    is_all_empty_reference_phrase,
    finalize_noop_review,
    private_reference_semantics_rules,
    render_compound_gesture_knowledge,
    COMPLEX_GESTURE_KNOWLEDGE,
    public_system_for,
    render_grouped_candidate_table,
    validate_jianzi_only,
    can_accept_empty_tool_turn,
    render_edit_preview,
)
from agents.abc_to_jianzipu.trajectory_replay import (
    compare_replay_to_patch_targets,
    replay_patches,
)
from agents.abc_to_jianzipu.reference_parser import AUDIT


class EditableProtocolTests(unittest.TestCase):
    def test_explicit_open_note_overrides_active_harmonic_region(self) -> None:
        tuning = {
            "open_strings": [
                {"string": index + 1, "pitch": pitch, "octave": octave}
                for index, (pitch, octave) in enumerate([
                    ("C", 3), ("D", 3), ("F", 3), ("G", 3),
                    ("A", 3), ("C", 4), ("D", 4),
                ])
            ]
        }
        report = AUDIT.audit({
            "metadata": {"tonic": "1=C", "tuning": tuning},
            "notes": [
                {"index": 0, "jianpu": "1", "jianzi": "泛起食指七徽勾一弦"},
                {"index": 1, "jianpu": "2", "jianzi": "散音擘七弦"},
            ],
        }, 50.0)
        detail = next(row for row in report["details"] if row["index"] == 1)
        self.assertEqual(detail["status"], "matched")

    def test_noop_review_gets_one_exact_conclusion(self) -> None:
        self.assertEqual(
            finalize_noop_review("逐音复核后，当前指法与衔接合理。"),
            "逐音复核后，当前指法与衔接合理。综上所述，无需修改。",
        )
        self.assertEqual(
            finalize_noop_review("逐音复核后可以保留。综上所述，无需修改。"),
            "逐音复核后可以保留。综上所述，无需修改。",
        )

    def test_guqinizer_targets_use_written_jianzi_text_even_when_fields_conflict(self) -> None:
        baseline = {"actions": [{
            "source_index": 7, "jianzi_text": "大指七徽勾4弦",
        }]}
        references = [{
            "source_index": 7, "text": "绰大指七徽勾4弦", "jianzi_text": "绰大指七徽勾4弦",
            "confidence_class": "conflicting", "confidence": 0.1,
            "evidence": ["pitch_audit_mismatched"],
        }]
        patches = infer_jianzi_text_patches(baseline, references)["patches"]
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["patch_type"], "SET_JIANZI_TEXT")
        self.assertEqual(patches[0]["after"]["jianzi_text"], "绰大指七徽勾4弦")

    def test_unusable_empty_reference_is_still_a_surface_target(self) -> None:
        baseline = {"actions": [{"source_index": 8, "jianzi_text": "散勾1弦"}]}
        references = [{
            "source_index": 8, "text": "", "jianzi_text": "",
            "confidence_class": "unusable", "confidence": 0.0,
        }]
        patches = infer_jianzi_text_patches(baseline, references)["patches"]
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0]["after"]["jianzi_text"], "")

    def test_none_reference_is_an_explicit_empty_surface_target(self) -> None:
        baseline = {"actions": [{"source_index": 9, "jianzi_text": "吟"}]}
        references = [{
            "source_index": 9, "text": None, "jianzi_text": None,
            "confidence_class": "weak", "confidence": 0.2,
        }]
        patches = infer_jianzi_text_patches(baseline, references)["patches"]
        self.assertEqual(patches[0]["after"]["jianzi_text"], "")

    def test_empty_tool_turn_can_retain_a_complete_plan_despite_surface_targets(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 7, "jianzi_text": "名指九徽勾4弦",
            }]},
            "input": {"metadata": {"tonic": "1=C"}, "notes_without_jianzi": [
                {"index": 7, "jianpu": "2", "abc": "D"},
            ]},
        }
        self.assertTrue(can_accept_empty_tool_turn(item))

    def test_empty_tool_turn_rejects_an_incomplete_fingering_plan(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 7, "jianzi_text": None,
            }]},
            "input": {"metadata": {"tonic": "1=C"}, "notes_without_jianzi": [
                {"index": 7, "jianpu": "2", "abc": "D"},
            ]},
        }
        self.assertFalse(can_accept_empty_tool_turn(item))

    def test_empty_tool_turn_accepts_runtime_completed_fingering_plan(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 7, "jianzi_text": None,
            }]},
            "input": {"notes_without_jianzi": [
                {"index": 7, "jianpu": "2", "abc": "D"},
            ]},
        }
        patches = [{"patch_type": "SET_JIANZI_TEXT", "source_index": 7,
                    "after": {"jianzi_text": "名指九徽勾4弦"}}]
        self.assertTrue(can_accept_empty_tool_turn(item, patches))

    def test_edit_preview_uses_plain_filled_label(self) -> None:
        item = {"input": {"notes_without_jianzi": [
            {"index": 7, "jianpu": "2", "abc": "D"},
        ]}}
        actions = [{
            "source_index": 7, "mode": "open", "string": 4,
            "attack": True, "jianzi_text": "挑4弦",
        }]
        patches = [{"source_index": 7, "after": {"jianzi_text": "挑4弦"}}]
        rendered = render_edit_preview(actions, patches, True, [], item)
        self.assertIn("7｜2｜已填写｜", rendered)
        self.assertNotIn("Agent 已填写", rendered)

    def test_edit_preview_places_pitch_warning_after_affected_row(self) -> None:
        item = {"input": {"notes_without_jianzi": [
            {"index": 7, "event_index": 3, "jianpu": "2", "abc": "D"},
        ]}}
        actions = [{
            "source_index": 7, "mode": "open", "string": 4,
            "attack": True, "jianzi_text": "散挑4弦",
        }]
        patches = [{"source_index": 7, "after": {"jianzi_text": "散挑4弦"}}]
        warnings = [{"source_index": 7, "code": "jianzi_pitch_mismatch"}]
        rendered = render_edit_preview(
            actions, patches, True, [], item, warnings
        )
        self.assertIn(
            "3｜2｜已填写｜[散挑4弦]:warning:音高不匹配",
            rendered,
        )

    def test_public_guqinizer_edit_plan_runs_pitch_audit_without_reference(self) -> None:
        item = {
            "score_key": "test", "phrase_id": "p0001",
            "input": {
                "metadata": {"tonic": "1=C"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": [
                    {"index": 0, "event_index": 0, "jianpu": "1̇"},
                    {"index": 1, "event_index": 1, "jianpu": "5"},
                ],
            },
            "baseline_plan": {"actions": [
                {"source_index": 0, "jianzi_text": "大指七徽挑六弦"},
                {"source_index": 1, "jianzi_text": ""},
            ]},
        }
        runtime = RealToolRuntime(item, {}, basic_fingering=False)
        result = runtime.invoke(
            "edit_plan", {"jianzi_rows": [[1, "注下七徽三分"]]}
        )
        self.assertTrue(result["result"]["valid"])
        self.assertIn(
            "1｜5｜已填写｜[注下七徽三分]:warning:音高不匹配",
            result["result"]["text"],
        )

    def test_edit_plan_reports_identical_rows_as_unchanged(self) -> None:
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "notes_without_jianzi": [{
                    "index": 25, "event_index": 25,
                    "jianpu": "3", "abc": "E",
                }],
            },
            "baseline_plan": {"actions": [{
                "source_index": 25,
                "jianzi_text": "名指九徽勾五弦",
            }]},
            "reference_plan": {"actions": []},
        }
        runtime = RealToolRuntime(item, {}, toward_reference=True)
        result = runtime.invoke(
            "edit_plan", {"jianzi_rows": [[25, "名指九徽勾五弦"]]}
        )
        self.assertTrue(result["result"]["valid"])
        self.assertNotIn("unchanged_event_indices", result["result"])
        self.assertIn("未变更｜25｜提交值与当前减字相同", result["result"]["text"])

    def test_edit_plan_omits_empty_unchanged_event_indices(self) -> None:
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "notes_without_jianzi": [{
                    "index": 25, "event_index": 25,
                    "jianpu": "3", "abc": "E",
                }],
            },
            "baseline_plan": {"actions": []},
            "reference_plan": {"actions": []},
        }
        runtime = RealToolRuntime(item, {}, toward_reference=True)
        result = runtime.invoke(
            "edit_plan", {"jianzi_rows": [[25, "名指九徽勾五弦"]]}
        )
        self.assertTrue(result["result"]["valid"])
        self.assertNotIn("unchanged_event_indices", result["result"])

    def test_private_prompt_teaches_canonical_jianzi_field_order(self) -> None:
        source = Path(
            "ABC_J/scripts/generate_teacher_tool_trajectories.py"
        ).read_text(encoding="utf-8")
        self.assertIn("名指九徽勾六弦", source)
        self.assertIn("泛音名指七徽勾四弦", source)
        self.assertIn("绰名指十徽八分勾三弦", source)
        self.assertIn("名指九徽打三弦", source)

    def test_pitch_warning_uses_normalized_score_tuning_and_does_not_block(self) -> None:
        open_strings = [
            {"string": index + 1, "pitch": pitch, "octave": octave}
            for index, (pitch, octave) in enumerate([
                ("C", 3), ("D", 3), ("F", 3), ("G", 3),
                ("A", 3), ("C", 4), ("D", 4),
            ])
        ]
        item = {
            "input": {
                "metadata": {
                    "tonic": "1=C",
                    "tuning": {"open_strings": open_strings},
                },
                # Deliberately differs from descriptive metadata on string 2.
                "normalized_tuning": {
                    "open_midi": [48, 48, 53, 55, 57, 60, 62],
                },
                "notes_without_jianzi": [{
                    "index": 1, "event_index": 0, "jianpu": "1̣", "abc": "C,",
                }],
            },
            "baseline_plan": {"actions": [{
                "source_index": 1, "jianzi_text": None,
            }]},
            "reference_plan": {"actions": []},
        }
        runtime = RealToolRuntime(item, {}, basic_fingering=True)
        matched = runtime.invoke(
            "edit_plan", {"jianzi_rows": [[0, "散挑2弦"]]}
        )
        self.assertTrue(matched["result"]["valid"])
        self.assertNotIn("音高不匹配", matched["result"]["text"])

        item["input"]["normalized_tuning"]["open_midi"][1] = 50
        runtime = RealToolRuntime(item, {}, basic_fingering=True)
        mismatched = runtime.invoke(
            "edit_plan", {"jianzi_rows": [[0, "散挑2弦"]]}
        )
        self.assertTrue(mismatched["result"]["valid"])
        self.assertIn(
            "0｜1̣｜已填写｜[散挑2弦]:warning:音高不匹配",
            mismatched["result"]["text"],
        )
        audit_warnings = runtime.calls[-1]["result"]["result"]["warnings"]
        self.assertTrue(any(
            warning.get("code") == "jianzi_pitch_mismatch"
            for warning in audit_warnings
        ))
        _, report = validate_jianzi_only(
            item,
            [{"patch_type": "SET_JIANZI_TEXT", "source_index": 1,
              "after": {"jianzi_text": "散挑2弦"}}],
            require_complete=False,
        )
        self.assertGreater(
            report["offline_quality"]["high_confidence_pitch_mismatches"], 0
        )
        self.assertTrue(report["offline_quality"]["training_eligible"])

    def test_empty_text_edit_stays_in_minimal_protocol(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "mode": None, "string": None,
                "hui": None, "left_finger": None, "right_finger": None,
                "techniques": [], "pre_attack_techniques": [], "attack": True,
                "jianzi_text": "散挑1弦",
            }]},
            "reference_plan": {"actions": []},
            "input": {
                "metadata": {"tonic": "1=C"},
                "notes_without_jianzi": [
                    {"index": 1, "jianpu": "1", "abc": "C"},
                ],
            },
        }
        runtime = RealToolRuntime(item, {}, toward_reference=True)
        result = runtime.invoke("edit_plan", {"jianzi_rows": []})
        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["valid"], result["result"]["text"])
        self.assertNotIn("pending_mode", result["result"]["text"])

    def test_public_edit_preview_never_leaks_reference_text(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "mode": "open", "string": 7,
                "hui": None, "left_finger": None, "right_finger": "挑",
                "techniques": [], "pre_attack_techniques": [], "attack": True,
                "jianzi_text": "散挑七弦",
            }]},
            "reference_plan": {"actions": [{
                "source_index": 1, "confidence_class": "verified",
                "jianzi_text": "散摘7弦",
            }]},
            "input": {
                "metadata": {"tonic": "1=C"},
                "notes_without_jianzi": [
                    {"index": 1, "jianpu": "2", "abc": "D"},
                ],
            },
        }
        runtime = RealToolRuntime(item, {}, toward_reference=True)
        result = runtime.invoke("edit_plan", {"jianzi_rows": []})
        public_text = str(result)
        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["valid"], public_text)
        self.assertNotIn("散摘7弦", public_text)
        self.assertNotIn("reference_jianzi_text_mismatch", public_text)

    def test_blank_plan_keeps_sustain_as_display_editable_row(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "mode": "open", "string": 1,
                "hui": None, "left_finger": None, "right_finger": "挑",
                "techniques": [], "pre_attack_techniques": [], "attack": True,
            }]},
            "input": {"notes_without_jianzi": [
                {"index": 1, "abc": "C", "jianpu": "1"},
                {"index": 2, "abc": "-C", "jianpu": "－（延音）"},
                {"index": 3, "abc": "z", "jianpu": "0（休止）"},
            ]},
        }
        plan = blank_plan_from_item(item)
        by_index = {a["source_index"]: a for a in plan["actions"]}
        self.assertEqual(by_index[2]["jianzi_text"], "")
        self.assertFalse(by_index[2]["attack"])
        self.assertIn(3, by_index)
        self.assertFalse(by_index[3]["attack"])
        self.assertEqual(by_index[3]["jianzi_text"], "")

    def test_blank_plan_adds_omitted_normal_notes_as_text_editable(self) -> None:
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "attack": True, "jianzi_text": None,
            }]},
            "input": {"notes_without_jianzi": [
                {"index": 1, "abc": "C", "jianpu": "1"},
                {"index": 2, "abc": "?", "jianpu": "5̣"},
                {"index": 3, "abc": "-?", "jianpu": "－（延音）"},
                {"index": 4, "abc": "|", "jianpu": "|"},
                {"index": 5, "abc": "z", "jianpu": "0（休止）"},
            ]},
        }
        by_index = {a["source_index"]: a for a in blank_plan_from_item(item)["actions"]}
        self.assertIn(2, by_index)
        self.assertTrue(by_index[2]["attack"])
        self.assertIsNone(by_index[2]["jianzi_text"])
        self.assertNotIn(4, by_index)
        self.assertIn(5, by_index)
        self.assertFalse(by_index[5]["attack"])

    def test_not_editable_reports_all_indices_in_one_error(self) -> None:
        with self.assertRaisesRegex(ValueError, r"not editable: 8, 9, 12"):
            expand_jianzi_rows(
                [[8, "散挑1弦"], [9, "散挑2弦"], [12, ""]],
                current_text={}, allowed_indices={1, 2},
            )

    def test_harmonic_region_replays_markers_before_phrase(self) -> None:
        item = {"score_key": "s", "input": {"event_range": {"start": 20}}}
        historical = {
            ("s", "p1"): {
                "input": {"event_range": {"start": 0}},
                "reference_plan": {"actions": [
                    {"source_index": 1, "text": "泛起食指七徽勾4弦"},
                ]},
            },
            ("s", "p2"): {
                "input": {"event_range": {"start": 10}},
                "reference_plan": {"actions": [
                    {"source_index": 11, "text": "挑6弦泛止"},
                ]},
            },
        }
        self.assertFalse(harmonic_region_at_phrase_start(item, historical))

    def test_pitch_audit_replays_previous_harmonic_hui(self) -> None:
        item = {
            "input": {
                "notes_without_jianzi": [
                    {"index": 2, "jianpu": "6", "abc": "D", "duration": "八分"},
                ],
                "phrase_handoff": {
                    "previous_phrase": {
                        "notes": [{"index": 1, "jianpu": "6", "abc": "D", "duration": "八分"}],
                        "actions": [{"source_index": 1, "mode": "harmonic", "hui": 6,
                                     "jianzi_text": "泛起勾二弦六徽"}],
                    }
                },
            }
        }
        from ABC_J.scripts.generate_teacher_tool_trajectories import pitch_audit_notes
        rows = pitch_audit_notes(item, {2: {"jianzi_text": "勾二弦六徽"}})
        self.assertEqual(rows[0]["jianzi"], "泛起勾1弦六徽")
        self.assertEqual(rows[-1]["jianzi"], "勾二弦六徽")

    def test_harmonic_audit_keeps_mode_across_rest_and_inherited_hui(self) -> None:
        report = AUDIT.audit({
            "metadata": {},
            "open_midi": [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0],
            "notes": [
                {"index": -1, "jianpu": None, "jianzi": "泛起勾1弦九徽"},
                {"index": 1, "jianpu": "0（休止）", "jianzi": ""},
                {"index": 2, "jianpu": "5", "jianzi": "名指九徽勾1弦"},
            ],
        }, 50.0)
        detail = next(row for row in report["details"] if row["index"] == 2)
        self.assertEqual(detail["status"], "matched")
        self.assertEqual(detail["pairs"][0]["jianzi_midi"], 67.0)

    def test_harmonic_audit_accepts_chinese_or_arabic_string_surface(self) -> None:
        open_midi = [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]
        for text in ("名指九徽勾1弦", "名指九徽勾一弦"):
            context = AUDIT.new_context()
            AUDIT.parse_jianzi("泛起勾1弦九徽", open_midi, context)
            pitches, reason = AUDIT.parse_jianzi(text, open_midi, context)
            self.assertIsNone(reason)
            self.assertEqual(pitches, [67.0])

    def test_text_only_intermediate_is_visible_without_structural_fields(self) -> None:
        table = _render_phrase_lines(
            [{"index": 1, "jianpu": "1", "abc": "C", "duration": "四分"}],
            [{"source_index": 1, "mode": None, "attack": True,
              "jianzi_text": "散挑1弦"}],
        )
        self.assertIn("[散挑1弦]", table)
        self.assertNotIn("[减字待填写]", table)

    def test_readonly_context_uses_neutral_placeholder_for_missing_text(self) -> None:
        table = _render_phrase_lines(
            [{"index": 1, "jianpu": "3", "abc": "C", "duration": "八分"}],
            [{"source_index": 1, "mode": None, "attack": False,
              "jianzi_text": None}], readonly=True,
        )
        self.assertIn("[空]", table)
        self.assertNotIn("[减字待填写]", table)

    def test_readonly_context_harmonic_missing_text_is_not_pending(self) -> None:
        table = _render_phrase_lines(
            [{"index": 1, "jianpu": "6", "abc": "D", "duration": "八分"}],
            [{"source_index": 1, "mode": "harmonic", "string": 2,
              "hui": 4, "attack": True, "jianzi_text": None}], readonly=True,
        )
        self.assertIn("[空]", table)
        self.assertNotIn("[减字待填写]", table)

    def test_readonly_context_missing_action_is_empty_not_pending(self) -> None:
        table = _render_phrase_lines(
            [{"index": 9, "jianpu": "5", "abc": "D", "duration": "四分"}],
            [], readonly=True,
        )
        self.assertIn("9｜5｜D｜四分｜[空]", table)
        self.assertNotIn("[减字待填写]", table)

    def test_readonly_context_normalizes_legacy_pending_literal_and_barline(self) -> None:
        table = _render_phrase_lines(
            [
                {"index": 1, "jianpu": "3", "abc": "C", "duration": "八分"},
                {"index": 2, "jianpu": "|", "abc": "|", "duration": "小节线"},
                {"index": 3, "jianpu": "4", "abc": "D", "duration": "八分"},
            ],
            [
                {"source_index": 1, "mode": None, "attack": True,
                 "jianzi_text": "减字待填写"},
                {"source_index": 2, "mode": None, "attack": False,
                 "jianzi_text": "减字待填写"},
                {"source_index": 3, "mode": "stopped", "string": 4,
                 "hui": 7, "attack": True, "jianzi_text": "减字待填写"},
            ],
            readonly=True,
        )
        self.assertEqual(table.count("[空]"), 2)
        self.assertIn("小节线", table)
        self.assertNotIn("[减字待填写]", table)

    def test_readonly_context_renders_explicit_empty_text_as_empty(self) -> None:
        table = _render_phrase_lines(
            [
                {"index": 1, "jianpu": "3", "abc": "C", "duration": "八分"},
                {"index": 2, "jianpu": "4", "abc": "D", "duration": "八分"},
            ],
            [
                {"source_index": 1, "mode": None, "attack": True,
                 "jianzi_text": ""},
                {"source_index": 2, "mode": "harmonic", "string": 4,
                 "hui": 7, "attack": True, "jianzi_text": ""},
            ],
            readonly=True,
        )
        self.assertEqual(table.count("[空]"), 2)
        self.assertNotIn("[减字待填写]", table)

    def test_current_repeat_copy_does_not_prelabel_omission(self) -> None:
        note = [{"index": 1, "jianpu": "3", "abc": "C", "duration": "八分",
                 "notation_omitted": True}]
        action = [{"source_index": 1, "mode": None, "attack": False,
                   "jianzi_text": None, "notation_omitted": True}]
        current = _render_phrase_lines(note, action, readonly=False)
        readonly = _render_phrase_lines(note, action, readonly=True)
        self.assertIn("[减字待填写]", current)
        self.assertNotIn("无（由于是再作部分，省略）", current)
        self.assertIn("无（由于是再作部分，省略）", readonly)

    def test_surface_digit_variants_are_reference_equivalent(self) -> None:
        self.assertEqual(canonical_jianzi_text("散摘7弦"),
                         canonical_jianzi_text("散摘七弦"))

    def test_private_reference_uses_same_table_grammar_as_editable_phrase(self) -> None:
        item = {
            "input": {"notes_without_jianzi": [
                {"index": 1, "jianpu": "2", "abc": "D", "duration": "八分"},
            ]},
            "reference_plan": {"actions": [{
                "source_index": 1, "jianzi_text": "挑4弦",
            }]},
        }

        reference = render_teacher_reference_gqs(item)

        self.assertIn("序号｜简谱｜ABC｜时值｜谱面减字", reference)
        self.assertIn("1｜2｜D｜八分｜[挑四弦]", reference)
        self.assertNotIn("音｜[", reference)

    def test_private_reference_distinguishes_blank_from_repeat_omission(self) -> None:
        item = {
            "input": {"notes_without_jianzi": [
                {"index": 1, "jianpu": "2", "abc": "D", "duration": "八分"},
                {"index": 2, "jianpu": "3", "abc": "E", "duration": "八分",
                 "notation_omitted": True},
            ]},
            "reference_plan": {"actions": [
                {"source_index": 1, "jianzi_text": None},
            ]},
        }
        reference = render_teacher_reference_gqs(item)
        self.assertIn("1｜2｜D｜八分｜[空]", reference)
        self.assertIn(f"2｜3｜E｜八分｜[{'无（由于是再作部分，省略）'}]", reference)

    def test_wholly_blank_phrase_is_excluded_but_repeat_omission_is_not(self) -> None:
        item = {
            "input": {"notes_without_jianzi": [{"index": 1, "jianpu": "1"}]},
            "reference_plan": {"actions": [{"source_index": 1, "jianzi_text": None}]},
        }
        self.assertTrue(is_all_empty_reference_phrase(item))
        item["input"]["notes_without_jianzi"][0]["notation_omitted"] = True
        self.assertFalse(is_all_empty_reference_phrase(item))

    def test_private_rules_distinguish_empty_target_from_repeat_omission(self) -> None:
        fingering = "".join(private_reference_semantics_rules("fingering_agent"))
        guqinizer = "".join(private_reference_semantics_rules("guqinization"))
        self.assertIn("该标记本身也是允许提交的减字文字", fingering)
        self.assertIn("可通过 edit_plan.jianzi_rows 直接填写", fingering)
        self.assertIn("沿用再作动作继承的减字", fingering)
        self.assertIn("不等于把该音置为空字符串", fingering)
        self.assertIn("大指七徽挑七弦”等同于“泛音大指七徽挑七弦", fingering)
        self.assertIn('":warning:音高不匹配"', fingering)
        self.assertIn("尽量核对该行的取音与减字并修正", fingering)
        self.assertIn("[空]是确定的谱面空显示目标", guqinizer)
        self.assertIn("设为空字符串", guqinizer)
        self.assertIn("不删除声音或演奏状态", guqinizer)
        self.assertIn("显式“散”只覆盖当前音", guqinizer)
        self.assertIn('":warning:音高不匹配"', guqinizer)

    def test_guqinizer_prompt_preserves_repeat_omission_marker(self) -> None:
        public = public_system_for("guqinization")
        private = "".join(private_reference_semantics_rules("guqinization"))
        self.assertNotIn('且与私有参考一致，必须保留', public)
        self.assertIn('且与私有参考一致，必须保留', private)

    def test_guqinizer_cannot_clear_base_repeat_omission_marker(self) -> None:
        marker = "无（由于是再作部分，省略）"
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "jianzi_text": marker,
            }]},
            "reference_plan": {"actions": [{
                "source_index": 1, "jianzi_text": marker,
            }]},
            "input": {"metadata": {"tonic": "1=C"},
                      "notes_without_jianzi": [{
                          "index": 1, "jianpu": "1", "abc": "C",
                      }]},
        }
        patches = [{"patch_type": "SET_JIANZI_TEXT", "source_index": 1,
                    "after": {"jianzi_text": ""}}]
        valid, report = validate_jianzi_only(
            item, patches, toward_reference=True, require_complete=True)
        self.assertFalse(valid)
        self.assertIn(
            "repeat_omission_marker_cleared",
            [problem.get("code") for problem in report["problems"]],
        )

    def test_guqinizer_may_clear_marker_when_private_reference_differs(self) -> None:
        marker = "无（由于是再作部分，省略）"
        item = {
            "baseline_plan": {"actions": [{
                "source_index": 1, "jianzi_text": marker,
            }]},
            "reference_plan": {"actions": [{
                "source_index": 1, "jianzi_text": "吟",
            }]},
            "input": {"metadata": {"tonic": "1=C"},
                      "notes_without_jianzi": [{
                          "index": 1, "jianpu": "1", "abc": "C",
                      }]},
        }
        patches = [{"patch_type": "SET_JIANZI_TEXT", "source_index": 1,
                    "after": {"jianzi_text": ""}}]
        valid, report = validate_jianzi_only(
            item, patches, toward_reference=True, require_complete=True)
        self.assertTrue(valid, report)

    def test_compound_knowledge_is_injected_only_for_present_reference_gesture(self) -> None:
        item = {"reference_plan": {"actions": [
            {"source_index": 1, "jianzi_text": "掐撮三声"},
        ]}}
        notes = render_compound_gesture_knowledge(item)
        self.assertEqual(len(notes), 1)
        self.assertIn("掐撮三声", notes[0])
        self.assertNotIn("掐撮三声", public_system_for("fingering_agent", basic=True))

    def test_cuo_knowledge_can_be_injected_for_the_basic_private_stage(self) -> None:
        notes = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "撮"}]},
        }))
        self.assertIn("双按音的撮较少用", notes)
        # The private renderer is stage-neutral; generate_one adds these notes
        # to both basic and Guqinizer private instructions.
        self.assertNotIn("双按音的撮较少用", public_system_for("fingering_agent", basic=True))

    def test_enhanced_compound_knowledge_requires_contextual_prerequisites(self) -> None:
        chuoshang = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "绰上四徽"}]},
        }))
        cuo = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "挑七弦绰"}]},
        }))
        xiao_cuo = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "撮"}]},
        }))
        qiacuo = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "掐撮三声"}]},
        }))
        self.assertIn("不需要、也不表示新的右手拨弦", chuoshang)
        self.assertIn("与右手取声配合", cuo)
        self.assertIn("双按音的撮较少用", xiao_cuo)
        self.assertIn("分别明确不同的左手按指、徽位", xiao_cuo)
        self.assertIn("前一减字通常应先给出", qiacuo)
        self.assertIn("此前同一对弦", qiacuo)

    def test_shared_slash_alias_injects_every_matching_entry(self) -> None:
        notes = render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "撮"}]},
        })
        self.assertEqual(len(notes), 2)
        self.assertTrue(any("复合技法知识｜撮：" in note for note in notes))
        self.assertTrue(any("复合技法知识｜大撮/撮：" in note for note in notes))

    def test_slash_alias_and_single_li_are_independently_injected(self) -> None:
        aliases = "\n".join(render_compound_gesture_knowledge({
            "reference_plan": {"actions": [
                {"jianzi_text": "滔起"}, {"jianzi_text": "历至五弦"},
            ]},
        }))
        self.assertIn("复合技法知识｜滔起：", aliases)
        self.assertIn("复合技法知识｜历：", aliases)
        self.assertEqual(
            COMPLEX_GESTURE_KNOWLEDGE["掐起"],
            COMPLEX_GESTURE_KNOWLEDGE["滔起"],
        )
        self.assertEqual(
            COMPLEX_GESTURE_KNOWLEDGE["上"],
            COMPLEX_GESTURE_KNOWLEDGE["绰上"],
        )
        self.assertEqual(
            COMPLEX_GESTURE_KNOWLEDGE["抓起"],
            COMPLEX_GESTURE_KNOWLEDGE["爪起"],
        )
        self.assertNotEqual(
            COMPLEX_GESTURE_KNOWLEDGE["绰"],
            COMPLEX_GESTURE_KNOWLEDGE["绰上"],
        )
        self.assertIn("历54弦", aliases)
        self.assertIn("食指七徽历54弦", aliases)

    def test_curated_single_character_gesture_is_injected(self) -> None:
        notes = render_compound_gesture_knowledge({
            "reference_plan": {"actions": [{"jianzi_text": "逗"}]},
        })
        self.assertEqual(len(notes), 1)
        self.assertIn("复合技法知识｜逗：", notes[0])

    def test_ruyi_knowledge_uses_enhanced_manual_explanation(self) -> None:
        item = {"reference_plan": {"actions": [
            {"source_index": 1, "jianzi_text": "如一"},
        ]}}
        notes = render_compound_gesture_knowledge(item)
        self.assertEqual(len(notes), 1)
        self.assertIn("同音高", notes[0])
        self.assertIn("同时剔出", notes[0])
        self.assertEqual(COMPLEX_GESTURE_KNOWLEDGE["如一"]["category"], "右手指法")

    def test_dependent_single_gestures_and_yinhua_are_injected(self) -> None:
        item = {"reference_plan": {"actions": [
            {"source_index": 1, "jianzi_text": "吟滑"},
            {"source_index": 2, "jianzi_text": "撞"},
            {"source_index": 3, "jianzi_text": "注下九徽"},
        ]}}
        notes = render_compound_gesture_knowledge(item)
        joined = "\n".join(notes)
        self.assertIn("复合技法知识｜吟滑：", joined)
        self.assertIn("复合技法知识｜撞：", joined)
        self.assertIn("复合技法知识｜注下：", joined)
        self.assertIn("[名指六徽勾二弦] [注下七徽九分]", joined)
        self.assertNotIn("复合技法知识｜吟：", joined)

    def test_public_prompt_explains_fieldwise_inheritance(self) -> None:
        for stage in ("fingering_agent", "guqinization"):
            prompt = public_system_for(stage, basic=stage == "fingering_agent")
            self.assertIn("按字段、按动作类型的局部继承", prompt)
            self.assertIn("吟、猱、撞、绰、注", prompt)
            self.assertIn("泛起至泛止之间继承泛音状态", prompt)
            self.assertIn("不能因某字段省略就把上一音的全部状态机械照搬", prompt)

    def test_public_prompt_does_not_expand_cuo_into_right_hand_attacks(self) -> None:
        basic_prompt = public_system_for("fingering_agent", basic=True)
        self.assertIn("普通撮本身就是右手指法", basic_prompt)
        guqinizer_prompt = public_system_for("guqinization")
        self.assertNotIn("普通撮本身就是右手指法", guqinizer_prompt)

    def test_minimal_public_edit_with_readonly_pitch_candidates(self) -> None:
        tools = public_tools_for("fingering_agent", basic=True)
        self.assertEqual({tool["name"] for tool in tools},
                         {"list_context", "expand_context", "get_pitch_candidates",
                          "edit_plan"})
        edit = next(tool for tool in tools if tool["name"] == "edit_plan")
        properties = edit["input_schema"]["properties"]
        self.assertEqual(set(properties), {"jianzi_rows"})
        self.assertEqual(edit["input_schema"]["required"], ["jianzi_rows"])
        self.assertIn("置空", edit["description"])
        self.assertIn("不会删除声音", edit["description"])
        self.assertIn("置空", properties["jianzi_rows"]["description"])
        self.assertIn("不会删除声音", properties["jianzi_rows"]["description"])

    def test_guqinizer_can_query_pitch_candidates_for_warning_repair(self) -> None:
        tools = public_tools_for("guqinization")
        self.assertIn("get_pitch_candidates", {tool["name"] for tool in tools})

    def test_empty_text_prompt_explains_compound_continuation_use(self) -> None:
        prompt = public_system_for("guqinization")
        self.assertIn("将该行 jianzi_text 置空", prompt)
        self.assertIn("不会删除声音或演奏状态", prompt)

    def test_guqinizer_prompt_requires_delta_rows_and_indexed_reasoning(self) -> None:
        prompt = public_system_for("guqinization")
        self.assertIn("确实需要改写的音", prompt)
        self.assertIn("不用重发无需改变的行", prompt)
        self.assertIn("逐一分析每个音应该使用什么指法/减字", prompt)
        self.assertIn("只把决定改写的音提交", prompt)

    def test_private_rules_separate_edit_direction_from_public_reasoning(self) -> None:
        source = (Path(__file__).resolve().parents[3] / "ABC_J" / "scripts"
                  / "generate_teacher_tool_trajectories.py")
        text = source.read_text(encoding="utf-8")
        self.assertIn("私有标注可以用于内部决定哪里需要改、往什么方向改", text)
        self.assertIn("公开 reasoning 的任务是解释修改为什么在古琴演奏上合理", text)
        self.assertIn("必须从头到尾按音序审阅完整个当前段", text)
        self.assertIn("可以逐音分析，也可以每次按一小组相邻音分析", text)
        self.assertIn("允许依据完整上下文作出与私有标注不同但合理的判断", text)
        self.assertIn("走手动作的同弦连续性", text)



    def test_public_schema_has_no_editable_technique_fields(self) -> None:
        edit = next(tool for tool in TOOLS if tool["name"] == "edit_plan")
        schema_text = repr(edit["input_schema"])
        self.assertNotIn("ADD_TECHNIQUE", schema_text)
        self.assertNotIn("REMOVE_TECHNIQUE", schema_text)
        self.assertNotIn("'technique'", schema_text)
        self.assertNotIn("'placement'", schema_text)

    def test_jianzi_text_derives_technique_without_mutating_attack(self) -> None:
        baseline = {"actions": [{
            "source_index": 1, "attack": True, "right_finger": "挑",
            "techniques": [], "pre_attack_techniques": [], "jianzi_text": None,
        }]}
        patch = [{"patch_id": "text-1", "patch_type": "SET_JIANZI_TEXT",
                  "source_index": 1, "after": {"jianzi_text": "吟"}}]
        action = replay_patches(baseline, patch).actions[0]
        self.assertEqual(action["techniques"], ["吟"])
        self.assertTrue(action["attack"])
        self.assertEqual(action["right_finger"], "挑")

    def test_empty_jianzi_text_does_not_erase_playable_semantics(self) -> None:
        baseline = {"actions": [{
            "source_index": 1, "attack": True, "right_finger": "挑",
            "techniques": ["吟"], "pre_attack_techniques": [], "jianzi_text": None,
        }]}
        patch = [{"patch_id": "hide-1", "patch_type": "SET_JIANZI_TEXT",
                  "source_index": 1, "after": {"jianzi_text": ""}}]
        action = replay_patches(baseline, patch).actions[0]
        self.assertTrue(action["attack"])
        self.assertEqual(action["right_finger"], "挑")
        self.assertEqual(action["techniques"], ["吟"])


class JianziRendererTests(unittest.TestCase):
    def test_open_surface_never_inserts_left_hand_none(self) -> None:
        action = {"mode": "open", "string": 7, "right_finger": "挑",
                  "left_finger": None, "attack": True, "techniques": [],
                  "jianzi_text": "散挑7弦"}
        self.assertEqual(render_jianzi_surface(action), "[散挑7弦]")

    def test_non_open_order_is_left_hui_right_string(self) -> None:
        action = {"mode": "stopped", "string": 4, "hui": 9.0,
                  "left_finger": "大指", "right_finger": "勾",
                  "attack": True, "techniques": ["吟"],
                  "jianzi_text": "大指九徽勾4弦吟"}
        self.assertEqual(render_jianzi_surface(action), "[大指九徽勾4弦吟]")

    def test_standalone_yin_does_not_repeat_inherited_hui(self) -> None:
        action = {"mode": "stopped", "string": 2, "hui": 7.9,
                  "left_finger": "大指", "right_finger": None,
                  "attack": False, "techniques": ["吟"], "jianzi_text": "吟"}
        self.assertEqual(render_jianzi_surface(action), "[吟]")

    def test_harmonic_double_string_and_slide_surfaces(self) -> None:
        chord = {"mode": "harmonic", "string": 4, "hui": 9.0,
                 "left_finger": "大指", "right_finger": "撮",
                 "string2": 3, "mode2": "open", "attack": True,
                 "techniques": [], "jianzi_text": "大指九徽撮4弦＋散3弦"}
        slide = {"mode": "stopped", "string": 4, "hui": 7.9,
                 "left_finger": "大指", "right_finger": None,
                 "attack": False, "techniques": ["绰"],
                 "jianzi_text": "绰上七徽九分"}
        self.assertEqual(render_jianzi_surface(chord), "[大指九徽撮4弦＋散3弦]")
        self.assertEqual(render_jianzi_surface(slide), "[绰上七徽九分]")

    def test_compound_gesture_surface_is_preserved_atomically(self) -> None:
        action = {"mode": "stopped", "string": 4, "hui": 9.0,
                  "left_finger": "大指", "right_finger": "撮",
                  "compound_gesture": "掐撮三声", "attack": True,
                  "techniques": [], "jianzi_text": "掐撮三声"}
        self.assertEqual(render_jianzi_surface(action), "[掐撮三声]")

    def test_empty_jianzi_text_keeps_action_but_renders_nothing(self) -> None:
        action = {"source_index": 260, "mode": "stopped", "string": 1,
                  "hui": 9.0, "left_finger": "大指", "right_finger": "挑",
                  "attack": True, "techniques": [], "jianzi_text": ""}
        self.assertEqual(render_jianzi_surface(action), "")
        replay = replay_patches(
            {"actions": [{**action, "jianzi_text": None}]},
            [{"patch_id": "hide-260", "patch_type": "SET_JIANZI_TEXT",
              "source_index": 260, "before": {"jianzi_text": None},
              "after": {"jianzi_text": ""}}],
        )
        self.assertTrue(replay.valid)
        self.assertEqual(replay.actions[0]["string"], 1)
        self.assertEqual(replay.actions[0]["jianzi_text"], "")

    def test_chuo_pre_attack_surface_precedes_left_hand_and_pluck(self) -> None:
        action = {"mode": "stopped", "string": 6, "hui": 5.6,
                  "left_finger": "大指", "right_finger": "挑", "attack": True,
                  "techniques": ["绰"], "pre_attack_techniques": ["绰"],
                  "jianzi_text": "绰大指五徽六分挑6弦"}
        self.assertEqual(render_jianzi_surface(action), "[绰大指五徽六分挑6弦]")

    def test_taoqi_surface_keeps_left_finger_and_hui(self) -> None:
        action = {"mode": "stopped", "string": 1, "hui": 10.0,
                  "left_finger": "名指", "right_finger": None, "attack": False,
                  "techniques": ["滔起"], "jianzi_text": "名指十徽滔起"}
        self.assertEqual(render_jianzi_surface(action), "[名指十徽滔起]")


class TeacherReferenceGqsTests(unittest.TestCase):
    def test_compacts_reference_and_omits_machine_only_rows(self) -> None:
        item = {
            "input": {"notes_without_jianzi": [
                {"index": 10, "abc": "A,2", "jianpu": "2̣", "jianpu_alt": None},
                {"index": 11, "abc": "B,2", "jianpu": "3̣", "jianpu_alt": None},
                {"index": 12, "abc": "C2", "jianpu": "4̣", "jianpu_alt": None},
            ]},
            "reference_plan": {"actions": [
                {"source_index": 10, "text": "掐撮三声", "attack": True,
                 "compound_gesture": "掐撮三声", "confidence_class": "verified",
                 "explicit_fields": ["attack", "compound_gesture"],
                 "inherited_fields": []},
                {"source_index": 11, "text": "绰上五徽", "attack": False,
                 "mode": "stopped", "string": 4, "hui": 5.0,
                 "techniques": ["绰"], "confidence_class": "weak",
                 "explicit_fields": ["attack", "hui", "mode"],
                 "inherited_fields": ["left_finger", "string"]},
                {"source_index": 12, "text": "", "attack": False,
                 "confidence_class": "unusable", "explicit_fields": [],
                 "inherited_fields": [], "diagnostics": ["empty_reference"]},
            ]},
        }

        rendered = render_teacher_reference_gqs(item)

        self.assertIn("当前段参考｜只读｜仅作改进方向", rendered)
        self.assertIn("掐撮三声", rendered)
        self.assertIn("绰上五徽", rendered)
        self.assertIn("序号｜简谱｜ABC｜时值｜谱面减字", rendered)
        self.assertIn("12｜4̣｜C2｜-｜[空]", rendered)
        self.assertNotIn("confidence", rendered)
        self.assertNotIn("explicit_fields", rendered)
        self.assertNotIn("inherited_fields", rendered)
        self.assertNotIn("diagnostics", rendered)
        self.assertNotIn("evidence", rendered)

    def test_reference_gqs_is_appended_as_real_multiline_text(self) -> None:
        reference_table = '当前段参考｜只读｜仅作改进方向\n1｜1｜C｜四分｜[散挑四弦]'
        rendered = render_teacher_system({"role": "教师"}, reference_table)

        self.assertIn('\n\n【教师私有参考谱】\n当前段参考｜', rendered)
        self.assertNotIn('reference_gqs', rendered)
        self.assertNotIn('\\n1｜', rendered)

    def test_annotation_gqs_keeps_empty_rows_for_visual_comparison(self) -> None:
        item = {
            "input": {"notes_without_jianzi": [
                {"index": 1, "abc": "D2", "jianpu": "5", "jianpu_alt": None},
                {"index": 2, "abc": "E2", "jianpu": "6", "jianpu_alt": None},
            ]},
            "reference_plan": {"actions": [
                {"source_index": 1, "text": "散挑7弦"},
                {"source_index": 2, "text": ""},
            ]},
        }

        rendered = render_annotation_gqs(item)

        self.assertIn('音｜[1,"5","散挑7弦"]', rendered)
        self.assertIn('音｜[2,"6",""]', rendered)


class TeacherQualityTests(unittest.TestCase):
    @staticmethod
    def _item() -> dict:
        actions = [{"source_index": index, "mode": "open", "string": 7,
                    "hui": None, "left_finger": None, "right_finger": "挑",
                    "attack": True, "techniques": [], "jianzi_text": "散挑7弦"}
                   for index in range(1, 6)]
        references = [{"source_index": index, "confidence_class": "verified",
                       "explicit_fields": ["right_finger"], "inherited_fields": [],
                       "right_finger": "勾", "attack": True, "techniques": []}
                      for index in range(1, 6)]
        references.append({"source_index": 99, "confidence_class": "weak",
                           "explicit_fields": [], "inherited_fields": ["mode", "hui"],
                           "mode": "harmonic", "hui": 9.0, "attack": False,
                           "techniques": []})
        notes = [{"index": index, "abc": "D4", "jianpu": "6̣"}
                 for index in range(1, 6)]
        return {
            "input": {
                "metadata": {"tonic": "1=F"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": notes,
            },
            "baseline_plan": {"actions": actions},
            "reference_plan": {"actions": references},
        }

    @staticmethod
    def _patch(index: int) -> dict:
        return {"patch_id": f"p-{index}", "patch_type": "CHANGE_FINGER",
                "source_index": index, "before": {"right_finger": "挑"},
                "after": {"right_finger": "勾"}, "confidence_class": "verified"}


    def test_compact_jianzi_rows_are_typed_and_idempotent(self) -> None:
        rows = [[245, "注下七徽九分"], [246, "吟"], [260, ""]]
        patches = expand_jianzi_rows(rows, current_text={245: None, 246: "吟", 260: None})
        self.assertEqual([patch["source_index"] for patch in patches], [245, 260])
        self.assertEqual(patches[0]["patch_type"], "SET_JIANZI_TEXT")
        self.assertEqual(patches[1]["after"]["jianzi_text"], "")

    def test_compact_jianzi_rows_normalize_quoted_decimal_indices(self) -> None:
        patches = expand_jianzi_rows(
            [["245", "注下七徽九分"]], current_text={245: None}
        )
        self.assertEqual(patches[0]["source_index"], 245)
        with self.assertRaises(ValueError):
            expand_jianzi_rows([["245.0", "吟"]], current_text={245: None})



    def test_batch_pitch_candidates_deduplicate_equal_target_pitch(self) -> None:
        item = self._item()
        item["phrase_id"] = "p0001"
        runtime = RealToolRuntime(item, {})

        result = runtime.invoke("get_pitch_candidates", {
            "source_indices": [1, 2, 3], "max_candidates": 3,
        })

        self.assertTrue(result["ok"], result)
        queries = runtime.calls[-1]["result"]["result"]["queries"]
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["source_indices"], [1, 2, 3])
        self.assertIn("来源｜1、2、3｜简谱｜6̣",
                      result["result"]["text"])

    def test_grouped_pitch_header_keeps_sources_but_one_jianpu_symbol(self) -> None:
        text = render_grouped_candidate_table(
            77.0,
            [{"mode": "open", "string": 1, "hui": None, "sounding_midi": 77.0,
              "confidence": "exact"}],
            ("open",),
            [{"event_index": 119, "source_index": 119, "jianpu": "1̣"},
             {"event_index": 121, "source_index": 121, "jianpu": "1̇"}],
        )
        self.assertIn("来源｜119、121｜简谱｜1̣", text)
        self.assertNotIn("简谱｜1̣、1̇", text)

    def test_chord_pitch_header_uses_the_symbol_for_that_chord_tone(self) -> None:
        item = self._item()
        item["input"]["notes_without_jianzi"][:2] = [
            {"index": 1, "abc": "[F,f]2", "jianpu": "1̣", "jianpu_alt": "1̇"},
            {"index": 2, "abc": "[F,f]2", "jianpu": "1̣", "jianpu_alt": "1̇"},
        ]
        item["phrase_id"] = "p0001"
        runtime = RealToolRuntime(item, {})
        result = runtime.invoke("get_pitch_candidates", {
            "source_indices": [1, 2], "max_candidates": 3,
        })
        self.assertTrue(result["ok"], result)
        text = result["result"]["text"]
        self.assertIn("目标音高｜MIDI 77｜来源｜1、2｜简谱｜1̇", text)
        self.assertNotIn("目标音高｜MIDI 77｜来源｜1、2｜简谱｜1̣ 1̇", text)

    def test_batch_pitch_candidates_accept_type_array(self) -> None:
        item = self._item()
        item["phrase_id"] = "p0001"
        runtime = RealToolRuntime(item, {})

        result = runtime.invoke("get_pitch_candidates", {
            "source_indices": [1, 2], "类型": ["按音", "散音"],
        })

        self.assertTrue(result["ok"], result)
        self.assertIn("目标音高", result["result"]["text"])
        self.assertIn("实得MIDI｜可信度", result["result"]["text"])
        self.assertNotIn("｜误差｜", result["result"]["text"])
        self.assertNotIn(".000", result["result"]["text"])

    def test_batch_pitch_candidates_normalize_chinese_modes(self) -> None:
        item = self._item()
        item["phrase_id"] = "p0001"
        runtime = RealToolRuntime(item, {})

        result = runtime.invoke("get_pitch_candidates", {
            "source_indices": [1, 2],
            "modes": ["按音", "散音", "泛音"],
        })

        self.assertTrue(result["ok"], result)
        queries = runtime.calls[-1]["result"]["result"]["queries"]
        candidates = [candidate for query in queries
                      for candidate in query["candidates"]]
        self.assertTrue(candidates)
        self.assertTrue(all(candidate["mode"] in {
            "stopped", "open", "harmonic"
        } for candidate in candidates))
        self.assertTrue(all(candidate["hui"] is None
                            for candidate in candidates
                            if candidate["mode"] == "open"))

    def test_batch_pitch_candidates_skips_non_sounding_indices(self) -> None:
        item = self._item()
        item["phrase_id"] = "p0001"
        runtime = RealToolRuntime(item, {})

        result = runtime.invoke("get_pitch_candidates", {
            "source_indices": [1, 99, 2],
        })

        self.assertTrue(result["ok"], result)
        self.assertIn("跳过不可解析或非发音序号｜99", result["result"]["text"])

    def test_partial_edit_preview_does_not_fail_on_unsubmitted_rows(self) -> None:
        item = self._item()
        item["baseline_plan"]["actions"][1]["jianzi_text"] = None
        runtime = RealToolRuntime(item, {}, basic_fingering=True)
        result = runtime.invoke("edit_plan", {"jianzi_rows": [[1, "散挑7弦"]]})

        self.assertTrue(result["ok"])
        self.assertTrue(result["result"]["valid"], result["result"]["text"])
        self.assertNotIn("pending_jianzi_text", result["result"]["text"])
        self.assertNotIn("errors", result["result"])
        self.assertEqual(runtime.calls[-1]["result"]["result"].get("errors"), [])

    def test_mixed_edit_batch_applies_legal_rows_and_warns_about_structural_rows(self) -> None:
        item = self._item()
        runtime = RealToolRuntime(item, {}, basic_fingering=True)
        result = runtime.invoke("edit_plan", {
            "jianzi_rows": [[1, "散勾7弦"], [99, "不应应用"]],
        })

        self.assertTrue(result["ok"], result)
        self.assertTrue(result["result"]["valid"], result["result"]["text"])
        self.assertIn("ignored_noneditable_jianzi_rows", result["result"]["text"])
        self.assertIn('"event_indices":[99]', result["result"]["text"])
        replay = replay_patches(
            item["baseline_plan"], runtime.accumulated_patches, strict_before=False,
        )
        by_index = {action["source_index"]: action for action in replay.actions}
        self.assertEqual(by_index[1]["jianzi_text"], "散勾7弦")

    def test_final_quality_check_still_requires_complete_jianzi_text(self) -> None:
        item = self._item()
        item["baseline_plan"]["actions"][1]["jianzi_text"] = None
        valid, report = validate_jianzi_only(item, [], require_complete=True)
        self.assertFalse(valid)
        self.assertEqual(
            [problem["source_index"] for problem in report["problems"]
             if problem.get("code") == "pending_jianzi_text"],
            [2],
        )














    def test_final_replay_uses_last_write_for_corrective_batches(self) -> None:
        item = self._item()
        patches = [
            {"patch_id": "first", "patch_type": "CHANGE_FINGER",
             "source_index": 1, "before": {}, "after": {"right_finger": "勾"}},
            {"patch_id": "correction", "patch_type": "CHANGE_FINGER",
             "source_index": 1, "before": {}, "after": {"right_finger": "抹"}},
        ]
        replay = replay_patches(item["baseline_plan"], patches, strict_before=False)
        self.assertTrue(replay.valid)
        self.assertEqual(compare_replay_to_patch_targets(replay.actions, patches), [])

    def test_repeat_omission_marker_and_empty_text_are_equivalent_targets(self) -> None:
        actions = [{"source_index": 1, "jianzi_text": ""}]
        patches = [{"patch_type": "SET_JIANZI_TEXT", "source_index": 1,
                    "after": {"jianzi_text": "无（由于是再作部分，省略）"}}]
        self.assertEqual(compare_replay_to_patch_targets(actions, patches), [])


    def test_pre_attack_placement_is_replayed_and_audited(self) -> None:
        item = self._item()
        patch = [{"patch_id": "pre-chuo", "patch_type": "ADD_TECHNIQUE",
                  "source_index": 1, "before": {},
                  "after": {"technique": "绰", "placement": "pre_attack"}}]
        replay = replay_patches(item["baseline_plan"], patch, strict_before=False)
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(replay.actions[0]["pre_attack_techniques"], ["绰"])
        self.assertEqual(compare_replay_to_patch_targets(replay.actions, patch), [])

    def test_add_technique_can_materialize_a_complete_pre_attack_action(self) -> None:
        patch = [{"patch_id": "new-pre-chuo", "patch_type": "ADD_TECHNIQUE",
                  "source_index": 99, "before": {},
                  "after": {"technique": "绰", "placement": "pre_attack",
                            "attack": True, "mode": "stopped", "string": 6,
                            "hui": 5.6, "left_finger": "大指",
                            "right_finger": "挑",
                            "jianzi_text": "绰大指五徽六分挑6弦"}}]
        replay = replay_patches({"actions": []}, patch)
        self.assertTrue(replay.valid, replay.errors)
        self.assertEqual(render_jianzi_surface(replay.actions[0]),
                         "[绰大指五徽六分挑6弦]")
        self.assertEqual(compare_replay_to_patch_targets(replay.actions, patch), [])



if __name__ == "__main__":
    unittest.main()
