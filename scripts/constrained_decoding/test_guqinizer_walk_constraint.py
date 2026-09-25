#!/usr/bin/env python3
"""Tests for scripts/guqinizer_walk_constraint.py (no torch/GPU needed)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jianpu_jianzi_pitch import parse_hui, position_pitch
from scripts.constrained_decoding.guqinizer_walk_constraint import (
    ENDPOINT_CHARSET,
    ST_ROWS,
    ST_SPAN,
    ST_VALUE,
    WalkHuiConstraintProcessor,
    build_walk_constraints,
    own_position,
    render_hui,
)

REPO = ROOT
EVAL_INPUT = REPO / "train/eval_inputs_v2_text_protocol/SCf7VJzZ_public.jsonl"
EVAL_OUTPUT = (
    REPO / "train/eval_outputs_v3_two_stage"
    / "remote_a100_qwen35_9b_final_20260917/SCf7VJzZ.jsonl"
)


def load_scfc7_phrase():
    inputs = {
        row["phrase_id"]: row["runtime_item"]
        for row in (
            json.loads(line) for line in EVAL_INPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    outputs = [
        json.loads(line) for line in EVAL_OUTPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    item = json.loads(json.dumps(inputs["p0001"]))
    item["baseline_plan"] = json.loads(json.dumps(outputs[0]["base"]["plan"]))
    return item


def build_mock_processor(constraints, tokens):
    """tokens: iterable of surface strings; ids assigned in order."""
    id_texts = {index: text for index, text in enumerate(tokens)}
    return WalkHuiConstraintProcessor(id_texts, constraints, prompt_length=0)


def greedy_run(processor, pieces):
    """Feed pieces, and whenever the state machine is inside a span, assert
    every piece continuation is admissible (mirrors what the logits mask
    enforces during real decoding)."""
    visited = []
    for piece in pieces:
        if processor.state == ST_SPAN:
            visited.append((processor.span_prefix, piece))
        processor._feed_chars(piece)
    return visited


class RenderHuiTest(unittest.TestCase):
    def test_roundtrip(self):
        for text in ("七徽", "七徽三分", "七徽六分", "十徽", "十徽八分", "十三徽",
                     "一徽", "一徽九分", "五徽六分"):
            self.assertEqual(render_hui(parse_hui(text)), text, text)


class BuildConstraintsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item = load_scfc7_phrase()
        cls.table = build_walk_constraints(cls.item)

    def test_open_midi_and_tonic(self):
        source = self.item["input"]
        self.assertEqual(source["normalized_tuning"]["open_midi"],
                         [48, 50, 53, 55, 57, 60, 62])
        self.assertEqual(source["metadata"]["tonic"], "1=C")

    def test_event_two_note7(self):
        # Base row: 大指七徽三分挑六弦 (source 2, event 2, jianpu "7" -> MIDI 71).
        allowed = self.table.allowed_for(2, self.table.initial_live.get(2))
        self.assertIn("七徽三分", allowed)          # clause 1: same as Base
        self.assertNotIn("七徽六分", allowed)        # the observed wrong rewrite
        self.assertNotIn("七徽", allowed)            # ~100 cents off
        # Every non-Base entry is pitch-correct on some string within 50 cents.
        open_midi = self.item["input"]["normalized_tuning"]["open_midi"]
        for surface in allowed:
            if surface == "七徽三分":
                continue
            hui = parse_hui(surface)
            cents = min(
                abs(position_pitch(string, hui, open_midi, "stopped")[0] - 71) * 100
                for string in range(1, 8)
            )
            self.assertLessEqual(cents, 50.0, surface)

    def test_allowed_set_is_base_plus_live_strings_only(self):
        # The union covers the Base string and the live string, not all seven:
        # 弦1's 四徽一分 family must stay out for note 2.
        allowed = self.table.allowed_for(2, self.table.initial_live.get(2))
        self.assertNotIn("四徽一分", allowed)
        self.assertNotIn("五徽二分", allowed)

    def test_quarantined_endpoint_is_wrong_on_every_string(self):
        open_midi = self.item["input"]["normalized_tuning"]["open_midi"]
        hui = parse_hui("七徽六分")
        best = min(
            abs(position_pitch(string, hui, open_midi, "stopped")[0] - 71) * 100
            for string in range(1, 8)
        )
        self.assertGreater(best, 50.0)

    def test_own_position_resolution(self):
        base = {
            action["source_index"]: action.get("jianzi_text")
            for action in self.item["baseline_plan"]["actions"]
        }
        self.assertEqual(own_position(base[2]), (6, 7.3))
        self.assertEqual(own_position(base[1]), (6, 7.0))
        self.assertEqual(own_position(base[5]), (5, 7.0))  # 撮 stopped partner
        self.assertIsNone(own_position("散挑七弦"))
        self.assertIsNone(own_position(""))

    def test_outside_hui_labels_are_anchors(self):
        # 徽外/徽外半 carry no numeric hui but do anchor (=Base) endpoints.
        self.assertEqual(own_position("大指徽外挑四弦"), (4, "徽外"))
        self.assertEqual(own_position("名指徽外半勾二弦"), (2, "徽外半"))

    def test_no_fractional_hui_past_thirteen(self):
        # The guqin has 13 hui; 十三徽几分 does not exist as notation.
        for allowed in self.table.preview_allowed().values():
            for surface in allowed:
                hui = parse_hui(surface)
                if isinstance(hui, str):
                    self.assertIn(surface, ("徽外", "徽外半"))
                    continue
                self.assertIsNotNone(hui, surface)
                self.assertLessEqual(float(hui), 13.0, surface)

    def test_build_includes_outside_hui_label(self):
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": [
                    {"index": 1, "event_index": 1, "jianpu": "2"},
                    {"index": 2, "event_index": 2, "jianpu": "2"},
                ],
            },
            "baseline_plan": {"actions": [
                {"source_index": 1, "jianzi_text": "大指徽外挑六弦"},
                {"source_index": 2, "jianzi_text": "散挑六弦"},
            ]},
        }
        table = build_walk_constraints(item)
        # Clause 1 (same as Base) and clause 2 via the live string left by
        # note 1 (弦6 徽外 = 60+1.9 semitones = 61.9 vs target 62, ~10 cents).
        self.assertIn("徽外", table.allowed_for(1, None))
        self.assertIn("徽外", table.allowed_for(2, table.initial_live.get(2)))
        self.assertEqual(table.initial_live.get(2), 6)

    def test_constraints_require_event_index_for_non_bar_notes(self):
        """The public tool protocol has no source-index fallback."""
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": [
                    {"index": 42, "event_index": None, "jianpu": "2"},
                ],
            },
            "baseline_plan": {"actions": [
                {"source_index": 42, "jianzi_text": "名指九徽勾五弦"},
            ]},
        }
        with self.assertRaisesRegex(ValueError, "require event_index"):
            build_walk_constraints(item)

    def test_constraints_key_by_event_index_across_bar_lines(self):
        """A source-row gap from a bar must not shift decoder constraints."""
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": [
                    {"index": 0, "event_index": 0, "jianpu": "2", "abc": "D"},
                    {"index": 1, "event_index": None, "jianpu": "|", "abc": "|", "duration": "小节线"},
                    {"index": 2, "event_index": 1, "jianpu": "7", "abc": "B"},
                ],
            },
            "baseline_plan": {"actions": [
                {"source_index": 0, "jianzi_text": "名指九徽勾五弦"},
                {"source_index": 2, "jianzi_text": "大指七徽三分挑六弦"},
            ]},
        }
        table = build_walk_constraints(item)
        self.assertIn("七徽三分", table.allowed_for(1, table.initial_live.get(1)))
        self.assertNotIn("九徽", table.allowed_for(1, table.initial_live.get(1)))

    def test_in_call_rows_update_the_live_string(self):
        # Rows the model has already completed inside the SAME tool call are
        # replayed before later rows open, so a walk sees the string the
        # Guqinizer itself just established.
        table = build_walk_constraints(self.item)
        vocab = MockVocab()
        processor = WalkHuiConstraintProcessor(vocab.id_texts, table, 0)
        processor._feed_chars('<parameter=jianzi_rows>[[1, "名指九徽勾五弦"], ')
        self.assertEqual(processor.stats["rows_replayed"], 1)
        self.assertEqual(processor._live_string(), None)  # row 2 not open yet
        processor._feed_chars('[2, "')
        # Row 1 left the left hand on 弦5, so the live string for event 2 is 5
        # even though the Base row for note 2 is on 弦6.
        self.assertEqual(processor._live_string(), 5)
        allowed = processor._row_allowed
        self.assertIn("七徽三分", allowed)  # static anchor from Base
        # 弦5 pitch-correct surfaces for target 71 join via the live string.
        self.assertIn("六徽四分", allowed)
        self.assertNotIn("四徽一分", allowed)  # other strings stay out

    def test_chuo_and_zhu_endpoints_must_move_from_previous_hui(self):
        item = {
            "input": {
                "metadata": {"tonic": "1=C"},
                "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                "notes_without_jianzi": [
                    {"index": 1, "event_index": 1, "jianpu": "2"},
                    {"index": 2, "event_index": 2, "jianpu": "2"},
                    {"index": 3, "event_index": 3, "jianpu": "2"},
                ],
            },
            "baseline_plan": {"actions": [
                {"source_index": 1, "jianzi_text": "名指十徽勾三弦"},
                {"source_index": 2, "jianzi_text": "名指十徽八分勾三弦"},
                {"source_index": 3, "jianzi_text": "名指十徽八分勾三弦"},
            ]},
        }
        processor = WalkHuiConstraintProcessor(MockVocab().id_texts,
                                               build_walk_constraints(item), 0)
        # Ten hui is the inherited position before event 2, hence it cannot
        # be the endpoint.  Ten hui eight fen is a real motion and remains.
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下')
        self.assertNotIn("十徽", processor._pending_allowed)
        self.assertIn("十徽八分", processor._pending_allowed)
        processor._feed_chars('十徽八分"], [3, "绰上')
        # The first in-call walk reached 10.8. A second 注下/绰上 to 10.8 is
        # now prohibited even though Base's own endpoint is also 10.8.
        self.assertNotIn("十徽八分", processor._pending_allowed)

    def test_same_endpoint_head_is_masked_before_it_is_emitted(self):
        item = {
            "input": {"metadata": {"tonic": "1=C"},
                      "normalized_tuning": {"open_midi": [48, 50, 53, 55, 57, 60, 62]},
                      "notes_without_jianzi": [
                          {"index": 1, "event_index": 1, "jianpu": None},
                          {"index": 2, "event_index": 2, "jianpu": None},
                      ]},
            "baseline_plan": {"actions": [
                {"source_index": 1, "jianzi_text": "名指十徽勾三弦"},
                {"source_index": 2, "jianzi_text": "名指十徽勾三弦"},
            ]},
        }
        processor = WalkHuiConstraintProcessor(MockVocab().id_texts,
                                               build_walk_constraints(item), 0)
        processor._feed_chars('<parameter=jianzi_rows>[[2, "')
        allowed = processor._allowed_ids_preventing_noop_walk_head()
        self.assertIsNotNone(allowed)
        # The no-op heads themselves (including a merged token) are blocked
        # before the model can emit an incomplete walk and enter a retry loop.
        for token_id, text in processor.id_texts.items():
            if text == "注下":
                self.assertNotIn(token_id, allowed)
            if text == "名指":
                self.assertIn(token_id, allowed)
        processor._feed_chars("绰")
        allowed = processor._allowed_ids_preventing_noop_walk_head()
        self.assertIsNotNone(allowed)
        for token_id, text in processor.id_texts.items():
            if text == "上":
                self.assertNotIn(token_id, allowed)

    def test_walk_head_is_masked_when_row_has_no_reliable_endpoint(self):
        """An unconstrained row must not admit an unverifiable walk."""
        processor = WalkHuiConstraintProcessor(MockVocab().id_texts, {}, 0)
        processor._feed_chars('<parameter=jianzi_rows>[[2, "')
        self.assertFalse(processor._row_allowed)
        allowed = processor._allowed_ids_preventing_noop_walk_head()
        self.assertIsNotNone(allowed)
        for token_id, text in processor.id_texts.items():
            if text == "注下":
                self.assertNotIn(token_id, allowed)
            if text == "名指":
                self.assertIn(token_id, allowed)
        # A separately tokenized 绰上 is also stopped before it completes.
        processor._feed_chars("绰")
        allowed = processor._allowed_ids_preventing_noop_walk_head()
        self.assertIsNotNone(allowed)
        for token_id, text in processor.id_texts.items():
            if text == "上":
                self.assertNotIn(token_id, allowed)


# --- state machine -----------------------------------------------------------

GUIQINIZER_OUTPUT_STYLE = (
    "逐音核对：第2音7改为注下七徽六分，承接同弦余音下滑。\n"
    "</think>\n"
    "<tool_call>\n"
    "<function=edit_plan>\n"
    "<parameter=jianzi_rows>\n"
    "[[2, {row2}], [6, \"历五弦\"], [22, \"绰大指七徽{fen}挑六弦\"]]\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
)

CHAR_PIECES = ["逐", "音", "核", "对", "：", "第", "2", "音", "7", "改", "为",
               "注", "下", "七", "徽", "六", "分", "，", "承", "接"]


class MockVocab:
    """Small id->text vocabulary with realistic multi-character merges."""

    TOKENS = [
        "逐音核对", "：", "第", "2", "音", "7", "改", "为", "注下", "注", "下",
        "七", "七徽", "徽", "三分", "六分", "徽三分", "徽六分", "分", "，",
        "承接", "同弦余音下滑", "。", "\n", "</think>", "<tool_call>",
        "<function=edit_plan>", "<parameter=jianzi_rows>", "[[", "]", ", ",
        "[", "]", '"', "历五弦", "大指", "挑", "六弦", "三分挑", "弦",
        "泛起", "勾", "五弦", "绰", "进复", "退复", "名指", "中指", "跪指",
        "十", "九", "八", "徽外", "挑六弦", " ", "</parameter>", "</function>",
        "</tool_call>", "<|im_end|>", " seven", "seven", "eight", "ten", "9", "7",
    ]

    def __init__(self):
        self.id_texts = {index: text for index, text in enumerate(self.TOKENS)}

    def tokenize(self, text):
        """Greedy longest-match mock tokenizer over TOKENS."""
        pieces = []
        cursor = 0
        while cursor < len(text):
            for length in range(min(8, len(text) - cursor), 0, -1):
                piece = text[cursor:cursor + length]
                if piece in self.TOKENS:
                    pieces.append(piece)
                    cursor += length
                    break
            else:
                pieces.append(text[cursor])
                cursor += 1
        return pieces


ALLOWED_EVENT_2 = frozenset({"七徽", "七徽三分", "七徽二分", "七徽四分"})
CONSTRAINTS = {2: ALLOWED_EVENT_2, 22: ALLOWED_EVENT_2}


class StateMachineTest(unittest.TestCase):
    def build(self, constraints=CONSTRAINTS):
        vocab = MockVocab()
        return (
            WalkHuiConstraintProcessor(vocab.id_texts, constraints, prompt_length=0),
            vocab,
        )

    def test_reasoning_is_never_constrained(self):
        processor, _ = self.build()
        # The reasoning section literally contains the forbidden endpoint; it
        # must not activate the span because no jianzi_rows region is open.
        processor._feed_chars("理由：注下七徽六分更佳。</think>\n")
        self.assertNotEqual(processor.state, ST_SPAN)
        self.assertEqual(processor.stats["spans_entered"], 0)

    def test_span_blocks_forbidden_endpoint_and_allows_correct_one(self):
        processor, vocab = self.build()
        text = GUIQINIZER_OUTPUT_STYLE.format(row2='"注下七徽六分"', fen="六分")
        for piece in vocab.tokenize(text):
            if processor.state == ST_SPAN:
                admissible = processor._token_admissible(processor.span_prefix, piece)
                if piece == "徽六分":
                    self.assertFalse(admissible, piece)
                if piece in ("徽三分", "三分", "徽"):
                    self.assertTrue(admissible, piece)
            processor._feed_chars(piece)
        # The forbidden surfaces were fed unmasked on purpose (both the
        # standalone 注下 walk and the pre-attack 绰 row); the state machine
        # records each.  During real decoding the masks (asserted above and
        # in the pending tests) make such tokens unpickable.
        self.assertEqual(processor.stats["mid_token_span_violations"], 2)
        self.assertEqual(processor.stats["spans_entered"], 2)

    def test_pending_mask_blocks_wrong_hui_integer(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下')
        self.assertIsNotNone(processor._pending_allowed)
        # The hui integer itself is constrained: no allowed surface starts
        # with 八, and a merged trigger+numeral token cannot smuggle it in.
        self.assertFalse(processor._token_admissible_pending("八", ALLOWED_EVENT_2, head_ambiguous=False))
        self.assertFalse(processor._token_admissible_pending("下八", ALLOWED_EVENT_2, head_ambiguous=False))
        self.assertTrue(processor._token_admissible_pending("七", ALLOWED_EVENT_2, head_ambiguous=False))
        self.assertTrue(processor._token_admissible_pending("下七", ALLOWED_EVENT_2, head_ambiguous=False))
        # 注下 is an unambiguous head: there is no abort any more — the model
        # must open an allowed endpoint (ASCII/。 escapes are all blocked).
        self.assertFalse(processor._token_admissible_pending("。", ALLOWED_EVENT_2, head_ambiguous=False))
        # 徽-initial tokens need an allowed 徽外-family label; ALLOWED_EVENT_2
        # has none, so both a 徽外 attempt and the malformed 徽三分 are banned.
        self.assertFalse(processor._token_admissible_pending("徽外", ALLOWED_EVENT_2, head_ambiguous=False))
        self.assertFalse(processor._token_admissible_pending("徽三分", ALLOWED_EVENT_2, head_ambiguous=False))
        allowed_ids = processor._allowed_ids_for_pending()
        self.assertIsNotNone(allowed_ids)
        for token_id, text in processor.id_texts.items():
            if text == "八":
                self.assertNotIn(token_id, allowed_ids)

    def test_walking_to_outside_hui_is_constrained(self):
        allowed = frozenset({"徽外", "七徽三分"})
        processor, _ = self.build(constraints={2: allowed})
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下')
        self.assertIsNotNone(processor._pending_allowed)
        self.assertTrue(processor._token_admissible_pending("徽外", allowed, head_ambiguous=False))
        self.assertFalse(processor._token_admissible_pending("徽外半", allowed, head_ambiguous=False))
        self.assertTrue(processor._token_admissible_pending("七", allowed, head_ambiguous=False))
        processor._feed_chars('徽外"]')
        self.assertEqual(processor.stats["spans_entered"], 1)
        self.assertEqual(processor.stats["spans_completed"], 1)
        self.assertEqual(processor.state, ST_ROWS)

    def test_greedy_constrained_generation_completes_row(self):
        processor, vocab = self.build()
        text = GUIQINIZER_OUTPUT_STYLE.format(row2='"注下七徽三分"', fen="三分")
        for piece in vocab.tokenize(text):
            if processor.state == ST_SPAN:
                self.assertTrue(
                    processor._token_admissible(processor.span_prefix, piece),
                    (processor.span_prefix, piece),
                )
            elif processor._pending_allowed is not None:
                self.assertTrue(
                    processor._token_admissible_pending(piece, processor._pending_allowed, head_ambiguous=False),
                    piece,
                )
            processor._feed_chars(piece)
        self.assertEqual(processor.stats["spans_entered"], 2)
        self.assertEqual(processor.stats["spans_completed"], 2)
        self.assertEqual(processor.stats["mid_token_span_violations"], 0)
        # After the second span the model keeps writing freely.
        self.assertNotEqual(processor.state, ST_SPAN)
        self.assertIsNone(processor._pending_allowed)

    def test_merged_boundary_token_is_admissible(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下七徽')
        self.assertEqual(processor.state, ST_SPAN)
        self.assertEqual(processor.span_prefix, "七徽")
        # "三分挑" merges the allowed fen with the following right-hand verb.
        self.assertTrue(processor._token_admissible("七徽", "三分挑"))
        processor._feed_chars('三分挑六弦"]')
        # The value closed; the scanner sits between rows again.
        self.assertEqual(processor.state, ST_ROWS)
        self.assertEqual(processor.stats["mid_token_span_violations"], 0)

    def test_unconstrained_row_and_base_positions(self):
        processor, vocab = self.build()
        rows = '[[6, "历五弦"], [3, "名指九徽勾五弦"], [9, "泛起大指九徽挑5弦"]]'
        text = f"<parameter=jianzi_rows>{rows}</parameter>"
        for piece in vocab.tokenize(text):
            processor._feed_chars(piece)
        self.assertEqual(processor.stats["spans_entered"], 0)

    def test_pre_attack_chuo_with_finger_triggers_span(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "绰大指七')
        self.assertEqual(processor.state, ST_SPAN)
        self.assertEqual(processor.span_prefix, "七")
        self.assertTrue(processor._token_admissible("七", "徽三分"))
        self.assertFalse(processor._token_admissible("七", "徽六分"))

    def test_json_envelope_form(self):
        processor, _ = self.build()
        processor._feed_chars(
            '{"decision_summary": "x", "tool_calls": [{"name": "edit_plan", '
            '"arguments": {"jianzi_rows": [[2, "注下七'
        )
        self.assertEqual(processor.state, ST_SPAN)
        self.assertFalse(processor._token_admissible("七", "徽六分"))
        self.assertTrue(processor._token_admissible("七", "徽三分"))

    def test_row_without_constraints_stays_free(self):
        processor, _ = self.build(constraints={99: ALLOWED_EVENT_2})
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下七徽六分"]]')
        self.assertEqual(processor.stats["spans_entered"], 0)
        self.assertEqual(processor.stats["unconstrained_rows"], 1)

    def test_mask_allows_only_admissible_ids(self):
        processor, vocab = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下七')
        allowed = processor._allowed_ids_for_prefix("七")
        self.assertIsNotNone(allowed)
        for token_id, text in vocab.id_texts.items():
            self.assertEqual(
                token_id in allowed,
                processor._token_admissible("七", text),
                text,
            )

    def test_safety_valve_on_dead_prefix(self):
        # A span whose allowed set cannot extend "七" at all must not be
        # maskable; the processor falls back to free decoding.
        processor, _ = self.build(constraints={2: frozenset({"十徽"})})
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下七')
        self.assertEqual(processor.state, ST_SPAN)
        self.assertIsNone(processor._allowed_ids_for_prefix("七"))



class AsciiEscapeRegressionTest(unittest.TestCase):
    """S2T7rDyJ 实测：模型把被禁徽位改写成英文数字（绰上 eight / 绰上 seven nine）。"""

    def build(self, allowed=ALLOWED_EVENT_2):
        vocab = MockVocab()
        return (
            WalkHuiConstraintProcessor(vocab.id_texts, {2: allowed}, 0),
            vocab,
        )

    def test_unambiguous_head_must_open_allowed_endpoint(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "绰上')
        self.assertIsNotNone(processor._pending_allowed)
        # 英文/阿拉伯数字/空格开头 token 一律禁止——包括带前导空格的 " seven"
        for piece in (" seven", "seven", "eight", "ten", "7", "9"):
            self.assertFalse(
                processor._token_admissible_pending(
                    piece, ALLOWED_EVENT_2, head_ambiguous=False),
                piece)
        # 无歧义头不允许中止（引号也不行）：必须写终点
        self.assertFalse(processor._token_admissible_pending(
            '"', ALLOWED_EVENT_2, head_ambiguous=False))
        # 允许集起点数字仍可写
        self.assertTrue(processor._token_admissible_pending(
            "七", ALLOWED_EVENT_2, head_ambiguous=False))

    def test_ambiguous_head_abort_still_blocks_ascii(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "挑四弦上')
        for piece in (" seven", "ten", "8"):
            self.assertFalse(
                processor._token_admissible_pending(
                    piece, ALLOWED_EVENT_2, head_ambiguous=True), piece)
        # 中文/结构符中止仍允许（false-positive 触发需要退路）
        self.assertTrue(processor._token_admissible_pending(
            "。", ALLOWED_EVENT_2, head_ambiguous=True))
        self.assertTrue(processor._token_admissible_pending(
            '"', ALLOWED_EVENT_2, head_ambiguous=True))

    def test_span_exit_blocks_ascii_escape(self):
        processor, _ = self.build()
        processor._feed_chars('<parameter=jianzi_rows>[[2, "注下七徽')
        self.assertEqual(processor.state, ST_SPAN)
        # 完整终点后的自由退出不允许 ASCII 逃逸字符开头
        processor._feed_chars('三分')
        for piece in (" seven", "9", "7"):
            self.assertFalse(processor._token_admissible(processor.span_prefix, piece), piece)
        # 引号/中文后续正常退出
        self.assertTrue(processor._token_admissible(processor.span_prefix, '"'))
        self.assertTrue(processor._token_admissible(processor.span_prefix, "挑"))

    def test_blocked_integer_forces_allowed_surface_end_to_end(self):
        # 允许集不含八：模型想写八时，八被禁、英文被禁，唯一出路是允许集终点
        allowed = frozenset({"九徽三分"})
        processor, _ = self.build(allowed=allowed)
        text = '<parameter=jianzi_rows>[[2, "绰上'
        for ch in text:
            if processor._pending_allowed is not None:
                break
            processor._feed_chars(ch)
        self.assertFalse(processor._token_admissible_pending("八", allowed, False))
        self.assertFalse(processor._token_admissible_pending("eight", allowed, False))
        self.assertTrue(processor._token_admissible_pending("九", allowed, False))
        # 模拟 greedy：只能选九，随后正常写完终点
        processor._feed_chars('九徽三分"]')
        self.assertEqual(processor.stats["spans_completed"], 1)
        self.assertEqual(processor.stats["mid_token_span_violations"], 0)


if __name__ == "__main__":
    unittest.main()
