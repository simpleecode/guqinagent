from __future__ import annotations

import unittest
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ABC_J.scripts.redact_teacher_reasoning import (
    BANNED_PUBLIC_TERMS,
    normalize_summary,
    parse_json_object,
    private_reference_texts,
    factual_tokens,
    rewrite_prompt,
    validate_rewritten_summary,
    turns_to_redact,
    validate_summary_batch,
    apply_summary_replacements,
    rewrite_trajectory,
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

    def test_rewrite_prompt_batches_indexed_summaries_with_context(self) -> None:
        prompt = rewrite_prompt({"agent_stage": "guqinization"}, [{"original_summary": "原文"}])
        self.assertIn("summary", prompt)
        self.assertIn("不要按音符拆成多条", prompt)
        self.assertIn('public_conversation', json.loads(prompt))
        self.assertIn('message_index', prompt)


class TrajectoryBatchRedactionTests(unittest.TestCase):
    def setUp(self):
        self.item = {'sample_id': 'score-p0001-stage', 'agent_stage': 'guqinization',
                     'tools': [{'name': 'edit_plan'}], 'messages': [
                         {'role': 'system', 'content': '公开规则'},
                         {'role': 'user', 'content': '当前谱面：三弦七徽'},
                         {'role': 'assistant', 'content': '根据参考，选择三弦七徽',
                          'tool_calls': [{'id': 'a', 'function': {'name': 'edit_plan',
                                         'arguments': '{"jianzi_rows":[[1,"勾三弦"]]}'}}]},
                         {'role': 'tool', 'tool_call_id': 'a', 'content': '{"text":"已设置四弦九徽"}'},
                         {'role': 'assistant', 'content': '这轮保留原样'},
                         {'role': 'assistant', 'content': '参考建议改为四弦九徽'},
                         {'role': 'assistant', 'content': None},
                     ]}
        self.payload = {'sample_id': self.item['sample_id'], 'summaries': [
            {'message_index': 5, 'summary': '改为四弦九徽'},
            {'message_index': 2, 'summary': '选择三弦七徽'},
        ]}

    def fake_client(self, responses):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            response = responses[len(calls)-1]
            return SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(response, ensure_ascii=False))])
        return SimpleNamespace(messages=SimpleNamespace(create=create)), calls

    def test_only_marked_assistant_indices_selected(self):
        self.assertEqual([t['message_index'] for t in turns_to_redact(self.item)], [2, 5])

    def test_one_request_replaces_by_index_even_if_response_reordered(self):
        original = deepcopy(self.item)
        client, calls = self.fake_client([self.payload])
        stats = {'api_calls': 0}
        result, audit = rewrite_trajectory(self.item, {}, client, 'glm-5.3', 3, stats)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats['api_calls'], 1)
        self.assertEqual([r['message_index'] for r in audit], [2, 5])
        self.assertEqual(result['messages'][2]['content'], '选择三弦七徽')
        self.assertEqual(result['messages'][5]['content'], '改为四弦九徽')
        result['messages'][2]['content'] = original['messages'][2]['content']
        result['messages'][5]['content'] = original['messages'][5]['content']
        self.assertEqual(result, original)
        self.assertEqual(self.item, original)
        sent = json.loads(calls[0]['messages'][0]['content'])
        self.assertEqual(len(sent['public_conversation']), len(original['messages']))

    def test_malformed_batch_fails_without_partial_changes(self):
        variants = []
        for indices in ([2], [2, 2], [2, 5, 4], ['2', 5], [True, 5]):
            p = deepcopy(self.payload)
            p['summaries'] = [{'message_index': i, 'summary': '选择三弦七徽'} for i in indices]
            variants.append(p)
        wrong_id = deepcopy(self.payload); wrong_id['sample_id'] = 'different'
        variants.append(wrong_id)
        bad_text = deepcopy(self.payload); bad_text['summaries'][0]['summary'] = None
        variants.append(bad_text)
        original = deepcopy(self.item)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_summary_batch(payload, self.item, turns_to_redact(self.item), {})
            self.assertEqual(self.item, original)

    def test_future_tool_fact_not_allowed_for_earlier_turn(self):
        payload = deepcopy(self.payload)
        payload['summaries'][1]['summary'] = '选择四弦九徽'
        with self.assertRaises(ValueError):
            validate_summary_batch(payload, self.item, turns_to_redact(self.item), {})

    def test_clean_trajectory_uses_no_model_calls(self):
        item = deepcopy(self.item)
        item['messages'][2]['content'] = '选择三弦七徽'
        item['messages'][5]['content'] = '改为四弦九徽'
        client, calls = self.fake_client([])
        result, audit = rewrite_trajectory(item, {}, client, 'glm-5.3', 3, {'api_calls': 0})
        self.assertEqual(result, item)
        self.assertEqual(audit, [])
        self.assertEqual(calls, [])

    def test_retry_is_whole_batch_not_each_turn(self):
        client, calls = self.fake_client([{'sample_id': 'wrong'}, self.payload])
        _, audit = rewrite_trajectory(self.item, {}, client, 'glm-5.3', 3, {'api_calls': 0})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(audit), 2)
        self.assertIn('retry_note', json.loads(calls[1]['messages'][0]['content']))

    def test_exhausted_invalid_output_fails_closed(self):
        client, calls = self.fake_client([{}, {}])
        with self.assertRaises(ValueError):
            rewrite_trajectory(self.item, {}, client, 'glm-5.3', 2, {'api_calls': 0})
        self.assertEqual(len(calls), 2)

    def test_non_assistant_replacement_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_summary_replacements(self.item, {3: '不能修改工具结果'})

    def test_cli_writes_exact_replacements_and_resume_skips_model(self):
        from ABC_J.scripts import redact_teacher_reasoning as module
        client, calls = self.fake_client([self.payload])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'source'; output = Path(folder)/'output'
            source.mkdir()
            private = {'sample_id': self.item['sample_id'],
                       'teacher_private': {'annotation_gqs': '不得发送给改写模型'}}
            for name, row in [('messages_train.jsonl', self.item),
                              ('teacher_trajectory_audit.jsonl', private)]:
                (source/name).write_text(json.dumps(row,ensure_ascii=False)+'\n')
            argv = ['redact_teacher_reasoning.py', '--input-dir', str(source),
                    '--output-dir', str(output), '--model', 'glm-5.3']
            with patch.object(module, 'make_client', return_value=client), patch('sys.argv', argv):
                self.assertEqual(module.main(), 0)
                self.assertEqual(module.main(), 0)
            self.assertEqual(len(calls), 1)
            result = json.loads((output/'messages_train.jsonl').read_text())
            self.assertEqual(result['reasoning_redaction']['rewritten_message_indices'], [2, 5])
            for index, before in enumerate(self.item['messages']):
                after = result['messages'][index]
                if index in (2, 5):
                    after = dict(after, content=before['content'])
                self.assertEqual(before, after)
            self.assertEqual(json.loads((output/'teacher_trajectory_audit.jsonl').read_text()), private)
            self.assertNotIn('不得发送给改写模型', str(calls))
            self.assertEqual(len((output/'reasoning_redaction_audit.jsonl').read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
