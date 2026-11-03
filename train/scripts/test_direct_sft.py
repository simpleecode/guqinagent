import json
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.abc_to_jianzipu.direct_generation import parse_answer, user_from_public_prompt
from evaluation.parser import final_actions
from train.scripts.build_direct_sft import committed_public_rows
from train.scripts.eval_direct import generate_http, run_score


def item(phrase='p0001', offset=0):
    return {'trajectory_id': 'S-' + phrase, 'score_key': 'S', 'phrase_id': phrase,
            'input': {'event_range': {'start': offset, 'end_exclusive': offset + 3},
                      'notes_without_jianzi': [
                          {'index': offset, 'event_index': 10 + offset},
                          {'index': offset + 1, 'event_index': None, 'abc': '|'},
                          {'index': offset + 2, 'event_index': 11 + offset}]}}


class DirectProtocolTests(unittest.TestCase):
    def test_http_payload_never_exposes_tools_or_requests_thinking(self):
        args = SimpleNamespace(model='guqin-direct', max_new_tokens=8192,
                               vllm_url='http://127.0.0.1:8999/v1', timeout=20)
        reply = io.BytesIO(json.dumps({'choices': [{'message': {'content': '{}'},
                                    'finish_reason': 'stop'}], 'usage': {'completion_tokens': 2}}).encode())
        with patch('urllib.request.urlopen', return_value=reply) as client:
            response = generate_http([{'role': 'user', 'content': 'score'}], args)
        payload = json.loads(client.call_args.args[0].data)
        self.assertNotIn('tools', payload)
        self.assertNotIn('tool_choice', payload)
        self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(response['usage']['completion_tokens'], 2)

    def test_event_numbers_map_to_source_with_empty_display(self):
        _, plan = parse_answer('{"jianzi_rows":[[10,"大指七徽勾一弦"],[11,""]]}', item())
        self.assertEqual(plan['actions'], [{'source_index': 0, 'jianzi_text': '大指七徽勾一弦'},
                                           {'source_index': 2, 'jianzi_text': ''}])
        self.assertEqual(final_actions({'direct': {'plan': plan}}), plan['actions'])

    def test_incomplete_duplicate_foreign_and_noninteger_rows_fail(self):
        for data in ([ [10, 'a'] ], [[10, 'a'], [10, 'b']], [[10, 'a'], [12, 'b']],
                     [[True, 'a'], [11, 'b']], [['10', 'a'], [11, 'b']]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_answer(json.dumps({'jianzi_rows': data}), item())

    def test_reasoning_or_tool_call_is_not_a_direct_answer(self):
        for text in ('<think>reason</think>{"jianzi_rows":[]}',
                     '{"jianzi_rows":[[10,"a"],[11,""]],"tool_calls":[]}', '```json\n{}\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_answer(text, item())

    def test_merge_all_successful_edits_ignore_failed_edits(self):
        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content':
            '【当前段 p0001｜待编辑】\n10｜1｜C｜四分｜[减字待填写]\n11｜0｜z｜四分｜[—]'}]
        for key, args, valid in [('one', {'jianzi_rows': [[10, 'first']]}, True),
                                 ('bad', {'unexpected': 1}, False),
                                 ('noop', {}, True),
                                 ('two', {'jianzi_rows': [[11, 'last']]}, True)]:
            messages.extend([{'role': 'assistant', 'tool_calls': [{'id': key, 'function': {
                'name': 'edit_plan', 'arguments': json.dumps(args)}}]},
                {'role': 'tool', 'tool_call_id': key, 'content': json.dumps({'ok': valid, 'result': {'valid': valid}})}])
        self.assertEqual(committed_public_rows(messages, item()), [[10, 'first'], [11, 'last']])

    def test_user_context_kept_without_tool_directions(self):
        text = '谱名｜example\n注｜先 list_context\n10｜1｜C｜四分｜[减字待填写]:warning:diagnosis'
        result = user_from_public_prompt(text)
        self.assertIn('10｜1｜C｜四分｜[减字待填写]', result)
        self.assertNotIn('list_context', result)
        self.assertNotIn(':warning:', result)
        self.assertNotIn('diagnosis', result)

    def test_failure_stops_suffix_and_previous_history_is_generated(self):
        sources = [{'sample_id': 'S-' + p, 'score_key': 'S', 'phrase_id': p,
                    'input_sha256': p, 'runtime_item': item(p, n * 3)}
                   for n, p in enumerate(('p0001', 'p0002', 'p0003'))]
        seen = []; emitted = []
        def prompt(runtime, previous, historical):
            seen.append((previous, dict(historical)))
            return []
        replies = iter([{'text': '{"jianzi_rows":[[10,"generated"],[11,""]]}'}, {'text': '{}'}])
        with patch('train.scripts.eval_direct.inference_messages', side_effect=prompt):
            run_score(sources, {}, lambda messages: next(replies), emitted.append)
        self.assertEqual(len(emitted), 2)
        self.assertEqual([r['protocol_valid'] for r in emitted], [True, False])
        self.assertEqual(seen[1][0]['reference_plan']['actions'][0]['jianzi_text'], 'generated')
        self.assertEqual(len(seen[1][1]), 1)
        # A failed saved attempt remains a failure on resume; never regenerate it.
        with patch('train.scripts.eval_direct.inference_messages', side_effect=AssertionError('must not regenerate')):
            run_score(sources, {r['sample_id']: r for r in emitted}, lambda _: None, lambda _: None)


if __name__ == '__main__':
    unittest.main()
