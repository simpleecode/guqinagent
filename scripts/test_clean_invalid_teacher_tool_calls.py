import json
import unittest
from copy import deepcopy
from scripts.clean_invalid_teacher_tool_calls import clean_messages


class CleanupTest(unittest.TestCase):
    def test_batched_third_error_preserves_valid_calls_and_receipts(self):
        calls = [{'id': str(i), 'function': {'name': name, 'arguments': '{}'}}
                 for i, name in enumerate(('expand_context', 'expand_context', 'bad'))]
        source = [{'role': 'assistant', 'content': 'reason', 'tool_calls': calls}]
        source += [{'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps(result)}
                   for i, result in enumerate(({'ok': False, 'error': 'context already shown'},
                                                {'ok': True, 'result': 'context'},
                                                {'ok': False, 'error': 'ValueError: unknown tool: bad'}))]
        original = deepcopy(source)
        cleaned, removed = clean_messages(source, {'expand_context'})
        self.assertEqual(removed, ['2'])
        self.assertEqual(cleaned[0]['tool_calls'], calls[:2])
        self.assertEqual(cleaned[1:], source[1:3])
        self.assertEqual(source, original)

    def test_no_unknown_receipt_no_removal(self):
        source = [{'role': 'assistant', 'tool_calls': [{'id': 'x', 'function': {'name': 'bad'}}]},
                  {'role': 'tool', 'tool_call_id': 'x', 'content': '{"ok": true}'}]
        self.assertEqual(clean_messages(source, set()), (source, []))


if __name__ == '__main__':
    unittest.main()
