import unittest
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from evaluation.run_eval import valid_prefix, run


class StrictPrefixTests(unittest.TestCase):
    def setUp(self):
        self.refs = [{'sample_id': str(i), 'runtime_item': {'input': {
            'notes_without_jianzi': [{'index': i}]}}} for i in range(4)]
        self.good = {'protocol_valid': True, 'direct': {'plan': {
            'actions': [{'source_index': 0, 'jianzi_text': '散勾一弦'}]}}}

    def test_failure_retains_prefix_not_suffix(self):
        prefix = valid_prefix({'0': self.good, '1': self.good, '2': {'protocol_valid': False}}, self.refs)
        self.assertEqual([r['sample_id'] for _, r in prefix], ['0', '1'])

    def test_missing_predecessor_cannot_be_reset(self):
        with self.assertRaises(ValueError):
            valid_prefix({'0': self.good, '2': self.good}, self.refs)

    def test_failure_cannot_be_followed_by_prediction(self):
        with self.assertRaises(ValueError):
            valid_prefix({'0': {'protocol_valid': False}, '1': self.good}, self.refs)

    def test_full_set_and_empty_prefix(self):
        self.assertEqual(len(valid_prefix({str(i): self.good for i in range(4)}, self.refs)), 4)
        self.assertEqual(valid_prefix({}, self.refs), [])

    def test_scoring_preserves_harmonic_history_and_failure_denominators(self):
        refs, predictions = [], []
        for i in range(4):
            text = '泛起大指七徽勾一弦' if i == 0 else '勾一弦'
            action = {'source_index': i, 'jianzi_text': text}
            item = {'score_key': 'T', 'phrase_id': f'p{i}', 'input': {
                'metadata': {'tonic': '1=C', 'tuning': {'open_strings': [
                    {'string': j+1, 'pitch': pitch, 'octave': octave}
                    for j, (pitch, octave) in enumerate([
                        ('C', 3), ('D', 3), ('F', 3), ('G', 3), ('A', 3), ('C', 4), ('D', 4)])]}},
                'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                'event_range': {'start': i, 'end_exclusive': i+1},
                'notes_without_jianzi': [{'index': i, 'event_index': i,
                                         'jianpu': '1', 'abc': 'C', 'duration': '四分'}]}}
            refs.append({'sample_id': str(i), 'score_key': 'T', 'runtime_item': item,
                         'reference': {'actions': [action]}})
            if i < 3:
                predictions.append({'sample_id': str(i), 'score_key': 'T', 'phrase_id': f'p{i}',
                    'protocol_valid': i < 2,
                    'direct': {'plan': {'actions': [action]} if i < 2 else None}})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, data in [('pred', predictions), ('refs', refs)]:
                (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in data))
            summary = run(Namespace(pred=directory/'pred', reference=directory/'refs',
                output_root=directory, experiment='eval', pitch_tolerance=50,
                model='test', strict_prefix=True))
            self.assertEqual(summary['completion']['attempted_failures'], 1)
            self.assertEqual(summary['completion']['missing_or_skipped'], 1)
            self.assertEqual(summary['completion']['valid_output_coverage'], .5)
            self.assertEqual(summary['metrics']['rule_warning_audited_phrases'], 2)
            scored = [json.loads(line) for line in (directory/'eval/per_event.jsonl').read_text().splitlines()]
            self.assertEqual(len(scored), 2)
            self.assertEqual(scored[1]['prediction']['tone_type'], 'harmonic')
            self.assertEqual(scored[1]['pitch_audit']['status'], 'matched')
