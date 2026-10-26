import unittest

from evaluation.fingering import field_counts, rates
from evaluation.ornament import ornament_metrics, technique_usage
from evaluation.techniques import extract_fingering_techniques
from evaluation.rules import replay_final_warnings
from evaluation.aggregate import final_warning_stats, metrics
from evaluation.parser import final_actions
from agents.abc_to_jianzipu.reference_parser import parse_reference_actions


class MetricsTests(unittest.TestCase):
    def test_single_stage_final_actions_are_evaluated(self):
        prediction = {'single_stage': {'plan': {'actions': [
            {'source_index': 7, 'jianzi_text': '散挑一弦'},
        ]}}}
        self.assertEqual(final_actions(prediction)[0]['jianzi_text'], '散挑一弦')

    def test_rule_metric_uses_final_plan_replay_not_intermediate_receipt(self):
        def sample(index, text, trace=None):
            item = {
                'score_key': 'T', 'phrase_id': f'p{index + 1:04d}',
                'input': {
                    'metadata': {'tonic': '1=C'},
                    'normalized_tuning': {'open_midi': [48, 50, 53, 55, 57, 60, 62]},
                    'event_range': {'start': index, 'end_exclusive': index + 1},
                    'notes_without_jianzi': [
                        {'index': index, 'event_index': index,
                         'jianpu': '1̣', 'abc': 'C,', 'duration': '四分'},
                    ],
                },
                'baseline_plan': {'actions': [{'source_index': index, 'jianzi_text': ''}]},
            }
            prediction = {
                'score_key': 'T', 'phrase_id': item['phrase_id'],
                'guqinizer': {'plan': {'actions': [
                    {'source_index': index, 'jianzi_text': text},
                ]}, 'trace': trace or []},
            }
            return prediction, {'runtime_item': item}

        stale_warning = [{'tool_results': [{'name': 'edit_plan', 'result': {
            'result': {'text': ':warning:音高不匹配'},
        }}]}]
        rows = [sample(0, '散勾三弦'), sample(1, '散挑一弦', stale_warning)]
        warnings = replay_final_warnings(rows)
        self.assertEqual([(row['phrase_id'], row['violation_type']) for row in warnings],
                         [('p0001', 'jianzi_pitch_mismatch')])
        stats = final_warning_stats(warnings, 2)
        self.assertEqual(stats['warning_count'], 1)
        self.assertEqual(stats['warning_phrases'], 1)
        self.assertEqual(stats['warning_rate'], 0.5)
        rule_metrics = metrics([], warnings, 50.0, audited_phrases=2)
        self.assertEqual(rule_metrics['rule_violations'], 1)
        self.assertEqual(rule_metrics['rule_violation_rate'], 0.5)
        incomplete = metrics([], [], 50.0, audited_phrases=0)
        self.assertIsNone(incomplete['rule_violation_rate'])
        self.assertEqual(incomplete['rule_warning_audited_phrases'], 0)

    def test_chinese_and_arabic_strings_have_identical_semantics(self):
        for chinese, arabic in (
            ('大指九徽挑五弦', '大指九徽挑5弦'),
            ('泛音大指七徽勾四弦', '泛音大指七徽勾4弦'),
            ('撮（大指七徽六弦按音＋一弦散音）', '撮（大指七徽6弦按音＋1弦散音）'),
        ):
            def parse(text):
                tuning = {'open_strings': [
                    {'pitch': pitch, 'octave': octave}
                    for pitch, octave in [('C', 2), ('D', 2), ('F', 2), ('G', 2), ('A', 2), ('C', 3), ('D', 3)]
                ]}
                return parse_reference_actions({'metadata': {'tuning': tuning}, 'notes': [{'index': 0, 'jianzi': text}]})[0]
            left, right = parse(chinese), parse(arabic)
            self.assertIsNotNone(left.string)
            for field in ('string', 'string2', 'mode', 'hui', 'left_finger', 'right_finger'):
                self.assertEqual(getattr(left, field), getattr(right, field), (chinese, field))

    def test_field_accuracy_and_f1(self):
        events = [
            {"reference": {"string": 5}, "prediction": {"string": 5}},
            {"reference": {"string": 6}, "prediction": {"string": 5}},
            {"reference": {"string": None}, "prediction": {"string": 7}},
        ]
        score = rates(field_counts(events, "string"))
        self.assertEqual(score["n"], 2)
        self.assertEqual(score["accuracy"], 0.5)
        self.assertAlmostEqual(score["f1"], 0.5)

    def test_ornaments_are_sets(self):
        events = [{"reference": {"ornaments": ["吟", "猱"]},
                   "prediction": {"ornaments": ["吟", "撞"]}}]
        score = ornament_metrics(events)
        self.assertAlmostEqual(score["precision"], 0.5)
        self.assertAlmostEqual(score["recall"], 0.5)

    def test_technique_usage_is_event_based(self):
        events = [
            {"reference": {"ornaments": ["吟"]}, "prediction": {"ornaments": ["吟", "吟"]}},
            {"reference": {"ornaments": []}, "prediction": {"ornaments": ["绰"]}},
        ]
        stats = technique_usage(events)["by_technique"]
        self.assertEqual(stats["吟"]["prediction_events"], 1)
        self.assertEqual(stats["吟"]["reference_events"], 1)
        self.assertAlmostEqual(stats["绰"]["rate_delta"], 0.5)
        self.assertNotIn("event_count_delta", stats["绰"])

    def test_technique_density_counts_events_not_mentions(self):
        events = [
            {"reference": {"ornaments": ["吟"]},
             "prediction": {"ornaments": ["吟", "吟"]}},
            {"reference": {"ornaments": ["猱"]},
             "prediction": {"ornaments": []}},
            {"reference": {"ornaments": []},
             "prediction": {"ornaments": ["绰", "上"]}},
            {"reference": {"ornaments": []}, "prediction": {"ornaments": []}},
        ]
        density = technique_usage(events)["technique_density"]
        self.assertEqual(density["prediction_events"], 2)
        self.assertEqual(density["reference_events"], 2)
        self.assertAlmostEqual(density["prediction_rate"], 0.5)
        self.assertAlmostEqual(density["reference_rate"], 0.5)

    def test_state_markers_are_not_performance_techniques(self):
        names = [item["name"] for item in extract_fingering_techniques(
            "泛起大指九徽挑五弦泛止撮"
        )]
        self.assertEqual(names, ["挑", "撮"])
