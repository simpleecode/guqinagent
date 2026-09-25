import unittest

from evaluation.fingering import field_counts, rates
from evaluation.ornament import ornament_metrics, technique_usage


class MetricsTests(unittest.TestCase):
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
