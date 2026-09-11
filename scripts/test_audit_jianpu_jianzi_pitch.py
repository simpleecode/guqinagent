import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_jianpu_jianzi_pitch.py")
SPEC = importlib.util.spec_from_file_location("audit_pitch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditPitchTests(unittest.TestCase):
    def setUp(self):
        self.opens = [48, 50, 53, 55, 57, 60, 62]

    def test_single_stopped_note(self):
        pitches, reason = MODULE.parse_jianzi("名指九徽勾5弦", self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], 64.01955, places=4)

    def test_compound_has_two_unordered_pitches(self):
        text = "撮（大指七徽九分6弦按音＋3弦散音）"
        pitches, reason = MODULE.parse_jianzi(text, self.opens)
        self.assertIsNone(reason)
        self.assertEqual(len(pitches), 2)
        pairs = MODULE.best_pairing([53, 69], pitches)
        self.assertEqual([p["jianpu_name"] for p in pairs], ["F3", "A4"])
        self.assertTrue(all(p["absolute_cents"] < 50 for p in pairs))

    def test_slide_without_inherited_string_is_skipped(self):
        pitches, reason = MODULE.parse_jianzi(
            "注下七徽九分", self.opens, MODULE.new_context()
        )
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "context_dependent_slide")

    def test_zhi_string_transition_is_skipped(self):
        pitches, reason = MODULE.parse_jianzi("至5弦", self.opens)
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "context_dependent_transition")

    def test_open_string(self):
        pitches, reason = MODULE.parse_jianzi("勾4弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [55])

    def test_open_string_accepts_chinese_string_number(self):
        pitches, reason = MODULE.parse_jianzi("散托七弦", self.opens)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62])

    def test_zhuaqi_releases_previous_thumb_stopped_string_to_open_pitch(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("爪起", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [57])
        self.assertIsNone(context["active_left_string"])
        self.assertIsNone(context["active_left_hui"])

    def test_zhuaqi_can_sound_with_another_open_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾5弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "散挑7弦爪起", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(pitches, [62, 57])

    def test_zhuaqi_requires_previous_thumb_stopped_note(self):
        context = MODULE.new_context()
        pitches, reason = MODULE.parse_jianzi("爪起", self.opens, context)
        self.assertEqual(pitches, [])
        self.assertEqual(
            reason, "zhuaqi_requires_previous_thumb_stopped_note"
        )

    def test_li_compact_surface_emits_ordered_open_strings(self):
        for text in ("历四三", "历四、三弦", "厉四三"):
            pitches, reason = MODULE.parse_jianzi(text, self.opens)
            self.assertIsNone(reason)
            self.assertEqual(pitches, [55, 53])

    def test_li_spans_two_consecutive_score_events(self):
        report = MODULE.audit({
            "metadata": {"tonic": "1=F", "tonic_degree1_midi": 65},
            "open_midi": self.opens,
            "notes": [
                {"index": 38, "jianpu": "2̣", "jianzi": "历四三"},
                {"index": 39, "jianpu": "1̣", "jianzi": ""},
            ],
        }, 50.0)
        self.assertEqual(report["details"][0]["status"], "matched")
        self.assertEqual(report["details"][0]["covered_indices"], [38, 39])

    def test_prefix_flat_tonic_is_accepted(self):
        self.assertEqual(
            MODULE.parse_tonic_midi({"tonic": "1=bE"}),
            MODULE.parse_tonic_midi({"tonic": "1=Eb"}),
        )

    def test_explicit_b_flat_uses_ordinary_register(self):
        # Only the App's legacy bare ``1=B`` label is pinned to Bb3.  An
        # explicit flat spelling is ordinary Bb4 unless score metadata carries
        # a capture-specific tonic_degree1_midi override.
        self.assertEqual(MODULE.parse_tonic_midi({"tonic": "1=bB"}), 70)
        self.assertEqual(MODULE.parse_tonic_midi({"tonic": "1=Bb"}), 70)

    def test_harmonic_mode_keeps_hui(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        first, reason = MODULE.parse_jianzi(
            "泛起名指十二徽挑5弦", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertEqual(first, [88])
        second, reason = MODULE.parse_jianzi("勾4弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(second, [86])

    def test_stopped_position_is_inherited_on_same_string(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        MODULE.parse_jianzi("大指六徽四分挑7弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], 76.0, delta=0.5)

    def test_stopped_position_is_inherited_on_another_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指七徽挑4弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("勾5弦", self.opens, context)
        expected, _ = MODULE.position_pitch(5, 7, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], expected, places=4)

    def test_slide_emits_endpoint_pitch_and_updates_context(self):
        context = {
            "harmonic": False, "harmonic_hui": None,
            "stopped_string": None, "stopped_hui": None,
        }
        MODULE.parse_jianzi("大指六徽四分挑7弦", self.opens, context)
        slide, reason = MODULE.parse_jianzi("注下七徽", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(slide, [74])
        pitches, reason = MODULE.parse_jianzi("挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [74])

    def test_plain_movement_updates_position_for_jiu(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
        slide, reason = MODULE.parse_jianzi("上七徽六分", self.opens, context)
        self.assertIsNone(reason)
        expected_slide, _ = MODULE.position_pitch(4, 7.6, self.opens)
        self.assertAlmostEqual(slide[0], expected_slide)
        pitches, reason = MODULE.parse_jianzi("就挑5弦", self.opens, context)
        self.assertIsNone(reason)
        expected, _ = MODULE.position_pitch(5, 7.6, self.opens)
        self.assertAlmostEqual(pitches[0], expected)

    def test_hu_slide_emits_endpoint_on_inherited_string(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
        pitches, reason = MODULE.parse_jianzi(
            "浒上七徽九分", self.opens, context
        )
        expected, _ = MODULE.position_pitch(4, 7.9, self.opens)
        self.assertIsNone(reason)
        self.assertAlmostEqual(pitches[0], expected)
        self.assertEqual(context["active_left_string"], 4)
        self.assertEqual(context["active_left_hui"], 7.9)

    def test_other_explicit_slide_types_emit_their_endpoint(self):
        for text, hui in (
            ("绰上五徽六分", 5.6),
            ("注下七徽九分", 7.9),
            ("淌九徽", 9.0),
            ("引上七徽六分", 7.6),
            ("进复七徽", 7.0),
            ("退复九徽", 9.0),
        ):
            with self.subTest(text=text):
                context = MODULE.new_context()
                MODULE.parse_jianzi("大指九徽勾4弦", self.opens, context)
                pitches, reason = MODULE.parse_jianzi(text, self.opens, context)
                expected, _ = MODULE.position_pitch(4, hui, self.opens)
                self.assertIsNone(reason)
                self.assertAlmostEqual(pitches[0], expected)
                self.assertEqual(context["active_left_hui"], hui)

    def test_pre_attack_chuo_is_not_forced_through_scalar_pitch_audit(self):
        data = {
            "metadata": {
                "tonic": "1=C",
                "tuning": {"open_strings": [
                    {"string": index, "pitch": pitch, "octave": octave}
                    for index, pitch, octave in (
                        (1, "C", 3), (2, "D", 3), (3, "F", 3), (4, "G", 3),
                        (5, "A", 3), (6, "C", 4), (7, "D", 4),
                    )
                ]},
            },
            "notes": [{"index": 0, "jianpu": "1", "jianzi": "绰大指五徽六分挑6弦"}],
        }
        report = MODULE.audit(data, 50.0)
        self.assertEqual(report["details"][0]["reason"],
                         "context_dependent_pre_attack_technique")
        self.assertEqual(report["details"][0]["status"], "skipped")

    def test_independent_chuo_is_not_forced_through_scalar_pitch_audit(self):
        data = {
            "metadata": {
                "tonic": "1=C",
                "tuning": {"open_strings": [
                    {"string": index, "pitch": pitch, "octave": octave}
                    for index, pitch, octave in (
                        (1, "C", 3), (2, "D", 3), (3, "F", 3), (4, "G", 3),
                        (5, "A", 3), (6, "C", 4), (7, "D", 4),
                    )
                ]},
            },
            "notes": [
                {"index": 0, "jianpu": "1", "jianzi": "大指七徽挑7弦"},
                {"index": 1, "jianpu": "3", "jianzi": "绰上五徽"},
            ],
        }
        report = MODULE.audit(data, 50.0)
        self.assertEqual(report["details"][1]["reason"], "context_dependent_slide")
        self.assertEqual(report["details"][1]["status"], "skipped")

    def test_jiu_requires_a_current_stopped_position(self):
        pitches, reason = MODULE.parse_jianzi(
            "就挑5弦", self.opens, MODULE.new_context()
        )
        self.assertEqual(pitches, [])
        self.assertEqual(reason, "current_stopped_position_missing")

    def test_open_mode_is_inherited_by_neighbouring_pluck(self):
        context = MODULE.new_context()
        first, reason = MODULE.parse_jianzi("散挑7弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(first, [62])
        second, reason = MODULE.parse_jianzi("勾5弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(second, [57])

    def test_explicit_stopped_note_ends_open_inheritance(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("散挑7弦", self.opens, context)
        stopped, reason = MODULE.parse_jianzi(
            "大指九徽勾5弦", self.opens, context
        )
        self.assertIsNone(reason)
        self.assertNotEqual(stopped, [57])
        inherited, reason = MODULE.parse_jianzi("挑5弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertAlmostEqual(inherited[0], stopped[0])

    def test_fanzhi_ends_harmonic_scope(self):
        context = MODULE.new_context()
        MODULE.parse_jianzi("泛起名指十二徽挑5弦", self.opens, context)
        MODULE.parse_jianzi("泛止", self.opens, context)
        pitches, reason = MODULE.parse_jianzi("勾4弦", self.opens, context)
        self.assertIsNone(reason)
        self.assertEqual(pitches, [55])

    def test_rest_with_fanqi_keeps_harmonic_scope(self):
        data = {
            "metadata": {
                "tonic": "1=C",
                "tuning": {"open_strings": [
                    {"string": index, "pitch": pitch, "octave": octave}
                    for index, pitch, octave in (
                        (1, "C", 3), (2, "D", 3), (3, "F", 3),
                        (4, "G", 3), (5, "A", 3), (6, "C", 4), (7, "D", 4),
                    )
                ]},
            },
            "notes": [
                {"index": 0, "jianpu": "0（休止）", "jianzi": "泛起名指十二徽挑5弦"},
                {"index": 1, "jianpu": "1", "jianzi": "勾4弦"},
            ],
        }
        row = MODULE.audit(data, 50)["details"][1]
        self.assertNotEqual(row.get("reason"), "harmonic_hui_missing")


if __name__ == "__main__":
    unittest.main()
