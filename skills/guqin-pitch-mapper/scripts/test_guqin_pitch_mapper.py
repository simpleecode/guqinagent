import importlib.util
import pathlib
import types
import unittest

PATH = pathlib.Path(__file__).with_name("guqin_pitch_mapper.py")
SPEC = importlib.util.spec_from_file_location("guqin_pitch_mapper", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def args(**kwargs):
    base = dict(open_midi=None, offsets=None, string=5, hui="9",
                mode="stopped", midi=None, jianpu=None, tonic="F",
                tonic_midi=None, octave=0, accidental=0,
                tolerance_cents=50, profile="sitongli")
    base.update(kwargs)
    return types.SimpleNamespace(**base)


class PitchMapperTests(unittest.TestCase):
    def test_stopped_fifth_string_ninth_hui_is_e4(self):
        result = M.position_result(args())
        self.assertEqual(result["pitches"][0]["nearest_midi"], 64)
        self.assertLess(abs(result["pitches"][0]["cents_error"]), 3)

    def test_compact_76_is_c5_on_seventh_string(self):
        result = M.position_result(args(string=7, hui="76"))
        self.assertEqual(result["pitches"][0]["nearest_midi"], 72)
        self.assertLess(abs(result["pitches"][0]["cents_error"]), 6)

    def test_harmonic_and_stopped_sixth_hui_are_distinct(self):
        stopped = M.position_result(args(string=1, hui="6", mode="stopped"))
        harmonic = M.position_result(args(string=1, hui="6", mode="harmonic"))
        self.assertEqual(stopped["pitches"][0]["nearest_midi"], 64)
        self.assertEqual(harmonic["pitches"][0]["nearest_midi"], 76)

    def test_fractional_harmonic_is_invalid(self):
        result = M.position_result(args(hui="76", mode="harmonic"))
        self.assertEqual(result["status"], "invalid")

    def test_reverse_lookup_contains_known_position(self):
        result = M.candidate_result(args(midi=64))
        self.assertTrue(any(c["string"] == 5 and c["hui"] == 9
                            and c["mode"] == "stopped"
                            for c in result["candidates"]))

    def test_validation_emits_mismatch(self):
        result = M.validate_result(args(midi=60))
        self.assertEqual(result["status"], "mismatch")
        self.assertFalse(result["match"])

    def test_offsets_change_open_string_pitch(self):
        result = M.position_result(args(string=2, hui=None, mode="open",
                                        offsets="0,1,0,0,0,0,0"))
        self.assertEqual(result["pitches"][0]["nearest_midi"], 51)

    def test_modern_profile_uses_documented_standard_octave(self):
        result = M.position_result(args(string=3, hui=None, mode="open",
                                        profile="modern"))
        self.assertEqual(result["pitches"][0]["name"], "F2")


class SkillVerifyDocumentTests(unittest.TestCase):
    # tuning, expected open string 3, harmonic 7/string 5,
    # stopped 7.6/string 7
    CASES = [
        ([36, 38, 41, 43, 45, 48, 50], 41, 57, 60),  # 正调
        ([36, 38, 41, 43, 46, 48, 50], 41, 58, 60),  # 蕤宾
        ([36, 38, 40, 43, 45, 48, 50], 40, 57, 60),  # 慢角
        ([36, 39, 41, 43, 46, 48, 51], 41, 58, 61),  # 清商: C♯4 = D♭4
        ([35, 38, 40, 43, 45, 47, 50], 40, 57, 60),  # 慢宫
        ([36, 36, 41, 43, 45, 48, 50], 41, 57, 60),  # 慢商
        ([34, 38, 41, 43, 46, 48, 50], 41, 58, 60),  # 无射
        ([36, 39, 41, 43, 46, 48, 50], 41, 58, 60),  # 凄凉
    ]

    def calculate(self, opens, string, hui=None, mode="open"):
        return M.position_result(args(
            open_midi=",".join(map(str, opens)),
            string=string, hui=hui, mode=mode, profile="modern"
        ))

    def test_eight_tunings_three_core_examples(self):
        for opens, expected_open, expected_harmonic, expected_stopped in self.CASES:
            with self.subTest(opens=opens, example="散勾三"):
                result = self.calculate(opens, 3)
                self.assertEqual(result["pitches"][0]["nearest_midi"], expected_open)
            with self.subTest(opens=opens, example="泛七挑五"):
                result = self.calculate(opens, 5, "7", "harmonic")
                self.assertEqual(result["pitches"][0]["nearest_midi"],
                                 expected_harmonic)
            with self.subTest(opens=opens, example="按七六挑七"):
                result = self.calculate(opens, 7, "76", "stopped")
                self.assertEqual(result["pitches"][0]["nearest_midi"],
                                 expected_stopped)
                self.assertLess(abs(result["pitches"][0]["cents_error"]), 5)

    def test_additional_zhengdiao_examples(self):
        opens = self.CASES[0][0]
        examples = [
            (6, None, "open", "C3"),
            (6, "7", "harmonic", "C4"),
            (6, "7", "stopped", "C4"),
            (6, "76", "stopped", "B♭3"),
            (3, "5", "harmonic", "C4"),
            (5, "4", "harmonic", "A4"),
            (7, None, "open", "D3"),
            (7, "7", "harmonic", "D4"),
            (7, "7", "stopped", "D4"),
            (7, "76", "stopped", "C4"),
        ]
        for string, hui, mode, expected in examples:
            with self.subTest(string=string, hui=hui, mode=mode):
                result = self.calculate(opens, string, hui, mode)
                self.assertEqual(result["pitches"][0]["name"], expected)

    def test_c4_reverse_lookup_preserves_same_pitch_different_timbres(self):
        result = M.candidate_result(args(
            midi=60, open_midi="36,38,41,43,45,48,50",
            profile="modern", tolerance_cents=10
        ))
        found = {
            (item["string"], item["hui"], item["mode"])
            for item in result["candidates"]
        }
        expected = {
            (3, 5, "harmonic"),
            (6, 7, "harmonic"),
            (6, 7.0, "stopped"),
            (7, 7.6, "stopped"),
        }
        self.assertTrue(expected.issubset(found))


if __name__ == "__main__":
    unittest.main()
