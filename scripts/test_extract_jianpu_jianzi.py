import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("extract_jianpu_jianzi.py")
SPEC = importlib.util.spec_from_file_location("extract_jianpu_jianzi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JianpuDecodeTests(unittest.TestCase):
    def test_demisemiquaver_pitch_is_not_a_rest(self):
        decoded = MODULE.decode_jianpu_token("d5'")
        self.assertFalse(decoded["rest"])
        self.assertEqual(decoded["pitch_name"], "5\u0307")
        self.assertEqual(decoded["duration_name"], "三十二分")

    def test_demisemiquaver_abc_duration(self):
        decoded = MODULE.decode_jianpu_token("d5'")
        abc, _ = MODULE.jianpu_to_abc(decoded, tonic="F")
        self.assertEqual(abc, "c'/2")

    def test_legacy_hardcoded_tuning_is_rejected(self):
        data = {
            "provenance": {
                "method": (
                    "Frida runtime object capture from authorized logged-in "
                    "Android app"
                )
            },
            "metadata": {
                "tuning": {
                    "name": "正调",
                    "value": [0, 0, 0, 0, 0, 0, 0, 0],
                }
            },
        }
        tuning = MODULE.resolve_tuning(data)
        self.assertIsNone(tuning["value"])
        self.assertEqual(tuning["source"], "legacy_hardcoded_value_rejected")

    def test_special_ornament_note(self):
        decoded = MODULE.decode_jianpu_token("y:1'/2'")
        self.assertTrue(decoded["ornament"])
        self.assertEqual(decoded["pitch_name"], "1̇（饰）")
        abc, previous = MODULE.jianpu_to_abc(
            decoded, tonic="F", previous_pitch="c"
        )
        self.assertEqual(abc, "{f}")
        self.assertEqual(previous, "c")

    def test_barline_is_not_a_rest(self):
        decoded = MODULE.decode_jianpu_token("|")
        self.assertTrue(decoded["barline"])
        self.assertEqual(MODULE.jianpu_to_abc(decoded)[0], "|")


class JianziDecodeTests(unittest.TestCase):
    def test_verified_right_hand_codes(self):
        glyph = {
            "std": {
                "tp": "zhyx", "a": "d", "b": "9", "d": "y", "e": "7"
            }
        }
        self.assertEqual(
            MODULE.decode_jianzi_glyph(glyph)["cn"],
            "大指九徽罨7弦",
        )

    def test_zhtyx_c_prefix_is_visible_chuo(self):
        glyph = {
            "std": {
                "tp": "zhtyx",
                "z": "d",
                "h": "7",
                "t": "c:",
                "y": "t",
                "x": "6",
            }
        }
        self.assertEqual(
            MODULE.decode_jianzi_glyph(glyph)["cn"],
            "绰大指七徽挑6弦",
        )

    def test_tyx_s_marker_is_visible_open_string(self):
        glyph = {"std": {"tp": "tyx", "c": "s:", "d": "g", "e": "5"}}
        self.assertEqual(MODULE.decode_jianzi_glyph(glyph)["cn"], "散勾5弦")

    def test_user_verified_compact_glyph_codes(self):
        cases = [
            ({"tp": "tx", "c": "sry:", "e": "5"}, "散如一5弦"),
            (
                {"tp": "zhtyx", "z": "m", "h": "7", "t": "f:",
                 "y": "g", "x": "3"},
                "泛音名指七徽勾3弦",
            ),
            ({"tp": "zhyx", "a": "m", "b": "X", "d": "dz", "e": "5"},
             "名指十徽打摘5弦"),
            ({"tp": "tyx", "c": "jiu:", "d": "yj", "e": "45"},
             "就摘涓45弦"),
        ]
        for std, rendered in cases:
            with self.subTest(std=std):
                self.assertEqual(
                    MODULE.decode_jianzi_glyph({"std": std})["cn"], rendered
                )

    def test_verified_decor_codes(self):
        expected = {
            ":zkzz": "从ㄱ再作",
            ":yh": "应合",
            ":tc": "推出",
            ":fh": "放合",
            ":ji": "急",
            ":ry": "如一",
            ":fc": "反撮",
            ":dy": "打圆",
            ":quz": "曲终",
            ":sl": "索铃",
            ":st": "双弹",
        }
        for raw, rendered in expected.items():
            with self.subTest(raw=raw):
                glyph = {"std": {"tp": "d", "d": raw}}
                self.assertEqual(
                    MODULE.decode_jianzi_glyph(glyph)["cn"], rendered
                )


if __name__ == "__main__":
    unittest.main()
