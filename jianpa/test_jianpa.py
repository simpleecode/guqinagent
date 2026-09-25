#!/usr/bin/env python3
"""jianpa 端到端测试：合成 MIDI / ABC → 转换 → 用项目自身 parse_jianpu 验证音高往返。"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
JIANKA = str(Path(__file__).resolve().parent)
if JIANKA not in sys.path:
    sys.path.insert(0, JIANKA)

from abc_reader import parse_abc
from key_detect import detect_key
from midi_reader import parse_midi
from scripts.audit_jianpu_jianzi_pitch import parse_jianpu


def write_smf(notes, tpq=480, beats_per_bar=4):
    """notes = [(start_q, dur_q, midi, vel)] → format-0 SMF bytes."""
    return _smf([notes], tpq=tpq)


def write_smf_two_track(right_hand, left_hand, tpq=480):
    """Piano-staff layout: one MTrk chunk per hand → format-1 SMF bytes."""
    return _smf([right_hand, left_hand], tpq=tpq)


def _smf(track_note_lists, *, tpq=480):
    tracks = []
    for notes in track_note_lists:
        events = []
        for start, dur, midi, vel in notes:
            events.append((int(start * tpq), 0x90, midi, vel))
            events.append((int((start + dur) * tpq), 0x80, midi, 0))
        events.sort()
        track = bytearray()
        if not tracks:
            track += b"\x00\xff\x58\x04\x04\x02\x18\x08"      # 4/4
            track += b"\x00\xff\x51\x03\x07\xa1\x20"          # 120bpm
        last = 0
        for tick, status, midi, vel in events:
            delta = tick - last
            last = tick
            var = bytearray()
            value = delta
            while True:
                var.insert(0, value & 0x7F)
                value >>= 7
                if not value:
                    break
            for i, byte in enumerate(var):
                if i < len(var) - 1:
                    var[i] |= 0x80
            track += bytes(var) + bytes([status, midi, vel])
        track += b"\x00\xff\x2f\x00"
        tracks.append(bytes(track))
    header = struct.pack(">HHH", 1 if len(tracks) > 1 else 0, len(tracks), tpq)
    out = b"MThd" + struct.pack(">I", 6) + header
    for track in tracks:
        out += b"MTrk" + struct.pack(">I", len(track)) + track
    return out


MELODY_C_MAJOR = [  # C 大调：1 2 3 4 | 5 6 7 1' |
    (0.0, 1.0, 60, 90), (1.0, 1.0, 62, 90), (2.0, 1.0, 64, 90), (3.0, 1.0, 65, 90),
    (4.0, 1.0, 67, 90), (5.0, 1.0, 69, 90), (6.0, 1.0, 71, 90), (7.0, 1.0, 72, 90),
    (8.0, 0.5, 71, 90), (8.5, 0.5, 69, 90), (9.0, 0.25, 67, 90), (9.25, 0.25, 65, 90),
    (9.5, 0.125, 64, 90), (9.625, 0.125, 62, 90),
]

ABC_SNIPPET = """X:1
T:Test Reel
M:4/4
L:1/8
K:G
|:D2|GABG d2BG|A2Ac B2GE|"""


class MidiReaderTest(unittest.TestCase):
    def test_roundtrip(self):
        data = parse_midi(write_smf(MELODY_C_MAJOR))
        self.assertEqual(data.ticks_per_quarter, 480)
        self.assertEqual(len(data.notes), 14)
        self.assertEqual(data.notes[0].midi, 60)
        self.assertAlmostEqual(data.notes[-1].start_tick / 480, 9.625)


class KeyDetectTest(unittest.TestCase):
    def test_c_major(self):
        pc, mode, score = detect_key([(s, d, m) for s, d, m, _ in MELODY_C_MAJOR])
        self.assertEqual(pc, 0)
        self.assertEqual(mode, "major")
        self.assertGreater(score, 0.8)


class AbcReaderTest(unittest.TestCase):
    def test_snippet(self):
        parsed = parse_abc(ABC_SNIPPET)
        self.assertEqual(parsed["mode"], "major")
        self.assertEqual(parsed["tonic_pc"], 7)  # K:G
        self.assertTrue(parsed["bars"])
        midis = [m for _, d, m in parsed["events"] if d > 0]
        self.assertIn(67, midis)  # G4
        self.assertIn(74, midis)  # d5


class RuntimeInvariantsTest(unittest.TestCase):
    def test_midi_to_runtime_roundtrip(self):
        from to_runtime import build_readable
        from midi_reader import parse_midi
        data = parse_midi(write_smf(MELODY_C_MAJOR))
        events = [(n.start_tick / data.ticks_per_quarter,
                   (n.end_tick - n.start_tick) / data.ticks_per_quarter, n.midi)
                  for n in data.notes]
        bars = [4.0, 8.0]
        readable = build_readable(events, bars, title="测试", tonic_pc=0, mode="major", beats_per_bar=4)
        readable["tonic_label"] = "1=C"
        self.assertEqual(readable["metadata"]["tonic"], "1=C")
        tonic = readable["tonic_degree1_midi"]
        self.assertEqual(tonic, 60)
        # 音高往返：每个 attack 行的 jianpu 解析回目标 MIDI
        attacks = [r for r in readable["notes"] if r["duration"] in ("四分", "八分", "十六分", "三十二分")]
        self.assertEqual(len(attacks), 14)
        target_by_order = [m for _, _, m, _ in MELODY_C_MAJOR]
        for row, target in zip(attacks, target_by_order):
            parsed = parse_jianpu(row["jianpu"], tonic)
            self.assertIsNotNone(parsed, row["jianpu"])
            self.assertAlmostEqual(parsed, target, delta=1e-6,
                                   msg=f"{row['jianpu']} -> {parsed} != {target}")
        # 小节线不占音序
        bars_rows = [r for r in readable["notes"] if r["duration"] == "小节线"]
        self.assertEqual(len(bars_rows), 2)
        # phrase 切分与运行时 schema
        from to_runtime import build_runtime
        rows = build_runtime(readable, score_key="JTEST01")
        self.assertTrue(rows)
        ordinals = []
        for payload in rows:
            notes = payload["runtime_item"]["input"]["notes_without_jianzi"]
            for n in notes:
                self.assertIn("event_index", n)
                if n["duration"] != "小节线":
                    ordinals.append(n["event_index"])
                else:
                    self.assertIsNone(n["event_index"])
        self.assertEqual(ordinals, sorted(ordinals))
        self.assertEqual(ordinals, list(range(len(ordinals))))
        # 输入文件与既有评估完全同构的字段
        for payload in rows:
            self.assertEqual(payload["schema_version"], "agent-eval-input-2.0")
            item = payload["runtime_item"]
            self.assertEqual(item["input"]["normalized_tuning"]["open_midi"],
                             [48, 50, 53, 55, 57, 60, 62])
            self.assertTrue(item["input"]["event_range"]["end_exclusive"] > item["input"]["event_range"]["start"])


class MelodyExtractionTest(unittest.TestCase):
    def _events(self, smf_bytes):
        import jianpa as jianpa_cli
        data = parse_midi(smf_bytes)
        return jianpa_cli._melody_events(data)

    def test_track_selection_drops_left_hand(self):
        right = [(0.0, 1.0, 72, 90), (1.0, 1.0, 74, 90), (2.0, 1.0, 76, 90)]
        left = [(0.0, 2.0, 45, 80), (2.0, 1.0, 40, 80)]  # A2/E2 bass
        events = self._events(write_smf_two_track(right, left))
        midis = sorted(m for _, _, m in events)
        self.assertEqual(midis, [72, 74, 76])

    def test_skyline_drops_accompaniment_under_sustain(self):
        melody = [(0.0, 2.0, 76, 90), (2.0, 1.0, 74, 90)]   # E5 sustains 2 beats
        filler = [(1.0, 0.5, 60, 80)]                        # C4 under the sustain
        events = self._events(write_smf([(0.0, 2.0, 76, 90), (1.0, 0.5, 60, 80), (2.0, 1.0, 74, 90)]))
        midis = [m for _, _, m in events]
        self.assertEqual(midis, [76, 74])

    def test_chord_partner_within_octave_kept(self):
        # Top voice E5=76 with C5=72 struck together → 撮 dyad (alt within 12)
        events = self._events(write_smf([(0.0, 1.0, 76, 90), (0.0, 1.0, 72, 80), (1.0, 1.0, 74, 90)]))
        self.assertEqual(len(events), 3)
        same_onset = [m for s, _, m in events if abs(s - 0.0) < 1e-6]
        self.assertEqual(sorted(same_onset), [72, 76])


class OctavePlacementTest(unittest.TestCase):
    def test_center_octaves_pulls_high_median_down(self):
        from to_runtime import center_octaves
        events = [(0.0, 1.0, 74), (1.0, 1.0, 76), (2.0, 1.0, 78), (3.0, 1.0, 74)]
        shifted = center_octaves(events)
        self.assertEqual([m for _, _, m in shifted], [62, 64, 66, 62])

    def test_center_octaves_leaves_centered_melody_alone(self):
        from to_runtime import center_octaves
        events = [(0.0, 1.0, 60), (1.0, 1.0, 64), (2.0, 1.0, 67)]
        self.assertEqual(center_octaves(events), events)

    def test_fold_outliers_moves_only_extremes(self):
        from to_runtime import fold_outliers
        events = [(0.0, 1.0, 48), (1.0, 1.0, 60), (2.0, 1.0, 88), (3.0, 1.0, 44), (4.0, 1.0, 79)]
        folded = [m for _, _, m in fold_outliers(events)]
        self.assertEqual(folded, [48, 60, 76, 56, 79])


class JianpuTextTest(unittest.TestCase):
    SNIPPET = """T:测试小调
1=Bb 4/4
3' 3' 5 6 | 1' - 7 6 |
5 6 1'&5 | #4/2 0/2 5 2 |
"""

    def test_parse_events_and_bars(self):
        from jianpu_reader import parse_jianpu_text
        parsed = parse_jianpu_text(self.SNIPPET)
        self.assertEqual(parsed["title"], "测试小调")
        self.assertEqual(parsed["tonic_pc"], 10)  # Bb
        self.assertEqual(parsed["beats_per_bar"], 4.0)
        events = parsed["events"]
        # 1=Bb → 锚定 Bb3=58；3' = +4 半音再升八度
        self.assertEqual(events[0][2], 58 + 4 + 12)
        self.assertEqual(events[0][1], 1.0)  # 默认四分
        # 延音 - ：第二小节 1' 占两拍
        tie = next(e for e in events if e[0] == 4.0)
        self.assertEqual((tie[1], tie[2]), (2.0, 58 + 12))
        # 撮 1'&5：两个同 onset 事件，时值相同
        dyad_main = next(e for e in events if e[0] == 10.0 and e[2] == 58 + 12)
        dyad_partner = next(e for e in events if e[0] == 10.0 and e[2] == 58 + 7)
        self.assertEqual(dyad_main[1], dyad_partner[1])
        # #4/2 八分变化音（第三小节只有 3 拍音符，小节线不占时值）
        sharp = events[-3]
        self.assertEqual((sharp[0], sharp[1], sharp[2]), (11.0, 0.5, 58 + 5 + 1))
        # 手写小节线位置（第一小节结束 = 拍 4）
        self.assertIn(4.0, parsed["bars"])

    def test_build_readable_pairs_dyad_once(self):
        from jianpu_reader import parse_jianpu_text
        from to_runtime import build_readable
        parsed = parse_jianpu_text(self.SNIPPET)
        readable = build_readable(parsed["events"], parsed["bars"], title="t",
                                  tonic_pc=parsed["tonic_pc"], mode="minor",
                                  beats_per_bar=4.0)
        # 撮行：高音（1'）为主、低音（5）为副，且伙伴音不再重复出现
        dyad_rows = [r for r in readable["notes"] if r.get("jianpu_alt")]
        self.assertEqual(len(dyad_rows), 1)
        self.assertEqual(dyad_rows[0]["jianpu"], "1\u0307")
        self.assertEqual(dyad_rows[0]["jianpu_alt"], "5")
        attacks = [r for r in readable["notes"] if r["duration"] in
                   ("二分", "四分", "八分", "十六分", "三十二分")
                   and not r["jianpu"].startswith(("－", "0"))]
        # 撮只占一个 attack 行：12 单音 + 1 撮
        self.assertEqual(len(attacks), 13)
        tonic = readable["tonic_degree1_midi"]
        for row in attacks:
            self.assertIsNotNone(parse_jianpu(row["jianpu"], tonic), row["jianpu"])


if __name__ == "__main__":
    unittest.main()
