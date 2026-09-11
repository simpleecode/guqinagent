from __future__ import annotations

import unittest

from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges


class PhraseSplitterTests(unittest.TestCase):
    @staticmethod
    def _counts(notes, ranges):
        return [sum(bool(note.get("sounding")) for note in notes[start:end])
                for start, end in ranges]

    def test_barline_free_tail_uses_hard_fallback_without_backtracking(self) -> None:
        notes = [{"sounding": True, "abc": "C"} for _ in range(10)]
        notes.append({"sounding": False, "abc": "|"})
        notes.extend({"sounding": True, "abc": "D"} for _ in range(60))
        ranges = split_phrase_ranges(
            notes, max_sounding=32,
            is_sounding=lambda note: bool(note.get("sounding")),
        )
        # The frozen v3 rule never backs up to the previous barline. With no
        # later barline it uses max_sounding + 8 as the hard fallback.
        self.assertEqual(ranges[0], (0, 41))
        self.assertTrue(all(count <= 40 for count in self._counts(notes, ranges)))
        self.assertEqual(sum(self._counts(notes, ranges)), 70)

    def test_section_boundary_is_preserved(self) -> None:
        notes = [
            {"sounding": True, "abc": "C", "section": {"number": 1}},
            {"sounding": True, "abc": "D", "section": {"number": 1}},
            {"sounding": True, "abc": "E", "section": {"number": 2}},
        ]
        self.assertEqual(
            split_phrase_ranges(notes, max_sounding=32,
                                is_sounding=lambda note: note["sounding"]),
            [(0, 2), (2, 3)],
        )

    def test_invalid_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_phrase_ranges([], max_sounding=0, is_sounding=lambda note: True)


if __name__ == "__main__":
    unittest.main()
