#!/usr/bin/env python3
"""jianpa — 在线旋律爬取 → 古琴 Agent 推理输入转换工具。

用法：
  python jianpa/jianpa.py crawl --source thesession --search "waltz" --limit 3
  python jianpa/jianpa.py crawl --source mutopia --search "sonata" --limit 3
  python jianpa/jianpa.py crawl --source url --url https://.../tune.abc
  python jianpa/jianpa.py convert cache/xxx.abc  --title 曲名 --score-key T001
  python jianpa/jianpa.py convert cache/xxx.mid  --tonic 1=F
转换产出（--out-dir 默认 jianpa/out）：
  <score-key>/jianpu_jianzi_readable.json   谱面中间格式（可接 PDF 出谱）
  <score-key>_public.jsonl                   eval_two_stage_score.py 公开输入
推理：
  python train/scripts/eval_two_stage_score.py --input jianpa/out/<key>_public.jsonl \
      --score-key <key> --constrain-walk-hui ... （与既有评估完全一致）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import crawl
from abc_reader import parse_abc
from jianpu_reader import parse_jianpu_text
from key_detect import detect_key, tonic_label
from midi_reader import parse_midi
from to_runtime import build_readable, center_octaves, fold_outliers, write_outputs


def _melody_events(data, *, onset_window_quarters: float = 0.04):
    """Melody extraction for polyphonic MIDI (piano/guitar etc.).

    When notes span several tracks (piano staves, guitar duo), keep only the
    track whose median pitch is highest — that is the melody/right-hand
    staff in Mutopia-style files.  Within the kept voices, a skyline pass
    drops accompaniment figures that start under a still-sounding higher
    note, and one chord partner within an octave below the top is kept (the
    guqin 撮 case), snapped to the group's start so build_readable pairs it
    as ``jianpu_alt``.
    """
    live = [n for n in data.notes if n.velocity > 0 and n.end_tick > n.start_tick]
    by_track: dict[int, list[MidiNote]] = {}
    for note in live:
        by_track.setdefault(note.track, []).append(note)
    if len(by_track) > 1:
        # Melody-track scoring for karaoke/pop layouts: the lead voice is
        # the track with the densest onsets covering most of the piece, in
        # a singable register, without heavy self-overlap.  "Highest median
        # pitch" alone mistakes sparse descant/pad tracks for the melody,
        # and remapped-percussion tracks (midi 0-1) must never win.
        sane = {t: [n for n in notes if 36 <= n.midi <= 96]
                for t, notes in by_track.items()}
        sane = {t: notes for t, notes in sane.items() if notes}
        pool = sane or by_track
        global_span = max(
            max(n.end_tick for n in notes) - min(n.start_tick for n in notes)
            for notes in pool.values())

        def melody_score(item):
            _, notes = item
            notes = sorted(notes, key=lambda n: n.start_tick)
            onsets = len({n.start_tick for n in notes})
            span = notes[-1].end_tick - notes[0].start_tick
            coverage = min(span / global_span, 1.0) if global_span else 0.0
            overlap = 1 - onsets / len(notes)
            median = sorted(n.midi for n in notes)[len(notes) // 2]
            register = 1.0 if 58 <= median <= 84 else 0.2
            return onsets * coverage * register * (1 - min(overlap, 0.9))

        live = by_track[max(pool.items(), key=melody_score)[0]]
    notes = sorted(live, key=lambda n: (n.start_tick, -n.midi))
    groups = []
    for note in notes:
        if groups and (note.start_tick - groups[-1][-1].start_tick) / data.ticks_per_quarter <= onset_window_quarters:
            groups[-1].append(note)
        else:
            groups.append([note])
    sounding: list[tuple[int, int]] = []  # (end_tick, midi)
    events = []
    for group in groups:
        start = group[0].start_tick
        sounding = [(end, midi) for end, midi in sounding if end > start]
        top = group[0]
        if sounding and top.midi < max(midi for _, midi in sounding):
            continue
        sounding.append((top.end_tick, top.midi))
        events.append((start / data.ticks_per_quarter,
                       (top.end_tick - start) / data.ticks_per_quarter, top.midi))
        for partner in group[1:]:
            if top.midi - partner.midi <= 12 and partner.midi != top.midi:
                sounding.append((partner.end_tick, partner.midi))
                events.append((start / data.ticks_per_quarter,
                               (partner.end_tick - start) / data.ticks_per_quarter,
                               partner.midi))
                break
    events.sort(key=lambda e: (round(e[0], 4), -e[2]))
    return events


def _events_from_midi(path: str):
    data = parse_midi(path)
    tpq = data.ticks_per_quarter
    events = _melody_events(data)
    bars = []
    bar_len = data.quarter_beats_per_bar
    if bar_len > 0:
        total = max((e[0] + e[1] for e in events), default=0.0)
        pos = bar_len
        while pos < total - 1e-6:
            bars.append(pos)
            pos += bar_len
    return events, bars, None


def _events_from_abc(path: str):
    parsed = parse_abc(Path(path).read_text(encoding="utf-8"))
    events = parsed["events"]
    bars = parsed["bars"]
    beats = parsed["beats_per_bar"]
    if not bars:
        total = max((e[0] + e[1] for e in events), default=0.0)
        pos = beats
        while pos < total - 1e-6:
            bars.append(pos)
            pos += beats
    return events, bars, parsed


def cmd_crawl(args):
    if args.source == "thesession":
        results = crawl.search_thesession(args.search, args.limit)
    elif args.source == "mutopia":
        results = crawl.search_mutopia(args.search, args.limit)
    else:
        results = [crawl.download_url(args.url)]
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


def cmd_convert(args):
    path = Path(args.file)
    suffix = path.suffix.lower()
    header_tonic = ""  # jianpu 文本头部调号（保留用户写法，如 Bb）
    if suffix in (".mid", ".midi"):
        events, bars, parsed = _events_from_midi(str(path))
        mode = None
    elif suffix in (".abc", ".txt", ".jp", ".jianpu"):
        text = path.read_text(encoding="utf-8")
        if suffix != ".abc" and not re.search(r"^[XK]:", text, re.M):
            parsed = parse_jianpu_text(text)
            events, bars, mode = parsed["events"], parsed["bars"], None
            header_tonic = parsed["tonic_label"]
        else:
            events, bars, parsed = _events_from_abc(str(path))
            mode = parsed["mode"]
    else:
        raise SystemExit(f"unsupported file type: {suffix} (use .mid/.midi/.abc/.jp)")

    sounding = [(s, d, m) for s, d, m in events if d > 0]
    if not sounding:
        raise SystemExit("no sounding notes found")
    detected_pc, detected_mode, confidence = detect_key(sounding)
    if args.tonic and args.tonic != "auto":
        label = args.tonic
        letter = label.split("=", 1)[1].strip()
        match = re.fullmatch(r"([A-G])([#b♯♭]?)", letter)
        if match is None:
            raise SystemExit(f"unsupported tonic label: {label}")
        accidental = match.group(2)
        pc = ("CDEFGAB".index(match.group(1))
              + (1 if accidental in ("#", "♯") else -1 if accidental in ("b", "♭") else 0)) % 12
        tonic_pc, mode = pc, (mode or detected_mode)
        auto = False
    elif header_tonic:
        label = f"1={header_tonic}"
        tonic_pc = parsed["tonic_pc"]
        mode = mode or detected_mode
        auto = False
    else:
        tonic_pc, mode = detected_pc, (mode or detected_mode)
        label = tonic_label(tonic_pc, mode)
        auto = True

    events = fold_outliers(center_octaves(events))
    title = args.title or (parsed or {}).get("title") or path.stem
    score_key = args.score_key or ("J" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6)))
    readable = build_readable(
        events, bars, title=title, tonic_pc=tonic_pc, mode=mode,
        beats_per_bar=4.0)
    readable["metadata"]["tonic"] = label
    readable["tonic_label"] = label
    summary = write_outputs(readable, score_key=score_key, out_dir=Path(args.out_dir))
    print(json.dumps({
        **summary, "score_key": score_key, "title": title,
        "tonic": label, "key_mode": mode,
        "detected": {"tonic_pc": detected_pc, "mode": detected_mode, "confidence": round(confidence, 3)},
        "tonic_auto": auto, "notes": len(readable["notes"])}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    crawl_parser = sub.add_parser("crawl", help="从公开源爬取旋律")
    crawl_parser.add_argument("--source", choices=("thesession", "mutopia", "url"), required=True)
    crawl_parser.add_argument("--search", default="")
    crawl_parser.add_argument("--url", default="")
    crawl_parser.add_argument("--limit", type=int, default=3)
    crawl_parser.set_defaults(func=cmd_crawl)

    convert_parser = sub.add_parser("convert", help="转换为推理输入")
    convert_parser.add_argument("file")
    convert_parser.add_argument("--title", default="")
    convert_parser.add_argument("--tonic", default="auto", help="auto 或 1=C/1=F/…")
    convert_parser.add_argument("--score-key", default="")
    convert_parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "out"))
    convert_parser.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
