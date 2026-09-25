#!/usr/bin/env python3
"""Dependency-free Standard MIDI File reader (format 0/1, tick-based).

Drums (channel 9) are dropped. Returns absolute-tick note events plus the
first tempo/time-signature meta events, which is all the converter needs.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


@dataclass
class MidiNote:
    start_tick: int
    end_tick: int
    midi: int
    velocity: int
    channel: int
    track: int = 0


@dataclass
class MidiFileData:
    ticks_per_quarter: int
    notes: list[MidiNote]
    quarter_beats_per_bar: float  # from time signature (default 4/4)
    key_signature: int | None     # sharps(+)/flats(-), may be None


def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def parse_midi(path_or_bytes) -> MidiFileData:
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File (missing MThd)")
    length = struct.unpack(">I", data[4:8])[0]
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE-timed MIDI is not supported")
    pos = 8 + length
    notes: list[MidiNote] = []
    beats_per_bar = 4.0
    key_signature = None
    for track_index in range(ntrks):
        if data[pos:pos + 4] != b"MTrk":
            pos = data.find(b"MTrk", pos) + 4
            if pos == 3:
                raise ValueError("corrupt MIDI: no more tracks")
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track = data[pos + 8:pos + 8 + track_len]
        pos += 8 + track_len
        tick = 0
        running = None
        i = 0
        active: dict[tuple[int, int], tuple[int, int]] = {}  # (ch,note)->(vel,start)
        while i < len(track):
            delta, i = _read_varlen(track, i)
            tick += delta
            status = track[i]
            if status < 0x80:
                if running is None:
                    break
                status = running
            else:
                i += 1
                running = status if status < 0xF0 else None
            kind = status & 0xF0
            channel = status & 0x0F
            if kind in (0x80, 0x90):
                note, vel = track[i], track[i + 1]
                i += 2
                if kind == 0x90 and vel:
                    active[(channel, note)] = (vel, tick)
                else:
                    hit = active.pop((channel, note), None)
                    if hit and channel != 9:
                        notes.append(MidiNote(hit[1], tick, note, hit[0], channel, track_index))
            elif kind in (0xA0, 0xB0, 0xE0):
                i += 2
            elif kind in (0xC0, 0xD0):
                i += 1
            elif status == 0xFF:
                meta_type = track[i]
                length, i = _read_varlen(track, i + 1)
                payload = track[i:i + length]
                i += length
                if meta_type == 0x58 and len(payload) >= 2:
                    numerator = payload[0]
                    denominator = 2 ** payload[1]
                    beats_per_bar = numerator * 4.0 / denominator
                elif meta_type == 0x59 and len(payload) >= 1:
                    key_signature = struct.unpack(">b", payload[:1])[0]
            elif status in (0xF0, 0xF7):
                length, i = _read_varlen(track, i)
                i += length
    notes.sort(key=lambda n: (n.start_tick, -n.midi))
    return MidiFileData(division, notes, beats_per_bar, key_signature)
