---
name: guqin-pitch-mapper
description: Map guqin jianzipu (减字谱) playing positions to sounding pitch, map a target pitch back to playable string/hui candidates, or validate jianzipu against numbered notation/absolute pitch under a specified tuning. Use for pitch checking, candidate generation, training-data audit, or agent feedback involving open, stopped, harmonic, slide-to-position, compound, or pitch-indeterminate guqin techniques.
---

# Guqin pitch mapper

Use `scripts/guqin_pitch_mapper.py` as the deterministic source of pitch
arithmetic. Do not estimate a hui position from memory when the script can
calculate it.

## Workflow

1. Normalize the tuning to seven open-string MIDI values. Prefer explicit
   `open_midi`; otherwise use 正调 plus per-string semitone offsets. The
   default `modern` profile uses `C2 D2 F2 G2 A2 C3 D3`; pass
   `--profile sitongli` for captured app data whose octave convention is one
   octave higher.
2. Normalize the event to one or more pitch-bearing actions:
   `open`, `stopped`, `harmonic`, or `slide`.
3. Run the matching command:

```powershell
python scripts/guqin_pitch_mapper.py position --string 5 --hui 9
python scripts/guqin_pitch_mapper.py pitch --midi 64
python scripts/guqin_pitch_mapper.py validate --string 5 --hui 9 --jianpu 3 --tonic F
python scripts/guqin_pitch_mapper.py position --profile sitongli --string 5 --hui 9
```

4. Consume the JSON result. Treat `status` as the agent signal:
   `exact` is safe to enforce; `approximate` is usable with its cents error;
   `inferred` depends on supplied context; `indeterminate` must not be scored
   as a wrong pitch.
5. Preserve every candidate for pitch-to-jianzi mapping. Rank by playability or
   phrase continuity outside this tool; pitch equivalence alone is many-to-one.

## Interpretation rules

- Treat an explicit string without a hui as an open string.
- Treat a string plus hui as a stopped note unless harmonic context is explicit.
- Allow harmonics only at exact hui 1–13. Reject fractional harmonic positions.
- Resolve `上`, `下`, `绰`, `注`, `进`, and `退` only when a destination hui or
  destination pitch is supplied. Otherwise return `indeterminate`.
- For a slide glyph containing only a target hui, inherit the previous
  pitch-bearing event's string and mode and label the result `inferred`.
- Split 撮/泼/剌 and other compounds into their component strings and return a
  pitch set. Do not collapse a chord to one pitch.
- Treat 吟、猱、撞 and similar ornaments as modifications of the preceding or
  anchored pitch, not independent fixed pitches.

Read [notation-and-confidence.md](references/notation-and-confidence.md) when
parsing raw Sitongli component dictionaries or explaining confidence. Read
[algorithm-evidence.md](references/algorithm-evidence.md) when auditing the
formula or comparing it with the reverse-engineered app behavior.

## Output contract

Always expose:

- `status`: `exact | approximate | inferred | indeterminate | invalid`
- `pitches`: zero or more `{midi, name, cents_error}` objects
- `evidence`: normalized tuning, string, hui, mode, and formula inputs
- `diagnostics`: machine-readable warning/error codes
- `candidates`: for reverse lookup, all positions inside the requested range

Never invent an octave, destination, string, or harmonic state. Return an
explicit diagnostic and the strongest partial evidence available.
