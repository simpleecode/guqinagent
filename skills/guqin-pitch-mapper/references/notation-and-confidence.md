# Notation and confidence

## Sitongli component shapes

Pitch-bearing component layouts observed in captured data:

- `zhyx`: `a=left finger, b=hui, d=right technique, e=string`
- `zhtyx`: `z=left finger, h=hui, y=right technique, x=string`
- `th`: `h=destination hui, t=slide direction`; inherit the prior string
- `tyx`, `yx`, `tx`, `x`: pluck/string forms; no hui means open string
- compound 撮: stopped component commonly uses `c=hui,d=string`; the paired
  open component commonly uses `g=string`
- `d`: ornament/control token; normally has no independent pitch

Compact hui codes use `X=10`, `X1/X2/X3=11/12/13`, and a trailing digit as
tenths between adjacent hui positions: `76=7.6`, `X8=10.8`, `X23=12.3`.
`Y` means 徽外 and has no single fixed pitch.

## Confidence semantics

| Status | Meaning | Validation use |
|---|---|---|
| `exact` | Exact hui/open-string arithmetic rounds to the requested semitone | hard signal |
| `approximate` | Physical position is between tempered semitones | soft signal; inspect cents |
| `inferred` | Pitch arithmetic is sound but string/mode came from context | soft signal |
| `indeterminate` | Technique lacks a fixed destination or required context | abstain |
| `invalid` | Impossible input, such as fractional-hui harmonic | reject representation |

Pitch equality defaults to ±50 cents. Tighten `--tolerance-cents` for clean
machine-authored data, but retain the unrounded MIDI float and cents error.

## Jianpu normalization

For a major-scale numbered pitch:

`midi = tonic_midi + [0,2,4,5,7,9,11][degree-1] + 12*octave + accidental`

Provide `--tonic-midi` when octave identity matters. A tonic name such as `F`
alone defaults to the F in MIDI octave 3 (`F3=53`) and is therefore an explicit
convention, not a fact inferred from the tuning name.

## Absolute-octave profiles

- `modern` (default): 正调 `C2 D2 F2 G2 A2 C3 D3`, matching
  `docs/skill_verify.md` and the convention in which fifth-string fourth-hui
  harmonic is `A4=440 Hz`.
- `sitongli`: `C3 D3 F3 G3 A3 C4 D4`, matching the captured app/export octave
  convention used by earlier reverse-engineering examples.

The profiles differ only by 12 semitones. Relative string/hui arithmetic is
identical. Prefer explicit `--open-midi` whenever the source states pitches.
