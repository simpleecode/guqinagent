# Algorithm evidence

The implementation follows the constants and control flow documented in
`../../../docs/PITCH_ALGORITHM_REVERSE_ENGINEERED.md`:

- Modern standard-pitch 正调 open strings: `C2 D2 F2 G2 A2 C3 D3`
- Sitongli runtime/export octave convention: `C3 D3 F3 G3 A3 C4 D4`
- major scale offsets: `[0,2,4,5,7,9,11]`
- harmonic hui semitone table:
  `[36,31,28,24,19,28,12,28,19,24,28,31,36]`
- pitch checking compares independently normalized integer/near-integer
  pitches rather than translating one notation text directly into the other

For stopped notes, this tool closes the fractional-hui gap with the physical
string-length formula. Hui coordinates measured from the bridge are:

`[1/8,1/6,1/5,1/4,1/3,2/5,1/2,3/5,2/3,3/4,4/5,5/6,7/8]`

Interpolate the coordinate linearly for `n.f` and compute:

`sounding_midi = open_midi + 12*log2(1 / coordinate)`

At exact hui, stopped-note intervals round to
`[36,31,28,24,19,16,12,9,7,5,4,3,2]`. Harmonics instead use the symmetric
app table above. This distinction is especially important at hui 6 and 8.

This exactly reproduces the known examples:

- 5th string, 9th hui: `A3 + 7.02 semitones ≈ E4`
- 7th string, 7.6 hui: `D4 + 10.04 semitones ≈ C5`

The fractional interpolation is a physics-derived completion, not yet a
byte-for-byte recovery of the app's hui virtual method. Results therefore
retain cents error and identify the formula in `evidence`.
