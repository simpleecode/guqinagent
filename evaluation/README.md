# GuqinAgent final jianzipu evaluation

This is a structured final-output evaluation, not a text-similarity score. It
uses the repository's authoritative `scripts/audit_jianpu_jianzi_pitch.py`
stateful pitch audit and `agents.abc_to_jianzipu.reference_parser` semantic
parser.  No training pipeline is modified.

## Run

```bash
cd /Volumes/F/work/Gunqin-agent
python -m evaluation.run_eval \
  --pred train/eval_outputs_v3_two_stage/remote_a100_qwen35_9b_final_20260917/SCf7VJzZ.jsonl \
  --reference ABC_J/agent_training/evaluation_text_protocol_v2/evaluation_pairs_test.jsonl \
  --experiment two_stage_smoke \
  --model qwen3.5-9b-lora
```

Results are written under `evaluation_results/<experiment>/` and are ignored
by Git. Prediction JSONL may be append-only: the last row for each
`sample_id` is used.

## Definitions

* **Pitch Accuracy @ 50 cents**: fraction of statefully parseable predicted
  sounding events whose audit result matches the numbered-notation target
  within the configured tolerance.
* **Pitch MAE**: mean absolute cents error over pitch pairs emitted by that
  same audit. Multi-tone events use its minimum-error pairing.
* **String accuracy**: exact primary-string match, only where the sealed
  reference explicitly supplies that field.
* **Left/right hand**: micro accuracy plus label F1 for explicitly annotated
  primary fingering fields. Missing reference values are excluded rather than
  treated as negatives.
* **Ornament P/R/F1**: micro set comparison of parsed `techniques` per event.
* **Technique Usage Statistics**: for every parsed technique,
  `prediction_events` / `reference_events` count note events containing that
  technique (not textual occurrences). `prediction_rate` /
  `reference_rate` divide those counts by all valid note events, and
  `rate_delta = prediction_rate - reference_rate`. Repeating a technique in
  one event still counts once. `technique_density` reports the same event
  incidence and rates for “contains at least one technique”, making an
  overall overly-plain / overly-dense prediction easy to spot.
* **Rule violation rate**: unique events with a deterministic parser-backed
  violation divided by pitch-evaluable events. Initial violations are pitch
  mismatch and unequivocal unresolvable/unplayable parser states.

`per_event.jsonl` retains structured prediction/reference fields, target MIDI,
and audit evidence. `violations.jsonl` has the precise state evidence.

## Deliberately out of scope / TODO

The first version does not infer subjective musical quality, phrase-level
ornament style, ergonomics of arbitrary multi-stop hand spans, or correctness
of context-dependent compound gestures. Those need a verified rule corpus or
expert labels; they must not be guessed from strings.
