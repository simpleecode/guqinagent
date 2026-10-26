# GuqinAgent final jianzipu evaluation

This is a structured final-output evaluation, not a text-similarity score. It
uses the repository's authoritative `scripts/audit_jianpu_jianzi_pitch.py`
stateful pitch audit and `agents.abc_to_jianzipu.reference_parser` semantic
parser.  No training pipeline is modified.

## Run

### Current inference artifacts

Current results are organized by method and score, not by run date:

```text
train/eval_outputs/two_stage/
  index.html
  S35eJqgV/                 # 梅花三弄
    inputs.jsonl
    predictions.jsonl
    result_manifest.json
    S35eJqgV.html
    S35eJqgV.pdf            # optional export, always this fixed name
  S1zt293d/                 # 凤求凰, same layout
  .work/inference/          # shared queue scratch data, not another formal version
```

`single_stage`, `direct`, and named ablations use separate method roots.
`train/scripts/publish_method_eval.py --method two_stage --inputs ...
--predictions ... --render` verifies complete ID/hash/protocol coverage and
publishes each score to its fixed directory. Updating a score replaces its
current result; stale PDFs are not carried over to changed predictions.
Replaced published directories go to recoverable trash, not version folders.
Legacy dated experiment directories are provenance, not the current entry point.

To refresh HTML and export a score PDF, use these fixed paths:

```bash
python ABC_J/scripts/visualize_running_eval_score.py \
  --score S35eJqgV \
  --input train/eval_outputs/two_stage/S35eJqgV/inputs.jsonl \
  --pulled train/eval_outputs/two_stage/S35eJqgV/predictions.jsonl \
  --skip-pull \
  --output train/eval_outputs/two_stage/S35eJqgV/S35eJqgV.html \
  --notation-pdf train/eval_outputs/two_stage/S35eJqgV/S35eJqgV.pdf
```

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
* **Tone-type accuracy (按/散/泛)**: exact match of replay-derived `stopped`,
  `open`, or `harmonic` mode against the annotation. Before scoring, all
  predicted and annotated phrases for each score are assembled in source
  order and replayed through the stateful pitch parser, so inherited 散音 and
  泛起…泛止 spans carry across phrase boundaries. Scores with incomplete
  prediction coverage omit this metric.
* **Tone-type distribution similarity**: `1 - TVD` between the normalized
  predicted and annotated 按/散/泛 event distributions, where
  `TVD = 0.5 * Σ|p(mode) - q(mode)|`. `1.0` means identical distributions;
  `0.0` means no overlap.
* **String accuracy**: exact primary-string match, only where the sealed
  reference explicitly supplies that field.
* **Left/right hand**: micro accuracy plus label F1 for explicitly annotated
  primary fingering fields. Missing reference values are excluded rather than
  treated as negatives.
* **Ornament P/R/F1**: micro set comparison of parsed `techniques` per event.
* **Performance-technique usage and diversity**: one unified, paper-facing
  statistic extracted from final notation using the curated
  `complex_fingering_explanations.jsonl` name list plus basic strokes
  (抹/挑/勾/剔, 吟/猱/绰/注, etc.). It covers right-hand, left-hand, and
  coordinated actions. Compound entries such as 勾剔、掐撮三声, and 撮 are
  each counted once per note event. Finger names, hui positions, sound modes
  (按/散/泛), and harmonic-region state markers 泛起/泛止 are not techniques.
  See `performance_technique_usage.by_technique` for event counts, rates, and
  deltas. It reports `distribution_similarity` (`1 - TVD`), raw unique count,
  effective vocabulary size (`exp(Shannon entropy)`), and event-level
  technique density. Ornament P/R/F1 above remains a separate event-level
  semantic-correctness metric, not a second distribution statistic.
* **Final replay warning count** (`rule_violations`): the number of warnings
  emitted when the final authored plans are replayed, phrase by phrase, through
  the same `validate_jianzi_only` rules used by `edit_plan`. This includes
  pitch, fingering, repeat, resonance, and other runtime warnings; warnings
  from intermediate tool turns are excluded. `rule_violation_rate` is the
  fraction of audited phrases with at least one final warning, while
  `rule_warnings_per_phrase` counts warning instances per audited phrase.
  Only complete scores with a final plan for every phrase are scored; skipped
  scores are listed in `warning_replay_incomplete_scores`, never reported as
  warning-free. This makes warning-feedback ablations directly comparable
  when they use the same test set and runtime version.

`per_event.jsonl` retains structured prediction/reference fields, target MIDI,
and audit evidence. `violations.jsonl` contains one record per final replay
warning with its runtime code, source index, and rule evidence. The separate
`final_tool_receipt_warning` statistic is a diagnostic of the last recorded
tool response, not the paper-facing final-state metric.

## Deliberately out of scope / TODO

The first version does not infer subjective musical quality, phrase-level
ornament style, ergonomics of arbitrary multi-stop hand spans, or correctness
of context-dependent compound gestures. Those need a verified rule corpus or
expert labels; they must not be guessed from strings.
