"""Prepare a Meihua suffix rerun with the preceding phrases frozen as history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'train/eval_inputs_v2_text_protocol/five_scores_latest_full/test_runtime.jsonl'
PREVIOUS = ROOT / 'train/eval_outputs_v3_two_stage/five_scores_latest_runtime_full_1gpu_20261018/predictions.jsonl'
OUTPUT = ROOT / 'train/eval_outputs_v3_two_stage/meihua_chuo_reaudit_work'


def rows(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=SOURCE)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--score', default='S35eJqgV')
    parser.add_argument('--start-phrase', default='p0005')
    args = parser.parse_args()
    inputs = [row for row in rows(args.inputs) if row['score_key'] == args.score]
    predictions = {row['sample_id']: row for row in rows(args.previous)}
    assert len(inputs) > 1 and len({row['sample_id'] for row in inputs}) == len(inputs)
    prefix = [row for row in inputs if row['phrase_id'] < args.start_phrase]
    assert prefix and len(prefix) < len(inputs)
    assert {row['phrase_id'] for row in inputs if row['phrase_id'] >= args.start_phrase}
    for row in prefix:
        old = predictions[row['sample_id']]
        assert old['input_sha256'] == row['input_sha256']
        assert old['protocol_valid'] and old['guqinizer']['ok']
    args.output_dir.mkdir(exist_ok=False)
    workers = args.output_dir / 'workers'
    workers.mkdir()
    (args.output_dir / 'inputs.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in inputs))
    (workers / 'predictions_00.jsonl').write_text(
        ''.join(json.dumps(predictions[row['sample_id']], ensure_ascii=False) + '\n'
                for row in prefix))
    print(json.dumps({'score': args.score, 'input_phrases': len(inputs),
                      'frozen_prefix': len(prefix), 'rerun_suffix': len(inputs) - len(prefix),
                      'output': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
