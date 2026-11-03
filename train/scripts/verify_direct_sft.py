#!/usr/bin/env python3
"""Preflight the full-artifact dataset without loading an LLM or private audits."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.direct_generation import PROTOCOL, SYSTEM, user_from_public_prompt


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'train/data/guqin_direct_sft')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--test-input', type=Path, required=True)
    parser.add_argument('--modern-input', type=Path, required=True)
    parser.add_argument('--splits', type=Path, default=ROOT / 'ABC_J/results/agent_dataset_splits.csv')
    args = parser.parse_args()
    manifest = json.loads((args.data / 'dataset_manifest.json').read_text())
    assert manifest['protocol'] == PROTOCOL
    assert sha(args.data / 'guqin_agent_train.jsonl') == manifest['dataset_sha256']
    assert sha(args.data / 'guqin_agent_train_metadata.jsonl') == manifest['metadata_sha256']
    assert sha(args.source_dir / 'guqin_agent_train.jsonl') == manifest['source_dataset_sha256'] == '984779ea61a9c582328518bf8d4e023826bf31b8c705f927c1dfbd9d97b00f4a'
    data = read(args.data / 'guqin_agent_train.jsonl')
    meta = read(args.data / 'guqin_agent_train_metadata.jsonl')
    original_data = read(args.source_dir / 'guqin_agent_train.jsonl')
    original_meta = read(args.source_dir / 'guqin_agent_train_metadata.jsonl')
    assert len(data) == len(meta) == len(original_data) == len(original_meta) == manifest['training_rows'] == 3976
    assert len({m['source_sample_id'] for m in meta}) == 3976
    assert [m['source_sample_id'] for m in meta] == [m['source_sample_id'] for m in original_meta]
    for row, m, original in zip(data, meta, original_data):
        assert row['tools'] == '[]' and set(row) == {'messages', 'tools'}
        messages = row['messages']
        assert [x['role'] for x in messages] == ['system', 'user', 'assistant']
        assert all(set(x) == {'role', 'content'} for x in messages)
        assert messages[0]['content'] == SYSTEM
        assert messages[1]['content'] == user_from_public_prompt(original['messages'][1]['content'])
        assert not any(v in str(messages) for v in ('<tool_call>', ':warning:', ':hint:', 'teacher-only', 'GQS'))
        answer = json.loads(messages[2]['content'])
        assert set(answer) == {'jianzi_rows'} and len(answer['jianzi_rows']) == m['target_event_rows']
        assert len({x[0] for x in answer['jianzi_rows']}) == len(answer['jianzi_rows'])
        current = messages[1]['content'].split('【当前段 ', 1)[1]
        current_ids = {int(line.split('｜', 1)[0]) for line in current.splitlines() if re.match(r'^\d+｜', line)}
        assert current_ids == {x[0] for x in answer['jianzi_rows']}
        assert all(isinstance(x[0], int) and not isinstance(x[0], bool) and isinstance(x[1], str) for x in answer['jianzi_rows'])
        assert hashlib.sha256(messages[2]['content'].encode()).hexdigest() == m['target_sha256']
        assert m['agent_stage'] == 'direct' and m['tool_call_turn_count'] == 0 and m['assistant_turn_count'] == 1
    tests = read(args.test_input); modern = read(args.modern_input)
    train = {m['score_key'] for m in meta}
    test_scores = {r['score_key'] for r in tests}
    modern_scores = {r['score_key'] for r in modern}
    assert len(train) == 153 and len(test_scores) == 48 and len(modern_scores) == 10
    assert not train & (test_scores | modern_scores)
    with args.splits.open(encoding='utf-8-sig') as handle:
        splits = {r['score_key']: r for r in csv.DictReader(handle)}
    assert all(splits[s]['split'] == 'train' for s in train)
    assert not {splits[s]['leakage_group_id'] for s in train} & {splits[s]['leakage_group_id'] for s in test_scores}
    report = {'valid': True, 'rows': 3976, 'train_scores': 153, 'assistant_turns': 3976, 'tool_calls': 0,
        'same_source_samples': True, 'same_initial_musical_inputs': True, 'complete_target_events': True,
        'score_overlap': [], 'traditional_group_overlap': [], 'modern_melody_independence': 'not established',
        'dataset_sha256': manifest['dataset_sha256']}
    (args.data / 'direct_validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
