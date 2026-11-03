#!/usr/bin/env python3
"""Keep only the public initial input and committed final artifact of each teacher trace."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.direct_generation import PROTOCOL, SYSTEM, event_sources, parse_answer, user_from_public_prompt


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def committed_public_rows(messages, item):
    """Replay successful public edit arguments; do not mistake the last delta for a full plan."""
    mapping = event_sources(item)
    initial = messages[1]['content'].split('【当前段 ', 1)[1]
    state = {}
    for line in initial.splitlines():
        if re.match(r'^\d+｜', line):
            fields = line.split('｜')
            value = re.split(r':(?:warning|hint):', fields[-1], maxsplit=1)[0]
            if value in ('[减字待填写]', '[空]', '[续音]', '[—]'):
                value = ''
            elif value.startswith('[') and value.endswith(']'):
                value = value[1:-1]
            else:
                raise ValueError(f'unknown initial display: {value}')
            state[int(fields[0])] = value
    if state.keys() != mapping.keys():
        raise ValueError('public prompt event set differs from common phrase input')
    pending = {}
    for message in messages[2:]:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            if function.get('name') == 'edit_plan':
                args = function['arguments']
                args = json.loads(args) if isinstance(args, str) else args
                pending[call['id']] = args
        if message.get('role') == 'tool' and message.get('tool_call_id') in pending:
            arguments = pending.pop(message['tool_call_id'])
            payload = json.loads(message['content'])
            receipt = payload.get('result') or {}
            if payload.get('ok') and receipt.get('valid') is True:
                if arguments and 'jianzi_rows' not in arguments:
                    raise ValueError('accepted edit has unknown argument schema')
                edits = arguments.get('jianzi_rows', [])  # A successful empty preview commits no changes.
                for index, text in edits:
                    if index not in mapping or not isinstance(text, str):
                        raise ValueError('edit outside the current phrase')
                    state[index] = text
    if pending:
        raise ValueError('edit with no actual tool receipt')
    return [[i, state[i]] for i in mapping]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'train/data/guqin_direct_sft')
    parser.add_argument('--replace-partial', action='store_true', help='Rebuild only an incomplete direct export from this builder')
    args = parser.parse_args()
    source = ROOT / 'ABC_J/agent_training/single_stage_audited'
    sft = source / 'sft_export'
    source_data = sft / 'guqin_agent_train.jsonl'
    assert sha(source_data) == '984779ea61a9c582328518bf8d4e023826bf31b8c705f927c1dfbd9d97b00f4a'
    metadata = read(sft / 'guqin_agent_train_metadata.jsonl')
    public = read(source / 'teacher_final_redacted/messages_train.jsonl')
    by_id = {r['sample_id']: r for r in public}
    assert len(metadata) == len(by_id) == len(public) == 3976
    common = {r['trajectory_id']: r for r in read(ROOT / 'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl')}
    # Extract only accepted final plans from the private audit. Never export private reasoning.
    plans = {}
    audit_path = source / 'teacher_final_redacted/teacher_trajectory_audit.jsonl'
    with audit_path.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['sample_id'] in plans:
                raise ValueError('duplicate final audit')
            plans[row['sample_id']] = row['teacher_private']['accepted_plan']
    assert set(plans) == set(by_id)
    train = {r['score_key'] for r in metadata}
    tests = read(ROOT / 'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl')
    modern = read(ROOT / 'train/eval_inputs/modern_audited.jsonl')
    test_scores = {r['score_key'] for r in tests}
    modern_scores = {r['score_key'] for r in modern}
    with (ROOT / 'ABC_J/results/agent_dataset_splits.csv').open(encoding='utf-8-sig') as handle:
        splits = {r['score_key']: r for r in csv.DictReader(handle)}
    assert len(train) == 153 and len(test_scores) == 48 and len(modern_scores) == 10
    assert not train & (test_scores | modern_scores)
    assert all(splits[s]['split'] == 'train' for s in train)
    assert not {splits[s]['leakage_group_id'] for s in train} & {splits[s]['leakage_group_id'] for s in test_scores}
    if args.output.exists():
        assert args.replace_partial and not (args.output / 'dataset_manifest.json').exists()
        assert {p.name for p in args.output.iterdir()} <= {'guqin_agent_train.jsonl', 'guqin_agent_train_metadata.jsonl'}
        assert all(r.get('messages', [{}])[0].get('content') == SYSTEM for r in read(args.output / 'guqin_agent_train.jsonl'))
    else:
        args.output.mkdir(parents=True)
    data_path = args.output / 'guqin_agent_train.jsonl'
    meta_path = args.output / 'guqin_agent_train_metadata.jsonl'
    with data_path.open('w', encoding='utf-8') as data, meta_path.open('w', encoding='utf-8') as meta:
        for n, original_meta in enumerate(metadata, 1):
            sid = original_meta['source_sample_id']
            row = by_id[sid]
            item = common[original_meta['source_trajectory_id']]
            answer = committed_public_rows(row['messages'], item)
            content = json.dumps({'jianzi_rows': answer}, ensure_ascii=False, separators=(',', ':'))
            _, parsed = parse_answer(content, item)
            accepted = {int(a['source_index']): str(a.get('jianzi_text') or a.get('text') or '') for a in plans[sid]['actions']}
            assert all(a['jianzi_text'] == accepted.get(a['source_index'], '') for a in parsed['actions']), sid
            current_sources = set(event_sources(item).values())
            assert not any(text for index, text in accepted.items() if index not in current_sources), sid
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': user_from_public_prompt(row['messages'][1]['content'])},
                        {'role': 'assistant', 'content': content}]
            assert not any(token in messages[1]['content'] for token in ('<tool_call>', ':warning:', ':hint:'))
            data.write(json.dumps({'messages': messages, 'tools': '[]'}, ensure_ascii=False) + '\n')
            m = dict(original_meta, agent_stage='direct', assistant_turn_count=1, tool_call_turn_count=0,
                     appended_terminal_assistant=False, training_trajectory_id=original_meta['source_trajectory_id'] + '-direct',
                     target_event_rows=len(answer), target_sha256=hashlib.sha256(content.encode()).hexdigest(),
                     public_input_sha256=hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest())
            meta.write(json.dumps(m, ensure_ascii=False) + '\n')
            if n % 500 == 0:
                print(f'{n}/3976 public committed artifacts verified', flush=True)
    info = json.loads((sft / 'dataset_info.json').read_text())
    # No function schema is provided to the tokenizer or model.
    info['guqin_agent_train']['columns'].pop('tools', None)
    (args.output / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    manifest = {'protocol': PROTOCOL, 'training_rows': 3976, 'train_scores': 153,
        'source_dataset_sha256': sha(source_data), 'source_public_sha256': sha(source / 'teacher_final_redacted/messages_train.jsonl'),
        'dataset_sha256': sha(data_path), 'metadata_sha256': sha(meta_path), 'target_source': 'same teacher committed final plans; verified against public accepted edit receipts',
        'public_context': 'same initial public musical context, tool-use directions and diagnostic suffixes removed',
        'supervision': 'one final JSON artifact, no reasoning, tool calls or tool feedback',
        'traditional_test_scores': 48, 'modern_test_scores': 10,
        'score_overlap': [], 'traditional_leakage_group_overlap': [],
        'modern_melody_group_independence': 'not established by score-ID separation',
        'inference': 'one generation per phrase; own generated prior history; no diagnostic repair'}
    (args.output / 'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
