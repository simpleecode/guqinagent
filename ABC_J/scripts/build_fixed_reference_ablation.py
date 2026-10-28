#!/usr/bin/env python3
"""Build isolated, deterministic single-stage reference-supervised tool traces."""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ABC_J/scripts'))
from generate_teacher_tool_trajectories import (  # noqa: E402
    RealToolRuntime, blank_plan_from_item, harmonic_region_at_phrase_start,
    public_system_for, public_tools_for, render_public_prompt,
)

SOURCE = ROOT / 'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl'
CANONICAL = ROOT / 'ABC_J/agent_training/single_stage_audited/teacher_final_redacted/messages_train.jsonl'


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def construct(item, historical):
    item = deepcopy(item)
    item['baseline_plan'] = blank_plan_from_item(item)
    item['harmonic_region_at_start'] = harmonic_region_at_phrase_start(item, historical)
    runtime = RealToolRuntime(item, historical, basic_fingering=True)
    notes = item['input']['notes_without_jianzi']
    event_indices = [int(n['event_index']) for n in notes if n.get('event_index') is not None
                     and not str(n.get('abc') or '').startswith(('-', 'z', 'x'))
                     and '休止' not in str(n.get('jianpu') or '')]
    if not event_indices:
        raise ValueError('no queryable pitch events')
    by_source = {int(n['index']): int(n['event_index']) for n in notes if n.get('event_index') is not None}
    edits = [[by_source[int(a['source_index'])], str(a.get('text') or '')]
             for a in item['reference_plan']['actions'] if int(a['source_index']) in by_source]
    if not edits or len({x[0] for x in edits}) != len(edits):
        raise ValueError('reference rows missing or duplicated')
    messages = [{'role': 'system', 'content': public_system_for('single_stage', basic=True)},
                {'role': 'user', 'content': render_public_prompt(item, 'single_stage', historical=historical)}]
    for number, name, arguments, reasoning in [
        (1, 'get_pitch_candidates', {'event_indices': event_indices, 'max_candidates': 6}, '先查询当前段音高对应的取音候选。'),
        (2, 'edit_plan', {'jianzi_rows': edits}, '结合当前段与取音候选，提交完整减字。'),
    ]:
        result = runtime.invoke(name, arguments)
        if not result.get('ok') or (name == 'edit_plan' and not result.get('result', {}).get('valid')):
            raise ValueError(f'{name} rejected: {result}')
        call_id = f'call_{number:02d}_01'
        messages.append({'role': 'assistant', 'content': reasoning, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}]})
        messages.append({'role': 'tool', 'tool_call_id': call_id, 'name': name, 'content': json.dumps(result, ensure_ascii=False)})
    messages.append({'role': 'assistant', 'content': '当前段减字已提交。'})
    sample = f"{item['trajectory_id']}-single_stage-teacher-tools"
    return {'schema_version': 'agent-messages-2.1', 'sample_id': sample, 'agent_stage': 'single_stage',
            'generation_mode': 'fixed_reference_real_tools_no_api', 'tools': public_tools_for('single_stage', basic=True),
            'messages': messages, 'provenance': {'source_trajectory_id': item['trajectory_id'], 'score_key': item['score_key'], 'split': item['split']},
            'termination': {'kind': 'accepted_edit_plan'}, 'verification': {'tools_really_executed': True}}, runtime.calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = read(SOURCE)
    with CANONICAL.open(encoding='utf-8') as stream:
        canonical_ids = [(json.loads(line).get('provenance') or {})['source_trajectory_id'] for line in stream if line.strip()]
    ids = set(canonical_ids)
    selected = [r for r in source if r['trajectory_id'] in ids]
    if len(selected) != len(ids) or len(ids) != len(canonical_ids):
        raise ValueError('source/canonical ID mismatch')
    history = {(r['score_key'], r['phrase_id']): r for r in source}
    if args.limit is not None:
        selected = selected[:args.limit]
    failures = []
    with (args.output / 'messages_train.jsonl').open('x', encoding='utf-8') as public, (args.output / 'tool_audit.jsonl').open('x', encoding='utf-8') as audit:
        for i, row in enumerate(selected, 1):
            try:
                record, calls = construct(row, history)
                public.write(json.dumps(record, ensure_ascii=False) + '\n')
                audit.write(json.dumps({'sample_id': record['sample_id'], 'calls': calls}, ensure_ascii=False) + '\n')
            except Exception as error:
                failures.append({'id': row['trajectory_id'], 'error': f'{type(error).__name__}: {error}'})
            if i % 50 == 0:
                print(f'{i}/{len(selected)} failures={len(failures)}', flush=True)
    report = {'expected': len(selected), 'failures': failures, 'source_sha256': digest(SOURCE), 'canonical_sha256': digest(CANONICAL)}
    (args.output / 'construction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
