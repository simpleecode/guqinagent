#!/usr/bin/env python3
"""Audit fixed-reference ablation traces against source and tool receipts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'ABC_J/agent_training/single_stage_fixed_reference_ablation'
SOURCE = ROOT / 'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl'
CANONICAL = ROOT / 'ABC_J/agent_training/single_stage_audited/teacher_final_redacted/messages_train.jsonl'
TEST = ROOT / 'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl'


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    source = {r['trajectory_id']: r for r in rows(SOURCE)}
    canonical = {(r.get('provenance') or {})['source_trajectory_id'] for r in rows(CANONICAL)}
    test = {r['score_key'] for r in rows(TEST)}
    seen, counts, issues = set(), Counter(), []
    for row, audit in zip(rows(DATA / 'messages_train.jsonl'), rows(DATA / 'tool_audit.jsonl'), strict=True):
        source_id = row['provenance']['source_trajectory_id']
        if source_id in seen:
            issues.append([source_id, 'duplicate_source_id'])
        seen.add(source_id)
        item = source[source_id]
        if item['score_key'] in test or item.get('split') != 'train':
            issues.append([source_id, 'test_or_split_overlap'])
        if row['sample_id'] != audit['sample_id']:
            issues.append([source_id, 'audit_id_mismatch'])
        calls = audit['calls']
        if [c['name'] for c in calls] != ['get_pitch_candidates', 'edit_plan']:
            issues.append([source_id, 'wrong_call_order'])
            continue
        if not all(c['result'].get('ok') for c in calls):
            issues.append([source_id, 'tool_error'])
        actions = calls[1]['result'].get('result', {}).get('preview_actions') or []
        actual = {int(a['source_index']): str(a.get('jianzi_text') or '') for a in actions}
        notes = {int(n['index']): int(n['event_index']) for n in item['input']['notes_without_jianzi'] if n.get('event_index') is not None}
        expected = {int(a['source_index']): str(a.get('text') or '') for a in item['reference_plan']['actions'] if int(a['source_index']) in notes}
        mismatches = [k for k, v in expected.items() if actual.get(k) != v]
        if mismatches:
            issues.append([source_id, 'preview_not_reference', mismatches[:12]])
        for w in calls[1]['result'].get('result', {}).get('warnings') or []:
            counts['warning:' + str(w.get('code'))] += 1
        counts['rows'] += 1
    if seen != canonical:
        issues.append(['dataset', 'source_id_mismatch', len(canonical - seen), len(seen - canonical)])
    report = {'valid': not issues, 'rows': counts['rows'], 'warning_counts': dict(counts), 'issues': issues,
              'public_sha256': hashlib.sha256((DATA / 'messages_train.jsonl').read_bytes()).hexdigest()}
    (DATA / 'reference_validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'valid': report['valid'], 'rows': counts['rows'], 'issues': len(issues), 'first_issues': issues[:5]}, ensure_ascii=False))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
