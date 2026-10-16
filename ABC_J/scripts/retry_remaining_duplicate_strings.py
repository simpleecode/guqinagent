"""Rerun only remaining final duplicate-string phrases and report improvement."""
import json
from pathlib import Path
import subprocess
import os
import sys

from prepare_duplicate_string_repair import hits, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
OUT = BASE / 'duplicate_string_retry'


def scan(path):
    affected = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            for action in row['teacher_private']['accepted_plan']['actions']:
                text = action.get('jianzi_text') or ''
                if hits(text):
                    pid = row['sample_id'].removesuffix('-teacher-tools').rsplit('-', 1)[0]
                    affected.setdefault(pid, []).append({'sample_id': row['sample_id'],
                        'source_index': action['source_index'], 'text': text})
    return affected


def main():
    before = scan(BASE / 'teacher_train_final/teacher_trajectory_audit.jsonl')
    assert len(before) == 12, f'Remaining set changed: {len(before)}'
    source = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    report = {'affected_phrases': len(before), 'selection': 'either stage final accepted plan only',
              'source_sha256': sha(source),
              'messages_sha256': sha(BASE / 'teacher_train_final/messages_train.jsonl'), 'cases': before}
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'regenerated_raw').exists()
    (OUT / 'affected_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'trajectory_ids.txt').write_text(''.join(pid + '\n' for pid in sorted(before)))
    print(f'Selected {len(before)} remaining phrases, {len({pid.split("-p")[0] for pid in before})} scores; both stages.', flush=True)
    subprocess.run(['bash', str(ROOT / 'ABC_J/scripts/run_duplicate_string_repair.sh')],
                   cwd=ROOT, env={**os.environ, 'REPAIR_NAME': 'duplicate_string_retry'}, check=True)
    raw = OUT / 'regenerated_raw'
    identities = []
    for line in (raw / 'messages_train.jsonl').open():
        row = json.loads(line)
        assert row.get('verification', {}).get('training_eligible')
        identities.append(row['sample_id'])
    expected = {f'{p}-{stage}-teacher-tools' for p in before
                for stage in ('fingering_agent', 'guqinization')}
    assert len(identities) == len(set(identities)) == 24 and set(identities) == expected
    after = scan(raw / 'teacher_trajectory_audit.jsonl')
    comparison = {'requested_phrases': 12, 'accepted_stage_rows': 24,
                  'before_final_duplicate_phrases': 12, 'after_final_duplicate_phrases': len(after),
                  'cleared_phrase_ids': sorted(set(before) - set(after)), 'remaining_cases': after,
                  'canonical_modified': False, 'redacted_or_merged': False}
    (OUT / 'comparison_report.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
