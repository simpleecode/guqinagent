"""Select accepted training phrases exposed as pitch mismatches by 绰 parsing."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import audit_jianpu_jianzi_pitch as pitch  # noqa: E402

BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
SOURCE = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
AUDIT = BASE / 'teacher_train_final/teacher_trajectory_audit.jsonl'
OUTPUT = BASE / 'chuo_pitch_repair'


def main() -> None:
    scores = defaultdict(lambda: {'notes': {}, 'ids': {}, 'actions': {}, 'metadata': None, 'open_midi': None})
    for line in SOURCE.open():
        record = json.loads(line)
        score_key = record['score_key']
        phrase_id = record['trajectory_id']
        entry = scores[score_key]
        entry['metadata'] = record['input']['metadata']
        entry['open_midi'] = record['input']['normalized_tuning']['open_midi']
        for note in record['input']['notes_without_jianzi']:
            index = note['index']
            entry['notes'][index] = note
            entry['ids'][index] = phrase_id

    for line in AUDIT.open():
        record = json.loads(line)
        if '-guqinization-' not in record['sample_id']:
            continue
        score_key = record['sample_id'].split('-', 1)[0]
        entry = scores.get(score_key)
        if entry is None:
            continue
        for action in (record['teacher_private'].get('accepted_plan') or {}).get('actions') or []:
            entry['actions'][action['source_index']] = action.get('jianzi_text') or ''

    cases = []
    candidate_count = 0
    for score_key, entry in scores.items():
        candidates = {}
        for index, value in entry['actions'].items():
            right = pitch.first_token(value, pitch.RIGHT_HAND_TECHNIQUES)
            if (right and '绰' in value and value.index('绰') < value.index(right)
                    and len(pitch.parse_string_numbers(value)) == 1
                    and pitch.parse_hui(value) is not None):
                candidates[index] = value
        if not candidates:
            continue
        candidate_count += len(candidates)
        notes = [{**entry['notes'][index], 'jianzi': entry['actions'].get(index, '')}
                 for index in sorted(entry['notes'])]
        report = pitch.audit({'metadata': entry['metadata'], 'open_midi': entry['open_midi'],
                              'notes': notes}, 50)
        for detail in report['details']:
            index = detail['index']
            if index in candidates and detail['status'] == 'mismatched':
                cases.append({'phrase_id': entry['ids'][index], 'score_key': score_key,
                              'source_index': index, 'jianzi': candidates[index],
                              'expected_midis': detail['expected_midis'],
                              'actual_midis': detail['actual_midis']})

    phrase_ids = sorted({case['phrase_id'] for case in cases})
    if len(phrase_ids) != 65 or len(cases) != 73:
        raise RuntimeError(f'Unexpected selection: {len(phrase_ids)} phrases, {len(cases)} notes')
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / 'trajectory_ids.txt').write_text(''.join(f'{phrase_id}\n' for phrase_id in phrase_ids))
    summary = {'affected_phrases': len(phrase_ids), 'affected_notes': len(cases),
               'candidate_actions': candidate_count, 'cases': cases}
    (OUTPUT / 'affected_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
