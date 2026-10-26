"""Compare teacher-context replay with generated-output-history replay."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts.prepare_final_warning_rebuild import BASE, SOURCE, AUDIT, OUT, sha256
from agents.ToolRuntime.runtime import validate_jianzi_only, harmonic_region_at_phrase_start

PLANS = {}
SAVED = {}


def replay_score(rows):
    reference_history = {(r['score_key'], r['phrase_id']): r for r in rows}
    cases = {}
    state_changes = []
    accepted_history = {}
    for row in sorted(rows, key=lambda r: int(r['input']['event_range']['start'])):
        sid = row['trajectory_id']
        item = deepcopy(row)
        item['baseline_plan'] = deepcopy(PLANS[sid])
        teacher_flag = harmonic_region_at_phrase_start(item, reference_history)
        accepted_flag = harmonic_region_at_phrase_start(item, accepted_history)
        item['harmonic_region_at_start'] = teacher_flag
        _, report = validate_jianzi_only(item, [], historical=reference_history)
        if teacher_flag != accepted_flag:
            state_changes.append({'sample_id': sid, 'teacher_context_harmonic': teacher_flag,
                                  'accepted_history_harmonic': accepted_flag})
        if report['warnings']:
            cases[sid] = report['warnings']
        item['reference_plan'] = PLANS[sid]
        accepted_history[(row['score_key'], row['phrase_id'])] = item
    return len(rows), cases, state_changes


def main():
    source = [json.loads(l) for l in SOURCE.open() if l.strip()]
    for line in AUDIT.open():
        if not line.strip():
            continue
        row = json.loads(line)
        suffix = '-guqinization-teacher-tools'
        if row['sample_id'].endswith(suffix):
            sid = row['sample_id'][:-len(suffix)]
            p = row['teacher_private']
            PLANS[sid] = p['accepted_plan']
            SAVED[sid] = p.get('jianzi_quality_report', {}).get('warnings') or []
    assert len(PLANS) == len(source) == 3976
    scores = {}
    for row in source:
        scores.setdefault(row['score_key'], []).append(row)
    cases = {}
    state_changes = []
    done = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        for count, found, changed in pool.map(replay_score, scores.values()):
            done += count
            cases.update(found)
            state_changes.extend(changed)
            print(f'Teacher-context replay {done}/3976', flush=True)
    previous = json.loads((OUT/'selection_report.json').read_text())
    keyset = lambda cs, code: {(sid, w.get('source_index')) for sid, ws in cs.items()
                              for w in ws if w['code'] == code}
    old_pitch = keyset(previous['cases'], 'jianzi_pitch_mismatch')
    current_pitch = keyset(cases, 'jianzi_pitch_mismatch')
    saved_pitch = keyset(SAVED, 'jianzi_pitch_mismatch')
    counts = Counter(w['code'] for ws in cases.values() for w in ws)
    report = {'basis': 'current runtime with original reference history used by teacher generation',
              'source_sha256': sha256(SOURCE), 'audit_sha256': sha256(AUDIT),
              'phrases': len(cases), 'warning_events': sum(counts.values()),
              'by_code': dict(counts.most_common()),
              'pitch_phrases': sum(any(w['code']=='jianzi_pitch_mismatch' for w in ws) for ws in cases.values()),
              'previous_pitch_events': len(old_pitch),
              'previous_pitch_disappeared_with_teacher_context': len(old_pitch-current_pitch),
              'pitch_retained_at_same_positions': len(old_pitch & current_pitch),
              'pitch_new_vs_previous_context': len(current_pitch-old_pitch),
              'saved_generation_pitch_events': len(saved_pitch),
              'current_pitch_also_in_saved_generation': len(current_pitch & saved_pitch),
              'current_pitch_not_in_saved_generation': len(current_pitch-saved_pitch),
              'harmonic_state_changed_phrases': state_changes, 'cases': cases}
    (OUT/'teacher_context_reaudit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','harmonic_state_changed_phrases'}},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
