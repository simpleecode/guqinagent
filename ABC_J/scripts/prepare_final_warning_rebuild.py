"""Select every final teacher plan still warning under the current runtime.

Replay accepted final plans, not patches against a possibly different baseline.
History uses the reference context actually supplied during teacher generation.
Generated-output history is a different, end-to-end evaluation condition.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import harmonic_region_at_phrase_start, validate_jianzi_only

BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
SOURCE = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
AUDIT = BASE / 'teacher_train_final/teacher_trajectory_audit.jsonl'
OUT = BASE / '.final_warning_rebuild'
RUNTIME_FILES = [
    'agents/ToolRuntime/runtime.py', 'agents/ToolRuntime/warning_checks.py',
    'agents/ToolRuntime/warning_registry.py', 'agents/ToolRuntime/warning_rendering.py',
    'scripts/audit_jianpu_jianzi_pitch.py',
    'ABC_J/scripts/generate_teacher_tool_trajectories.py',
    'agents/abc_to_jianzipu/teacher_trajectory.py',
]
PLANS = {}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def replay_score(rows):
    history = {(r['score_key'], r['phrase_id']): r for r in rows}
    cases = {}
    for row in sorted(rows, key=lambda r: int(r['input']['event_range']['start'])):
        sid = row['trajectory_id']
        item = deepcopy(row)
        item['baseline_plan'] = deepcopy(PLANS[sid])
        item['harmonic_region_at_start'] = harmonic_region_at_phrase_start(item, history)
        valid, report = validate_jianzi_only(item, [], historical=history)
        if not valid:
            raise ValueError(f'Invalid accepted final plan: {sid}: {report.get("problems")}')
        mapping = {int(n['index']): n.get('event_index')
                   for n in item['input']['notes_without_jianzi']}
        warnings = [dict(w, event_index=mapping.get(w.get('source_index')))
                    for w in report.get('warnings') or []]
        if warnings:
            cases[sid] = warnings
    return len(rows), cases


def main():
    if (OUT / 'selection_report.json').exists() or (OUT / 'regenerated_raw').exists():
        raise SystemExit('Selection already exists; do not overwrite an active build.')
    manifest = {f: sha256(ROOT / f) for f in RUNTIME_FILES}
    source_sha = sha256(SOURCE)
    audit_sha = sha256(AUDIT)
    source = [json.loads(l) for l in SOURCE.open() if l.strip()]
    expected = {r['trajectory_id'] for r in source}
    assert len(source) == len(expected) == 3976
    seen = set()
    for line in AUDIT.open():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row['sample_id']
        assert sid not in seen, sid
        seen.add(sid)
        suffix = '-guqinization-teacher-tools'
        if sid.endswith(suffix):
            PLANS[sid[:-len(suffix)]] = row['teacher_private']['accepted_plan']
    assert len(seen) == 7952 and set(PLANS) == expected
    scores = defaultdict(list)
    for row in source:
        scores[row['score_key']].append(row)
    cases = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=4,
                             mp_context=multiprocessing.get_context('fork')) as pool:
        for count, found in pool.map(replay_score, scores.values()):
            completed += count
            cases.update(found)
            print(f'Replayed {completed}/3976; warning phrases={len(cases)}', flush=True)
    assert source_sha == sha256(SOURCE) and audit_sha == sha256(AUDIT)
    assert manifest == {f: sha256(ROOT / f) for f in RUNTIME_FILES}
    counts = Counter(w['code'] for ws in cases.values() for w in ws)
    phrases = Counter(code for ws in cases.values() for code in {w['code'] for w in ws})
    ids = sorted(cases)
    report = {
        'basis': 'current runtime; accepted final plans; original teacher reference history',
        'source_sha256': source_sha, 'audit_sha256': audit_sha,
        'runtime_sha256': manifest, 'source_phrases': len(source),
        'selected_phrases': len(ids), 'selected_events': sum(counts.values()),
        'selected_codes': dict(counts.most_common()),
        'selected_phrases_by_code': dict(phrases.most_common()),
        'annotation_aligned_warnings_excluded': False,
        'selected_ids': ids, 'cases': cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'selection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'trajectory_ids.txt').write_text(''.join(sid+'\n' for sid in ids))
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','selected_ids'}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
