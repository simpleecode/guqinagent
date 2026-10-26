"""Promote corrected selection and retain only in-selection valid progress."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts.prepare_final_warning_rebuild import OUT, SOURCE, AUDIT, sha256


def read_rows(path):
    return [json.loads(l) for l in path.open() if l.strip()] if path.exists() else []


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def main():
    processes = subprocess.check_output(['ps', '-axo', 'pid,command'], text=True)
    for line in processes.splitlines():
        if ('run_teacher_batch_parallel.py' in line or 'generate_teacher_tool_trajectories.py' in line) and '.final_warning_rebuild' in line:
            raise SystemExit('Old generator is still alive; refusing migration')
    corrected = json.loads((OUT/'teacher_context_reaudit_report.json').read_text())
    original = json.loads((OUT/'selection_report.json').read_text())
    assert corrected['basis'].endswith('used by teacher generation')
    assert sha256(SOURCE) == original['source_sha256'] == corrected['source_sha256']
    assert sha256(AUDIT) == original['audit_sha256'] == corrected['audit_sha256']
    assert all(sha256(ROOT/f)==v for f,v in original['runtime_sha256'].items())
    ids = set(corrected['cases'])
    assert len(ids)==766 and corrected['warning_events']==1135
    sources = read_rows(SOURCE)
    shards = [[] for _ in range(7)]
    by_score = {}
    for r in sources:
        if r['trajectory_id'] in ids:
            by_score.setdefault(r['score_key'], []).append(r['trajectory_id'])
    for score, members in sorted(by_score.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        i = min(range(7), key=lambda j: (len(shards[j]), j))
        shards[i].extend(members)
    assert {s for shard in shards for s in shard} == ids
    trash = Path('/Volumes/F/.Trashes/501')
    assert trash.is_dir()
    backup = Path(tempfile.mkdtemp(prefix='guqin_superseded_warning_selection_', dir=trash))
    old_raw = OUT/'regenerated_raw'
    shutil.move(str(old_raw), backup/'regenerated_raw')
    for name in ('selection_report.json', 'trajectory_ids.txt', 'coordinator.log'):
        if (OUT/name).exists():
            shutil.move(str(OUT/name), backup/name)
    files = ('messages_train.jsonl','teacher_trajectory_audit.jsonl',
             'fingering_intermediates.jsonl','checkpoint.jsonl')
    retained = {name:{} for name in files}
    removed = 0
    def phrase(row):
        sid = row.get('trajectory_id') or (row.get('provenance') or {}).get('source_trajectory_id')
        if sid:
            return sid
        sid = row.get('sample_id','')
        for suffix in ('-fingering_agent-teacher-tools','-guqinization-teacher-tools'):
            if sid.endswith(suffix):
                return sid[:-len(suffix)]
        return None
    for name in files:
        for path in (backup/'regenerated_raw/.parallel_workers').glob(f'*/output/{name}'):
            for row in read_rows(path):
                sid = phrase(row)
                if sid not in ids:
                    removed += 1
                    continue
                key = row.get('sample_id') or row.get('trajectory_id')
                retained[name][key] = row
    for i, members in enumerate(shards):
        worker = old_raw/'.parallel_workers'/f'finalwarning_{i:02d}'
        output = worker/'output'
        output.mkdir(parents=True)
        (worker/'trajectory_ids.txt').write_text(''.join(s+'\n' for s in members))
        selected = set(members)
        for name in files:
            write_rows(output/name, [r for r in retained[name].values() if phrase(r) in selected])
    report = {**original, 'basis': corrected['basis'], 'selected_ids': sorted(ids),
              'selected_phrases': len(ids), 'selected_events': corrected['warning_events'],
              'selected_codes': corrected['by_code'], 'cases': corrected['cases'],
              'selected_phrases_by_code': {c:sum(any(w['code']==c for w in ws) for ws in corrected['cases'].values())
                                           for c in corrected['by_code']},
              'context_correction_applied': True}
    (OUT/'selection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (OUT/'trajectory_ids.txt').write_text(''.join(s+'\n' for s in sorted(ids)))
    summary = {'phrases':len(ids), 'stages_expected':2*len(ids), 'worker_sizes':list(map(len,shards)),
               'retained_records':{name:len(rows) for name,rows in retained.items()},
               'excluded_records':removed, 'recoverable_previous_build':str(backup)}
    (OUT/'selection_correction_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
