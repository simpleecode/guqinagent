"""Promote four verified recaptures; replace only two affected train scores.

Run only with the teacher coordinator/workers stopped. Old source/phrase
products and worker snapshots go to a recoverable, task-specific Trash folder.
"""
import hashlib
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / 'ABC_J/agent_training/two_stage_audited'
KEYS = {'S9j70QqA', 'SH8hkkfN', 'SMcokJse', 'SZ7pMCs3'}
TRAIN_KEYS = {'SMcokJse', 'SZ7pMCs3'}

def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    ps = subprocess.check_output(['ps', '-axo', 'command'], text=True)
    assert not any(('generate_teacher_tool_trajectories.py --input' in l or
                    'run_teacher_batch_parallel.py --input' in l) for l in ps.splitlines()), 'stop live teachers first'
    resume = len(sys.argv) == 3
    trash = Path(sys.argv[2]) if resume else Path(tempfile.mkdtemp(prefix='guqin_before_recapture_phrase_replace.', dir=Path.home()/'.Trash'))
    work = Path(sys.argv[1]) if resume else Path(tempfile.mkdtemp(prefix='recapture-phrase-replace-', dir=METHOD))
    report = {'backup': str(trash), 'staging': str(work), 'source_keys': sorted(KEYS)}
    old_raw = METHOD/'inferred_all/inferred_trajectories_train.jsonl'
    old_selected = METHOD/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    old_counts = Counter(r['score_key'] for r in rows(old_raw))
    old_selected_counts = Counter(r['score_key'] for r in rows(old_selected))
    for key in ([] if resume else sorted(KEYS)):
        source = ROOT/'ABC_J/recapture_duplicate_titles'/key
        raw = json.loads((source/'raw_data.json').read_text())
        readable = json.loads((source/'jianpu_jianzi_readable.json').read_text())
        assert len(readable['notes']) == raw['metadata']['notes_length_metadata']
        assert raw['metadata']['score_key'] == key
        replacement = work/'final'/key
        shutil.copytree(source, replacement)
        (replacement/'out').mkdir(exist_ok=True)
        for name in ('raw_data.json', 'data.json'):
            shutil.move(replacement/name, replacement/'out'/name)
        canonical = ROOT/'ABC_J/final'/key
        backup = trash/'final'/key
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(canonical, backup)
        shutil.move(replacement, canonical)
    with (ROOT/'ABC_J/results/dataset_split_groups.csv').open(encoding='utf-8-sig') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames
        manifest_rows=[r for r in reader if r['score_key'] in TRAIN_KEYS]
    assert len(manifest_rows)==2 and all(r['split']=='train' for r in manifest_rows)
    with (work/'affected_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(manifest_rows)
    with (work/'infer.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/infer_agent_trajectories.py',
            '--manifest', str(work/'affected_manifest.csv'),
            '--split-manifest', 'ABC_J/results/agent_dataset_splits.csv',
            '--mapped-dir', 'ABC_J/final', '--output-dir', str(work/'affected_inferred'),
            '--score-key', 'SMcokJse', '--score-key', 'SZ7pMCs3'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    new_affected = list(rows(work/'affected_inferred/inferred_trajectories_train.jsonl'))
    assert {r['score_key'] for r in new_affected} == TRAIN_KEYS
    affected_by_key = {key:[r for r in new_affected if r['score_key']==key] for key in TRAIN_KEYS}
    def combined():
        inserted = set()
        for r in rows(old_raw):
            key = r['score_key']
            if key in TRAIN_KEYS:
                if key not in inserted:
                    yield from affected_by_key[key]
                    inserted.add(key)
            else:
                yield r
        assert inserted == TRAIN_KEYS
    write(work/'inferred_all/inferred_trajectories_train.jsonl', combined())
    shutil.copy2(METHOD/'inferred_all/inferred_trajectories_test.jsonl', work/'inferred_all/inferred_trajectories_test.jsonl')
    with (work/'selection.log').open('w') as log:
        subprocess.run([sys.executable, 'ABC_J/scripts/export_pitch_eligible_trajectories.py',
            '--inferred-dir', str(work/'inferred_all'), '--splits','train','test',
            '--unfiltered-split','test', '--materialize-selected-inferred',
            '--output-dir', str(work/'pitch_eligible_train_test')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    selected = work/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    # Non-affected selected inputs must be exactly reusable, not just same IDs.
    before = {r['trajectory_id']:r for r in rows(old_selected) if r['score_key'] not in TRAIN_KEYS}
    after = {r['trajectory_id']:r for r in rows(selected) if r['score_key'] not in TRAIN_KEYS}
    assert before == after, 'non-affected selected inputs changed; do not resume blindly'
    new_counts = Counter(r['score_key'] for r in rows(work/'inferred_all/inferred_trajectories_train.jsonl'))
    new_selected_counts = Counter(r['score_key'] for r in rows(selected))
    test_keys={r['score_key'] for r in rows(work/'inferred_all/inferred_trajectories_test.jsonl')}
    assert not test_keys & set(new_selected_counts)
    report.update(old_raw_total=sum(old_counts.values()), new_raw_total=sum(new_counts.values()),
        old_selected_total=sum(old_selected_counts.values()), new_selected_total=sum(new_selected_counts.values()),
        unaffected_selected_reused=len(before), affected_counts={key:{'raw_before':old_counts[key],
        'raw_after':new_counts[key],'selected_before':old_selected_counts[key],
        'selected_after':new_selected_counts[key]} for key in sorted(TRAIN_KEYS)})
    canonical_raw = METHOD/'inferred_all'
    backup_raw = trash/'inferred_all'
    shutil.copytree(canonical_raw, backup_raw)
    shutil.move(work/'inferred_all/inferred_trajectories_train.jsonl', old_raw)
    for old_manifest in canonical_raw.glob('*.json'):
        if old_manifest.name != 'input_compatibility_status.json':
            shutil.move(old_manifest, backup_raw/old_manifest.name)
    (canonical_raw/'input_compatibility_status.json').write_text(json.dumps({'status':'recapture_updated',
        'report':str(METHOD/'recapture_phrase_replacement_report.json'),
        'test_unchanged':True,'affected_scores':sorted(TRAIN_KEYS)},ensure_ascii=False,indent=2)+'\n')
    selection_manifest = work/'pitch_eligible_train_test/selection_manifest.json'
    manifest=json.loads(selection_manifest.read_text())
    manifest['inputs']['inferred_dir']=str(canonical_raw)
    selection_manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.move(METHOD/'pitch_eligible_train_test', trash/'pitch_eligible_train_test')
    shutil.move(work/'pitch_eligible_train_test', METHOD/'pitch_eligible_train_test')
    # Preserve only complete paired records on byte-identical inputs.
    teacher=METHOD/'teacher_latest_runtime_raw'
    checkpoints={r['trajectory_id']:r for r in rows(teacher/'checkpoint.jsonl')}
    stages={}
    for r in rows(teacher/'messages_train.jsonl'):
        source=r['sample_id'].split('-fingering_agent-')[0].split('-guqinization-')[0]
        stages.setdefault(source,set()).add(r['agent_stage'])
    valid_ids={r['trajectory_id'] for r in rows(METHOD/'pitch_eligible_train_test/inferred_trajectories_train.jsonl')}
    keep={tid for tid,r in checkpoints.items() if r['status']=='completed' and tid in valid_ids
          and tid.split('-p')[0] not in TRAIN_KEYS and stages.get(tid)=={'fingering_agent','guqinization'}}
    def source_id(r):
        return r.get('trajectory_id') or str(r.get('sample_id','')).split('-fingering_agent-')[0].split('-guqinization-')[0]
    for name in ('messages_train.jsonl','teacher_trajectory_audit.jsonl','fingering_intermediates.jsonl','teacher_rejected_io.jsonl','checkpoint.jsonl'):
        p=teacher/name
        if not p.exists():continue
        dest=trash/'teacher_root'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        if name=='checkpoint.jsonl':write(p,[checkpoints[tid] for tid in sorted(keep)])
        else:
            temp=work/name;write(temp,(r for r in rows(p) if source_id(r) in keep));shutil.move(temp,p)
    shutil.move(teacher/'.parallel_workers', trash/'teacher_workers')
    for name in ('generation_report.json','run.log'):
        if (teacher/name).exists():shutil.move(teacher/name,trash/('old_'+name))
    report.update(teacher_complete_phrases_retained=len(keep), teacher_stages_retained=2*len(keep),
        selected_input_sha256=hashlib.sha256((METHOD/'pitch_eligible_train_test/inferred_trajectories_train.jsonl').read_bytes()).hexdigest(),
        status='complete', test_inputs_unchanged=True)
    (METHOD/'recapture_phrase_replacement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
