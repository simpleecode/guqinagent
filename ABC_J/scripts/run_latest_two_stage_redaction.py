"""Dependency-aware, bounded-memory GLM redaction and final dataset promotion."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'ABC_J/agent_training/two_stage_audited'
RAW=BASE/'teacher_latest_runtime_raw'
WORK=BASE/'latest_runtime_redaction'


def rows(path):
    for line in path.open():
        if line.strip():yield json.loads(line)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024):digest.update(data)
    return digest.hexdigest()


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dependency-pid',type=int)
    parser.add_argument('--resolution-report',type=Path)
    args=parser.parse_args()
    if args.dependency_pid:
        print(f'Waiting for missing-stage completion PID {args.dependency_pid}; no redaction API calls yet.',flush=True)
    while args.dependency_pid:
        p=subprocess.run(['ps','-p',str(args.dependency_pid),'-o','command='],capture_output=True,text=True)
        if p.returncode or not p.stdout.strip():break
        assert 'run_missing_teacher_completion.py' in p.stdout,'Dependency PID reused'
        time.sleep(5)
    completion=json.loads((args.resolution_report or BASE/'missing_stage_completion/replacement_report.json').read_text())
    assert not completion['pending_phrases'],'Completion has unresolved phrases; refusing incomplete training data'
    source=BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    source_ids={r['trajectory_id'] for r in rows(source)}
    assert len(source_ids)==3976
    excluded=set(completion.get('excluded_phrase_ids') or [])
    assert excluded<=source_ids
    source_ids-=excluded
    expected={f'{sid}-{stage}-teacher-tools' for sid in source_ids for stage in ['fingering_agent','guqinization']}
    ids=[];routing={}
    for r in rows(RAW/'messages_train.jsonl'):
        if (r.get('provenance') or {}).get('source_trajectory_id') in excluded:continue
        assert r['sample_id'] not in routing
        routing[r['sample_id']]=len(ids)%7;ids.append(r['sample_id'])
    assert set(ids)==expected,'Raw accepted messages incomplete or contain unexpected IDs'
    hashes={name:sha(RAW/name) for name in ['messages_train.jsonl','teacher_trajectory_audit.jsonl']}
    WORK.mkdir(exist_ok=True)
    inputs=WORK/'inputs';inputs.mkdir(exist_ok=True)
    assert not any(inputs.iterdir()),'Input shards already exist; inspect before resuming'
    for i in range(7):(inputs/f'worker_{i}').mkdir()
    for name in ['messages_train.jsonl','teacher_trajectory_audit.jsonl']:
        handles=[(inputs/f'worker_{i}'/name).open('w') for i in range(7)]
        seen=set()
        try:
            for line in (RAW/name).open():
                r=json.loads(line);sid=r['sample_id']
                if sid.removesuffix('-teacher-tools').rsplit('-',1)[0] in excluded:continue
                assert sid in routing and sid not in seen
                seen.add(sid);handles[routing[sid]].write(line)
        finally:
            for f in handles:f.close()
        assert seen==expected
    (WORK/'source_manifest.json').write_text(json.dumps({'rows':len(ids),'phrases':len(source_ids),'sha256':hashes,'excluded_phrase_ids':sorted(excluded)},indent=2)+'\n')
    processes=[]
    for i in range(7):
        out=WORK/f'worker_{i}'
        log=(WORK/f'worker_{i}.log').open('ab')
        p=subprocess.Popen([sys.executable,str(ROOT/'ABC_J/scripts/redact_teacher_reasoning.py'),
            '--input-dir',str(inputs/f'worker_{i}'),'--output-dir',str(out),'--model','glm-5.3'],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        processes.append((i,p,log));print(f'started redaction worker {i}: {p.pid}',flush=True)
    statuses=[]
    for i,p,log in processes:
        code=p.wait();log.close();statuses.append(code)
        print(f'redaction worker {i} exited {code}',flush=True)
    assert not any(statuses),'A redaction worker failed; no final dataset promotion'
    assert all(sha(RAW/name)==digest for name,digest in hashes.items()),'Raw source changed during redaction'
    candidate=WORK/'merged_candidate';assert not candidate.exists();candidate.mkdir()
    database=WORK/'merge.sqlite'
    assert not database.exists()
    db=sqlite3.connect(database)
    db.execute('PRAGMA cache_size=-8192')
    db.execute('CREATE TABLE records (kind TEXT,id TEXT,payload TEXT,PRIMARY KEY(kind,id))')
    reports=[]
    with (candidate/'reasoning_redaction_audit.jsonl').open('w') as changes:
        for i in range(7):
            worker=WORK/f'worker_{i}'
            report=json.loads((worker/'reasoning_redaction_report.json').read_text())
            assert not report.get('failures') and report.get('failed',0)==0,report
            reports.append(report)
            for kind,name in [('public','messages_train.jsonl'),('private','teacher_trajectory_audit.jsonl')]:
                for line in (worker/name).open():
                    r=json.loads(line);assert r['sample_id'] in expected
                    db.execute('INSERT INTO records VALUES (?,?,?)',(kind,r['sample_id'],line))
            path=worker/'reasoning_redaction_audit.jsonl'
            if path.exists():
                with path.open() as f:shutil.copyfileobj(f,changes)
            db.commit()
    for kind in ['public','private']:
        assert db.execute('SELECT count(*) FROM records WHERE kind=?',(kind,)).fetchone()[0]==len(expected)
    with (candidate/'messages_train.jsonl').open('w') as public,(candidate/'teacher_trajectory_audit.jsonl').open('w') as private:
        for original in rows(RAW/'messages_train.jsonl'):
            sid=original['sample_id']
            if (original.get('provenance') or {}).get('source_trajectory_id') in excluded:continue
            line=db.execute('SELECT payload FROM records WHERE kind=? AND id=?',('public',sid)).fetchone()[0]
            rewritten=json.loads(line)
            assert rewritten['tools']==original['tools'] and rewritten['provenance']==original['provenance']
            assert len(rewritten['messages'])==len(original['messages'])
            for before,after in zip(original['messages'],rewritten['messages']):
                before=deepcopy(before);after=deepcopy(after)
                if before.get('role')=='assistant':before.pop('content',None);after.pop('content',None)
                assert before==after,(sid,'non-reasoning message fields changed')
            public.write(line)
            private.write(db.execute('SELECT payload FROM records WHERE kind=? AND id=?',('private',sid)).fetchone()[0])
    db.close();database.unlink()
    report={'schema_version':'reasoning-redaction-merged-1.0','requested':len(ids),'written':len(ids),
            'failed':0,'public_private_ids_match':True,'source_sha256':hashes,'workers':reports,'excluded_phrase_ids':sorted(excluded)}
    (candidate/'reasoning_redaction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    run(ROOT/'scripts/validate_teacher_agent_messages.py','--input-dir',candidate)
    export=ROOT/'train/data/guqin_agent_sft_two_stage_latest_runtime'
    assert not export.exists(),'Export already exists; refusing overwrite'
    run(ROOT/'train/scripts/export_sft_dataset.py','--input-dir',candidate,'--output-dir',export)
    final=BASE/'teacher_train_final'
    if final.exists():
        backup=Path(tempfile.mkdtemp(prefix='guqin-before-latest-final-',dir='/Users/lijiawei/.Trash'))
        shutil.move(str(final),str(backup/'teacher_train_final'))
        report['previous_final_backup']=str(backup)
    candidate.rename(final)
    report['final_directory']=str(final);report['sft_export']=str(export)
    (WORK/'completion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'Redaction, validation, export and promotion complete: {final}',flush=True)


if __name__=='__main__':main()
