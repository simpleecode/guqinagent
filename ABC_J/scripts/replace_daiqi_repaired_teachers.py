"""Replace complete repaired pairs only; stream and preserve unrelated records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True)
p.add_argument('--repair',type=Path,required=True);p.add_argument('--ids',type=Path,required=True)
p.add_argument('--report',type=Path,required=True)
a=p.parse_args();wanted=set(a.ids.read_text().split())

def rows(path):
    if not path.exists():return
    for line in path.open():
        if line.strip():yield json.loads(line)

public={r['sample_id']:r for r in rows(a.repair/'messages_train.jsonl')}
private={r['sample_id']:r for r in rows(a.repair/'teacher_trajectory_audit.jsonl')}
complete=set()
for phrase in wanted:
    ids=[f'{phrase}-{stage}-teacher-tools' for stage in ['fingering_agent','guqinization']]
    if all(sid in public and sid in private
           and public[sid].get('verification',{}).get('training_eligible')
           and private[sid].get('verification',{}).get('replay_passed')
           for sid in ids):complete.add(phrase)
assert complete,'No complete verified repaired pairs; originals unchanged'
intermediates={r['trajectory_id']:r for r in rows(a.repair/'fingering_intermediates.jsonl')}
assert complete<=intermediates.keys()
trash_root=Path('/Volumes/F/.Trashes')/str(os.getuid())
assert trash_root.is_dir() and trash_root.stat().st_dev==a.canonical.stat().st_dev
backup=Path(tempfile.mkdtemp(prefix='guqin-before-daiqi-replacement-',dir=trash_root))
staging=Path(tempfile.mkdtemp(prefix='.daiqi-replacement-',dir=a.canonical.parent))
report={'replaced_phrases':len(complete),'pending_phrases':sorted(wanted-complete),
        'backup':str(backup),'canonical':str(a.canonical),'files':{}}
for filename,key,new in [
    ('messages_train.jsonl','sample_id',public),
    ('teacher_trajectory_audit.jsonl','sample_id',private),
    ('fingering_intermediates.jsonl','trajectory_id',intermediates),
]:
    existing_ids=set();changed=0;unchanged=hashlib.sha256();verified=hashlib.sha256()
    def phrase_of(identity):
        return identity if key=='trajectory_id' else identity.removesuffix('-teacher-tools').rsplit('-',1)[0]
    target=staging/filename
    with target.open('w') as out:
        for line in (a.canonical/filename).open():
            r=json.loads(line);identity=r[key]
            assert identity not in existing_ids
            existing_ids.add(identity)
            if phrase_of(identity) in complete:
                assert identity in new
                out.write(json.dumps(new[identity],ensure_ascii=False)+'\n');changed+=1
            else:
                unchanged.update(line.encode());out.write(line)
        for identity,r in new.items():
            if identity not in existing_ids and phrase_of(identity) in complete:
                out.write(json.dumps(r,ensure_ascii=False)+'\n');changed+=1
    with target.open() as f:
        for line in f:
            r=json.loads(line)
            if phrase_of(r[key]) not in complete:verified.update(line.encode())
    assert unchanged.digest()==verified.digest(),'Unrelated records changed'
    report['files'][filename]={'changed_or_added':changed,'unrelated_sha256':unchanged.hexdigest()}
# Validate all output before making any canonical mutation.
for filename in report['files']:
    shutil.copy2(a.canonical/filename,backup/filename)
for filename in report['files']:
    os.replace(staging/filename,a.canonical/filename)
shutil.rmtree(staging)
with (a.canonical/'checkpoint.jsonl').open('a') as out:
    for r in rows(a.repair/'checkpoint.jsonl'):
        if r.get('trajectory_id') in complete:out.write(json.dumps(r,ensure_ascii=False)+'\n')
report['archived_stale_statistics']=[]
for filename in ['token_distribution_raw.json','over8192_raw.tsv','problem_statistics_raw.json']:
    path=a.canonical/filename
    if path.exists():
        shutil.move(str(path),str(backup/filename))
        report['archived_stale_statistics'].append(filename)
a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
