"""Install the user-selected Youlan capture as a held-out replacement."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.infer_agent_trajectories import infer_score
from train.scripts.build_text_protocol_eval_set import public_runtime_item, fingerprint

def read(path):
    return [json.loads(l) for l in path.open() if l.strip()]

def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))

def refresh_inventories():
    data=read(ROOT/'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl')
    inv=ROOT/'ABC_J/results/split_inventories'
    for name, values in [('test_score_ids.txt',sorted({r['score_key'] for r in data})),
                         ('test_phrase_ids.txt',sorted(r['sample_id'] for r in data))]:
        (inv/name).write_text(''.join(v+'\n' for v in values))

def main():
    key = 'ST4GAejh'
    source = ROOT/'ABC_J/test_replacements'/key
    black = set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
    group = 'LG0101'
    final = ROOT/'ABC_J/final'/key
    assert not final.exists(), 'Replacement already installed'
    # Raw evidence is copied unchanged; no original capture is rewritten.
    final.mkdir()
    (final/'out').mkdir()
    for name in ('raw_data.json', 'data.json'):
        shutil.copy2(source/name, final/'out'/name)
    for name in ('jianpu_jianzi_mapped.json','jianpu_jianzi_mapped.md','jianpu_jianzi_readable.json'):
        shutil.copy2(source/name, final/name)
    row = {'score_key':key, 'leakage_group_id':group, 'split':'test'}
    phrases = infer_score(row, 16, mapped_dir=ROOT/'ABC_J/final')
    assert phrases and all(p['split']=='test' for p in phrases)
    train = ROOT/'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    assert not any(r['score_key']==key or r.get('score_family_id')==group for r in read(train))
    train_sha = hashlib.sha256(train.read_bytes()).hexdigest()
    backup = Path(tempfile.mkdtemp(prefix='guqin_test_youlan_replaced_',dir='/Volumes/F/.Trashes/501'))
    def preserve(path):
        dest=backup/path.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
    method=ROOT/'ABC_J/agent_training/two_stage_audited'
    for path in (method/'inferred_all/inferred_trajectories_test.jsonl',
                 method/'pitch_eligible_train_test/inferred_trajectories_test.jsonl'):
        preserve(path)
        old=read(path)
        write(path,[r for r in old if r['score_key'] not in black and r['score_key']!=key]+phrases)
    target=ROOT/'train/eval_inputs_v2_text_protocol/test50'
    publics=[]; sealed=[]
    for p in phrases:
        out={'schema_version':'agent-eval-input-2.0','sample_id':p['trajectory_id'],
             'split':'test','score_key':key,'phrase_id':p['phrase_id'],'runtime_item':public_runtime_item(p)}
        out['input_sha256']=fingerprint(out)
        publics.append(out)
        sealed.append({**out,'reference':{'actions':p['reference_plan']['actions']},'training_allowed':False})
    for name, extra in [('test_runtime.jsonl',publics),('evaluation_pairs_test.jsonl',sealed)]:
        path=target/name;preserve(path)
        keep=[r for r in read(path) if r['score_key'] not in black and r['score_key']!=key]
        write(path,keep+extra)
    for filename in ('agent_dataset_splits.csv','dataset_split_groups.csv'):
        path=ROOT/'ABC_J/results'/filename;preserve(path)
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); fields=reader.fieldnames; records=list(reader)
        original=next(r for r in records if r['score_key']=='SurzVQ02')
        new=dict(original);new.update(score_key=key,split='test')
        if filename=='dataset_split_groups.csv':
            new.update(score_id='11492',candidate_title='碣石调·幽兰',author='淇淇',
                       all_titles='碣石调·幽兰',notes_length='1360',tonic='C',
                       pitch_match_rate='',collection_status='complete',final_data_path=str(final),rounds='replacement')
        for r in records:
            if r['score_key'] in black:r['split']='excluded'
        records.append(new)
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    allpublic=read(target/'test_runtime.jsonl')
    assert len({r['score_key'] for r in allpublic})==48
    assert len({r['sample_id'] for r in allpublic})==len(allpublic)
    assert train_sha==hashlib.sha256(train.read_bytes()).hexdigest()
    preserve(target/'export_manifest.json')
    manifest=json.loads((target/'export_manifest.json').read_text())
    manifest.update(scores=48,phrases=len(allpublic),replacement_score=key,
                    excluded_score_keys=sorted(black),
                    public_sha256=hashlib.sha256((target/'test_runtime.jsonl').read_bytes()).hexdigest(),
                    sealed_sha256=hashlib.sha256((target/'evaluation_pairs_test.jsonl').read_bytes()).hexdigest())
    (target/'export_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    report={'score_key':key,'replaced':'SurzVQ02','phrases':len(phrases),'test_scores':48,
            'test_phrases':len(allpublic),'training_unchanged_sha256':train_sha,'backup':str(backup),
            'server_test_inputs_updated':False,'running_evaluation_inputs_changed':False}
    (source/'replacement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    refresh_inventories()
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
