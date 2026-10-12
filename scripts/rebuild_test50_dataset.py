#!/usr/bin/env python3
"""Expand held-out repertoire to 50; reuse byte-identical accepted teacher rows.

Keep the existing 22 singleton tests, add all 25 eligible two-version groups
and three deterministically sampled three-version groups. Other versions of
held-out works are excluded, not put into train. Raw captures are never changed.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.scripts.build_text_protocol_eval_set import add_event_indices, event_indices_by_score, fingerprint, public_runtime_item
from agents.abc_to_jianzipu.reference_parser import AUDIT

METHOD = ROOT / 'ABC_J/agent_training/two_stage_audited'
RESULTS = ROOT / 'ABC_J/results'
SEED = 20261012


def read(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line), line


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def choose(groups, protected):
    existing = {key for key, members in groups.items() if any(r['split']=='test' for r in members)}
    assert len(existing)==22 and all(len(groups[g])==1 for g in existing), 'Unexpected current split; refuse replay'
    two = {g for g in groups if len(groups[g])==2 and g not in protected}
    assert len(two)==25, len(two)
    three = sorted((g for g in groups if len(groups[g])==3 and g not in protected),
                   key=lambda g: hashlib.sha256(f'{SEED}:{g}'.encode()).hexdigest())[:3]
    selected = existing | two | set(three)
    assert len(selected)==50
    representatives = {}
    for group in selected:
        # Choose annotation quality, not student performance, as the version criterion.
        representatives[group] = max(groups[group], key=lambda r: (
            float(r.get('pitch_match_rate') or 0), int(r.get('notes_length') or 0), r['score_key']))['score_key']
    excluded = {r['score_key'] for g in selected for r in groups[g] if r['score_key']!=representatives[g]}
    test = set(representatives.values())
    train = {r['score_key'] for g, members in groups.items() if g not in selected for r in members}
    assert not train & test and not train & excluded
    assert not protected & selected
    return train, test, excluded, selected, representatives, three


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    with (RESULTS/'dataset_split_groups.csv').open(encoding='utf-8-sig', newline='') as f:
        manifest=list(csv.DictReader(f))
    groups=defaultdict(list)
    for row in manifest:
        groups[row['leakage_group_id']].append(row)
    old_report=json.loads((RESULTS/'dataset_split_report.json').read_text())
    protected=set(old_report['groups_forced_to_train']) | set(old_report['development_groups_forced_to_train'])
    train_scores,test_scores,excluded,heldout,representatives,three=choose(groups,protected)
    score_split={k:'train' for k in train_scores} | {k:'test' for k in test_scores} | {k:'excluded_test_alternate' for k in excluded}
    counts=Counter(); current_ids=set(); retained_ids=set()
    for row,_ in read(METHOD/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'):
        current_ids.add(row['trajectory_id'])
        if row['score_key'] in train_scores:
            retained_ids.add(row['trajectory_id'])
    for split in ('train','test'):
        for row,_ in read(METHOD/'inferred_all'/f'inferred_trajectories_{split}.jsonl'):
            counts[score_split[row['score_key']]]+=1
    entries=[{'group':g,'score_key':representatives[g], 'title':groups[g][0]['canonical_seed'],
              'version_count':len(groups[g]), 'excluded_versions':[r['score_key'] for r in groups[g] if r['score_key']!=representatives[g]]}
             for g in sorted(heldout)]
    report={'schema_version':'test50-incremental-rebuild-1.0','seed':SEED,
            'selection':'22 retained singleton groups + 25 two-version groups + 3 seeded three-version groups; one test version per work',
            'representative_policy':'highest captured annotation pitch_match_rate, then notes_length, then score_key; no model-performance selection',
            'splits':{'train':{'scores':len(train_scores),'source_phrases':counts['train'],'selected_phrases':len(retained_ids)},
                      'test':{'scores':len(test_scores),'phrases':counts['test'],'selection_applied':False},
                      'excluded_test_alternate':{'scores':len(excluded),'phrases':counts['excluded_test_alternate']}},
            'reused_training_phrases':len(retained_ids),'removed_training_phrases':len(current_ids-retained_ids),
            'new_training_phrases':0,'changed_retained_inputs':0,'teacher_generation_required':0,
            'three_version_additions':[e for e in entries if e['group'] in three],
            'test_scores':entries,'train_score_keys':sorted(train_scores),'excluded_score_keys':sorted(excluded),
            'protected_train_groups':sorted(protected),
            'leakage_check':{'train_test_group_overlap':[],'test_versions_per_work':1,'passed':True}}
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not args.apply:
        return
    work=Path(tempfile.mkdtemp(prefix='test50-rebuild-',dir=METHOD))
    staged_all=work/'inferred_all'; staged_selected=work/'pitch_eligible_train_test'
    staged_all.mkdir(); staged_selected.mkdir()
    test_rows=[]; source_train_count=0
    handles={s:(staged_all/f'inferred_trajectories_{s}.jsonl').open('w',encoding='utf-8') for s in ('train','test')}
    try:
        for split in ('train','test'):
            for row,line in read(METHOD/'inferred_all'/f'inferred_trajectories_{split}.jsonl'):
                target=score_split[row['score_key']]
                if target=='train':
                    assert split=='train', 'Unexpected new training input: generate before promotion'
                    handles[target].write(line)  # Retained inference inputs are byte-identical.
                    source_train_count+=1
                elif target=='test':
                    row['split']='test'; test_rows.append(row)
        test_rows.sort(key=lambda r:(r['score_key'],r['input']['event_range']['start']))
        for row in test_rows:
            handles['test'].write(json.dumps(row,ensure_ascii=False)+'\n')
    finally:
        for h in handles.values():h.close()
    old_summary=json.loads((METHOD/'inferred_all/inferred_trajectories_summary.json').read_text())
    old_summary.update(scores=len(train_scores|test_scores),trajectories={'train':source_train_count,'test':len(test_rows)},
                       split_manifest=str(RESULTS/'agent_dataset_splits.csv'),excluded_test_alternate_score_keys=sorted(excluded))
    # Historical reference-class/patch aggregates no longer describe this filtered corpus.
    for key in ('reference_action_classes','all_patch_types','verified_supervised_patch_types'):
        old_summary.pop(key,None)
    dump(staged_all/'inferred_trajectories_summary.json',old_summary)
    with (staged_selected/'inferred_trajectories_train.jsonl').open('w',encoding='utf-8') as out:
        for row,line in read(METHOD/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'):
            if row['trajectory_id'] in retained_ids:out.write(line)
    shutil.copy2(staged_all/'inferred_trajectories_test.jsonl',staged_selected/'inferred_trajectories_test.jsonl')
    test_ids={r['trajectory_id'] for r in test_rows}
    for name,values in [('pitch_eligible_phrase_ids_train.txt',retained_ids),('pitch_eligible_phrase_ids_test.txt',test_ids),('pitch_eligible_phrase_ids.txt',retained_ids|test_ids)]:
        (staged_selected/name).write_text(''.join(x+'\n' for x in sorted(values)))
    selection=json.loads((METHOD/'pitch_eligible_train_test/selection_manifest.json').read_text())
    selection['splits']['train'].update(source_phrase_count=source_train_count,selected_phrase_count=len(retained_ids))
    for prefix in ('ordinary_blank_run','all_empty_reference'):
        key=prefix+'_excluded_phrase_ids'
        values=[x for x in selection['splits']['train'].get(key,[]) if x.split('-p')[0] in train_scores]
        selection['splits']['train'][key]=values
        selection['splits']['train'][prefix+'_excluded_phrase_count']=len(values)
    selection['splits']['test'].update(source_phrase_count=len(test_rows),selected_phrase_count=len(test_rows),selection_applied=False)
    selection['total_selected_phrase_count']=len(retained_ids)+len(test_rows)
    selection['split_rebuild_report']=str(RESULTS/'dataset_split_report.json')
    dump(staged_selected/'selection_manifest.json',selection)

    teacher_targets=[]
    for method,base_name,multiplier in [('two_stage_audited','teacher_train_final',2),('single_stage_audited','teacher_final_redacted',1)]:
        base=ROOT/'ABC_J/agent_training'/method/base_name
        target=work/method;target.mkdir()
        public_ids=set();private_ids=set(); actual_counts={}
        for name in ('messages_train.jsonl','teacher_trajectory_audit.jsonl','reasoning_redaction_audit.jsonl'):
            count=0
            with (target/name).open('w',encoding='utf-8') as out:
                for row,line in read(base/name):
                    sid=row.get('sample_id','')
                    match=re.match(r'^(.+-p\d+)(?:-|$)',sid)
                    if match and match.group(1) in retained_ids:
                        out.write(line);count+=1
                        if name=='messages_train.jsonl':public_ids.add(sid)
                        elif name=='teacher_trajectory_audit.jsonl':private_ids.add(sid)
            actual_counts[name]=count
        assert public_ids==private_ids and len(public_ids)==multiplier*len(retained_ids)
        run(ROOT/'scripts/validate_teacher_agent_messages.py','--input-dir',target)
        dump(target/'split_rebuild_report.json',{'status':'complete','reused_unchanged':len(public_ids),'new_generation':0,
             'removed_phrase_count':len(current_ids-retained_ids),'public_private_ids_match':True,'byte_identical_retained_rows':True})
        teacher_targets.append((base,target))

    eval_target=work/'test50';eval_target.mkdir()
    maps=event_indices_by_score(test_rows)
    public=[];sealed=[]
    for raw in test_rows:
        item=add_event_indices(raw,maps[raw['score_key']])
        tune=item['input']['normalized_tuning']
        assert list(tune['open_midi'])==AUDIT.parse_open_midi(item['input']['metadata']),item['trajectory_id']
        runtime=public_runtime_item(item)
        # Clear handoff-derived starting scope: inference will replay generated history.
        row={'schema_version':'agent-eval-input-2.0','sample_id':item['trajectory_id'],'split':'test',
             'score_key':item['score_key'],'phrase_id':item['phrase_id'],'runtime_item':runtime}
        row['input_sha256']=fingerprint(row)
        public.append(row)
        sealed.append({**row,'reference':{'actions':item['reference_plan']['actions']},'training_allowed':False})
    assert len(public)==len(test_ids) and len({r['score_key'] for r in public})==50
    write_rows(eval_target/'test_runtime.jsonl',public);write_rows(eval_target/'evaluation_pairs_test.jsonl',sealed)
    dump(eval_target/'export_manifest.json',{'source':str(METHOD/'inferred_all/inferred_trajectories_test.jsonl'),
         'phrases':len(public),'scores':50,'test_filtering':False,'normalized_tuning_verified':True,
         'train_test_score_overlap':[],'train_test_leakage_group_overlap':[],
         'public_sha256':hashlib.sha256((eval_target/'test_runtime.jsonl').read_bytes()).hexdigest(),
         'sealed_sha256':hashlib.sha256((eval_target/'evaluation_pairs_test.jsonl').read_bytes()).hexdigest()})
    sft=work/'sft';
    run(ROOT/'train/scripts/export_sft_dataset.py','--input-dir',teacher_targets[0][1],'--output-dir',sft)
    run(ROOT/'train/scripts/validate_sft_export.py','--input-dir',sft)
    sft_manifest=json.loads((sft/'dataset_manifest.json').read_text())
    sft_manifest['source_dir']=str(METHOD/'teacher_train_final')
    dump(sft/'dataset_manifest.json',sft_manifest)

    # All verification succeeded. Replace canonical products, preserve old ones in Trash.
    trash=Path(tempfile.mkdtemp(prefix='guqin_before_test50.',dir=Path.home()/'.Trash'))
    replacements=[(METHOD/'inferred_all',staged_all),(METHOD/'pitch_eligible_train_test',staged_selected),*teacher_targets,
                  (ROOT/'train/eval_inputs_v2_text_protocol/test50',eval_target),
                  (ROOT/'train/data/guqin_agent_sft_two_stage_test50',sft)]
    for index,(base,candidate) in enumerate(replacements):
        if base.exists():shutil.move(str(base),str(trash/f'{index}_{base.name}'))
        base.parent.mkdir(parents=True,exist_ok=True);candidate.rename(base)
    for filename in ('dataset_split_groups.csv','agent_dataset_splits.csv','dataset_split_report.json'):
        shutil.copy2(RESULTS/filename,trash/filename)
    for row in manifest:row['split']=score_split[row['score_key']]
    with (RESULTS/'dataset_split_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    with (RESULTS/'agent_dataset_splits.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['leakage_group_id','score_key','split','seed']);w.writeheader()
        w.writerows({k:r[k] for k in ('leakage_group_id','score_key','split')}|{'seed':SEED} for r in manifest)
    report.update(status='complete',backup_trash=str(trash),canonical_eval=str(ROOT/'train/eval_inputs_v2_text_protocol/test50'),
                  canonical_sft=str(ROOT/'train/data/guqin_agent_sft_two_stage_test50'))
    dump(RESULTS/'dataset_split_report.json',report)
    inventory=RESULTS/'split_inventories'
    if inventory.exists():shutil.move(str(inventory),str(trash/'split_inventories'))
    inventory.mkdir()
    for split,keys,ids in [('train',train_scores,retained_ids),('test',test_scores,test_ids)]:
        (inventory/f'{split}_score_ids.txt').write_text(''.join(x+'\n' for x in sorted(keys)))
        (inventory/f'{split}_phrase_ids.txt').write_text(''.join(x+'\n' for x in sorted(ids)))
    (inventory/'excluded_score_ids.txt').write_text(''.join(x+'\n' for x in sorted(excluded)))
    (inventory/'removed_training_phrase_ids.txt').write_text(''.join(x+'\n' for x in sorted(current_ids-retained_ids)))
    dump(inventory/'split_inventories_report.json',report)
    # Old full-training viewers and problem subsets refer to removed samples.
    for method in ('two_stage_audited','single_stage_audited'):
        for name in ('full_training_visualization','problem_visualization'):
            path=ROOT/'ABC_J/agent_training'/method/name
            if path.exists():shutil.move(str(path),str(trash/f'{method}_{name}'))
    work.rmdir()
    print('COMPLETE '+json.dumps(report['splits'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
