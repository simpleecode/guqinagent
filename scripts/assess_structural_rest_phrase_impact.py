#!/usr/bin/env python3
"""Shadow-only assessment of structural tokens falsely rendered as rests.

Normalize identified structural rows in memory, retain source indexes, and
compare frozen phrase segmentation and public ordinals. No production inputs,
teacher trajectories, captures or parser code are edited.
"""
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
from scripts.audit_jianpu_jianzi_pitch import parse_jianpu,parse_tonic_midi,replay_harmonic_scope


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def main():
    method=ROOT/'ABC_J/agent_training/two_stage_audited'
    audit=json.loads((ROOT/'ABC_J/results/raw_rest_duration_audit.json').read_text())
    bad=defaultdict(dict)
    for case in audit['structural_cases']:bad[case['score']][int(case['source_index'])]=case['raw_token']
    selected={r['trajectory_id'] for r in rows(method/'pitch_eligible_train_test/inferred_trajectories_train.jsonl')}
    old_scores=defaultdict(list)
    for split in ('train','test'):
        for row in rows(method/'inferred_all'/f'inferred_trajectories_{split}.jsonl'):
            # Retain only small prompt/annotation fields, never the large canonical patch trees.
            old_scores[row['score_key']].append({k:row[k] for k in ('trajectory_id','score_key','phrase_id','split','input','reference_plan')})
    summaries={};cases=[];reuse=[];affected=[];selected_counts=Counter();old_test_counts=Counter()
    parser_baseline_disagreement=[]
    for score,phrases in sorted(old_scores.items()):
        phrases.sort(key=lambda r:r['input']['event_range']['start'])
        split=phrases[0]['split'];metadata=phrases[0]['input']['metadata']
        note_map={n['index']:n for r in phrases for n in r['input']['notes_without_jianzi']}
        notes=[deepcopy(note_map[k]) for k in sorted(note_map)]
        assert [n['index'] for n in notes]==list(range(len(notes))),score
        changes={k:v for k,v in bad[score].items() if k in note_map}
        corrected=deepcopy(notes)
        for index in changes:
            corrected[index].update(jianpu='|',jianpu_alt=None,abc='|',duration='小节线',event_index=None)
        ordinal=0
        for note in corrected:
            if note['abc']=='|' or note['jianpu']=='|' or note['duration']=='小节线':note['event_index']=None
            else:note['event_index']=ordinal;ordinal+=1
        tonic=parse_tonic_midi(metadata)
        sounding=lambda n:any(parse_jianpu(n.get(k),tonic) is not None for k in ('jianpu','jianpu_alt'))
        old_computed=split_phrase_ranges(notes,max_sounding=16,is_sounding=sounding)
        actual=[(r['input']['event_range']['start'],r['input']['event_range']['end_exclusive']) for r in phrases]
        if old_computed!=actual:parser_baseline_disagreement.append(score)
        assert old_computed==actual, f'Frozen splitter cannot reproduce old ranges: {score}'
        new_ranges=split_phrase_ranges(corrected,max_sounding=16,is_sounding=sounding)
        new_range_ids={span:f'{score}-p{index:04d}' for index,span in enumerate(new_ranges,1)}
        reference={a['source_index']:a.get('jianzi_text') or a.get('text') or '' for r in phrases for a in r['reference_plan']['actions']}
        old_scope={};new_scope={};active_old=False;active_new=False
        for old,new in zip(notes,corrected):
            index=old['index'];old_scope[index]=active_old;new_scope[index]=active_new
            text=reference.get(index,'')
            active_old=replay_harmonic_scope(active_old,old,text)
            active_new=replay_harmonic_scope(active_new,new,text)
        first_change=min(changes,default=len(notes))
        score_stats=Counter();selected_score_stats=Counter()
        for number,row in enumerate(phrases,1):
            pid=row['trajectory_id'];start=row['input']['event_range']['start'];end=row['input']['event_range']['end_exclusive'];span=(start,end)
            direct=sorted(k for k in changes if start<=k<end)
            numbering=any(notes[k].get('event_index')!=corrected[k].get('event_index') for k in range(start,end))
            same_span=span in new_range_ids;same_id=new_range_ids.get(span)==pid
            scope_change=old_scope[start]!=new_scope[start]
            earlier_history_changed=start>first_change
            local_changed=bool(direct or numbering)
            safe=same_span and same_id and not local_changed and not earlier_history_changed
            flags={'direct_structural_row_changes':bool(direct),'public_ordinals_changed':numbering,
                   'old_span_not_preserved':not same_span,'phrase_id_or_range_changed':not (same_span and same_id),
                   'reference_harmonic_scope_at_start_changed':scope_change,
                   'earlier_score_history_changed':earlier_history_changed,'strictly_reusable':safe}
            score_stats.update(k for k,v in flags.items() if v)
            in_train_selection=pid in selected
            if in_train_selection:
                selected_score_stats.update(k for k,v in flags.items() if v)
                selected_counts.update(k for k,v in flags.items() if v)
                (reuse if safe else affected).append(pid)
            if split=='test':old_test_counts.update(k for k,v in flags.items() if v)
            cases.append({'phrase_id':pid,'split':split,'in_selected_train':in_train_selection,
                          'old_span':[start,end],'matching_new_phrase_id':new_range_ids.get(span),
                          'changed_source_indices':direct,**flags})
        summaries[score]={'split':split,'title':metadata.get('score_title'),
                          'structural_rows_fixed':len(changes),'first_changed_source_index':None if not changes else first_change,
                          'old_phrase_count':len(phrases),'new_phrase_count':len(new_ranges),
                          'old_phrase_effect_counts':dict(score_stats),'selected_train_effect_counts':dict(selected_score_stats),
                          'old_ranges':actual,'shadow_ranges':new_ranges}
    assert len(reuse)+len(affected)==len(selected)==3995
    selected_cases=[case for case in cases if case['in_selected_train']]
    direct_or_range_or_scope=[case for case in selected_cases if case['direct_structural_row_changes']
                             or case['phrase_id_or_range_changed'] or case['reference_harmonic_scope_at_start_changed']]
    ordinal_history_only=[case for case in selected_cases if not case['strictly_reusable']
                          and not (case['direct_structural_row_changes'] or case['phrase_id_or_range_changed']
                                   or case['reference_harmonic_scope_at_start_changed'])]
    stats={'selected_train_phrases':len(selected),'strictly_reusable_training_phrases':len(reuse),
           'training_phrases_requiring_regeneration_or_deeper_replay':len(affected),
           'direct_or_range_or_reference_scope_changed_training_phrases':len(direct_or_range_or_scope),
           'local_content_range_reference_scope_same_but_ordinals_history_changed':len(ordinal_history_only),
           'selected_train_effect_counts':dict(selected_counts),'test_old_phrase_effect_counts':dict(old_test_counts),
           'original_train_raw_phrases':sum(v['old_phrase_count'] for v in summaries.values() if v['split']=='train'),
           'shadow_train_raw_phrases':sum(v['new_phrase_count'] for v in summaries.values() if v['split']=='train'),
           'original_test_phrases':sum(v['old_phrase_count'] for v in summaries.values() if v['split']=='test'),
           'shadow_test_phrases':sum(v['new_phrase_count'] for v in summaries.values() if v['split']=='test'),
           'scores_with_changed_boundaries':sum(v['old_ranges']!=v['shadow_ranges'] for v in summaries.values()),
           'splitter_baseline_disagreements':parser_baseline_disagreement}
    report={'status':'complete','policy':'Read-only shadow normalization of structural bars; no original capture, formal input, trajectory or model changed',
            'assumptions':['Recognize structural rows as non-sounding separators while preserving raw source indexes.',
                           'Keep the current work-level train/test split and tuning unchanged.',
                           'Normalize separator ABC to | for existing frozen splitter; retain original raw token in report.',
                           'Reuse only identical old phrase ID/range, current note fields/ordinals and unchanged earlier history.',
                           'Runtime list_context/expand_context permit earlier phrases only; unchanged prefixes do not read changed future phrases.',
                           'Harmonic start-state comparison uses source annotations, not hypothetical newly generated teacher plans.',
                           'No pitch filtering, baseline route/reference-patch rebuilding, repeat expansion or grace-token repairs simulated.'],
            'counts':stats,'per_score':summaries,'per_old_phrase':cases,
            'strict_reuse_phrase_ids':sorted(reuse),'affected_selected_train_phrase_ids':sorted(affected)}
    out=ROOT/'ABC_J/results/structural_rest_phrase_impact.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print('Sx4mbdGp',json.dumps(summaries['Sx4mbdGp'],ensure_ascii=False))


if __name__=='__main__':main()
