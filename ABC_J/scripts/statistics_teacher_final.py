"""Stream accepted final-stage audits, never union intermediate warnings."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
a=p.parse_args()
sources={}
for line in a.source.open():
    r=json.loads(line)
    sources[r['trajectory_id']]={
        'score_key':r['score_key'],'phrase_id':r['phrase_id'],
        'title':r['input']['metadata'].get('score_title') or r['input']['metadata'].get('title'),
        'events':{int(n['index']):n.get('event_index') for n in r['input']['notes_without_jianzi']},
    }
seen=set();stages=defaultdict(set);stage_counts=Counter();codes=defaultdict(Counter)
cases=[];repeat_cases=[];stored_warning_phrases=defaultdict(set)
for line in (a.directory/'teacher_trajectory_audit.jsonl').open():
    r=json.loads(line);sid=r['sample_id']
    assert sid not in seen,sid
    seen.add(sid)
    phrase,stage=sid.removesuffix('-teacher-tools').rsplit('-',1)
    assert phrase in sources,phrase
    stages[phrase].add(stage);stage_counts[stage]+=1
    private=r['teacher_private'];report=private.get('jianzi_quality_report') or {}
    warnings=report.get('warnings') or []
    if not warnings:continue
    info=sources[phrase]
    actions={int(x['source_index']):x for x in (private.get('accepted_plan') or {}).get('actions',[])}
    warnings=[dict(w,event_index=info['events'].get(w.get('source_index')),
                   jianzi_text=actions.get(w.get('source_index'),{}).get('jianzi_text')) for w in warnings]
    case={'sample_id':sid,'trajectory_id':phrase,'agent_stage':stage,
          'score_key':info['score_key'],'title':info['title'],'warnings':warnings}
    cases.append(case);stored_warning_phrases[stage].add(phrase)
    codes[stage].update(w.get('code','unknown') for w in warnings)
    repeat=[w for w in warnings if 'repeat' in w.get('code','')]
    if repeat:repeat_cases.append(dict(case,warnings=repeat))
missing={}
for phrase in sources:
    absent={'fingering_agent','guqinization'}-stages[phrase]
    if absent:missing[phrase]=sorted(absent)
report={'audit_basis':'stored quality report of accepted final plan; not union of tool rounds',
        'data_state':'raw teacher messages before GLM redaction',
        'source_phrases':len(sources),'accepted_stage_trajectories':len(seen),
        'by_stage':dict(stage_counts),'missing_stages':missing,
        'complete_phrases':sum(v=={'fingering_agent','guqinization'} for v in stages.values()),
        'final_warning_phrase_count':len(stored_warning_phrases['guqinization']),
        'either_stage_final_warning_phrase_count':len(set().union(*stored_warning_phrases.values())) if stored_warning_phrases else 0,
        'warning_stage_trajectory_counts':{stage:len(ids) for stage,ids in stored_warning_phrases.items()},
        'warning_codes':{stage:dict(count) for stage,count in codes.items()},
        'repeat_final_guqinizer_phrase_count':len({c['trajectory_id'] for c in repeat_cases if c['agent_stage']=='guqinization'}),
        'repeat_either_stage_phrase_count':len({c['trajectory_id'] for c in repeat_cases}),
        'cases':cases,'repeat_cases':repeat_cases}
(a.directory/'final_warning_statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(a.directory/'repeat_warning_cases.json').write_text(json.dumps(repeat_cases,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cases','repeat_cases','missing_stages']},ensure_ascii=False,indent=2))
print('missing phrase count:',len(missing))
