"""Audit optional hint refinements before publishing any changed score."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import rows,write_rows,validate
from evaluation.rules import replay_final_warnings
from evaluation.parser import final_actions
from agents.ToolRuntime.hint_checks import repeated_pitch_hints


def main():
    out=ROOT/'train/eval_outputs/two_stage/.work/hint_refinement'
    prior=out.parent/'warning_suffix_retry'
    selection=json.loads((out/'selection_report.json').read_text())
    for name,digest in selection['sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    inputs,predictions=rows(out/'inputs.jsonl'),rows(out/'predictions.jsonl')
    sources,finals=validate(inputs,predictions)
    _,baseline=validate(inputs,rows(out/'baseline.jsonl'))
    targets=set(selection['target_ids'])
    frozen=rows(out/'frozen_predictions.jsonl')
    assert len(finals)==684 and len(targets)==75 and len(frozen)==609
    assert all(finals[r['sample_id']]==r for r in frozen)
    assert all(finals[sid]['base']==baseline[sid]['base'] for sid in targets)
    grouped=defaultdict(list)
    for source in inputs:grouped[source['score_key']].append(source)
    warnings=[];old_warnings=[];held={};approved=[]
    def position(w):return w['phrase_id'],w['event_id'],w['violation_type']
    for score,source_rows in sorted(grouped.items()):
        old=replay_final_warnings([(baseline[s['sample_id']],s) for s in source_rows])
        new=replay_final_warnings([(finals[s['sample_id']],s) for s in source_rows])
        old_warnings.extend(old);warnings.extend(new)
        introduced={position(w) for w in new}-{position(w) for w in old}
        if introduced:
            held[score]={'introduced':sorted(introduced),'old_count':len(old),'new_count':len(new)}
        else:approved.append(score)
    records=[]
    total_tokens=0;total_rounds=0;total_passes=0
    for case in selection['cases']:
        sid=case['sample_id'];new=finals[sid];old=baseline[sid]
        indices={n for h in case['hints'] for n in h['source_indices']}
        def uses(record):
            return [{'source_index':a['source_index'],'text':a.get('jianzi_text') or a.get('text') or ''}
                    for a in final_actions(record) if int(a['source_index']) in indices
                    and re.search('轮|锁',a.get('jianzi_text') or a.get('text') or '')]
        trace=new['guqinizer'].get('trace') or []
        seen=any(':hint:' in json.dumps(turn.get('tool_results') or [],ensure_ascii=False) for turn in trace)
        tokens=None  # Stored traces do not include token usage; logs below do.
        passes=int(new.get('guqinizer_passes') or 0)
        total_rounds+=len(trace);total_passes+=passes
        records.append({'sample_id':sid,'score_key':case['score_key'],'hint_seen_in_tool_receipt':seen,
                        'before_uses':uses(old),'after_uses':uses(new),
                        'remaining_hint_runs':len(repeated_pitch_hints(sources[sid]['runtime_item'],final_actions(new))),
                        'generated_tokens':tokens,'rounds':len(trace),'passes':passes,
                        'plan_changed':new['guqinizer']['plan']!=old['guqinizer']['plan']})
    full_inputs=rows(prior/'combined_inputs.jsonl')
    full_baseline={r['sample_id']:r for r in rows(prior/'combined_predictions.jsonl')}
    combined=[finals[s['sample_id']] if s['score_key'] in approved else full_baseline[s['sample_id']]
              for s in full_inputs]
    validate(full_inputs,combined)
    write_rows(out/'approved_combined_inputs.jsonl',full_inputs)
    write_rows(out/'approved_combined_predictions.jsonl',combined)
    usage=Counter(); logged_rounds=0
    for path in sorted((out/'workers').glob('worker_*.log')):
        for line in path.read_text(errors='replace').splitlines():
            start=line.find('{"event": "eval_round"')
            if start<0:continue
            event,_=json.JSONDecoder().raw_decode(line[start:])
            if event.get('sample_id') in targets:
                usage[event['sample_id']]+=int(event.get('generated_tokens') or 0)
                logged_rounds+=1
    for record in records:
        record['generated_tokens']=usage.get(record['sample_id'])
    report={'unique_ids':len(finals),'frozen_exact':len(frozen),'base_exact':len(targets),
            'raw_old_warning_count':len(old_warnings),'raw_new_warning_count':len(warnings),
            'raw_new_warning_codes':dict(Counter(w['violation_type'] for w in warnings)),
            'raw_warnings':warnings,'approved_scores':approved,'held_scores':held,
            'hint_receipt_phrases':sum(r['hint_seen_in_tool_receipt'] for r in records),
            'adopted_phrases':sum(bool(r['after_uses']) for r in records),
            'adopted_occurrences':sum(len(r['after_uses']) for r in records),
            'remaining_hint_phrases':sum(r['remaining_hint_runs']>0 for r in records),
            'generated_tokens':sum(usage.values()) if usage else None,
            'token_cost_source':'worker eval_round logs (including retries)',
            'logged_rounds':logged_rounds,'rounds':total_rounds,'passes':total_passes,
            'records':records,'publication_status':'not yet published; held scores retain baseline'}
    (out/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','raw_warnings')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
