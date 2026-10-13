#!/usr/bin/env python3
"""Read-only raw-capture vs mapped-rest check; do not repair score products."""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    stats=Counter();durations=Counter();rest_durations=Counter();fallbacks=Counter()
    structure_cases=[];long_rest_cases=[];missing=[];unrecognized=[];by_score=Counter()
    paths=sorted((ROOT/'ABC_J/final').glob('*/out/raw_data.json'))+sorted((ROOT/'ABC_J/modern').glob('S*/raw_data.json'))
    for path in paths:
        raw=json.loads(path.read_text(encoding='utf-8'))
        score_dir=path.parent.parent if path.parent.name=='out' else path.parent
        score=score_dir.name
        notes=(raw.get('raw_score_data') or {}).get('notes') or raw.get('notes') or []
        mapped_path=score_dir/'jianpu_jianzi_mapped.json'
        if not mapped_path.exists():mapped_path=score_dir/'mapped/jianpu_jianzi_mapped.json'
        mapped={n['index']:n for n in json.loads(mapped_path.read_text())['notes']} if mapped_path.exists() else {}
        stats['scores_scanned']+=1
        for note in notes:
            stats['raw_rows']+=1
            token=str(note.get('note') or '')
            decoded=note.get('note_decoded') or {}
            dur=decoded.get('duration') or {}
            pitch=(decoded.get('pitch') or {}).get('value')
            name=str(dur.get('name'));value=dur.get('value');code=dur.get('code')
            if not dur:
                stats['rows_without_decoded_duration']+=1
                if score not in missing:missing.append(score)
            else:
                durations[(name,str(value),str(code))]+=1
            target=mapped.get(note.get('index'))
            is_mapped_rest=bool(target and target.get('jianpu',{}).get('rest'))
            if is_mapped_rest:
                stats['mapped_rest_rows']+=1
                if token not in ('0','q0','s0','d0'):
                    fallbacks[token]+=1
                # A structural token has no zero-pitch rest; bar/repeat syntax is explicit evidence.
                if '|' in token or token in (':','[1','[2','[3','[4'):
                    case={'score':score,'source_index':note.get('index'),'raw_token':token,
                          'runtime_duration':dur,'runtime_pitch':pitch,
                          'mapped_jianpu':target['jianpu'].get('pitch_name'),
                          'mapped_duration':target['jianpu'].get('duration_name'),'mapped_abc':target.get('abc')}
                    structure_cases.append(case);by_score[score]+=1
            if str(pitch)=='0' or token in ('0','q0','s0','d0'):
                stats['explicit_zero_pitch_rest_rows']+=1
                rest_durations[(name,str(value),str(code))]+=1
                if isinstance(value,(int,float)) and value>=1 or name in ('semibreve','whole','wholeNote'):
                    long_rest_cases.append({'score':score,'source_index':note.get('index'),'raw_token':token,
                          'runtime_duration':dur,'runtime_pitch':pitch,'mapped':target})
            if code not in (None,'','q','s','d','y') and len(unrecognized)<30:
                unrecognized.append({'score':score,'index':note.get('index'),'token':token,'duration':dur})
    def table(counter):
        return [{'runtime_name':key[0],'runtime_value':key[1],'code':key[2],'rows':count} for key,count in counter.most_common()]
    report={'counts':dict(stats),'runtime_duration_encodings':table(durations),'explicit_rest_duration_encodings':table(rest_durations),
            'whole_or_long_rest_candidates':long_rest_cases,'mapped_rest_nonstandard_raw_tokens':dict(fallbacks),
            'structural_token_misread_as_rest_rows':len(structure_cases),'structural_token_affected_scores':len(by_score),
            'structural_token_counts':dict(Counter(c['raw_token'] for c in structure_cases)),
            'affected_scores':dict(by_score),'structural_cases':structure_cases,
            'scores_with_missing_runtime_duration':missing,'unrecognized_duration_examples':unrecognized,
            'policy':'Read-only diagnosis; raw captures and derived datasets were not modified.'}
    out=ROOT/'ABC_J/results/raw_rest_duration_audit.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('structural_cases','whole_or_long_rest_candidates')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
