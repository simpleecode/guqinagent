#!/usr/bin/env python3
"""Count final Guqinizer audit warnings and token-overlength phrases, streaming audits."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def phrase(sample):
    return re.sub(r'-(?:fingering_agent|guqinization|single_stage)-teacher-tools$', '', sample)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--overlength',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    selected=set();stage_counts=Counter()
    messages=args.input_dir/'messages_train.jsonl'
    digest=hashlib.sha256()
    with messages.open('rb') as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                row=json.loads(line); selected.add(phrase(row['sample_id']));stage_counts[row['agent_stage']]+=1
    warned=set();warned_any_stage=set();warnings_by_phrase={};codes=Counter();warned_stages=Counter();seen=set()
    with (args.input_dir/'teacher_trajectory_audit.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():continue
            row=json.loads(line);sample=row['sample_id'];pid=phrase(sample)
            assert pid in selected and sample not in seen
            seen.add(sample)
            warnings=((row.get('teacher_private') or {}).get('jianzi_quality_report') or {}).get('warnings') or []
            stage='guqinization' if '-guqinization-' in sample else 'fingering_agent'
            if warnings:
                warned_stages[stage]+=1;warned_any_stage.add(pid)
                if stage=='guqinization':
                    warned.add(pid);warnings_by_phrase[pid]=warnings
                    codes.update(w.get('code','unknown') for w in warnings)
    assert len(seen)==sum(stage_counts.values())
    long_rows=[]
    with args.overlength.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            assert int(row['tokens'])>8192
            assert phrase(row['source_sample_id']) in selected
            long_rows.append(row)
    long={phrase(r['source_sample_id']) for r in long_rows}
    report={'definition':'over8192 in either serialized stage OR warnings in final Guqinizer jianzi_quality_report; not intermediate tool warnings',
            'warning_source':'stored final private audit after accepted plan, not a new full-score re-audit',
            'training_phrases':len(selected),'stage_trajectories':sum(stage_counts.values()),
            'messages_sha256':digest.hexdigest(),'over8192_stage_trajectories':len(long_rows),'over8192_phrases':len(long),
            'final_guqinizer_warning_phrases':len(warned),'overlap_phrases':len(long&warned),
            'problem_phrases':len(long|warned),'problem_rate':len(long|warned)/len(selected),
            'final_warning_stage_counts':dict(warned_stages),'warning_codes':dict(codes),
            'either_stage_final_warning_phrases':len(warned_any_stage),
            'either_stage_warning_or_overlength_phrases':len(warned_any_stage|long),
            'over8192_samples':long_rows,'final_guqinizer_warning_phrase_ids':sorted(warned),
            'problem_phrase_ids':sorted(long|warned),'final_warnings_by_phrase':warnings_by_phrase}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list) and k!='final_warnings_by_phrase'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
