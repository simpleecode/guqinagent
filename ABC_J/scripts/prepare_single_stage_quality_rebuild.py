"""Audit single-stage final teachers under their original reference history."""
import sys,json,csv
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ABC_J.scripts import prepare_final_warning_rebuild as shared

BASE=ROOT/'ABC_J/agent_training/single_stage_audited'
OUT=BASE/'.quality_rebuild'

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--length-tsv',type=Path)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'regenerated_raw').exists(), 'Generation already started'
    source=[json.loads(s) for s in shared.SOURCE.open() if s.strip()]
    expected={r['trajectory_id'] for r in source}
    audit=BASE/'teacher_raw/teacher_trajectory_audit.jsonl'
    fingerprints={f:shared.sha256(ROOT/f) for f in shared.RUNTIME_FILES+['agents/ToolRuntime/hint_checks.py']}
    source_sha=shared.sha256(shared.SOURCE);audit_sha=shared.sha256(audit)
    report_path=OUT/'warning_audit_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        assert report['source_sha256']==source_sha and report['audit_sha256']==audit_sha
        assert report['runtime_sha256']==fingerprints
    else:
        suffix='-single_stage-teacher-tools'
        for s in audit.open():
            r=json.loads(s);sid=r['sample_id'];assert sid.endswith(suffix)
            key=sid[:-len(suffix)];assert key not in shared.PLANS
            shared.PLANS[key]=r['teacher_private']['accepted_plan']
        assert len(source)==len(expected)==3976 and set(shared.PLANS)==expected
        grouped=defaultdict(list)
        for row in source:grouped[row['score_key']].append(row)
        cases={};done=0
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
            for count,found in pool.map(shared.replay_score,grouped.values()):
                cases.update(found);done+=count
                print(f'Replayed {done}/3976; warning phrases={len(cases)}',flush=True)
        assert shared.sha256(audit)==audit_sha and shared.sha256(shared.SOURCE)==source_sha
        assert fingerprints=={f:shared.sha256(ROOT/f) for f in fingerprints}
        report={'source_sha256':source_sha,'audit_sha256':audit_sha,'runtime_sha256':fingerprints,
                'basis':'accepted single-stage final plans; current runtime; actual teacher reference history',
                'source_phrases':len(source),'warning_phrases':len(cases),
                'warning_count':sum(map(len,cases.values())),
                'warning_codes':dict(Counter(w['code'] for ws in cases.values() for w in ws)),
                'cases':cases}
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','runtime_sha256')},ensure_ascii=False),flush=True)
    if not args.length_tsv:return
    long_rows=list(csv.DictReader(args.length_tsv.open(),delimiter='\t'))
    long_ids={r['source_sample_id'].removesuffix('-single_stage-teacher-tools') for r in long_rows}
    assert long_ids<=expected
    selected=set(report['cases'])|long_ids
    final={**report,'selected_ids':sorted(selected),'selected_phrases':len(selected),
           'long_phrases':len(long_ids),'warning_and_long':len(long_ids & set(report['cases'])),
           'length_threshold':8192,'length_tsv_sha256':shared.sha256(args.length_tsv),
           'sft_sha256':shared.sha256(BASE/'sft_export/guqin_agent_train.jsonl'),
           'raw_public_sha256':shared.sha256(BASE/'teacher_raw/messages_train.jsonl')}
    (OUT/'selection_report.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    (OUT/'trajectory_ids.txt').write_text(''.join(s+'\n' for s in sorted(selected)))
    print('SELECTED',len(selected),'WARNING',report['warning_phrases'],'LONG',len(long_ids),flush=True)

if __name__=='__main__':main()
