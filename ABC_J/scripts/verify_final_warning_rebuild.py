"""Verify rebuilt pairs and compare final warnings in teacher-visible context."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agents.ToolRuntime.runtime import validate_jianzi_only, harmonic_region_at_phrase_start
from ABC_J.scripts.prepare_final_warning_rebuild import BASE, SOURCE, OUT, sha256

HISTORY={}
PLANS={}


def read(path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def verify_one(source):
    sid=source['trajectory_id']
    item=deepcopy(source)
    item['baseline_plan']=PLANS[sid]
    hist=HISTORY[source['score_key']]
    item['harmonic_region_at_start']=harmonic_region_at_phrase_start(item,hist)
    valid,report=validate_jianzi_only(item,[],historical=hist)
    if not valid:
        raise ValueError(f'Invalid final plan: {sid}')
    return sid,report['warnings']


def main():
    selection=json.loads((OUT/'selection_report.json').read_text())
    assert sha256(SOURCE)==selection['source_sha256']
    assert all(sha256(ROOT/f)==digest for f,digest in selection['runtime_sha256'].items())
    wanted=set(selection['selected_ids'])
    assert len(wanted)==766
    expected={f'{sid}-{stage}-teacher-tools' for sid in wanted
              for stage in ('fingering_agent','guqinization')}
    raw=OUT/'regenerated_raw'
    public=read(raw/'messages_train.jsonl')
    audits=read(raw/'teacher_trajectory_audit.jsonl')
    assert len(public)==len(audits)==len(expected)==1532
    assert {r['sample_id'] for r in public}=={r['sample_id'] for r in audits}==expected
    assert all(r['verification']['training_eligible'] for r in public)
    assert all(r['verification']['replay_passed'] for r in audits)
    checkpoints={r['trajectory_id']:r for r in read(raw/'checkpoint.jsonl')}
    assert set(checkpoints)==wanted
    assert not any(r.get('status')=='attempted_with_failure' for r in checkpoints.values())
    for r in audits:
        if r['sample_id'].endswith('-guqinization-teacher-tools'):
            PLANS[r['sample_id'].removesuffix('-guqinization-teacher-tools')]=r['teacher_private']['accepted_plan']
    sources=read(SOURCE)
    for row in sources:
        HISTORY.setdefault(row['score_key'],{})[(row['score_key'],row['phrase_id'])]=row
    selected=[r for r in sources if r['trajectory_id'] in wanted]
    cases={}
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        for number,(sid,warnings) in enumerate(pool.map(verify_one,selected),1):
            if warnings:cases[sid]=warnings
            if number%100==0:print(f'Replayed {number}/766',flush=True)
    counts=Counter(w['code'] for ws in cases.values() for w in ws)
    keyset=lambda cs:{(sid,w.get('source_index'),w['code']) for sid,ws in cs.items() for w in ws}
    before=keyset(selection['cases']);after=keyset(cases)
    report={'verified_phrases':766,'verified_stage_rows':1532,'failures':0,
        'basis':'current runtime; same original reference history used during teacher generation',
        'before_warning_phrases':766,'before_warning_events':selection['selected_events'],
        'after_warning_phrases':len(cases),'after_warning_events':sum(counts.values()),
        'after_codes':dict(counts.most_common()),
        'original_warning_positions_resolved':len(before-after),
        'original_warning_positions_remaining':len(before&after),
        'new_warning_positions':len(after-before),'cases':cases,
        'raw_public_sha256':sha256(raw/'messages_train.jsonl'),
        'raw_audit_sha256':sha256(raw/'teacher_trajectory_audit.jsonl')}
    (OUT/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
