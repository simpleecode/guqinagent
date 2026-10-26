"""Freeze accepted prefixes and prepare one latest-runtime suffix per score."""
from pathlib import Path
import json
import hashlib
from collections import defaultdict
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluation.rules import replay_final_warnings
from train.scripts.visualize_completed_eval import rows,write_rows,validate
from train.scripts.run_vllm_eval_queue import balanced_shards

def main():
    old=ROOT/'train/eval_outputs/two_stage/.work/test50'
    out=ROOT/'train/eval_outputs/two_stage/.work/warning_suffix_retry'
    assert not out.exists(), 'Refuse to overwrite retry progress'
    inputs=rows(old/'eligible_inputs.jsonl');pred=rows(old/'eligible_predictions.jsonl')
    sources,finals=validate(inputs,pred)
    by_score=defaultdict(list)
    for s in inputs:by_score[s['score_key']].append((finals[s['sample_id']],s))
    selected=[];boundaries={};warnings=[]
    black=set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
    for score,samples in sorted(by_score.items()):
        if score in black:continue
        cases=replay_final_warnings(samples)
        if not cases:continue
        warnings.extend(cases)
        start=min(w['phrase_id'] for w in cases)
        boundaries[score]=start
        selected.extend(s for _,s in samples)
    assert selected
    out.mkdir();(out/'workers').mkdir()
    write_rows(out/'inputs.jsonl',selected)
    frozen=[]
    for i,shard in enumerate(balanced_shards(selected,4)):
        prefix=[finals[s['sample_id']] for s in shard if s['phrase_id']<boundaries[s['score_key']]]
        frozen.extend(prefix);write_rows(out/'workers'/f'predictions_{i:02d}.jsonl',prefix)
    write_rows(out/'frozen_prefix.jsonl',frozen)
    files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
           'agents/ToolRuntime/warning_registry.py','agents/ToolRuntime/warning_rendering.py',
           'scripts/audit_jianpu_jianzi_pitch.py','train/scripts/eval_two_stage_score.py',
           'train/scripts/run_vllm_eval_queue.py','train/configs/sampling_retry_schedule.json']
    report={'scores':len(boundaries),'total_phrases':len(selected),'frozen_phrases':len(frozen),
            'recomputed_phrases':len(selected)-len(frozen),'start_phrase_by_score':boundaries,
            'warning_count':len(warnings),'warnings':warnings,
            'input_sha256':hashlib.sha256((out/'inputs.jsonl').read_bytes()).hexdigest(),
            'runtime_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (out/'selection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('warnings','runtime_sha256')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
