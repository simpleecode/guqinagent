"""Validate the completed original test50 run and publish its eligible subset."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import rows,write_rows,validate,publish

def main():
    p=ROOT/'train/eval_outputs/two_stage/.work/test50'
    inputs=rows(p/'original_inputs.jsonl');preds=rows(p/'predictions.jsonl')
    validate(inputs,preds)
    assert len(inputs)==len(preds)==999
    assert hashlib.sha256((p/'original_inputs.jsonl').read_bytes()).hexdigest()=='250c19dba2dd2b39161148e5dcb7c29b428398a234a4a3dd237d19688a281269'
    black=set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
    eligible=[r for r in inputs if r['score_key'] not in black]
    ids={r['sample_id'] for r in eligible}
    write_rows(p/'eligible_inputs.jsonl',eligible)
    write_rows(p/'eligible_predictions.jsonl',[r for r in preds if r['sample_id'] in ids])
    write_rows(p/'eligible_references.jsonl',[r for r in rows(p/'original_references.jsonl') if r['sample_id'] in ids])
    assert len({r['score_key'] for r in eligible})==47
    mark=ROOT/'Mark1';manifest=json.loads((mark/'manifest.json').read_text())
    for rel,digest in manifest['sha256'].items():
        if rel.endswith('predictions.jsonl') and '/.work/' not in rel:
            assert hashlib.sha256((mark/rel).read_bytes()).hexdigest()==digest
    out=ROOT/'train/eval_outputs/two_stage'
    for key in black:
        target=out/key
        if target.exists():
            trash=Path(tempfile.mkdtemp(prefix='guqin_blacklisted_eval_',dir='/Volumes/F/.Trashes/501'))
            shutil.move(str(target),trash/key)
    publish(p/'eligible_inputs.jsonl',p/'eligible_predictions.jsonl','two_stage',render=True)
    subprocess.run([sys.executable,'-m','evaluation.run_eval','--pred',str(p/'eligible_predictions.jsonl'),
        '--reference',str(p/'eligible_references.jsonl'),'--experiment','two_stage_final',
        '--model','qwen35_9b_lora_two_stage_final_2gpu_bs12'],cwd=ROOT,check=True)

if __name__=='__main__':main()
