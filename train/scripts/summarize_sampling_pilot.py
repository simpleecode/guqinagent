"""Compare the fixed paired sampling pilot without dropping failed phrases."""
from pathlib import Path
import json
import subprocess
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import rows,write_rows,validate

def main():
    p=ROOT/'train/eval_outputs/sampling_pilot'
    inputs=rows(p/'inputs.jsonl'); ids={r['sample_id'] for r in inputs}
    refs=rows(ROOT/'train/eval_outputs/two_stage/.work/test50/original_references.jsonl')
    write_rows(p/'references.jsonl',[r for r in refs if r['sample_id'] in ids])
    assert len(ids)==33
    result={}
    for arm in ('greedy','precision'):
        directory=p/arm;pred=rows(directory/'predictions.jsonl');validate(inputs,pred)
        assert len(pred)==33
        trace=[t for r in pred for stage in ('base','guqinizer') for t in r[stage].get('trace',[])]
        rounds=[]
        for f in (directory/'workers').glob('worker_*.log'):
            for line in f.open(errors='replace'):
                start=line.find('{"event": "eval_round"')
                if start>=0:rounds.append(json.loads(line[start:]))
        subprocess.run([sys.executable,'-m','evaluation.run_eval','--pred',str(directory/'predictions.jsonl'),
            '--reference',str(p/'references.jsonl'),'--experiment',f'sampling_{arm}',
            '--model','qwen35_9b_lora_two_stage_final_2gpu_bs12'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        evaluation=json.loads((ROOT/f'evaluation_results/sampling_{arm}/summary.json').read_text())
        result[arm]={'sampling':json.loads((directory/'sampling_params.json').read_text()),
                     'phrases':33,'protocol_valid':sum(bool(r['protocol_valid']) for r in pred),
                     'failed_stages':sum(not r[s].get('ok',False) for r in pred for s in ('base','guqinizer')),
                     'logged_generation_rounds':len(rounds),
                     'generated_tokens':sum(r.get('generated_tokens',0) for r in rounds),
                     'truncated_rounds':sum(bool(r.get('truncated')) for r in rounds),
                     'repeated_text_rounds':sum(any(n>=5 for n in Counter(
                         str(t.get('raw_output',''))[i:i+40] for i in range(max(0,len(str(t.get('raw_output','')))-39))).values()) for t in trace),
                     'pitch_accuracy':evaluation['metrics']['pitch_accuracy_at_50_cents'],
                     'pitch_events':evaluation['metrics']['pitch_evaluable_events'],
                     'final_warning':evaluation['final_warning']}
        print(arm,json.dumps(result[arm],ensure_ascii=False),flush=True)
    (p/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
