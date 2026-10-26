"""Publish only score-level hint refinements without new rule violations."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import publish,rows,validate


def main():
    out=ROOT/'train/eval_outputs/two_stage/.work/hint_refinement'
    report=json.loads((out/'verification_report.json').read_text())
    assert report['unique_ids']==684 and report['frozen_exact']==609 and report['base_exact']==75
    inputs,predictions=rows(out/'inputs.jsonl'),rows(out/'predictions.jsonl')
    validate(inputs,predictions)
    usage=Counter();logged_rounds=0
    for path in (out/'workers').glob('worker_*.log'):
        for line in path.read_text(errors='replace').splitlines():
            start=line.find('{"event": "eval_round"')
            if start<0:continue
            event,_=json.JSONDecoder().raw_decode(line[start:])
            usage[event['sample_id']]+=int(event.get('generated_tokens') or 0);logged_rounds+=1
    assert len(usage)==75, 'Require token logs for every target'
    for record in report['records']:record['generated_tokens']=usage[record['sample_id']]
    report.update(generated_tokens=sum(usage.values()),logged_rounds=logged_rounds,
                  token_cost_source='worker eval_round logs, including retries')
    (out/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,'-m','evaluation.run_eval',
                    '--pred',str(out/'approved_combined_predictions.jsonl'),
                    '--reference',str(out.parent/'test50/eligible_references.jsonl'),
                    '--experiment','two_stage_hint_refinement',
                    '--model','qwen35_9b_lora_two_stage_final_2gpu_bs12'],cwd=ROOT,check=True)
    summary=json.loads((ROOT/'evaluation_results/two_stage_hint_refinement/summary.json').read_text())
    assert summary['final_warning']['warning_count']<=19
    mark=ROOT/'Mark1';manifest=json.loads((mark/'manifest.json').read_text())
    for rel,digest in manifest['sha256'].items():
        if rel.endswith('predictions.jsonl') and '/.work/' not in rel:
            assert hashlib.sha256((mark/rel).read_bytes()).hexdigest()==digest
    if report['approved_scores']:
        publish(out/'inputs.jsonl',out/'predictions.jsonl','two_stage',render=True,
                selected_scores=report['approved_scores'])
    report['publication_status']='approved scores published locally; held scores retain old results'
    (out/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'completion_report.json').write_text(json.dumps({
        'published_scores':report['approved_scores'],'held_scores':report['held_scores'],
        'final_warning':summary['final_warning'],
        'pitch_accuracy_at_50_cents':summary['metrics']['pitch_accuracy_at_50_cents'],
        'generated_tokens':report['generated_tokens'],'remote_publication':'pending'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
