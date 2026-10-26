"""Verify isolated warning repairs without regenerating any frozen suffix."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sys,subprocess,hashlib,zipfile,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import rows,write_rows,validate,publish,refresh_index
from evaluation.rules import replay_final_warnings


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-name',default='warning_segment_retry')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--publish-verified',action='store_true',help='Use existing verified snapshot only if input/prediction hashes match')
    args=parser.parse_args()
    assert Path(args.work_name).name==args.work_name and not args.work_name.startswith('.')
    root=ROOT/'train/eval_outputs/two_stage';out=root/'.work'/args.work_name
    selection=json.loads((out/'selection_report.json').read_text())
    snapshot_hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest()
                     for n in ('inputs.jsonl','predictions.jsonl','baseline.jsonl','frozen_predictions.jsonl')}
    if args.publish_verified:
        report=json.loads((out/'verification_report.json').read_text())
        assert report['snapshot_sha256']==snapshot_hashes
        approved=report['approved_scores'];held=report['held_scores']
        return finish(root,out,approved,held)
    for name,digest in selection['sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    inputs,predictions=rows(out/'inputs.jsonl'),rows(out/'predictions.jsonl')
    sources,finals=validate(inputs,predictions);_,baseline=validate(inputs,rows(out/'baseline.jsonl'))
    frozen=rows(out/'frozen_predictions.jsonl');targets=set(selection['target_ids'])
    assert len(finals)==selection['total_phrases'] and len(targets)==selection['target_phrases'] and len(frozen)==selection['frozen_phrases']
    assert all(finals[r['sample_id']]==r for r in frozen)
    both=selection.get('rerun_stages')=='both'
    if not both:assert all(finals[sid]['base']==baseline[sid]['base'] for sid in targets)
    else:
        for sid in targets:
            for stage in ('base','guqinizer'):
                assert finals[sid][stage].get('trace'), (sid,stage,'no actual trace')
                assert finals[sid][stage] != baseline[sid][stage], (sid,stage,'not regenerated')
    grouped=defaultdict(list)
    for source in inputs:grouped[source['score_key']].append(source)
    approved=[];held={};new_all=[];checks=[]
    def pos(w):return w['phrase_id'],w['event_id'],w['violation_type']
    for score,source_rows in sorted(grouped.items()):
        old=replay_final_warnings([(baseline[s['sample_id']],s) for s in source_rows])
        new=replay_final_warnings([(finals[s['sample_id']],s) for s in source_rows]);new_all.extend(new)
        introduced={pos(w) for w in new}-{pos(w) for w in old}
        if introduced:held[score]={'introduced':sorted(introduced),'old':len(old),'new':len(new)}
        else:approved.append(score)
    for sid in sorted(targets):
        r=finals[sid];trace=r['guqinizer'].get('trace',[])
        checks.append({'sample_id':sid,'passes':r.get('guqinizer_passes'),
            'edit_calls':sum(c.get('name')=='edit_plan' for t in trace for c in t.get('tool_calls',[])),
            'reminders':sum((t.get('stop_check') or {}).get('action')=='remind' for t in trace),
            'stop_checks':[t['stop_check'] for t in trace if 'stop_check' in t],
            'final_reply':r['guqinizer'].get('final_reply')})
    full_inputs=[];full_predictions=[]
    for manifest in sorted(root.glob('*/result_manifest.json')):
        full_inputs.extend(rows(manifest.parent/'inputs.jsonl'))
        full_predictions.extend(rows(manifest.parent/'predictions.jsonl'))
    combined={r['sample_id']:r for r in full_predictions}
    combined.update({sid:r for sid,r in finals.items() if sources[sid]['score_key'] in approved})
    merged=[combined[s['sample_id']] for s in full_inputs];validate(full_inputs,merged)
    assert len(merged)==979
    write_rows(out/'combined_inputs.jsonl',full_inputs);write_rows(out/'combined_predictions.jsonl',merged)
    report={'snapshot_sha256':snapshot_hashes,'unique_ids':len(finals),'frozen_exact':len(frozen),'base_exact':0 if both else len(targets),
            'rerun_stages':'both' if both else 'guqinizer','approved_scores':approved,
            'held_scores':held,'raw_final_warnings':new_all,'raw_final_warning_count':len(new_all),
            'raw_warning_codes':dict(Counter(w['violation_type'] for w in new_all)),
            'checks':checks,'policy':'no suffix regeneration; reject scores with new violations'}
    (out/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if args.audit_only:return
    finish(root,out,approved,held)


def finish(root,out,approved,held):
    subprocess.run([sys.executable,'-m','evaluation.run_eval','--pred',str(out/'combined_predictions.jsonl'),
        '--reference',str(root/'.work/test50/eligible_references.jsonl'),
        '--experiment','two_stage_current','--model','qwen35_9b_lora_two_stage_final_2gpu_bs12'],cwd=ROOT,check=True)
    mark=ROOT/'Mark1';m=json.loads((mark/'manifest.json').read_text())
    for f,d in m['sha256'].items():
        if f.endswith('predictions.jsonl') and '/.work/' not in f:assert hashlib.sha256((mark/f).read_bytes()).hexdigest()==d
    if approved:
        publish(out/'inputs.jsonl',out/'predictions.jsonl','two_stage',render=True,selected_scores=approved,
                references=root/'.work/test50/eligible_references.jsonl')
        subprocess.run(['node','/Users/lijiawei/.codex/plugins/cache/openai-primary-runtime/pdf/26.1022.11436/skills/pdf/container_tools/mark_artifact_operation_started.mjs',
            '--operation-kind','create','--expected-output-count',str(len(approved)),'--output-format','pdf'],check=True)
        for score in approved:
            d=root/score
            subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/visualize_running_eval_score.py'),
                '--score',score,'--input',str(d/'inputs.jsonl'),'--pulled',str(d/'predictions.jsonl'),
                '--references',str(root/'.work/test50/eligible_references.jsonl'),
                '--skip-pull','--output',str(d/f'{score}.html'),'--notation-pdf',str(d/f'{score}.pdf')],check=True)
    refresh_index(root)
    pdfs=[p.parent/f'{p.parent.name}.pdf' for p in sorted(root.glob('*/result_manifest.json'))]
    assert len(pdfs)==47
    for p in pdfs:subprocess.run(['pdfinfo',str(p)],check=True,stdout=subprocess.DEVNULL)
    with zipfile.ZipFile(root/'all_scores_pdf.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in pdfs:z.write(p,f'{p.parent.name}/{p.name}')
    (out/'completion_report.json').write_text(json.dumps({'published_scores':approved,'held_scores':held,
        'pdf_count':len(pdfs),'remote_sync':'pending','verified':True},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
