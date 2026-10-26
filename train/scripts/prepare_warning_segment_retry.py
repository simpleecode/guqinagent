"""Freeze whole-score model history; rerun only warning-bearing Guqinizer segments."""
from pathlib import Path
import sys,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train.scripts.visualize_completed_eval import rows,write_rows,validate
from train.scripts.run_vllm_eval_queue import balanced_shards


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--score')
    parser.add_argument('--phrase',action='append')
    parser.add_argument('--current-warnings',action='store_true',help='Audit current published model results; include p0062 prompt repair')
    parser.add_argument('--targets-from',type=Path,help='Reuse an audited selection without redoing the full audit')
    parser.add_argument('--exclude-target',action='append',default=[])
    parser.add_argument('--baseline-predictions',type=Path,help='Use a verified candidate snapshot as model history instead of published predictions')
    parser.add_argument('--work-name',default='warning_segment_retry')
    args=parser.parse_args()
    if bool(args.score)!=bool(args.phrase):parser.error('--score and --phrase must be supplied together')
    if not args.work_name or Path(args.work_name).name!=args.work_name or args.work_name.startswith('.'):
        parser.error('work-name must be a simple directory name')
    root=ROOT/'train/eval_outputs/two_stage'
    out=root/'.work'/args.work_name
    assert not out.exists(),'Do not overwrite progress'
    if args.targets_from:
        assert not args.score and not args.current_warnings
        audited=json.loads(args.targets_from.read_text())
        targets=set(audited['target_ids'])-set(args.exclude_target)
        violations=[w for w in audited['warnings'] if f"{w['piece_id']}-{w['phrase_id']}" in targets]
        scores={sid.rsplit('-p',1)[0] for sid in targets}
    elif args.current_warnings:
        from evaluation.rules import replay_final_warnings
        assert not args.score, 'Use either current-warnings or explicit score'
        violations=[]
        black=set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or folder.name.startswith('.') or folder.name in black:
                continue
            if not (folder/'predictions.jsonl').exists():continue
            source=rows(folder/'inputs.jsonl');pred=rows(folder/'predictions.jsonl')
            _,plans=validate(source,pred)
            violations.extend(replay_final_warnings([(plans[s['sample_id']],s) for s in source]))
        targets={f"{v['piece_id']}-{v['phrase_id']}" for v in violations}
        targets.add('S35eJqgV-p0062')
        scores={sid.rsplit('-p',1)[0] for sid in targets}
    elif args.score:
        from train.scripts.visualize_completed_eval import safe_name
        safe_name(args.score)
        for phrase in args.phrase:safe_name(phrase)
        targets={f'{args.score}-{p}' for p in args.phrase};scores={args.score};violations=[]
    else:
        violations=rows(ROOT/'evaluation_results/two_stage_hint_refinement/violations.jsonl')
        targets={f"{v['piece_id']}-{v['phrase_id']}" for v in violations}
        scores={v['piece_id'] for v in violations}
        assert len(targets)==10 and len(scores)==9
    inputs=[];predictions=[]
    candidate=rows(args.baseline_predictions) if args.baseline_predictions else None
    for score in sorted(scores):
        inputs.extend(rows(root/score/'inputs.jsonl'))
        predictions.extend([r for r in candidate if r['score_key']==score]
                           if candidate is not None else rows(root/score/'predictions.jsonl'))
    _,finals=validate(inputs,predictions)
    assert targets<=set(finals)
    out.mkdir();(out/'workers').mkdir()
    write_rows(out/'inputs.jsonl',inputs);write_rows(out/'baseline.jsonl',predictions)
    frozen=[r for r in predictions if r['sample_id'] not in targets]
    write_rows(out/'frozen_predictions.jsonl',frozen)
    for i,shard in enumerate(balanced_shards(inputs,min(16,len(scores)))):
        write_rows(out/'workers'/f'predictions_{i:02d}.jsonl',
                   [finals[r['sample_id']] for r in shard if r['sample_id'] not in targets])
    files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
           'agents/ToolRuntime/warning_registry.py','agents/ToolRuntime/warning_rendering.py',
           'agents/ToolRuntime/hint_checks.py','scripts/audit_jianpu_jianzi_pitch.py',
           'train/scripts/eval_two_stage_score.py','train/scripts/run_vllm_eval_queue.py',
           'ABC_J/scripts/generate_teacher_tool_trajectories.py',
           'train/configs/sampling_retry_schedule.json']
    report={'target_ids':sorted(targets),'target_phrases':len(targets),'scores':len(scores),
            'baseline_source':str(args.baseline_predictions) if args.baseline_predictions else 'published',
            'total_phrases':len(inputs),'frozen_phrases':len(frozen),'warnings':violations,
            'policy':'Guqinizer only; base and every non-target segment including suffix frozen; never expand reruns',
            'sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest()
                      for n in ['inputs.jsonl','baseline.jsonl','frozen_predictions.jsonl']},
            'runtime_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (out/'selection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('warnings','runtime_sha256','sha256')},ensure_ascii=False))


if __name__=='__main__':main()
