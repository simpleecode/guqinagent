"""Seal the paired inference pilot: same public input, weights and sampling."""
from pathlib import Path
import argparse,hashlib,json

ROOT=Path(__file__).resolve().parents[2]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=('pilot','full','youlan'),default='pilot')
    args=parser.parse_args()
    out=ROOT/'train/eval_outputs/warning_ablation'/args.scope
    source=out/'inputs.jsonl';rows=[json.loads(s) for s in source.open() if s.strip()]
    expected={'pilot':33,'full':979,'youlan':52}[args.scope]
    assert len(rows)==len({r['sample_id'] for r in rows})==expected
    assert len({r['score_key'] for r in rows})=={'pilot':4,'full':47,'youlan':1}[args.scope]
    assert not {'SCwngKTr','SeWok3pZ','SurzVQ02'} & {r['score_key'] for r in rows}
    if args.scope=='youlan':
        assert {r['score_key'] for r in rows}=={'ST4GAejh'}
    else:
        assert 'ST4GAejh' not in {r['score_key'] for r in rows}
    assert all(r.get('reference') is None and (r.get('runtime_item') or {}).get('reference_plan') is None for r in rows)
    adapter=Path('/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_two_stage_final_2gpu_bs12')
    model=Path('/home/20223393ljw/models/Qwen3.5-9B')
    files=['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
           'agents/ToolRuntime/warning_rendering.py','agents/ToolRuntime/warning_registry.py',
           'agents/ToolRuntime/hint_checks.py','agents/abc_to_jianzipu/teacher_trajectory.py',
           'scripts/audit_jianpu_jianzi_pitch.py','train/scripts/eval_two_stage_score.py',
           'train/scripts/run_vllm_eval_queue.py','ABC_J/scripts/generate_teacher_tool_trajectories.py']
    weights=list(adapter.glob('*.safetensors'))
    assert weights,'Adapter weights missing'
    report={'input_sha256':sha(source),'sample_ids':[r['sample_id'] for r in rows],
            'scores':len({r['score_key'] for r in rows}),'adapter':str(adapter),
            'adapter_sha256':{str(p):sha(p) for p in weights+[adapter/'adapter_config.json']},
            'base_model':str(model),'base_config_sha256':sha(model/'config.json'),
            'sampling':json.loads((ROOT/'train/configs/sampling_retry_schedule.json').read_text()),
            'runtime_sha256':{f:sha(ROOT/f) for f in files},
            'arms':{'on':'diagnostic warnings/hints and warning-driven followups',
                    'off':'no diagnostic warnings/hints or warning-driven followups; structural validation stays'},
            'offline_scoring':'same full runtime; never disable scoring; keep failures in denominator'}
    target=out/'experiment_manifest.json'
    if args.scope in ('full','youlan'):
        pilot=json.loads((out.parent/'pilot/experiment_manifest.json').read_text())
        for key in ('adapter_sha256','base_config_sha256','runtime_sha256','sampling'):
            assert report[key]==pilot[key],f'Full experiment differs from verified pilot: {key}'
    if target.exists():assert json.loads(target.read_text())==report,'Experiment identity changed; refuse to mix runs'
    else:target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'Paired {args.scope} manifest verified: {expected} identical public inputs, fixed adapter/runtime/sampling',flush=True)

if __name__=='__main__':main()
