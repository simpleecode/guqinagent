"""Wait for the repair pipeline, fill missing stages, then verified replacement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'ABC_J/agent_training/two_stage_audited'
CANONICAL=BASE/'teacher_latest_runtime_raw'
OUTPUT=BASE/'missing_stage_completion'
SOURCE=BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'


def read_rows(path):
    if not path.exists():return
    for line in path.open():
        if line.strip():yield json.loads(line)


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-pid',type=int,required=True)
    args=p.parse_args()
    print(f'Waiting for Daiqi repair pipeline PID {args.dependency_pid}; no API requests until it exits.',flush=True)
    while True:
        proc=subprocess.run(['ps','-p',str(args.dependency_pid),'-o','command='],text=True,capture_output=True)
        if proc.returncode or not proc.stdout.strip():break
        if 'run_daiqi_state_repair.sh' not in proc.stdout:
            raise RuntimeError('Dependency PID was reused; refusing to infer successful completion')
        time.sleep(5)
    assert (BASE/'daiqi_state_repair/replacement_report.json').exists(), 'Repair did not reach replacement; completion remains blocked'
    expected=json.loads((BASE/'daiqi_state_repair/affected_report.json').read_text())['source_sha256']
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==expected
    sources={r['trajectory_id']:r for r in read_rows(SOURCE)}
    public={r['sample_id']:r for r in read_rows(CANONICAL/'messages_train.jsonl')}
    stages={}
    for r in public.values():
        phrase=r['provenance']['source_trajectory_id']
        stages.setdefault(phrase,set()).add(r['agent_stage'])
    missing={phrase:sorted({'fingering_agent','guqinization'}-stages.get(phrase,set()))
             for phrase in sources if stages.get(phrase,set())!={'fingering_agent','guqinization'}}
    if not missing:
        print('All phrases already complete; nothing to generate.',flush=True);return
    fresh=sorted(phrase for phrase in missing if 'fingering_agent' in missing[phrase])
    guqinizer=sorted(set(missing)-set(fresh))
    output=OUTPUT/'regenerated_raw'
    assert not output.exists(),'Completion output already exists; inspect it before resuming'
    output.mkdir(parents=True)
    wanted_fingers={f'{phrase}-fingering_agent-teacher-tools' for phrase in guqinizer}
    with (output/'messages_train.jsonl').open('w') as f:
        for sid in sorted(wanted_fingers):
            assert public[sid]['verification'].get('training_eligible')
            f.write(json.dumps(public[sid],ensure_ascii=False)+'\n')
    copied=set()
    with (output/'teacher_trajectory_audit.jsonl').open('w') as f:
        for r in read_rows(CANONICAL/'teacher_trajectory_audit.jsonl'):
            if r['sample_id'] in wanted_fingers:
                copied.add(r['sample_id']);f.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert copied==wanted_fingers
    intermediates={r['trajectory_id']:r for r in read_rows(CANONICAL/'fingering_intermediates.jsonl')}
    assert set(guqinizer)<=intermediates.keys()
    intermediate_path=OUTPUT/'reused_fingering_intermediates.jsonl'
    with intermediate_path.open('w') as f:
        for phrase in guqinizer:f.write(json.dumps(intermediates[phrase],ensure_ascii=False)+'\n')
    (output/'fingering_intermediates.jsonl').write_text(intermediate_path.read_text())
    (OUTPUT/'trajectory_ids.txt').write_text(''.join(x+'\n' for x in sorted(missing)))
    (OUTPUT/'fresh_ids.txt').write_text(''.join(x+'\n' for x in fresh))
    (OUTPUT/'guqinizer_ids.txt').write_text(''.join(x+'\n' for x in guqinizer))
    plan={'missing_phrases':len(missing),'regenerate_both':fresh,'reuse_fingering_and_generate_guqinizer':guqinizer,
          'missing_stages':missing,'source_sha256':expected}
    (OUTPUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    common=[ROOT/'ABC_J/scripts/run_teacher_batch_parallel.py','--input',SOURCE,'--output-dir',output,
            '--target-count',len(sources),'--workers',7,'--balance-by-score','--model','glm-5.3',
            '--min-interval','0.5','--max-tool-rounds',24,'--max-attempts',3,
            '--allow-private-reasoning-leakage','--include-guqinizer-no-op']
    if fresh:run(*common,'--trajectory-id-file',OUTPUT/'fresh_ids.txt')
    if guqinizer:run(*common,'--trajectory-id-file',OUTPUT/'guqinizer_ids.txt',
                    '--stage','guqinization','--worker-prefix','guqinizer',
                    '--intermediate-input',intermediate_path)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==expected
    run(ROOT/'ABC_J/scripts/replace_daiqi_repaired_teachers.py',
        '--canonical',CANONICAL,'--repair',output,'--ids',OUTPUT/'trajectory_ids.txt',
        '--report',OUTPUT/'replacement_report.json')
    run(ROOT/'ABC_J/scripts/statistics_teacher_final.py','--directory',CANONICAL,'--source',SOURCE)
    run(ROOT/'ABC_J/scripts/visualize_final_warning_subset.py','--directory',CANONICAL,'--source',SOURCE,
        '--output-dir',CANONICAL/'final_warning_visualization/visualization_sharded')
    print('Missing-stage completion and verified merge finished.',flush=True)


if __name__=='__main__':main()
