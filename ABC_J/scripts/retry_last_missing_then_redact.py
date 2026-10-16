"""One final Guqinizer attempt; explicitly exclude on failure, then redact."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'ABC_J/agent_training/two_stage_audited'
WORK=BASE/'last_missing_retry'
PHRASE='SMcjde29-p0016'


def read(path):
    for line in path.open():
        if line.strip():yield json.loads(line)


def main():
    WORK.mkdir(exist_ok=True)
    output=WORK/'regenerated_raw'
    assert not output.exists(),'Final retry already exists; inspect before rerunning'
    output.mkdir()
    old=BASE/'missing_stage_completion/regenerated_raw'
    finger=f'{PHRASE}-fingering_agent-teacher-tools'
    for name,key,value in [('messages_train.jsonl','sample_id',finger),
                           ('teacher_trajectory_audit.jsonl','sample_id',finger),
                           ('fingering_intermediates.jsonl','trajectory_id',PHRASE)]:
        selected=[r for r in read(old/name) if r.get(key)==value]
        assert len(selected)==1,(name,value)
        (output/name).write_text(json.dumps(selected[0],ensure_ascii=False)+'\n')
    source=BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    with (WORK/'score_context.jsonl').open('w') as out:
        for r in read(source):
            if r['score_key']=='SMcjde29':out.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('Retrying only SMcjde29-p0016 Guqinizer; existing Fingering reused; one stage attempt.',flush=True)
    process=subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/generate_teacher_tool_trajectories.py'),
        '--input',str(WORK/'score_context.jsonl'),'--output-dir',str(output),
        '--trajectory-id',PHRASE,'--stage','guqinization','--resume',
        '--intermediate-input',str(output/'fingering_intermediates.jsonl'),
        '--model','glm-5.3','--min-interval','0.5','--max-tool-rounds','24','--max-attempts','1',
        '--allow-private-reasoning-leakage','--include-guqinizer-no-op'],cwd=ROOT)
    public={r['sample_id']:r for r in read(output/'messages_train.jsonl')}
    success=(process.returncode==0 and public.get(f'{PHRASE}-guqinization-teacher-tools',{}).get('verification',{}).get('training_eligible'))
    result={'pending_phrases':[],'retried_phrase':PHRASE,'retry_succeeded':bool(success),
            'excluded_phrase_ids':[] if success else [PHRASE],
            'exclusion_authority':'User: retry once; omit this phrase if it still fails'}
    if success:
        (WORK/'trajectory_ids.txt').write_text(PHRASE+'\n')
        subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/replace_daiqi_repaired_teachers.py'),
            '--canonical',str(BASE/'teacher_latest_runtime_raw'),'--repair',str(output),
            '--ids',str(WORK/'trajectory_ids.txt'),'--report',str(WORK/'replacement_report.json')],cwd=ROOT,check=True)
        print('Final Guqinizer retry succeeded and was merged.',flush=True)
    else:
        print('Final retry failed; both stages of this phrase excluded from final training dataset as authorized.',flush=True)
    (WORK/'resolution_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/run_latest_two_stage_redaction.py'),
        '--resolution-report',str(WORK/'resolution_report.json')],cwd=ROOT,check=True)


if __name__=='__main__':main()
