"""Stream a final-Guqinizer-warning subset into the existing sharded viewer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args()
report=json.loads((a.directory/'final_warning_statistics.json').read_text())
cases=[c for c in report['cases'] if c['agent_stage']=='guqinization']
selected={c['trajectory_id'] for c in cases}
scores={c['score_key'] for c in cases}
counts=Counter();phrases=defaultdict(set)
for case in cases:
    for w in case['warnings']:
        counts[w['code']]+=1
        phrases[w['code']].add(case['trajectory_id'])
summary={'phrases':len(selected),'scores':len(scores),'warning_count':sum(counts.values()),
         'basis':report['audit_basis'],'data_state':report['data_state'],
         'categories':{code:{'warnings':count,'phrases':len(phrases[code])} for code,count in counts.most_common()}}
a.output_dir.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='guqin-final-warning-') as name:
    temp=Path(name);ids=set()
    with (temp/'messages.jsonl').open('w') as out:
        for line in (a.directory/'messages_train.jsonl').open():
            r=json.loads(line)
            if r.get('provenance',{}).get('source_trajectory_id') in selected:
                ids.add(r['sample_id']);out.write(line)
    audits=set()
    with (temp/'audit.jsonl').open('w') as out:
        for line in (a.directory/'teacher_trajectory_audit.jsonl').open():
            r=json.loads(line)
            if r['sample_id'] in ids:
                audits.add(r['sample_id']);out.write(line)
    assert ids==audits and len(ids)==2*len(selected)
    with (temp/'source.jsonl').open('w') as out:
        for line in a.source.open():
            r=json.loads(line)
            if r['score_key'] in scores:out.write(line)
    subprocess.run([sys.executable,str(Path(__file__).with_name('visualize_all_trajectories_hierarchical_sharded.py')),
        '--input',str(temp/'messages.jsonl'),'--audit',str(temp/'audit.jsonl'),
        '--source',str(temp/'source.jsonl'),'--output-dir',str(a.output_dir),
        '--notice',f'原始教师轨迹，尚未GLM脱敏。仅展示最终Guqinizer存档审计仍有warning的{len(selected)}个phrase；每个phrase包含两阶段，保留完整曲谱源上下文。'],check=True)
(a.output_dir/'warning_categories.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
