"""Combine completed repairs; newer repeat repair takes phrase-level precedence."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]/'ABC_J/agent_training/two_stage_audited'
older = BASE/'ordinary_open_reattack_repair'
newer = BASE/'repeat_warning_dependency_repair'
out = BASE/'open_repeat_combined_repair'
assert not out.exists(), 'Combined batch already exists; do not overwrite'
ids1 = set((older/'trajectory_ids.txt').read_text().split())
ids2 = set((newer/'trajectory_ids.txt').read_text().split())
assert len(ids1) == 248 and len(ids2) == 21 and len(ids1 & ids2) == 3
wanted = ids1 | ids2
raw = out/'regenerated_raw'
raw.mkdir(parents=True)
(out/'trajectory_ids.txt').write_text(''.join(p+'\n' for p in sorted(wanted)))
expected = {f'{p}-{s}-teacher-tools' for p in wanted for s in ('fingering_agent','guqinization')}
counts = {}
for filename, identity_key in [('messages_train.jsonl','sample_id'),
        ('teacher_trajectory_audit.jsonl','sample_id'),
        ('fingering_intermediates.jsonl','trajectory_id'), ('checkpoint.jsonl','trajectory_id')]:
    chosen = {}
    for directory, ids in [(older, ids1-ids2), (newer, ids2)]:
        for line in (directory/'regenerated_raw'/filename).open():
            row = json.loads(line)
            identity = row[identity_key]
            phrase = identity.removesuffix('-teacher-tools').rsplit('-',1)[0] if identity_key == 'sample_id' else identity
            if phrase not in ids:continue
            if filename == 'messages_train.jsonl':
                assert row['verification'].get('training_eligible'), identity
            elif filename == 'teacher_trajectory_audit.jsonl':
                assert row['verification'].get('replay_passed'), identity
            if filename != 'checkpoint.jsonl':assert identity not in chosen, identity
            chosen[identity] = line
    assert set(chosen) == (expected if identity_key == 'sample_id' else wanted), filename
    with (raw/filename).open('w') as f:
        for identity in sorted(chosen):f.write(chosen[identity])
    counts[filename] = len(chosen)
report = {'phrases':len(wanted),'overlap':len(ids1&ids2),
    'precedence':str(newer),'files':counts,'input_batches':[str(older),str(newer)]}
(out/'combined_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
