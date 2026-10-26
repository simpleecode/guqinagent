"""Select linked contradictory repeat warnings in the same tool receipt."""
import json
from pathlib import Path
BASE = Path('ABC_J/agent_training/two_stage_audited')
old = BASE/'ordinary_open_reattack_repair'
replaced = set((old/'trajectory_ids.txt').read_text().split())
out = BASE/'repeat_warning_dependency_repair'
out.mkdir(exist_ok=True)
affected = {}
invalid_codes = {'missing_repeat_anchor', 'repeat_anchor_instruction_overlap',
                 'repeat_numbered_start_missing', 'repeat_numbered_pitch_mismatch',
                 'repeat_from_head_pitch_mismatch'}
for path, skip in [(BASE/'teacher_train_final/teacher_trajectory_audit.jsonl', replaced),
                   (old/'regenerated_raw/teacher_trajectory_audit.jsonl', set())]:
    for line in path.open():
        r = json.loads(line)
        phrase = r['sample_id'].removesuffix('-teacher-tools').rsplit('-',1)[0]
        if phrase in skip:continue
        for turn, call in enumerate(r['teacher_private'].get('tool_execution_log') or []):
            if call.get('name') != 'edit_plan':continue
            result = call.get('result') or {};result = result.get('result',result)
            warnings = result.get('warnings') or []
            invalid = {w.get('command_source_index',w.get('source_index')) for w in warnings
                       if w.get('code') in invalid_codes}
            for w in warnings:
                if w.get('code') == 'repeat_omission_marker_missing' and w.get('command_source_index') in invalid:
                    affected.setdefault(phrase,[]).append({'sample_id':r['sample_id'],
                        'turn':turn,'command_source_index':w['command_source_index'],
                        'warnings':warnings,'source':str(path)})
(out/'trajectory_ids.txt').write_text(''.join(k+'\n' for k in sorted(affected)))
report = {'affected_phrases':len(affected),'overlap_with_open_repair':len(set(affected)&replaced),
          'selection':'same edit_plan receipt; omission references the invalid command source index',
          'cases':affected}
(out/'affected_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
