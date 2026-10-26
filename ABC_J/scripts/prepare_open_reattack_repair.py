"""Select ordinary open reattacks absent at the same location in annotation."""
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import pitch_audit_notes
from agents.abc_to_jianzipu.reference_parser import AUDIT

base = ROOT/'ABC_J/agent_training/two_stage_audited'
out = base/'ordinary_open_reattack_repair'
out.mkdir(exist_ok=True)
source = base/'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
sources = {r['trajectory_id']:r for r in map(json.loads,source.open())}
affected = {}
counts = {}
for line in (base/'teacher_train_final/teacher_trajectory_audit.jsonl').open():
    r = json.loads(line)
    key = r['sample_id'].removesuffix('-teacher-tools').rsplit('-',1)[0]
    if key not in sources:
        continue
    item = sources[key]
    refs = {a['source_index']: {**a, 'jianzi_text':a.get('jianzi_text')
            if a.get('jianzi_text') is not None else a.get('text','')}
            for a in item['reference_plan']['actions']}
    def check(actions):
        notes = pitch_audit_notes(item, actions)
        return AUDIT.ordinary_open_reattack_warnings({
            'metadata':item['input']['metadata'],
            'open_midi':item['input']['normalized_tuning']['open_midi'], 'notes':notes})
    allowed = {(w['previous_index'],w['index']) for w in check(refs)}
    private = r['teacher_private']
    snapshots = [('final',private['accepted_plan']['actions'])]
    for turn, call in enumerate(private.get('tool_execution_log') or []):
        if call.get('name') != 'edit_plan':continue
        result = call.get('result') or {}
        result = result.get('result',result)
        if result.get('preview_actions'):
            snapshots.append((f'tool_{turn}',result['preview_actions']))
    for location, actions in snapshots:
        hits = [w for w in check({a['source_index']:a for a in actions})
                if (w['previous_index'],w['index']) not in allowed]
        if hits:
            case = affected.setdefault(key, {'score_key':item['score_key'],'cases':[]})
            case['cases'].append({'sample_id':r['sample_id'],'location':location,'warnings':hits})
            counts[location.split('_')[0]] = counts.get(location.split('_')[0],0)+len(hits)
report = {'affected_phrases':len(affected),'scores':len({c['score_key'] for c in affected.values()}),
          'selection':'Any final or tool preview in either stage; same adjacent source-index pair absent in annotation; phrase-local conservative selection',
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'cases':affected,'occurrences':counts}
(out/'affected_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(out/'trajectory_ids.txt').write_text(''.join(k+'\n' for k in sorted(affected)))
print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False))
