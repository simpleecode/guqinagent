"""Compare pre/post Daiqi-state pitch replay at every accepted tool preview."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agents.ToolRuntime.runtime import pitch_audit_notes
from agents.abc_to_jianzipu.inverse_baseline import AUDIT


def old_auditor():
    path=ROOT/'scripts/audit_jianpu_jianzi_pitch.py'
    code=path.read_text()
    changes=[
        ("        # Dict order is the fallback recency order, not first-ever use order.\n        context['left_positions'].pop(str(resolved_finger), None)\n",''),
        ("    if requested:\n        return (requested, int(positions[requested]['string'])) if requested in positions else None\n",
         '    if requested and requested in positions:\n        return requested, int(positions[requested]["string"])\n'),
        ("    if active_finger in positions:\n        # An intervening open-string pluck clears melodic inheritance, but\n        # need not release a finger still pressing a DIFFERENT string.\n        return str(active_finger), int(positions[active_finger]['string'])\n",''),
    ]
    for before,after in changes:
        assert code.count(before)==1,before
        code=code.replace(before,after,1)
    begin=code.index('        stopped_component = next(')
    end=code.index('        context["sound_mode"] = None',begin)
    code=code[:begin]+'        remember_left_position(context, stopped_string, hui)\n'+code[end:]
    begin=code.index('        # Release only the sounded open strings;')
    end=code.index('        context["active_left_string"] = None',begin)
    code=code[:begin]+code[end:]
    module=types.ModuleType('pre_daiqi_fix_auditor');module.__file__=str(path)
    exec(compile(code,str(path),'exec'),module.__dict__)
    return module


def advance(module,context,notes,opens):
    values={}
    for note in notes:
        if module.is_structural_note(note):continue
        text=note.get('jianzi') or ''
        if module.advance_context_for_bare_rest(context,note,text):continue
        actual,reason=module.parse_jianzi(text,opens,context)
        values[note['index']]={'actual':actual,'reason':reason}
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    source={r['trajectory_id']:r for l in a.source.open() if (r:=json.loads(l))}
    history={(r['score_key'],r['phrase_id']):r for r in source.values()}
    intermediates={r['trajectory_id']:r['plan'] for l in (a.directory/'fingering_intermediates.jsonl').open() if (r:=json.loads(l))}
    old=old_auditor();affected={};scanned=0;previews=0
    for line in (a.directory/'teacher_trajectory_audit.jsonl').open():
        record=json.loads(line);sid=record['sample_id']
        phrase,stage=sid.removesuffix('-teacher-tools').rsplit('-',1)
        item=source[phrase];private=record['teacher_private']
        calls=private.get('tool_execution_log') or []
        plans=[('final',private['accepted_plan']['actions'])]
        for number,call in enumerate(calls):
            payload=(call.get('result') or {}).get('result') or {}
            if call.get('name')=='edit_plan' and isinstance(payload.get('preview_actions'),list):
                plans.append((f'edit_plan_{number+1}',payload['preview_actions']))
        if stage=='guqinization' and phrase in intermediates:
            plans.append(('initial_guqinizer',intermediates[phrase]['actions']))
        # Build/replay the full original read-only history once per stage.
        rows=pitch_audit_notes(item,{},history)
        current_indices={n['index'] for n in item['input']['notes_without_jianzi']}
        prefix=[n for n in rows if n['index'] not in current_indices]
        if not any(('带起' in (n.get('jianzi') or '') or '爪起' in (n.get('jianzi') or '')) for n in prefix) and not any(
                '带起' in str(plan) or '爪起' in str(plan) for _,plan in plans):
            scanned+=1;continue
        opens=item['input']['normalized_tuning']['open_midi']
        old_context=old.new_context();new_context=AUDIT.new_context()
        advance(old,old_context,prefix,opens);advance(AUDIT,new_context,prefix,opens)
        seen=set()
        for location,actions in plans:
            key=json.dumps(actions,ensure_ascii=False,sort_keys=True)
            if key in seen:continue
            seen.add(key);previews+=1
            by_index={int(x['source_index']):x for x in actions}
            notes=[dict(n,jianzi=(by_index.get(n['index']) or {}).get('jianzi_text') or '')
                   for n in item['input']['notes_without_jianzi']]
            before=advance(old,deepcopy(old_context),notes,opens)
            after=advance(AUDIT,deepcopy(new_context),notes,opens)
            changes=[]
            for n in notes:
                index=n['index']
                if before.get(index)!=after.get(index):
                    changes.append({'source_index':index,'event_index':n.get('event_index'),
                                    'jianzi':n['jianzi'],'before':before.get(index),'after':after.get(index)})
            if changes:
                entry=affected.setdefault(phrase,{'score_key':item['score_key'],'phrase_id':item['phrase_id'],
                    'title':item['input']['metadata'].get('score_title'),'stages':[]})
                entry['stages'].append({'sample_id':sid,'location':location,'changes':changes})
        scanned+=1
        if scanned%100==0:print(f'scanned {scanned} stages; affected {len(affected)} phrases',flush=True)
    report={'scanned_stages':scanned,'compared_plans':previews,'affected_phrases':len(affected),
            'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
            'auditor_sha256':hashlib.sha256((ROOT/'scripts/audit_jianpu_jianzi_pitch.py').read_bytes()).hexdigest(),
            'cases':affected}
    (a.output_dir/'affected_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output_dir/'trajectory_ids.txt').write_text(''.join(x+'\n' for x in sorted(affected)))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
