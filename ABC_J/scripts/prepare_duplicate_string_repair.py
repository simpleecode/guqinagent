"""Select phrases with a redundant same-string prefix in authored jianzi."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.jianzi_surface_checks import duplicate_string_prefixes as hits
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
OUT = BASE / 'duplicate_string_repair'
# Only inspect actual edit text / accepted plan text, never prose or prompts.
# Do not cross punctuation separating multi-string components.
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert hits('泛音二弦四徽勾二弦')
    assert hits('大指2弦七徽挑二弦')
    assert not hits('泛音四徽勾二弦')
    assert not hits('撮（大指七徽六弦泛音＋中指七徽一弦泛音）')
    assert not hits('大指七徽历三二弦')
    assert hits('名指二弦徽外剔二弦')
    assert not hits('散勾四弦 大指七徽勾四弦')
    assert not hits('拂一弦至五弦名指十徽八分抹五弦')
    assert not (OUT / 'regenerated_raw').exists(), 'Repair generation already started'
    OUT.mkdir(exist_ok=True)
    source = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    sources = {r['trajectory_id']: r for line in source.open() if (r := json.loads(line))}
    affected = {}

    def add(row, index, text, location, turn=None):
        found = hits(text)
        if not found:
            return
        sid = row['sample_id']
        phrase = sid.removesuffix('-teacher-tools').rsplit('-', 1)[0]
        assert phrase in sources
        case = affected.setdefault(phrase, {'score_key': sources[phrase]['score_key'], 'cases': []})
        case['cases'].append({'sample_id': sid, 'index': index, 'index_kind':
                             'event_index' if location == 'submitted' else 'source_index',
                             'text': text, 'location': location, 'turn': turn, 'hits': found})

    messages = BASE / 'teacher_train_final/messages_train.jsonl'
    audit = BASE / 'teacher_train_final/teacher_trajectory_audit.jsonl'
    for line in messages.open():
        row = json.loads(line)
        for turn, message in enumerate(row['messages']):
            if message['role'] != 'assistant':
                continue
            for call in message.get('tool_calls') or []:
                function = call.get('function', {})
                if function.get('name') != 'edit_plan':
                    continue
                arguments = function.get('arguments') or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                entries = arguments.get('jianzi_rows') or []
                if isinstance(entries, dict):
                    entries = list(entries.items())
                for entry in entries:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        add(row, entry[0], entry[1], 'submitted', turn)
                    elif isinstance(entry, dict):
                        for key, value in entry.items():
                            if isinstance(value, str):
                                add(row, key, value, 'submitted', turn)
    for line in audit.open():
        row = json.loads(line)
        for action in row['teacher_private']['accepted_plan']['actions']:
            add(row, action['source_index'], action.get('jianzi_text') or '', 'final')
    assert affected
    report = {'affected_phrases': len(affected),
              'scores': len({case['score_key'] for case in affected.values()}),
              'selection': 'any actual edit_plan row or accepted final plan; includes later-corrected cases',
              'source_sha256': sha(source), 'messages_sha256': sha(messages),
              'cases': affected}
    (OUT / 'affected_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'trajectory_ids.txt').write_text(''.join(p + '\n' for p in sorted(affected)))
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
