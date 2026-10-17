"""Read-only corpus investigation; emits summary to stdout."""
import collections
import json
import re
from pathlib import Path
from agents.ToolRuntime.runtime import missing_attack_warnings

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
sources = {}
reference_stats = collections.Counter()
for line in (BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl').open():
    x = json.loads(line)
    key = (x['score_key'], x['phrase_id'])
    refs = {a['source_index']: a for a in x['reference_plan']['actions']}
    sources[key] = {'input': x['input'], 'refs': refs}
    reference_actions = {k: {'jianzi_text': a.get('jianzi_text') if a.get('jianzi_text') is not None else a.get('text', '')}
                         for k, a in refs.items()}
    ws = missing_attack_warnings(x, reference_actions)
    reference_stats['phrases'] += bool(ws)
    reference_stats['events'] += len(ws)

stats = collections.defaultdict(collections.Counter)
cases = collections.defaultdict(list)
for line in (BASE/'teacher_train_final/teacher_trajectory_audit.jsonl').open():
    x = json.loads(line)
    match = re.match(r'(.+)-(p\d+)-(fingering_agent|guqinization)-', x['sample_id'])
    if not match:
        continue
    score, phrase, stage = match.groups()
    source = sources[(score, phrase)]
    plan = x['teacher_private']['accepted_plan']
    actions = {a['source_index']: a for a in plan['actions']}
    ws = missing_attack_warnings(source, actions)
    c = stats[stage]
    c['total_phrases'] += 1
    c['affected_phrases'] += bool(ws)
    c['events'] += len(ws)
    notes = {n['index']: n for n in source['input']['notes_without_jianzi']}
    for w in ws:
        idx = w['source_index']
        prev = w['previous_attack_source_index']
        ref = source['refs'].get(idx, {})
        reftext = ref.get('jianzi_text') if ref.get('jianzi_text') is not None else ref.get('text', '')
        c['reference_nonblank' if reftext else 'reference_blank'] += 1
        if notes[idx].get('abc') == notes[prev].get('abc'):
            c['same_abc_including_duration'] += 1
        if re.sub(r'\d|/', '', notes[idx].get('abc', '')) == re.sub(r'\d|/', '', notes[prev].get('abc', '')):
            c['same_pitch_spelling'] += 1
        if len(cases[stage]) < 12:
            cases[stage].append({'phrase': score+'-'+phrase, 'event': notes[idx].get('event_index'),
                                 'abc': notes[idx].get('abc'), 'previous': actions[prev].get('jianzi_text'),
                                 'reference': reftext})
print(json.dumps({'teacher_final': stats, 'source_reference': reference_stats, 'examples': cases}, ensure_ascii=False, indent=2))
