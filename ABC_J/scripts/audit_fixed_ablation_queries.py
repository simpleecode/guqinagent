#!/usr/bin/env python3
"""Audit query coverage using the project's sounding-pitch resolver."""
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import target_midis  # noqa: E402

data = ROOT / 'ABC_J/agent_training/single_stage_fixed_reference_ablation'
source = ROOT / 'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl'
items = {r['trajectory_id']: r for r in map(json.loads, source.open())}
public = {r['sample_id']: r for r in map(json.loads, (data/'messages_train.jsonl').open())}
counts = Counter()
problems = []
for line in (data/'tool_audit.jsonl').open():
    audit = json.loads(line)
    row = public[audit['sample_id']]
    item = items[row['provenance']['source_trajectory_id']]
    calls = audit['calls']
    queried = set(calls[0]['arguments']['event_indices'])
    expected = set()
    for note in item['input']['notes_without_jianzi']:
        event = note.get('event_index')
        abc = str(note.get('abc') or '')
        jianpu = str(note.get('jianpu') or '')
        if event is not None and not abc.startswith(('-', 'z', 'x')) and '休止' not in jianpu:
            try:
                sounding = target_midis(item, int(note['index']))
            except ValueError as exc:
                if 'has no sounding target' not in str(exc):
                    raise
                sounding = []
            if sounding:
                expected.add(int(event))
    missing, extra = expected - queried, queried - expected
    if missing:
        problems.append({'id': item['trajectory_id'], 'missing': sorted(missing)})
    counts['extra_query_events'] += len(extra)
    counts['extra_query_phrases'] += bool(extra)
    counts['phrases'] += 1
(data/'query_validation_report.json').write_text(json.dumps({'valid': not problems, 'counts': counts, 'problems': problems}, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'valid': not problems, 'counts': counts, 'problem_count': len(problems), 'examples': problems[:3]}, ensure_ascii=False))
raise SystemExit(bool(problems))
