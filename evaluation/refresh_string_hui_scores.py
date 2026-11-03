"""Add joint position scores without changing saved pitch/warning evidence."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .aggregate import per_piece
from .fingering import resolved_positions, string_hui_metrics
from .pitch import audit_plan
from .run_eval import load_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
CACHE = {}


def refresh(name):
    directory = ROOT / 'evaluation_results' / name
    config = json.loads((directory / 'config.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    path = Path(config['reference_path'])
    if str(path).startswith('/Volumes/F/work/Gunqin-agent/'):
        path = ROOT / path.relative_to('/Volumes/F/work/Gunqin-agent')
    if not path.is_absolute():
        path = ROOT / path
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_key = (sha, config['pitch_tolerance_cents'])
    if cache_key not in CACHE:
        scores = defaultdict(list)
        for ref in load_jsonl(path, 'sample_id').values():
            scores[ref['score_key']].append(ref)
        resolved = {}
        for score, refs in scores.items():
            payload = refs[0]['runtime_item']['input']
            notes = {}
            for ref in refs:
                actions = {int(a['source_index']): a.get('jianzi_text') or a.get('text') or ''
                           for a in ref.get('reference', {}).get('actions', [])}
                for note in ref['runtime_item']['input']['notes_without_jianzi']:
                    if note.get('index') is not None:
                        i = int(note['index'])
                        notes[i] = {**note, 'jianzi': actions.get(i, '')}
            audit = audit_plan({'metadata': payload['metadata'],
                'open_midi': (payload.get('normalized_tuning') or {}).get('open_midi'),
                'notes': [notes[i] for i in sorted(notes)]}, config['pitch_tolerance_cents'])
            for row in audit['details']:
                if row.get('index') is not None:
                    resolved[(score, int(row['index']))] = resolved_positions(row)
        CACHE[cache_key] = resolved
    events = [json.loads(line) for line in (directory / 'per_event.jsonl').read_text().splitlines()]
    assert len(events) == summary['events']
    assert len({(e['piece_id'], e['phrase_id'], e['event_id']) for e in events}) == len(events)
    assert not summary['warning_replay_incomplete_scores'], 'Full-score reference refresh requires complete predictions'
    prediction_path = Path(config['prediction_path'])
    if str(prediction_path).startswith('/Volumes/F/work/Gunqin-agent/'):
        prediction_path = ROOT / prediction_path.relative_to('/Volumes/F/work/Gunqin-agent')
    if not prediction_path.is_absolute():
        prediction_path = ROOT / prediction_path
    prediction_sha = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    original_audits = [e['pitch_audit'] for e in events]
    for event in events:
        event['reference']['string_hui_positions'] = CACHE[cache_key].get((event['piece_id'], event['event_id']), [])
        event['prediction']['string_hui_positions'] = resolved_positions(event['pitch_audit'])
    summary['metrics']['string_hui'] = string_hui_metrics(events)
    summary['string_hui_scoring_version'] = 'guqin-string-hui-1'
    config['string_hui_refresh'] = {'version': 'guqin-string-hui-1', 'reference_sha256': sha,
        'prediction_sha256': prediction_sha,
        'pitch_and_warning_evidence': 'unchanged; reference positions replayed over complete scores',
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / 'evaluation').glob('*.py')}}
    violations = [json.loads(line) for line in (directory / 'violations.jsonl').read_text().splitlines()]
    excluded = set(summary['warning_replay_incomplete_scores'])
    audited = {(e['piece_id'], e['phrase_id']) for e in events if e['piece_id'] not in excluded}
    assert original_audits == [e['pitch_audit'] for e in events]
    write_jsonl(directory / 'per_event.jsonl', events)
    write_jsonl(directory / 'per_piece.jsonl', per_piece(events, violations, config['pitch_tolerance_cents'], audited))
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (directory / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    assert hashlib.sha256(prediction_path.read_bytes()).hexdigest() == prediction_sha
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    print(name, json.dumps(summary['metrics']['string_hui']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', nargs='+')
    for name in parser.parse_args().experiments:
        refresh(name)
