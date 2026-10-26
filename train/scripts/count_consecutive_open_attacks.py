"""Count adjacent single-string open attacks using the shared pitch parser."""
import collections
import json
import re
from pathlib import Path
from agents.abc_to_jianzipu.reference_parser import AUDIT
from agents.abc_to_jianzipu.score_structure import is_structural_note

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'ABC_J/agent_training/two_stage_audited'
sources = {}
for line in (BASE/'pitch_eligible_train_test/inferred_trajectories_train.jsonl').open():
    r = json.loads(line)
    sources[(r['score_key'], r['phrase_id'])] = r
plans = collections.defaultdict(dict)
for line in (BASE/'teacher_train_final/teacher_trajectory_audit.jsonl').open():
    r = json.loads(line)
    m = re.match(r'(.+)-(p\d+)-(fingering_agent|guqinization)-', r['sample_id'])
    if m:
        score, phrase, stage = m.groups()
        plans[stage][(score, phrase)] = {a['source_index']: a.get('jianzi_text')
            for a in r['teacher_private']['accepted_plan']['actions']}
plans['source_reference'] = {key: {a['source_index']: a.get('jianzi_text')
    if a.get('jianzi_text') is not None else a.get('text', '')
    for a in r['reference_plan']['actions']} for key, r in sources.items()}

result = {}
for stage, stage_plans in plans.items():
    ctx = None
    last_score = None
    last_index = None
    previous = None
    count = collections.Counter()
    phrases = set()
    scores = set()
    plain_phrases = set()
    examples = []
    for key in sorted(sources):
        score, phrase = key
        row = sources[key]
        open_midi = row['input']['normalized_tuning']['open_midi']
        for n in row['input']['notes_without_jianzi']:
            idx = n['index']
            if score != last_score or last_index is None or idx != last_index + 1:
                ctx = AUDIT.new_context()
                previous = None
            last_score, last_index = score, idx
            if is_structural_note(n):
                continue
            text = stage_plans.get(key, {}).get(idx) or ''
            note = {**n, 'jianzi': text}
            if AUDIT.advance_context_for_bare_rest(ctx, note, text):
                previous = None
                continue
            actual, reason = AUDIT.parse_jianzi(text, open_midi, ctx)
            attacks = ctx.get('current_note_attacks') or []
            opened = (reason is None and len(actual) == 1 and len(attacks) == 1
                      and attacks[0]['mode'] == 'open')
            # Strict adjacent rows, ignoring only structural barlines. Empty
            # rows, ties, ornaments and unparsed gestures break this chain.
            plain = bool(re.fullmatch(r'(?:散音|散)?[抹挑勾剔擘托劈打摘搓][一二三四五六七1-7]弦', text))
            current = {'phrase': score+'-'+phrase, 'event': n.get('event_index'),
                       'source_index': idx, 'text': text, 'plain': plain,
                       'string': attacks[0]['string']} if opened else None
            if current and previous and current['string'] == previous['string']:
                count['pairs'] += 1
                phrases.add(current['phrase']); scores.add(score)
                if not previous.get('in_run'):
                    count['runs'] += 1
                current['in_run'] = True
                if current['plain'] and previous['plain']:
                    count['plain_pairs'] += 1
                    plain_phrases.add(current['phrase'])
                if len(examples) < 12:
                    examples.append({'previous': previous, 'current': current})
            previous = current
    result[stage] = {**count, 'affected_phrases': len(phrases),
        'affected_scores': len(scores), 'plain_affected_phrases': len(plain_phrases),
        'examples': examples}
print(json.dumps(result, ensure_ascii=False, indent=2))
