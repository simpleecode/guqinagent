#!/usr/bin/env python3
"""Verify complete modern evaluations and save costs and common-event comparisons."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    root = ROOT / 'train/eval_outputs/modern_comparison'
    source = rows(root / 'inputs.jsonl')
    assert hashlib.sha256((root / 'inputs.jsonl').read_bytes()).hexdigest() == '8c37783873f22f07f397e96ebe07d7009e086c69e7687f57a03a45d3c2079400'
    sources = {r['sample_id']: r for r in source}
    assert len(source) == len(sources) == 91 and len({r['score_key'] for r in source}) == 10
    refs = rows(ROOT / 'train/eval_inputs/modern_sealed/evaluation_pairs_test.jsonl')
    assert len(refs) == 91 and {r['sample_id'] for r in refs} == sources.keys()
    assert all(r['input_sha256'] == sources[r['sample_id']]['input_sha256'] and r['runtime_item']['input'] == sources[r['sample_id']]['runtime_item']['input'] for r in refs)
    report = {'scores': 10, 'phrases': 91, 'arms': {}, 'common_events': {}}
    event_sets = {}
    for method in ('two_stage', 'single_stage', 'two_stage_off', 'fixed_workflow'):
        pred = rows(root / method / 'predictions.jsonl')
        assert len(pred) == len({r['sample_id'] for r in pred}) == 91
        assert {r['sample_id'] for r in pred} == sources.keys()
        for r in pred:
            assert r['protocol_valid'] and r['input_sha256'] == sources[r['sample_id']]['input_sha256']
            assert r['score_key'] == sources[r['sample_id']]['score_key']
            stages = ('base', 'guqinizer') if method in ('two_stage', 'two_stage_off') else ('single_stage',)
            assert all(r.get(s, {}).get('ok') and isinstance(r[s].get('plan'), dict) for s in stages)
            if method == 'two_stage_off':
                assert r['guqinizer_passes'] == 1
                for stage in stages:
                    for t in r[stage].get('trace', []):
                        public = json.dumps(t.get('tool_results', []), ensure_ascii=False)
                        dynamic = json.dumps([m for m in t.get('request_messages', []) if m.get('role') in ('user', 'tool')], ensure_ascii=False)
                        assert not any(v in public or v in dynamic for v in (':warning:', ':hint:', '警告｜', '当前方案仍有告警', '上一轮出现音高警告'))
                        assert not t.get('pitch_gate', {}).get('warned_now')
                        assert t.get('stop_check', {}).get('action') != 'remind'
        experiment = ROOT / 'evaluation_results' / ('modern_' + method)
        summary = json.loads((experiment / 'summary.json').read_text())
        assert summary['matched_samples'] == 91 and not summary['unmatched_prediction_ids']
        assert not summary['warning_replay_incomplete_scores'] and not summary['tone_type_replay_incomplete_scores']
        assert sum(summary['rule_warning_audited_phrases_by_score'].values()) == 91
        events = rows(experiment / 'per_event.jsonl')
        indexed = {(r['piece_id'], r['phrase_id'], r['event_id']): r for r in events}
        assert len(indexed) == len(events)
        event_sets[method] = indexed
        rounds = []
        logs = list((root / method / 'workers').glob('worker_*.log'))
        assert len(logs) == 8
        for log in logs:
            for line in log.open(errors='replace'):
                start = line.find('{"event": "eval_round"')
                if start >= 0:
                    rounds.append(json.loads(line[start:]))
        assert rounds
        evaluable = [e for e in events if e['pitch_audit'].get('status') in ('matched', 'mismatched')]
        target = [e for e in events if e.get('target_midi')]
        report['arms'][method] = {
            'metrics': summary['metrics'], 'final_warning': summary['final_warning'],
            'target_events': len(target), 'matched_pitch_events': sum(e['pitch_audit'].get('status') == 'matched' for e in evaluable),
            'pitch_coverage': len(evaluable) / len(target),
            'generation_rounds': len(rounds), 'generated_tokens': sum(r.get('generated_tokens', 0) for r in rounds),
            'tool_calls': sum(len(r.get('tool_calls', [])) for r in rounds),
            'truncated_rounds': sum(bool(r.get('truncated')) for r in rounds),
        }
    for a, b in (('two_stage', 'two_stage_off'), ('single_stage', 'fixed_workflow')):
        assert event_sets[a].keys() == event_sets[b].keys()
        common = [k for k in event_sets[a] if all(event_sets[m][k]['pitch_audit'].get('status') in ('matched', 'mismatched') for m in (a, b))]
        report['common_events'][a + '_vs_' + b] = {'events': len(common), 'pitch_accuracy': {
            m: sum(event_sets[m][k]['pitch_audit']['status'] == 'matched' for k in common) / len(common) for m in (a, b)}}
    (root / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({m: {k: v for k, v in data.items() if k not in ('metrics', 'final_warning')} for m, data in report['arms'].items()}, ensure_ascii=False))
    print(json.dumps(report['common_events']))


if __name__ == '__main__':
    main()
