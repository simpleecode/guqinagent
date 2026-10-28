"""Audit paired diagnostic-feedback ablation; retain every planned phrase."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.scripts.visualize_completed_eval import rows, write_rows, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--existing-on', action='store_true', help='compare full OFF with frozen current formal ON; not a paired rerun')
    args = parser.parse_args()
    assert not args.existing_on or args.scope == 'full'
    pilot = ROOT / 'train/eval_outputs/warning_ablation' / args.scope
    inputs = rows(pilot / 'inputs.jsonl')
    manifest = json.loads((pilot / 'experiment_manifest.json').read_text())
    ids = {r['sample_id'] for r in inputs}
    expected = 33 if args.scope == 'pilot' else 979
    assert len(ids) == len(inputs) == expected
    assert ids == set(manifest['sample_ids'])
    assert hashlib.sha256((pilot / 'inputs.jsonl').read_bytes()).hexdigest() == manifest['input_sha256']
    references = [r for r in rows(ROOT / 'train/eval_outputs/two_stage/.work/test50/original_references.jsonl')
                  if r['sample_id'] in ids]
    assert len(references) == expected and {r['sample_id'] for r in references} == ids
    write_rows(pilot / 'references.jsonl', references)
    current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in manifest['runtime_sha256']}
    result = {'input_sha256': manifest['input_sha256'], 'phrases': expected, 'scores': manifest['scores'],
              'runtime_matches_inference_snapshot': current_hashes == manifest['runtime_sha256'],
              'scoring_runtime_sha256': current_hashes, 'arms': {}}
    if args.existing_on:
        result['scoring_files_different_from_inference'] = [
            name for name, digest in current_hashes.items() if digest != manifest['runtime_sha256'][name]]
        baseline = json.loads((pilot / 'on_baseline_manifest.json').read_text())
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                   for name, digest in baseline['files'].items()), 'Formal ON baseline changed'
        prior = json.loads((pilot / 'off/direct_gpu2_launch.json').read_text())
        finals = {r['sample_id']: r for r in rows(pilot / 'off/predictions.jsonl')}
        assert prior['baseline_count'] == 940 and len(finals) == 979
        assert all(hashlib.sha256(json.dumps(finals[sid], sort_keys=True, ensure_ascii=False).encode()).hexdigest() == digest
                   for sid, digest in prior['baseline_row_sha256'].items()), 'Resumed prefix changed'
        result['resumed_940_records_unchanged'] = True
    for arm in ('on', 'off'):
        reused = args.existing_on and arm == 'on'
        directory = pilot / ('on_existing' if reused else arm)
        if reused:
            directory.mkdir(exist_ok=True)
            stored = [r for name in baseline['files'] if name.endswith('/predictions.jsonl') for r in rows(ROOT / name)]
            write_rows(directory / 'predictions.jsonl', stored)
        predictions = rows(directory / 'predictions.jsonl')
        validate(inputs, predictions)
        if arm == 'off':
            assert all(r.get('guqinizer_passes') == 1 for r in predictions), 'OFF has extra Guqinizer passes'
        sampling = None if reused else json.loads((directory / 'sampling_params.json').read_text())
        if not reused:
            assert sampling == manifest['sampling']
        policy = ({'source': 'existing revised full-system result; mixed revision budgets, not a paired ON rerun'}
                  if reused else json.loads((directory / 'feedback_policy.json').read_text()))
        if not reused:
            assert policy['warning_feedback'] == policy['hint_feedback'] == arm
        trace = [t for r in predictions for s in ('base', 'guqinizer') for t in r[s].get('trace', [])]
        leaks = []
        for r in predictions:
            for stage in ('base', 'guqinizer'):
                for t in r[stage].get('trace', []):
                    public = json.dumps(t.get('tool_results', []), ensure_ascii=False)
                    dynamic = json.dumps([m for m in t.get('request_messages', [])
                                          if m.get('role') in ('user', 'tool')], ensure_ascii=False)
                    markers = (':warning:', ':hint:', '警告｜', '当前方案仍有告警', '上一轮出现音高警告')
                    hits = [m for m in markers if m in public or m in dynamic]
                    if arm == 'off' and (hits or t.get('pitch_gate', {}).get('warned_now')
                                         or t.get('stop_check', {}).get('action') == 'remind'):
                        leaks.append({'sample_id': r['sample_id'], 'stage': stage,
                                      'round': t.get('round'), 'markers': hits})
        rounds = []
        logs = list((directory / 'workers').glob('worker_*.log'))
        logs += list((directory / 'logs_before_direct_gpu2').glob('worker_*.log'))
        for log in logs:
            for line in log.open(errors='replace'):
                start = line.find('{"event": "eval_round"')
                if start >= 0:
                    rounds.append(json.loads(line[start:]))
        assert rounds or reused, 'Missing generation-cost logs'
        experiment = 'warning_ablation_' + ('' if args.scope == 'pilot' else 'full_') + arm
        subprocess.run([sys.executable, '-m', 'evaluation.run_eval', '--pred', str(directory / 'predictions.jsonl'),
                        '--reference', str(pilot / 'references.jsonl'), '--experiment', experiment,
                        '--model', 'qwen35_9b_lora_two_stage_final_2gpu_bs12'], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        evaluation = json.loads((ROOT / 'evaluation_results' / experiment / 'summary.json').read_text())
        result['arms'][arm] = {
            'sampling': sampling, 'feedback_policy': policy, 'protocol_valid': sum(r['protocol_valid'] for r in predictions),
            'failed_stages': sum(not r[s].get('ok', False) for r in predictions for s in ('base', 'guqinizer')),
            'generated_tokens': None if reused else sum(r.get('generated_tokens', 0) for r in rounds),
            'generation_rounds': None if reused else len(rounds),
            'truncated_rounds': None if reused else sum(bool(r.get('truncated')) for r in rounds),
            'tool_calls': None if reused else sum(len(r.get('tool_calls', [])) for r in rounds),
            'guqinizer_passes': sum(r.get('guqinizer_passes', 0) for r in predictions),
            'feedback_leaks': leaks, 'metrics': evaluation['metrics'], 'final_warning': evaluation['final_warning']}
    events = {arm: {(r['piece_id'], r['phrase_id'], r['event_id']): r
                    for r in rows(ROOT / 'evaluation_results' / ('warning_ablation_' + ('' if args.scope == 'pilot' else 'full_') + arm) / 'per_event.jsonl')}
              for arm in ('on', 'off')}
    assert events['on'].keys() == events['off'].keys()
    common = [key for key in events['on'] if all(events[a][key]['pitch_audit'].get('status')
                                                in ('matched', 'mismatched') for a in events)]
    result['common_pitch_evaluable_events'] = len(common)
    result['common_pitch_accuracy'] = {
        a: sum(events[a][k]['pitch_audit']['status'] == 'matched' for k in common) / len(common)
        for a in events}
    result['limitations'] = [
        ('Single seed and 33 phrases from four scores, not a full-test estimate.' if args.scope == 'pilot'
         else 'Single seed, 47 eligible scores / 979 phrases; excludes three blacklisted scores and the new Youlan replacement.'),
        'OFF removes hints and warning-driven followups as well as warning text (bundled diagnostic-feedback ablation).',
        'PitchAcc uses evaluable events; also report common-event comparison because coverage differs.',
        'Identical initial seed does not imply identical later random draws after dialogue paths diverge.']
    if args.existing_on:
        result['limitations'].append('ON reuses revised formal results; OFF is one full inference run with interruption/resume. Not a strict paired causal ablation or fair cumulative cost comparison.')
    (pilot / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({a: {k: v for k, v in r.items() if k not in ('metrics', 'sampling')}
                      for a, r in result['arms'].items()}, ensure_ascii=False, indent=2))
    assert not result['arms']['off']['feedback_leaks'], 'OFF feedback leakage detected; do not interpret as clean ablation'


if __name__ == '__main__':
    main()
