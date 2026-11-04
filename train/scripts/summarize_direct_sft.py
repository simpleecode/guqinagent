"""Verify saved Direct outputs and compare them with trajectory-trained students.

No inference or new scoring: reads independently scored, strict-prefix outputs.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SHA = '30c0e8fcd8fb7387d86f66677f8596c24327a693eced4f5ac8131dfcb27d0913'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(name):
    return {(r['piece_id'], r['phrase_id'], r['event_id']): r
            for r in rows(ROOT / 'evaluation_results' / name / 'per_event.jsonl')}


def main():
    result = {'schema': 'direct-sft-comparison-1', 'adapter_sha256': ADAPTER_SHA,
              'scope': 'full modern test; traditional music scores conditional on valid contiguous prefixes',
              'comparability': 'same final teacher targets; different tool access and inference budgets',
              'scoring_code_sha256': {str(p.relative_to(ROOT)): sha(p)
                  for p in sorted((ROOT / 'evaluation').glob('*.py'))},
              'datasets': {}}
    for scope, experiment, baseline, source, refs in [
        ('test48', 'direct_sft_current48', 'single_stage_current48',
         'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl',
         'train/eval_outputs/warning_ablation/current48/on/references.jsonl'),
        ('modern', 'modern_direct_sft', 'modern_single_stage',
         'train/eval_outputs/modern_comparison/inputs.jsonl',
         'train/eval_inputs/modern_sealed/evaluation_pairs_test.jsonl'),
    ]:
        directory = ROOT / 'train/eval_outputs/direct_sft' / scope
        config = read(directory / 'direct_config.json')
        summary = read(ROOT / 'evaluation_results' / experiment / 'summary.json')
        run = read(directory / 'run_summary.json')
        prediction_rows = rows(directory / 'predictions.jsonl')
        inputs = {r['sample_id']: r for r in rows(ROOT / source)}
        references = {r['sample_id']: r for r in rows(ROOT / refs)}
        assert inputs.keys() == references.keys()
        assert sha(ROOT / source) == config['input_sha256']
        assert config['checkpoint_sha256'] == ADAPTER_SHA
        assert config['generations_per_phrase'] == 1 and config['tools'] is False and config['thinking'] is False
        assert len(prediction_rows) == len({r['sample_id'] for r in prediction_rows}) == run['attempted']
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        truncations = failures = 0
        for record in prediction_rows:
            sid = record['sample_id']
            assert record['input_sha256'] == inputs[sid]['input_sha256'] == references[sid]['input_sha256']
            assert inputs[sid]['runtime_item']['input'] == references[sid]['runtime_item']['input']
            direct = record['direct']
            assert direct['model_generations'] == 1 and direct['tool_calls'] == 0
            assert all(m.get('role') != 'tool' for m in direct['request_messages'])
            failures += not record['protocol_valid']
            generation = direct['generation']
            truncations += generation['finish_reason'] == 'length'
            for field in usage:
                assert field in generation['usage'], 'Missing usage is not zero'
                usage[field] += generation['usage'][field]
        assert failures == run['attempted_failures']
        completion = summary['completion']
        assert completion['protocol_valid'] == completion['scored_valid_prefix_units'] == run['protocol_valid']
        assert completion['missing_or_skipped'] == run['skipped_after_invalid_predecessor']
        direct_events = events(experiment)
        baseline_events = events(baseline)
        assert direct_events.keys() <= baseline_events.keys()
        pitch_statuses = {'matched', 'mismatched'}
        common = [key for key, event in direct_events.items()
                  if event['pitch_audit'].get('status') in pitch_statuses
                  and baseline_events[key]['pitch_audit'].get('status') in pitch_statuses]
        common_pitch = {'events': len(common)}
        for name, records in [('direct', direct_events), ('single_stage', baseline_events)]:
            correct = sum(records[k]['pitch_audit']['status'] == 'matched' for k in common)
            common_pitch[name + '_matched'] = correct
            common_pitch[name + '_accuracy'] = correct / len(common) if common else None
        target_count = sum(bool(r['target_midi']) for r in baseline_events.values())
        matched = sum(e['pitch_audit'].get('status') == 'matched' for e in direct_events.values())
        result['datasets'][scope] = {
            'experiment': experiment, 'completion': completion,
            'metrics': summary['metrics'],
            'full_test_target_events': target_count,
            'pitch_evaluable_over_full_target': summary['metrics']['pitch_evaluable_events'] / target_count,
            'verified_correct_over_full_target': matched / target_count,
            'verified_correct_events': matched,
            'common_evaluable_pitch': common_pitch,
            'usage': usage, 'generations': len(prediction_rows), 'tool_calls': 0,
            'truncated_generations': truncations,
            'prediction_sha256': sha(directory / 'predictions.jsonl'),
            'public_input_sha256': sha(ROOT / source), 'reference_sha256': sha(ROOT / refs),
            'scoring_summary_sha256': sha(ROOT / 'evaluation_results' / experiment / 'summary.json'),
        }
    output = ROOT / 'train/eval_outputs/direct_sft/comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
