"""Verify and publish the warning-suffix run; retain a reproducible hint shortlist."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.scripts.visualize_completed_eval import rows, write_rows, validate, publish
from evaluation.rules import replay_final_warnings
from evaluation.parser import final_actions
from agents.ToolRuntime.hint_checks import repeated_pitch_hints


def main():
    work = ROOT/'train/eval_outputs/two_stage/.work/warning_suffix_retry'
    baseline = work.parent/'test50'
    selection = json.loads((work/'selection_report.json').read_text())
    assert hashlib.sha256((work/'inputs.jsonl').read_bytes()).hexdigest() == selection['input_sha256']
    inputs, predictions = rows(work/'inputs.jsonl'), rows(work/'predictions.jsonl')
    sources, finals = validate(inputs, predictions)
    assert len(finals) == 818
    frozen = rows(work/'frozen_prefix.jsonl')
    assert len(frozen) == 175
    assert all(finals[r['sample_id']] == r for r in frozen)
    frozen_ids = {r['sample_id'] for r in frozen}
    assert len(set(finals)-frozen_ids) == 643
    combined_inputs = rows(baseline/'eligible_inputs.jsonl')
    combined = {r['sample_id']: r for r in rows(baseline/'eligible_predictions.jsonl')}
    combined.update(finals)
    combined_predictions = [combined[r['sample_id']] for r in combined_inputs]
    validate(combined_inputs, combined_predictions)
    assert len(combined_predictions) == 979
    assert len({r['score_key'] for r in combined_inputs}) == 47
    write_rows(work/'combined_inputs.jsonl', combined_inputs)
    write_rows(work/'combined_predictions.jsonl', combined_predictions)
    groups = defaultdict(list)
    hints = []
    for source in combined_inputs:
        prediction = combined[source['sample_id']]
        groups[source['score_key']].append((prediction, source))
        found = repeated_pitch_hints(source['runtime_item'], final_actions(prediction))
        if found:
            hints.append({'sample_id': source['sample_id'], 'score_key': source['score_key'],
                          'phrase_id': source['phrase_id'], 'hints': found})
    warnings = [w for samples in groups.values() for w in replay_final_warnings(samples)]
    old_positions = {(w['piece_id'], w['phrase_id'], w['event_id'], w['violation_type'])
                     for w in selection['warnings']}
    new_positions = {(w['piece_id'], w['phrase_id'], w['event_id'], w['violation_type'])
                     for w in warnings}
    hint_report = {'phrases': len(hints), 'runs': sum(len(h['hints']) for h in hints),
                   'scores': len({h['score_key'] for h in hints}), 'cases': hints,
                   'status': 'candidate shortlist only; not mandatory technique substitutions',
                   'minimum_guqinizer_passes': len(hints), 'maximum_guqinizer_passes': 3*len(hints)}
    (work/'hint_selection_report.json').write_text(json.dumps(hint_report, ensure_ascii=False, indent=2)+'\n')
    report = {'expected': 818, 'protocol_valid': len(finals), 'frozen_exact': len(frozen),
              'recomputed': 643, 'full_scores': 47, 'full_phrases': 979,
              'original_warning_count': len(selection['warnings']), 'final_warning_count': len(warnings),
              'warning_phrases': len({(w['piece_id'], w['phrase_id']) for w in warnings}),
              'warning_scores': len({w['piece_id'] for w in warnings}),
              'codes': dict(Counter(w['violation_type'] for w in warnings)),
              'original_positions_removed': len(old_positions-new_positions),
              'original_positions_remaining': len(old_positions & new_positions),
              'new_positions': len(new_positions-old_positions), 'warnings': warnings,
              'hint_phrases': hint_report['phrases'], 'hint_runs': hint_report['runs'],
              'hint_scores': hint_report['scores'],
              'runtime_note': 'latest local replay; inference did not include optional hint rendering',
              'predictions_sha256': hashlib.sha256((work/'predictions.jsonl').read_bytes()).hexdigest()}
    (work/'verification_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='warnings'}, ensure_ascii=False), flush=True)
    # Rendering may be resumed without replacing already-published identical results.
    selected = set(selection['start_phrase_by_score'])
    root = ROOT/'train/eval_outputs/two_stage'
    pending = []
    for score in selected:
        expected = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in
                           sorted((r for r in predictions if sources[r['sample_id']]['score_key']==score),
                                  key=lambda r: sources[r['sample_id']]['phrase_id']))
        target = root/score/'predictions.jsonl'
        if not target.exists() or target.read_text()!=expected or not (root/score/f'{score}.html').exists():
            pending.append(score)
    if pending:
        publish(work/'inputs.jsonl', work/'predictions.jsonl', 'two_stage', render=True,
                selected_scores=pending)
    subprocess.run([sys.executable, '-m', 'evaluation.run_eval', '--pred', str(work/'combined_predictions.jsonl'),
                    '--reference', str(baseline/'eligible_references.jsonl'), '--experiment', 'two_stage_warning_retry',
                    '--model', 'qwen35_9b_lora_two_stage_final_2gpu_bs12'], cwd=ROOT, check=True)
    (work/'completion_report.json').write_text(json.dumps({'published_scores':29, 'verified':True,
          'metrics':'evaluation_results/two_stage_warning_retry/summary.json',
          'hint_shortlist':'hint_selection_report.json'}, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
