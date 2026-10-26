"""Select optional same-pitch refinements, keeping all other model plans frozen."""
from pathlib import Path
import hashlib
import json
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.scripts.visualize_completed_eval import rows, write_rows, validate
from train.scripts.run_vllm_eval_queue import balanced_shards
from evaluation.parser import final_actions
from agents.ToolRuntime.hint_checks import basic_or_blank
from agents.abc_to_jianzipu.score_structure import is_structural_note


def main():
    previous = ROOT/'train/eval_outputs/two_stage/.work/warning_suffix_retry'
    out = previous.parent/'hint_refinement'
    if out.exists():
        raise SystemExit('Refusing to replace existing hint-refinement progress')
    source = rows(previous/'combined_inputs.jsonl')
    baseline = rows(previous/'combined_predictions.jsonl')
    _, finals = validate(source, baseline)
    shortlist = json.loads((previous/'hint_selection_report.json').read_text())
    # Conservatively exclude blanks following a compound/repeat/ornament in
    # the complete score stream, including across phrase boundaries. A blank
    # is not by itself evidence that a fresh repeated-note gesture is needed.
    grouped = defaultdict(list)
    for row in source:
        grouped[row['score_key']].append(row)
    covered = defaultdict(set)
    for score, items in grouped.items():
        previous_action = None
        for item in sorted(items, key=lambda r:r['phrase_id']):
            actions = {int(a['source_index']): a for a in final_actions(finals[item['sample_id']])}
            for note in item['runtime_item']['input']['notes_without_jianzi']:
                if is_structural_note(note):
                    continue
                index = int(note['index'])
                action = actions.get(index)
                text = str((action or {}).get('jianzi_text') or (action or {}).get('text') or '').strip()
                if note.get('notation_omitted') or (not text and previous_action and not basic_or_blank(previous_action)):
                    covered[score].add(index)
                if str(note.get('abc') or '').lower().startswith(('z', 'x')):
                    previous_action = None
                elif text:
                    previous_action = action
    selected, excluded = [], []
    for case in shortlist['cases']:
        kept = []
        for hint in case['hints']:
            if covered[case['score_key']].intersection(hint['source_indices']):
                excluded.append({'sample_id':case['sample_id'], 'hint':hint,
                                 'reason':'blank already potentially covered by preceding non-basic gesture/repeat'})
            else:
                kept.append(hint)
        if kept:
            selected.append({**case, 'hints':kept})
    targets = {r['sample_id'] for r in selected}
    scores = {r['score_key'] for r in selected}
    if not targets:
        raise SystemExit('No safe candidates remain')
    inputs = [r for r in source if r['score_key'] in scores]
    frozen = [finals[r['sample_id']] for r in inputs if r['sample_id'] not in targets]
    out.mkdir()
    (out/'workers').mkdir()
    write_rows(out/'inputs.jsonl', inputs)
    write_rows(out/'baseline.jsonl', [finals[r['sample_id']] for r in inputs])
    write_rows(out/'frozen_predictions.jsonl', frozen)
    for index, shard in enumerate(balanced_shards(inputs, min(16, len(scores)))):
        write_rows(out/'workers'/f'predictions_{index:02d}.jsonl',
                   [finals[r['sample_id']] for r in shard if r['sample_id'] not in targets])
    files = ['agents/ToolRuntime/runtime.py','agents/ToolRuntime/warning_checks.py',
             'agents/ToolRuntime/warning_registry.py','agents/ToolRuntime/warning_rendering.py',
             'agents/ToolRuntime/hint_checks.py','scripts/audit_jianpu_jianzi_pitch.py',
             'train/scripts/eval_two_stage_score.py','train/scripts/run_vllm_eval_queue.py',
             'train/configs/sampling_retry_schedule.json']
    report = {'target_phrases':len(targets), 'target_runs':sum(len(r['hints']) for r in selected),
              'scores':len(scores), 'total_phrases':len(inputs), 'frozen_phrases':len(frozen),
              'target_ids':sorted(targets), 'cases':selected, 'excluded_runs':excluded,
              'guqinizer_pass_budget':[len(targets), 3*len(targets)],
              'selection_policy':'optional hint; no forced adoption; Fingering frozen',
              'sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                        for name in ['inputs.jsonl','baseline.jsonl','frozen_predictions.jsonl']},
              'runtime_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (out/'selection_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','target_ids','runtime_sha256','excluded_runs')},ensure_ascii=False))


if __name__ == '__main__':
    main()
