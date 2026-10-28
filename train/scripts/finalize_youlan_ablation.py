"""Validate and score the 48-score ablation with the held-out Youlan."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train.scripts.visualize_completed_eval import rows, write_rows, validate


def main():
    out = ROOT / 'train/eval_outputs/warning_ablation'
    full, youlan = out / 'full', out / 'youlan'
    inputs = rows(ROOT / 'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl')
    refs = rows(full / 'references.jsonl') + rows(youlan / 'references.jsonl')
    assert len(inputs) == len(refs) == 1031 and len({r['sample_id'] for r in inputs}) == 1031
    assert {r['sample_id'] for r in inputs} == {r['sample_id'] for r in refs}
    results = {}
    for arm in ('on', 'off'):
        base = full / ('on_existing' if arm == 'on' else 'off') / 'predictions.jsonl'
        new = rows(youlan / arm / 'predictions.jsonl')
        validate(rows(youlan / 'inputs.jsonl'), new)
        policy = json.loads((youlan / arm / 'feedback_policy.json').read_text())
        assert policy['warning_feedback'] == policy['hint_feedback'] == arm
        if arm == 'off':
            assert all(r['guqinizer_passes'] == 1 for r in new)
            for r in new:
                for stage in ('base', 'guqinizer'):
                    for t in r[stage].get('trace', []):
                        content = json.dumps(t.get('tool_results', []), ensure_ascii=False)
                        content += json.dumps([m for m in t.get('request_messages', []) if m.get('role') in ('user', 'tool')], ensure_ascii=False)
                        assert not any(s in content for s in (':warning:', ':hint:', '警告｜', '当前方案仍有告警', '上一轮出现音高警告'))
        pred = rows(base) + new
        validate(inputs, pred)
        target = out / 'current48' / arm
        target.mkdir(parents=True, exist_ok=True)
        write_rows(target / 'predictions.jsonl', pred)
        write_rows(target / 'references.jsonl', refs)
        subprocess.run([sys.executable, '-m', 'evaluation.run_eval', '--pred', str(target / 'predictions.jsonl'),
                        '--reference', str(target / 'references.jsonl'), '--experiment', 'warning_ablation_current48_' + arm,
                        '--model', 'qwen35_9b_lora_two_stage_final_2gpu_bs12'], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        results[arm] = json.loads((ROOT / 'evaluation_results' / ('warning_ablation_current48_' + arm) / 'summary.json').read_text())
    save = out / 'current48' / 'summary.json'
    save.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    for arm, result in results.items():
        print(arm, 'pitch', result['metrics']['pitch_accuracy_at_50_cents'], 'coverage', result['metrics']['pitch_evaluable_events'],
              'warnings', result['final_warning']['warning_count'], flush=True)


if __name__ == '__main__':
    main()
