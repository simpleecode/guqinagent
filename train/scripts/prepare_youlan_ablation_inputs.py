"""Extract Youlan public inputs and keep sealed references local for scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'train/eval_inputs_v2_text_protocol/test50'
    output = ROOT / 'train/eval_outputs/warning_ablation/youlan'
    output.mkdir(exist_ok=True)
    evidence = {}
    ids = None
    for original, target in [('test_runtime.jsonl', 'inputs.jsonl'), ('evaluation_pairs_test.jsonl', 'references.jsonl')]:
        path = source / original
        selected = [r for r in (json.loads(line) for line in path.open() if line.strip()) if r['score_key'] == 'ST4GAejh']
        assert len(selected) == len({r['sample_id'] for r in selected}) == 52
        current = {r['sample_id'] for r in selected}
        assert ids is None or current == ids
        ids = current
        if target == 'inputs.jsonl':
            assert all(r.get('reference') is None and r['runtime_item'].get('reference_plan') is None for r in selected)
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
        destination = output / target
        if destination.exists():
            assert destination.read_text() == text, 'Existing extraction changed'
        else:
            destination.write_text(text)
        evidence[original] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'input_selection_report.json').write_text(json.dumps({
        'score': 'ST4GAejh', 'phrases': 52, 'source_sha256': evidence,
        'sealed_reference_policy': 'local offline scoring only; upload inputs.jsonl only'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
