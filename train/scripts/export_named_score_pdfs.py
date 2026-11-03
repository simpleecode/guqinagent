#!/usr/bin/env python3
"""Export verified ancient and modern model scores using the standard PDF renderer."""
import json
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts.visualize_running_eval_score import render_notation_pdf


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modern-ablations-only', action='store_true')
    args = parser.parse_args()
    eval_root = ROOT / 'train/eval_outputs'
    ancient_inputs = ROOT / 'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl'
    ancient_refs = eval_root / 'warning_ablation/current48/on/references.jsonl'
    modern_root = eval_root / 'modern_comparison'
    output_root = eval_root / 'score_pdfs'
    jobs = [
        ('古谱', '双阶段', ancient_inputs, eval_root / 'warning_ablation/current48/on/predictions.jsonl', ancient_refs),
        ('古谱', '单阶段', ancient_inputs, eval_root / 'single_stage_final/results/predictions.jsonl', ancient_refs),
        ('现代曲', '双阶段', modern_root / 'inputs.jsonl', modern_root / 'two_stage/predictions.jsonl', None),
        ('现代曲', '单阶段', modern_root / 'inputs.jsonl', modern_root / 'single_stage/predictions.jsonl', None),
    ]
    if args.modern_ablations_only:
        jobs = [('现代曲', '双阶段关闭反馈', modern_root / 'inputs.jsonl', modern_root / 'two_stage_off/predictions.jsonl', None),
                ('现代曲', '固定流程', modern_root / 'inputs.jsonl', modern_root / 'fixed_workflow/predictions.jsonl', None)]
    for category, method, input_path, prediction_path, reference_path in jobs:
        inputs, result = read(input_path), read(prediction_path)
        sources = {r['sample_id']: r for r in inputs}
        predictions = {r['sample_id']: r for r in result}
        assert len(inputs) == len(sources) == len(result) == len(predictions)
        assert sources.keys() == predictions.keys()
        assert len(inputs) == (1031 if category == '古谱' else 91)
        groups = defaultdict(list)
        for source in inputs:
            sid = source['sample_id']; row = predictions[sid]
            assert row['input_sha256'] == source['input_sha256'] and row['protocol_valid'], sid
            stages = ('base', 'guqinizer') if method in ('双阶段', '双阶段关闭反馈') else ('single_stage',)
            assert all(row.get(s, {}).get('ok') and isinstance(row[s].get('plan'), dict) for s in stages), sid
            groups[source['score_key']].append(source)
        assert len(groups) == (48 if category == '古谱' else 10)
        references = None
        if reference_path:
            references = {}
            for row in read(reference_path):
                sid = row.get('sample_id') or f"{row['score_key']}-{row['phrase_id']}"
                actions = (row.get('reference') or {}).get('actions') or (row.get('reference_plan') or {}).get('actions') or []
                references[sid] = {int(a['source_index']): str(a.get('jianzi') or a.get('jianzi_text') or a.get('text') or '') for a in actions if a.get('source_index') is not None}
            assert references.keys() == sources.keys()
        folder = output_root / category
        folder.mkdir(parents=True, exist_ok=True)
        for score, source in sorted(groups.items()):
            source.sort(key=lambda r: r['phrase_id'])
            title = source[0]['runtime_item']['input']['metadata'].get('score_title') or score
            safe_title = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', title).strip().rstrip('.') or score
            destination = folder / f'{method}_{safe_title}_{score}.pdf'
            if destination.exists():
                raise FileExistsError(destination)
            tuning = source[0]['runtime_item']['input'].get('normalized_tuning') or {}
            render_notation_pdf(source, predictions, score, title, tuning, destination,
                                notes_per_row=12, reference_by_id=references)
            print(destination, flush=True)


if __name__ == '__main__':
    main()
