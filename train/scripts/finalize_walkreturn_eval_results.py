#!/usr/bin/env python3
"""Merge suffix retries, check full benchmark coverage, score and render existing viewers."""
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'harmonicrestfix_checkpoint2574_walkreturn_repeatanchor_suffix_3gpu_20261011'
OUT = ROOT / 'train/eval_outputs_v3_two_stage' / NAME
INPUT = ROOT / 'train/eval_inputs_v2_text_protocol/renderfix_20261007/test_runtime.jsonl'
REFERENCES = INPUT.with_name('evaluation_pairs_test.jsonl')
BASE = ROOT / 'train/eval_outputs_v3_two_stage/harmonicrestfix_checkpoint2574_newtest_2gpu_20261011/remote_snapshot/predictions_complete.jsonl'


def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(*command):
    subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)


def main():
    inputs = rows(INPUT)
    expected = {r['sample_id']: r for r in inputs}
    assert len(expected) == len(inputs), 'Duplicate input IDs'
    refs = rows(REFERENCES)
    assert {r['sample_id'] for r in refs} == set(expected), 'Reference IDs differ'
    originals, retries = rows(BASE), rows(OUT / 'predictions.jsonl')
    assert len({r['sample_id'] for r in retries}) == len(retries), 'Duplicate retry IDs'
    combined = {r['sample_id']: r for r in originals}
    combined.update({r['sample_id']: r for r in retries})
    assert set(combined) == set(expected), 'Incomplete/extra benchmark predictions'
    for sid, pred in combined.items():
        assert pred.get('protocol_valid'), f'Invalid protocol: {sid}'
        assert pred['input_sha256'] == expected[sid]['input_sha256'], f'Input changed: {sid}'
    final = OUT / 'final_predictions.jsonl'
    with final.open('w', encoding='utf-8') as handle:
        for source in inputs:
            handle.write(json.dumps(combined[source['sample_id']], ensure_ascii=False) + '\n')
    report = {'phrases': len(combined), 'scores': len({r['score_key'] for r in inputs}),
              'retry_records': len(retries), 'retained_other_scores': len(combined)-len(retries),
              'protocol_valid': len(combined), 'missing': [], 'duplicates': 0}
    (OUT / 'final_merge_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    run('-m', 'evaluation.run_eval', '--pred', final, '--reference', REFERENCES,
        '--experiment', NAME, '--model', 'Qwen3.5-9B checkpoint-2574; max-guqinizer-passes=3')
    run(ROOT / 'scripts/export_eval_score_markdown.py', '--evaluation-dir', ROOT/'evaluation_results'/NAME,
        '--name', NAME+'.md')
    viewer = OUT / 'visualization_all'
    viewer.mkdir(exist_ok=True)
    entries = []
    for number, score in enumerate(sorted({r['score_key'] for r in inputs}), 1):
        score_inputs = [r for r in inputs if r['score_key'] == score]
        title = score_inputs[0]['runtime_item']['input']['metadata'].get('score_title', '')
        target = viewer / (score+'.html')
        run(ROOT / 'ABC_J/scripts/visualize_running_eval_score.py', '--score', score,
            '--input', INPUT, '--references', REFERENCES, '--pulled', final,
            '--skip-pull', '--output', target)
        assert target.stat().st_size > 0
        entries.append(f'<li><a href="{score}.html">{html.escape(score)}｜{html.escape(title)}</a>｜{len(score_inputs)} phrase</li>')
        print(f'Visualization {number}/{report["scores"]}: {score}', flush=True)
    index = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整评估结果</title><style>body{max-width:1000px;margin:30px auto;background:#181818;color:#eee;font:16px/1.8 system-ui}a{color:#a9ccff}</style><h1>checkpoint-2574 完整评估结果</h1>'
    index += f'<p>{report["scores"]}首曲谱，{report["phrases"]}个phrase；已合并复／进退与再作标记修复后的续跑结果。</p><ul>' + ''.join(entries) + '</ul></html>'
    (viewer/'index.html').write_text(index, encoding='utf-8')
    print('COMPLETE '+str(viewer/'index.html'), flush=True)


if __name__ == '__main__':
    main()
