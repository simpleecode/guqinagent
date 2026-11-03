#!/usr/bin/env python3
"""Render validated modern predictions with the shared notation overview."""
import html
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts.visualize_running_eval_score import notation_overview


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=('two_stage', 'single_stage', 'two_stage_off', 'fixed_workflow'), default=('two_stage', 'single_stage'))
    args = parser.parse_args()
    labels = {'two_stage': '双阶段', 'single_stage': '单阶段', 'two_stage_off': '双阶段关闭反馈', 'fixed_workflow': '固定流程'}
    root = ROOT / 'train/eval_outputs/modern_comparison'
    inputs = read(root / 'inputs.jsonl')
    sources = {row['sample_id']: row for row in inputs}
    assert len(inputs) == len(sources) == 91
    groups = defaultdict(list)
    for row in inputs:
        groups[row['score_key']].append(row)
    assert len(groups) == 10
    for method in args.methods:
        rows = read(root / method / 'predictions.jsonl')
        predictions = {row['sample_id']: row for row in rows}
        assert len(rows) == len(predictions) == 91 and predictions.keys() == sources.keys()
        for sid, row in predictions.items():
            assert row['input_sha256'] == sources[sid]['input_sha256']
            assert row['score_key'] == sources[sid]['score_key'] and row['protocol_valid']
            stages = ('base', 'guqinizer') if method in ('two_stage', 'two_stage_off') else ('single_stage',)
            assert all(row.get(stage, {}).get('ok') and isinstance(row[stage].get('plan'), dict) for stage in stages), sid
        output = root / 'views' / method
        output.mkdir(parents=True, exist_ok=True)
        links = []
        for score, source in sorted(groups.items()):
            title = source[0]['runtime_item']['input']['metadata'].get('score_title') or score
            overview = notation_overview(source, predictions)
            body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
            body += '''<style>:root{--muted:#555}body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}.notation-strip{margin:1em 0;border:1px solid #bbb;padding:.5em}.notation-strip header{display:flex;justify-content:space-between}.notation-grid{display:grid;grid-template-columns:3em repeat(var(--note-count),minmax(70px,1fr));overflow-x:auto}.notation-cell,.notation-label{padding:.4em;border:1px solid #ddd;min-height:2em}.notation-cell small{display:block;color:#777;font-size:10px}.jianzi{white-space:pre-wrap}</style>'''
            body += '<h1>' + html.escape(title) + '</h1><p>方法：' + labels[method] + '。模型生成结果；本页面不展示参考标注或参考一致性评分。</p>' + overview + '</html>'
            (output / (score + '.html')).write_text(body, encoding='utf-8')
            links.append('<li><a href="' + html.escape(score) + '.html">' + html.escape(title) + '</a>（' + str(len(source)) + ' 段）</li>')
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>' + labels[method] + '现代曲结果</h1><p>10 首、91 段；无参考标注展示。</p><ul>' + ''.join(links) + '</ul>', encoding='utf-8')
        print(method, len(rows), len(groups), output)


if __name__ == '__main__':
    main()
