#!/usr/bin/env python3
"""Validate a complete evaluation and reuse the per-score viewer for HTML shards."""
import argparse
import csv
import html
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def partition(path, temp, suffix):
    ids=set()
    for row in rows(path):
        sid=row['sample_id']
        if sid in ids:raise ValueError(f'Duplicate sample: {sid}')
        ids.add(sid)
        score=row['score_key']
        if Path(score).name!=score:raise ValueError(score)
        with (temp/f'{score}.{suffix}.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    return ids


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--label',default='完整两阶段评估')
    parser.add_argument('--expected-scores',type=int,required=True)
    parser.add_argument('--test-subset',action='store_true',
                        help='Visualize a verified subset of the current test inputs')
    args=parser.parse_args()
    sources={}; scores=defaultdict(list);titles={}
    for row in rows(args.inputs):
        sid=row['sample_id'];score=row['score_key']
        assert sid not in sources
        sources[sid]=row['input_sha256'];scores[score].append(sid)
        titles[score]=row['runtime_item']['input']['metadata'].get('score_title','')
    assert len(scores)==args.expected_scores
    seen=set()
    for row in rows(args.predictions):
        sid=row['sample_id']
        assert sid not in seen and sid in sources,sid
        assert row.get('protocol_valid') and row['input_sha256']==sources[sid],sid
        seen.add(sid)
    assert seen==set(sources), 'Incomplete benchmark'
    # Check the CURRENT split, rather than validating only against a stale input.
    with (ROOT/'ABC_J/results/agent_dataset_splits.csv').open(encoding='utf-8-sig') as handle:
        split_rows=list(csv.DictReader(handle))
    current={r['score_key'] for r in split_rows if r['split']=='test'}
    if args.test_subset:
        assert set(scores)<=current, 'Subset contains scores outside the current test split'
        canonical={r['sample_id']:r['input_sha256'] for r in rows(
            ROOT/'train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl')}
        assert all(canonical.get(sid)==digest for sid,digest in sources.items()), 'Subset inputs differ from current canonical inputs'
    else:
        assert set(scores)==current, 'Evaluation score IDs differ from current test split'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    entries=[]
    with tempfile.TemporaryDirectory(prefix='guqin-eval-score-') as dirname:
        temp=Path(dirname)
        assert partition(args.inputs,temp,'input')==seen
        assert partition(args.predictions,temp,'prediction')==seen
        assert partition(args.references,temp,'reference')==seen
        for number,score in enumerate(sorted(scores),1):
            target=args.output_dir/(score+'.html')
            subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/visualize_running_eval_score.py'),
                '--score',score,'--input',str(temp/f'{score}.input.jsonl'),
                '--references',str(temp/f'{score}.reference.jsonl'),
                '--pulled',str(temp/f'{score}.prediction.jsonl'),
                '--skip-pull','--output',str(target)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            content=target.read_text(encoding='utf-8')
            assert 'prompt reconstruction failed' not in content,score
            assert '等待输出' not in content,score
            entries.append(f'<li><a href="{score}.html">{html.escape(score)}｜{html.escape(titles[score])}</a><span>{len(scores[score])} phrase</span></li>')
            print(f'[{number}/{len(scores)}] {score}: {len(scores[score])}',flush=True)
    page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(args.label)+'</title><style>body{max-width:1000px;margin:30px auto;padding:0 18px;background:#181818;color:#eee;font:16px/1.8 system-ui}a{color:#a9ccff;text-decoration:none}li{display:flex;justify-content:space-between;border-bottom:1px solid #444;padding:8px}ul{padding:0;list-style:none}span{color:#bbb}</style></head><body><h1>'+html.escape(args.label)+'</h1>'
    scope='当前测试集子集' if args.test_subset else '当前测试集'
    page+=f'<p>{scope}：{len(scores)}首，{len(seen)}个phrase；全部完成，输入哈希与当前划分核验通过。</p><ul>'+''.join(entries)+'</ul></body></html>'
    (args.output_dir/'index.html').write_text(page,encoding='utf-8')
    report={'scores':len(scores),'phrases':len(seen),'protocol_valid':len(seen),'current_test_split_matches':True,
            'test_subset':args.test_subset,
            'source_inputs':str(args.inputs.resolve()),'predictions':str(args.predictions.resolve()),
            'index':str((args.output_dir/'index.html').resolve()),'all_shards_validated':True}
    (args.output_dir/'visualization_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
