#!/usr/bin/env python3
"""Validate and visualize evaluations with the original dark score-sharded index.

Use --publish --method NAME to store one current inputs/predictions/manifest
under train/eval_outputs/NAME/SCORE. Add --render for HTML and --notation-pdf
for PDFs. Old results go to recoverable Trash; changed predictions invalidate
old PDFs and the aggregate PDF ZIP. Without --publish, the original explicit
--references/--output-dir/--expected-scores visualization workflow is unchanged.
"""
import argparse
import csv
import html
import json
import subprocess
import sys
import tempfile
import hashlib
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT_ROOT=ROOT/'train/eval_outputs'


def rows(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, data):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',value):
        raise ValueError(f'Unsafe method/score name: {value}')
    return value


def validate(inputs, predictions):
    sources={r['sample_id']:r for r in inputs};finals={r['sample_id']:r for r in predictions}
    if len(sources)!=len(inputs) or len(finals)!=len(predictions) or sources.keys()!=finals.keys():
        raise ValueError('Input/prediction IDs must be unique and completely covered')
    for sid,r in finals.items():
        if (r.get('input_sha256')!=sources[sid]['input_sha256'] or not r.get('protocol_valid')
                or not r.get('base',{}).get('ok') or not r.get('guqinizer',{}).get('ok')
                or r.get('score_key')!=sources[sid]['score_key']):
            raise ValueError(f'Invalid final prediction: {sid}')
    return sources,finals


def index_page(entries, label, notice=''):
    # One index template for both visualization and managed result storage.
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(label)
        +'</title><style>body{max-width:1000px;margin:30px auto;padding:0 18px;background:#181818;'
        'color:#eee;font:16px/1.8 system-ui}a{color:#a9ccff;text-decoration:none}li{display:flex;'
        'justify-content:space-between;border-bottom:1px solid #444;padding:8px}ul{padding:0;list-style:none}'
        'span{color:#bbb}</style></head><body><h1>'+html.escape(label)+'</h1><p>'+html.escape(notice)
        +'</p><ul>'+''.join(entries)+'</ul></body></html>')


def refresh_index(method_root):
    entries=[];pdfs=[];phrases=0
    for manifest in sorted(method_root.glob('*/result_manifest.json')):
        r=json.loads(manifest.read_text());score=safe_name(r['score_key']);phrases+=r['phrases']
        pdf=manifest.parent/f'{score}.pdf'
        pdf_link=f' · <a href="{score}/{score}.pdf">PDF</a>' if pdf.exists() else ''
        if pdf.exists():pdfs.append(f'<li><a href="{score}/{score}.pdf">{html.escape(r.get("title") or score)}</a></li>')
        entries.append(f'<li><a href="{score}/{score}.html">{score}｜{html.escape(r.get("title") or score)}</a>'
                       f'<span>{r["phrases"]} phrase{pdf_link}</span></li>')
    (method_root/'index.html').write_text(index_page(entries,'完整两阶段评估',
        f'{len(entries)} 首，{phrases} 个 phrase；按曲目打开。'),encoding='utf-8')
    (method_root/'pdf_index.html').write_text(index_page(pdfs,'当前有效 PDF'),encoding='utf-8')


def _retire(path):
    trash=Path('/Volumes/F/.Trashes/501')
    if not trash.is_dir():
        trash=Path.home()/'.local/share/Trash/files';trash.mkdir(parents=True,exist_ok=True)
    backup=Path(tempfile.mkdtemp(prefix='guqin_eval_replaced_',dir=trash))
    shutil.move(str(path),str(backup/path.name))
    return backup/path.name


def render_score(score, inputs, predictions, target, references=None, pdf=False, notes_per_row=12,
                 include_request_messages=False):
    command=[sys.executable,str(ROOT/'ABC_J/scripts/visualize_running_eval_score.py'),
             '--score',score,'--input',str(inputs),'--pulled',str(predictions),'--skip-pull','--output',str(target)]
    if references:command.extend(['--references',str(references)])
    if include_request_messages:command.append('--include-request-messages')
    if pdf:command.extend(['--notation-pdf',str(target.with_suffix('.pdf')),
                           '--notation-pdf-notes-per-row',str(notes_per_row)])
    subprocess.run(command,cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    content=target.read_text(encoding='utf-8')
    if 'prompt reconstruction failed' in content:raise ValueError(f'Invalid rendered prompt: {score}')


def publish(inputs_path, predictions_path, method, *, render=False, output_root=OUTPUT_ROOT,
            selected_scores=None, references=None, notation_pdf=False, notes_per_row=12,
            include_request_messages=False):
    """Store one verified current result per method/score, using this viewer."""
    method_root=output_root/safe_name(method)
    if method_root.is_symlink():raise ValueError('Refusing to publish through a method symlink')
    inputs=rows(inputs_path);predictions=rows(predictions_path);sources,finals=validate(inputs,predictions)
    grouped=defaultdict(list)
    for source in inputs:
        score=safe_name(source['score_key'])
        if selected_scores is None or score in selected_scores:grouped[score].append(source)
    if selected_scores is not None and set(grouped)!=set(selected_scores):
        raise ValueError('Requested score absent from complete input')
    method_root.mkdir(parents=True,exist_ok=True);published=[]
    for score,source in sorted(grouped.items()):
        source.sort(key=lambda r:r['phrase_id']);target=method_root/score
        if target.is_symlink():raise ValueError('Refusing to replace a score symlink')
        stage=Path(tempfile.mkdtemp(prefix=f'.visualize-{score}-',dir=method_root));backup=None
        try:
            write_rows(stage/'inputs.jsonl',source)
            write_rows(stage/'predictions.jsonl',[finals[s['sample_id']] for s in source])
            digest=hashlib.sha256((stage/'predictions.jsonl').read_bytes()).hexdigest()
            unchanged=(target/'predictions.jsonl').exists() and hashlib.sha256((target/'predictions.jsonl').read_bytes()).hexdigest()==digest
            metadata={'method':method,'score_key':score,'phrases':len(source),
                'title':source[0]['runtime_item']['input']['metadata'].get('score_title',''),
                'source_inputs':str(inputs_path.resolve()),'source_predictions':str(predictions_path.resolve()),
                'predictions_sha256':digest}
            (stage/'result_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
            if render or notation_pdf:
                render_score(score,stage/'inputs.jsonl',stage/'predictions.jsonl',stage/f'{score}.html',
                             references,notation_pdf,notes_per_row,include_request_messages)
            if unchanged:
                if not notation_pdf:
                    for p in target.glob('*.pdf'):shutil.copy2(p,stage/p.name)
                if not (render or notation_pdf) and (target/f'{score}.html').exists():shutil.copy2(target/f'{score}.html',stage/f'{score}.html')
            # Invalid exports must not remain advertised after prediction changes.
            if (not unchanged or notation_pdf) and (method_root/'all_scores_pdf.zip').exists():_retire(method_root/'all_scores_pdf.zip')
            if target.exists():backup=_retire(target)
            try:stage.rename(target)
            except Exception:
                if backup is not None:shutil.move(str(backup),str(target))
                raise
            published.append(str(target))
        finally:
            if stage.exists():shutil.rmtree(stage)
    refresh_index(method_root)
    return published


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
    parser.add_argument('--references',type=Path)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--publish',action='store_true',help='Validate and replace managed current results')
    parser.add_argument('--method',default='two_stage')
    parser.add_argument('--render',action='store_true',help='Render HTML when publishing')
    parser.add_argument('--include-request-messages',action='store_true',
                        help='Opt in to full per-round request messages in HTML')
    parser.add_argument('--score',action='append',help='Publish only these score IDs')
    parser.add_argument('--score-subdirs',action='store_true',
                        help='Keep existing <score>/<score>.html paths')
    parser.add_argument('--label',default='完整两阶段评估')
    parser.add_argument('--expected-scores',type=int)
    parser.add_argument('--test-subset',action='store_true',
                        help='Visualize a verified subset of the current test inputs')
    parser.add_argument('--allow-partial',action='store_true',
                        help='Show pending phrases and explicitly label an incomplete snapshot')
    parser.add_argument('--notation-pdf', action='store_true',
                        help='Export one final-notation PDF per score beside its HTML')
    parser.add_argument('--notation-pdf-notes-per-row', type=int, default=12)
    args=parser.parse_args()
    if args.publish:
        if args.allow_partial:parser.error('Managed publishing requires complete results')
        root=args.output_dir.parent if args.output_dir else OUTPUT_ROOT
        if args.output_dir and args.output_dir.name!=args.method:
            parser.error('--output-dir must end with --method in publishing mode')
        print(json.dumps({'published':publish(args.inputs,args.predictions,args.method,render=args.render,
            output_root=root,selected_scores=args.score,references=args.references,
            notation_pdf=args.notation_pdf,notes_per_row=args.notation_pdf_notes_per_row,
            include_request_messages=args.include_request_messages)},ensure_ascii=False))
        return
    if not args.output_dir or not args.references or args.expected_scores is None:
        parser.error('Visualization requires --output-dir, --references and --expected-scores')
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
    assert args.allow_partial or seen==set(sources), 'Incomplete benchmark'
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
        assert partition(args.inputs,temp,'input')==set(sources)
        assert partition(args.predictions,temp,'prediction')==seen
        assert partition(args.references,temp,'reference')==set(sources)
        for number,score in enumerate(sorted(scores),1):
            prediction_path=temp/f'{score}.prediction.jsonl'
            prediction_path.touch(exist_ok=True)
            target=(args.output_dir/score/(score+'.html') if args.score_subdirs
                    else args.output_dir/(score+'.html'))
            target.parent.mkdir(parents=True,exist_ok=True)
            render_score(score,temp/f'{score}.input.jsonl',prediction_path,target,
                         temp/f'{score}.reference.jsonl',args.notation_pdf,args.notation_pdf_notes_per_row,
                         args.include_request_messages)
            content=target.read_text(encoding='utf-8')
            assert 'prompt reconstruction failed' not in content,score
            if not args.allow_partial:
                assert '等待输出' not in content,score
            completed=sum(sid in seen for sid in scores[score])
            href=target.relative_to(args.output_dir).as_posix()
            pdf_href=target.with_suffix('.pdf').relative_to(args.output_dir).as_posix()
            pdf_link = f' · <a href="{pdf_href}">PDF</a>' if target.with_suffix('.pdf').exists() else ''
            entries.append(f'<li><a href="{href}">{html.escape(score)}｜{html.escape(titles[score])}</a><span>{completed}/{len(scores[score])} phrase{pdf_link}</span></li>')
            print(f'[{number}/{len(scores)}] {score}: {len(scores[score])}',flush=True)
    scope='当前测试集子集' if args.test_subset else '当前测试集'
    status='全部完成' if seen==set(sources) else '部分结果，仍在运行'
    page=index_page(entries,args.label,f'{scope}：{len(scores)}首，已完成{len(seen)}/{len(sources)}个phrase；{status}，输入哈希与当前划分核验通过。')
    (args.output_dir/'index.html').write_text(page,encoding='utf-8')
    report={'scores':len(scores),'phrases':len(seen),'protocol_valid':len(seen),'current_test_split_matches':True,
            'include_request_messages':args.include_request_messages,
            'test_subset':args.test_subset,'complete':seen==set(sources),'expected_phrases':len(sources),
            'source_inputs':str(args.inputs.resolve()),'predictions':str(args.predictions.resolve()),
            'index':str((args.output_dir/'index.html').resolve()),'all_shards_validated':True}
    (args.output_dir/'visualization_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
