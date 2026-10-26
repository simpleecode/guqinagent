"""Replay a local worker snapshot and render only complete warning-free scores."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import html
from collections import defaultdict
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agents.ToolRuntime.runtime import validate_jianzi_only
from train.scripts.visualize_completed_eval import rows,write_rows

def render_index(report):
    links=''.join(f'<li><a href="{r["score"]}/{r["score"]}.pdf">{html.escape(r["title"] or r["score"])}</a></li>'
                  for r in report if 'pdf' in r)
    (ROOT/'train/eval_outputs/two_stage/pdf_index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>最终零警告曲谱 PDF</title>'
        '<h1>当前运行时重审：最终零警告曲谱 PDF</h1><ul>'+links+'</ul>')

def main():
    if '--all-current' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--all-current',action='store_true')
        parser.add_argument('--references',type=Path,default=ROOT/'train/eval_outputs/two_stage/.work/test50/eligible_references.jsonl')
        args=parser.parse_args()
        from train.scripts.visualize_completed_eval import validate,refresh_index
        root=ROOT/'train/eval_outputs/two_stage'
        report=[]
        for manifest in sorted(root.glob('*/result_manifest.json')):
            out=manifest.parent;score=out.name
            validate(rows(out/'inputs.jsonl'),rows(out/'predictions.jsonl'))
            subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/visualize_running_eval_score.py'),
                '--score',score,'--input',str(out/'inputs.jsonl'),'--pulled',str(out/'predictions.jsonl'),
                '--skip-pull','--output',str(out/f'{score}.html'),
                '--references',str(args.references),
                '--notation-pdf',str(out/f'{score}.pdf')],check=True)
            report.append({'score':score,'pdf':str(out/f'{score}.pdf')})
        refresh_index(root)
        import zipfile
        for row in report:
            subprocess.run(['pdfinfo',row['pdf']],check=True,stdout=subprocess.DEVNULL)
        with zipfile.ZipFile(root/'all_scores_pdf.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for row in report:
                p=Path(row['pdf']);archive.write(p,f'{row["score"]}/{p.name}')
        (root/'pdf_export_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('PDF_COUNT',len(report),flush=True)
        return
    work=ROOT/'train/eval_outputs/two_stage/.work/test50'
    sources={};preds={}
    for f in sorted((work/'workers').glob('input_*.jsonl')):
        for r in rows(f):sources[r['sample_id']]=r
    for f in sorted((work/'workers').glob('predictions_*.jsonl')):
        for r in rows(f):
            assert r['sample_id'] not in preds
            preds[r['sample_id']]=r
    black=set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
    scores=defaultdict(list)
    for r in sources.values():scores[r['score_key']].append(r)
    report=[]
    for score,ss in sorted(scores.items()):
        if score in black or not all(s['sample_id'] in preds for s in ss):continue
        history={};warnings=[];valid=True
        for s in sorted(ss,key=lambda r:r['phrase_id']):
            r=preds[s['sample_id']]
            if (r['input_sha256']!=s['input_sha256'] or not r.get('protocol_valid')
                    or not r.get('base',{}).get('ok') or not r.get('guqinizer',{}).get('ok')):
                valid=False;break
            item=deepcopy(s['runtime_item']);item['baseline_plan']=deepcopy(r['guqinizer']['plan'])
            _,audit=validate_jianzi_only(item,[],require_complete=False,historical=history)
            warnings.extend({'phrase':s['phrase_id'],**w} for w in audit['warnings'])
            item['reference_plan']=deepcopy(r['guqinizer']['plan']);history[(score,s['phrase_id'])]=item
        row={'score':score,'title':ss[0]['runtime_item']['input']['metadata'].get('score_title'),
             'phrases':len(ss),'valid':valid,'warnings':warnings}
        report.append(row)
        print(score,'warnings',len(warnings),'valid',valid,flush=True)
        if not valid or warnings:continue
        out=ROOT/'train/eval_outputs/two_stage'/score;out.mkdir(exist_ok=True)
        ss.sort(key=lambda r:r['phrase_id'])
        write_rows(out/'inputs.jsonl',ss);write_rows(out/'predictions.jsonl',[preds[s['sample_id']] for s in ss])
        subprocess.run([sys.executable,str(ROOT/'ABC_J/scripts/visualize_running_eval_score.py'),
            '--score',score,'--input',str(out/'inputs.jsonl'),'--pulled',str(out/'predictions.jsonl'),
            '--skip-pull','--output',str(out/f'{score}.html'),'--notation-pdf',str(out/f'{score}.pdf')],check=True)
        row['pdf']=str(out/f'{score}.pdf')
    (work/'clean_pdf_report.json').write_text(json.dumps({'runtime_sha256':hashlib.sha256(
        (ROOT/'scripts/audit_jianpu_jianzi_pitch.py').read_bytes()).hexdigest(),'scores':report},ensure_ascii=False,indent=2)+'\n')
    render_index(report)
    print('PDF_COUNT',sum('pdf' in r for r in report),flush=True)

if __name__=='__main__':main()
