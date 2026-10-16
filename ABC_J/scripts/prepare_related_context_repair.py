"""Select accepted teachers whose stored related-context prompts must change."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from agents.ToolRuntime.runtime import related_repeat_phrases


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--messages',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    sources={r['trajectory_id']:r for l in a.source.open() if (r:=json.loads(l))}
    history={(r['score_key'],r['phrase_id']):r for r in sources.values()}
    cache={};affected={};reasons={}
    for line in a.messages.open():
        r=json.loads(line);phrase=r['provenance']['source_trajectory_id']
        prompt=next((str(m.get('content') or '') for m in r['messages'] if m.get('role')=='user'),'')
        old_ids=re.findall(r'【只读相关段\s+([^｜]+)｜',prompt)
        previous_match='相关匹配｜已展示的前一段' in prompt
        if not old_ids and not previous_match:continue
        previous=re.search(r'【只读前一段\s+([^｜]+)｜',prompt)
        previous_id=previous.group(1) if previous else None
        if phrase not in cache:cache[phrase]=related_repeat_phrases(sources[phrase],history)
        found=cache[phrase]
        new_ids=[v['phrase_id'] for v in found if v['phrase_id']!=previous_id]
        new_previous=bool(found and found[0]['phrase_id']==previous_id)
        changes=[]
        if previous and old_ids and prompt.index('【只读相关段')>previous.start():changes.append('related_block_order')
        if old_ids!=new_ids or previous_match!=new_previous:changes.append('related_selection_changed')
        if changes:
            case=affected.setdefault(phrase,{'score_key':sources[phrase]['score_key'],'stages':[]})
            case['stages'].append({'sample_id':r['sample_id'],'reasons':changes,
                'old_related':old_ids,'new_related':new_ids,'old_previous_match':previous_match,
                'new_previous_match':new_previous})
            for change in changes:reasons.setdefault(change,set()).add(phrase)
    report={'affected_phrases':len(affected),'score_count':len({v['score_key'] for v in affected.values()}),
            'reason_phrase_counts':{k:len(v) for k,v in reasons.items()},
            'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
            'source_messages_sha256':hashlib.sha256(a.messages.read_bytes()).hexdigest(),
            'cases':affected}
    (a.output_dir/'affected_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (a.output_dir/'trajectory_ids.txt').write_text(''.join(x+'\n' for x in sorted(affected)))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
