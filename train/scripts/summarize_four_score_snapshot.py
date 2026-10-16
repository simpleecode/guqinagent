"""Merge a pulled worker snapshot and audit only final authored plans."""
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import validate_jianzi_only
from ABC_J.scripts.visualize_running_eval_score import final_warnings

output = Path(sys.argv[1])
inputs = [json.loads(l) for l in (ROOT/'train/eval_inputs_v2_text_protocol/test50_structurefix_four_scores/test_runtime.jsonl').open()]
sources = {r['sample_id']: r for r in inputs}
predictions = {}
for path in sorted((output/'workers').glob('predictions_*.jsonl')):
    for line in path.open():
        r = json.loads(line)
        assert r['sample_id'] not in predictions
        assert r['input_sha256'] == sources[r['sample_id']]['input_sha256']
        predictions[r['sample_id']] = r
with (output/'predictions.jsonl').open('w') as f:
    for sid in sorted(predictions):
        f.write(json.dumps(predictions[sid], ensure_ascii=False)+'\n')
history = {}
cases = []
summary = {}
for source in sorted(inputs, key=lambda r:(r['score_key'],r['phrase_id'])):
    score = source['score_key']
    counts = summary.setdefault(score, {'total':0,'completed':0,'warning_phrases':0,'warning_count':0,'codes':{}})
    counts['total'] += 1
    if source['sample_id'] not in predictions:
        continue
    r = predictions[source['sample_id']]
    assert r['protocol_valid'] and r['base']['ok'] and r['guqinizer']['ok']
    counts['completed'] += 1
    item = deepcopy(source['runtime_item'])
    item['baseline_plan'] = deepcopy(r['guqinizer']['plan'])
    _, report = validate_jianzi_only(item, [], require_complete=False, historical=history)
    warnings = report['warnings']
    mapping = {int(n['event_index']):int(n['index']) for n in item['input']['notes_without_jianzi'] if n.get('event_index') is not None}
    receipt = final_warnings(r['guqinizer'], mapping)
    counts['warning_count'] += len(warnings)
    counts['warning_phrases'] += bool(warnings)
    for w in warnings:
        code = w['code']; counts['codes'][code] = counts['codes'].get(code,0)+1
    if warnings or receipt:
        cases.append({'sample_id':r['sample_id'],'warnings':warnings,'final_receipt_warnings':receipt,'guqinizer_passes':r['guqinizer_passes']})
    item['reference_plan'] = deepcopy(r['guqinizer']['plan'])
    history[(score,source['phrase_id'])] = item
report = {'completed':len(predictions),'expected':len(inputs),'by_score':summary,'cases':cases}
(output/'final_warning_reaudit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
