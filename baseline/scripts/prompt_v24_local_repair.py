"""Repair only scored pitch-mismatch rows from a frozen V9 development output."""
from __future__ import annotations
import json,re
from pathlib import Path
import prompt_candidate_paired_experiment as exp
from agents.ToolRuntime.runtime import RealToolRuntime
from baseline.gepa_guqin.glm_client import GlmAnthropicClient,GlmBackend,tool_calls
from baseline.gepa_guqin.metric import phrase_report
ROOT=Path(__file__).resolve().parent
SYSTEM="""你是 V9 的局部音高校对器。只可修改输入明确列出的 allowed_event_indices；其他行不可见也不可修改。依据给出的真实候选表，选择能命中目标音高且最接近原来音色/写法的单一替换。不得重做整段、不得改变未标记行、不得编造弦徽。若没有把握，返回原行。只调用 submit。"""
TOOL={"name":"submit","description":"提交允许音序的替换","input_schema":{"type":"object","properties":{"jianzi_rows":{"type":"array","items":{"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],"minItems":2,"maxItems":2}}},"required":["jianzi_rows"],"additionalProperties":False}}
def main():
 d=json.loads((ROOT/'prompt_v9_dev_replay_result.json').read_text(encoding='utf-8'))
 src=next(r for r in d['rows'] if r['sample_id']=='S4nhaTGN-p0001' and r['arm']=='v9')
 ids=[int(x) for x in re.findall(r'音高不匹配音序：([0-9、]+)',src['feedback'])[0].split('、')]
 raw=exp.select_explicit_rows('train',[src['sample_id']])[0]; example=exp.prepare(raw); runtime=RealToolRuntime(example['runtime_item'],{})
 candidates=runtime.invoke('get_pitch_candidates',{'event_indices':ids})['result']['text']
 allowed=[[i,t] for i,t in src['prediction']['jianzi_rows'] if i in ids]
 client=GlmAnthropicClient(min_interval=.5,max_retries=1); backend=GlmBackend(client)
 response=backend.generate(system=SYSTEM,messages=[{'role':'user','content':json.dumps({'allowed_event_indices':ids,'original_rows':allowed,'candidate_table':candidates},ensure_ascii=False)}],tools=[TOOL],temperature=0)
 calls=[c for c in tool_calls(response) if c['name']=='submit']; proposed=calls[0]['arguments']['jianzi_rows'] if len(calls)==1 else allowed
 proposed=[[int(i),str(t)] for i,t in proposed if int(i) in ids]
 if [i for i,_ in proposed] != ids: proposed=allowed
 final=[[i,(dict(proposed).get(i,t) if i in ids else t)] for i,t in src['prediction']['jianzi_rows']]
 pred={**src['prediction'],'jianzi_rows':final}; report=phrase_report(example,pred,{})
 repaired=float(report['score']); base=float(src['score']); accept=repaired>base and report['metrics']['components'].get('pitch',0)>=src['components']['pitch'] and report['metrics']['rule_violations']<=src['rule_violations']
 out={'kind':'v24_local_repair','sample_id':src['sample_id'],'allowed_event_indices':ids,'v9_score':base,'proposal_score':repaired,'accepted':accept,'v9_prediction':src['prediction'],'proposal':pred,'report':report,'usage':client.snapshot()}
 (ROOT/'prompt_v24_local_repair_result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:out[k] for k in ['v9_score','proposal_score','accepted']},ensure_ascii=False))
if __name__=='__main__':main()
