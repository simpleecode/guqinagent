"""Report saved teacher warnings and replay accepted plans with the current runtime."""
import ast
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import validate_jianzi_only, harmonic_region_at_phrase_start
from agents.ToolRuntime.warning_registry import PUBLIC_WARNING_CODES

BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
SOURCE = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
AUDIT = BASE / 'teacher_train_final/teacher_trajectory_audit.jsonl'
OUT = BASE / 'warning_distribution'
REPLAY_AUDITS = {}
REPLAY_HISTORIES = {}

def replay_row(row):
    phrase = row['trajectory_id']
    history = REPLAY_HISTORIES[row['score_key']]
    item = dict(row)
    item['baseline_plan'] = REPLAY_AUDITS[phrase + '-fingering_agent-teacher-tools']['accepted_plan']
    item['harmonic_region_at_start'] = harmonic_region_at_phrase_start(item, history)
    _, report = validate_jianzi_only(
        item, REPLAY_AUDITS[phrase + '-guqinization-teacher-tools']['accepted_patches'],
        toward_reference=False, historical=history)
    return phrase, report.get('warnings') or []

def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

class Counts:
    def __init__(self):
        self.events = Counter()
        self.phrases = defaultdict(set)
        self.unique = defaultdict(set)
    def add(self, warnings, phrase, stage):
        for w in warnings:
            code = w.get('code', 'unknown')
            self.events[code] += 1
            self.phrases[code].add(phrase)
            self.unique[code].add((phrase, stage, w.get('source_index')))
    def result(self):
        return {'events': sum(self.events.values()),
                'phrases': len(set().union(*self.phrases.values())) if self.phrases else 0,
                'by_code': {c: {'events': n, 'phrases': len(self.phrases[c]),
                                'unique_stage_positions': len(self.unique[c])}
                            for c,n in self.events.most_common()}}

def main():
    sources = list(rows(SOURCE))
    saved_final = defaultdict(Counts)
    process = defaultdict(Counts)
    audits = {}
    for row in rows(AUDIT):
        sid = row['sample_id']
        phrase, stage = sid.removesuffix('-teacher-tools').rsplit('-', 1)
        assert sid not in audits
        p = audits[sid] = row['teacher_private']
        saved_final[stage].add(p.get('jianzi_quality_report', {}).get('warnings', []), phrase, stage)
        for entry in p.get('tool_execution_log', []):
            if entry.get('name') != 'edit_plan':
                continue
            payload = entry.get('result') or {}
            payload = payload.get('result', payload)
            process[stage].add(payload.get('warnings') or [], phrase, stage)
        audits[sid] = {k:p[k] for k in ('accepted_plan','accepted_patches')}
    assert len(sources) == 3976 and len(audits) == 7952
    history = {}
    for row in sources:
        final = audits[row['trajectory_id'] + '-guqinization-teacher-tools']['accepted_plan']
        historic = dict(row)
        historic['reference_plan'] = deepcopy(final)
        history[(row['score_key'], row['phrase_id'])] = historic
    current = Counts()
    cases = {}
    score_histories = defaultdict(dict)
    for key,value in history.items():
        score_histories[key[0]][key] = value
    global REPLAY_AUDITS, REPLAY_HISTORIES
    REPLAY_AUDITS, REPLAY_HISTORIES = audits, score_histories
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        for number, (phrase, warnings) in enumerate(pool.map(replay_row, sources, chunksize=8), 1):
            current.add(warnings, phrase, 'guqinization')
            if warnings:
                cases[phrase] = warnings
            if number % 250 == 0:
                print(f'Replayed {number}/{len(sources)}', flush=True)
    inventory = set()
    for path in [ROOT/'agents/ToolRuntime/runtime.py', ROOT/'agents/ToolRuntime/warning_checks.py',
                 ROOT/'scripts/audit_jianpu_jianzi_pitch.py']:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys,node.values):
                    if isinstance(key,ast.Constant) and key.value=='code' and isinstance(value,ast.Constant) and isinstance(value.value,str):
                        inventory.add(value.value)
    public_codes = set(PUBLIC_WARNING_CODES)
    data = {'source_phrases':len(sources), 'stage_trajectories':len(audits),
            'audit_sha256':hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
            'inventory': sorted(inventory),
            'public_warning_codes':sorted(public_codes),
            'saved_final': {s:c.result() for s,c in saved_final.items()},
            'process_occurrences': {s:c.result() for s,c in process.items()},
            'current_runtime_final': current.result(),
            'current_runtime_history':'accepted final Guqinizer plans for earlier training phrases',
            'current_cases':cases}
    OUT.mkdir(exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    summary = ['# 两阶段教师轨迹告警分布', '',
               f'源数据：{len(sources)} phrase；{len(audits)} 条阶段轨迹。', '',
               f'当前公开谱面告警类别：{len(public_codes)}。', '',
               '过程列为每次 edit_plan 的结构化审计告警，重复轮次累计；最终列仅取最终 Guqinizer。',
               '当前重审使用正式接受的历史 Guqinizer 方案；与生成时参考历史不同。', '',
               '| code | 生成时最终事件 | 最终phrase | 过程Fingering次数 | 过程Guqinizer次数 | 当前重审事件 | 当前phrase |',
               '|---|---:|---:|---:|---:|---:|---:|']
    sf=data['saved_final'].get('guqinization',{}).get('by_code',{})
    pf=data['process_occurrences'].get('fingering_agent',{}).get('by_code',{})
    pg=data['process_occurrences'].get('guqinization',{}).get('by_code',{})
    cf=data['current_runtime_final']['by_code']
    for c in sorted(public_codes | set(sf) | set(pf) | set(pg) | set(cf)):
        summary.append(f'| {c} | {sf.get(c,{}).get("events",0)} | {sf.get(c,{}).get("phrases",0)} | {pf.get(c,{}).get("events",0)} | {pg.get(c,{}).get("events",0)} | {cf.get(c,{}).get("events",0)} | {cf.get(c,{}).get("phrases",0)} |')
    (OUT/'report.md').write_text('\n'.join(summary)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='current_cases'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
