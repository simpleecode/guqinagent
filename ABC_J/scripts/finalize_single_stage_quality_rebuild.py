"""Validate, redact, and build a single-stage candidate without promoting it.

All 3976 IDs are retained. Only the selected 1100 trajectories are replaced;
unselected JSONL lines remain byte-identical. Resuming never discards outputs.
"""
import hashlib
import importlib.util
import json
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/single_stage_audited'
BUILD = BASE / '.quality_rebuild'
RAW = BUILD / 'regenerated_raw'
WORK = BUILD / 'redaction_work'
SUFFIX = '-single_stage-teacher-tools'


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def overlay(old, replacement, target, expected):
    patches = {}
    for line in replacement.open():
        sid = json.loads(line)['sample_id']
        assert sid not in patches
        patches[sid] = line
    assert set(patches) == expected
    seen = set()
    with target.open('w') as output:
        for line in old.open():
            sid = json.loads(line)['sample_id']
            assert sid not in seen
            seen.add(sid)
            output.write(patches.get(sid, line))
    assert len(seen) == 3976 and expected <= seen


def verify_raw(selection, wanted, expected):
    source = ROOT / selection.get('source', 'ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl')
    assert sha(source) == selection['source_sha256']
    assert sha(BASE / 'teacher_raw/teacher_trajectory_audit.jsonl') == selection['audit_sha256']
    snapshot = BUILD / 'runtime_snapshot'
    assert all(sha(snapshot / name) == digest for name, digest in selection['runtime_sha256'].items())
    public = list(rows(RAW / 'messages_train.jsonl'))
    assert len(public) == len(expected) == 1100
    assert {r['sample_id'] for r in public} == expected
    assert all(r['agent_stage'] == 'single_stage' and r['verification']['training_eligible'] for r in public)
    checkpoint = {r['trajectory_id']: r for r in rows(RAW / 'checkpoint.jsonl')}
    assert set(checkpoint) == wanted and all(r['status'] == 'completed' for r in checkpoint.values())
    # Import and replay using the exact generation snapshot, not the mutable workspace runtime.
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location('quality_snapshot_audit', snapshot / 'ABC_J/scripts/prepare_final_warning_rebuild.py')
    shared = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = shared
    spec.loader.exec_module(shared)
    for r in rows(BASE / 'teacher_raw/teacher_trajectory_audit.jsonl'):
        assert r['sample_id'].endswith(SUFFIX)
        shared.PLANS[r['sample_id'][:-len(SUFFIX)]] = r['teacher_private']['accepted_plan']
    replacement_ids = set()
    for r in rows(RAW / 'teacher_trajectory_audit.jsonl'):
        assert r['sample_id'] not in replacement_ids
        replacement_ids.add(r['sample_id'])
        assert r['verification']['replay_passed']
        shared.PLANS[r['sample_id'][:-len(SUFFIX)]] = r['teacher_private']['accepted_plan']
    assert replacement_ids == expected and len(shared.PLANS) == 3976
    grouped = defaultdict(list)
    for row in rows(source):
        grouped[row['score_key']].append(row)
    cases = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('fork')) as pool:
        for count, found in pool.map(shared.replay_score, grouped.values()):
            completed += count
            cases.update(found)
            print(f'Replayed {completed}/3976; warning phrases={len(cases)}', flush=True)
    assert completed == 3976
    codes = Counter(w['code'] for warnings in cases.values() for w in warnings)
    before = {(sid, w.get('source_index'), w['code']) for sid, warnings in selection['cases'].items() for w in warnings}
    after = {(sid, w.get('source_index'), w['code']) for sid, warnings in cases.items() for w in warnings}
    report = {'verified_replacements': 1100, 'full_phrases': 3976, 'failed': 0,
              'basis': 'generation snapshot; original teacher reference history; accepted final plans',
              'before_warning_phrases': selection['warning_phrases'], 'before_warning_count': selection['warning_count'],
              'after_warning_phrases': len(cases), 'after_warning_count': sum(codes.values()), 'after_codes': dict(codes),
              'original_warning_positions_resolved': len(before - after),
              'original_warning_positions_remaining': len(before & after), 'new_warning_positions': len(after - before),
              'cases': cases, 'raw_public_sha256': sha(RAW / 'messages_train.jsonl'),
              'raw_audit_sha256': sha(RAW / 'teacher_trajectory_audit.jsonl')}
    save(BUILD / 'verification_report.json', report)
    return public


def main():
    selection = json.loads((BUILD / 'selection_report.json').read_text())
    wanted = set(selection['selected_ids'])
    expected = {sid + SUFFIX for sid in wanted}
    WORK.mkdir(exist_ok=True)
    paths = [RAW / 'messages_train.jsonl', RAW / 'teacher_trajectory_audit.jsonl',
             BASE / 'teacher_raw/messages_train.jsonl', BASE / 'teacher_raw/teacher_trajectory_audit.jsonl',
             BASE / 'teacher_final_redacted/messages_train.jsonl', BASE / 'teacher_final_redacted/teacher_trajectory_audit.jsonl',
             ROOT / 'ABC_J/scripts/redact_teacher_reasoning.py']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    lock = WORK / 'source_hashes.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == fingerprints, 'Sources changed; refusing mixed resume'
    else:
        save(lock, fingerprints)
    verification = BUILD / 'verification_report.json'
    if verification.exists():
        report = json.loads(verification.read_text())
        assert report['raw_public_sha256'] == sha(RAW / 'messages_train.jsonl')
        assert report['raw_audit_sha256'] == sha(RAW / 'teacher_trajectory_audit.jsonl')
        assert report['verified_replacements'] == 1100 and report['failed'] == 0
        assert all(sha(BUILD / 'runtime_snapshot' / name) == digest
                   for name, digest in selection['runtime_sha256'].items())
        public = list(rows(RAW / 'messages_train.jsonl'))
        assert len(public) == len(expected) and {r['sample_id'] for r in public} == expected
        print('Reusing verified immutable raw audit; resuming missing redaction only.', flush=True)
    else:
        public = verify_raw(selection, wanted, expected)
    worker_root = WORK / 'workers'
    worker_root.mkdir(exist_ok=True)
    processes = []
    for i in range(7):
        log = (worker_root / f'worker_{i}.log').open('ab')
        proc = subprocess.Popen([sys.executable, str(ROOT / 'ABC_J/scripts/redact_teacher_reasoning.py'),
                                 '--input-dir', str(RAW), '--output-dir', str(worker_root / f'worker_{i}'),
                                 '--model', 'glm-5.3', '--shard-count', '7', '--shard-index', str(i)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log))
        print(f'Redaction worker {i} started: PID {proc.pid}', flush=True)
    codes = []
    for proc, log in processes:
        codes.append(proc.wait())
        log.close()
    # Avoid a second parallel burst when individual calls hit rate limits.
    # The redactor resumes by sample ID, so successful rows incur no new calls.
    for i, code in enumerate(codes):
        if not code:
            continue
        with (worker_root / f'worker_{i}.log').open('ab') as log:
            retried = subprocess.run([sys.executable, str(ROOT / 'ABC_J/scripts/redact_teacher_reasoning.py'),
                                     '--input-dir', str(RAW), '--output-dir', str(worker_root / f'worker_{i}'),
                                     '--model', 'glm-5.3', '--shard-count', '7', '--shard-index', str(i)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        codes[i] = retried.returncode
    assert not any(codes), f'Redaction failures remain; resume missing only: {codes}'
    subset = WORK / 'redacted_subset'
    if not subset.exists():
        run('scripts/merge_redaction_workers.py', '--source', RAW, '--worker-root', worker_root, '--output-dir', subset)
    originals = {r['sample_id']: r for r in public}
    rewritten_ids = set()
    for r in rows(subset / 'messages_train.jsonl'):
        sid = r['sample_id']
        assert sid not in rewritten_ids
        rewritten_ids.add(sid)
        before = originals[sid]
        assert before['tools'] == r['tools'] and before['provenance'] == r['provenance']
        assert len(before['messages']) == len(r['messages'])
        for old, new in zip(before['messages'], r['messages']):
            old, new = deepcopy(old), deepcopy(new)
            if old['role'] == 'assistant':
                old.pop('content', None)
                new.pop('content', None)
            assert old == new, f'Tool/nonassistant mutation: {sid}'
    assert rewritten_ids == expected
    candidate = WORK / 'full_candidate'
    candidate.mkdir(exist_ok=True)
    for filename in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl'):
        overlay(BASE / 'teacher_final_redacted' / filename, subset / filename, candidate / filename, expected)
    with (candidate / 'reasoning_redaction_audit.jsonl').open('w') as out:
        for line in (BASE / 'teacher_final_redacted/reasoning_redaction_audit.jsonl').open():
            if json.loads(line).get('sample_id') not in expected:
                out.write(line)
        with (subset / 'reasoning_redaction_audit.jsonl').open() as source:
            shutil.copyfileobj(source, out)
    save(candidate / 'reasoning_redaction_report.json', {
        'requested': 3976, 'written': 3976, 'failed': 0, 'replaced_single_stage_rows': 1100,
        'public_private_ids_match': True, 'nonassistant_and_tool_content_unchanged': True,
        'incremental_report': json.loads((subset / 'reasoning_redaction_report.json').read_text())})
    # Keep the exact redaction overlay as evidence. Public SFT cleaning is a
    # separate, reported transformation of explicitly failed unknown calls.
    training = WORK / 'training_candidate'
    if not training.exists():
        run('scripts/clean_invalid_teacher_tool_calls.py', '--input-dir', candidate, '--output-dir', training)
    run('scripts/validate_teacher_agent_messages.py', '--input-dir', training)
    export = WORK / 'sft_candidate'
    run('train/scripts/export_sft_dataset.py', '--input-dir', training, '--output-dir', export)
    run('train/scripts/validate_sft_export.py', '--input-dir', export)
    assert fingerprints == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(BUILD / 'candidate_validation_report.json', {
        'status': 'candidate_validated_not_promoted', 'rows': 3976, 'replacements': 1100,
        'messages_sha256': sha(training / 'messages_train.jsonl'),
        'raw_redaction_overlay_sha256': sha(candidate / 'messages_train.jsonl'),
        'protocol_cleanup': json.loads((training / 'invalid_tool_cleanup_report.json').read_text()),
        'sft_sha256': sha(export / 'guqin_agent_train.jsonl'),
        'length_measurement_pending': True, 'formal_data_unchanged': True})
    print('Candidate validated; fresh full-template length measurement required before promotion.', flush=True)


if __name__ == '__main__':
    main()
