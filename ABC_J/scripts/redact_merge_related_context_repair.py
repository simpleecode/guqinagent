"""Redact only repaired pairs, validate a full overlay, then promote one version."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_latest_two_stage_redaction import rows, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
REPAIR = BASE / 'related_context_repair'
RAW = REPAIR / 'regenerated_raw'
WORK = REPAIR / 'redaction_work'
FINAL = BASE / 'teacher_train_final'
EXPORT = ROOT / 'train/data/guqin_agent_sft_two_stage_latest_runtime'


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def overlay(old, replacement, target, expected):
    """Stream large private audits and preserve unrelated lines byte-for-byte."""
    replacements = {}
    for line in replacement.open():
        r = json.loads(line)
        sid = r['sample_id']
        assert sid not in replacements
        replacements[sid] = line
    assert set(replacements) == expected
    seen = set()
    with target.open('w') as out:
        for line in old.open():
            sid = json.loads(line)['sample_id']
            assert sid not in seen
            seen.add(sid)
            out.write(replacements.get(sid, line))
    assert expected <= seen
    assert len(seen) == 7952, len(seen)


def main():
    global REPAIR, RAW, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair-dir', type=Path, default=REPAIR)
    parser.add_argument('--expected-phrases', type=int, default=177)
    parser.add_argument('--resume-redaction', action='store_true',
                        help='retry only missing redacted samples using unchanged existing shards')
    args = parser.parse_args()
    REPAIR = args.repair_dir.resolve()
    assert REPAIR.parent == BASE.resolve()
    RAW = REPAIR / 'regenerated_raw'
    WORK = REPAIR / 'redaction_work'
    wanted = set((REPAIR / 'trajectory_ids.txt').read_text().split())
    assert len(wanted) == args.expected_phrases
    expected = {f'{p}-{stage}-teacher-tools' for p in wanted
                for stage in ('fingering_agent', 'guqinization')}
    originals = {r['sample_id']: r for r in rows(RAW / 'messages_train.jsonl')}
    assert set(originals) == expected
    assert all(r.get('verification', {}).get('training_eligible') for r in originals.values())
    private_ids = set()
    for r in rows(RAW / 'teacher_trajectory_audit.jsonl'):
        assert r['sample_id'] not in private_ids
        private_ids.add(r['sample_id'])
        assert r.get('verification', {}).get('replay_passed')
    assert private_ids == expected
    initial = {str(p): sha(p) for p in (RAW / 'messages_train.jsonl',
               RAW / 'teacher_trajectory_audit.jsonl', FINAL / 'messages_train.jsonl',
               FINAL / 'teacher_trajectory_audit.jsonl', EXPORT / 'guqin_agent_train.jsonl')}
    WORK.mkdir(exist_ok=args.resume_redaction)
    lock = WORK / 'source_hashes.json'
    if lock.exists():
        assert json.loads(lock.read_text()) == initial, 'Source changed since redaction started'
    else:
        if args.resume_redaction:
            assert not (REPAIR / 'raw_replacement_report.json').exists()
            selection = json.loads((REPAIR / 'selection_report.json').read_text())
            assert sha(FINAL / 'teacher_trajectory_audit.jsonl') == selection['audit_sha256']
            verification = json.loads((REPAIR / 'verification_report.json').read_text())
            assert sha(RAW / 'messages_train.jsonl') == verification['raw_public_sha256']
            assert sha(RAW / 'teacher_trajectory_audit.jsonl') == verification['raw_audit_sha256']
        lock.write_text(json.dumps(initial, indent=2) + '\n')
    assert not (WORK / 'redacted_subset').exists(), 'Resume applies only before the merge step'
    routing = {sid: i % 7 for i, sid in enumerate(originals)}
    inputs = WORK / 'inputs'
    for i in range(7):
        (inputs / f'worker_{i}').mkdir(parents=True, exist_ok=args.resume_redaction)
    for name in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl'):
        if args.resume_redaction:
            original_rows = {r['sample_id']: r for r in rows(RAW / name)}
            for i in range(7):
                existing = list(rows(inputs / f'worker_{i}' / name))
                assert {r['sample_id'] for r in existing} == {s for s,j in routing.items() if j==i}
                assert all(r == original_rows[r['sample_id']] for r in existing)
            continue
        handles = [(inputs / f'worker_{i}' / name).open('w') for i in range(7)]
        try:
            for line in (RAW / name).open():
                handles[routing[json.loads(line)['sample_id']]].write(line)
        finally:
            for handle in handles:
                handle.close()
    processes = []
    for i in range(7):
        log = (REPAIR / f'redaction_worker_{i}.log').open('ab')
        process = subprocess.Popen([sys.executable, str(ROOT / 'ABC_J/scripts/redact_teacher_reasoning.py'),
                    '--input-dir', str(inputs / f'worker_{i}'), '--output-dir', str(WORK / f'worker_{i}'),
                    '--model', 'glm-5.3'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log))
        print(f'GLM redaction worker {i}: PID {process.pid}, {sum(v == i for v in routing.values())} rows', flush=True)
    codes = []
    for process, log in processes:
        codes.append(process.wait())
        log.close()
    assert not any(codes), codes
    subset = WORK / 'redacted_subset'
    run('scripts/merge_redaction_workers.py', '--source', RAW, '--worker-root', WORK, '--output-dir', subset)
    for rewritten in rows(subset / 'messages_train.jsonl'):
        original = originals[rewritten['sample_id']]
        assert rewritten['tools'] == original['tools'] and rewritten['provenance'] == original['provenance']
        assert len(rewritten['messages']) == len(original['messages'])
        for before, after in zip(original['messages'], rewritten['messages']):
            before, after = deepcopy(before), deepcopy(after)
            if before.get('role') == 'assistant':
                before.pop('content', None)
                after.pop('content', None)
            assert before == after, rewritten['sample_id']
    candidate = WORK / 'full_candidate'
    candidate.mkdir()
    for name in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl'):
        overlay(FINAL / name, subset / name, candidate / name, expected)
    with (candidate / 'reasoning_redaction_audit.jsonl').open('w') as out:
        for line in (FINAL / 'reasoning_redaction_audit.jsonl').open():
            if json.loads(line).get('sample_id') not in expected:
                out.write(line)
        with (subset / 'reasoning_redaction_audit.jsonl').open() as changes:
            shutil.copyfileobj(changes, out)
    report = {'schema_version': 'reasoning-redaction-incremental-1.0', 'requested': 7952,
              'written': 7952, 'failed': 0, 'replaced_phrases': len(wanted), 'replaced_stage_rows': len(expected),
              'public_private_ids_match': True, 'previous_redaction_report':
              json.loads((FINAL / 'reasoning_redaction_report.json').read_text()),
              'incremental_redaction_report': json.loads((subset / 'reasoning_redaction_report.json').read_text())}
    (candidate / 'reasoning_redaction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    run('scripts/validate_teacher_agent_messages.py', '--input-dir', candidate)
    export_candidate = WORK / 'sft_candidate'
    run('train/scripts/export_sft_dataset.py', '--input-dir', candidate, '--output-dir', export_candidate)
    run('train/scripts/validate_sft_export.py', '--input-dir', export_candidate)
    manifest_path = export_candidate / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_dir'] = str(FINAL)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert all(sha(Path(path)) == digest for path, digest in initial.items()), 'Input changed during redaction; refusing promotion'
    run('ABC_J/scripts/replace_daiqi_repaired_teachers.py', '--canonical', BASE / 'teacher_latest_runtime_raw',
        '--repair', RAW, '--ids', REPAIR / 'trajectory_ids.txt', '--report', REPAIR / 'raw_replacement_report.json')
    raw_report = json.loads((REPAIR / 'raw_replacement_report.json').read_text())
    assert not raw_report['pending_phrases'] and raw_report['replaced_phrases'] == len(wanted)
    trash_root = Path('/Volumes/F/.Trashes') / str(os.getuid())
    assert trash_root.is_dir() and trash_root.stat().st_dev == BASE.stat().st_dev
    trash = Path(tempfile.mkdtemp(prefix=f'guqin-{REPAIR.name}-merged-', dir=trash_root))
    shutil.move(str(FINAL), str(trash / FINAL.name))
    candidate.rename(FINAL)
    shutil.move(str(EXPORT), str(trash / EXPORT.name))
    export_candidate.rename(EXPORT)
    completion = {'phrases': 3976, 'stage_rows': 7952, 'replaced_phrases': len(wanted),
                  'redaction_model': 'glm-5.3', 'final': str(FINAL), 'export': str(EXPORT),
                  'messages_sha256': sha(FINAL / 'messages_train.jsonl'),
                  'export_sha256': sha(EXPORT / 'guqin_agent_train.jsonl'),
                  'backup_in_trash': str(trash), 'remote_training_data_changed': False}
    # These are now superseded temporary copies, not independent training versions.
    shutil.move(str(WORK), str(trash / WORK.name))
    shutil.move(str(RAW), str(trash / 'regenerated_raw'))
    (REPAIR / 'completion_report.json').write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
