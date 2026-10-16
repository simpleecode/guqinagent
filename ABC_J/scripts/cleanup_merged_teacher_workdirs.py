"""Recoverably retire explicitly verified, already merged teacher workdirs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def read(relative):
    return json.loads((BASE / relative).read_text())


def main():
    completion = read('related_context_repair/completion_report.json')
    final = BASE / 'teacher_train_final'
    assert digest(final / 'messages_train.jsonl') == completion['messages_sha256']
    assert read('teacher_train_final/messages_validation_report.json')['valid']
    assert not read('daiqi_state_repair/replacement_report.json')['pending_phrases']
    assert not read('last_missing_retry/resolution_report.json')['pending_phrases']
    assert read('recapture_phrase_replacement_report.json')['status'] == 'complete'
    source = BASE / 'pitch_eligible_train_test/inferred_trajectories_train.jsonl'
    phrases = {json.loads(line)['trajectory_id'] for line in source.open() if line.strip()}
    assert len(phrases) == 3976
    expected = {f'{pid}-{stage}-teacher-tools' for pid in phrases
                for stage in ('fingering_agent', 'guqinization')}
    for directory in ('teacher_latest_runtime_raw', 'teacher_train_final'):
        ids = [json.loads(line)['sample_id'] for line in (BASE / directory / 'messages_train.jsonl').open() if line.strip()]
        assert len(ids) == len(set(ids)) == 7952 and set(ids) == expected
    view = read('full_training_visualization/visualization_sharded/source_manifest.json')
    assert view['source_sha256'] == completion['messages_sha256']
    targets = [
        'daiqi_state_repair', 'last_missing_retry', 'latest_runtime_redaction',
        'missing_stage_completion', 'recapture-phrase-replace-qrv_otja',
        'related_context_repair', 'walk_return_retry', 'numbered_repeat_visualization',
        'teacher_latest_runtime_raw/.parallel_workers',
        'teacher_latest_runtime_raw/final_warning_visualization',
        'teacher_latest_runtime_raw/final_warning_statistics.json',
        'teacher_latest_runtime_raw/repeat_warning_cases.json',
        'full_training_visualization.log',
    ]
    paths = [BASE / name for name in targets]
    assert all(p.exists() and not p.is_symlink() and p.resolve().is_relative_to(BASE) for p in paths)
    preserved = [BASE / name for name in ('inferred_all', 'pitch_eligible_train_test',
                 'teacher_latest_runtime_raw', 'teacher_train_final', 'full_training_visualization')]
    assert not any(keep == p or keep.is_relative_to(p) for keep in preserved for p in paths)
    reports = {}
    for p in paths:
        if p.is_dir():
            for report in p.glob('*.json'):
                try:
                    reports[str(report.relative_to(BASE))] = json.loads(report.read_text())
                except (ValueError, UnicodeError):
                    pass
    sizes = {str(p.relative_to(BASE)): (sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                                      if p.is_dir() else p.stat().st_size) for p in paths}
    # Use this volume's Trash: a home-directory Trash would copy gigabytes
    # onto the system disk and can exhaust it during cross-device moves.
    trash_root = Path('/Volumes/F/.Trashes') / str(os.getuid())
    assert trash_root.is_dir() and trash_root.stat().st_dev == BASE.stat().st_dev
    trash = Path(tempfile.mkdtemp(prefix='guqin-merged-workdir-cleanup-', dir=trash_root))
    manifest = {'trash': str(trash), 'bytes_moved': sum(sizes.values()), 'targets': sizes,
                'preserved_directories': [str(p) for p in preserved], 'completion_reports': reports,
                'final_messages_sha256': completion['messages_sha256'], 'remote_modified': False,
                'moves': []}
    audit = BASE / 'cleanup_manifest.json'
    for p in paths:
        relative = p.relative_to(BASE)
        destination = trash / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(destination))
        manifest['moves'].append({'from': str(p), 'to': str(destination)})
        audit.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    assert all(p.exists() for p in preserved)
    print(json.dumps({'trash': str(trash), 'moved_targets': len(paths),
                      'bytes_moved': manifest['bytes_moved'], 'manifest': str(audit)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
