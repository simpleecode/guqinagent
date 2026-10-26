"""Post-install verification and recoverable cleanup of redundant work copies."""
import json
from pathlib import Path
import shutil
import tempfile

from finalize_single_stage_quality_rebuild import ROOT, BASE, BUILD, RAW, WORK, SUFFIX, rows, sha, save, run


def untouched_digest(path, selected):
    import hashlib
    digest = hashlib.sha256()
    ids = set()
    retained = 0
    with path.open('rb') as stream:
        for line in stream:
            sid = json.loads(line)['sample_id']
            assert sid not in ids
            ids.add(sid)
            if sid not in selected:
                digest.update(line)
                retained += 1
    assert len(ids) == 3976 and retained == 2876
    return digest.hexdigest(), ids


def main():
    completion = json.loads((BUILD / 'completion_report.json').read_text())
    assert completion['status'] == 'installed_no_training_submitted'
    backup = Path(completion['backup'])
    selection = json.loads((BUILD / 'selection_report.json').read_text())
    selected = {sid + SUFFIX for sid in selection['selected_ids']}
    manifest = json.loads((BASE / 'manifest.json').read_text())
    assert sha(ROOT / manifest['source']) == selection['source_sha256'] == manifest['source_sha256']
    comparisons = {}
    for directory in ('teacher_raw', 'teacher_final_redacted'):
        for name in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl'):
            old, old_ids = untouched_digest(backup / directory / name, selected)
            new, new_ids = untouched_digest(BASE / directory / name, selected)
            assert old == new and old_ids == new_ids
            comparisons[directory + '/' + name] = new
    final = BASE / 'teacher_final_redacted'
    exported = BASE / 'sft_export'
    run('scripts/validate_teacher_agent_messages.py', '--input-dir', final)
    run('train/scripts/validate_sft_export.py', '--input-dir', exported)
    assert sha(final / 'messages_train.jsonl') == manifest['redacted_messages_sha256']
    assert sha(exported / 'guqin_agent_train.jsonl') == completion['sft_sha256'] == manifest['sft_dataset_sha256']
    for r in rows(final / 'messages_train.jsonl'):
        assert r['agent_stage'] == 'single_stage'
    protected = json.loads((ROOT / 'train/eval_outputs/warning_ablation/full/on_baseline_manifest.json').read_text())['files']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    mark1 = ROOT / 'Mark1'
    mark1_files = {name: digest for name, digest in json.loads((mark1 / 'manifest.json').read_text())['sha256'].items()
                   if name.startswith('inference/') and name.endswith('/predictions.jsonl')
                   and len(Path(name).parts) == 3 and not Path(name).parts[1].startswith('.')}
    assert len(mark1_files) == 5 and all(sha(mark1 / name) == digest for name, digest in mark1_files.items())
    report = {'status': 'verified', 'phrases': 3976, 'replaced': 1100, 'unchanged': 2876,
              'unchanged_line_digests': comparisons, 'source_train_sha256': manifest['source_sha256'],
              'mark1_predictions_unchanged': 5, 'main_evaluation_fingerprints_unchanged': True,
              'sft_sha256': completion['sft_sha256'], 'training_submitted': False}
    save(BUILD / 'post_install_verification.json', report)
    # Canonical raw and final now contain the source evidence and accepted data.
    # Keep aggregate reports and snapshot, but retire duplicate shard/candidate copies.
    trash = Path(tempfile.mkdtemp(prefix='guqin_single_stage_quality_workcopies_', dir='/Volumes/F/.Trashes/501'))
    removed = []
    for path in (WORK, RAW / '.parallel_workers'):
        if path.exists():
            target = trash / path.name
            shutil.move(str(path), str(target))
            removed.append({'source': str(path), 'recoverable_trash': str(target)})
    save(BUILD / 'workcopy_cleanup_report.json', {'recoverable': True, 'moved': removed})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
