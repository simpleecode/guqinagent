"""Recoverably install a validated full single-stage overlay; never train."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile

from finalize_single_stage_quality_rebuild import ROOT, BASE, BUILD, RAW, WORK, SUFFIX, rows, sha, save, overlay


def main():
    completion = BUILD / 'completion_report.json'
    assert not completion.exists(), 'Already installed; inspect completion report instead'
    validation = json.loads((BUILD / 'candidate_validation_report.json').read_text())
    measurement = json.loads((BUILD / 'candidate_length_verification.json').read_text())
    stats = json.loads((BUILD / 'candidate_token_distribution.json').read_text())
    verification = json.loads((BUILD / 'verification_report.json').read_text())
    selection = json.loads((BUILD / 'selection_report.json').read_text())
    locked = json.loads((WORK / 'source_hashes.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in locked.items())
    assert validation['rows'] == stats['rows'] == 3976 and validation['replacements'] == 1100
    assert validation['sft_sha256'] == measurement['sft_sha256'] == sha(WORK / 'sft_candidate/guqin_agent_train.jsonl')
    assert measurement['report_sha256'] == sha(BUILD / 'candidate_token_distribution.json')
    final = WORK / 'training_candidate'
    assert sha(final / 'messages_train.jsonl') == validation['messages_sha256']
    assert json.loads((final / 'messages_validation_report.json').read_text())['valid']
    export = WORK / 'sft_candidate'
    assert json.loads((export / 'sft_export_validation_report.json').read_text())['valid']
    expected = {sid + SUFFIX for sid in selection['selected_ids']}
    raw_candidate = WORK / 'raw_candidate'
    raw_candidate.mkdir(exist_ok=True)
    for source in (BASE / 'teacher_raw').iterdir():
        if source.is_file() and source.name not in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl', 'checkpoint.jsonl', 'generation_report.json'):
            shutil.copy2(source, raw_candidate / source.name)
    for name in ('messages_train.jsonl', 'teacher_trajectory_audit.jsonl'):
        overlay(BASE / 'teacher_raw' / name, RAW / name, raw_candidate / name, expected)
    old_checkpoint = {r['trajectory_id']: r for r in rows(BASE / 'teacher_raw/checkpoint.jsonl')}
    updates = {r['trajectory_id']: r for r in rows(RAW / 'checkpoint.jsonl')}
    assert set(updates) == set(selection['selected_ids'])
    assert set(updates) <= set(old_checkpoint) and len(old_checkpoint) == 3976
    old_checkpoint.update(updates)
    assert all(r['status'] == 'completed' for r in old_checkpoint.values())
    with (raw_candidate / 'checkpoint.jsonl').open('w') as stream:
        for r in old_checkpoint.values():
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (raw_candidate / 'teacher_rejected_io.jsonl').open('ab') as output:
        with (RAW / 'teacher_rejected_io.jsonl').open('rb') as source:
            shutil.copyfileobj(source, output)
    save(raw_candidate / 'generation_report.json', {
        'accepted': 3976, 'failures': [], 'workflow': 'single_stage', 'by_stage': {'single_stage': 3976},
        'replaced': 1100, 'incremental_generation_report': json.loads((RAW / 'generation_report.json').read_text()),
        'quality_report': str(BUILD / 'verification_report.json')})
    dataset_manifest = json.loads((export / 'dataset_manifest.json').read_text())
    dataset_manifest['source_dir'] = str(BASE / 'teacher_final_redacted')
    save(export / 'dataset_manifest.json', dataset_manifest)
    manifest_path = BASE / 'manifest.json'
    previous = json.loads(manifest_path.read_text())
    manifest = deepcopy(previous)
    manifest.update(status='validated_with_quality_report', redacted_messages_sha256=validation['messages_sha256'],
                    sft_dataset_sha256=validation['sft_sha256'], raw_messages_sha256=sha(raw_candidate / 'messages_train.jsonl'),
                    raw_audit_sha256=sha(raw_candidate / 'teacher_trajectory_audit.jsonl'),
                    quality_replaced_phrases=1100, remaining_warning_count=verification['after_warning_count'],
                    remaining_warning_phrases=verification['after_warning_phrases'],
                    over_8192=stats['over_8192'], training_ready_at_cutoff_8192=stats['over_8192'] == 0,
                    quality_report=str(BUILD / 'verification_report.json'),
                    length_report=str(BUILD / 'candidate_token_distribution.json'))
    trash = Path('/Volumes/F/.Trashes/501')
    assert trash.is_dir()
    backup = Path(tempfile.mkdtemp(prefix='guqin_single_stage_quality_previous_', dir=trash))
    save(backup / 'manifest.json', previous)
    pairs = [(raw_candidate, BASE / 'teacher_raw'), (final, BASE / 'teacher_final_redacted'), (export, BASE / 'sft_export')]
    installed = []
    archived = []
    try:
        for source, target in pairs:
            os.rename(target, backup / target.name)
            archived.append(target)
            os.rename(source, target)
            installed.append((source, target))
        temporary_manifest = BUILD / 'promoted_manifest.json'
        save(temporary_manifest, manifest)
        os.replace(temporary_manifest, manifest_path)
    except BaseException:
        for source, target in reversed(installed):
            os.rename(target, source)
        for target in reversed(archived):
            os.rename(backup / target.name, target)
        raise
    save(completion, {'status': 'installed_no_training_submitted', 'phrases': 3976, 'replaced': 1100,
                      'remaining_warning_count': verification['after_warning_count'],
                      'remaining_warning_phrases': verification['after_warning_phrases'],
                      'over_8192': stats['over_8192'], 'training_ready_at_cutoff_8192': stats['over_8192'] == 0,
                      'sft_sha256': validation['sft_sha256'], 'backup': str(backup)})
    print(completion.read_text(), flush=True)


if __name__ == '__main__':
    main()
