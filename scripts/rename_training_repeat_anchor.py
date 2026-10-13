#!/usr/bin/env python3
"""Rename the repeat anchor in canonical local training products only."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWO = ROOT/'ABC_J/agent_training/two_stage_audited'
SINGLE = ROOT/'ABC_J/agent_training/single_stage_audited/teacher_final_redacted'
SFT = ROOT/'train/data/guqin_agent_sft_two_stage_test50'
PATTERN = re.compile(r'再作起点(?!ㄱ)')


def replace(value):
    if isinstance(value, str):
        return PATTERN.sub('再作起点ㄱ', value)
    if isinstance(value, list):
        return [replace(v) for v in value]
    if isinstance(value, dict):
        return {k: replace(v) for k, v in value.items()}
    return value


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    targets = [TWO/'inferred_all/inferred_trajectories_train.jsonl',
               TWO/'pitch_eligible_train_test/inferred_trajectories_train.jsonl']
    for base in (TWO/'teacher_train_final', SINGLE):
        targets.extend(base/name for name in ('messages_train.jsonl',
                       'teacher_trajectory_audit.jsonl', 'reasoning_redaction_audit.jsonl'))
    targets.extend(SFT/name for name in ('guqin_agent_train.jsonl', 'guqin_agent_train_metadata.jsonl'))
    assert all(path.is_file() for path in targets)
    backup = Path(tempfile.mkdtemp(prefix='guqin_before_anchor_rename.', dir=Path.home()/'.Trash'))
    report = {'replacement': '再作起点 → 再作起点ㄱ', 'files': [], 'standalone_final_markers': {},
              'test_inputs_modified': False, 'source_captures_modified': False,
              'backup': str(backup)}
    for path in targets:
        temporary = path.with_name('.anchor_rename_'+path.name)
        count = changed = 0
        standalone = []
        before_hash = sha(path)
        with path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as out:
            for line in source:
                row = json.loads(line); count += 1
                if path.name == 'teacher_trajectory_audit.jsonl':
                    for action in ((row.get('teacher_private') or {}).get('accepted_plan') or {}).get('actions', []):
                        text = str(action.get('jianzi_text') or action.get('text') or '').strip()
                        if re.fullmatch(r'[〔【\[]?再作起点(?:ㄱ)?[〕】\]]?', text):
                            standalone.append({'sample_id': row['sample_id'], 'source_index': action.get('source_index'), 'text': text})
                renamed = replace(row)
                if renamed != row:
                    changed += 1
                    out.write(json.dumps(renamed, ensure_ascii=False)+'\n')
                else:
                    out.write(line)
        if changed:
            original = backup/path.relative_to(ROOT)
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), original)
            temporary.replace(path)
        else:
            temporary.unlink()
        if path.name == 'teacher_trajectory_audit.jsonl':
            report['standalone_final_markers'][str(path.relative_to(ROOT))] = standalone
        report['files'].append({'path': str(path.relative_to(ROOT)), 'rows': count,
                                'changed_rows': changed, 'before_sha256': before_hash, 'after_sha256': sha(path)})
        print(path.relative_to(ROOT), count, 'rows;', changed, 'changed', flush=True)
    manifest_path = SFT/'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(source_messages_sha256=sha(TWO/'teacher_train_final/messages_train.jsonl'),
                    dataset_sha256=sha(SFT/'guqin_agent_train.jsonl'),
                    metadata_sha256=sha(SFT/'guqin_agent_train_metadata.jsonl'))
    manifest['repeat_anchor_rename'] = '再作起点ㄱ'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    report['token_statistics_stale'] = True
    report['visualization_not_regenerated'] = True
    (TWO/'repeat_anchor_rename_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
