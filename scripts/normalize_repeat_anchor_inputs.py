#!/usr/bin/env python3
"""Normalize anchor names in current derived scores and split annotations.

Raw evidence and archived inference request/result snapshots are not changed.
"""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.rename_training_repeat_anchor import replace, PATTERN, sha
from train.scripts.build_text_protocol_eval_set import fingerprint


def main():
    method = ROOT/'ABC_J/agent_training/two_stage_audited'
    targets = []
    for base in (ROOT/'ABC_J/final', ROOT/'ABC_J/modern'):
        for name in ('jianpu_jianzi_mapped.json', 'jianpu_jianzi_readable.json', 'jianpu_jianzi_mapped.md'):
            targets.extend(base.rglob(name))
    targets = sorted(set(targets))
    for base in (method/'inferred_all', method/'pitch_eligible_train_test'):
        targets.extend(base/f'inferred_trajectories_{split}.jsonl' for split in ('train', 'test'))
    evaluation = ROOT/'train/eval_inputs_v2_text_protocol/test50'
    targets.extend(evaluation/name for name in ('test_runtime.jsonl', 'evaluation_pairs_test.jsonl'))
    backup = Path(tempfile.mkdtemp(prefix='guqin_before_split_anchor_rename.', dir=Path.home()/'.Trash'))
    report = {'backup': str(backup), 'changed_files': [], 'raw_evidence_modified': False,
              'archived_predictions_modified': False, 'positions_and_phrase_ids_changed': False}
    for path in targets:
        if not path.exists():
            continue
        temp = path.with_name('.anchor_normalize_'+path.name)
        changed = 0
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8') as src, temp.open('w', encoding='utf-8') as out:
                for line in src:
                    if not PATTERN.search(line):
                        out.write(line); continue
                    row = json.loads(line); renamed = replace(row)
                    if renamed == row:
                        out.write(line); continue
                    changed += 1
                    # Eval input identity excludes the sealed reference.
                    if 'runtime_item' in renamed and 'input_sha256' in renamed:
                        public = {k: v for k, v in renamed.items()
                                  if k not in {'input_sha256', 'reference', 'training_allowed'}}
                        renamed['input_sha256'] = fingerprint(public)
                    out.write(json.dumps(renamed, ensure_ascii=False)+'\n')
        else:
            text = path.read_text(encoding='utf-8')
            changed = len(PATTERN.findall(text))
            temp.write_text(PATTERN.sub('再作起点ㄱ', text), encoding='utf-8')
        if changed:
            original = backup/path.relative_to(ROOT)
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), original); temp.replace(path)
            report['changed_files'].append({'path': str(path.relative_to(ROOT)), 'changed_rows_or_occurrences': changed})
        else:
            temp.unlink()
    manifest_path = evaluation/'export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['public_sha256'] = sha(evaluation/'test_runtime.jsonl')
    manifest['sealed_sha256'] = sha(evaluation/'evaluation_pairs_test.jsonl')
    manifest['repeat_anchor_label'] = '再作起点ㄱ'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (method/'split_anchor_normalization_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'changed_files': len(report['changed_files']), 'backup': str(backup),
                      'positions_and_phrase_ids_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
