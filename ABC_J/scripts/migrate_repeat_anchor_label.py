"""Normalize current derived/training labels, preserving historical evidence."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.repeat_anchor import normalize_repeat_anchor_text
from train.scripts.build_text_protocol_eval_set import fingerprint

BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
FINAL = BASE / 'teacher_train_final'
EXPORT = ROOT / 'train/data/guqin_agent_sft_two_stage_latest_runtime'


def normalize(value):
    if isinstance(value, str):
        if '\\u' in value and value.lstrip().startswith(('{', '[')):
            try:
                parsed = json.loads(value)
                changed = normalize(parsed)
                if parsed != changed:
                    return json.dumps(changed, ensure_ascii=False)
            except (ValueError, TypeError):
                pass
        return normalize_repeat_anchor_text(value)
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def main():
    work = BASE / 'repeat_anchor_label_migration'
    work.mkdir(exist_ok=False)
    candidate = work / 'candidate_final'
    candidate.mkdir()
    report = {'label': '[再作标记ㄱ]', 'raw_capture_or_teacher_evidence_modified': False,
              'historical_eval_snapshots_modified': False, 'remote_modified': False,
              'semantic_repair_performed': False, 'changes': [], 'promoted': []}
    staged = []
    targets = []
    for base in (ROOT / 'ABC_J/final', ROOT / 'ABC_J/modern'):
        for filename in ('jianpu_jianzi_readable.json', 'jianpu_jianzi_mapped.json', 'jianpu_jianzi_mapped.md'):
            targets.extend(base.glob(f'*/{filename}'))
    for folder in ('inferred_all', 'pitch_eligible_train_test'):
        targets.extend(BASE / folder / f'inferred_trajectories_{split}.jsonl' for split in ('train', 'test'))
    evaluation = ROOT / 'train/eval_inputs_v2_text_protocol/test50'
    targets.extend(evaluation / name for name in ('test_runtime.jsonl', 'evaluation_pairs_test.jsonl'))
    targets.append(FINAL / 'messages_train.jsonl')
    for path in sorted(set(targets)):
        assert path.is_file() and not path.is_symlink()
        target = work / 'staged' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        changed = count = 0
        if path.suffix == '.jsonl':
            with path.open() as source, target.open('w') as out:
                for line in source:
                    if not line.strip():
                        out.write(line)
                        continue
                    original = json.loads(line)
                    updated = normalize(original)
                    if 'runtime_item' in updated and 'input_sha256' in updated:
                        public = {k: v for k, v in updated.items()
                                  if k not in {'input_sha256', 'reference', 'training_allowed'}}
                        updated['input_sha256'] = fingerprint(public)
                    changed += updated != original
                    count += 1
                    out.write(json.dumps(updated, ensure_ascii=False) + '\n' if updated != original else line)
        elif path.suffix == '.json':
            original = json.loads(path.read_text())
            updated = normalize(original)
            changed = int(updated != original)
            target.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + '\n')
        else:
            original = path.read_text()
            updated = normalize_repeat_anchor_text(original)
            changed = int(updated != original)
            target.write_text(updated)
        if path == FINAL / 'messages_train.jsonl':
            target.rename(candidate / 'messages_train.jsonl')
        elif changed:
            staged.append((path, target))
        else:
            target.unlink()
        if changed:
            report['changes'].append({'file': str(path.relative_to(ROOT)), 'changed_rows_or_document': changed,
                                      'rows': count, 'before_sha256': sha(path)})
    # Hard-link immutable private history rather than rewriting real requests.
    for filename in ('teacher_trajectory_audit.jsonl', 'reasoning_redaction_audit.jsonl', 'reasoning_redaction_report.json'):
        os.link(FINAL / filename, candidate / filename)
    run('scripts/validate_teacher_agent_messages.py', '--input-dir', candidate)
    export_candidate = work / 'sft_candidate'
    run('train/scripts/export_sft_dataset.py', '--input-dir', candidate, '--output-dir', export_candidate)
    run('train/scripts/validate_sft_export.py', '--input-dir', export_candidate)
    trash_root = Path('/Volumes/F/.Trashes') / str(os.getuid())
    assert trash_root.stat().st_dev == BASE.stat().st_dev
    backup = Path(tempfile.mkdtemp(prefix='guqin-before-repeat-label-', dir=trash_root))
    report['backup'] = str(backup)
    report_path = BASE / 'repeat_anchor_label_migration_report.json'
    for path, target in staged:
        old = backup / path.relative_to(ROOT)
        old.parent.mkdir(parents=True, exist_ok=True)
        path.rename(old)
        target.rename(path)
        report['promoted'].append(str(path.relative_to(ROOT)))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for current, replacement in ((FINAL, candidate), (EXPORT, export_candidate)):
        old = backup / current.relative_to(ROOT)
        old.parent.mkdir(parents=True, exist_ok=True)
        current.rename(old)
        replacement.rename(current)
        report['promoted'].append(str(current.relative_to(ROOT)))
    export_manifest = EXPORT / 'dataset_manifest.json'
    exported = json.loads(export_manifest.read_text())
    exported['source_dir'] = str(FINAL)
    exported['repeat_anchor_label'] = '[再作标记ㄱ]'
    export_manifest.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + '\n')
    # Update canonical manifests, not historical snapshot manifests.
    manifest_path = evaluation / 'export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(public_sha256=sha(evaluation / 'test_runtime.jsonl'),
                    sealed_sha256=sha(evaluation / 'evaluation_pairs_test.jsonl'),
                    repeat_anchor_label='[再作标记ㄱ]')
    old_manifest = backup / manifest_path.relative_to(ROOT)
    old_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, old_manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    selection = BASE / 'pitch_eligible_train_test/selection_manifest.json'
    old_selection = backup / selection.relative_to(ROOT)
    old_selection.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection, old_selection)
    info = json.loads(selection.read_text())
    info['repeat_anchor_label'] = '[再作标记ㄱ]'
    selection.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    report.update(complete=True, messages_sha256=sha(FINAL / 'messages_train.jsonl'),
                  sft_sha256=sha(EXPORT / 'guqin_agent_train.jsonl'),
                  token_statistics_stale=True, visualization_stale=True,
                  public_messages_are_normalized_derivatives=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    # Keep only the report, recoverable old copies, and one canonical result.
    shutil.move(str(work), str(backup / 'migration_work'))
    print(json.dumps({k: v for k, v in report.items() if k not in {'changes', 'promoted'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
