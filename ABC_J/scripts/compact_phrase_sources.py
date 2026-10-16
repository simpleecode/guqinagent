"""Remove unused inverse-diff data while proving all retained fields unchanged."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'ABC_J/agent_training/two_stage_audited'
DROP = {'canonical_trajectory', 'patches', 'unresolved_differences', 'quality'}


def compact(record):
    return {key: value for key, value in record.items() if key not in DROP}


def retained_digest(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def main():
    files = [BASE / directory / f'inferred_trajectories_{split}.jsonl'
             for directory in ('inferred_all', 'pitch_eligible_train_test')
             for split in ('train', 'test')]
    assert all(p.is_file() and not p.is_symlink() for p in files)
    trash_root = Path('/Volumes/F/.Trashes') / str(os.getuid())
    assert trash_root.stat().st_dev == BASE.stat().st_dev
    backup = Path(tempfile.mkdtemp(prefix='guqin-before-phrase-compaction-', dir=trash_root))
    report = {'removed_fields': sorted(DROP), 'backup': str(backup), 'files': [],
              'baseline_plan_preserved': True, 'reference_plan_preserved': True,
              'teacher_and_sft_files_modified': False, 'remote_modified': False}
    staged = []
    # Stage and validate every file before promoting any of them.
    for path in files:
        target = path.with_suffix('.compact-staging')
        assert not target.exists()
        before_hash, after_hash = hashlib.sha256(), hashlib.sha256()
        count = 0
        with target.open('w') as out, path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                original = json.loads(line)
                assert 'input' in original and 'reference_plan' in original and 'baseline_plan' in original
                value = compact(original)
                before_hash.update(retained_digest(value) + b'\n')
                out.write(json.dumps(value, ensure_ascii=False) + '\n')
                count += 1
        verified_count = 0
        with target.open() as source:
            for line in source:
                value = json.loads(line)
                assert not DROP.intersection(value)
                after_hash.update(retained_digest(value) + b'\n')
                verified_count += 1
        assert count == verified_count and before_hash.digest() == after_hash.digest()
        relative = path.relative_to(BASE)
        record = {'file': str(relative), 'rows': count, 'bytes_before': path.stat().st_size,
                  'bytes_after': target.stat().st_size, 'retained_content_sha256': before_hash.hexdigest()}
        report['files'].append(record)
        staged.append((path, target, relative))
    report_path = BASE / 'phrase_source_compaction_report.json'
    report['promoted'] = []
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for path, target, relative in staged:
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        target.rename(path)
        report['promoted'].append(str(relative))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    report['complete'] = True
    report['bytes_before'] = sum(r['bytes_before'] for r in report['files'])
    report['bytes_after'] = sum(r['bytes_after'] for r in report['files'])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
