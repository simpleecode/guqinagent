#!/usr/bin/env python3
"""Rebuild phrase inputs with the fixed parser, retaining current score splits.

Does not regenerate teacher messages, run training, or synchronize servers.
Promotes only verified source renderings, inferred seeds, selection and test inputs.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.infer_agent_trajectories import AUDIT
from agents.abc_to_jianzipu.score_structure import structural_token_info, is_structural_note
from train.scripts.build_text_protocol_eval_set import public_runtime_item, fingerprint

METHOD = ROOT / 'ABC_J/agent_training/two_stage_audited'


def read_rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalize_selected_paths(base):
    """Replace staging provenance before temporary work is removed."""
    for split in ('train', 'test'):
        path = base/f'inferred_trajectories_{split}.jsonl'
        temporary = base/f'.{split}_provenance.jsonl'
        with temporary.open('w', encoding='utf-8') as f:
            for row in read_rows(path):
                row['provenance']['source'] = str(ROOT/'ABC_J/final'/row['score_key']/'jianpu_jianzi_readable.json')
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        temporary.replace(path)
    path = base/'selection_manifest.json'
    selection = json.loads(path.read_text())
    selection['inputs']['inferred_dir'] = str(METHOD/'inferred_all')
    dump(path, selection)


def main():
    manifest = ROOT / 'ABC_J/results/dataset_split_groups.csv'
    with manifest.open(encoding='utf-8-sig') as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if r['split'] in {'train', 'test'}]
    assert Counter(r['split'] for r in rows) == {'train': 158, 'test': 50}
    train_groups = {r['leakage_group_id'] for r in rows if r['split'] == 'train'}
    test_groups = [r['leakage_group_id'] for r in rows if r['split'] == 'test']
    assert len(set(test_groups)) == 50 and not train_groups & set(test_groups)
    blacklisted = set(json.loads((ROOT/'ABC_J/results/score_blacklist.json').read_text())['score_keys'])
    assert not blacklisted & {r['score_key'] for r in rows}
    status_path = METHOD/'structural_rebuild_status.json'
    prior_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    pending = Path(prior_status.get('staging', '/nonexistent'))
    work = pending if prior_status.get('status') == 'running' and pending.is_dir() else Path(tempfile.mkdtemp(prefix='structural-input-rebuild-', dir=METHOD))
    dump(METHOD/'structural_rebuild_status.json', {'status': 'running', 'staging': str(work)})
    mapped = work/'mapped'; mapped.mkdir(exist_ok=True)
    source_changes = []
    with (work/'run.log').open('w', encoding='utf-8') as log:
        def run(*args):
            subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        for number, row in enumerate(rows, 1):
            key = row['score_key']; source = ROOT/'ABC_J/final'/key
            old = json.loads((source/'jianpu_jianzi_readable.json').read_text())
            raw = json.loads((source/'out/raw_data.json').read_text())
            # Carry verified metadata overrides into a temporary capture copy,
            # never rewrite the original raw evidence.
            metadata = raw.setdefault('metadata', {})
            metadata['tonic'] = old['metadata']['tonic'].removeprefix('1=')
            for field in ('jianpu_storage_tonic', 'jianpu_storage_note',
                          'jianpu_octave_shift', 'jianpu_octave_note', 'tonic_degree1_midi'):
                if field in old['metadata']:
                    metadata[field] = old['metadata'][field]
            tuning = old['metadata']['tuning']
            metadata['tuning'] = {'name': tuning['name'], 'value': tuning['value'],
                                  'source': 'preserved_verified_rendering_metadata'}
            capture = work/'capture.json'; dump(capture, raw)
            target = mapped/key
            run(ROOT/'scripts/extract_jianpu_jianzi.py', '--raw', capture,
                '--out-dir', target, '--tonic', metadata['tonic'])
            new = json.loads((target/'jianpu_jianzi_readable.json').read_text())
            assert AUDIT.parse_open_midi(new['metadata']) == AUDIT.parse_open_midi(old['metadata']), key
            assert new['metadata']['tuning']['name'] == tuning['name'], key
            assert len(new['notes']) == len(old['notes']), key
            changed = []
            for before, after in zip(old['notes'], new['notes']):
                assert before['index'] == after['index'], key
                if before != after:
                    if is_structural_note(after):
                        assert structural_token_info(after.get('structure_token') or after['abc']), (key, after)
                        changed.append(before['index'])
                    else:
                        raise ValueError(f'Unexpected non-structural rendering change: {key}:{before["index"]}: {before!r} -> {after!r}')
            source_changes.append({'score_key': key, 'changed_structure_source_indices': changed})
            if number % 10 == 0:
                print(f'rendered {number}/{len(rows)}', flush=True)
        staged_manifest = work/'manifest.csv'
        with staged_manifest.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        inferred = work/'inferred_all'; selected = work/'pitch_eligible_train_test'
        run(ROOT/'scripts/infer_agent_trajectories.py', '--manifest', staged_manifest,
            '--split-manifest', ROOT/'ABC_J/results/agent_dataset_splits.csv',
            '--mapped-dir', mapped, '--output-dir', inferred)
        print('phrase reconstruction complete; filtering train only', flush=True)
        run(ROOT/'ABC_J/scripts/export_pitch_eligible_trajectories.py',
            '--inferred-dir', inferred, '--splits', 'train', 'test', '--unfiltered-split', 'test',
            '--materialize-selected-inferred', '--output-dir', selected)
    counts = {}; compatibility = {}
    for split in ('train', 'test'):
        old_inputs = {r['trajectory_id']: r for r in read_rows(METHOD/'inferred_all'/f'inferred_trajectories_{split}.jsonl')}
        count = 0; unchanged = []; changed = []
        for row in read_rows(inferred/f'inferred_trajectories_{split}.jsonl'):
            row['provenance']['source'] = str(ROOT/'ABC_J/final'/row['score_key']/'jianpu_jianzi_readable.json')
            old = old_inputs.get(row['trajectory_id'])
            (unchanged if old and old['input'] == row['input'] else changed).append(row['trajectory_id'])
            count += 1
        counts[split] = count
        compatibility[split] = {'unchanged_local_input_ids': unchanged, 'changed_or_new_input_ids': changed,
                                 'removed_ids': sorted(set(old_inputs) - set(unchanged) - set(changed))}
        # Rewrite provenance to the promoted canonical path (not temp staging).
        path = inferred/f'inferred_trajectories_{split}.jsonl'
        rewritten = work/f'{split}.jsonl'
        with rewritten.open('w', encoding='utf-8') as f:
            for row in read_rows(path):
                row['provenance']['source'] = str(ROOT/'ABC_J/final'/row['score_key']/'jianpu_jianzi_readable.json')
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        rewritten.replace(path)
    test_target = work/'test50'; test_target.mkdir()
    with (test_target/'test_runtime.jsonl').open('w', encoding='utf-8') as public, (test_target/'evaluation_pairs_test.jsonl').open('w', encoding='utf-8') as sealed:
        for row in read_rows(inferred/'inferred_trajectories_test.jsonl'):
            assert row['input']['normalized_tuning']['open_midi'] == AUDIT.parse_open_midi(row['input']['metadata'])
            runtime = public_runtime_item(row)
            out = {'schema_version': 'agent-eval-input-2.0', 'sample_id': row['trajectory_id'],
                   'split': 'test', 'score_key': row['score_key'], 'phrase_id': row['phrase_id'],
                   'runtime_item': runtime}
            out['input_sha256'] = fingerprint(out)
            public.write(json.dumps(out, ensure_ascii=False)+'\n')
            sealed.write(json.dumps({**out, 'reference': {'actions': row['reference_plan']['actions']},
                                     'training_allowed': False}, ensure_ascii=False)+'\n')
    selected_counts = {s: sum(1 for _ in read_rows(selected/f'inferred_trajectories_{s}.jsonl')) for s in ('train', 'test')}
    normalize_selected_paths(selected)
    assert selected_counts['test'] == counts['test']
    dump(test_target/'export_manifest.json', {'source': str(METHOD/'inferred_all/inferred_trajectories_test.jsonl'),
        'phrases': counts['test'], 'scores': 50, 'test_filtering': False, 'normalized_tuning_verified': True,
        'train_test_score_overlap': [], 'train_test_leakage_group_overlap': [],
        'public_sha256': hashlib.sha256((test_target/'test_runtime.jsonl').read_bytes()).hexdigest(),
        'sealed_sha256': hashlib.sha256((test_target/'evaluation_pairs_test.jsonl').read_bytes()).hexdigest()})
    report = {'status': 'complete', 'score_split_preserved': True, 'scores': {'train': 158, 'test': 50},
              'raw_phrases': counts, 'selected_phrases': selected_counts,
              'source_changes': source_changes, 'input_compatibility': compatibility,
              'teacher_messages_rebuilt': False, 'existing_teacher_and_sft_require_compatibility_review': True,
              'server_synchronized': False}
    trash_root = Path.home()/'.Trash'; trash_root.mkdir(exist_ok=True)
    trash = Path(tempfile.mkdtemp(prefix='guqin_before_structure_rebuild.', dir=trash_root))
    for row in rows:
        for name in ('jianpu_jianzi_mapped.json', 'jianpu_jianzi_mapped.md', 'jianpu_jianzi_readable.json'):
            old_path = ROOT/'ABC_J/final'/row['score_key']/name
            destination = trash/'mapped'/row['score_key']/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), destination)
            shutil.move(str(mapped/row['score_key']/name), old_path)
    for name in ('inferred_all', 'pitch_eligible_train_test'):
        shutil.move(str(METHOD/name), trash/name)
        shutil.move(str(work/name), METHOD/name)
    eval_base = ROOT/'train/eval_inputs_v2_text_protocol/test50'
    shutil.move(str(eval_base), trash/'test50')
    shutil.move(str(test_target), eval_base)
    report['recoverable_previous_products'] = str(trash)
    dump(METHOD/'structural_rebuild_report.json', report)
    dump(METHOD/'structural_rebuild_status.json', {'status': 'complete', 'report': str(METHOD/'structural_rebuild_report.json')})
    for base in (METHOD/'teacher_train_final', ROOT/'ABC_J/agent_training/single_stage_audited/teacher_final_redacted',
                 ROOT/'train/data/guqin_agent_sft_two_stage_test50'):
        dump(base/'input_compatibility_status.json', {'status': 'legacy_inputs_do_not_use_for_new_training',
             'reason': 'Structure parser repair changed phrase boundaries and public ordinals; messages are retained without rewriting.',
             'report': str(METHOD/'structural_rebuild_report.json')})
    # Keep only the small run log, remove staging captures and extra ABC/text artifacts.
    shutil.move(str(work/'run.log'), METHOD/'structural_rebuild.log')
    shutil.rmtree(work)
    print(json.dumps({k: report[k] for k in ('status', 'scores', 'raw_phrases', 'selected_phrases', 'recoverable_previous_products')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
