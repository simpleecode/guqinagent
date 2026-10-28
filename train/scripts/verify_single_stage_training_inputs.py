"""Validate the sealed single-stage export and current train/test separation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'train/data/guqin_agent_sft_single_stage_final'


def main():
    manifest = json.loads((DATA / 'dataset_manifest.json').read_text())
    for filename, key in [('guqin_agent_train.jsonl', 'dataset_sha256'),
                          ('guqin_agent_train_metadata.jsonl', 'metadata_sha256')]:
        digest = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
        assert digest == manifest[key]
    assert manifest['dataset_sha256'] == '984779ea61a9c582328518bf8d4e023826bf31b8c705f927c1dfbd9d97b00f4a'
    metadata = [json.loads(line) for line in (DATA / 'guqin_agent_train_metadata.jsonl').open() if line.strip()]
    assert len(metadata) == len({r['source_sample_id'] for r in metadata}) == manifest['training_rows'] == 3976
    assert all(r['agent_stage'] == 'single_stage' for r in metadata)
    train = {r['score_key'] for r in metadata}
    test = set((ROOT / 'ABC_J/results/split_inventories/test_score_ids.txt').read_text().split())
    assert len(train) == 153 and len(test) == 48 and 'ST4GAejh' in test and not train & test
    with (ROOT / 'ABC_J/results/agent_dataset_splits.csv').open(encoding='utf-8-sig') as stream:
        splits = {r['score_key']: r for r in csv.DictReader(stream)}
    assert all(splits[k]['split'] == 'train' for k in train)
    assert all(splits[k]['split'] == 'test' for k in test)
    assert not {splits[k]['leakage_group_id'] for k in train} & {splits[k]['leakage_group_id'] for k in test}
    print(json.dumps({'valid': True, 'rows': 3976, 'train_scores': 153, 'test_scores': 48,
                      'cutoff_len': 8192, 'overlength_policy': 'standard trainer preprocessing; raw trajectories unchanged'}), flush=True)


if __name__ == '__main__':
    main()
