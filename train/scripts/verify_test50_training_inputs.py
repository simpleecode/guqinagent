#!/usr/bin/env python3
"""Fail closed if an obsolete split or train/test leakage reaches training."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    data=ROOT/'train/data/guqin_agent_sft_two_stage_test50'
    evaluation=ROOT/'train/eval_inputs_v2_text_protocol/test50'
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    validation=json.loads((data/'sft_export_validation_report.json').read_text())
    assert validation['valid'] and validation['rows']==manifest['training_rows']==7990
    assert sha(data/'guqin_agent_train.jsonl')==manifest['dataset_sha256']
    assert sha(data/'guqin_agent_train_metadata.jsonl')==manifest['metadata_sha256']
    expected=json.loads((evaluation/'export_manifest.json').read_text())
    assert expected['scores']==50 and expected['phrases']==999 and expected['test_filtering'] is False
    assert sha(evaluation/'test_runtime.jsonl')==expected['public_sha256']
    assert sha(evaluation/'evaluation_pairs_test.jsonl')==expected['sealed_sha256']
    with (ROOT/'ABC_J/results/agent_dataset_splits.csv').open(encoding='utf-8-sig') as handle:
        splits={r['score_key']:r for r in csv.DictReader(handle)}
    metadata=list(rows(data/'guqin_agent_train_metadata.jsonl'))
    train={r['score_key'] for r in metadata}
    test_rows=list(rows(evaluation/'test_runtime.jsonl'));test={r['score_key'] for r in test_rows}
    assert len(test_rows)==len({r['sample_id'] for r in test_rows})==999
    assert len(test)==50 and not train & test
    assert all(splits[k]['split']=='train' for k in train)
    assert all(splits[k]['split']=='test' for k in test)
    groups=Counter(splits[k]['leakage_group_id'] for k in test)
    assert len(groups)==50 and set(groups.values())=={1}
    assert not {splits[k]['leakage_group_id'] for k in train}&set(groups)
    assert {'SBAwX43Q','ST65Msa6','SDcE3Geq','SMcjde29','STGsvZsA','SbEnRCX9','SkAD4fmu'}<=train
    stages=Counter(r['agent_stage'] for r in metadata)
    assert dict(stages)=={'fingering_agent':3995,'guqinization':3995}
    print(json.dumps({'valid':True,'training_rows':7990,'training_phrases':3995,'test_scores':50,
                      'test_phrases':999,'train_test_group_overlap':[],'all_file_hashes_match':True},ensure_ascii=False))


if __name__=='__main__':main()
