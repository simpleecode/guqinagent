#!/usr/bin/env python3
"""Use the real trainer tokenizer and verify every complete artifact is supervised."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data = Path(config['dataset_dir'])
    cache = Path(config['tokenized_path'])
    dataset_sha = hashlib.sha256((data / 'guqin_agent_train.jsonl').read_bytes()).hexdigest()
    assert dataset_sha == json.loads((data / 'dataset_manifest.json').read_text())['dataset_sha256']
    request = {'dataset_sha256': dataset_sha, 'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()}
    identity = data / 'tokenized_build_request.json'
    if identity.exists():
        assert json.loads(identity.read_text()) == request
    else:
        assert not cache.exists(), 'Refuse cache without build provenance'
        identity.write_text(json.dumps(request, indent=2) + '\n')
    from datasets import DatasetDict, load_from_disk
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    options = dict(config, use_cpu=True, bf16=False, fp16=False, report_to='none',
                   quantization_bit=None, enable_liger_kernel=False, use_unsloth_gc=False)
    model_args, data_args, training_args, _, _ = get_train_args(options)
    module = load_tokenizer(model_args)
    tokenizer = module['tokenizer']
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if not cache.exists():
        try:
            get_dataset(template, model_args, data_args, training_args, stage='sft', **module)
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
    cached = load_from_disk(str(cache))
    train = cached['train'] if isinstance(cached, DatasetDict) else cached
    assert len(train) == 3976
    lengths = []
    supervised_tokens = 0
    with (data / 'guqin_agent_train.jsonl').open(encoding='utf-8') as handle:
        for encoded, line in zip(train, handle):
            target = json.loads(line)['messages'][-1]['content']
            supervised = [token for token, label in zip(encoded['input_ids'], encoded['labels']) if label != -100]
            assert supervised and target in tokenizer.decode(supervised, skip_special_tokens=False), 'Missing or cropped artifact supervision'
            supervised_tokens += len(supervised)
            lengths.append(len(encoded['input_ids']))
    assert len(lengths) == 3976 and max(lengths) <= 8192
    manifest = {'dataset_sha256': dataset_sha, 'rows': 3976, 'template': config['template'],
        'cutoff_len': config['cutoff_len'], 'train_on_prompt': config['train_on_prompt'],
        'mask_history': config['mask_history'], 'enable_thinking': config['enable_thinking'],
        'complete_supervised_artifacts': 3976, 'max_tokens': max(lengths),
        'mean_tokens': sum(lengths) / len(lengths), 'supervised_tokens': supervised_tokens}
    (data / 'tokenized_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
