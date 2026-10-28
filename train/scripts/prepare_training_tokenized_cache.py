"""Run the installed LLaMA-Factory data pipeline on CPU before GPU scheduling."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-rows', type=int, default=7952)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data = Path(config['dataset_dir'])
    destination = Path(config['tokenized_path'])
    assert not destination.exists(), 'Refuse stale or partial tokenized cache'
    config.update(use_cpu=True, bf16=False, fp16=False, report_to='none',
                  quantization_bit=None, enable_liger_kernel=False, use_unsloth_gc=False,
                  preprocessing_num_workers=16, overwrite_cache=True)
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    model_args, data_args, training_args, _, _ = get_train_args(config)
    module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(module['tokenizer'], data_args)
    datasets = get_dataset(template, model_args, data_args, training_args, stage='sft', **module)
    assert len(datasets['train_dataset']) == args.expected_rows
    h = hashlib.sha256((data / 'guqin_agent_train.jsonl').read_bytes()).hexdigest()
    manifest = {'dataset_sha256': h, 'rows': args.expected_rows, 'template': config['template'],
                'cutoff_len': config['cutoff_len'], 'tokenized_path': str(destination),
                'train_on_prompt': config['train_on_prompt'], 'mask_history': config['mask_history']}
    (data / 'tokenized_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
