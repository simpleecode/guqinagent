#!/usr/bin/env python3
"""One tool-free generation per phrase, preserving each score's generated history."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.direct_generation import PROTOCOL, inference_messages, parse_answer


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_http(messages, args):
    # Tool schemas, tool_choice, candidate queries and diagnostic retries are deliberately absent.
    payload = {'model': args.model, 'messages': messages, 'max_tokens': args.max_new_tokens,
               'temperature': 0.0, 'chat_template_kwargs': {'enable_thinking': False}}
    request = urllib.request.Request(args.vllm_url.rstrip('/') + '/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    choice = result['choices'][0]
    return {'text': choice['message'].get('content') or '', 'finish_reason': choice.get('finish_reason'),
            'returned_tool_calls': choice['message'].get('tool_calls') or [],
            'usage': result.get('usage') or {}, 'seconds': time.monotonic() - start}


def run_score(sources, prior, generate, emit):
    previous = None
    historical = {}
    for source in sorted(sources, key=lambda r: int(r['runtime_item']['input']['event_range']['start'])):
        sid = source['sample_id']
        item = deepcopy(source['runtime_item'])
        record = prior.get(sid)
        if record is None:
            messages = inference_messages(item, previous, historical)
            response = generate(messages)
            try:
                if response.get('returned_tool_calls'):
                    raise ValueError('unexpected tool call in direct output')
                _, plan = parse_answer(response['text'], item)
                error = None
            except (ValueError, TypeError, KeyError) as exc:
                plan, error = None, f'{type(exc).__name__}: {exc}'
            record = {'schema_version': 'agent-eval-prediction-1.3', 'inference_mode': PROTOCOL,
                'sample_id': sid, 'score_key': source['score_key'], 'phrase_id': source['phrase_id'],
                'split': source.get('split'), 'input_sha256': source['input_sha256'],
                'protocol_valid': plan is not None, 'parse_error': error,
                'direct': {'ok': plan is not None, 'plan': plan, 'generation': response,
                           'request_messages': messages, 'model_generations': 1, 'tool_calls': 0},
                'jianzi_rows': [[a['source_index'], a['jianzi_text']] for a in (plan or {}).get('actions', [])]}
            emit(record)
        if not record['protocol_valid']:
            break  # No reference or blank-history rescue after a predecessor failure.
        item['reference_plan'] = deepcopy(record['direct']['plan'])
        previous = item
        historical[(source['score_key'], source['phrase_id'])] = item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vllm-url', default='http://127.0.0.1:8999/v1')
    parser.add_argument('--model', default='guqin-direct')
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if urlparse(args.vllm_url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('This evaluator expects a local vLLM server')
    if args.workers < 1:
        parser.error('workers must be positive')
    source = read(args.input)
    sources = {r['sample_id']: r for r in source}
    if len(sources) != len(source):
        raise ValueError('duplicate input ID')
    grouped = defaultdict(list)
    for r in source:
        if r.get('reference') is not None or r['runtime_item'].get('reference_plan') is not None:
            raise ValueError('private reference in inference input')
        grouped[r['score_key']].append(r)
    config = {'protocol': PROTOCOL, 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'checkpoint_sha256': args.checkpoint_sha256, 'model': args.model,
              'max_new_tokens': args.max_new_tokens, 'temperature': 0.0,
              'thinking': False, 'tools': False, 'generations_per_phrase': 1,
              'history': 'own_generated_only; skip_suffix_after_failure'}
    if args.dry_run:
        for group in grouped.values():
            first = min(group, key=lambda r: int(r['runtime_item']['input']['event_range']['start']))
            inference_messages(first['runtime_item'], None, {})
        print(json.dumps(dict(config, scores=len(grouped), phrases=len(source), dry_run=True)))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / 'direct_config.json'
    if config_path.exists():
        assert args.resume and json.loads(config_path.read_text()) == config, 'Refuse changed direct experiment'
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    output = args.output_dir / 'predictions.jsonl'
    existing = read(output) if output.exists() else []
    if existing and not args.resume:
        raise ValueError('existing predictions require --resume')
    prior = {r['sample_id']: r for r in existing}
    assert len(prior) == len(existing)
    for sid, record in prior.items():
        assert sid in sources and record['input_sha256'] == sources[sid]['input_sha256']
        assert record.get('inference_mode') == PROTOCOL
    # A resumed valid record may not follow a failed/missing predecessor.
    for group in grouped.values():
        gap = False
        for r in sorted(group, key=lambda r: int(r['runtime_item']['input']['event_range']['start'])):
            record = prior.get(r['sample_id'])
            if record is not None and gap:
                raise ValueError('resume history has a missing or failed predecessor')
            gap = gap or record is None or not record['protocol_valid']
    lock = threading.Lock()
    with output.open('a', encoding='utf-8') as handle:
        def emit(record):
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                prior[record['sample_id']] = record
                print(json.dumps({'event': 'direct_phrase', 'sample_id': record['sample_id'],
                    'protocol_valid': record['protocol_valid'], 'attempted': len(prior), 'expected': len(source)}), flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(grouped))) as pool:
            futures = [pool.submit(run_score, group, prior, lambda messages: generate_http(messages, args), emit)
                       for group in grouped.values()]
            for future in as_completed(futures):
                future.result()
    valid = sum(bool(r['protocol_valid']) for r in prior.values())
    skipped = sorted(sources.keys() - prior.keys())
    summary = {'expected': len(source), 'attempted': len(prior), 'protocol_valid': valid,
        'attempted_failures': len(prior) - valid, 'valid_output_coverage': valid / len(source),
        'skipped_after_invalid_predecessor': len(skipped), 'skipped_sample_ids': skipped,
        'model_generations': len(prior), 'tool_calls': 0}
    (args.output_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
