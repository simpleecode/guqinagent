"""Show final numbered-repeat teachers using the existing score-sharded viewer."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import NUMBERED_REPEAT_PATTERN


def rows(path):
    for line in path.open():
        if line.strip():
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for record in rows(args.directory / 'teacher_trajectory_audit.jsonl'):
        sid = record['sample_id']
        if not sid.endswith('-guqinization-teacher-tools'):
            continue
        matches = []
        for action in record['teacher_private']['accepted_plan']['actions']:
            text = str(action.get('jianzi_text') or '')
            if match := re.search(NUMBERED_REPEAT_PATTERN, text):
                matches.append({'source_index': action['source_index'],
                                'target_event_index': int(match.group(1)), 'jianzi_text': text})
        if matches:
            selected[sid.removesuffix('-guqinization-teacher-tools')] = matches
    assert selected, 'No numbered-repeat phrases in final accepted Guqinizer plans'
    scores = {pid.split('-p', 1)[0] for pid in selected}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='guqin-numbered-repeat-') as name:
        temp = Path(name)
        sample_ids = set()
        with (temp / 'messages.jsonl').open('w') as out:
            for r in rows(args.directory / 'messages_train.jsonl'):
                if r.get('provenance', {}).get('source_trajectory_id') in selected:
                    assert r['sample_id'] not in sample_ids
                    sample_ids.add(r['sample_id'])
                    out.write(json.dumps(r, ensure_ascii=False) + '\n')
        audit_ids = set()
        with (temp / 'audit.jsonl').open('w') as out:
            for r in rows(args.directory / 'teacher_trajectory_audit.jsonl'):
                if r['sample_id'] in sample_ids:
                    assert r['sample_id'] not in audit_ids
                    audit_ids.add(r['sample_id'])
                    out.write(json.dumps(r, ensure_ascii=False) + '\n')
        assert sample_ids == audit_ids and len(sample_ids) == len(selected) * 2
        with (temp / 'source.jsonl').open('w') as out:
            for r in rows(args.source):
                if r['score_key'] in scores:
                    out.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['trajectory_id'] in selected:
                    mapping = {n['index']: n.get('event_index') for n in r['input']['notes_without_jianzi']}
                    for match in selected[r['trajectory_id']]:
                        match['event_index'] = mapping[match['source_index']]
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name('visualize_all_trajectories_hierarchical_sharded.py')),
            '--input', str(temp / 'messages.jsonl'), '--audit', str(temp / 'audit.jsonl'),
            '--source', str(temp / 'source.jsonl'), '--output-dir', str(args.output_dir),
            '--notice', f'已GLM脱敏的正式训练数据。仅展示最终Guqinizer计划使用“从数字音序再作”的{len(selected)}个phrase，包含两阶段实际轨迹，保留完整曲谱源上下文；不包含仅在中间轮提过、最终撤销的数字再作。',
        ], check=True)
    summary = {
        'phrase_count': len(selected), 'score_count': len(scores),
        'stage_trajectories': len(sample_ids),
        'source_messages': str((args.directory / 'messages_train.jsonl').resolve()),
        'selection_basis': 'numbered repeat in final accepted Guqinizer plan',
        'phrases': selected,
    }
    (args.output_dir / 'selection_manifest.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'phrases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
