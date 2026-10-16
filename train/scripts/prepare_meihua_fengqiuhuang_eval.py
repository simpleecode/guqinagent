"""Extract two complete scores from the current public test50 snapshot."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'train/eval_inputs_v2_text_protocol/test50'
target = ROOT / 'train/eval_inputs_v2_text_protocol/meihua_fengqiuhuang_latest'
keys = {'S35eJqgV', 'S1zt293d'}
target.mkdir(exist_ok=True)
manifest = {'score_titles': {'S35eJqgV': '梅花三弄', 'S1zt293d': '凤求凰'},
            'source': str(source), 'test_filtering': False, 'files': {}}
ids = []
for filename in ('test_runtime.jsonl', 'evaluation_pairs_test.jsonl'):
    selected = []
    for line in (source / filename).open():
        row = json.loads(line)
        if row['score_key'] in keys:
            selected.append((row, line))
    assert Counter(r['score_key'] for r, _ in selected) == {'S35eJqgV': 66, 'S1zt293d': 6}
    identities = {r['sample_id'] for r, _ in selected}
    assert len(identities) == len(selected) == 72
    ids.append(identities)
    output = target / filename
    with output.open('w') as handle:
        for _, line in selected:
            handle.write(line)
    manifest['files'][filename] = {'rows': 72, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
assert ids[0] == ids[1]
(target / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
