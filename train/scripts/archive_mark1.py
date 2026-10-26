"""Name the accepted model Mark1 and retire superseded inference snapshots."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
archive = ROOT / 'Mark1'
archive.mkdir(exist_ok=True)
trash = (Path('/Volumes/F/.Trashes/501') if str(ROOT).startswith('/Volumes/F/')
         else Path.home() / '.local/share/Trash/files')
trash.mkdir(parents=True, exist_ok=True)
retired = Path(tempfile.mkdtemp(prefix='guqin_mark1_retired_', dir=trash))
moves = (json.loads((archive / 'manifest.json').read_text()).get('retired', [])
         if (archive / 'manifest.json').exists() else [])
evals = ROOT / 'train/eval_outputs/two_stage'
target = archive / 'inference'
if not target.exists():
    assert evals.is_dir() and not evals.is_symlink()
    shutil.move(str(evals), target)
    evals.symlink_to(target, target_is_directory=True)

model = Path('/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016')
if model.is_dir() and not model.is_symlink():
    assert (model / 'adapter_model.safetensors').is_file()
    shutil.move(str(model), archive / 'weights')
    model.symlink_to(archive / 'weights', target_is_directory=True)

old = ROOT / 'train/eval_outputs_v3_two_stage'
if old.is_dir() and not old.is_symlink():
    shutil.move(str(old), retired / old.name)
    moves.append({'source': str(old), 'trash': str(retired / old.name)})

snapshot = archive / 'runtime'
snapshot.mkdir(exist_ok=True)
for rel in ['agents/ToolRuntime', 'agents/abc_to_jianzipu',
            'scripts/audit_jianpu_jianzi_pitch.py',
            'ABC_J/scripts/visualize_running_eval_score.py',
            'train/configs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12.yaml']:
    src = ROOT / rel
    dst = snapshot / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    elif src.exists():
        shutil.copy2(src, dst)
manifest = {'name': 'Mark1', 'base_model': 'Qwen3.5-9B',
            'adapter_origin': str(model), 'inference': 'inference',
            'retired': moves,
            'sha256': {str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in archive.rglob('*') if p.is_file() and p.name != 'manifest.json'}}
(archive / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'archive': str(archive), 'retired': moves}, ensure_ascii=False))
