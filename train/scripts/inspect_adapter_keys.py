import json
import sys

from safetensors import safe_open


adapter_dir = sys.argv[1]
with open(f"{adapter_dir}/adapter_config.json", encoding="utf-8") as handle:
    print(json.dumps(json.load(handle), ensure_ascii=False, indent=2))
with safe_open(f"{adapter_dir}/adapter_model.safetensors", framework="pt") as handle:
    keys = list(handle.keys())
print(f"key_count={len(keys)}")
print("\n".join(keys[:20]))
