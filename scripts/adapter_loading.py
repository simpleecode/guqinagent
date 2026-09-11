"""Safe loading helpers for PEFT adapters used by evaluation scripts."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_LORA_SUFFIX = re.compile(r"\.lora_(A|B)\.weight$")
_PEFT_PREFIXES = ("base_model.model.", "base_model.")


def _parameter_key(saved_key: str, adapter_name: str = "default") -> str:
    """Convert a PEFT saved key to the corresponding loaded parameter key."""
    return _LORA_SUFFIX.sub(
        lambda match: f".lora_{match.group(1)}.{adapter_name}.weight", saved_key
    )


def _mapped_saved_key(saved_key: str, key_mapping: dict[str, str] | None) -> str:
    """Apply PEFT's key mapping to a saved key for validation."""
    if not key_mapping:
        return saved_key
    prefix = next((p for p in _PEFT_PREFIXES if saved_key.startswith(p)), None)
    if prefix is None:
        raise RuntimeError(f"unexpected PEFT key prefix: {saved_key}")
    body = saved_key.removeprefix(prefix)
    for pattern, replacement in key_mapping.items():
        body_new, count = re.subn(pattern, replacement, body)
        body = body_new
        if count:
            break
    return prefix + body


def _choose_key_mapping(model: Any, saved_keys: list[str]) -> dict[str, str] | None:
    """Choose the mapping that matches the base model's actual module paths."""
    module_names = set(dict(model.named_modules()))
    targets = []
    for key in saved_keys:
        body = key
        for prefix in _PEFT_PREFIXES:
            if body.startswith(prefix):
                body = body.removeprefix(prefix)
                break
        body = _LORA_SUFFIX.sub("", body)
        targets.append(body)

    direct = sum(target in module_names for target in targets)
    legacy_mapping = {r"^model\.language_model\.": "model."}
    legacy = sum(
        re.sub(r"^model\.language_model\.", "model.", target) in module_names
        for target in targets
    )
    if direct == len(targets):
        return None
    if legacy == len(targets):
        return legacy_mapping
    raise RuntimeError(
        "Cannot resolve LoRA key mapping against the loaded base model: "
        f"direct={direct}/{len(targets)}, model.language_model→model={legacy}/{len(targets)}"
    )


def load_adapter_checked(model: Any, adapter_path: str | Path, *, adapter_name: str = "default") -> Any:
    """Attach an adapter without key rewriting and verify every saved tensor.

    PEFT's convenience loader uses ``strict=False`` internally.  A bad key
    mapping can therefore look successful while leaving all LoRA B matrices
    at zero initialisation.  Compare the saved adapter state with loaded
    parameters and fail loudly on any missing or changed tensor.
    """
    import torch
    from peft import PeftModel
    from peft.utils.save_and_load import load_peft_weights

    adapter_path = str(adapter_path)
    saved = load_peft_weights(adapter_path, device="cpu")
    if not saved:
        raise RuntimeError(f"LoRA adapter has no tensors: {adapter_path}")

    key_mapping = _choose_key_mapping(model, list(saved))
    wrapped = PeftModel.from_pretrained(
        model, adapter_path, adapter_name=adapter_name, key_mapping=key_mapping
    ).eval()
    loaded = {
        name: parameter
        for name, parameter in wrapped.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }

    expected = {
        _parameter_key(_mapped_saved_key(key, key_mapping), adapter_name): value
        for key, value in saved.items()
    }
    missing = [key for key in expected if key not in loaded]
    if missing:
        preview = ", ".join(missing[:4])
        raise RuntimeError(
            f"LoRA adapter key mismatch: {len(missing)}/{len(expected)} saved tensors "
            f"are absent after loading; examples: {preview}"
        )

    mismatched: list[str] = []
    for key, saved_value in expected.items():
        actual = loaded[key].detach().float().cpu()
        wanted = saved_value.detach().float().cpu()
        if actual.shape != wanted.shape or not torch.allclose(actual, wanted, rtol=0.0, atol=1e-6):
            mismatched.append(key)
            if len(mismatched) >= 4:
                break
    if mismatched:
        raise RuntimeError(
            "LoRA adapter tensors did not match after loading; "
            f"examples: {', '.join(mismatched)}"
        )

    b_keys = [key for key in expected if ".lora_B." in key]
    nonzero_b = sum(bool(torch.count_nonzero(loaded[key].detach()).item()) for key in b_keys)
    if b_keys and nonzero_b == 0:
        raise RuntimeError(
            "LoRA adapter loaded but every lora_B tensor is zero; "
            "this usually indicates an incorrect key mapping or an untrained adapter"
        )

    print(
        f"LoRA adapter verified: {len(expected)} tensors, "
        f"{len(b_keys)} lora_B tensors ({nonzero_b} nonzero), "
        f"key_mapping={'none' if key_mapping is None else 'model.language_model→model'}, "
        f"path={adapter_path}",
        flush=True,
    )
    return wrapped
