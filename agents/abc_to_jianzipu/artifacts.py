from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Small, atomic artifact store used alongside LangGraph checkpoints."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, value: Any) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, target)
        return target

    def write_text(self, name: str, value: str) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, target)
        return target
