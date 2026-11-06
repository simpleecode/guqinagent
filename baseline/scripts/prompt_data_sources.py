"""Current user-selected data sources (2026-10-29); no legacy fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
MAIN_TEST = REPO / "train/eval_inputs_v2_text_protocol/test.jsonl"
TEST50 = REPO / "train/eval_inputs_v2_text_protocol/test50"


def training_identity() -> dict[str, str]:
    if not TRAIN.is_file():
        raise SystemExit(f"New training source is missing: {TRAIN}")
    digest = hashlib.sha256()
    with TRAIN.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"source": str(TRAIN.resolve()), "source_sha256": digest.hexdigest()}


def require_current_split(identity: dict[str, str]) -> None:
    manifest_path = TRAIN.parent / "prompt_iteration_subsplit_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            "新版训练数据尚未建立迭代划分：请先在新版数据上生成并校验划分名单。"
            "旧数据的样本名单和检查点不能继续用于新版训练。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_sha256") != identity["source_sha256"]:
        raise SystemExit("Iteration split does not match the current training dataset SHA-256.")
