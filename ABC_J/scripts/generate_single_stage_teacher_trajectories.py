#!/usr/bin/env python3
"""Generate direct, final-notation teacher trajectories in one agent stage.

Unlike the paired Fingering -> Guqinizer corpus, every selected phrase starts
from a blank display plan and one agent iterates with the public tools until it
has produced the final notation.  Private reference material is used only by
the teacher runner; write ``messages_train.jsonl`` through the existing
redaction workflow before it becomes SFT data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_teacher_tool_trajectories import (  # noqa: E402
    _RateLimitedClient, blank_plan_from_item, generate_with_retries,
    infer_jianzi_text_patches, is_all_empty_reference_phrase,
    load_dotenv, validate_normalized_tuning,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def score_bucket(score_key: str, count: int) -> int:
    digest = hashlib.blake2b(score_key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-id", action="append")
    parser.add_argument("--trajectory-id-file", type=Path)
    parser.add_argument("--score", action="append", help="limit to score_key; repeatable")
    parser.add_argument("--max-tool-rounds", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-interval", type=float)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-private-reasoning-leakage", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in 0..shard-count-1")

    load_dotenv(ROOT / ".env")
    from anthropic import Anthropic
    model = args.model or os.getenv("GLM_MODEL", "glm-5.3")
    if not model.lower().startswith("glm"):
        raise SystemExit("single-stage teacher generation is GLM-only")
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        raise SystemExit("GLM API credentials are not configured")
    interval = args.min_interval if args.min_interval is not None else float(
        os.getenv("TEACHER_MIN_INTERVAL", "2")
    )
    client = _RateLimitedClient(Anthropic(
        api_key=os.environ["GLM_API_KEY"], base_url=os.environ["GLM_BASE_URL"].rstrip("/"),
        timeout=120.0, max_retries=1,
    ), interval)

    rows = read_jsonl(args.input)
    for row in rows:
        validate_normalized_tuning(row)
    historical = {(row["score_key"], row["phrase_id"]): row for row in rows}
    wanted = set(args.trajectory_id or [])
    if args.trajectory_id_file:
        wanted.update(line.strip() for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines() if line.strip())
    wanted_scores = set(args.score or [])
    eligible = [row for row in rows if not is_all_empty_reference_phrase(row)]
    if wanted:
        eligible = [row for row in eligible if row["trajectory_id"] in wanted]
    if wanted_scores:
        eligible = [row for row in eligible if row["score_key"] in wanted_scores]
    if args.shard_count > 1:
        eligible = [row for row in eligible if score_bucket(str(row["score_key"]), args.shard_count) == args.shard_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "public": args.output_dir / "messages_train.jsonl",
        "private": args.output_dir / "teacher_trajectory_audit.jsonl",
        "rejected": args.output_dir / "teacher_rejected_io.jsonl",
        "checkpoint": args.output_dir / "checkpoint.jsonl",
    }
    if not args.resume and any(path.exists() for path in paths.values()):
        raise SystemExit(f"output directory is not empty; use --resume: {args.output_dir}")
    checkpoint = read_jsonl(paths["checkpoint"])
    completed = {str(row.get("trajectory_id")) for row in checkpoint if row.get("status") == "completed"}
    if not args.retry_failed:
        completed.update(str(row.get("trajectory_id")) for row in checkpoint if row.get("status") == "attempted_with_failure")
    selected = [row for row in eligible if row["trajectory_id"] not in completed]
    mode = "a" if args.resume else "w"
    counts = Counter(row.get("agent_stage") for row in read_jsonl(paths["public"]))
    failures: list[dict] = []
    rejected_traces: list[dict] = []
    with (paths["public"].open(mode, encoding="utf-8", newline="\n") as public_out,
          paths["private"].open(mode, encoding="utf-8", newline="\n") as private_out,
          paths["checkpoint"].open(mode, encoding="utf-8", newline="\n") as checkpoint_out):
        for position, source in enumerate(tqdm(selected, total=len(selected), desc="single-stage trajectories", unit="phrase"), 1):
            item = deepcopy(source)
            item["baseline_plan"] = blank_plan_from_item(item)
            targets = infer_jianzi_text_patches(item["baseline_plan"], item["reference_plan"]["actions"])["patches"]
            print(f"phrase {position}/{len(selected)}: {item['trajectory_id']}", flush=True)
            try:
                public, private = generate_with_retries(
                    client, model, item, "single_stage", targets, historical,
                    args.max_tool_rounds, args.max_attempts, objective="toward_reference",
                    failure_traces=rejected_traces,
                    allow_private_reasoning_leakage=args.allow_private_reasoning_leakage,
                )
                private_out.write(json.dumps(private, ensure_ascii=False) + "\n")
                if public["verification"].get("training_eligible"):
                    public_out.write(json.dumps(public, ensure_ascii=False) + "\n")
                    counts["single_stage"] += 1
                checkpoint_out.write(json.dumps({"trajectory_id": item["trajectory_id"], "status": "completed"}, ensure_ascii=False) + "\n")
            except Exception as exc:
                failure = {"trajectory_id": item["trajectory_id"], "stage": "single_stage", "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                checkpoint_out.write(json.dumps({"trajectory_id": item["trajectory_id"], "status": "attempted_with_failure", "stage": "single_stage"}, ensure_ascii=False) + "\n")
            public_out.flush(); private_out.flush(); checkpoint_out.flush()
    with paths["rejected"].open(mode, encoding="utf-8", newline="\n") as rejected_out:
        for trace in rejected_traces:
            rejected_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
    report = {"schema_version": "single-stage-teacher-report-1.0", "workflow": "direct_final_from_blank", "model": model, "requested": len(eligible), "selected": len(selected), "accepted": counts["single_stage"], "rejected": len(failures), "failures": failures}
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if counts["single_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
