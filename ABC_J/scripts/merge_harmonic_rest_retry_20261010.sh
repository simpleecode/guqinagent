#!/usr/bin/env bash
set -euo pipefail

cd /Volumes/F/work/Gunqin-agent
ROOT='ABC_J/agent_training/renderfix_train_test_20261008'
BASE="$ROOT/teacher_train_final_allblacklistclean_walk501_20261009"
REDACTED="$ROOT/teacher_harmonic_rest_leakage_retry_redacted_20261009"
OUTPUT="$ROOT/teacher_train_final_allblacklistclean_walk501_harmonicrestfix_20261010"

python -u scripts/merge_repaired_guqinizer_dataset.py \
  --base "$BASE" \
  --replacement "$REDACTED/shard_00" \
  --replacement "$REDACTED/shard_01" \
  --replacement "$REDACTED/shard_02" \
  --replacement "$REDACTED/shard_03" \
  --replacement "$REDACTED/shard_04" \
  --replacement "$REDACTED/shard_05" \
  --replacement "$REDACTED/shard_06" \
  --output "$OUTPUT"

python scripts/validate_teacher_agent_messages.py --input-dir "$OUTPUT"
