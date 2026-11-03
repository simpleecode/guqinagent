#!/usr/bin/env python3
"""Expose actual fixed-ablation tool audits to the shared message validator."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data = root / 'ABC_J/agent_training/single_stage_fixed_reference_ablation'
with (data / 'tool_audit.jsonl').open(encoding='utf-8') as source, (data / 'teacher_trajectory_audit.jsonl').open('x', encoding='utf-8') as target:
    for line in source:
        if line.strip():
            row = json.loads(line)
            target.write(json.dumps({'sample_id': row['sample_id'], 'teacher_private': {'tool_execution_log': row['calls'], 'teacher_model': None, 'generation_mode': 'deterministic_no_api'}}, ensure_ascii=False) + '\n')
