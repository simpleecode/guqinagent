param([Parameter(Mandatory=$true)][int]$ShardIndex)
$ErrorActionPreference = 'Stop'
$project = 'D:\sitongli\sitongli-guanshanyue'
$root = 'C:\Users\30343\.codex\visualizations\2026\09\03\01a065f1-9f9a-77a2-97e1-effc6361e996\gqs_v12_train_run1'
$python = 'D:\Program Files\miniconda\python.exe'
$script = Join-Path $project 'ABC_J\scripts\generate_teacher_tool_trajectories.py'
$input = Join-Path $project 'ABC_J\agent_training\inferred_gqs_v12_tuningfix_20260914\inferred_trajectories_train.jsonl'
$ids = Join-Path $project 'ABC_J\agent_training\pitch_eligible_gqs_v12_train\pitch_eligible_phrase_ids.txt'
$output = Join-Path $root "messages_gqs_v12_train_run1_shard_$ShardIndex"
& $python $script --input $input --trajectory-id-file $ids `
  --max-tool-rounds 24 --max-attempts 6 --allow-private-reasoning-leakage `
  --include-guqinizer-no-op --model glm-5.3 --min-interval 1 `
  --score-shard-count 6 --score-shard-index $ShardIndex --resume --retry-failed `
  --output-dir $output
