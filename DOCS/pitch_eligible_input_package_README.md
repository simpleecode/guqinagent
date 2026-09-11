# pitch-eligible 输入包

包内已将完整 train 源文件放置到：

`ABC_J/agent_training/pitch_eligible_gqs_v12_train/inferred_trajectories_train.jsonl`

它包含 4,691 个 train phrase；真正需要生成的是 3,927 个 phrase，必须同时使用：

`ABC_J/agent_training/pitch_eligible_gqs_v12_train/pitch_eligible_phrase_ids.txt`

解包后从项目根目录运行：

```bash
python scripts/run_teacher_batch_parallel.py \
  --input ABC_J/agent_training/pitch_eligible_gqs_v12_train/inferred_trajectories_train.jsonl \
  --trajectory-id-file ABC_J/agent_training/pitch_eligible_gqs_v12_train/pitch_eligible_phrase_ids.txt \
  --output-dir ABC_J/agent_training/messages_gqs_v12_pitch_eligible_full_parallel8_v6 \
  --target-count 3927 \
  --workers 8 \
  --score-shard-count 8 \
  --model glm-5.3 \
  --max-tool-rounds 24 \
  --max-attempts 6 \
  --min-interval 1 \
  --allow-private-reasoning-leakage \
  --include-guqinizer-no-op
```

已有 `.parallel_workers/worker_00` 至 `worker_07` 时使用同一个输出目录续跑，不要删除断点。
