# guqin-agent Mac 迁移包

这是一个精简的可迁移包，包含当前代码、DOCS、GQS 1.2 源数据、训练/评估脚本和最新公开训练数据。
早期重复教师轨迹、反编译工作目录、运行缓存、旧模型输出和 API 密钥未包含。

## 解包后

1. 复制 `.env.example` 为 `.env`，填写需要的 API key；不要把 Windows 机器上的 `.env` 直接带入版本库。
2. 使用 Python 3.11 或更新版本创建虚拟环境，安装 `requirements-agent.txt`。
3. 按 `DOCS/agent_training_handoff.md` 和 `DOCS/agent_sft_training_guide.md` 恢复数据生成、SFT 导出和评估流程。
4. `ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3_public/` 是当前已脱敏公开轨迹；
   `train/data/guqin_agent_sft_v5_pure_cuo_final_v3/` 是对应 SFT 数据。
5. 当前 GQS 1.2 继续生成所需输入位于 `ABC_J/agent_training/inferred_gqs_v12/`，训练 split 清单位于
   `ABC_J/agent_training/pitch_eligible_gqs_v12_train/`。
6. 源数据获取/解析脚本保留在 `scripts/`；`ABC_J/final/`、`ABC_J/round2/`、`ABC_J/candidates/`、
   `ABC_J/results/` 以及 `ABC_J/agent_training/inferred_*`、`gqs_*` 和 `reference_trajectories_*` 是保留的
   源数据中间结果，可用于复现最终 GQS 与筛选过程。

包内路径统一使用相对路径，适合在 macOS 解包后直接运行；Windows 专用的 `.env`、缓存和历史实验副本需在目标机重新配置。
