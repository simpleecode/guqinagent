# 古琴 Agent SFT 训练与服务器部署指南

更新：2026-08-31。本文只覆盖当前第一版 Qwen3.5 工具轨迹 SFT；教师生成、私有标注审计和
reasoning 脱敏的完整历史见 `agent_training_handoff.md`。

## 1. 当前可训练数据

当前训练输入是：

```text
ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v2/
```

其中 `messages_train.jsonl` 已通过公开 reasoning 脱敏和
`validate_teacher_agent_messages.py`。私有文件 `teacher_trajectory_audit.jsonl` 只用于内部
审计，**绝不上传或导出到训练服务器**。

当前规模：8,612 条脱敏教师消息（Fingering 4,306、Guqinizer 4,306）；纯“撮＋中文数字”
候选已完成定向重跑并清零。私有审计文件仍保留原始标注和 reasoning，不能上传。

## 2. 完整多轮 trajectory 的监督方式

这不是 Qwen3.5 不支持多轮，也不是工具调用不原生。Qwen3.5 和服务器上的
LLaMA-Factory 0.9.6 都能表示：

```text
system → user → assistant（工具调用） → tool（观察） → assistant（下一次工具调用） → ...
```

需要修正一个容易混淆的说法：LLaMA-Factory 0.9.6.dev0 的 SFT 处理器会把记录拆成多轮
`source → target` 对；在默认 `mask_history=false` 下，每一轮 assistant/function target 都会
计算 loss。只有设置 `mask_history=true` 时，它才会倒序保留上下文，并只训练最后一轮 target。

当前采用一条完整 Agent 轨迹对应一条 SFT 样本：

```text
system → user → assistant A → tool A → assistant B → tool B → assistant final
```

其中：

- `mask_history=false`，所有 assistant/function target 都计算 loss；
- `train_on_prompt=false`，user 与 tool observation 只提供上下文，不计算内容 loss；
- 每个 assistant 的 reasoning 与工具调用合并为同一个 assistant `content`；
- 若原始轨迹最后停在 tool result 且没有后续 assistant，导出时删除该末尾 tool result；
- 若未来轨迹含有真正的 assistant final answer，则完整保留并参与 loss。

这样不再复制历史前缀，8,612 条教师轨迹导出为 8,612 条训练记录。导出目录的 metadata
保留 `source_sample_id`、曲目、assistant 回合数和裁掉的末尾 tool result 数量。

### 原生工具调用与公开 reasoning

服务器的 LLaMA-Factory 确实可读取 OpenAI 风格的 `tool_calls`；但该版本会将带
`tool_calls` 的 assistant 内容替换为函数 JSON，从而丢弃本项目已脱敏的公开 reasoning。
为保留 reasoning，导出器将工具调用写为 Qwen3.5 官方格式的 XML 文本，并把它与 reasoning
一起放进 assistant 监督目标：

```text
公开 reasoning
<tool_call>
<function=edit_plan>
...
</function>
</tool_call>
```

工具定义仍通过顶层 `tools` 字段注入 Qwen3.5 模板；模型在推理时仍会生成可解析的原生
Qwen3.5 工具调用表面。

### labels 抽检

`inspect_sft_labels.py` 直接调用服务器安装的 LLaMA-Factory 0.9.6.dev0
`SupervisedDatasetProcessor`，随机抽取完整轨迹并核对实际 `input_ids/labels`：每一轮 assistant
target 必须完整有 label，user/tool observation 对应的 source 内容必须为 `IGNORE_INDEX`。
只检查渲染文本不足以证明 loss mask 正确，因此该抽检是训练启动前的硬门槛。

## 3. 本地导出和校验

```powershell
conda run -n guqin-agent python train/scripts/export_sft_dataset.py `
  --input-dir ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v2 `
  --output-dir train/data/guqin_agent_sft_v5_pure_cuo_final_v2

conda run -n guqin-agent python train/scripts/validate_sft_export.py `
  --input-dir train/data/guqin_agent_sft_v5_pure_cuo_final_v2
```

导出目录：

| 文件 | 用途 | 是否上传服务器 |
|---|---|---|
| `guqin_agent_train.jsonl` | LLaMA-Factory 的稳定 OpenAI-format 训练数据 | 是 |
| `dataset_info.json` | LLaMA-Factory 数据集注册信息 | 是 |
| `dataset_manifest.json` | 数据来源、hash、规模和口径 | 是 |
| `guqin_agent_train_metadata.jsonl` | 本地回合/曲目分析索引，不参与训练 | 可选 |
| `sft_export_validation_report.json` | 本地导出校验报告 | 建议上传 |

当前导出为 8,612 条完整轨迹：Fingering 4,306、Guqinizer 4,306；共保留 22,684 个
assistant 回合。

## 4. 服务器环境与训练

服务器约定沿用 `D:\new_cip\服务器使用与网络方案.md`：4×RTX 3090 24GB、
`~/models/Qwen3.5-9B`、`~/LLaMA-Factory` 固定提交、已有 `cip` 环境与 3090 显存补丁。

不要直接改 `cip`。在当前项目上传到例如 `~/guqin-agent/` 后：

```bash
cd ~/guqin-agent
bash train/server/bootstrap_guqin_sft_env.sh
```

该命令会从 `cip` 克隆 `guqin-sft` 环境，复用已有依赖和 LLaMA-Factory checkout。

### Jobber：GPU 实验队列

服务器上的 `~/code/training/jobber.py` 可顺序管理训练和评测任务。提交任务时只声明需要的
GPU 数量，Jobber 会等待足够空闲的 GPU，并自动设置 `CUDA_VISIBLE_DEVICES`；它不会抢占、停止
或接管手工启动的进程。首次使用或服务器重启后启动守护进程（重复执行无害）：

```bash
PY=/home/20223393ljw/miniconda3/envs/cip/bin/python
$PY ~/code/training/jobber.py start
$PY ~/code/training/jobber.py status --verbose
$PY ~/code/training/jobber.py ps
```

Jobber 管理的命令必须前台运行，不能附加 `nohup`、`setsid`、`&` 或手动设置
`CUDA_VISIBLE_DEVICES`。例如，先把真实两条样本 smoke 放入队列：

```bash
$PY ~/code/training/jobber.py submit --gpus 4 --name guqin-sft-smoke --cwd ~/guqin-agent -- \
  "bash train/server/run_sft.sh smoke"
```

smoke 通过后提交完整训练：

```bash
$PY ~/code/training/jobber.py submit --gpus 4 --name guqin-sft-full --cwd ~/guqin-agent -- \
  "bash train/server/run_sft.sh full"
```

评测任务也使用同样的 `submit` 形式，将队列命令末尾替换为实际的前台评测命令。常用的
插队、暂停、恢复、取消和日志查看：

```bash
$PY ~/code/training/jobber.py move J00007 --before J00006
$PY ~/code/training/jobber.py pause
$PY ~/code/training/jobber.py resume
$PY ~/code/training/jobber.py cancel J00007          # 运行中先发送 TERM
$PY ~/code/training/jobber.py cancel J00007 --force  # TERM 无效时才使用
$PY ~/code/training/jobber.py logs J00007 --tail 80
$PY ~/code/training/jobber.py ps J00007
```

任务状态、日志和退出码位于 `~/code/training/runtime/gpu_queue/`；默认每 15 秒检查一次，
显存占用不高于 512 MiB 的 GPU 才视为可用。`status --verbose` 会显示任务 PID、进程组、启动
时间、命令和日志路径。Jobber 只查询和管理自己提交的任务，不会显示或中止未通过 Jobber
提交的手动进程；详情见服务器上的 `~/code/training/JOBBER.md`。

若不使用 Jobber，先运行真实两条样本 smoke：

```bash
cd ~/guqin-agent
PROJECT_ROOT=$PWD ENV_NAME=guqin-sft bash train/server/run_sft.sh smoke
```

若不使用 Jobber，通过后按手工后台方式启动完整训练：

```bash
cd ~/guqin-agent
setsid nohup env PROJECT_ROOT=$PWD ENV_NAME=guqin-sft \
  bash train/server/run_sft.sh full > ~/guqin_sft_v2_full.log 2>&1 < /dev/null &
```

训练配方：Qwen3.5-9B、4-bit QLoRA、bf16、4 卡 DDP、有效 batch 32、3 epoch。
服务器真实 Qwen3.5 tokenizer 实测 P50 3,455、P95 5,655、P99 6,319、最大 17,679 token，
因此 `cutoff_len` 固定为 18,432；不能沿用原先按回合样本的 2,048。该长度仍须通过
4×RTX 3090 显存 smoke。

在 2,048 cutoff 下，8,922 条中有 8,523 条超长，截断率 95.53%。普通四卡 DDP 不会合并
显存，每张卡仍需处理完整序列，因此不能靠增加 DDP 卡数解决。当前配方额外启用
`use_unsloth_gc=true`（将 checkpoint hidden states 卸载到 CPU）和 Liger kernel；待 GPU
空闲后再逐级 smoke。若 18,432 仍无法运行，只能在“压缩不参与 loss 的 user/tool 上下文”
与“改用支持 Qwen3.5 的 Megatron context/tensor parallel”之间选择；前者不改变 assistant
reasoning/tool call，但不再逐字保留 observation，后者工程与依赖成本明显更高且不支持 4-bit。

## 5. 必须先过的 preflight

`run_sft.sh` 会调用 `train/scripts/preflight_sft.py`，使用 Qwen3.5 chat template 检查：

- 全部训练消息能被模板渲染；
- p50/p95/p99/max token 长度；
- 超过当前 `cutoff_len` 的轨迹数量与比例。

任何轨迹超过当前 cutoff，全量训练都会被拒绝，因为从头截断可能使后续 assistant 回合完全
失去监督。此时应调整上下文长度并重新做 3090 显存 smoke；不要静默截断再训练。

2026-08-28 的服务器抽检随机选择第 787、5152、5617、7807、8652 条，共覆盖 8 个
assistant 回合。所有回合均满足：source 内容被 mask、target 完整监督、reasoning/tool call
文本完整保留、实际 labels 与处理器预期逐 token 一致。

## 6. 验证与测试边界

当前教师批只覆盖原始 train split，不能从 train phrase 随机切出 validation/test 并声称没有
乐曲泄漏。第一版训练配置不启用 eval；后续应在曲级独立的 validation split 生成同口径教师
轨迹，或以独立 agent 评测脚本衡量硬约束、工具调用和减字质量。原始 validation/test 分区持续
封存，直到评测协议冻结。
