# Direct SFT 对照

## 问题与口径

比较从公开输入直接预测最终减字谱的流程，与学习教师工具交互轨迹的单阶段流程。Direct 每段只生成一次最终 JSON，不提供工具、不查询候选、不展示诊断反馈，也不按 warning 追加修改；关闭模型显式思维链。各曲顺序推理，后段仅继承本模型已成功生成的前段；协议失败后停止该曲后续段，并分别报告尝试失败和跳过。

该比较同时改变监督形式和推理交互预算。它检验完整流程，不能把全部差异归因于训练轨迹本身；相同 epoch 数也不意味着相同监督 token 数。

## 数据

- 唯一数据目录：`train/data/guqin_direct_sft/`。
- 3976 条、153 首，样本 ID 与当前单阶段 SFT 逐项一致；传统测试 48 首和现代曲 10 首的曲目 ID 均无交叉，传统 train/test 的 leakage group 无交叉。现代曲旋律层面独立性仍未由此建立。
- 输入保留单阶段 SFT 的初始公开音乐上下文，移除工具使用指令和诊断后缀；使用 Direct 专用输出指令。
- 目标为同一批教师轨迹的最终实际提交谱面。按真实成功回执合并全部 `edit_plan` 增量，再逐行比对 private audit 的 `accepted_plan`；只导出最终减字文本，不导出私有推理或参考请求。
- 每条仅 `system/user/assistant` 三条消息，`tools=[]`，无中间推理或工具回执。每个演奏事件都输出一行，显式保留合法的空字符串，排除小节线。
- `guqin_agent_train.jsonl` SHA256：`edba4abf12f1d2834bd079ea30f7ed1ef50238b2a631db6948712b18d102f7e9`。
- 该目标取自教师最终结果；与按原始标注填写的固定流程轨迹消融有区别，不能将两者混为同一个基线。

数据构造：`train/scripts/build_direct_sft.py`。构造及验证均不调用模型 API。

```bash
conda run -n guqin-agent python train/scripts/build_direct_sft.py
conda run -n guqin-agent python train/scripts/verify_direct_sft.py \
  --source-dir ABC_J/agent_training/single_stage_audited/sft_export \
  --test-input train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl \
  --modern-input train/eval_inputs/modern_audited.jsonl
```

已验收的正式数据目录存在时，构造器拒绝覆盖。`--replace-partial` 仅用于修复本构造器尚未写完 manifest 的不完整导出，不接受替换已验收数据。

## 训练

配置：`train/configs/qwen35_9b_lora_direct_sft_1gpu_bs12.yaml`；入口：`train/scripts/run_direct_sft_train.sh`。

沿用 Qwen3.5-9B、4bit all-LoRA、单卡 batch1、累积12、3 epoch、lr2e-4、cutoff8192。采用 `qwen3_5` 模板、`enable_thinking=false`，仅监督 assistant 最终答案。独立权重目录：`/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_direct_sft_1gpu_bs12`。

`prepare_direct_training_cache.py` 使用服务器实际安装的 LLaMA-Factory 做 CPU 预处理，逐条解码监督标签，确认完整目标字符串确实出现在标签中。若训练框架保存缓存后直接退出，则加载已存缓存继续验收，不删除缓存。验收后的 `tokenized_manifest.json` 记录真实行数、最大长度、监督 token 数和完整目标覆盖数。训练前还要求专用单卡连续三次已用显存不超过512 MiB。

本次实际缓存验收：3976/3976 份最终答案完整进入监督，最大2395 token、均值1412.760 token、单个 epoch 共852410个监督 token，无需裁剪。数据及 metadata SHA256 与启动时一致。

训练于2026年11月3日03:05（服务器日志时间）完成：3 epoch、996/996次优化更新，训练耗时6小时4分21.15秒，平均训练 loss 为0.2215267（不是测试指标）。最终 `adapter_model.safetensors` 为86,630,872字节，SHA256：`30c0e8fcd8fb7387d86f66677f8596c24327a693eced4f5ac8131dfcb27d0913`。权重保留在上述独立服务器目录；必要日志、训练状态、配置及数据缓存验收报告已保存到 `train/eval_outputs/direct_sft_training_audit/`，未复制全部 checkpoint。

## 推理与评分

共享协议：`agents/abc_to_jianzipu/direct_generation.py`；推理：`train/scripts/eval_direct.py`；单卡服务入口：`train/scripts/run_direct_vllm_eval.py`。

模型返回 `{"jianzi_rows":[[演奏事件序号,"减字文本"],...]}`；推理器严格检查完整覆盖、重复/外来序号、行类型，并映射回原始 source index。每曲串行，可按曲并发最多8个客户端。恢复任务须匹配输入、checkpoint 和协议，禁止跳过失败前驱后用空历史补跑。HTTP 请求不含 tools/tool_choice；一次生成上限8192 token，上下文16384，temperature0，不按失败答案追加模型调用。网络请求失败会中止任务供诊断，不能当作音乐协议失败悄悄续跑。

```bash
# 服务入口在已部署 Direct 代码的冻结 runtime 中运行，使用已训练完成的 adapter。
CUDA_VISIBLE_DEVICES=0 conda run -n guqin-vllm-019 python train/scripts/run_direct_vllm_eval.py \
  --input train/eval_outputs/base_two_stage/inputs.jsonl \
  --output-dir train/eval_outputs/direct_sft/test48 \
  --base-model /home/20223393ljw/models/Qwen3.5-9B \
  --adapter /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_direct_sft_1gpu_bs12
```

现代曲换成 `train/eval_outputs/modern_comparison/inputs.jsonl`，使用独立的 `train/eval_outputs/direct_sft/modern` 输出目录。`evaluation.parser.final_actions` 已支持 `direct.plan`；输出可用相同 `evaluation.run_eval`、同一密封参考及完整历史规则离线评分。

2026年11月3日已启动两组正式评估，统一入口 `train/scripts/run_direct_sft_eval.sh`：GPU0、8个客户端，依次古谱48首1031段和现代曲10首91段，端口8999；运行于服务器冻结 runtime，独立目录 `train/eval_outputs/direct_sft/`，进度见 `coordinator.log`。入口核验最终 adapter 和两组公共输入 SHA256，连续三次可用显存不少于20 GiB后启动模型；不干预其他进程。当前尚无验收后的测试成绩。
