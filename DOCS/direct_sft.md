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

两组正式推理及离线评分已完成，结果独立保存在 `train/eval_outputs/direct_sft/`。ID、逐输入 SHA、权重指纹、每次生成无工具及用量均已核验；失败或缺失没有用空历史补跑。

## 最终评估结果（2026-11-03）

| 指标 | 古谱：仅有效连续前缀的条件诊断 | 现代曲：完整91段 |
|---|---:|---:|
| 预期 / 实际尝试 / 协议有效 | 1031 / 1005 / 1003 | 91 / 91 / 91 |
| 尝试后失败 / 前驱失败跳过 | 2 / 26 | 0 / 0 |
| 有效输出覆盖率 | 97.2842% | 100% |
| 条件 PitchAcc@50c | 85.2773% | 79.0916% |
| 可评估音高事件 / 音高配对数 | 14,909 / 15,571 | 1,387 / 1,622 |
| 音高 MAE（cents） | 57.2911 | 90.5756 |
| 弦徽 F1 | 37.3686% | 40.7368% |
| 左手 / 右手 / 音色 F1 | 59.9796 / 51.1717 / 70.6069% | 57.0080 / 59.3612 / 68.5596% |
| Orn. F1 / Tech. Sim. | 27.6042% / 0.753580 | 43.4368% / 0.836374 |
| 最终规则违例 / 涉及片段 | 2703 / 628 | 378 / 78 |
| 规则违例比率（仅已评估有效片段） | 628/1003 = 62.6122% | 78/91 = 85.7143% |
| 与轨迹单阶段共同可评估事件 | 14,000 | 1,315 |
| 共同事件 PitchAcc：Direct / 轨迹单阶段 | 85.6429% / 98.8214% | 80.7605% / 98.0228% |
| vLLM报告输入 / 输出 token | 1,222,415 / 218,369 | 114,109 / 21,990 |
| 生成次数 / 工具调用 / 输出截断 | 1005 / 0 / 0 | 91 / 0 / 0 |

古谱失败为 `SeBx4Lvv-p0004` 和 `SncH0ay9-p0023` 的 JSON 格式错误，均正常 stop 而非 token 截断；尝试后协议成功率1003/1005=99.8010%，不能与有效输出覆盖率1003/1031混用。音乐指标只针对1003段有效连续前缀，按曲从首段顺序重放，没有逐phrase重置历史；失败2段与跳过26段仍单列，不能把该组当作完整测试集。论文古谱主表已填入条件实测值并加脚注，注明仅针对1003段有效连续前缀；完整分母及共同事件诊断保留在附录。

音高可评估覆盖若以全测试目标事件作分母，古谱为14,909/18,128=82.2429%，确认正确事件12,714/18,128=70.1346%；现代为1,387/1,528=90.7723%，确认正确1,097/1,528=71.7932%。这些比率与条件音高准确率不同。

证据：`evaluation_results/direct_sft_current48/`、`evaluation_results/modern_direct_sft/`；核验与共同事件汇总 `train/eval_outputs/direct_sft/comparison.json`，可用 `train/scripts/summarize_direct_sft.py` 重建。评分入口新增 `--strict-prefix`，只评连续有效前缀并保留失败/跳过分母；不改变既有完整主评估结果。现代参考覆盖和弦徽规则与其他方法一致。

Direct在现代曲的装饰标签一致性接近轨迹单阶段，但音高准确率和规则一致性明显较低；这说明最终谱面模仿不等同于音高约束下的工具交互生成。工具访问、推理预算和监督形式同时变化，不能将差距全归因为训练轨迹。上述 token 是保存响应中的实测用量，不与论文中其他方法的重分词 API 等价成本混为同一种统计。
