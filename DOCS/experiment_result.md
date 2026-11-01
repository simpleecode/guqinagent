# 论文实验结果记录

## 自动评估分布指标补充

以下数值直接取自各组已保存的 `summary.json`：`metrics.performance_technique_usage.distribution_similarity` 与 `metrics.tone_type.distribution_similarity`，均为 `1 - total_variation_distance`，范围 0–1，越高越接近参考分布。技法按每事件去重计数后归一化，不检验具体事件位置或顺序。

| 方法 | 技法分布相似度 | 音色分布相似度 | 最终告警段占比 |
|---|---:|---:|---:|
| 两阶段完整系统 ON | 0.730055 | 0.774339 | 0.194% |
| 两阶段诊断反馈 OFF | 0.761643 | 0.781012 | 78.468% |
| 单阶段 SFT | 0.761828 | 0.811330 | 14.258% |
| 固定流程轨迹 SFT | 0.840082 | 0.832045 | 95.829% |
| GLM-5.3 ReAct + Handbook | 0.632294 | 0.843893 | 5.044% |

证据分别为 `evaluation_results/warning_ablation_current48_on/summary.json`、`evaluation_results/warning_ablation_current48_off/summary.json`、`evaluation_results/single_stage_current48/summary.json`、`evaluation_results/fixed_reference_ablation_current48/summary.json`、`evaluation_results/react_handbook_full_test50/summary.json`。现有“弦号 F1”仅比较弦号；尚无弦—徽位置联合 F1，不能将这些已有数值更名为“弦徽 F1”。

更新日期：2026-10-30。仅记录已验证数值。

## 1. 两阶段系统与诊断反馈比较

当前测试集包含48首、1031个phrase，包括替换版幽兰ST4GAejh的52段；三首有问题的原谱已排除。两组均采用Qwen3.5-9B两阶段LoRA权重，全部1031段协议有效。

| 指标 | 当前完整系统ON | 关闭诊断反馈OFF |
|---|---:|---:|
| PitchAcc@50c | 99.9810% | 86.2370% |
| 可评估音高事件 | 15,758 | 15,360 |
| 共同14,289音的PitchAcc | 99.9790% | 86.9410% |
| 音高MAE | 2.3081 cents | 64.9931 cents |
| 弦号F1 | 39.761% | 38.554% |
| 左手指法F1 | 57.119% | 56.599% |
| 右手指法F1 | 46.009% | 45.659% |
| 音色F1 | 62.543% | 61.670% |
| 装饰音F1 | 20.184% | 22.285% |
| 最终warning | 2 | 3436 |
| 含warning段数 | 2/1031 | 809/1031 |

ON启用完整系统的诊断反馈；OFF关闭warning、hint及告警驱动后续修订，保留结构验证和音高候选工具。两组使用相同的两阶段LoRA权重。评分均沿各首完整历史使用相同当前规则；推理封存运行时与评分代码的指纹差异已保留审计。

OFF实际公开回执及user/tool消息未检出诊断反馈泄漏，每段只有一个Guqinizer pass。OFF合计5382次生成轮次、1,565,947生成token。

ON残留两条音高warning：SxC3qbtd-p0004，以及幽兰ST4GAejh-p0047 source_index 1220（目标E7、实际E6）。F1衡量与原标注的一致性，不等于所有可演奏方案的质量；OFF装饰音F1略高，不能声称ON所有指标均更优。

证据：`evaluation_results/warning_ablation_current48_on/summary.json`、`evaluation_results/warning_ablation_current48_off/summary.json`；逐曲可视化见`train/eval_outputs/two_stage/index.html`。

## 2. 单阶段与两阶段比较

两组使用相同的 48 首、1031 个测试 phrase，按各曲完整历史使用相同规则评分；单阶段模型为独立训练的 Qwen3.5-9B LoRA。两组的 1031 段均协议有效。

| 指标 | 单阶段 | 两阶段完整系统 |
|---|---:|---:|
| PitchAcc@50c | 98.7324% | 99.9810% |
| 可评估音高事件 | 15,935 | 15,758 |
| 共同 14,580 音的 PitchAcc | 98.8066% | 99.9794% |
| 音高 MAE | 7.8798 cents | 2.3081 cents |
| 弦号 F1 | 42.610% | 39.761% |
| 左手指法 F1 | 57.917% | 57.119% |
| 右手指法 F1 | 47.697% | 46.009% |
| 音色 F1 | 65.629% | 62.543% |
| 装饰音 F1 | 24.622% | 20.184% |
| 最终 warning | 381 | 2 |
| 含 warning 段数 | 147/1031 | 2/1031 |

单阶段每次最多 16 轮，两阶段每次阶段最多 8 轮、Guqinizer 最多 3 次 pass；两阶段正式结果包含定向修订，不能解释为严格匹配预算的纯架构消融。音高可评估覆盖不同，因此同时给出共同可评估事件结果。F1 衡量与原标注的一致性，而不是所有可演奏方案的质量。

单阶段日志记录 13,954 次生成轮、2,944,380 个生成 token、14,470 次工具调用和 27 次生成截断；包含续跑前后的已记录调用，不应与经多次定向修订的两阶段累计成本直接作为公平速度比。证据：`evaluation_results/single_stage_current48/summary.json`、`evaluation_results/warning_ablation_current48_on/summary.json`。

## 3. 原始 Qwen3.5-9B 与两阶段 Agent

未加载项目 SFT 权重的 Qwen3.5-9B 使用两阶段 Agent，在同一 48 首、1031 段测试集上按曲谱顺序推理。若某段无法生成协议有效结果，停止该曲后续推理，不用空历史或原标注替代模型历史。

| 指标 | 原始 Qwen3.5-9B |
|---|---:|
| 实际尝试段数 | 59/1031（5.72%） |
| 尝试后协议有效 | 11/59（18.64%） |
| 尝试后协议失败 | 48/59（81.36%） |
| 有效输出覆盖率 | 11/1031（1.07%） |
| 前驱失败而跳过 | 972/1031（94.28%） |
| 全测试集音高与指法等音乐指标 | 未完成 |

已尝试的 59 段来自 48 首曲谱，输入哈希全部匹配；11/1031 是有效输出覆盖率，不是协议成功率。未尝试的 972 段不计入协议失败分母。原始模型只在有效输出上的音乐指标若后续计算，应明确标为条件性诊断，不与 SFT 模型的全量音乐指标直接比较。

48 个失败段中，13 个在 Fingering 阶段失败、35 个在 Guqinizer 阶段失败。按逐轮日志统计，45 段至少发生一次输出截断，26 段至少有一次尝试到达第 8 轮，31 段至少一次连续 3 轮调用音高候选查询；这些现象可能重叠，不是互斥的失败原因分类。

原始模型实验和两阶段 SFT 评估入口使用同一基础模型、两阶段运行框架及工具定义；请求输出上限均为 8192 token，每次阶段最多 8 轮、最多 3 次尝试，Guqinizer 最多 3 次 pass，上下文上限为 16384 token。原始模型不加载 LoRA；SFT 系统结果包含既有定向修订，故不视为严格配对的纯权重消融。正式 SFT 结果由多个运行批次构成，不能仅凭统一入口配置断言全部历史批次的聊天模板和其他运行细节逐项完全一致。

证据：`train/eval_outputs/base_two_stage_audit/run_summary.json`、`predictions.jsonl` 和 `workers/worker_*.log`。

## 固定教师轨迹构造消融

使用与单阶段 SFT 相同的 48 首、1031 段测试集和完整曲谱历史。消融模型以固定通用推理、一次真实音高候选查询和一次按原标注编辑构造的单阶段轨迹训练；推理时启用诊断反馈，每阶段最多 16 轮。两组均有 1031/1031 段协议有效。训练轨迹和部分 system/user 提示词存在差异，不能将比较解释为严格单变量因果消融。

| 指标 | 固定教师轨迹 | 单阶段 SFT |
|---|---:|---:|
| PitchAcc@50c | 26.7778% | 98.7324% |
| 可评估音高事件 | 13,556 | 15,935 |
| 共同 12,792 音的 PitchAcc | 26.9309% | 98.7805% |
| 音高 MAE | 517.6244 cents | 7.8798 cents |
| 弦号 F1 | 25.903% | 42.610% |
| 左手指法 F1 | 54.965% | 57.917% |
| 右手指法 F1 | 43.505% | 47.697% |
| 音色 F1 | 59.102% | 65.629% |
| 装饰音 F1 | 31.446% | 24.622% |
| 最终 warning | 11,596 | 381 |
| 含 warning 段数 | 988/1031 | 147/1031 |

固定轨迹组日志记录 11,426 次生成轮、2,913,821 个生成 token、12,225 次工具调用和 38 次生成截断；最终 warning 按完整曲谱历史重放统计，其中 9,926 条为音高不匹配。F1 衡量与原标注的一致性；固定轨迹组装饰音 F1 较高，不能声称单阶段 SFT 的所有指标均更优。证据：`evaluation_results/fixed_reference_ablation_current48/summary.json`、`evaluation_results/single_stage_current48/summary.json`。

## 4. ReAct + Handbook（GLM-5.3，prompt-only 基线）

同一 48 首、1031 段密封测试集上的强提示词基线：GLM-5.3（API）+ 生产种子提示词（`public_system_for("single_stage")`）+ 固定 73 条目技法手册，单阶段 agent 循环（最多 8 轮、最多 2 次尝试、temperature 0.25），工具与评分管线和 SFT 系统相同。1031 段全部协议有效、无重复 ID（首晚 S35eJqgV 一段协议失败中断该曲，续跑已补齐）。

| 指标 | ReAct + Handbook |
|---|---:|
| PitchAcc@50c | 99.7604% |
| 可评估音高事件 | 15,862 |
| 音高 MAE | 2.5422 cents |
| 弦号 F1 | 36.514% |
| 左手指法 F1 | 43.609% |
| 右手指法 F1 | 40.835% |
| 音色 F1 | 54.555% |
| 装饰音 F1 | 9.492% |
| 最终 warning | 102 |
| 含 warning 段数 | 52/1031 |

warning 构成：音高失配 37、泛音同弦余韵 29、低徽过紧 19、散音同弦再按 8+8、拇指低位 1。GLM-5.3 为远大于本地 9B 的闭源模型，本组数字回答的是"拥有相同工具与长尾技法知识时，强 prompt 基线能到什么水平"：音高接近上限（有音高候选工具兜底），但装饰音 F1（9.492%）低于两阶段 SFT（20.184%）和单阶段 SFT（24.622%），弦号与音色 F1 也全面低于 SFT 系统——支持"韵味与指法选择需要轨迹蒸馏而非更强提示词"的主张。该组不加载任何本地权重，与 SFT 各组不是同模型配对消融，跨组比较只作论文六臂表的行间对照。

复跑：`bash baseline/gepa_guqin/scripts/run_react_handbook_full.sh`（断点续跑，rollout 后自动事件级评估）。

证据：`baseline/gepa_guqin/runs/react_handbook_full_test50/`（预测含逐轮轨迹、启动配置、评估摘要快照，已入库）；事件级明细在本地 `evaluation_results/react_handbook_full_test50/`。

