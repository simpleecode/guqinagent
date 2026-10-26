# 古琴 Agent 训练交接：当前有效状态

更新：2026-10-16（北京时间）。仅维护当前状态；历史修复细节看报告、代码和测试，不再追加操作流水账。

正式推理产物统一入口（2026-10-22整理）：`train/eval_outputs/<方法>/<曲谱ID>/`。
主实验方法为 `two_stage`，每首固定保留 `inputs.jsonl`、`predictions.jsonl`、
`result_manifest.json`、`<曲谱ID>.html` 和可选 `<曲谱ID>.pdf`；总览为方法目录的
`index.html`。同方法更新时替换该曲当前结果，旧正式目录移入可恢复废纸篓；
不得继续新建带日期/重试后缀的正式目录。共享队列工作文件使用
`train/eval_outputs/two_stage/.work/inference/`，完成后由
`train/scripts/publish_method_eval.py` 校验并分曲发布。J00327已改为此布局。
旧 `train/eval_outputs_v3_two_stage/` 中的历史实验不再作为当前入口，不批量删除未核对的实验凭据。

## 1. 当前训练

- 实验室：`20223393ljw@219.216.65.119`；仓库 `/home/20223393ljw/guqin-agent`。
- **J00271**：3 GPU（当前0/1/2）、3 epoch、有效 batch size **12 = 每卡1 × 梯度累积4 × 3卡**。新标签版数据；先完成预检/CPU分词并确认新任务入队，再取消旧J00270；旧PGID341312已无残留。已确认实际完成step 1/1989（启动核查快照），三卡各约21.6GB/100%利用率。J00267已成功完成。
- Qwen3.5-9B；4bit bitsandbytes LoRA、target=all；bf16、Liger、梯度检查点；LR 2e-4、cosine、warmup 0.03、cutoff 8192；每epoch保存、保留3个checkpoint。从基座新训，不恢复旧optimizer。
- 配置：`train/configs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12.yaml`；入口：`train/scripts/run_latest_runtime_train_3gpu.sh`。
- 远端数据快照：`/home/20223393ljw/guqin-agent/train/data/guqin_agent_sft_repeat_marker_g/`；含预先在CPU生成的7952条tokenized缓存，加载前核对tokenized_manifest与数据SHA；没有改模板或监督mask。50项相关测试、训练格式与test50隔离/指纹校验通过。
- 远端输出：`/home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016/`；队列日志 `/home/20223393ljw/code/training/runtime/gpu_queue/logs/J00271.log`。旧权重和输出未覆盖。
- **训练环境guqin-sft；推理环境guqin-vllm-019；cip仅运行jobber，不是训练环境。** 旧J00264已取消，J00266加载时因其他用户占卡OOM，勿重复提交。

```bash
ssh 20223393ljw@219.216.65.119 'tail -f /home/20223393ljw/jianzi/outputs/qwen35_9b_lora_repeat_marker_g_3gpu_bs12_20261016/train.log'
ssh 20223393ljw@219.216.65.119 '/home/20223393ljw/miniconda3/envs/cip/bin/python /home/20223393ljw/code/training/jobber.py status'
```

jobber running/分配卡号不等于锁住显存。耗时预检应提前做；不可变数据可按SHA复用已通过预检。必须确认模型加载和实际step，不能只报入队。换任务先入队再停旧任务；不动其他用户进程，不覆盖正在训练的数据快照。

## 2. 唯一有效数据位置

本地根目录 `/Volumes/F/work/Gunqin-agent`。下表路径相对 `ABC_J/agent_training/two_stage_audited/`。

| 产物 | 路径 | 数量/用途 |
|---|---|---|
| 筛选前phrase | `inferred_all/inferred_trajectories_{train,test}.jsonl` | train4950、test999；重建/重新筛选的上游 |
| 筛选后train | `pitch_eligible_train_test/inferred_trajectories_train.jsonl` | **3976 phrase、153首；教师输入** |
| test | `pitch_eligible_train_test/inferred_trajectories_test.jsonl` | **999 phrase、50首；不筛选** |
| 筛选说明 | `pitch_eligible_train_test/selection_manifest.json` | 规则与准入清单 |
| 原始教师对话 | `teacher_latest_runtime_raw/messages_train.jsonl` | 未脱敏，7952条阶段轨迹 |
| 私有审计 | `teacher_latest_runtime_raw/teacher_trajectory_audit.jsonl` | 请求/响应、工具执行、最终方案；禁止直接训练 |
| **最终脱敏教师** | **`teacher_train_final/messages_train.jsonl`** | **3976 phrase × 两阶段 = 7952条** |
| 最终校验 | `teacher_train_final/messages_validation_report.json` | valid=true，两阶段各3976，错误0 |
| 全量可视化 | `full_training_visualization/visualization_sharded/index.html` | 153曲谱分片；尚未刷新本次114phrase替换，暂属上一版 |

[可视化索引](/Volumes/F/work/Gunqin-agent/ABC_J/agent_training/two_stage_audited/full_training_visualization/visualization_sharded/index.html)

实际SFT在仓库的 `train/data/guqin_agent_sft_two_stage_latest_runtime/guqin_agent_train.jsonl`，配套metadata、dataset_info、manifest和校验报告；不是直接训练inferred文件。

正式评估输入：`train/eval_inputs_v2_text_protocol/test50/test_runtime.jsonl`；密封标注同目录 `evaluation_pairs_test.jsonl`，指纹看 `export_manifest.json`。划分清单 `ABC_J/results/agent_dataset_splits.csv`；无validation。训练/测试按曲组隔离，测试50首各取一个版本；不要混回旧测试集或旧pitch-match-70 benchmark。

## 3. 源数据与教师数据的区别

- inferred每行一个phrase，保留ID、split、metadata、normalized_tuning、无减字音符、handoff、baseline_plan、reference_plan和来源。
- reference_plan含原谱标注，供教师参考、筛选与评估；**不能把完整inferred直接发给评估模型**。
- 源baseline_plan的算法弦徽不作Fingering答案：先清空选择，只借音序建立空白计划；Guqinizer接收实际Fingering结果。运行时同名字段是“本阶段初稿”，仍需要。
- 上下游四个源JSONL已删除旧顶层canonical_trajectory、patches、unresolved_differences、quality；其余字段逐条一致，4975个筛选后phrase空白计划一致、203首公开prompt抽查一致。
- `scripts/infer_agent_trajectories.py` CLI默认精简输出；历史算法补丁导出器需要 `--include-legacy-inference-fields`。不要误删运行时新产生的patches/quality。
- 教师messages每行是一个阶段的完整实际对话，不是一个音；公开脱敏对话与私有原始审计分开保存。

## 4. 现行生成流程

12phrase及最后一条重试已完成：先将 `duplicate_string_last_retry/` 成功的S45fs5uq-p0044两阶段覆盖到12phrase候选，再GLM脱敏、整集合并和SFT校验；凭据 `duplicate_string_retry/completion_report.json`。重新扫描整份7952条实际edit_plan及全部最终方案，重复弦号均为0。已同步为J00270新训练快照；旧训练和已完成评估未覆盖。本次脱敏临时副本已移外置盘废纸篓。

教师生成现有定向自动重试：共享 `agents/abc_to_jianzipu/jianzi_surface_checks.py` 检测重复弦号，`generate_with_retries` 在阶段返回验收时检查实际edit_plan行及最终方案（中间轮错了后面改好也拒绝，避免写入错误示范）；不扫描用户prompt/assistant解释文字，不增加提示词或强制user消息。命中只重跑该阶段，沿用max_attempts=3，超限记录failure，不接受错误结果。规则与定向筛选共用；未改推理runtime warning。相关27项unittest通过。

最近局部修复已完成：`two_stage_audited/duplicate_string_repair/`，114phrase/72首、228条，两阶段全部成功；GLM-5.3脱敏、整集消息和SFT校验通过，已替换canonical raw/final/SFT，保持3976pair。两阶段最终仍含重复弦号的phrase由113降到12，非全部清除；比较见retry_comparison.json。未加新提示词或验收约束。完成凭据completion_report.json，总日志redaction_merge.log；旧正式版、本次重造raw及脱敏worker已移到外置盘废纸篓，仅留报告日志。远端训练快照与评估未改，仍对应上一版数据。

1. 从 `ABC_J/final/` 当前谱源解析并重建phrase，按当前划分筛选；禁止恢复旧inferred_gqs_v12等淘汰源。
2. 只筛train：每阶段至少2个匹配可解析音高，或匹配数至少占可解析数一半；另排除普通前置减字后连续异常空音。详细规则以selection_manifest和筛选代码为准；test不筛。
3. Fingering从空白起步，Guqinizer基于其结果古琴化；真实执行工具，不伪造成功回执。
4. 教师GLM-5.3；实测配置7worker、0.5s间隔、24工具轮、3次阶段尝试，优先整曲均衡分片，不自行增加并发。
5. 429/临时网络错误重试同一请求，保留前轮内存状态、不插强制user指令；未实现逐轮磁盘恢复，中断后未完成阶段仍需重跑。
6. 默认允许教师摘要引用私有参考、包含Guqinizer no-op；随后**必须GLM改写脱敏**，不以正则替换冒充完成。
7. warning由真实工具回执告知模型，模型自行决定修改或结束；不得因仍有warning就伪写“全部通过”，也不自动判全部失败。

prompt顺序：**只读相关段 → 只读前一段 → 当前段**。相关段仅检索同曲严格更早历史，连续匹配至少9个音高，取最早有效匹配；候选整段或匹配范围内超过半数减字为空则跳过。前一段不重复插入，无匹配不留空标题。教师与学生历史来源须明确，不偷偷以标注替换学生预测历史。

公开音序用event_index，结构行不占可编辑音序；source_index是内部映射，不能混用。`edit_plan.jianzi_rows`直接写减字文本，允许上下文继承和省略，不强迫每音写满字段。

## 5. 共享审计及验收口径

- 共享runtime：`agents/ToolRuntime/runtime.py`；音高审计：`scripts/audit_jianpu_jianzi_pitch.py`；prompt：`agents/abc_to_jianzipu/teacher_trajectory.py`。工具、筛选、可视化应共用状态规则，不另起一套泛音判定。
- 泛起/泛止、撮多弦、左手位置继承、带起及复/进/退都会改变状态；末尾泛止在本音发声后生效。同弦同徽泛音按实际继承徽位豁免余韵warning，不依赖右手技法枚举。
- 泛音分数徽位提示：`泛音只能在徽位上弹出（请使用整数徽位，或先泛止并改用按音）`。泛音区间走手需审计；走手跨度阈值501 cents。拂类音高/连续技法余韵豁免按当前代码及测试，不擅自扩展。
- 再作标签现规范为 `[再作标记ㄱ]`；解析兼容旧名称，提取器:k及新生成内容用新名。提示词说明标记的是可重复乐句的起始位置，通常附乐句首音，不是再作指令，不能临时放在紧邻指令的休止/空行凑数。支持从头、从ㄱ、从数字音序再作；数字再作按实际省略长度匹配。**从ㄱ再作仍缺引用内容/空区间审计，改名和提示词不代表此漏洞已修好。**
- 单个/连续省略首行检查前一个有效减字是否有再作指令，后续连续行不重复报警；更早的起点标记不能代替执行指令。省略用专用文本，不留空。
- 反复和小节结构不得当休止，保留 `|:`、`||:`、`:||`、`:||[2` 等表示；不能用休止时值猜结构。
- 最终warning统计只看最后接受方案，不累加中间轮；区分“最终Guqinizer”与“两阶段任一”、存档回执与当前运行时重审。**目前没有合并后的最新全量warning数字，不能声称0。**
- J00271新标签快照已重新做SFT预检：超过8192为56/7952（0.704%），p50=4975、p95=6828、p99=7936、最大11344。cutoff8192会截断，不表示删除样本。报告在远端新数据目录preflight_report.json。
- 可视化优先显示保存的真实公开请求，缺失时才明确标记重建；warning按正确音序对齐，重放保留曲谱上下文，大集按曲目分片。

## 6. 常用代码入口

| 入口（相对仓库） | 用途 |
|---|---|
| `scripts/infer_agent_trajectories.py` | phrase源重建 |
| `ABC_J/scripts/export_pitch_eligible_trajectories.py` | train筛选及规范调弦校验 |
| `ABC_J/scripts/run_teacher_batch_parallel.py` | 并行及按ID局部重跑 |
| `ABC_J/scripts/generate_teacher_tool_trajectories.py` | 教师真实工具轨迹 |
| `ABC_J/scripts/redact_teacher_reasoning.py` | GLM脱敏 |
| `scripts/validate_teacher_agent_messages.py` | 最终教师消息校验 |
| `train/scripts/export_sft_dataset.py`、`validate_sft_export.py` | SFT导出/校验 |
| `train/scripts/verify_test50_training_inputs.py` | 划分、防泄漏和指纹校验 |
| `train/scripts/preflight_sft.py` | 实际模板token预检 |
| `train/scripts/eval_two_stage_score.py`、`run_vllm_eval_queue.py` | 评估推理/多卡调度 |
| `ABC_J/scripts/visualize_all_trajectories_hierarchical_sharded.py` | 教师分片可视化 |

本地Python：`/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python`。

```bash
# 仓库根目录执行
/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python scripts/validate_teacher_agent_messages.py --input-dir ABC_J/agent_training/two_stage_audited/teacher_train_final
/opt/homebrew/Caskroom/miniforge/base/envs/guqin-agent/bin/python train/scripts/validate_sft_export.py --input-dir train/data/guqin_agent_sft_two_stage_latest_runtime
```

局部修复按sample_id替换完整两阶段pair；“只追加未见ID”不能替换旧结果。缺失/失败需明确报告，脱敏及校验完成再提升正式集。审计较大，要流式/物理分片处理，不能7worker各加载全量审计压满内存。

新评估启动前核对代码同步、公开测试集SHA、调弦名/七弦MIDI、adapter、GPU分片和Guqinizer上限（当前多轮方案通常总计最多3次，以启动参数为准）；每卡所属worker全完后释放模型实例。

前一轮评估 **J00268已succeeded**，依赖J00267，2GPU：梅花三弄S35eJqgV（66phrase）、凤求凰S1zt293d（6phrase），来自test50，共72phrase。入口 `train/scripts/run_meihua_fengqiuhuang_final_eval.sh` 仍固定旧adapter，不能直接当J00270评估脚本。输入 `train/eval_inputs_v2_text_protocol/meihua_fengqiuhuang_latest/test_runtime.jsonl`，密封标注单独存同目录，不传模型。公开输入SHA256 `e4ffe3f041fa8224326b1fdc7419af053ba0829aad0d8cf87343d18b47c97ef7`。

J00268使用J00267输出根目录的最终adapter，启动前强制epoch>=2.99、global_step=max_steps且最终文件非空。两阶段、Guqinizer最多3次、8192输出/16384上下文、8工具轮、8worker/2server。旧评估输出 `/home/20223393ljw/guqin-agent/train/eval_outputs_v3_two_stage/related_context_final_meihua_fengqiuhuang_2gpu/`，日志eval.log；队列日志 `/home/20223393ljw/code/training/runtime/gpu_queue/logs/J00268.log`。远端已完成，本地可视化仍可能是旧部分快照，刷新需重新拉取。

## 7. 指纹、清理和后续边界

- 本地新标签最终教师messages SHA256：`231b8933772ab8f207d814018a2b9d50e348350bde5f704148ea0a43ee1b6f36`；本地SFT SHA256：`f6dfc07c3c6efff228b5c4f85a19e3e5b89f559fa58ab6f1eccd551f185e0bab`。
- J00271已用本地f6dfc07c开头新标签快照重训，代码和test50规范化输入已同步。旧J00270（773149cc快照）已取消；旧J00267/J00268用03d06bae快照，不要混同。未追加新评估任务。
- 标签迁移凭据 `repeat_anchor_label_migration_report.json`：249个文件改名，其中公开教师4174/7952行变更；整集消息、SFT、test50防泄漏及指纹校验通过。处理当前final/modern派生谱面、四份phrase源、test50标注和正式脱敏消息/SFT，ID/音序/音高/划分未重建。原始采集、teacher_latest_runtime_raw、私有原始请求/审计、历史预测和单阶段历史实验未改，旧标签继续兼容；因此最终公开消息是名称规范化派生版，不可冒称字节级原请求。旧文件在外置盘废纸篓，新语义提示没有回填成旧教师真的看过的请求；可视化及token统计待刷新。
- 方法目录 `cleanup_manifest.json` 保存已清理目标及历史完成报告；旧修复/worker/过时可视化已入废纸篓，不得再从历史worker resume。
- `phrase_source_compaction_report.json` 保存精简校验。源文件字节SHA已变，旧报告的源SHA属于历史快照，不和新文件机械比对。
- 保留筛选前上游、当前筛选源、canonical raw、最终脱敏、SFT及最新可视化，因用途不同不是冗余版本；临时副本合并后清理，只维护一版正式结果。
- 外置盘内容移至 `/Volumes/F/.Trashes/501/`，不要复制几个GB到系统盘废纸篓；不主动清空废纸篓。
- 单阶段目录 `ABC_J/agent_training/single_stage_audited/` 含历史 `teacher_final_redacted/` 等产物，**尚未确认与当前3976phrase/test50和最新runtime对齐，不能直接当当前对照训练集**。
- 后续只更新对应章节；不再追加旧任务流水账、失效路径或旧数据“唯一有效”声明。任何同步/重训不覆盖正在使用的远端快照。
