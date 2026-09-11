# ABC→减字谱 Agent 训练与评估实验计划

状态：v1 方案冻结候选，数据口径更新至 2026-08-13。核心目标是构建可训练、可评分、可复现的
Agent 轨迹，而不是仅生成最终减字谱文本。

当前实现、数据规模、复现入口和后续交接详见
[agent_training_handoff.md](agent_training_handoff.md)。

当前 v2 产物状态：241 首整曲 `teacher.gqs` 已无损生成；由其直接产生 4,855 条 phrase
（train/validation/test = 3,374/488/993），最大 32 个有声音符。严格合格 train phrase
1,192 条中已生成 2,403 条职责严格的 messages，并通过上下文连续性和 messages 结构验证；
其中 Guqinizer 仅保留前置左右手完整的 238 条。

以下 4,490/1,132 等数字为 v1 对照：4,490/4,490 条反推轨迹已通过确定性 patch 回放。只有 train 被展开为
full trajectory、patch、direct generation 与曲级 tuning decision 四类训练视图，共 39,771 条记录；
validation/test 只保留 166/417 条标注评估配对并标记 `training_allowed: false`。审计未发现
family 跨分区、重复 ID、答案字段泄漏或非法七弦调弦。

### 调弦决策的正式定义

调弦不是调号到固定定弦的映射，而是先判断“是否值得改弦”、再选择替代方案的曲级决策：

1. 正调是默认路线；先检查目标调的主音、属音和高频重要音能否利用散音/泛音；
2. 估计曲目对散音的需求，并区分独奏与合奏：独奏更需要古琴承担“底”；
3. 只有替代调弦在散音、泛音和常用徽位上的收益超过改弦阈值时才离开正调；
4. 输出必须包含七弦实际 `open_midi`，不能仅输出名称；同名调弦可能有音高变体；
5. 标注谱采用的调弦属于示范/偏好监督，不应被表述为唯一正确分类标签。

曲级调弦训练数据只有 train 176 条；validation/test 的 21/44 首只用于指标评估。每首曲只生成
一条，避免长曲按 phrase 数被重复加权。

### Phrase 上下文交付口径

phrase 切分不能切断减字谱状态。当前统一采用可折叠序列：输入包含前一 phrase 的完整已完成
减字序列和当前 phrase，不提供后文预览；每个更早 phrase 各有一个 `context://...` 引用，
需要时通过 `expand_context` 单独展开。第一 phrase 只包含当前序列。序列是事实来源，结构化边界状态只作为
可重建缓存和审计结果。

section 是强切分信号，但不是无条件状态重置点。跨节保留必要尾部状态并显式标记边界；长休止、
明确终止以及新的显式手法才按规则覆盖或清空对应字段。训练时前段标注输出作为 teacher-forced
历史可见，但当前 phrase 的答案仍不可进入输入。

前文口径与暴露偏差（2026-08-18 决定）：教师轨迹生成时，前一段与 `expand_context` 返回的
是标注方案；学生推理时前文是其自生成方案。这是有意的 exposure bias 结构，需在 §8 的对比
实验中量化：同一 validation phrase 集，前文=标注方案 与 前文=自生成级联 两种设置下分别报告
硬约束指标与标注距离，差异作为级联误差的正式测量。

专项审计显示，4,490 条 phrase 中有 2,139 条首动作依赖前文继承，因此必须保留完整前段；
旧版 entry/exit 审计结果只作为迁移基线，后续应改为序列连续性与折叠引用审计。

## 1. 冻结的 Agent 框架

训练和推理的基本单位是 `phrase`，不是整首曲，也不是单音。每个 phrase 同时携带
前后各 2～4 个音的只读边界上下文。

```text
ABC + tuning
  → deterministic parser / phrase splitter
  → deterministic baseline route + advisory candidates
  → Fingering Agent
       ↔ get_pitch_candidates（参考，不是白名单）
       ↔ calculate_guqin_pitch
       ↔ score_transition
       ↔ audit_plan_patch
  → Guqinization Agent
       ↔ technique-context rules
       ↔ pitch / inheritance audit
  → deterministic phrase merge
  → compiler + hard audit
  → Critic / reward model（只评分，不绕过 hard gate）
  → best-of-N 或人工选择
```

### 1.1 职责边界

- Parser、调弦解释、音高计算、继承展开、编译和硬审计必须是确定性的。
- 候选生成器只提供搜索启发，不限制 Agent 提出候选外徽位。
- Route Search 与 Global Route Merger 确定弦徽路线；Fingering Agent 按曲序补全并优化左右手指法。
- Guqinization Agent 处理走手、散泛替换、吟猱绰注、掐起、撞及结构性技法。
- Critic 不直接改谱；它输出分项诊断和偏好，用于 best-of-N 与偏好训练。
- 全曲总控只负责调度、边界合并、回退和保存轨迹。

当前代码中的全曲 `fingering_optimizer` 保留作端到端基线；训练数据生成前应改为
phrase worker。`score_analyst` 和 `style_critic` 默认不应在每次推理中无条件调用：
前者的确定性信息先由 parser 提供，后者仅在多候选选择或评测时调用。

## 2. 轨迹数据格式

每个 JSONL 样本对应一个 phrase：

```json
{
  "trajectory_version": "1.0",
  "trajectory_id": "...",
  "score_family_id": "...",
  "score_key": "...",
  "phrase_id": "...",
  "split": "train|validation|test",
  "input": {
    "abc_events": [],
    "tuning": {},
    "left_context": [],
    "right_context": [],
    "baseline_plan": {},
    "reference_plan": {}
  },
  "steps": [
    {"type": "observation", "content": {}},
    {"type": "tool_call", "name": "...", "arguments": {}},
    {"type": "tool_result", "name": "...", "content": {}},
    {"type": "decision", "patch": {}, "evidence": []},
    {"type": "audit", "content": {}}
  ],
  "output_plan": {},
  "reference_plan": {},
  "rewards": {},
  "provenance": {
    "model": "...",
    "prompt_version": "...",
    "skill_versions": {},
    "dataset_version": "..."
  }
}
```

不保存 API key、请求头或隐藏思维链。保存可观察的模型输出、工具调用、工具结果、
patch、审计诊断和最终选择。训练目标是“可验证的决策轨迹”，不是让模型模仿未经
验证的自由散文。

## 3. 评估指标

不把所有维度过早压成一个总分。论文主表报告完整向量；只有在专家评分校准权重后，
才报告辅助综合分。

### 3.1 硬约束层

1. `PitchAcc@50c`：可判定攻击音中，音高误差不超过 50 cents 的比例。
2. `PitchMAE`：可判定音高对的平均绝对 cents 误差。
3. `PitchCoverage`：目标攻击音中成功展开并比较的比例。
4. `AttackCoverage`：目标攻击点是否恰好对应一个演奏动作。
5. `StructurePreservation`：section、phrase、休止、连音和攻击点边界保持率。
6. `CompileValidity`：减字结构闭合、字段合法且可回译的比例。

硬约束不通过时，样本不能靠“韵”得分补偿。

### 3.2 与标注谱的相似度

对齐到同一 ABC event 后分别计算，不要求整字完全相同：

- `Mode-F1`：散、泛、按取音方式；
- `StringAcc`：弦号一致率；
- `HuiAcc@0.2`：徽位误差不超过 0.2 徽；
- `RightHand-F1`：勾、挑、抹、剔、托、擘、打、摘、撮等；
- `LeftFinger-F1`：大、食、中、名、跪等；
- `Technique-F1`：吟、猱、绰、注、撞、逗、掐起、带起、进复、退复等；
- `ActionEditDistance`：结构化动作序列的归一化编辑距离。

参考谱不是唯一正确答案，因此这些指标衡量“接近标注版本”，不等同于绝对质量。

### 3.3 可演奏性与连贯性

- 左手物理移动距离及速度惩罚；
- 反向移动次数与短时大跳率；
- 右手连续同指、低效过弦和不自然组合率；
- phrase 边界入口/出口兼容性；
- 相对参考谱的 `TransitionCostRatio`；
- 规则非法动作率。

### 3.4 “韵”的可计算代理指标

“韵”拆成四层，禁止只用技法数量冒充韵味：

1. `Yun-Technique`：技法事件与参考谱的加权 Precision/Recall/F1；
2. `Yun-Context`：技法是否出现在合适的节奏、音程、长音、同音、句尾等上下文；
3. `Yun-Distribution`：生成谱的技法密度、散泛按比例、左右手序列、走手跨度等
   与训练语料分布的距离；应按曲体/长度/调弦分组计算，而非全库混为一谈；
4. `Yun-Expert`：琴家盲评或 A/B 偏好，是“韵”的主评价。建议维度为
   古琴语汇自然度、气韵连贯、装饰克制、音色组织、整体偏好。

`Yun-Distribution` 只能说明“像语料”，不能证明艺术质量；统计指标与专家评价必须
同时报告。后续可训练 style classifier/reward model，并验证其与专家排序的相关性。

### 3.5 辅助综合分

在专家校准前只作工程排序：

```text
HardPass = PitchAcc@50c × PitchCoverage × AttackCoverage × CompileValidity
SoftScore = 0.30 ReferenceAction
          + 0.25 Playability
          + 0.20 Yun-Technique
          + 0.15 Yun-Context
          + 0.10 Yun-Distribution
Overall = HardPass × SoftScore
```

正式论文权重应在 validation 集上依据专家偏好拟合，不能用 test 集调权重。

## 4. 当前数据盘点

截至 2026-08-19（以 `dataset_split_groups.csv` 为唯一清单；metric 基线已在 tonic 修复后
重跑，三处音高率源已对齐）：

- 清单包含 241 首、104 个 `leakage_group_id`；其中 round1 182 首、round2 59 首；
- 241 首均已有七弦调弦信息并可进入音高审计；
- 加权 `PitchAcc@50c` 为 77.84%；
- 加权 `PitchCoverage` 为 58.69%；
- 加权 `PitchJointSuccess` 为 45.68%；
- 单谱音高准确率中位数为 86.00%（233 首可评，8 首无音高率）；
- 每百个目标音平均包含 27.23 个技法标记；
- 已按 leakage group 划分为 train/validation/test：176/21/44 首，按目标音量占比
  69.91%/9.90%/20.19%。

这意味着不能随机按 score_key 切分：同名、同 ABC 指纹、同源版本必须归入同一个
`score_family_id`，否则训练/测试会严重泄漏。旧 pitch audit 也有大量不可判定事件，
所以必须同时报告 accuracy 与 coverage，不能只看命中率。

## 5. 两条 API 探索样本

两条样本只用于验证轨迹与指标，不参与最终 test：

1. 短谱：高音高覆盖、技法较少，用于验证硬约束和指法路径；
2. 中等长度谱：包含分节、走手、散泛切换和装饰，用于验证“韵”指标。

每条运行：reference、deterministic baseline、Agent best-of-N 三组；报告完整指标向量、
工具调用轨迹、被拒 patch 和成本。探索完成后把两首锁入 train/dev，禁止再进入 test。

## 6. 数据划分

先建立 `score_family_id`：标准化曲名 + ABC 旋律指纹 + 来源/版本关系。以 family 为组，
进行分层划分：建议 train/validation/test = 70/10/20。分层变量包括长度、调号、调弦、
音高覆盖、技法密度、散泛按比例和 section 数。专家盲评子集从 test 中预先抽取，
在模型训练和权重拟合期间保持封存。

## 7. 轨迹构造算法

### 7.0 当前反推实现（inverse-1.0）

截至 2026-08-12，已实现并全量运行：

```text
只读 ABC/简谱/调弦
  → stopped-only answer-blind baseline
  → 标注减字谱状态展开（显式字段/继承字段分离）
  → verified / weak / conflicting / unusable action 分级
  → baseline→reference typed minimal patch
  → 推定必要工具调用
  → action-level mask + canonical proxy trajectory
```

共生成 4,490 条 phrase 反推轨迹；训练集 3,122 条中，1,132 条满足当前严格
SFT 门槛。训练分区的全部轨迹中共有 62,569 个 verified patch；其中位于上述
1,132 条严格合格轨迹内、推荐进入第一版 SFT 的有 37,331 个。其构成为：
`REPOSITION` 11,755、`CHANGE_MODE` 9,263、`CHANGE_FINGER` 8,867、
`ADD_TECHNIQUE` 6,602、`NO_OP` 241。其余 verified patch 暂不应绕过 phrase 级质量
门槛直接进入完整轨迹训练；weak patch 只作上下文或低权重辅助，不作为默认强监督。
该轨迹是“可验证的最小代理路径”，不声称恢复标注者真实思维过程。

三个 split 的严格合格规模如下：

| split | 全部 phrase | 严格 SFT 合格 | 合格轨迹内 verified patch |
|---|---:|---:|---:|
| train | 3,122 | 1,132 | 37,331 |
| validation | 441 | 166 | 5,464 |
| test | 927 | 417 | 11,239 |
| 合计 | 4,490 | 1,715 | 54,034 |

validation/test 中的合格轨迹只用于调参与评估，禁止并入训练。

### 7.1 参考轨迹

从标注减字谱反解 `reference_plan`，逐 phrase 对齐 ABC event；为每个参考动作生成：

- 当时可见上下文；
- 候选位置与参考位置（参考可在候选外）；
- 音高和转移成本验证；
- 与 baseline 的最小 patch；
- 技法上下文标签；
- 最终审计与奖励。

规则可唯一解释的步骤生成确定性轨迹；存在多解时保留为弱监督，不伪造唯一理由。

### 7.2 探索与反事实轨迹

对每个 phrase 生成 N 个候选轨迹，经 hard gate 后，用自动分项指标和专家/critic 排序。
保存 chosen/rejected 对，并保留“为什么拒绝”的审计诊断。优先收集能改变决策的工具
调用，不鼓励无意义地查询所有候选。

### 7.3 与当前 Agent 框架对齐（messages v3）

- 训练阶段统一使用 `fingering_agent`、`guqinization`、`plan_auditor`，不再混用
  `fingering_backbone`／`fingering_optimizer`；
- Route Search 与 Global Route Merger 是确定性算法；Fingering Agent 负责左右手，
  Guqinizer 负责走手、音色变化和装饰；
- 一个谱面事件是最小局部决策单元。同一事件的模式、弦徽和左右手字段原子提交，不把每个
  字段伪装成一次独立推理；
- 多事件仍按谱序形成多轮 messages，并在末尾确定性重放审计；在线调用默认最多 24 轮，
  可由 `AGENT_MAX_TOOL_ROUNDS` 调整；
- v3 仍是从标注反推并经确定性重放验证的 teacher-forced 轨迹。下一层数据应让教师模型在
  私有答案指导下决定何时查询候选、核验音高或展开前文，再只保留可重放、无答案泄漏的轨迹。

该下一层生成器现已实现为 `scripts/generate_teacher_tool_trajectories.py`。教师侧采用单一
JSON 信封仅表达 `tool_calls`；通过审计的 `edit_plan` 由运行器直接提交并结束阶段，随后确定性转换为 Qwen3.5 原生
`assistant.tool_calls → tool` messages。批量生成时只写入通过者；失败原因和每次失败的真实
教师输入输出分别记录在生成报告与 `teacher_rejected_io.jsonl`。Guqinizer 样本还必须先通过
“Fingering Agent 输出无待定左右手”的前置检查。

新版 `--basic-intermediate` 不要求 Fingering 与标注指法一致。Fingering Agent 从仅保留音高、
节奏和攻击点的空白动作开始，自主查询候选并选择散按泛、弦徽和左右手；随后 Guqinizer 分步
缩小与标注的字段／技法距离。音乐上不同但
合法的基础指法不淘汰，只保留非空覆盖、工具真实性、音高、完整性、可重放和私有信息隔离等
技术门槛。中间体独立缓存，可从失败的 Guqinizer 阶段续跑。

## 8. 训练与测试顺序

1. 实现 patch 回放器，确认从 baseline 顺序应用 patch 后能还原目标状态，并重新执行
   音高、结构、攻击点和编译硬审计；
2. 实现统一导出器，从同一源轨迹生成三种数据视图：完整 Agent 轨迹、verified
   动作级样本、直接生成辅助任务；
3. 动作级 SFT：训练 schema、模式切换、弦徽换位、指法和技法添加；
4. 完整轨迹 SFT：训练工具选择、多步 patch 和最终审核；直接生成作为辅助任务；
5. preference optimization：使用同一 phrase 的 chosen/rejected 反事实轨迹；
6. 可选 reward model：只有专家偏好量足够且一致性达标后再训练；
7. best-of-N + deterministic verifier 作为强基线；
8. 冻结 test 上报告 hard、reference、playability、Yun 与专家盲评；
9. 消融：无工具、无候选、无 critic、单音/phrase/全曲上下文、无轨迹训练；
10. 前文口径对比（暴露偏差）：validation 上对比 前文=标注方案 与 前文=自生成级联 两种
    推理设置，报告硬约束与标注距离的差值，作为级联误差的正式测量。

数据口径更新（2026-08-18）：全部训练轨迹改由 MiniMax 教师管线批量生成（含曲级调弦决策，
由 `scripts/generate_teacher_tuning_decisions.py` 产出）；反推轨迹的动作级 SFT 退出训练，
仅保留为评估基线与消融对照。教师轨迹接受口径同步放宽：音高超差与攻击点／技法类违规记为
警告级诊断，Guqinizer 距离要求为不上升（允许持平）。

当前严格训练集导出 1,132 条完整轨迹、37,331 条动作级记录和 1,132 条
直接生成记录，即 39,595 条主任务物理记录；另有 176 条曲级调弦训练记录。三种主视图共享同一批 1,132 个独立 phrase，
不能把物理记录数误报为独立样本数，也不能在展开后重新随机划分。

两条 MiniMax API 探索样本只承担运行时冒烟测试，不是批量训练数据构造的前置条件，
不应阻塞回放器和导出器的实现。

任何训练前必须保存 dataset manifest、split manifest、prompt/skill 版本和随机种子。

## 9. 论文参考文献与用途

### [R1] StyleRank：符号音乐风格相似度

Jeff Ens and Philippe Pasquier. **Quantifying Musical Style: Ranking Symbolic
Music Based on Similarity to a Style.** Proceedings of the 20th International
Society for Music Information Retrieval Conference (ISMIR), pp. 870–877, 2019.

- 论文：[ISMIR 公开 PDF](https://archives.ismir.net/ismir2019/paper/000107.pdf)
- 预印本：[arXiv:2003.06226](https://arxiv.org/abs/2003.06226)
- 本项目用途：支持 `Yun-Distribution` / style similarity 的设计。可借鉴其
  “以一个语料集合定义风格、按多组可解释特征计算相似度并与人类判断校验”的思路。
  本项目不能直接照搬 MIDI 特征，而应换成古琴专属的散泛按比例、弦徽路线、左右手
  序列、走手与装饰上下文等特征。

建议 BibTeX：

```bibtex
@inproceedings{ens2019stylerank,
  title     = {Quantifying Musical Style: Ranking Symbolic Music Based on Similarity to a Style},
  author    = {Ens, Jeff and Pasquier, Philippe},
  booktitle = {Proceedings of the 20th International Society for Music Information Retrieval Conference},
  pages     = {870--877},
  year      = {2019}
}
```

### [R2] FMD：生成符号音乐的分布距离

Jan Retkowski, Jakub Stępniak, and Mateusz Modrzejewski. **Frechet Music
Distance: A Metric for Generative Symbolic Music Evaluation.** arXiv:2412.07948,
2024；后续收录于第 39 届 AAAI Conference on Artificial Intelligence 论文集。

- 论文：[arXiv:2412.07948](https://arxiv.org/abs/2412.07948)
- 官方实现：[frechet-music-distance](https://github.com/jryban/frechet-music-distance)
- 本项目用途：支持“不要只逐谱比较参考答案，还要比较生成集合和真实集合的整体分布”
  这一评估层。后续可先用 ABC/MIDI embedding 计算通用 FMD，再研究古琴动作序列
  embedding；FMD 不能代替音高硬审计、逐事件技法 F1 或琴家盲评。

建议 BibTeX（最终投稿前需从正式 AAAI 论文集补齐卷号和页码）：

```bibtex
@article{retkowski2024fmd,
  title   = {Frechet Music Distance: A Metric for Generative Symbolic Music Evaluation},
  author  = {Retkowski, Jan and St{\k{e}}pniak, Jakub and Modrzejewski, Mateusz},
  journal = {arXiv preprint arXiv:2412.07948},
  year    = {2024}
}
```

### 引用边界

- [R1] 用于论证“风格相似度可由语料定义并用可解释特征估计”；不能据此声称自动指标
  已经等同古琴审美判断。
- [R2] 用于论证“生成集合与参考集合应做分布级比较”；不能据此把 FMD 当作单谱质量分。
- 本项目关于“韵”的核心贡献仍应是古琴专属技法/上下文指标及琴家偏好实验，而不是
  简单复用通用符号音乐指标。
