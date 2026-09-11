# ABC → 减字谱 Agent：交付文档

更新时间：2026-08-28  
当前状态：**训练源数据已冻结**（`inferred_v6`，2026-08-28，冻结声明见 §7.67）；
`inferred_v6` 数据与教师管线就绪；40 条分层 pilot 已完成，最终干净集合为
`messages_pilot_v44_final`（Fingering 40＋Guqinizer 38＝78 样本）。v4 修复了复合写法“泛止”
未结束泛音作用域的全库污染；`inferred_v3` 及基于 v3 且未通过 v4 重审的古琴化样本均为历史产物，
不得进入训练。v5 在 v4 基础上新增多声复合动作的原子语义；v6 又补齐“独立续音绰”和
“拨弦前置绰”的结构差异、动作物化与重放审计。新教师生成必须使用 v6。v44 pilot 抽样中没有
这些新增动作，因此原78条仍有效，但不能作为 v5/v6 新语义的质量证据。

本文是当前工程状态的唯一交接入口。框架原则见
[abc_to_jianzipu_agent_framework.md](abc_to_jianzipu_agent_framework.md)，训练与评估方案见
[agent_training_experiment_plan.md](agent_training_experiment_plan.md)。

## 0. 2026-09-11 最新数据状态

- 当前完整两阶段教师数据为
  `ABC_J/agent_training/messages_glm_full_two_stage_equalized_blacklist_filtered_20260911/`。
  共 7,806 条：Fingering 3,903 条、Guqinizer 3,903 条；两阶段 phrase ID 集合完全一致，
  公开／私有审计 ID 一一对应。
- `SQquA21B`（《梅花三弄》）已加入 `ABC_J/results/score_blacklist.json`。该曲泛音段的
  ABC／简谱目标与标注减字呈系统性一八度偏差，`SQquA21B-p0008` 的音高警告多数固定为
  `+1200 cents`，会诱使 Guqinizer 反复追逐无法通过局部编辑消除的警告。因此整首 24 个
  phrase 均已从当前训练源剔除，两阶段合计删除 48 条；后续选择、抽取、训练与评估均应按
  score key 整首排除。
- 全量 Guqinizer 相对上一版完整 3,927 条数据的标注命中统计（剔除该曲前）：下降 160、
  上升 676、持平 3,091；总命中率 `32544/46819`（69.51%）→ `34117/46808`（72.89%）。
- 当前层级可视化脚本已移动至
  `ABC_J/scripts/visualize_all_trajectories_hierarchical.py`；根目录 `scripts/` 下不再保留副本。
- 注意：当前合并数据继承了 `SzQNSg3J-p0022-guqinization-teacher-tools` 的公开 reasoning
  遗留私有来源措辞，协议校验会报 3 个同源词项；在重新导出最终 SFT 前必须对该条重新脱敏
  或替换，不能把当前目录直接宣称为最终可训练导出。

## 1. 当前目标

目标是训练一个将 ABC notation 转为古琴减字谱的工具型 Agent。训练对象是可观察、可执行、
可审核的工具调用与局部编辑，不包含隐藏思维链。

```text
ABC／简谱 + 分节 + 调弦
  → Fingering Agent：从空白动作选择散／按／泛、弦徽、左右手
  → Guqinizer：加入走手、装饰、吟猱及更贴近标注的处理
                 并以原子动作保留掐撮／掐拨剌／掐拂歷／历拂
  → edit_plan：真实重放、预览与硬审计
  → 通过即直接提交，编译为减字谱
```

一个训练样本对应一个 phrase；输入保留当前段、完整前一段和按需展开的更早段。调弦是曲级
显式输入：候选生成和音高核验始终使用该样本的 `normalized_tuning.open_midi`，而不是根据
调号临时猜测。

## 2. 已冻结的工作流

### 2.1 Fingering Agent

- 从仅保留音高、时值、攻击点的空白动作开始；不继承 Route Search 的散按泛、弦或徽位。
- 先自主调用 `get_pitch_candidates`；可用 `类型=泛音｜散音｜按音` 限定候选。逐音全查还是只查
  不确定的音由教师决定，但本阶段至少要成功查询一次候选。
- 选择散／按／泛、琴弦、徽位、左右手，并用 `edit_plan` 提交局部 patch。
- 硬约束（不满足即拒绝）：动作覆盖不变、攻击动作的弦徽与左右手完整、patch 可确定性重放、
  攻击点不变（改 attack 可绕过右手指法要求，故保持硬约束）。基础阶段添加技法为警告级诊断，
  交由 Guqinizer 距离审计兜底。
- 音高采用分级审计：普通散／按／泛起音和结构完整的双弦动作可直接计算，基础阶段超过
  ±50 音分为硬错误；走手续音、绰注撞逗、进退复、掐起带起等不能可靠简化为单一 MIDI 的
  动作只记 `pitch_not_directly_auditable`，不因工具无法证明而拒绝。报告必须同时给出
  `directly_checked`、`not_directly_auditable`、`target_unavailable` 和 `mismatches`，不得把
  “可计算子集 100%”表述成“全部技法 100%”。

### 2.2 Guqinizer

- 输入为已通过 Fingering 审计的基础方案。
- 可调用候选、音高核验和前文展开工具；负责走手、音色变化、装饰及向标注靠近的局部修改。
- 不要求机械复刻标注，但与可信标注的字段／技法距离不得上升；允许持平——已达标或无法
  改进时，一次通过全部硬审计的确认性提交即可结束。同时保持可重放与左右手完整。
- 古琴化不得用 weak／inherited 参考把原本 ±50 音分内的普通可计算音改坏；verified 人工位置
  可高于标量工具，复杂技法仍走语义／上下文审计。`撮／泼／剌` 在两个阶段都要求 `string2`
  伙伴弦，缺失即硬拒绝。
- 左手技法续音是续音：右手不拨弦。独立式标注（单独一行“猱”／“撞”／“进复七徽”／
  “带起”等，语料约 3,300 个）全部 attack=false 无右手，Guqinizer 用完整 patch 将
  attack 改为 false（清空右手）并以 ADD_TECHNIQUE 加技法；合法技法集为左手技法全集
  （绰注上下淌吟猱往来进复退复撞逗带起掐起抓起），谱面渲染如 [绰上七徽九分]、
  [猱十徽]、[进复七徽]。复合式（“大指十徽八分剔1弦吟猱”）保持 attack=true 与右手，
  谱面减字缀技法串。无技法的去攻击（attack_removed_without_slide_technique）被拒；
  基础阶段禁止改 attack（fingering_changed_attack 为硬约束，防止借续音绕过右手指法要求）。
- “绰”有两种结构，不能只作为无位置的 technique 后缀处理：`绰上五徽` 是独立续音，必须
  `CHANGE_ATTACK` 为 false、清空右手，再 `ADD_TECHNIQUE`；`绰大指五徽六分挑6弦` 是一次
  带前置动作的真实拨弦，必须保持 attack=true 与“挑”，并以
  `ADD_TECHNIQUE.after.placement=pre_attack` 记录。内部字段
  `pre_attack_techniques=["绰"]` 使渲染稳定为 `[绰大指五徽六分挑6弦]`，而不是错误的
  `[大指五徽六分挑6弦绰]`。可信参考显式断言 attack，错误地“只加绰但仍保留拨弦”会被
  `reference_attack_mismatch` 硬拒绝；前置绰误放到拨弦主体之后则由
  `reference_pre_attack_technique_mismatch` 硬拒绝。

### 2.3 上下文

- 默认提供：当前 phrase、完整前一 phrase、曲名与规范调弦。
- 更早段不塞入主 prompt；先 `list_context`，再一次只 `expand_context` 一段。
- 前文是只读已确认方案，section 及继承状态由其自然保留。
- 已知暴露偏差口径：教师轨迹的前文取自标注方案；学生推理时的前文是自生成方案。两者差异
  通过对比实验量化（见实验计划 §8），不在此处静默。

## 3. 教师生成与 Qwen 训练协议

MiniMax 教师模型看到私有参考标注用于规划；公开训练数据绝不包含参考动作、答案字段或隐藏
推理。教师每轮只生成一个 JSON 工具信封：

```json
{
  "decision_summary": "可选的简短、公开可核查理由",
  "tool_calls": [
    {"name": "get_pitch_candidates", "arguments": {"source_indices": [647, 648]}}
  ]
}
```

运行器真实执行工具，并确定性转换为 Qwen3.5 的原生训练消息：

```text
assistant.tool_calls → tool
```

`edit_plan` 预览一旦通过全部阶段审计，即为提交并终止本阶段；接受要求**当前批次重放有效**
（畸形批次不得借“空累计＝距离持平”的确认路径蒙混）。教师不再生成 `final`。因此公开
训练 messages 可以结束于最后一批 tool 结果，样本标记：

```json
{"termination": {"kind": "accepted_edit_plan"}}
```

这避免了重复提交、额外 API 回合，以及 `final` 与预览不一致的问题。

## 4. 工具接口

| 工具 | 作用 | 关键输入／输出 |
|---|---|---|
| `get_pitch_candidates` | 为目标音列举散／按／泛候选 | 输入 `source_index` 或 `source_indices`，可选 `类型` 与 `max_candidates`（默认：批量 8、单音 12）；多音和弦（撮等）逐个音给出候选表，标签用各音 ABC 音名（如 `和弦音1（C,）`）；也可直接以 `target_midi` 指定任意音高查询（不依赖 index）。输出可读候选表：方式、弦徽、实得 MIDI、音分误差、可信度；音分差相同时散音优先。它不决定左右手。 |
| `calculate_guqin_pitch` | 核验一个指定位置的音高 | `source_index`、`mode`、`string`、可选 `hui`；多音和弦默认与最高音比较，可用 `target_midi` 指定比较目标。 |
| `edit_plan` | 应用 patch 的真实预览 | 返回中文变化表、谱面减字、可应用性与错误；基础指法阶段还会返回待定左右手等硬错误。多批次提交时预览与审计针对**累计** patch 状态（后续批次可修正前批取值），因此每次预览都反映当前完整方案。紧凑行第 7 位支持双弦技法（撮／泼／剌）的伙伴弦：整数＝散音伙伴弦（语料主流），对象＝`{string, mode, hui}` 显式伙伴位。独立续音绰使用 `CHANGE_ATTACK`＋`ADD_TECHNIQUE`；拨弦前置绰的 ADD patch 必须带 `placement=pre_attack`。若 Route 无基础动作，ADD patch 可携带完整结构快照并物化该行。 |
| `list_context` | 查看可展开的更早段 | 不返回谱面动作。 |
| `expand_context` | 展开一个更早 phrase | 只读已确认方案。 |

`get_pitch_candidates` 的公开返回只保留可读候选表；私有审计文件保留完整候选对象。候选位置是
专业判断的证据，不是硬白名单。

## 5. 再作物化与减字显示约定

再作类标记在 gqs 的 `jianzi` 字段内，语料共 894 处（再作 283、再作起点 296、从ㄱ再作 233、
从头再作 23、再作二声 45、再作三声 14），覆盖 122/241 首。inferred 生成以
`--materialize-repeats` 展开为演奏现实，两类情况：

1. **跨度重复**（音符带真实减字＋重复指令，如“大指九徽挑5弦从ㄱ再作”）：重复段未写出，
   物化为真实音符副本。跨度＝起点（`再作起点`；无起点则取最近边界：休止／小节线／段起／
   上一标记）到标记音（含）；`从头再作` 从曲首；二声／三声展开两／三份。
2. **标记省略音**（音符在音高流中、jianzi 仅为“再作”）：谱面以再作替代其减字，直接标记。

所有省略音携带 `notation_omitted=True`：Agent 仍须填写完整指法（散按泛、弦徽、左右手均过
硬审计），但谱面减字固定渲染占位 `[无（由于是再作部分，省略）]`——prompt 表格、edit_plan
预览与编译器三处一致。省略音不携带参考标注（继承展开生成的 attack=false 空续音引用已剥
离），距离审计对其不计分。

物化后全音符序号重排，v2 数据不可与 v3 混用。当前数据状态（2026-08-20 重分节）：phrase 6,798
（train/validation/test = 4,739/654/1,405）；引用类计数与 v2
完全一致（verified 54,739／conflicting 15,581／unusable 31,115），weak 51,264→50,745 的差异即剥离的
虚假续音。

分节规则（`--max-sounding 16`，冻结于 2026-08-20）：自然分节边界（`<一>` 等）永远优先，绝不
跨分节；段内发音音超过 16 个后，截到**下一条**小节线（不回退，完整携带溢出小节）；无小节线
的段落（潇湘水云等 36 首源数据无线）在 24 个发音音处强制截断兜底。发音音分布：中位 19、
p95 24、max 30。

源抽取器 [extract_jianpu_jianzi.py](../scripts/extract_jianpu_jianzi.py) 对 `zhyx` 的可读顺序为：

```text
左手指 + 徽位 + 右手指 + 弦
名指五徽勾3弦
```

工具预览与编译器已同步该顺序；预览额外显示“谱面减字”列，例如 `[大指四徽勾3弦]`（再作
省略音为占位）。内部结构保持独立字段，避免靠文本解析编辑：主位
`mode/string/hui/left_finger/right_finger`，双弦技法（撮／泼／剌）的伙伴位
`string2/mode2/hui2`，以及拨弦主体之前的 `pre_attack_techniques`。`mode` 与 `mode2` 共用同一
值域（内部英文，界面一律转中文）：

| 内部值 | 中文 | 语义 |
|---|---|---|
| `open` | 散音 | 空弦直接拨 |
| `stopped` | 按音 | 左手按弦取音（与 open string 相对的西方弦乐术语） |
| `harmonic` | 泛音 | 轻触徽位泛音点 |

候选查询的 `类型` 参数（泛音｜散音｜按音）在运行器内映射到同一值域。表格的简谱列对
和弦音显示完整双谱字（如 `4̣̣ 1̣`）——第二个谱字本就存储在 `jianpu_alt` 中（全库 4,199 个
和弦音符 100% 双值），直接拼用；非和弦音符的 `jianpu_alt` 是装饰性替代音，不参与显示。和弦
的音高目标同样以存储双谱字解析为准（记谱真相）。ABC 列已于 2026-08-21 修复并与简谱全库
一致：`jianpu_abc_pitch` 改为绝对 MIDI 锚定（复用 AUDIT 的度1表，含 "1=B"=Bb3 与
`tonic_degree1_midi` 覆盖）、显式拼写变音记号（`_B`/`^F`）、消费 decoder 的 accidental
（♯4 等简谱变音）；241 首已全量重抽并级联重建 gqs 与 v3，主音 83,151 个零偏差，
phrase ID 与 pilot 清单不变，音高匹配率（>40% 共 223 首）不受影响。

## 6. 关键代码与产物

| 位置 | 作用 |
|---|---|
| `scripts/generate_teacher_tool_trajectories.py` | MiniMax 教师轨迹生成、真实工具运行、提交与私有审计；支持 `--shard-count/--shard-index` 分片并行。 |
| `scripts/run_teacher_generation_parallel.py` | 并行分片启动器：派生 N 个生成进程、汇总日志并合并分片产物与报告。 |
| `scripts/generate_teacher_tuning_decisions.py` | 曲级调弦决策的教师生成：真实工具（调弦目录／资源核验／提交审计），标注调弦仅作私有示范监督。 |
| `scripts/sample_pilot_phrases.py` | pilot 分层抽样：上下文角色 × 长度/技法密度分箱，输出 `pilot_ids.txt`。 |
| `agents/abc_to_jianzipu/teacher_trajectory.py` | Phrase 的可读公开 prompt 与上下文渲染。 |
| `agents/abc_to_jianzipu/trajectory_replay.py` | patch 确定性重放。 |
| `agents/abc_to_jianzipu/compiler.py` | 方案编译为可读减字谱。 |
| `scripts/validate_teacher_agent_messages.py` | 公开／私有隔离、工具配对、终止协议校验。 |
| `agents/abc_to_jianzipu/qwen35_serialization.py` | 使用官方 Qwen3.5 chat template 序列化训练 messages。 |
| `scripts/visualize_agent_trajectories.py` | 公开训练轨迹与 Qwen 序列化可视化。 |
| `scripts/visualize_teacher_io.py` | 私有 MiniMax 原始请求／响应可视化；含阶段导航。 |
| `agents/abc_to_jianzipu/repeat_materializer.py` | 再作物化：标记解析、重复段展开、省略标记与占位约定。 |
| `scripts/infer_agent_trajectories.py` | gqs → inferred 轨迹；`--materialize-repeats` 展开再作。 |
| `ABC_J/agent_training/inferred_v6/inferred_trajectories_train.jsonl` | 当前唯一可用于新教师生成的 train phrase 输入（再作、泛止、多声原子动作、两类绰语义及全部减字代码映射已修复）。**已于 2026-08-28 冻结**（见 §7.67），也是生成器默认输入。 |

生成目录的固定文件约定：

| 文件 | 是否可用于训练 | 内容 |
|---|---|---|
| `messages_train.jsonl` | 是 | Qwen 原生工具消息，未含私有答案。 |
| `teacher_trajectory_audit.jsonl` | 否 | 私有标注目标、完整工具执行日志、教师实际 I/O。 |
| `teacher_rejected_io.jsonl` | 否 | 被拒尝试的真实 I/O，用于调试。 |
| `fingering_intermediates.jsonl` | 否 | Fingering 通过后的中间方案，供 Guqinizer 续跑。 |
| `generation_report.json` | 否 | 通过／拒绝数量和失败摘要。 |

## 7. 已完成的真实验证

### 7.1 2026-08-22：40 条分层 pilot（当前准入基线）

最终训练目录：`ABC_J/agent_training/messages_pilot_v44_final/`。

- 40/40 Fingering 通过；39 条存在 v4 古琴化目标，其中 38 条通过（97.4%），`ShnpF8Qg-p0036`
  因弱继承撮缺伙伴弦且会引入近半音偏差而剔除；`SumLbkVi-p0014` 本来无古琴化改进目标；
- 最终 78 个训练样本，公开／私有 ID 一一对应，协议验证 `valid=true`，v4 内容重审 0 失败；
- Fingering 音高：直接核验 733，复杂／不支持动作 3，目标不可解析 0，分歧 0；
- Guqinizer 音高：直接核验 595，复杂／上下文动作 104，目标不可解析 0，分歧 0；两阶段合计
  107 次“不直接核验”均明确记录，未冒充音高通过；
- Guqinizer 参考距离总计 926 → 195，改善 731；accepted patch 中位 21.5，工具调用中位 6，
  edit_plan 调用中位 2；
- `[无N弦…]` 0，非散音“弦在徽前”0；所有 verification 字段为 true；全套 48 项测试通过；
- 可复现报告：`generation_report.json`、`messages_validation_report.json`、`pilot_analysis.json`；查看器：
  `trajectory_viewer_qwen35.html`、`teacher_io_viewer.html`。

pilot 首轮用 4 workers 触发 MiniMax Token Plan 429，出现 23 个阶段失败；改为单线程并按
Fingering 中间体续跑后基本恢复。最终 `historical_rejected_attempts=101` 是所有重试过程的教师
I/O 审计条数，不是最终样本失败数。生成器现已加入 429 指数退避＋随机抖动及逐 phrase flush；
但新的安全并发尚未实测，当前套餐仍按 `workers=1` 运行。

本轮发现并修复两个此前会污染训练的缺陷：

1. 参考解析器只识别独立一行 `泛止`，不识别常见的 `3弦<br>泛止`，且休止重置 context 时没有
   同步重置局部 harmonic 状态；导致后续普通按／散音被批量伪标为泛音。v4 相比 v3：
   `CHANGE_MODE` 30,378 → 16,007，verified 54,739 → 56,711，conflicting 15,581 → 14,671；
2. Guqinizer 审计漏检 `撮／泼／剌` 的 `string2`；现已与 Fingering 一致作硬约束。

`messages_pilot_v32`～`v43` 是诊断／恢复过程目录，不得直接拼入训练；只有
`messages_pilot_v44_final/messages_train.jsonl` 是本轮准入文件。

### 7.2 2026-08-23：多声复合动作修复（历史 v5）

- 新增独立字段 `compound_gesture` 与 patch `SET_COMPOUND_GESTURE`。`掐撮一/二/三声`、
  `掐拨剌一/二/三声`、`掐拂歷二/三声`、`历拂`不再被拆成普通`撮／剌`子串，也不再退化为
  `attack=false` 空动作；动作名是显式强语义证据（verified），但音高按
  `context_dependent_compound_gesture` 跳过单值硬判；
- 普通`撮／泼／剌`仍严格要求 `string2`。原子复合动作不套该规则；2个“普通撮弦位＋掐撮三声”
  混合文本会分别保留普通双弦位置和原子复合名；全库原子动作子串泄漏为0；
- v5 共6798条 phrase（train 4739／validation 654／test 1405），确定性重放6798/6798通过；
  共267个复合动作参考、涉及259个 phrase，其中215个直接形成 verified
  `SET_COMPOUND_GESTURE` patch。另51个因旧 Route 基线为空暂列 unresolved；批量教师管线的
  `--basic-intermediate` 会先从零补全基础动作，再从全部 verified/weak 参考重新推导 Guqinizer
  target，因此这些动作不会在实际教师生成阶段继续丢失；
- 当时回归测试87/87通过（agent 54＋scripts 33）。该版本现只作历史对照；新批次使用 v6。

### 7.3 2026-08-23：两类“绰”结构修复（当前 v6）

- `.gqs` 对 `绰上五徽` 与 `绰大指五徽六分挑6弦` 原本即可无损往返；本次补齐的是结构解析、
  `edit_plan` patch、确定性重放、音高审计和最终减字渲染，不是修改 gqs 文法；
- 独立绰显式 `attack=false/right_finger=null`，前置绰显式
  `attack=true/pre_attack_techniques=["绰"]`。前置绰属于上下文相关动作，不强行通过不可靠的
  单值音高工具；这只豁免标量音高判定，不豁免弦、徽、左右手、attack、技法位置与重放审计；
- v6 共6798条 phrase（train 4739／validation 654／test 1405），格式与曲级防泄漏校验通过，
  确定性重放6798/6798通过。`绰上五徽` 共295处，295/295 均作为 weak 结构监督、精确渲染且
  全部 attack=false；旧分类中18处 conflicting 的根因只是简谱寄存器／八度约定与继承弦五徽的
  单值换算相差102～2402音分，并非“绰”结构冲突，现不再整条屏蔽。所有独立走手仍更新目标
  徽位与演奏上下文，但不以单一 MIDI 决定其 reference 是否可监督。`绰大指五徽六分挑6弦`
  共13处，13/13 verified、精确渲染、attack=true 且位置为 pre_attack；
- 修复了 Route 无基础动作时技法行只生成空壳的问题：ADD patch 现在可物化完整独立／前置动作。
  同时补齐重放器遗漏的 `string2/mode2/hui2` 状态，防止无基础动作的撮／泼／剌静默丢伙伴弦；
- 当前回归测试98/98通过（agent 63＋scripts 35）。`inferred_v5`及更早版本只保留历史对照；
  新批次默认已切换到`inferred_v6`。

### 7.4 历史验证

最新批次：`ABC_J/agent_training/messages_prompt_review_v23/`（2026-08-21，10 条 pilot 头部
phrase，全部协议修复就位后的产出）：

- 通过：10/10 phrase（Fingering 10＋Guqinizer 10＝20 样本），拒绝 0（含此前连续超轮次
  失败的 SFzXxqnV-p0069）；
- 终止一致性：末条 edit_plan 预览全部“可应用”（0 例报错收尾）；
- 双弦撮 4 个提交全部带伙伴弦；变音 ABC 拼写（`_B`/`_E` 等）152 处；prompt 双谱字正常；
- 教师轮次 1～7（中位 2）；token 均值：单阶段样本约 22.5K 上下文（新输入 8.9K＋缓存读
  12.9K＋输出 0.6K），单 phrase 两阶段约 45K；
- 公共／私有样本 ID 一一对应，工具调用配对、重放与泄漏检查通过；
- Qwen3.5 官方 chat template 已验证支持以最后一批 tool 结果结束的消息序列。

批次历程（每轮都修出并固化了问题）：

| 批次 | 结果 | 修出的问题 |
|---|---|---|
| v20 | 4/10（v2 旧数据，存档 `_v2basis`） | max_tokens 截断 → 紧凑行（信封省 79%）；生成器默认输入误指 v2 |
| v21 | 19/20 | JSON 截断根修＋失败反馈重试；多批次预览误报（累计校验）；空累计蒙混漏洞 |
| v22 | 19/20 | 撮双弦支持落地；走手续音协议缺口（attack 语义）确诊 |
| v23 | 20/20 | 左手技法全集＋复合式渲染；ABC 全库修复后的首批干净产出 |

调弦轨迹：3/3 通过（正调，决策与标注逐弦零音分差，工具链 get_tuning_catalog →
check_scale_resources → submit_tuning）；改弦曲的区分度待 176 条全量。

可视化入口（最新批次）：

- `ABC_J/agent_training/messages_prompt_review_v23/trajectory_viewer_qwen35.html`
- `ABC_J/agent_training/messages_prompt_review_v23/teacher_io_viewer.html`

### 7.5 2026-08-24：v6 复杂指法 MiniMax 单条真实样例

- 输入：`inferred_v6` train 中的 `S0FKGjFt-p0011`（《樵歌》），包含原子复合动作 `掐撮三声`，用于专项检查 v5/v6 新协议。
- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax/`。
- MiniMax-M3 两阶段均完成：Fingering 1 条、Guqinizer 1 条，共 2 条公开训练样本；拒绝 0、最终失败 0。
- 协议校验：`valid=true`，公开/私有 sample ID 一一对应，真实工具调用、确定性重放、目标字段审计、私有答案防泄漏全部通过。
- 最终音高审计：Fingering 直接核验 14、不可直接核验 5、分歧 0；Guqinizer 直接核验 11、不可直接核验 8、分歧 0。13 个“不直接核验”来自复合/不支持的右手动作、续音/结构动作及 `context_dependent_compound_gesture`，均明确记录，未伪装为音高通过。
- Guqinizer 与参考的结构距离从 13 降到 4；`SET_COMPOUND_GESTURE` 在 source 259 保留完整 `掐撮三声`，未拆成普通撮弦，也未退化成空动作。
- 可视化：`trajectory_viewer_qwen35.html`（公开 SFT 视图）与 `teacher_io_viewer.html`（私有目标/调用审计视图）。

### 7.6 2026-08-24：教师参考改为紧凑 GQS 视图

- 教师模型的基础指法阶段不再接收完整 `reference_actions` 对象数组；当前格式是真正的纯谱面 GQS 片段，只保留 `index / jianpu / jianzi`。可信等级、`explicit_fields/inherited_fields` 仅供内部筛选与审计，不向教师展示。
- 仅将 `verified`、`weak` 且具有实际音乐信息的参考行送给教师；`unusable/conflicting`、空参考行、evidence、diagnostics 和大量 null 字段不进入模型上下文。
- 完整 `reference_actions` 仍原样保存在 `teacher_trajectory_audit.jsonl`，并继续供确定性目标审计、距离计算和重放使用；此次改动只改变教师可见提示，不降低机器硬约束。
- 在 `S0FKGjFt-p0011` 上，教师参考从 24 个完整对象、9827 字符缩为 8 行、215 字符，字符量减少 97.8%。GQS 作为系统提示中的独立多行文本追加，不再嵌套为 JSON 字符串，因此模型不会看到 `\\n`、转义引号或 `reference_gqs` 包装。
- `generate_basic_direct` 备用路径同步切换到紧凑 GQS，避免主路径与回退路径提示不一致。
- 第一版紧凑 smoke 位于 `ABC_J/agent_training/messages_complex_sample_v6_minimax_gqs/`，虽然 2/2 接受且审计通过，但它使用了已废弃的“GQS 外壳＋字段掩码＋嵌套 JSON”混合格式，只能作诊断对照，**不得进入训练**。当时测得的轮次/token 改善不能代替当前纯 GQS 的线上验证。
- 纯 GQS 格式的真实 MiniMax smoke 已输出到 `ABC_J/agent_training/messages_complex_sample_v6_minimax_pure_gqs/`：同一 `S0FKGjFt-p0011` 的 Fingering/Guqinizer 2/2 接受、拒绝 0，协议与重审 `valid=true`，最终音高分歧 0，`掐撮三声` 在 source 259 完整保留。教师系统中不存在 `reference_gqs` 包装，GQS 片段不含可信等级或字段掩码。
- 该纯 GQS 样例总教师轮次 10、工具调用 14、无效 edit 5；输入 token 23341、缓存读取 61849、输出 2451。Guqinizer 参考距离 15→8（改善 7），低于已废弃混合格式 smoke 的 14→4，因此“格式可用”已验证，但单样本质量波动仍需在下一轮受控 pilot 统计，不能仅凭此例认定纯 GQS 一定提高终稿贴合度。
- 当前回归测试 103/103（agent 66、scripts 37）。

### 7.7 2026-08-24：轨迹页默认显示最终版与标注版对比

- 新生成的私有 `teacher_trajectory_audit.jsonl` 增加 phrase 级 `annotation_gqs`，完整保留每个序号的 `index / jianpu / jianzi`；Guqinizer 阶段也保存确定性重放后的完整 `accepted_plan`。这些字段不进入公开 `messages_train.jsonl`。
- `visualize_agent_trajectories.py` 默认自动读取 `messages_train.jsonl` 同目录的 `teacher_trajectory_audit.jsonl`，在每个样本顶部显示“最终版本 vs 标注版本”逐序号表格，列为序号、简谱、最终减字、标注减字及一致/不同状态；差异高亮但不自动判错，因为合理替代指法也可能不同。
- 可用 `--audit` 显式指定其他审计文件。现有旧批次若没有 `annotation_gqs/accepted_plan`，页面保持兼容但不显示对比；下一次真实生成起自动具备。
- 当前回归测试 103/103（agent 66、scripts 37）。

### 7.8 2026-08-24：空标注行与“删除动作”语义

- 教师私有 GQS 现在列出当前 phrase 的全部序号，包括 `jianzi=""` 的行；`S0FKGjFt-p0011` 因此会明确显示 247–249、252、255、257–258、260–268 等空标注，而不是从序号表中消失。Fingering 与 Guqinizer 两阶段都接收这份完整 GQS。
- 提示明确规定：空 `jianzi` 表示没有可靠直接标注，不等于休止或删除声音；不得仅因空白清空可演奏字段。此规则避免把“缺标注”错误训练成“删音”。
- 当前 patch 协议可将个别字段设为 null、移除 technique 或把起音改为续音，但没有删除整个 action 的正式语义。发现并修复 Guqinizer 审计漏洞：过去把 `string` 清成 null 会跳过音高审计；现在缺 mode/string/hui/left_finger 等可演奏字段会硬失败。
- 如果后续确认某些空行是由 `掐撮三声` 等复合动作覆盖，正确建模应是保留音乐事件并显式设置谱面省略/复合动作覆盖范围，而不是删除 action 或清空弦徽。该 span/notation-omitted 语义需依据标注约定另行实现，不能仅凭空字符串自动推断。
- 当前回归测试 104/104（agent 67、scripts 37）。
- 最新完整空行 smoke：`ABC_J/agent_training/messages_complex_sample_v6_minimax_full_gqs/`。同一 `S0FKGjFt-p0011` 两阶段 2/2 接受、拒绝 0，协议/重放 `valid=true`，最终音高分歧 0；Fingering 与 Guqinizer 私有系统均包含 24 行完整 GQS、无可信等级或字段掩码，source 259 的 `掐撮三声` 完整保留。轨迹页已生成两个“最终版 vs 标注版”对比区。
- 本轮 Fingering 3 轮/3 工具/1 次无效 edit；Guqinizer 8 轮/10 工具/6 次无效 edit，参考距离 16→9。无效提交主要是 `attack_removed_without_slide_technique` 与会破坏原正确音高的改位，不是清空 string 绕审计。说明完整空行格式正确且防删音生效，但单例工具效率偏低，后续 pilot 应统计空标注是否诱发不必要修改。

### 7.9 2026-08-24：候选取音按实际音高自动去重

- `get_pitch_candidates` 继续以 `source_index/source_indices` 为主要输入；模型不负责把简谱、调号、八度和变音自行换算成 MIDI。批量查询由工具解析每个事件及和弦音，再按实际 `target_midi` 分组。
- 每个唯一音高只计算并输出一份候选表，同时返回 `source_indices` 与逐来源标签；同音高的不同序号仍可根据上下文选择不同指法。和弦序号可进入多个音高组。`target_midi` 输入继续保留，仅用于脱离谱面事件的特殊查询。
- Fingering 提示已从“为每个起音调用”改为优先一次传入批量 `source_indices`；工具描述同步说明自动去重。确定性完成查询的审计仍按原始调用参数覆盖全部 source index，不因分组而丢失事件。
- 在 `S0FKGjFt-p0011` 的 19 个起音上，实际归并为 6 个 MIDI 组 `[59,57,55,62,60,48]`，组大小 `[5,5,7,1,1,2]`；候选公开文本从模拟旧格式 5334 字符降为 1621 字符，减少 69.6%。
- 当前回归测试 105/105（agent 68、scripts 37）。

### 7.10 2026-08-24：`滔起` 续音解析与批次报错解释

- `edit_plan` 返回 `ok=true, valid=false` 时，表示工具执行成功、但整批累计方案不可应用；错误按 `source_index` 定位，不能把同批 251/254 的错误误认为 259 的 `SET_COMPOUND_GESTURE` 失败。`S0FKGjFt-p0011` 的 source 259 始终正确渲染为 `[掐撮三声]`。
- source 254 的 `guqinization_introduced_pitch_mismatch` 可靠：简谱目标 MIDI 60，改成散音4弦为 MIDI 55，较原正确主干低 500 音分；原文 `至4弦` 只明确弦号，不能把继承的散音模式当成可靠目标。
- source 251 暴露真实词表缺口：抽取器早已把 `tq:` 定义为左手动作 `滔起`，但参考解析器、续音合法技法表、单值音高豁免和渲染未同步。现已全链路加入 `滔起`；`名指十徽滔起` 解析为 `attack=false + techniques=["滔起"] + 名指/十徽`，diff 同时生成 `CHANGE_ATTACK` 与 `ADD_TECHNIQUE`，最终精确渲染 `[名指十徽滔起]`。没有放宽 `attack_removed_without_slide_technique` 硬审计。
- 已从 `teacher.gqs` 重新生成当前 `inferred_v6`：6798 条（train 4739、validation 654、test 1405），新增后总 patch 统计含 `ADD_TECHNIQUE=31707`；确定性重放 6798/6798 通过。
- 当前回归测试 107/107（agent 70、scripts 37）。

### 7.11 2026-08-24：候选音高去重后的真实 MiniMax 单条验证

- 使用当前 `inferred_v6` 与 MiniMax-M3 重新生成复杂样例 `S0FKGjFt-p0011`，产物位于 `ABC_J/agent_training/messages_complex_sample_v6_minimax_dedup/`；Fingering 与 Guqinizer 共 2/2 accepted、0 rejected，消息协议、公开/私有隔离与确定性重放均通过。
- Fingering 对 19 个起音/和弦目标分两次批量查询，工具按实际 MIDI 自动合并为 6 个音高组；首批 11 个序号合并为 5 组，后批 8 个序号（含和弦分音）合并为 3 组，其中 MIDI 57 与 55 已跨序号复用候选表。该调用证明去重接口在真实教师轨迹中生效，且没有丢失来源序号。
- 最终音高审计：Fingering 直接检查 14、不可直接检查 5、分歧 0；Guqinizer 直接检查 13、不可直接检查 6、分歧 0。不可直接检查项均按复合动作或续音原因显式记录，没有伪装成通过。
- Guqinizer 参考距离从 12 降到 3；与标注文字完全一致的关键行包括 `[注下七徽九分]`、`[注下九徽]`、`[名指十徽滔起]`、`[散摘7弦]`、`[掐撮三声]`。source 254 保持音高正确的 `[散挑6弦]`，没有为贴合弱参考“至4弦”而制造 -500 音分错误。
- 本轮 Fingering 10 次工具调用、Guqinizer 5 次工具调用；可视化 `trajectory_viewer_qwen35.html` 已包含“最终版本 vs 标注版本”，私有教师输入输出见 `teacher_io_viewer.html`。

### 7.12 2026-08-24：独立“吟”不重复继承徽位

- “吟”有两类谱面结构：与本次拨弦合写时保持 `attack=true`，渲染为 `[勾5弦吟]` 等完整拨弦式；独立成行、承接上一音时为 `attack=false`，只渲染 `[吟]`。后一类内部仍可继承上一音的 mode/string/hui/left_finger 供上下文与演奏状态使用，但这些继承字段不得再次显示成 `[吟七徽九分]`。
- `jianzi_renderer` 与教师轨迹辅助渲染已同步该规则；`[注下七徽九分][吟]` 现在保持两个独立减字，attack=true 的后缀吟仍保持原样。
- 新增两种形态的回归测试，当前 agent 71、scripts 37，共 108/108 通过。

### 7.13 2026-08-24：最终减字改由 Agent 直接填写

- GQS `teacher-gqs-1.1` 使用可空 `jianzi_text`：`null` 表示没有可直接监督的最终文字，非空字符串是目标减字，`""` 表示动作继续演奏但该行不显示减字。原先短暂加入的 `jianzi_hidden/SET_JIANZI_HIDDEN` 已废弃，不得再生成。
- 最终谱面不再由代码依据弦、徽、左右手和技法拼接。动作的 `jianzi_text=null` 时只显示 `[减字待填写]`；Agent 必须通过 `SET_JIANZI_TEXT` 设置最终文字，Guqinizer 尚有任一 null 时由 `pending_jianzi_text` 硬拒绝。空字符串是唯一明确的无显示值。
- `edit_plan` 预览同时展示完整结构化动作与减字填写状态（待填写／Agent 已填写／空且动作保留）。结构化字段继续独立接受音高、左右手、弦徽和复合动作审计，填写文字不能绕过这些约束。
- 对明确带“声”的多声复合减字，GQS 构建器把其后连续、空标注且可演奏的音的 `jianzi_text` 设为 `""`；遇到延音、休止、分段或下一条减字即停止。`S0FKGjFt` 中 259 为 `掐撮三声`，260–266 为空，267–268 不受影响。
- 全库生成 `SET_JIANZI_TEXT=60,455`，其中 verified 监督 38,218；`NO_OP` 已消除。`inferred_v6` 重建后 6798/6798 确定性重放通过，当前测试 109/109（agent 72、scripts 37）。

### 7.14 2026-08-24：精简 `edit_plan.jianzi_rows` 与真实 MiniMax 验证

- `edit_plan` 新增定长批量输入 `jianzi_rows=[[source_index,jianzi_text],...]`；运行器自动展开为 `SET_JIANZI_TEXT`，补齐 patch 类型、before 与 ID。空字符串合法，表示动作保留但该行不显示减字。复杂结构编辑仍使用 `patches`，两种输入可在一次调用中并存。
- `jianzi_rows` 会拒绝错误行长、错误类型与同批重复序号；跨批重复提交若与当前累计状态相同则自动忽略，避免 LLM 重发整段时制造冗余监督。当前测试 110/110（agent 73、scripts 37）。
- 前两次真实调用均在 Fingering 阶段因 MiniMax 非法 JSON／耗尽轮次失败，失败目录分别为 `messages_complex_sample_v6_minimax_agent_jianzi/` 与 `..._retry1/`，不得进入训练。第三次 `..._retry2/` 成功：2/2 accepted、0 rejected，协议与重审通过，最终音高分歧 0，参考距离 17→3。
- 成功样例中 Guqinizer 最终 19 个动作全部具有非 null `jianzi_text`；246=`吟`，259=`掐撮三声`，260–266=`""`。该次模型曾把同一批 19 行重复提交三次，产物保留真实调用历史；其后加入的幂等过滤会在新轨迹中消除这些重复，不回写篡改旧调用。

### 7.15 2026-08-24：`edit_plan` Schema 收紧审计

- 真实调用证明上一版更新不完整：`patches` 与 `jianzi_rows` 同时允许文本编辑，`after` 可带任意字段，导致模型重复提交 `SET_JIANZI_TEXT`、发明 `slide=true`、把整串“注下七徽九分”当 technique，并把普通“撮”误设为 compound gesture。
- 当前公开 Schema 已改为单一路径：最终文字只能通过 `jianzi_rows`；`patches` 的 enum 不再暴露 `SET_JIANZI_TEXT/NO_OP`。完整 patch 的 `after` 使用字段白名单与 `additionalProperties=false`，technique 直接使用合法 token enum。
- 运行时再做语义白名单：ADD_TECHNIQUE 必须属于合法左手技法表；SET_COMPOUND_GESTURE 必须完整匹配复合动作词法，普通“撮”不能通过。未定义字段由 patch-target 检查在当轮立即拒绝，且无效批次不进入累计状态。
- verified 参考的非 null `jianzi_text` 现在是硬约束；总体距离改善不能再掩盖关键减字写错。当前测试 110/110。
- `messages_complex_sample_v6_minimax_agent_jianzi_idempotent/` 虽通过旧审计，但仍含双路径重复；`..._compact_retry2/` 通过旧宽松质量门槛但实际技法错误。两者均只作失败诊断，**不得进入训练**。Schema 收紧后的真实 MiniMax 样例尚待重新生成。

### 7.16 2026-08-24：移除教师可编辑的 `technique`

- `technique`、`placement`、`ADD_TECHNIQUE`、`REMOVE_TECHNIQUE` 已从公开 `edit_plan` Schema 和提示词移除。教师只需在 `jianzi_rows` 写最终减字；工具从非空 `jianzi_text` 自动派生内部技法及其前置关系。
- 空字符串仍只表示“该行动作存在但谱面不显示”，不会清除其演奏状态。内部技法字段继续用于音高豁免、上下文和标注审计，但不再让 LLM 重复编辑同一事实。
- 弦、徽、散按泛、左右手仍影响可演奏性或音高，不能删除。结构化复合动作修正和 `CHANGE_ATTACK` 暂时保留作无充分文字依据时的纠错入口；有最终减字时优先从文字推导。
- 兼容性边界：写 `jianzi_text` 不会暗改 `attack`、`right_finger` 或 `compound_gesture`，避免破坏旧补丁顺序；这些演奏结构仍须显式编辑。当前测试 113/113（agent 76、scripts 37），全库导出与重放 6798/6798 通过、rejected=0。
- 教师当前谱面与 `edit_plan` 预览不再单列“技法”；原“动作／左手／右手”合并为一列“演奏状态”。信息仍供可演奏性和音高判断使用，但表格从 9 列缩至 6 列，避免最终减字与派生技法重复展示。

### 7.17 2026-08-24：精简协议 MiniMax 实测与枚举补强

- 前两次实测分别因非法 envelope／工具轮次耗尽失败。根因是紧凑行的 mode 未枚举且重复动作行不幂等；已限定 `open/stopped/harmonic`、兼容归一 `press→stopped`，并自动过滤与累计状态相同的字段。
- 第三次 `messages_complex_sample_v6_minimax_simplified_retry3` 两阶段 2/2 accepted、协议验证通过，但模型把“注下／吟／猱”写入 `right_finger`，暴露左右手仍允许任意字符串。该样例仅作诊断，不得训练。
- 当前公开 Schema 与运行时均对 `left_finger/right_finger` 使用抽取器合法词表；技法文字不能再冒充左右手。修复后测试 114/114。
- 枚举修复后的 `..._simplified_retry4` 为 2/2 accepted、协议验证通过，关键复合式 259 与 260–266 显示范围正确；但模型把 weak 非空参考 245 清为空以绕开结构冲突，因此仍只作诊断、不得训练。质量门槛现新增 `nonempty_reference_jianzi_dropped`：verified/weak 的非空参考不可被清空，但仍允许 weak 行采用非空合理替代，不扩大为逐字硬匹配。
- Fingering 除完整基础演奏状态外，也必须用 `jianzi_rows` 为每个动作填写基础减字初稿；任一 `jianzi_text=null` 由 `pending_jianzi_text` 拒绝。Guqinizer 在该初稿上修订走手、复合技法和空显示范围，不再从整段“减字待填写”开始。
- `edit_plan` 的公开返回不再同时发送 `text` 内的“错误｜[...]”和重复的顶层 `errors` 数组；模型只看到一次错误信息。结构化 `errors` 继续保存在私有工具审计记录中，验证能力不变。

### 7.18 2026-08-24：极简减字文本协议

- 新教师轨迹的唯一编辑入口是 `edit_plan.jianzi_rows=[[source_index,jianzi_text],...]`。公开 Schema、提示词和预览均不再暴露 `patches`、mode、弦、徽、左右手、attack、technique 或 compound_gesture；当前段表格为“序号／简谱／ABC／时值／谱面减字”，预览为“序号／简谱／减字显示／谱面减字”。
- Fingering 必须填写每个演奏事件的基础减字初稿；Guqinizer 只在初稿上润色走手、复合技法和空显示范围。两个阶段的新增监督 patch 类型均仅为 `SET_JIANZI_TEXT`。旧结构 patch 与内部动作字段继续保留，专供历史数据确定性回放，不进入新模型协议。
- 音高检查直接把整段 Agent 减字送入成熟的上下文解析器。只有简谱和减字均可高置信解析且误差超过50音分时，公开返回 `jianzi_pitch_mismatch` 警告；吟猱、绰注、复合动作、继承动作或未知新词法等不可可靠标量化情况不报错、不阻塞提交。
- 协议硬条件仅保留：合法 JSON/二元行、演奏事件序号、同批不重复、每个动作 `jianzi_text` 非 null、verified 文本精确一致、verified/weak 非空参考不可被清空。音高 warning 不阻塞模型对话。
- 私有审计保存完整 `jianzi_quality_report` 与 `offline_quality.training_eligible`。若最终仍有高置信音高冲突，该阶段写入审计并计入 quarantine，但不写入 `messages_train.jsonl`；因此创作过程保持灵活，明显错音不会进入监督。
- 公开 `edit_plan` 不重复输出顶层 errors；结构化 problems/warnings 仅在私有工具审计保留。当前测试117/117（agent80、scripts37）；旧 `inferred_v6` 全量导出6798/6798有效、rejected=0。

## 8. 复现命令

在仓库根目录、`guqin-agent` 环境中运行。MiniMax 凭据只从本地 `.env` 读取，绝不写入数据、
报告或版本库。

单条真实冒烟：

```powershell
& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\infer_agent_trajectories.py `
  --gqs-dir ABC_J\agent_training\gqs --materialize-repeats `
  --output-dir ABC_J\agent_training\inferred_v6

& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\generate_teacher_tool_trajectories.py `
  --input ABC_J\agent_training\inferred_v6\inferred_trajectories_train.jsonl `
  --basic-intermediate `
  --trajectory-id S0tu8BXC-p0019 `
  --limit 1 --max-tool-rounds 24 --max-attempts 3 `
  --output-dir ABC_J\agent_training\messages_prompt_review_v24
```

pilot 生成（pilot ID 清单来自 `ABC_J\agent_training\pilot_sampling\pilot_ids.txt`；当前 Token Plan
先用单线程，且必须显式传 ID，不能在 4 个 shard 后各自 `--limit 40`，否则会最多选中 160 条）：

```powershell
$pilotArgs = Get-Content ABC_J\agent_training\pilot_sampling\pilot_ids.txt |
  Where-Object { $_.Trim() } | ForEach-Object { '--trajectory-id'; $_.Trim() }

& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\generate_teacher_tool_trajectories.py `
  --input ABC_J\agent_training\inferred_v6\inferred_trajectories_train.jsonl `
  --basic-intermediate --limit 40 --max-tool-rounds 24 --max-attempts 3 `
  @pilotArgs --output-dir ABC_J\agent_training\messages_pilot_next
```

曲级调弦教师生成（含离线自测 `--self-test`）：

```powershell
& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\generate_teacher_tuning_decisions.py `
  --limit 5 --output-dir ABC_J\agent_training\tuning_teacher
```

校验与可视化：

下列 `messages_pilot_v44_final` 是历史 v4 pilot，因此分析命令仍显式使用 v4 source；新 v6 批次
应把 `--source` 改为 v6。

```powershell
& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\validate_teacher_agent_messages.py `
  --input-dir ABC_J\agent_training\messages_pilot_v44_final

& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\analyze_teacher_pilot.py `
  --input-dir ABC_J\agent_training\messages_pilot_v44_final `
  --source ABC_J\agent_training\inferred_v4\inferred_trajectories_train.jsonl

$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\visualize_agent_trajectories.py `
  ABC_J\agent_training\messages_pilot_v44_final\messages_train.jsonl `
  -o ABC_J\agent_training\messages_pilot_v44_final\trajectory_viewer_qwen35.html `
  --qwen35-template

& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\visualize_teacher_io.py `
  ABC_J\agent_training\messages_pilot_v44_final\teacher_trajectory_audit.jsonl `
  -o ABC_J\agent_training\messages_pilot_v44_final\teacher_io_viewer.html
```

批量生成前，应先选定独立输出目录并小批量检查失败样本；不要覆盖已有验证目录。`--limit`、
`--trajectory-id` 可用于分批与断点调查；已有 Fingering 中间体可通过 `--intermediate-input` 续跑
Guqinizer。

## 9. 已知限制与下一步

1. 40 条分层 pilot 已完成，模型质量上具备扩大生产的可行性；429 指数退避／抖动重试和每
   phrase flush 已落地。下一批建议先按 `workers=1` 扩到约 200 条验证长时间稳定性；如要提高
   并发，先单独测 `workers=2`，不建议直接恢复 4 workers 全量。
2. 基础方案允许与标注不同；它是合格的中间体，Guqinizer 才进一步逼近标注。不能把“与标注
   不同”直接当作失败。
3. 普通多弦动作（撮／泼／剌）已支持：动作含 `string2`（伙伴弦）与 `mode2`；语料分布为按音＋散音 80%
   （2,517）、双按 11%、双散 7%，故紧凑行第 7 位裸整数默认散音伙伴弦，双按需显式
   `{"string":N,"mode":"stopped"}`。硬审计要求 `right_finger=撮/泼/剌` 时 `string2` 必填，
   音高按双音各自配对核验，距离按弦组（不分先后）比较。参考解析器从标注文本解析双弦与伙伴
   mode（明文“按音”优先于泛音段继承，伙伴段“散／按音”各自显式化）。渲染约定：散伙伴带
   “散”标记（`＋散3弦`），双按伙伴共享主位徽不重复标（`＋6弦`）。多声复合动作另走
   `compound_gesture`／`SET_COMPOUND_GESTURE`，保留完整名称、跳过单值音高硬判，不要求
   `string2`，不得拆成普通撮或剌。
   再作已按 §5 物化处理；无起点再作的跨度取最近边界是 v1 启发式，pilot 应抽查其展开正确性。
4. 训练／验证／测试必须继续遵守 `ABC_J/results/dataset_split_groups.csv` 的曲级泄漏分组；不得
   在 phrase 展开后重新随机划分。
5. 建议先批量生成一个受控 train 子集，统计：接受率、工具轮数、候选调用量、终止原因、token
   长度、基础方案音高和可演奏性；确认稳定后再扩大到完整 train。
6. 训练 Qwen3.5-9B 时，全部训练轨迹出自教师管线（含曲级调弦决策）；反推轨迹与确定性调弦
   导出退出训练，仅作评估基线与消融对照。
7. 并行批量脚本仍可分片生成后自动合并，但当前实测 4 workers 会触发 Token Plan 429；自动
   退避已经落地，安全并发仍待 `workers=2` 小批实测，`workers=1` 是当前准入设置。重试目录用
   `merge_teacher_retry_outputs.py` 按 sample ID 后写覆盖并可显式剔除已知坏样本。
8. 左手技法协议已进入 40 条真实 pilot；复杂／上下文动作共 104 个未做错误的单值音高硬判。
   该 pilot 没有命中 v5 多声复合动作或 v6 前置绰；批量生产时应单独统计
   `SET_COMPOUND_GESTURE` 的提交率、接受率与最终谱面保留率，并分别统计独立绰
   attack=false、前置绰 attack=true＋pre_attack 以及普通后缀技法的采纳率。无起点再作的跨度
   启发式仍需扩大抽查；SFzXxqnV-p0069 已能经单线程重试完成。

## 7.19 极简减字文本协议：MiniMax 真实单条复验（2026-08-24）

- 样本：`S0FKGjFt-p0011`，模型：`MiniMax-M3`，流程：`basic_intermediate_to_annotation`。
- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_text_only`。
- 结果：Fingering 1 条、Guqinizer 1 条均接受；`quarantined=0`、`rejected=0`；公开/私有 sample ID 一致，消息校验通过。
- 两个阶段公开工具均只暴露 `edit_plan.jianzi_rows=[[source_index,jianzi_text], ...]`，本次公开轨迹没有结构 patch；最终 `training_eligible=true`。
- Fingering 首轮即提交 19 个演奏事件：245/246 等填写实际减字，247--249、260--266 等复合动作覆盖范围使用空字符串，未再出现 `[减字待填写]`。
- “吟”以独立 `[吟]` 保存，没有错误附带徽位；259 为 `[掐撮三声]`，260--266 保留声音而谱面减字为空，符合本轮目标。
- Guqinizer 曾把小节线 252/258、延音 257/267、休止 268 也提交给 `jianzi_rows`，工具逐项拒绝，删除非演奏序号后成功。最终轨迹合法，但提示词后续宜明确写成“只提交待编辑表中具有演奏事件的序号；不要提交小节线、延音和休止”，可减少无价值重试。
- Guqinizer 将部分阿拉伯数字改为汉字；253 因可靠参考要求精确一致而被退回 `散摘7弦`，254/256 的纯字形改写获准。该行为不产生音高错误，但批量前应决定数字字形是否需要统一规范，避免训练模型学习两套等价写法。
- 离线音高审计：24 行中仅 253 可高置信比较且匹配；其余复杂走手、复合技法、空显示或无明确弦位均跳过，没有把不可解析误判成失败。
- 可视化：`trajectory_viewer_qwen35.html`（公开轨迹及最终/标注对比）与 `teacher_io_viewer.html`（教师原始输入输出）。

## 7.20 Fingering 参考可见与候选取音职责修正（2026-08-25）

- Fingering 仍可在教师私有提示中逐行查看最终 GQS 标注；它不是盲生成阶段。
- 为防止该阶段退化成逐字抄标注，恢复只读 `get_pitch_candidates`：Fingering 必须先把当前段全部演奏事件批量查询，工具按目标 MIDI 去重，再依据候选、简谱、ABC、上下文和专业判断生成基础减字。
- 编辑面继续保持极简：唯一写工具仍是 `edit_plan.jianzi_rows`；候选工具不修改方案，也不恢复弦、徽、左右手等结构字段。
- 阶段边界：Fingering 的每个演奏事件必须生成非空基础减字，不照抄最终标注中的复杂走手、复合技法或空显示范围；Guqinizer 再依据最终 GQS 加入复杂技法和显示为空的覆盖范围。
- 公开/私有工具 schema 已统一，Fingering 均能看到 `get_pitch_candidates`；Guqinizer 不暴露该工具。
- 旧占位符 `[取音与左右手待定]` 已改为与当前文本协议一致的 `[减字待填写]`。
- 补充规则：两个阶段都不得向 `jianzi_rows` 提交小节线、延音或休止。
- 回归：agents 80/80、scripts 37/37，总计 117/117；`py_compile` 通过。

## 7.21 候选取音恢复后的 MiniMax 真实复验（2026-08-25）

- 同一样本 `S0FKGjFt-p0011` 输出到 `ABC_J/agent_training/messages_complex_sample_v6_minimax_pitch_grounded`。
- 结果：Fingering 1、Guqinizer 1 均接受；隔离 0、拒绝 0；消息校验通过，二者 `training_eligible=true`。
- Fingering 工具顺序为 `get_pitch_candidates -> edit_plan -> edit_plan`：首轮一次提交全部 19 个演奏事件，工具按实际 MIDI 去重为 6 组候选，证明候选取音职责已真实恢复而非只写在提示词中。
- Fingering 最终给全部 19 个演奏事件填写非空减字；Guqinizer 随后将 247--249、255、260--266 等复合动作覆盖行改为空显示，并推进到最终标注。因此两阶段不再直接完全一致。
- 尚存边界问题：Fingering 对教师 GQS 中已有非空减字的行仍沿用若干复杂写法（如注下、滔起、掐撮三声），主要在标注空白行依据候选生成新基础取音；当前属于“候选取音参与的参考驱动混合初稿”，还不是完全剥离复杂技法的纯基础初稿。
- 可视化：`trajectory_viewer_qwen35.html`；教师原始请求/响应：`teacher_io_viewer.html`。

## 7.22 删除结构字段负面提示（2026-08-25）

- 从 Fingering、Guqinizer 的公开角色提示及教师私有规则中删除“不要输出或维护弦、徽、左右手、attack、technique 或结构 patch”等冗余负面说明。
- 活动写接口本来已经只包含 `edit_plan.jianzi_rows`，无上述结构字段，故不再向模型介绍不存在的编辑概念。
- Fingering 的只读 `get_pitch_candidates` 仍返回方式、弦和徽位，因为这些是候选取音证据，不是可维护状态；Guqinizer 不暴露该工具。
- 内部 typed patch/replay 继续作为确定性审计实现细节保留，不出现在活动工具 schema 或角色提示中。
- 回归测试 117/117 通过。

## 7.23 Fingering 空显示改为柔性偏好（2026-08-25）

- 学生可见 Fingering 提示不再出现“不要逐行照抄最终标注……”；防止教师机械复制最终 GQS 的约束仅保留在教师私有提示中。
- 学生提示改为正向柔性要求：依据候选、上下文和专业判断填写基础减字，尽量填写每个演奏事件，只有谱面关系明显需要留空时才留空。
- 删除代码硬校验 `empty_basic_jianzi_text` 及 `require_nonempty`；Fingering 的空字符串不再导致整次 `edit_plan` 无效。
- 教师私有提示同样把“每行必须非空”改为“尽量填写；明显需要留空时可空”，但继续保留不机械照抄最终复杂技法/空显示范围的教师控制。
- 回归测试 117/117，编译检查通过。

## 7.24 工具调用理由提示放宽（2026-08-25）

- 学生 Agent 提示由“每次工具调用前可用一句简短、可由公开谱面核查的摘要说明理由”简化为“每次工具调用前说明理由”。
- 教师 `decision_summary` 的语义改为“基于公开谱面、前文或已有工具结果的理由或思考过程”。
- 删除80字与“一句话”限制；运行时仅保留500字符的异常输出保护，以及合法 JSON 所需的未转义引号/换行限制。
- 回归测试 117/117 通过。

## 7.25 放宽理由后的 MiniMax 复验与泄漏修复（2026-08-25）

- 首次输出 `messages_complex_sample_v6_minimax_reasoning_flex` 两阶段均接受，但 Fingering 的公开理由出现“GQS提示”并复述私有参考；该版仅作诊断，不得训练。
- 补充公开摘要禁词 `GQS / 教师私有 / 最终标注`，避免教师理由泄漏私有目标；同时删除公开任务末尾残留的“不要提交弦徽、左右手或结构 patch”。回归测试 117/117。
- 修正版输出 `messages_complex_sample_v6_minimax_reasoning_flex_v2`：未再出现上述私有来源词。Fingering 调用顺序为候选查询、历史目录及三段按需展开、编辑提交。
- 修正版 Fingering 被离线过滤隔离：254 写成 `[大指九徽勾4弦]`，解析为 D4，而简谱目标为 C4，相差约 +201.955 音分；Guqinizer 将其改为 `[至4弦]` 后通过并进入公开训练消息。因此该目录公开轨迹仅有 Guqinizer 1 条，教师 I/O 查看器含 Fingering 隔离诊断和 Guqinizer 共2条。

## 7.26 教师提示源头防止公开理由泄漏 GQS（2026-08-25）

- 教师规则明确说明 `decision_summary` 会原样进入学生可见轨迹，只能依据学生已看到的谱面、前文和真实工具结果写理由或思考过程。
- 教师被明确禁止在公开摘要中提及 GQS、教师私有参考、最终标注、参考/目标答案，也不得把仅由私有 GQS 得知的具体减字包装成公开依据；私有目标只能用于内部选择 `tool_calls`。
- 输出契约的 forbidden 段同步加强；生成后禁词检测继续作为第二层兜底。
- 新增“公开摘要包含 GQS 必须拒绝”回归测试；agents 81/81、scripts 37/37，总计118/118，编译检查通过。

## 7.27 教师防泄漏提示真实复验（2026-08-25）

- 输出：`ABC_J/agent_training/messages_complex_sample_v6_minimax_public_reasoning_guard`；Fingering、Guqinizer 各1条接受，隔离0、拒绝0，二者均 `training_eligible=true`，消息校验通过。
- Fingering 两次公开理由只使用当前谱面、前段写法和候选工具结果；未出现 GQS、教师私有、最终标注、reference/teacher/target 等来源词，教师共2轮完成，没有触发泄漏重试。说明源头提示对本次 MiniMax 调用有效。
- Guqinizer 本次省略了理由，原因是教师输出契约仍将 `decision_summary` 标成可选，与公开系统“每次调用前说明理由”冲突。现已改为每次工具调用必填；解析器会拒绝缺少摘要的教师 envelope。
- 测试相应改为“省略 decision_summary 必须拒绝”；总计118/118，编译检查通过。当前已生成轨迹保留该无理由 Guqinizer 调用作为诊断；下一次生成才应用必填修正。

## 7.28 Guqinizer 当前段未显示 Fingering 初稿修复（2026-08-25）

- 根因：Guqinizer 的 `stage_item.baseline_plan` 已正确替换为 Fingering `accepted_plan`，工具运行也确实基于初稿；但文本协议的 action 没有 `mode`，`_action_columns` 在 `mode is None` 时无条件渲染 `[减字待填写]`，忽略了已存在的 `jianzi_text`。
- 修复：当结构字段为空但 `jianzi_text` 已设置时，当前段直接显示该减字；`jianzi_text=""` 显示为空；仅 `jianzi_text is None` 才显示 `[减字待填写]`。
- 因此新生成的 Guqinizer 用户提示会真实展示 Fingering 初稿，模型可见状态、内部 baseline 和工具预览三者一致。
- 新增文本中间体无结构字段仍可见的回归测试；agents 82/82、scripts 37/37，总计119/119，编译检查通过。旧查看器不会回溯改变。

## 7.29 泛音区间状态与真实复验（2026-08-25）

- 对 `inferred_v6` 去重扫描：`泛起`713处，其中705处位于减字开头（98.9%）；`泛止`635处，其中613处位于减字末尾（96.5%），280处为独立“泛止”。少数例外含单字内“泛起…泛止”及“泛止曲终”，故提示采用“常规写法”而非绝对语法。
- 新增按同曲早期 confirmed reference 顺序重放“泛起/泛止”的 phrase 起始状态；同一减字同时含两个标志时也按文本顺序处理。
- 学生提示在“当前段”后增加：`泛音区间｜当前段开始时是/否`，以及“进入时在减字开头加泛起、结束时在减字末尾加泛止”的常规写法提示。
- `S0FKGjFt-p0011` 起始状态正确显示“否”；该曲实际到 p0030/index673 才泛起，p0032/index747 泛止。
- 真实输出：`messages_complex_sample_v6_minimax_harmonic_state`；Fingering、Guqinizer各1条接受，隔离0、拒绝0，均可训练，消息校验通过。
- 本次Guqinizer当前段已正确显示Fingering初稿而非全量占位符；所有公开工具调用均带理由。
- 回归：agents 83/83、scripts 37/37，总计120/120，编译检查通过。

## 7.30 独立泛止与延音行编辑支持（2026-08-25）

- 去重统计：280处独立 `[泛止]` 中，238处位于延音事件（85.0%），42处位于非延音事件。因此“泛止只能拼在当前减字末尾”不符合数据。
- 公共提示改为两种常规合法形式：可在当前减字末尾添加“泛止”；也常在随后的延音行单独填写“泛止”，不得强行合并。
- 文本初稿 scaffold 现在为延音事件增加 `attack=false / jianzi_text=""` 的 display-only action，使 `edit_plan.jianzi_rows` 能真实编辑延音行；小节线和休止仍不可编辑。
- Fingering 的候选查询提示收窄为“有明确简谱音高的起音”，避免把延音提交给音高候选；Fingering/Guqinizer 均可在延音行写独立控制减字。
- 新增延音可编辑、休止不可编辑测试；agents 84/84、scripts 37/37，总计121/121，编译检查通过。

## 7.31 空文本变更误入旧结构审计与 Fingering 机械复制（2026-08-25）

- `messages_complex_sample_v6_minimax_harmonic_state` 中 Fingering 虽调用候选工具，最终仍与标注逐行一致；教师私有提示已加强：当参考 GQS 含走手、复合技法或成片空显示时，应依据候选重建基础取音初稿，不应逐行照抄，最终化内容留给 Guqinizer；若参考 GQS 本身简单、基础且可直接演奏，则允许初稿与其一致，不得为了制造差异而改写。该约束仍为提示层偏好，不新增僵硬的“必须不同”硬门槛。
- Guqinizer 首次调用 `edit_plan.jianzi_rows=[]`；由于没有文本变化，内部 patch 列表为空。旧代码以 `patches and all(...)` 判断文本协议，空列表因此误入遗留结构审计，产生 `pending_mode/pending_string/reference_attack_mismatch` 等无关错误。
- 修复：真实文本运行时及其终止/最终审计显式调用 `validate_jianzi_only`，不再通过 patch 非空性猜测协议；空文本变更现在按当前 `jianzi_rows` 协议验证。遗留结构化 validator 的空 patch 行为保留，仅用于历史测试与回放。
- 新增空 `jianzi_rows` 不得产生 `pending_mode` 的运行时回归测试；agents 85/85、scripts 37/37，总计122/122，编译检查通过。旧轨迹中的错误保留为历史诊断。

## 10. 最小验收清单

- `messages_train.jsonl` 与 `teacher_trajectory_audit.jsonl` 的 sample ID 一一对应；
- 所有公开工具调用均有真实 tool 返回；
- 每条样本以 `termination.kind = accepted_edit_plan` 终止，且其最后一条 edit_plan 预览
  为“可应用”（终止与可见反馈一致）；
- 每个提交方案可重放，且通过相应阶段硬审计；
- 公开 messages 中不含 `teacher_private`、`reference_plan`、`reference_actions`、
  `target_patches` 或答案措辞；
- Qwen3.5 chat template 可无错误序列化；
- 任何导出与日志都不包含 `.env`、API key、请求头或隐藏思维链。

## 7.32 删除教师生成器旧结构化协议兼容层（2026-08-25）

- `generate_teacher_tool_trajectories.py` 已删除旧的结构化编辑协议：`PATCH_AFTER_SCHEMA`、`calculate_guqin_pitch`、compact patch 展开/规范化、旧 Fingering/Guqinizer 结构化质量校验及旧 direct generator。
- 删除了先构造旧结构化教师指令、随后被文本协议覆盖的死代码；教师可写接口现在唯一为 `edit_plan.jianzi_rows=[[source_index,jianzi_text], ...]`。
- 删除旧 `exact_reference` 生成分支、`--basic-intermediate` 开关及仅服务旧分支的结构化筛选参数。生成器现在固定执行 `basic_intermediate_to_annotation` 两阶段流程，可用 `--stage` 限制阶段。
- 活动 patch 词汇校验只接受内部 `SET_JIANZI_TEXT`；不再兼容 `CHANGE_FINGER`、`REPOSITION`、`SET_COMPOUND_GESTURE`、`ADD_TECHNIQUE` 等教师提交格式。
- 保留底层 `trajectory_replay`、音高审计和 `inverse_diff`：它们分别用于安全应用文本、离线质量过滤和构造私有参考方向，不属于教师可见协议。
- Fingering 的教师约束表述为：不应提交与参考 GQS 逐行完全一致的复杂最终化结果；若参考本身简单、基础且可直接演奏，则允许一致，不为制造差异而改写。
- 删除仅覆盖旧结构化兼容行为的测试后，当前回归为 agents 65/65、scripts 37/37，总计 102/102；`py_compile` 通过。

## 7.33 删除兼容层后的 MiniMax 单条复验（2026-08-25）

- 样本 `S0FKGjFt-p0011`，输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_text_protocol_clean_v2`。
- 首次复验暴露清理回归：预览函数仍调用 `render_jianzi_surface`，但其导入被误判为旧代码删除，导致 `edit_plan` 连续返回 `NameError`。已恢复必要导入；这属于文本预览依赖，不是旧结构化协议兼容层。
- 修复后 MiniMax-M3 生成 Fingering 1 条、Guqinizer 1 条，均接受且 `training_eligible=true`；隔离 0、拒绝 0、质量问题 0、警告 0。
- 消息验证通过：2 条轨迹，公开/私有 sample ID 一致，无校验错误。
- 已生成 `trajectory_viewer_qwen35.html`（含最终版本与标注版本对比）和 `teacher_io_viewer.html`。

## 7.34 禁止公开工具反馈泄漏教师参考（2026-08-25）

- 在 `messages_complex_sample_v6_minimax_text_protocol_clean_v2` 发现公开 `edit_plan` 返回 `reference_jianzi_text_mismatch`，并包含 `expected`/`actual`；它泄漏了 253 的“散摘7弦”、259 的“掐撮三声”及 260--266 的空显示范围。该目录已由新版校验器判定为无效，不得用于训练。
- 根因是 Guqinizer 的公开预览错误复用了 `toward_reference=True` 的教师私有比较。现已改为公开工具始终只做可应用性与可可靠解析的音高检查；参考逐行差异仅写入私有审计 `private_reference_review`，不作为公开硬门槛。
- 接受终止不再要求与参考字符串完全一致；模型可保留音高正确、可演奏的等价写法。教师仍可在私有提示中使用 GQS 进行规划，但不得通过工具反馈把答案传给学生。
- 学生可见 Guqinizer system prompt 中“教师 GQS”已改为“专业判断”；消息校验器新增禁止公开出现 `GQS`、`教师私有`、`最终标注` 及两个参考反馈错误码。
- 新增回归测试验证：即使私有参考为“散摘7弦”、当前文本为“散挑七弦”，公开 `edit_plan` 仍可应用，且结果中不出现参考文本或差异码。当前回归 agents 66/66、scripts 37/37，总计 103/103。
- 修复后复验目录 `messages_complex_sample_v6_minimax_no_reference_leak` 未出现参考反馈泄漏，但本次两个阶段均因独立音高过滤进入 quarantine，未导出公开训练消息；这属于模型本次取音质量问题，不是泄漏回归。

## 7.36 Fingering 基础技法范围收窄（2026-08-25）

- Fingering 的学生可见提示和教师私有规则现明确：基础初稿只使用泛音、按音、散音，以及右手抹、挑、勾、剔、擘、托、打、摘。
- 绰、注、吟、猱、走手及其他复杂技法不再由 Fingering 主动生成，交给 Guqinizer 润色；这只是阶段职责约束，不改变 `jianzi_rows` 接口。
- 新增方向定义：绰上为左手由较低徽位滑向较高徽位，注下为由较高徽位滑向较低徽位。
- 回归测试保持通过：agents 66/66、scripts 37/37，总计 103/103，编译检查通过。

## 7.37 基础技法范围提示后的 MiniMax 重跑（2026-08-27）

- 样本 `S0FKGjFt-p0011` 输出至 `ABC_J/agent_training/messages_complex_sample_v6_minimax_basic_modes_v2`。
- MiniMax-M3 两阶段均成功：Fingering 1、Guqinizer 1，隔离 0、拒绝 0，训练资格通过。
- 消息校验通过（2/2）；已生成 `trajectory_viewer_qwen35.html`（最终版—标注版对比）和 `teacher_io_viewer.html`。

## 7.38 只读空减字、等价弦号与复合技法知识映射（2026-08-27）

- 只读前一段中 `jianzi_text is None` 的行现在显示中性占位 `[空]`，不再显示会误导模型的 `[减字待填写]`；当前待编辑段仍保留 `[减字待填写]`。
- 私有参考复核新增表面规范化：`7弦` 与 `七弦` 等数字写法视为等价，不再因字形差异产生参考不一致。
- 新增教师私有复合技法知识表。目前收录 `掐撮三声`：左手名指按弦、大指在上一音位作罨，掐起后右手撮一声；再罨并先正后反连续掐起两次，最后再撮一声，共八声。只有参考中出现该动作时才注入教师提示，未映射动作不臆造解释。
- Fingering 右手基础动作集合增加 `撮`，定义为食指、中指挑勾并作、同得一声；复杂走手仍交给 Guqinizer。
- 回归测试：agents 69/69、scripts 37/37，总计 106/106，编译检查通过。

## 7.39 复合技法映射版本 MiniMax 重跑（2026-08-27）

- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_gesture_map_v1`，样本 `S0FKGjFt-p0011`。
- Fingering 与 Guqinizer 各生成并接受 1 条，隔离 0、拒绝 0；消息校验通过（2/2）。
- 教师私有 system 已实际注入 `掐撮三声` 知识和绰上/注下方向说明；公开轨迹未包含私有参考反馈。
- 已生成轨迹对比查看器和教师 I/O 查看器。

## 7.42 分阶段提示词边界修正（2026-08-27）

- Fingering 教师提示不再包含“绰上/注下方向”或“收到预览后只修正变化行”等 Guqinizer 工作规则。
- Guqinizer 的公开提示明确当前段来自只会基础指法的初稿 Agent，并要求在不破坏音高和可演奏性的前提下适当加入高级指法，使减字更丰富有韵味。
- Guqinizer 私有提示明确 GQS 只是改进方向，不要求逐字复制，并保留绰上/注下方向和预览修正规则。
- Fingering 公开提示改为“填写或编辑曲谱只能使用 `edit_plan.jianzi_rows`”，并要求尽量覆盖每个发音事件的减字。
- 回归：agents 69/69、scripts 37/37，总计 106/106；抽查确认 Fingering 不含上述 Guqinizer 规则，Guqinizer 含新增职责说明。

## 7.43 分阶段提示词重跑结果（2026-08-27）

- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_prompt_layers_v1`。
- MiniMax 两阶段均完成调用但各因离线音高过滤进入 quarantine（Fingering 与 Guqinizer 各 1 条），没有公开训练消息；这不是提示词泄漏或协议错误。
- 教师 I/O 查看器已生成；轨迹查看器因 `messages_train.jsonl` 为空，仅显示 0 条样本，不能作为最终轨迹验收。

## 7.44 分阶段提示词重跑通过（2026-08-27）

- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_prompt_layers_v2`。
- MiniMax-M3 两阶段均成功，隔离 0、拒绝 0，消息校验通过；已生成最终版—标注版轨迹查看器和教师 I/O 查看器。
- 关键复核：Fingering 本轮将 251 写为“挑四”，音高审计通过；此前误写 3 弦导致的约 502 音分错误未再出现。

## 7.45 `jianzi_rows` 字符串序号容错（2026-08-27）

- MiniMax 偶尔把 `source_index` 输出为 JSON 字符串，如 `["245","注下七徽九分"]`，尽管 Schema 声明为整数。
- Schema 和描述现更明确要求 JSON 整数；运行时同时容错纯十进制字符串并规范化为整数，避免可恢复格式错误打断整轮。
- 浮点字符串（如 `"245.0"`）、任意文本、布尔值和浮点数仍严格拒绝，防止错误编辑到其他音。
- 回归测试：agents 70/70、scripts 37/37，总计 107/107，编译检查通过。

## 7.46 Guqinizer 单阶段重跑（2026-08-27）

- 复用 `messages_complex_sample_v6_minimax_prompt_layers_v2/fingering_intermediates.jsonl` 中已通过的 Fingering 中间体，仅生成 Guqinizer。
- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_guqinizer_retry_v1`；Guqinizer 1 条接受，隔离 0、拒绝 0。
- 消息校验通过（1/1）；已生成单阶段轨迹查看器和教师 I/O 查看器。

## 7.47 项目内复杂指法知识库与“如一”映射（2026-08-27）

- 已将 `C:\Users\30343\Downloads\古琴减字谱复杂指法解释.jsonl` 复制到项目内 `agents/abc_to_jianzipu/knowledge/complex_fingering_explanations.jsonl`，原始知识库共 157 条。
- 生成器加载该文件，拆分斜杠别名并按动作名去重；当前有 162 个唯一检索键。冲突的同名解释会在启动时直接报错，避免静默覆盖。
- 映射注入仍只发生在 Guqinizer 私有提示：仅对长度至少两字的动作或“特殊/左右手配合指法”类别作最长匹配，避免普通单字（如挑、吟、指）泛滥注入；更具体的“掐撮三声”会覆盖“掐撮”。
- “如一”使用手工增强解释：两弦一按一散，按音徽位与散弦同音高，右手同时剔两弦，如同一声。
- 回归：agents 71/71、scripts 37/37，总计 108/108，编译检查通过。

## 7.40 修正复合技法知识的阶段注入范围（2026-08-27）

- 复核 `messages_complex_sample_v6_minimax_gesture_map_v1` 后确认：`掐撮三声` 映射此前被无条件注入 Fingering 和 Guqinizer 两个教师提示，超出了 Fingering 职责。
- 现已改为仅在 Guqinizer（`toward_reference`）阶段注入；Fingering 只接收基础取音规则和候选工具结果，不接收复合技法知识。
- 新增回归断言，确保 Fingering 公共提示不含 `掐撮三声`；agents 69/69、scripts 37/37 通过，编译检查通过。

## 7.41 阶段映射边界重跑与可视化（2026-08-27）

- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_gesture_map_v3`，样本 `S0FKGjFt-p0011`。
- MiniMax-M3 两阶段均成功并通过音高过滤：Fingering 1、Guqinizer 1，隔离 0、拒绝 0；消息校验通过。
- 私有提示抽查确认：Fingering 不含 `复合技法知识｜掐撮三声`，Guqinizer 含该映射；Fingering 含右手“撮”基础动作规则。
- 已生成轨迹最终版—标注版对比查看器和教师 I/O 查看器。

## 7.35 无参考反馈泄漏版本重跑（2026-08-25）

- 输出目录：`ABC_J/agent_training/messages_complex_sample_v6_minimax_no_reference_leak_v2`，样本 `S0FKGjFt-p0011`。
- MiniMax-M3 两阶段均成功：Fingering 1、Guqinizer 1，隔离 0、拒绝 0，均 `training_eligible=true`。
- 消息校验通过，公开/私有 ID 一致；公开消息扫描未发现 `reference_jianzi_text_mismatch`、`nonempty_reference_jianzi_dropped`、`expected`、`教师 GQS`、`教师私有` 或 `最终标注`。

## 7.48 可视化首屏合并两阶段最终版对比（2026-08-27）

- `scripts/visualize_agent_trajectories.py` 现在会按源轨迹（`source_trajectory_id`）把 Fingering、Guqinizer 和标注合并为一张首屏对比表，而不是在两个阶段各显示一张重复的“最终版 vs 标注版”表。
- 表列固定为“序号｜简谱｜Fingering 最终版｜Guqinizer 最终版｜标注版本”；显式空减字显示 `[空]`，阶段没有该行则显示 `—`。消息轨迹仍完整保留在表格之后，便于同时看结果和过程。
- 已重新生成通过样例的查看器：`ABC_J/agent_training/messages_complex_sample_v6_minimax_prompt_layers_v2/trajectory_viewer_qwen35.html`。
- 新增合并表回归测试；当前 agents 71/71、scripts 38/38，总计 109/109 通过。

## 7.49 《水调歌头》泛音段 MiniMax 两阶段验证（2026-08-27）

- 新选样本 `SB4hDksv-p0001`（《水调歌头》开篇，事件 0--27），与此前《樵歌》`p0011` 的复合动作样例不同，覆盖泛起、泛音区间、延音 `泛止` 与和弦。
- MiniMax-M3 两阶段均成功：Fingering 1、Guqinizer 1，隔离 0、拒绝 0；两条均 `training_eligible=true` 且无校验错误。
- 产物位于 `ABC_J/agent_training/messages_harmonic_sample_v2/`；已生成含首屏三方合并表的 `trajectory_viewer_qwen35.html`。

## 7.50 Guqinizer 增量编辑与公开逐音分析约束（2026-08-27）

- Guqinizer 的公开与教师提示均明确：调用 `edit_plan.jianzi_rows` 只能列出相对当前初稿确实改写的音，不得为了整齐而重发未变化行。
- 每次 Guqinizer 提交前须逐音审阅；学生可见 `decision_summary` 必须逐一说明改写序号，或列出采用同一判断的相邻音组及其序号，且仅能依据公开谱面、前文和工具结果。
- `教师提示`、`系统提示`、`提示词`、`教师规则`、`私有规则` 已加入公开摘要泄漏拦截词；此类表述会在 envelope 解析阶段直接拒绝，不再进入训练轨迹。
- 新增提示泄漏及 Guqinizer 增量提示回归；当前 agents 73/73、scripts 38/38，总计 111/111 通过。

## 7.51 宽容增量提示版《水调歌头》复验（2026-08-27）

- 依用户偏好，未把“只提交改动行”做成硬拒绝：`edit_plan` 对重复文本仍保持既有幂等过滤，避免不必要地中断模型；该要求只通过 Guqinizer 提示词引导。
- 重跑目录：`ABC_J/agent_training/messages_harmonic_sample_v7_delta_prompt/`。MiniMax-M3 Fingering、Guqinizer 各接受 1 条，隔离 0、拒绝 0。
- 本轮 Guqinizer 实际只提交一行 `jianzi_rows=[[0,"泛起食指七徽勾三弦"]]`；公开摘要逐项讨论了 0、10、13、16、21 号，未包含 GQS、教师提示、系统提示或其他私有来源措辞。
- 已生成首屏 Fingering／Guqinizer／标注三方对比：`trajectory_viewer_qwen35.html`。

## 7.52 三方对比表单元格匹配高亮（2026-08-27）

- 首屏对比表的 Fingering 与 Guqinizer 单元格现在独立与标注版本比较；一致的格子以绿色高亮，而不是只在行级给出结果。
- 比较会忽略空白及一至七／1--7 的无害字形差异；例如 `散挑七弦` 与 `散挑7弦` 会标绿，`挑` 与 `摘` 不会被误判一致。
- 已重新生成 `messages_harmonic_sample_v7_delta_prompt/trajectory_viewer_qwen35.html`；脚本测试 38/38 通过。

## 7.53 《水调歌头》Guqinizer 单阶段重跑与配套阶段可视化（2026-08-27）

- 复用 `messages_harmonic_sample_v7_delta_prompt/fingering_intermediates.jsonl`，只重跑 `SB4hDksv-p0001` 的 Guqinizer；结果接受 1、隔离 0、拒绝 0。
- 本轮 Guqinizer 审阅后确认初稿无需改写，真实调用为 `jianzi_rows=[]`；这是保留而非失败，未伪造增量修改。
- 可视化脚本新增 `--companion-input`／`--companion-stage`，可以把单阶段重跑与对应前阶段公开轨迹、各自审计合并为同一首屏三方表。该样例查看器位于 `messages_harmonic_sample_v8_guqinizer_retry/trajectory_viewer_qwen35.html`。

## 7.54 私有参考与当前段统一表格式的验证（2026-08-27）

- `render_teacher_reference_gqs` 改为与待编辑段相同的五列表格：`序号｜简谱｜ABC｜时值｜谱面减字`；常见 `1弦`--`7弦` 表面字形同步为一弦--七弦。参考仍保留真正不同的减字，避免把格式统一误变成答案覆盖。
- 同表格式首次验证 `messages_harmonic_sample_v9_reference_table/` 中，Guqinizer 实际将 16、21 改为 `注下七徽九分`、`注下七徽六分`，说明格式统一能帮助其识别局部润色机会；但该轮公开理由含“与参考一致”“按提示”，不得作为训练轨迹。
- 已把“与参考一致／参考使用／参考中／按提示”加入公开理由泄漏拦截，并在 `messages_harmonic_sample_v10_reference_table_guard/` 重跑。该轮 1 条接受、零隔离拒绝，公开轨迹未泄漏私有来源，但模型保守地提交空增量；体现出 MiniMax 的采样波动，不能用单次空增量否定格式改动。
- v10 已生成 `trajectory_viewer_qwen35.html`（配套 Fingering 三方对比）与 `teacher_io_viewer.html`；当前 agents 75/75、scripts 38/38，总计 113/113 测试通过。

## 7.55 移除 decision_summary 长度限制与重试提示污染（2026-08-27）

- `extract_decision_summary` 不再限制 500 字；仍保留非空、合法 JSON 和公开来源泄漏拦截，API 的 `max_tokens` 仍是模型响应总量上限而非摘要字段长度限制。
- `generate_one` 重试时不再把 `上一次尝试失败：...` 注入教师私有 `rules`；错误只保存在内部失败轨迹，避免实现细节污染教师判断。
- 新增长摘要回归测试；当前 agents 76/76、scripts 38/38，总计 114/114 通过。

## 7.56 精简教师私有参考标签（2026-08-27）

- 将教师系统输入中的 `【教师私有参考谱｜与当前段同表格式】` 精简为 `【教师私有参考谱】`；表格结构仍保留在参考正文中，不再把实现细节写进标签。
- agents 76/76、scripts 38/38 回归通过。

## 7.57 逐音分析提示改写（2026-08-27）

- Guqinizer 公开提示改为：先逐一分析每个音应使用的指法/减字，再决定是否编辑；只把决定改写的音提交给 `edit_plan`。
- 教师私有规则进一步要求重点关注当前稿与标注的不同点，并逐一分析差异音的音高、技法和减字取舍。
- agents 76/76、scripts 38/38 回归通过。

## 7.58 私有决策与公开 reasoning 分层（2026-08-27）

- 教师私有规则现在明确两层职责：私有标注可用于内部决定“哪里需要改、往什么方向改”；公开 `decision_summary`/reasoning 不再要求假装没看标注，而是解释该修改为何在古琴演奏上合理。
- 公开 reasoning 仍禁止提及 GQS、标注、参考答案或私有提示，也不得把“因为标注这样写”作为理由；应分析前后音、走手方向、按音连续性、音高和时值等公开音乐依据。
- 新增分层提示回归；当前 agents 79/79、scripts 38/38，总计 117/117 通过。

## 7.59 分层提示 MiniMax 重跑（2026-08-27）

- 输出目录：`ABC_J/agent_training/messages_harmonic_sample_v11_reasoning_split/`；复用 `messages_harmonic_sample_v7_delta_prompt` 的 Fingering 中间初稿，Guqinizer 1 条接受、隔离 0、拒绝 0。
- 本轮公开 reasoning 逐音审阅后只提交 source 10 的撮写法调整，理由围绕泛音段、按音连续性和十六分音符时值展开，未提及私有标注或教师提示。
- 已生成首屏三方对比 `trajectory_viewer_qwen35.html` 与教师输入输出 `teacher_io_viewer.html`。

## 7.60 随机片段两阶段验证（2026-08-27）

- 首次随机抽到 `Sq6pbetG-p0030`，其 26 行参考全部为空/不可监督，因此 Fingering 后没有 Guqinizer 目标，未将其冒充完整两阶段样本。
- 重新随机抽取 `SmMYcR5r-p0034`（17 个发音事件、9 条可用标注），MiniMax-M3 Fingering 与 Guqinizer 各接受 1 条，隔离 0、拒绝 0。
- 产物位于 `ABC_J/agent_training/messages_random_SmMYcR5r_p0034/`，已生成三方对比 `trajectory_viewer_qwen35.html` 和逐轮教师 I/O `teacher_io_viewer.html`。
- 该样本显示基础阶段能按音高候选逐事件填写；Guqinizer 虽最终只实际改变局部行，但原始调用仍提交了多行幂等文本，提示词引导尚未完全稳定，不能据此宣称模型已解决增量调用问题。

## 7.61 修复 verified 减字不一致的训练资格审计（2026-08-27）

- 发现 `validate_jianzi_only` 过去只把 verified 参考文字不一致记录在 `private_reference_review`，却没有影响 `offline_quality.training_eligible`；因此 `SmMYcR5r-p0034` 的 734 等错写仍可能显示可训练。
- 现已将 verified 参考不一致/非空清空纳入训练资格硬条件；weak 参考仍仅供方向和诊断，不扩大硬门槛。回归覆盖 verified mismatch，agents 80/80、scripts 38/38，总计 118/118 通过。
- 既有轨迹文件不回写篡改；按新审计规则复核时，`messages_random_SmMYcR5r_p0034` 应淘汰，不得进入训练。

## 7.58 MiniMax-M3 原生 thinking 试验（2026-08-27）

- 生成器增加了受控环境变量 `MINIMAX_THINKING=enabled`；默认仍为 `disabled`。启用时请求使用原生 `thinking`，响应中的思考块只保存在私有教师 I/O，不进入公开消息。
- 对已授权的 `SB4hDksv-p0001` 做了一次试验；请求超过一分钟无响应且进程不再增长，终止后没有生成轨迹或报告。不能据此把 thinking 直接用于批量生产，需先确认 MiniMax 当前端点的 thinking 参数和超时支持。

## 7.59 adaptive thinking 单样本 smoke 与完整响应链（2026-08-27）

- 按要求将 thinking 试为 `{"type":"adaptive"}`，先以 1 样本、1 round 调试。`max_tokens=8000` 时 MiniMax 返回的 `response.content` 只有 thinking 块，`thinking_tokens=8000`，`stop_reason=max_tokens`，没有后续普通 text/tool。
- 将 smoke 上限临时提高至 16000 后请求持续挂起，未返回可用 `response.content`，已终止；两个目录均为空调试产物，不得训练。
- 多轮链路已改为把完整 `response.content` 序列化后作为下一轮 assistant 消息（包括 thinking）；公开消息仍只从 text 块生成 `decision_summary` 和工具记录。
- 生产默认恢复为 `thinking={"type":"disabled"}`；仅显式设置 `MINIMAX_THINKING=adaptive` 才启用 adaptive，避免端点不兼容导致批量任务挂起。`MINIMAX_MAX_TOKENS` 可用于受控 smoke 覆盖默认 8000。

## 7.62 GLM-4.5 同片段两阶段对照（2026-08-27）

- 根目录 `.env` 已支持 GLM 配置：`GLM_API_KEY`、`GLM_BASE_URL`、`GLM_MODEL`；密钥不写入轨迹、不写入 handoff，也不应提交到版本库。生成器在模型名以 `glm` 开头时自动选择 GLM 配置，默认使用 `thinking={"type":"disabled"}`，避免该端点只返回 thinking 块而没有可解析的普通文本。
- 对 `SmMYcR5r-p0034` 重跑两阶段：第一次 Guqinizer 的 3 次尝试均因公开摘要复述私有参考被拒；第二次重试成功，`messages_glm_SmMYcR5r_p0034_v3/` 中 Fingering、Guqinizer 各接受 1 条，隔离 0、拒绝 0，工具真实执行、回放和字段审计均通过。
- GLM Fingering 为 17 个发音事件填写了基础减字；Guqinizer 提交 10 行增量修改。按标注语义归一化后，这 10 行均匹配，但 719、726、727、732、738 仍与标注不一致（前者未从“散挑四弦”改为“挑三弦”，后四行未清空）。因此该样本说明 GLM 比先前 MiniMax 更愿意做局部高级技法修改，但仍不能视为高质量监督样本。
- 该轮公开 `decision_summary` 出现“参考谱面建议”等私有来源表述，并逐字复述若干私有减字；现有 `private_leakage_passed` 未捕获这些变体，故该轨迹只能用于模型/审计诊断，不能直接纳入训练。后续应扩充公开泄漏词形和“私有具体文本复述”检测，再评估 GLM 批量生产资格。
- 已生成：`ABC_J/agent_training/messages_glm_SmMYcR5r_p0034_v3/trajectory_viewer_qwen35.html`（首屏三方对比）与 `teacher_io_viewer.html`（教师 I/O）。

## 7.63 澄清空字符串的 edit_plan 语义（2026-08-27）

- `edit_plan.jianzi_rows` 的空字符串现在明确表示“将该行 `jianzi_text` 置空”，不会删除声音、attack 或其他演奏状态。
- 当上一行的复合减字已经覆盖后续多个动作时，后续行可提交空字符串，表示不重复显示减字；这是常见用法，但不是空字符串本身的底层语义。
- 已同步更新工具 description、参数 schema、教师规则和预览标签（显示为“已置空”），并新增回归断言；`test_teacher_quality` 当前 35/35 通过。

## 7.64 训练数据末端增加 reasoning 脱敏重写阶段（2026-08-27）

- 依当前决定，教师模型阶段允许保留原始 reasoning，即使其中出现私有参考或教师提示的复述；原始教师 I/O 继续只用于内部审计、问题诊断和质量分析，不直接作为学生可见训练文本。
- 在训练数据生成流程末尾新增一个独立的“reasoning 脱敏重写”阶段：输入为已接受的教师轨迹，输出为不泄露私有来源的公开 reasoning。该阶段只改写可见的 `decision_summary`／reasoning 文本，不改动工具调用、`jianzi_rows`、最终减字、样本 ID、阶段标签或审计结果。
- 脱敏重写应保留每个修改音的序号、修改内容和音乐学依据（音高、时值、前后音连接、指法可演奏性、走手方向等），删除或改写“GQS、标注、教师提示、参考谱面、参考建议”等来源表述及私有具体文本复述。
- 需要同时保存原始版与脱敏版：原始版放在私有教师 I/O／审计产物中，脱敏版才进入学生可见消息和最终训练集；脱敏后应重新执行 JSON、工具调用回放、字段一致性和公开泄漏检查。
- 当前 `messages_glm_SmMYcR5r_p0034_v3` 可作为该后处理阶段的验证样本：两阶段调用已接受，但 Guqinizer 的公开摘要含私有来源措辞，正好覆盖脱敏重写的目标场景。

## 7.65 reasoning 脱敏重写脚本落地与样例验证（2026-08-27）

- 新增 `scripts/redact_teacher_reasoning.py`。用法：`python scripts/redact_teacher_reasoning.py --input-dir <教师轨迹目录> --output-dir <脱敏目录> --model <模型>`。
- 脚本按“每个带工具调用的 assistant turn 单独重写”处理，要求模型返回单个 `{"summary": ...}`；失败样本默认不写入脱敏训练输出，避免把未脱敏原文混入最终数据。
- 输出包含脱敏后的 `messages_train.jsonl`、原样复制的私有 `teacher_trajectory_audit.jsonl`、私有 before/after 对照 `reasoning_redaction_audit.jsonl` 和 `reasoning_redaction_report.json`。除公开摘要外，不修改工具调用、工具结果、减字、样本 ID 或阶段字段；仅追加 `generation_mode` 后缀和 `reasoning_redaction` 状态。
- 脱敏校验包括来源词、私有参考具体文本和弦/徽/音高等数值事实保护；新增 `scripts/test_redact_teacher_reasoning.py`。目前与教师质量测试合计 41 项通过。
- 用 GLM-4.5 对 `messages_glm_SmMYcR5r_p0034_v3` 实测输出到 `messages_glm_SmMYcR5r_p0034_v8_redacted/`：2/2 条成功，0 失败；`validate_teacher_agent_messages.py` 报告 valid=true，工具声明及调用结构保持一致。该目录可作为脱敏阶段的功能样例，不代表其原始 Guqinizer 决策已达到训练质量门槛。

## 7.66 恢复复杂指法知识映射（2026-08-27）

- 此节原记录是误解用户意图后的临时恢复操作，已撤销；用户删除这些条目是有意精简，不应按下载源补回。
- 当前项目映射恢复为用户原先保留的 145 条，继续沿用既有手工筛选结果；下载目录文件仅作为参考，不作为自动同步源。

## 7.67 双吟映射吸收与训练源数据冻结（2026-08-28）

- 最后一批减字代码映射（`:tples` 家族、`:tflss/:tfles`、`:yin:sua` 双吟等，共 10 个）完成后发现：`inferred_v6` 仍是 2026-08-24 版本，缺 08-27 的双吟映射（15 处 `:yin:sua` 裸码残留在参考文本中）。已用当前 `teacher.gqs`（teacher-gqs-1.1，含全部新映射，241 首往返校验通过）确定性重生成 `inferred_v6`。
- 重生成校验：三个 split 的 phrase ID 与旧 v6 **逐一相同**（train/validation/test = 4,739/654/1,405，历史样本与 pilot 引用不受影响）；参考文本原始代码残留 **0**；双吟 15 处、掐拨剌二声 9 处、掐拂歷三声 15 处等新名字全部进入参考；`SET_COMPOUND_GESTURE=215` 等既有语义统计保持。
- **训练源数据自此冻结**：`ABC_J/agent_training/inferred_v6/` 为最终训练源数据，是后续所有教师批量生成的固定输入。冻结含义：
  1. phrase ID 与曲级 split 划分不再变动，任何已生成样本的 `source_trajectory_id` 引用持续有效；
  2. 此后任何数据层修改（抽取器映射、参考解析、分节、再作物化、语义补全）都必须走 readable → gqs → v6 完整级联重生成，并重新验证 phrase ID 不变性与既有样本引用一致性；不满足不变性的重生成视为破坏冻结，须排查而不是接受；
  3. `inferred_v3/v4/v5` 仅作历史对照，不得用于新教师生成；基于旧版本生成且未通过当前审计重审的 messages 目录同样不得进入训练。
- 减字代码映射状态：全库 241 首 readable jianzi 的原始代码残留为 0（含 round2 路径），App 减字编码到可读文本的映射对训练语料已完整封闭。

## 7.68 GLM-4.5 串行批量 40 条（2026-08-28）

- 使用冻结后的 `ABC_J/agent_training/inferred_v6/inferred_trajectories_train.jsonl`，固定尝试前 40 个唯一 `trajectory_id`；未运行 reasoning 脱敏阶段。
- 输出目录：`ABC_J/agent_training/messages_batch_glm_40/`。单进程、无 shard 并发，每次 API 请求间隔 2 秒；生成器启用 `--resume`，依据既有 public/private/rejected/checkpoint 记录跳过已尝试 ID，避免断点续跑重复写入。
- 结果：40 个唯一片段均已尝试；Fingering 接受 38 条、Guqinizer 接受 11 条，公开/私有消息共 49 条，隔离 0。29 个阶段失败，主要是 Guqinizer 公开 reasoning 触发私有参考复述审计，另有少量 Fingering 同类失败；详细错误保存在 `generation_report.json` 与 `teacher_rejected_io.jsonl`。
- 已生成 `trajectory_viewer_qwen35.html`（两阶段与标注对比）和 `teacher_io_viewer.html`（教师 I/O）；`validate_teacher_agent_messages.py` 报告 valid=true，49 条 public/private ID 对齐且无重复。批量报告的 `source_phrases` 已校正为实际尝试数 40。
- 本批不因失败自动修改提示词或放宽审计；失败样本不直接进入训练。脱敏阶段仍按第 7.64--7.65 节作为独立后处理，未并入本批。

## 7.69 原始 reasoning 允许泄露后的批次补跑（2026-08-28）

- 用户已明确：本批先保留原始 reasoning，后续再做脱敏重写；因此生成器新增显式开关 `--allow-private-reasoning-leakage`。默认仍为严格拒绝，只有原始采样阶段显式开启时才接受含私有来源的 `decision_summary`，不改变工具回放、字段和音高审计。
- 新增 `--retry-failed`：在 `--resume` 下只重试 `attempted_with_failure` 阶段，已接收样本不重跑；失败阶段的 Fingering 中间结果优先复用。此前批次因泄露门槛失败的阶段已按此规则补跑。
- 最终批次目录仍为 `ABC_J/agent_training/messages_batch_glm_40/`：40 个唯一片段，Fingering 接受 40 条，Guqinizer 接受 39 条，共 79 条两阶段消息；唯一未完成阶段为 `S0FKGjFt-p0002` 的 Guqinizer，其多轮 JSON 输出不稳定，已保留一条去重后的失败记录，不再继续无止境重试。
- `messages_train.jsonl` 与 `teacher_trajectory_audit.jsonl` 均 79 行且 `sample_id` 唯一；checkpoint 的 40 个唯一片段中 39 个 `completed`、1 个 `attempted_with_failure`。`teacher_rejected_io.jsonl` 保留各次尝试的原始诊断，不作为训练集。
- 因原始 reasoning 按决定允许泄露，`validate_teacher_agent_messages.py` 对该批报告 `valid=false`，仅提示 4 条公开摘要含 `GQS`；这属于预期的脱敏前状态，**本目录不得直接作为学生可见训练集**，须先运行第 7.64--7.65 节的 reasoning 脱敏阶段。
- 已重新生成 `trajectory_viewer_qwen35.html`（两阶段最终版本/标注对比）和 `teacher_io_viewer.html`（教师 I/O）；未运行脱敏阶段，未改动冻结训练源数据，也未引入并发。

## 7.70 全量剩余片段并发批量完成（2026-08-28）

- 在用户明确授权将剩余片段（含私有标注和未脱敏 reasoning）发送到 GLM API 后，先安全停止单进程，再启动 4 个隔离 worker；每个 worker 使用 0.5 秒请求间隔，独立写入 shard，协调器完成去重合并，避免并发交错写总文件。
- 4 个 worker 均以退出码 0 完成。当前总目录 `ABC_J/agent_training/messages_batch_glm_40/` 的 checkpoint 覆盖冻结训练源 4,739 个唯一片段；共接收 8,763 条消息，其中 Fingering 4,694 条、Guqinizer 4,069 条；188 个阶段失败，主要是模型输出不符合单对象 JSON 协议，已在报告中按 `(trajectory_id, stage)` 去重。
- 产物行数：`messages_train.jsonl` 8,763，`teacher_trajectory_audit.jsonl` 8,781，`fingering_intermediates.jsonl` 4,708，`checkpoint.jsonl` 4,770，`teacher_rejected_io.jsonl` 1,510。worker 合并按 `sample_id`/`trajectory_id` 去重，未重复写入成功样本。
- `generation_report.json` 记录 `source_phrases=4,739`、`workers=4`、`min_interval=0.5`、`worker_exit_codes=[0,0,0,0]`。由于本批保留原始 reasoning，公开消息仍可能含私有来源词；验证脚本的 `valid=false` 属于脱敏前预期状态，不能直接发布为学生可见训练集。
- 脱敏阶段仍未运行；正式训练前必须按第 7.64--7.65 节对全量 accepted 消息执行 reasoning 脱敏、重新校验，并按质量策略处理 `training_eligible=false` 与 188 个失败阶段。

## 7.71 JSON 协议解析兼容修复与失败片段补跑（2026-08-28）

- 修复 `scripts/generate_teacher_tool_trajectories.py` 的教师响应解析：改用 `JSONDecoder.raw_decode` 提取首个完整对象，安全处理前后说明文字/代码围栏；仅对缺失末尾括号做保守补全；兼容唯一、无歧义的 `{"tool_turn": {...}}` 包装。
- `tool_calls: []` 现可作为“本段无需修改”的协议结果；只有当前基线仍存在待填写演奏音时才要求模型继续提交 `jianzi_rows`，避免把真正的空操作误报为协议失败。报告按 `(trajectory_id, stage)` 去重，历史原始尝试仍保留在 `teacher_rejected_io.jsonl`。
- 新增/更新协议回归测试，教师工具、质量、GQS、脱敏测试合计 59 项通过。
- 对上一轮剩余协议失败的 64 个唯一片段（对应 127 条重复失败记录）补跑：新增接受 36 条，当前报告为 accepted 8,930、Fingering 4,717、Guqinizer 4,213、未解决 37 条；其中 35 条仍为模型多轮协议/工具轮次问题，2 条为 GLM 1301 平台拦截。未运行 reasoning 脱敏。

## 7.72 reasoning 脱敏增加实时进度与断点续跑（2026-08-28）

- `scripts/redact_teacher_reasoning.py` 现在逐样本追加写入 `messages_train.jsonl` 和 `reasoning_redaction_audit.jsonl`，不再等全部任务结束后一次性写文件；中断时已完成样本可直接复用，避免重复调用 API。
- 每个输出目录实时维护 `reasoning_redaction_progress.json`，包含 `status`、`completed`、`failed`、`remaining`、`api_calls`、`last_sample_id` 和更新时间；临时断行会被安全跳过。最终仍生成 `reasoning_redaction_report.json`。
- 旧的四路并发脱敏因出现 `APIConnectionError` 已停止；先以单路完成 worker 0 并确认网络稳定，再依次恢复 worker 1--3。当前剩余三个分片使用独立输出目录并行运行，持续观察连接失败率；全部完成后再合并、校验，不把中途目录当作最终训练集。

## 7.73 全量 reasoning 脱敏完成（2026-08-28）

- 四个分片均已完成并合并到 `ABC_J/agent_training/messages_batch_glm_40_redacted/`；新增 `scripts/merge_redaction_shards.py` 按训练源原始顺序合并，检查跨分片重复 ID 与审计键冲突。
- 输入 accepted 消息 8,935 条，脱敏后写入 8,889 条；46 条因模型改写仍复述私有减字/来源词、数值校验失败、JSON 无法解析或 API 连接失败而保留在报告失败列表，未混入公开训练输出。
- 输出行数：`messages_train.jsonl` 8,889、私有 `teacher_trajectory_audit.jsonl` 8,889、`reasoning_redaction_audit.jsonl` 15,158。公开 assistant reasoning 的来源词扫描为 0 命中（工具结果中的同名词不计入 reasoning）。
- `validate_teacher_agent_messages.py` 已通过：`valid=true`，公私有 ID 对齐，无结构错误；阶段统计为 Fingering 4,713、Guqinizer 4,176。脱敏目录可作为学生可见训练候选集，但 46 条失败样本需按质量策略另行重试或排除。

## 7.74 脱敏失败样本并行重试（2026-08-28）

- 从 7.73 报告提取 46 条失败样本，分成 4 路独立调用 GLM-4.5；33 条重试成功，13 条仍未通过（主要是私有减字复述、来源词或数值校验）。
- 33 条成功结果已合并回 `ABC_J/agent_training/messages_batch_glm_40_redacted/`，并按冻结训练源顺序重排；新增 `scripts/merge_redaction_retry.py`，同时补齐私有审计与 before/after 记录。
- 当前最终输出 8,922 条，失败 13 条；`validate_teacher_agent_messages.py` 再次通过：`valid=true`，公私有 ID 对齐，无结构错误。Fingering 4,716、Guqinizer 4,206。
- 13 条失败样本继续保留在 `reasoning_redaction_report.json` 中，不进入公开训练集；如需进一步重试，应针对具体失败原因调整脱敏提示或本地清理策略。

## 7.75 Qwen3.5 工具轨迹 SFT 训练代码与首版数据导出（已由 7.76 取代，2026-08-28）

- 新增 `train/scripts/export_sft_dataset.py`：从 `messages_batch_glm_40_redacted/` 导出 LLaMA-Factory 0.9.6 的 `openai` 数据集格式。每一个带工具调用的 assistant 回合独立成为一条 SFT 样本；历史公开消息/工具结果为上下文，当轮公开 reasoning 与 Qwen3.5 XML 工具调用为监督目标。这样既不丢弃 reasoning，也不把整条多轮轨迹只监督到最后一步。
- 工具定义序列化为 JSON 字符串，训练 jsonl 顶层固定为 `messages` 与 `tools` 两列，规避 Hugging Face Arrow 对异构工具 schema 的推断问题。导出不包含任何 `teacher_trajectory_audit.jsonl`、教师原始 I/O 或环境密钥。
- 首版公开训练集已导出到 `train/data/guqin_agent_sft_v1/`：源消息 8,922，工具回合样本 15,230（Fingering 10,234、Guqinizer 4,996）；`validate_sft_export.py` 通过，工具调用表面、公开 reasoning 隔离和 stable schema 均无错误。
- 训练配置与服务器脚本：`train/configs/guqin_agent_lora_4x3090.yaml`、`train/scripts/preflight_sft.py`、`train/scripts/render_train_config.py`、`train/server/`。沿用服务器 `~/models/Qwen3.5-9B`、固定提交 `~/LLaMA-Factory`（0.9.6.dev0）和经过 4×RTX 3090 验证的 bf16/4-bit QLoRA 配方；`bootstrap_guqin_sft_env.sh` 从现有 `cip` 环境克隆为 `guqin-sft`，不改动原环境。
- 尚未上传服务器、未克隆服务器环境、未启动 smoke 或全量训练。正式执行前必须先用 `train/server/run_sft.sh smoke` 做真实两条样本检查；完整 preflight 若发现 2048 截断率超过 5% 会拒绝启动。当前教师批只覆盖 train split，未生成独立 validation 教师轨迹，因此首版配置不做 eval，不能从 train phrase 随机拆出伪验证集。

## 7.76 完整多轮 trajectory SFT 导出（2026-08-28）

- 纠正 7.75 的框架判断：服务器 LLaMA-Factory 0.9.6.dev0 在 `mask_history=false` 时会对多轮中每个 assistant/function target 计算 loss；只有 `mask_history=true` 才只训练最后一轮。因此不再按工具回合拆样本。
- `export_sft_dataset.py` 已改为一条完整 Agent 轨迹对应一条训练记录；每个 assistant 的脱敏 reasoning 与 Qwen3.5 XML tool call 合并在 `content`，避免 OpenAI converter 用原生 `tool_calls` function JSON 覆盖 reasoning。末尾没有后续 assistant 的 tool result 会被删除，其他 user/tool observation 均保留为上下文。
- 新数据位于 `train/data/guqin_agent_sft_v2_full_trajectory/`：8,922 条完整轨迹（Fingering 4,716、Guqinizer 4,206），含 15,239 个 assistant 回合和 15,267 个 tool-call XML block；`validate_sft_export.py` 报告 valid=true。
- 新增 `inspect_sft_labels.py`，将直接调用服务器实际 `SupervisedDatasetProcessor` 随机检查 labels；训练配置显式固定 `mask_history=false`、`train_on_prompt=false`。启动脚本要求真实模板截断率为 0，并在训练前完成 labels 抽检。
- 用户明确授权后，新版脱敏训练 JSONL 与检查脚本已上传实验室服务器。真实 Qwen3.5 tokenizer 全量 preflight：8,922 条均可渲染，P50=3,455、P95=5,655、P99=6,319、max=17,679；在 32,768 下截断率为 0，据此将正式候选 `cutoff_len` 收紧为 18,432。
- 服务器实际 LLaMA-Factory 0.9.6.dev0 labels 抽检随机选择 5 条轨迹（第 787、5152、5617、7807、8652 行），共 8 个 assistant 回合：`mask_history=false`、`train_on_prompt=false`；所有 assistant reasoning/tool call target 完整有 label，所有 user/tool observation source 内容被 mask，实际 labels 与预期逐 token 一致，报告 valid=true。
- 尚未启动 GPU smoke 或正式训练。下一步是以 `CUTOFF_LEN=18432` 做 4×RTX 3090 两条轨迹 smoke；若 OOM，应先调整显存配方，不能降低 cutoff 静默截断完整轨迹。

## 7.77 完整轨迹的 2048 显存约束评估（2026-08-28）

- 服务器真实 tokenizer 以 `cutoff_len=2048` 全量统计：8,922 条中 8,523 条超过上限，比例 95.53%；直接训练会截断绝大多数轨迹并漏掉后续 assistant targets，不可接受。
- 普通 4 卡 DDP 不会合并单卡显存，每张卡仍处理完整序列；当前服务器四张 3090 均被另一个 Qwen3.5 评测任务占用约 12.1GB，未启动本项目 smoke，也未停止或干扰该任务。
- 当前 QLoRA 配方新增 `use_unsloth_gc=true`，使用该 LLaMA-Factory 内置的 hidden-state CPU offload gradient checkpointing，并继续启用 Liger；待 GPU 释放后再从短到长做显存 smoke。
- 当前 checkout 的 v1 Ulysses sequence parallel 明确拒绝 Qwen3.5；Megatron Bridge 则支持 `qwen3_5`、LoRA、context/tensor parallel，但不支持量化模型，且服务器尚未安装 megatron-bridge。它是保持逐字完整 18k 轨迹的高成本备选，不作为当前默认方案。
- 若硬件最终只能承受 2,048，唯一务实的数据路线是保持完整轮次顺序和所有 assistant reasoning/tool call，确定性压缩不参与 loss 的 tools schema、user 表格与 tool observations，并剔除 assistant target 本身已超过 2,048 的异常长重试轨迹；这会保留完整决策链，但不再逐字保留全部上下文。不得用普通从头截断伪装成完整轨迹训练。

## 7.78 缺失“只读前一段”上下文的定向修复（2026-08-29）

- 长轨迹可视化发现少数非首段的公开 user prompt 没有 `【只读前一段 <phrase_id>｜confirmed_readonly】`。审计确认这不是 SFT 导出或 reasoning 脱敏删除了上下文：原始 `messages_batch_glm_40/` 中相同记录也缺失该段，而冻结训练源 `inferred_v6` 中已有前段数据。旧 assistant 决策是在缺少该上下文的条件下作出，不能事后仅补 prompt 文本。
- 新增 `scripts/repair_phrase_context_dataset.py`，提供：`audit`（按当前冻结输入审计真实缺口）、`prepare-intermediates`（为 Guqinizer 组合原/新 Fingering 初稿）、`combine`（组合阶段产物）和 `merge`（按 `sample_id` 在新目录原子替换，不修改旧批次）。修复清单保存在 `ABC_J/agent_training/messages_batch_glm_40_context_repair/repair_manifest.json`。
- 经用户确认后，最小范围重跑：9 条 Fingering、29 条 Guqinizer（共 30 个 source phrase、38 个旧 accepted stage 样本）。9 条 Fingering 全部成功；29 条 Guqinizer 中 22 条产生新的编辑轨迹，另 7 条在新 Fingering 初稿下已无须高级修改，按生成器正常语义不产生空操作 Guqinizer 轨迹。
- 另有 2 条重跑 Fingering 仅含延音/休止，模型输出直接说明而没有工具调用；它们不满足完整工具轨迹 SFT 的最低结构，和上述 7 条无须编辑的 Guqinizer 一样从训练集排除，而不保留旧的错误条件样本。故最终净替换 29 条、删除 9 条：去敏公开消息从 8,922 变为 **8,913**（Fingering 4,714、Guqinizer 4,199）。
- 31 条实际重建轨迹已单独运行 GLM-4.5 reasoning 脱敏，成功 31、失败 0；原始修复版在 `messages_batch_glm_40_context_repaired_raw/`，最终公开训练候选在 `messages_batch_glm_40_context_repaired/`。后者 `validate_teacher_agent_messages.py` 通过，public/private ID 对齐；修复后的非首段缺失前段上下文审计为 0。
- 已从最终公开目录重新导出完整多轮数据到 `train/data/guqin_agent_sft_v2_full_trajectory_context_repaired/`：8,913 行、14,970 个 assistant turn、15,002 个 Qwen3.5 XML tool-call block；`validate_sft_export.py` 为 `valid=true`。该目录取代 7.76 的旧 v2 导出，后续服务器 preflight / labels 抽检 / 训练应使用它；旧目录保留为可追溯历史版本。

## 7.79 公开 user prompt 去除重复协议标题（2026-08-29）

- 按用户要求，`render_public_prompt()` 不再输出 `任务｜fingering_agent` / `任务｜guqinization`；阶段身份已在 system prompt、`agent_stage` 元数据和工具轨迹中明确，无须在 user 内容重复。
- 同时去除末尾的 `要求｜通过 edit_plan.jianzi_rows 提交减字文字。`；唯一编辑入口仍由 system prompt 和 `edit_plan` 的工具 schema 约束，删除这句不改变协议。
- 已运行 teacher protocol / quality / GQS 回归测试，53 项通过。此变更仅影响后续新生成轨迹；当前已冻结、验证通过的 8,913 条训练候选不回写改动，避免输入与既有 assistant 决策不一致。

### 既有训练候选的确定性文本清理（同日）

- 用户确认这两段内容也可直接从既有公开 user 消息中作精确字符串删除。新增 `scripts/clean_legacy_user_prompt.py`：仅对 `role=user` 删除开头 `任务｜fingering_agent` / `任务｜guqinization` 与末尾完全匹配的要求句；不改 assistant reasoning、tool call、tool result、谱面表格或私有审计。
- 已原位处理 `messages_batch_glm_40_context_repaired_raw/`（8,926 条原始 accepted 消息）和 `messages_batch_glm_40_context_repaired/`（8,913 条公开脱敏训练候选）。公开集 8,913 个 user 消息均精确删除两处；清理报告分别保存在两个目录的 `legacy_user_prompt_cleanup_report.json`。
- 已重新运行公开消息校验及 v2 完整轨迹导出：`validate_teacher_agent_messages.py` 与 `validate_sft_export.py` 均 `valid=true`；训练行数、assistant turn 和 tool-call block 统计不变。最终 SFT 数据仍为 `train/data/guqin_agent_sft_v2_full_trajectory_context_repaired/`，但其 `source_messages_sha256` 已更新为 `80b9d547…`。

## 7.80 expand_context 重复前段调用修复（2026-08-29）

- 发现 `SFzXxqnV-p0069-fingering_agent-teacher-tools` 在 user 已包含 `【只读前一段 p0068｜confirmed_readonly】` 时，仍调用 `expand_context(p0068)`。全量公开训练候选审计后，`expand_context` 仅有这一次调用，且唯一属于重复前段，不存在“真正查更早段”的既有轨迹。
- `render_public_prompt()` 在已给出前段时新增简短注记：前段直接使用，只有需要更早段时才 `list_context → expand_context`。`expand_context` 的 schema 描述同步明确禁止重复展开；运行时也会返回错误而非泄漏/重复返回内容。
- `expand_context` 的成功返回由原先的结构化 `actions` 数组改为只读简表文本：`【只读更早段 …】` 加 `序号｜简谱｜ABC｜时值｜谱面减字`，与 user 当前段/`edit_plan` 预览一致，不再把内部 action 结构直接暴露给模型。
- 该 Fingering 已用 GLM-4.5 重新生成：调用链为 `get_pitch_candidates → edit_plan → edit_plan`，不再调用 `expand_context`；该条 reasoning 脱敏成功。其下游 Guqinizer 首两轮（每轮 3 次 API 尝试）因模型没有按 JSON 包络输出而未接受；不是音乐判断失败。
- 针对 GLM 的一种无歧义变体新增保守兼容：当模型先输出非空公开 reasoning，紧随其后只输出一个合法 `{"tool_calls": [...]}` 对象时，解析器把该前缀归入 `decision_summary`；没有 reasoning 的 calls-only JSON 仍拒绝。新增两项回归测试，teacher protocol / quality / GQS 共 55 项通过。
- 提高重试次数后，Guqinizer 成功接受并经 reasoning 脱敏后插回 Fingering 之后；其调用链为 `edit_plan → edit_plan`，同样不调用 `expand_context`。最终公开训练候选恢复为 **8,913**（Fingering 4,714、Guqinizer 4,199），完整多轮导出为 8,913 行、14,970 assistant turn、15,002 tool-call block；公开消息和 SFT 导出校验均 `valid=true`。当前最终集的 `expand_context` 调用数为 **0**。后续服务器 preflight/训练继续使用同一目录 `train/data/guqin_agent_sft_v2_full_trajectory_context_repaired/`。

## 7.81 get_pitch_candidates 首轮批量查询提示优化与定向重跑（2026-08-29）

- 针对长轨迹 token 激增的主要来源——Fingering 逐音调用 `get_pitch_candidates`——仅调整提示引导，不增加硬性放行条件：当当前段有 4 个或以上可解析发音时，首次查询尽量一次提交至少 4 个（最好全部）起音序号；只有后续需要聚焦核查单音时才逐音查询。工具 schema 仍允许 `source_index` 单音调用，保留灵活性。
- 同步更新了公开 system prompt、教师私有 Fingering 规则和工具描述；新增提示明确要求“不要把首次查询拆成逐音调用”。回归测试 55 项全部通过。
- 全量审计发现只有 5 个已接受片段确实受影响（有至少 4 个可解析发音但首轮少于 4 个）：`SFzXxqnV-p0069`、`SQAf6j3a-p0014`、`SaBzwJPS-p0026`、`SmFP1nH0-p0002`、`Sqg2C0GU-p0012`。这 5 条 Fingering 与其下游 Guqinizer 均以 GLM-4.5 串行重跑，Fingering 5/5、Guqinizer 5/5 接受，失败 0。
- 新轨迹显示首轮批量分别为 6、4、13、11、4 个 source index；原先分别为 3、1、1、3、3。5 条 Fingering 的 API 请求数由 17 降为 11，输入 token（GLM usage）由 24,193 降为 22,971，输出 token 由 7,512 降为 4,099；这验证了批量查询能减少轮次和输出开销，但不是对全量 Qwen3.5 序列长度的直接测量。
- 新 raw/公开候选目录：`ABC_J/agent_training/messages_batch_glm_40_pitch_batch_repair/`（原始阶段结果）与 `ABC_J/agent_training/messages_batch_glm_40_pitch_batch_repaired/`（脱敏、合并后公开候选）；对应完整多轮 SFT：`train/data/guqin_agent_sft_v2_pitch_batch_repaired/`。公开消息校验、SFT 导出与 SFT 校验均 `valid=true`，仍为 8,913 条（Fingering 4,714、Guqinizer 4,199）。旧的 `context_repaired` 目录保留，未覆盖。
- 本轮未重新上传服务器：上传新生成公开数据的外部传输需要针对该新 payload 的单独授权；因此 Qwen3.5 官方 tokenizer 的 P50/P95/P99/max 尚未更新。若要确认全量 token 统计，下一步只需明确授权后上传新的 `guqin_agent_sft_v2_pitch_batch_repaired/guqin_agent_train.jsonl`，不必再次调用教师模型。

## 7.82 新候选集 Qwen3.5 tokenizer 长度复测（2026-08-29）

- 本机缓存的 Qwen3.5-9B tokenizer 与 chat template 已对 `train/data/guqin_agent_sft_v2_pitch_batch_repaired/guqin_agent_train.jsonl` 全量复测；无需再次上传服务器即可复现官方模板统计。
- 8,913 条均可渲染，`P50=3,382`、`P95=5,608`、`P99=6,217`、`max=12,599`；超过 2,048 的 8,477 条（95.108%），超过 18,432 的 0 条。该统计对应首轮批量优化后的候选集；其中 12,599 的最长样本是 `ShnpF8Qg-p0052-fingering_agent-teacher-tools`（第 2,665 行）。
- 后续针对批量参数失败的样本重跑后，最新候选集的最大值已降为 `8,564`（P50=3,382、P95=5,608、P99=6,212；超过 2,048 仍为 8,477 条），最长样本变为 `Ssn9EWFF-p0020-fingering_agent-teacher-tools`（第 6,734 行）。
- 最新最长轨迹可视化：`C:\Users\30343\\.codex\\visualizations\\2026\\08\\27\\01a04324-2f64-76f1-9983-9acfa9c17711\\max-token-trajectory-pitch-batch-v2.html`。

## 7.83 批量音高查询失败的鲁棒性修复（2026-08-29）

- 调查确认旧最长样本的首轮 `get_pitch_candidates` 传入了 `类型=["按音","散音"]`；旧运行时代码只按单字符串处理，触发 `TypeError: unhashable type: 'list'`，随后模型退化为 21 次逐音调用，造成 12,599 token 峰值。
- 工具 schema 现在同时接受单个类型字符串和字符串数组；批量 `source_indices` 中混入不存在、休止、小节线或无法解析音高的序号时，跳过无效序号并继续处理有效音，在结果中给出跳过说明，不再让整批失败。新增两个回归测试，测试总数 57 项全部通过。
- 重新生成 `ShnpF8Qg-p0052` 与 `SaljUbT2-p0060` 的两阶段轨迹，Fingering/Guqinizer 均 2/2 成功；前者的音高查询由 23 次（1 次失败 + 22 次逐音）降为 1 次批量，后者由 5 次降为 1 次批量。最新 SFT 候选目录为 `train/data/guqin_agent_sft_v2_pitch_batch_repaired_v2/`，公开消息和 SFT 校验均 `valid=true`，轨迹总数仍为 8,913。

## 7.84 `edit_plan` 不可编辑索引汇总、休止行开放与定向重跑（2026-08-29）

- 调查 `messages_batch_glm_40_pitch_batch_repaired_v2_audit/teacher_trajectory_audit.jsonl`：共有 896 个阶段样本出现过 `jianzi source_index is not editable`，其中 159 个轨迹含至少 2 个不同索引，故建立 `ABC_J/agent_training/edit_not_editable_multi_ids.txt` 作为定向重跑清单。旧错误的主要成因是模型把宽范围表格中的小节线也提交给 `edit_plan`；这不是减字内容缺失。
- `expand_jianzi_rows()` 现在收集同一调用中的全部不可编辑索引，一次返回 `jianzi source_index is not editable: i, j, ...`，不再遇到第一个索引就截断；普通音、延音和休止行均可作为文字编辑行，小节线仍保持结构性不可编辑。休止不进入音高查询，但可承载 `[走猱]` 等延续减字（例如 `0（休止）` 行）。公开/私有提示同步说明该语义。
- `blank_plan_from_item()` 会把路由基线遗漏的普通音和休止/延音补入 text-only 可编辑骨架；休止初始为无起音、空文本，模型可填写走猱、猱、吟、泛止等减字。新增测试覆盖遗漏普通音、休止行和多索引汇总；教师协议/质量/GQS 回归测试共 59 项通过。
- 按清单对 159 条轨迹用 GLM-4.5、4 个隔离分片、请求间隔 0.5 秒完整重跑两阶段。4 个 worker 均退出码 0；重跑目录 `ABC_J/agent_training/messages_batch_glm_40_not_editable_rerun_v2/`，接受 207 个阶段样本（Fingering 149、Guqinizer 58），97 个阶段仍因主要是 `decision_summary` 私有引用复述/协议不稳定而失败。剩余的不可编辑错误出现在 29 个阶段样本、71 个不同索引，且全部确认是小节线；每条调用已一次汇总，不再重复逐索引报错。
- 与最新批量候选合并时，自动丢弃两条“只有 system/user/assistant、没有工具回合”的无操作 Guqinizer 记录，保留原有可训练版本；最终合并目录 `ABC_J/agent_training/messages_batch_glm_40_not_editable_merged_v3/`，公开/私有各 8,916 行，`validate_teacher_agent_messages.py` 通过。完整多轮 SFT 导出到 `train/data/guqin_agent_sft_v2_not_editable_merged_v1/`，`export_sft_dataset.py` 成功，8,916 条轨迹、14,724 个 assistant turn、14,721 个工具回合。
- 该合并目录是“重跑候选/审计”而非脱敏最终集：它以尚未脱敏的最新批量候选为基底，SFT 结构导出成功但 `validate_sft_export.py` 因继承的 8 条私有来源词命中而报告 `valid=false`；正式训练前仍须走 reasoning 脱敏，不能直接上传此目录。

## 7.85 原始 reasoning 接受开关补正重试（2026-08-29）

- 复核 7.84 后确认：用户既定流程允许原始 `decision_summary` 含私有信息，后续统一脱敏；7.84 的 159 条定向重跑命令漏传 `--allow-private-reasoning-leakage`，导致报告中的 97 个阶段失败主要是假性严格门槛，不应当作模型质量失败。
- 已按正确策略补重：10 条 Fingering 阶段全部成功；87 条 Guqinizer 阶段中 77 条成功、10 条仍是模型 JSON 包络/轮次协议失败。补重目录分别为 `ABC_J/agent_training/messages_batch_glm_40_not_editable_retry_v3_fingering/` 与 `ABC_J/agent_training/messages_batch_glm_40_not_editable_retry_v3_guqinizer/`，均保留原始 reasoning 供后续脱敏。
- 合并后的审计候选为 `ABC_J/agent_training/messages_batch_glm_40_not_editable_merged_v4/`，公开/私有 ID 对齐，`validate_teacher_agent_messages.py` 报告 valid=true；与 v3 相比新增 1 条此前缺失的 Guqinizer 阶段，最终 8,917 条（Fingering 4,714、Guqinizer 4,203）。该目录仍是未脱敏候选，不得直接作为学生训练集。

## 7.86 最后 10 条 Guqinizer 重试、全量脱敏与 token 统计（2026-08-29）

- 对 7.85 剩余的 10 条 Guqinizer 协议失败样本再次重试（`--allow-private-reasoning-leakage`、最多 5 次尝试）：7 条成功，3 条仍为 JSON 包络/轮次协议失败（`SXBbuM3G-p0002`、`Sgm0yeKr-p0004`、`ShnpF8Qg-p0002`）。没有继续无限重试；失败记录保留在 `ABC_J/agent_training/messages_batch_glm_40_not_editable_retry_v4_guqinizer/`。
- 当前完整候选 `messages_batch_glm_40_not_editable_merged_v5/` 经四路脱敏后写入 `ABC_J/agent_training/messages_batch_glm_40_redacted_v1/`；8,917 条全部处理成功，实际脱敏 API 调用 212 次。发现 1 条无工具调用的无操作 Guqinizer 记录，按 SFT 最低结构要求移除，得到最终公开脱敏目录 `ABC_J/agent_training/messages_batch_glm_40_redacted_v2/`：8,916 条，Fingering 4,714、Guqinizer 4,202；公开消息校验 valid=true。
- 完整多轮 SFT 导出：`train/data/guqin_agent_sft_v2_not_editable_redacted_v1/`，8,916 条；`validate_sft_export.py` valid=true。该目录已不含私有来源词，可作为训练候选集。
- 使用本机缓存 Qwen3.5-9B tokenizer 与 chat template 统计 `token_distribution.json`：总计 8,916 条，P50=3,300.5，P90=5,302.5，P95=5,592，P99=6,170.7，最大=8,142；超过 2,048 的 8,469 条（95.0%），超过 8,192/16,384 的均为 0。Fingering：P50=4,725、P95=5,832、P99=6,379.9、max=8,142；Guqinizer：P50=2,446、P95=2,937、P99=3,206.9、max=4,384。最长样本为数据第 766 行 `S3gMUKsg-p0013-fingering_agent-teacher-tools`。

## 7.87 Guqinizer 协议失败的额外重试结论（2026-08-29）

- 对剩余 3 条协议失败样本增加到最多 10 次尝试：`SXBbuM3G-p0002`、`Sgm0yeKr-p0004`、`ShnpF8Qg-p0002` 中前两条（SXB、Shnp）在额外重试中成功；`Sgm0yeKr-p0004` 再次 10 次尝试仍无法输出合法包络，失败原因均为顶层字段/缺失 `decision_summary`，不是私有信息过滤。
- 前两条本已在此前合并候选中存在成功版本，故最终脱敏训练集内容和 token 统计不变；`Sgm0yeKr-p0004` 保留失败审计，不进入训练。当前最终脱敏集仍为 8,916 条，`token_distribution.json` 统计保持有效。

## 7.88 重试轨迹实际替换与 token 统计更正（2026-08-29）

- 复核发现 7.87 的“前两条本已存在、内容不变”表述不准确：`SXBbuM3G-p0002` 和 `ShnpF8Qg-p0002` 虽然 sample_id 已存在，但新版本确实由 8 条消息缩短为 4 条消息，已重新脱敏并替换进 `messages_batch_glm_40_redacted_v4/`。
- 更新后的 SFT 数据为 `train/data/guqin_agent_sft_v2_not_editable_redacted_v2/`；两条样本 token 分别由 2,804→2,370、3,029→2,254。由于仅替换 2/8,916 条且它们不是最长样本，总体分布只发生极小变化：P50=3,300.5、P95=5,592、P99=6,170.7、max=8,142，超过 2,048 仍为 8,469 条。
- 新增 `train/scripts/measure_token_distribution.py`，统计文件为该 SFT 目录下的 `token_distribution.json`；`validate_sft_export.py` valid=true。后续若重试成功版本已存在同一 sample_id，必须比较内容哈希后再决定是否替换，不能只按 ID 判断“无需更新”。
- 首轮批量参数失败的具体结论：旧 `get_pitch_candidates` 遇到混入的非发音序号即整批失败；本轮修复后的批量查询错误数为 0，休止/小节线只会在被查询时被跳过并说明。后续若再出现不可编辑错误，优先检查是否为小节线；普通音和休止不应再被基线覆盖范围误拒。

## 7.89 `edit_plan` 局部提交不再被未提交行阻塞（2026-08-29）

- 修复 `scripts/generate_teacher_tool_trajectories.py`：`edit_plan.jianzi_rows` 现在允许只提交本轮已经决定的部分行；未提交的其他音保留现状，不再在本轮返回 `pending_jianzi_text`。Fingering 仍要求在最终结束前为所有发音事件提供字符串（必要时为空字符串），Guqinizer 可按需提交任意局部改写。
- `validate_jianzi_only()` 增加 `require_complete` 区分：工具回合采用局部校验，最终结束/训练资格校验仍采用完整覆盖校验；因此没有放松非法索引、重复行或高置信音高错误的检查。
- 同步更新 edit_plan schema、公开 system prompt 与教师提示，明确“可以分批提交已确定的音；未提交的音不会使本次预览失败”。新增回归测试；教师协议/质量/GQS 共 60 项通过。
- 从 `messages_batch_glm_40_not_editable_merged_v7/teacher_trajectory_audit.jsonl` 定位到 198 条曾出现 `pending_jianzi_text` 的 Fingering 轨迹，使用 GLM-4.5 定向重跑：首轮 184 条成功，另重试 14 条成功 8 条，最后对 6 条允许暂存私有 reasoning 后全部成功；198 条新审计均不再出现 `pending_jianzi_text`。
- 对应重跑候选原始目录为 `ABC_J/agent_training/messages_batch_glm_40_partial_edit_rerun_v1/`；合并候选为 `ABC_J/agent_training/messages_batch_glm_40_partial_edit_merged_v4/`。合并后公开/私有各 8,918 条，Fingering 4,714、Guqinizer 4,204；`validate_teacher_agent_messages.py` 为 `valid=true`，全量审计中该错误为 0。删除了既有无工具 Guqinizer 无操作记录 `SH8hkkfN-p0013-guqinization-teacher-tools`。
- 该合并候选仍含本轮 6 条暂存私有 reasoning，属于待脱敏原始候选，不能直接作为学生训练集；应先对替换样本做 reasoning 脱敏，再重新导出/校验 SFT。旧的 `messages_batch_glm_40_redacted_v4` 与 `train/data/guqin_agent_sft_v2_not_editable_redacted_v2` 未被原位覆盖。

## 7.90 局部提交候选脱敏与 SFT 预处理复核（2026-08-29）

- 对 `messages_batch_glm_40_partial_edit_merged_v4` 完成四分片 reasoning 脱敏，输出 `ABC_J/agent_training/messages_batch_glm_40_partial_edit_redacted_v2/`；8,918/8,918 写入成功，失败 0，私有审计原样保留。公开消息校验 `valid=true`，公开/私有 ID 对齐。
- 发现 26 条新候选含“纯 reasoning assistant 紧接 tool-call assistant”的连续 assistant 记录。更新 `train/scripts/export_sft_dataset.py`，在预处理层合并相邻 assistant，保留全部 reasoning 与 tool call 顺序，以满足 LLaMA-Factory 交替规则；不修改原始轨迹。导出到 `train/data/guqin_agent_sft_v2_partial_edit_redacted_v3/`，SFT 校验 `valid=true`。
- 当前脱敏 SFT 共 8,918 条（Fingering 4,714、Guqinizer 4,204），assistant turns 14,694、tool-call blocks 14,714。Qwen3.5 tokenizer 统计：P50=3,302.5、P90=5,314.3、P95=5,627、P99=6,361.8、max=9,417；超过 2,048 的 8,468 条，超过 8,192 的 8 条，超过 16,384 的 0 条。最长为第 366 行 `S34JRKE9-p0020-fingering_agent-teacher-tools`。
- 与此前 `not_editable_redacted_v2`（P50=3,300.5、P95=5,592、P99=6,170.7、max=8,142）相比，中位数基本不变、尾部略变长；局部提交修复主要改善协议成功率，不会自动缩短已有上下文。新目录是可训练候选，但若受 8,192 上限约束仍需处理这 8 条超长样本。

## 7.91 `get_pitch_candidates` 输出精简（2026-08-29）

- 候选表的来源由“序号＋重复简谱标签”改为“来源序号＋去重简谱值”（例如 `目标音高｜MIDI 74｜来源｜452、457、458、460、463、470｜简谱｜6`）；表头中的误差列移除；实得 MIDI 统一显示 1 位小数（例如 `74.0`）。
- 同步更新教师轨迹生成器与运行时 `tool_broker`，候选计算、容差判断和内部审计仍保留完整数值，不影响音高校验；`top_candidate` 只保留模型后续决策所需的 mode/string/hui/sounding_midi 字段。
- 教师质量、框架、工具协议与 GQS 测试共 72 项通过。

## 7.92 已生成轨迹的候选表文本压缩脚本（2026-08-29）

- 新增 `scripts/compact_pitch_tool_results.py`：输入旧版 `messages_train.jsonl`，仅重写 tool 消息 JSON 中的 `result.text`；工具参数、候选对象、私有审计和其他消息均不修改。旧来源表会压缩为“来源序号＋去重简谱”，候选表删除误差列并将实得 MIDI 保留 1 位小数。
- 脚本采用输入/输出分离，先写新文件并打印变更行数，避免覆盖原始轨迹。示例：`python scripts/compact_pitch_tool_results.py --input <old>/messages_train.jsonl --output <new>/messages_train.jsonl`。
- 已在当前脱敏候选上试跑：4,683 条消息轨迹、4,767 个 tool 消息发生文本压缩；工具调用参数和内部审计仍保持原样。新生成轨迹则直接使用运行时的精简格式，无需再经过此脚本。

## 7.93 已生成轨迹压缩后的 token 复测与最长轨迹视图（2026-08-29）

- 将 `messages_batch_glm_40_partial_edit_redacted_v2/messages_train.jsonl` 以输入/输出分离方式压缩为候选表精简版，临时输出位于线程可视化目录的 `messages_train_compact_test.jsonl`；随后按同一完整多轮 SFT 预处理导出到 `compact_sft/`，未覆盖正式训练目录。
- 精简后 8,918 条的 Qwen3.5-9B chat-template token 分布：P50=3,131.5、P90=4,620.3、P95=4,880.15、P99=5,490.32、max=8,612；超过 2,048 的仍为 8,468 条，超过 8,192 的 1 条，超过 16,384 的 0 条。Fingering：P50=4,173、P95=5,107.75、max=8,612；Guqinizer：P50=2,446、P95=2,937.85、max=4,384。
- 与压缩前正式 SFT（P50=3,302.5、P95=5,627、max=9,417）相比，P50 减少 171、P95 减少 746.85、最大值减少 805；超过 2,048 的数量未变，说明压缩主要改善长尾而不是让大多数样本跌破 2,048。
- 修复了压缩脚本将含有“｜简谱｜”的来源表头误识别为候选行的问题；例如 `451（简谱1（饰））` 现在正确输出为 `来源｜451｜简谱｜1（饰）`，不会再变成 `451.0｜1（饰）`。全量 31,477 个来源表头扫描未发现残留的十进制来源序号。
- 精简后最长 8 条轨迹可视化：`C:\Users\30343\\.codex\\visualizations\\2026\\08\\27\\01a04324-2f64-76f1-9983-9acfa9c17711\\longest-trajectories-compact.html`；token 排名首条为第 366 行 `S34JRKE9-p0020-fingering_agent-teacher-tools`（8,612 tokens）。

## 7.94 4×RTX 3090 显存 smoke（2026-08-30）

- 使用转换后最长轨迹 `S34JRKE9-p0020-fingering_agent-teacher-tools`（8,612 tokens）做 4 卡、batch=1、单步测试，依次覆盖 cutoff 2,048/4,096/6,144/8,192，并分别测试 `use_unsloth_gc=false/true`。
- 非 offload 的峰值显存（各卡最高值）分别为 22,569/22,569/23,411/23,891 MiB；offload 分别为 22,581/22,585/22,583/22,759 MiB。所有配置训练主体均完成；8,192 非 offload 在销毁 NCCL 进程组时出现 CUDA OOM 警告（训练指标已产生但不算干净通过），8,192 offload 无该警告。
- 日志中的单步耗时（`1/train_steps_per_second`）为：非 offload 20.83/45.45/50.00/52.63 秒，offload 20.83/23.26/26.32/29.41 秒。每组首次运行包含 Triton/模型冷启动，offload 看起来更快不能直接解释为加速；但从显存安全和 8,192 可运行性看，offload 明显更稳。
- 汇总结果保存在 `C:\Users\30343\\.codex\\visualizations\\2026\\08\\27\\01a04324-2f64-76f1-9983-9acfa9c17711\\vram_smoke_results.tsv`；测试配置为 `train/server/vram_smoke_template.yaml`，串行 runner 为 `train/server/run_vram_smoke.sh`。

## 7.96 再作省略音的当前段渲染修复与定向重跑（2026-08-30）

- 发现旧 `teacher_trajectory.py` 无论当前段还是只读前段，只要音符带 `notation_omitted=true` 就强制渲染 `无（由于是再作部分，省略）`，并在当前段追加“再作省略音”说明；这会把尚未由 Agent 判断的当前段状态提前泄露。
- 已修复为：只有 `readonly=true` 的前段才显示再作省略标记和说明；当前段统一按普通待编辑行显示 `[减字待填写]`，不再预判“再作”。新增回归测试 `test_current_repeat_copy_does_not_prelabel_omission`；教师质量测试 41/41 通过。
- 新增 `scripts/find_current_repeat_placeholder_ids.py` 扫描受影响轨迹，当前候选共定位 361 个轨迹（686 条阶段消息）。使用 GLM-4.5、允许原始私有 reasoning、4 分片重跑：首轮因默认 `limit=2` 实际只跑 8 个轨迹；随后排除已完成项补跑其余 353 个轨迹。Fingering 361 条全部成功，Guqinizer 新结果 324 条成功；39 条无新 Guqinizer 结果，其中多数是新 Fingering 推导后的无操作，另有 1 条协议失败。对 3 条异常 Guqinizer 再定向重试，成功 2 条；不把旧 Guqinizer 与新 Fingering 混拼。
- 原始覆盖候选为 `ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_merged_v2/`；脱敏后剔除 7 条只有 system/user/assistant、无工具回合的无操作 Guqinizer，最终公开候选为 `ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_redacted_v2/`，共 8,970 条（Fingering 4,714、Guqinizer 4,256），当前段再作标记残留扫描为 0，`validate_teacher_agent_messages.py` 为 `valid=true`。
- 完整多轮 SFT 导出到 `train/data/guqin_agent_sft_v2_repeat_prompt_redacted_v1/`，8,970 条；`validate_sft_export.py` 为 `valid=true`。原始私有候选和重试失败审计均保留，公开训练只使用脱敏 SFT。

## 7.95 Fingering-only 片段的 Guqinizer 补跑与脱敏（2026-08-30）

- 在当前 `messages_batch_glm_40_partial_edit_merged_v4/` 中核对到 4,714 条 Fingering 与 4,204 条 Guqinizer：516 条表面上缺 Guqinizer。用当前 Fingering 私有审计中的 `accepted_plan` 重新推导参考差异后，454 条实际为 `NO_OP`，只有 62 条存在需要 Guqinizer 处理的目标；新增 `scripts/prepare_guqinizer_retry.py` 自动重建 4,714 条中间体并生成去重后的 62 条清单。
- 62 条使用 GLM-4.5、4 个隔离分片、最多 5 次尝试，显式开启 `--allow-private-reasoning-leakage`：首轮因并行脚本每个分片默认 `limit=2` 只成功 8 条；排除这 8 条后补跑剩余 54 条，全部成功。原始候选合并到 `ABC_J/agent_training/messages_batch_glm_40_partial_edit_merged_v5/`，共 8,980 条（Fingering 4,714、Guqinizer 4,266）。
- 脱敏严格在补跑完成后进行。为避免重复调用，使用已脱敏的 `messages_batch_glm_40_partial_edit_redacted_v2/` 覆盖旧样本，仅对 62 条新 Guqinizer 原始 reasoning 做四分片脱敏；脱敏工作目录为 `messages_batch_glm_40_partial_edit_redaction_workers_v3/`，合并后为 `messages_batch_glm_40_partial_edit_redacted_v3/`。
- 脱敏后的校验发现 2 条新 Guqinizer 只有 system/user/assistant、没有工具回合（模型错误地输出了无操作结果）：`S3CDXQNy-p0002-guqinization-teacher-tools`、`S3JNAahW-p0006-guqinization-teacher-tools`。已从 raw 与公开候选中剔除，最终脱敏目录为 `ABC_J/agent_training/messages_batch_glm_40_partial_edit_redacted_v4/`，8,978 条，Fingering 4,714、Guqinizer 4,264；`validate_teacher_agent_messages.py` 为 `valid=true`。
- 完整多轮 SFT 已导出到 `train/data/guqin_agent_sft_v2_partial_edit_redacted_v4/`：8,978 条，assistant turns 14,764，SFT 校验 `valid=true`。454 条 NO_OP 片段没有 Guqinizer 是预期行为，不再重复调用；两条无工具误输出已保留在原始重试审计中但不进入训练集。

## 7.97 Guqinizer 进入条件改为减字文字差异（2026-08-30）

- 复核发现“无 Guqinizer”并不等于 Fingering 文本与标注一致：旧选择逻辑只对 `verified/weak` 的结构化参考字段运行 `infer_minimal_patches()`，大量含 `绰`、复合技法或文字层级差异的标注因被标为 `conflicting/unusable` 而被误判为 no-op。
- 新增 `infer_jianzi_text_patches()`，Guqinizer 是否需要运行现在直接比较 baseline 与参考的 `jianzi_text`/`text`（保留空白和中文/阿拉伯数字的无害归一化）；非空文字差异即生成 `SET_JIANZI_TEXT` 目标，不再依赖 mode/string/hui/左右手等结构化字段。只有不安全的空参考不会触发清空。
- `generate_teacher_tool_trajectories.py` 的续跑完成判定和 Guqinizer target 选择、`prepare_guqinizer_retry.py` 的补跑清单均已切换到文字差异逻辑。新增回归测试覆盖 `大指七徽勾4弦`→`绰大指七徽勾4弦` 以及不安全空参考保护；教师质量/逆向差异测试 55 项通过（guqin-agent 环境）。
- 该修复只改变后续生成/补跑的选择条件，尚未自动重跑当前 8,970 条公开候选；现有 `no-guqinizer-fingering-v2.html` 中“不对应”现象正是旧选择条件留下的结果。若要补齐，应按新文字差异清单重跑缺失 Guqinizer，再做脱敏和合并。

## 7.98 按减字文字差异补跑 Fingering-only Guqinizer（2026-08-30）

- 用 `scripts/prepare_guqinizer_retry.py` 基于最新 Fingering 中间体重算目标，得到 39 条确有非空 `jianzi_text` 差异、此前却没有 Guqinizer 的轨迹；425 条仍为文字层面的无操作，不再重复调用。
- 首次批量续跑看似 39 条全部连接失败，实际是 `--resume` 将中间输入中 425 条无操作样本计入全局 `limit`，导致没有选中重试池，只沿用了旧失败报告。已修复生成器的断点计数：`limit` 只作用于过滤后的候选池；单条和双条探针均验证通过。
- 使用 GLM-4.5、串行请求、最小间隔 3 秒、最多 5 次尝试和指数退避，39/39 条 Guqinizer 全部成功，0 条协议/连接失败。输出目录：`ABC_J/agent_training/messages_text_jianzi_guqinizer_retry_v1/`。
- 已按 sample ID 后写覆盖合并回最新原始候选：`ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_text_jianzi_merged_v1/`，共 9,009 条（Fingering 4,714、Guqinizer 4,295），公开/私有 ID 对齐。该目录仍是原始私有 reasoning 候选，进入训练前需按既定流程脱敏并重新导出 SFT。

## 7.99 映射谱减字稀疏/尾部空值审计（2026-08-30）

- 新增 `scripts/audit_mapped_jianzi_quality.py`，对训练源 176 个 score key 的 `jianpu_jianzi_mapped.md` 统计非空减字比例和末尾连续空减字。小节线采用转义 `\|`，审计脚本已按 Markdown 转义规则解析，避免把小节线误计为空减字。
- 非空比例严格低于 50% 的 13 个谱建议整谱剔除；非空比例至少 50% 但末尾连续空减字不少于 10 音的 11 个谱建议截断尾部；其余 152 个谱暂保留。修正转义小节线解析后，`SMcokJse` 为 1059/1696（62.4%）且无连续空尾，不属于剔除对象；`SkYX92CP` 为 6/1245（0.5%），属于明显稀疏源。
- 当前只生成非破坏性审计报告，未覆盖原始 `final/`、`round2/final/` 或现有轨迹：`ABC_J/agent_training/source_quality_audit_v1/quality_report.md` 与 `quality_report.json`。报告给出 11 个尾部截断的最后非空序号和第一段连续空值序号；执行清洗前需据此重建受影响的 GQS/推理片段，不能直接改写原始谱文件。

## 7.100 映射谱质量过滤后的 GQS 与推理重建（2026-08-30）

- 按 §7.99 的规则生成了非破坏性质量过滤源：`ABC_J/agent_training/gqs_quality_filtered_v1/`。原始
  `ABC_J/agent_training/gqs/`、`final/`、`round2/final/` 和既有轨迹均未改写。
- 13 个非空比例低于 50% 的谱整谱剔除：`S2M83aBA`、`S9GH7RvP`、`SAZ9S3Na`、`SPHtJJ6X`、
  `SeWok3pZ`、`SfcEnVgP`、`SgKMecQx`、`ShXKQjut`、`Sk0YhP5A`、`SkYX92CP`、`Sq6pbetG`、
  `SurzVQ02`、`SxUPcN1Q`。11 个非空比例达标但连续空尾不少于 10 音的谱按审计报告的最后非空
  序号截断；共移除 GQS 音符行 747 行。具体截断位置及原始统计见
  `ABC_J/agent_training/source_quality_audit_v1/quality_report.json`，执行记录见同目录
  `filter_application_report.json`。
- 过滤后的清单为 `ABC_J/agent_training/source_quality_audit_v1/dataset_split_groups_filtered.csv` 与
  `agent_dataset_splits_filtered.csv`（各 228 行）。
- 基于过滤后 GQS 重建推理数据到 `ABC_J/agent_training/inferred_quality_filtered_v1/`：228 首、
  train/validation/test = 4,345/654/1,405，共 6,404 段；`validate_agent_trajectories.py` 报告
  `valid=true`，无重复或 split 泄漏。相较旧 `inferred_v3`，训练段减少 394 条，来自整谱剔除和
  尾部截断后的重新分节，而不是删除旧轨迹文件。
- 这只是新的推理源和中间数据，尚未重新调用教师模型。后续教师轨迹必须从
  `inferred_quality_filtered_v1` 重新生成；不能把受影响 24 个 score key 的旧教师轨迹与新源混用。

## 7.101 质量过滤的低成本合并版本（2026-08-30）

- 为避免无必要地重建全部 phrase，新增 `scripts/merge_quality_filter_minimal.py`，以旧
  `inferred_v3` 为主体执行非破坏性筛选：删除 13 个整谱的 361 个训练 phrase，删除 11 个
  截断谱中完全位于空尾后的 33 个 phrase，仅用过滤后重建结果替换 11 个跨截断边界的 phrase。
- 输出为 `ABC_J/agent_training/inferred_quality_filtered_minimal_v1/`，与完整重建版相比拥有完全
  相同的 6,404 个 trajectory ID（train/validation/test = 4,345/654/1,405），且
  `validate_agent_trajectories.py` 报告 `valid=true`。合并细节见该目录的
  `minimal_merge_report.json`。
- 后续教师模型重跑只需覆盖 11 个跨截断边界的 phrase（来自 11 个 score key）；13 个整谱和
  33 个尾部 phrase 已直接删除，不需要调用模型；其余 152 个谱的旧教师轨迹可以复用。精确的
  11 个待重跑 trajectory ID 见 `ABC_J/agent_training/source_quality_audit_v1/crossing_teacher_trajectory_ids.txt`。

## 7.102 跨边界教师轨迹低成本重跑（2026-08-30）

- 使用 `inferred_quality_filtered_minimal_v1` 定向重跑清单中的 11 条 Fingering，GLM-4.5
  全部成功；输出原始候选为 `ABC_J/agent_training/messages_quality_filter_minimal_retry_v2_fingering/`。
- 基于新的 Fingering 中间结果继续跑 Guqinizer：10 条成功，`S8Tq46CE-p0021` 经新 Fingering
  判断为无 Guqinizer 目标，因此没有沿用旧 Guqinizer 结果。输出为
  `ABC_J/agent_training/messages_quality_filter_minimal_retry_v2_guqinizer/`。
- 通过 `merge_teacher_retry_outputs.py` 将 21 条新阶段结果覆盖回旧候选，并移除该条过期的
  `S8Tq46CE-p0021-guqinization-teacher-tools`，得到
  `ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_text_jianzi_quality_filtered_minimal_merged_v1/`：
  共 9,008 条（Fingering 4,714、Guqinizer 4,294），公开/私有 ID 对齐。该目录仍含原始私有
  reasoning，不能直接训练；需按既定流程脱敏后再导出 SFT。
- 首次误用默认 `--limit=2` 的连接失败审计保留在
  `messages_quality_filter_minimal_retry_v1_fingering/`，不计入最终合并结果；后续重跑已显式
  `--limit=11` 并使用网络授权完成。

## 7.103 教师候选按最小过滤推理集对齐（2026-08-30）

- 发现 7.102 的中间合并目录仍包含被删 phrase 的旧 teacher rows，已用
  `scripts/filter_teacher_trajectories_to_inferred.py` 按
  `inferred_quality_filtered_minimal_v1` 的 trajectory ID 做最终筛选。
- 最终原始教师候选为
  `ABC_J/agent_training/messages_batch_glm_40_repeat_prompt_text_jianzi_quality_filtered_minimal_final_v2/`：
  8,552 条 accepted teacher rows（Fingering 4,322、Guqinizer 4,230），公开/私有 ID 对齐；
  被过滤掉的 450 条来自已删除的整谱/尾部 phrase，另剔除 6 条历史遗留的孤立 Guqinizer
  rows（没有对应 Fingering）。该目录仍含私有 reasoning，必须先脱敏。
- `S8Tq46CE-p0021` 的新 Fingering 没有 Guqinizer 目标，因此最终只保留 Fingering；没有把旧的
  Guqinizer 结果带回。原始重试目录和过滤报告均保留。

## 7.104 无 Guqinizer 阶段复核（2026-08-30）

- 使用 `scripts/prepare_guqinizer_retry.py` 对最终候选中的 4,322 条 Fingering 重新计算文字差异：
  92 条没有 Guqinizer，全部为 `no_op`，可确认不需要补跑；没有发现 actionable 缺口。
- 92 条清单和判定保存在 `ABC_J/agent_training/source_quality_audit_v1/guqinizer_gap_review_v1/selection_report.json`，
  可直接查看 `no_op_ids`。孤立 Guqinizer 清单为
  `orphan_guqinizer_sample_ids.txt`，已从最终 v2 候选移除。
  `inferred_quality_filtered_v1` 保留作为完整重建对照，不作为必须使用的版本。

## 7.105 no-op 复核轨迹、显式停止回合与末尾工具结果保留（2026-08-30）

- no-op 生成提示改为只要求教师复核当前稿并说明为何可直接保留；不再要求教师声明“没有私有标注”，也不再要求输出 JSON 或 `tool_calls=[]`。no-op 教师请求不附带私有 GQS，最终消息为普通 assistant 复核意见。
- 92 条 no-op 使用 GLM-4.5、6 并发分片生成，最终 92/92 成功；原始结果保存在
  `ABC_J/agent_training/messages_guqinizer_noop_public_v5_raw/`。抽样可视化为
  `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/guqinizer-noop-review-v5.html`。
- 为避免模型学习不到“工具成功后停止”，`generate_teacher_tool_trajectories.py`、
  `validate_teacher_agent_messages.py` 和 `train/scripts/export_sft_dataset.py` 已支持完整结构：
  `assistant(tool_call) → tool(result) → assistant(final)`；历史末尾只有 tool result 的 accepted rows 在预处理时追加阶段化停止句，不再删除 tool observation。明确的 no-op 轨迹允许 `system → user → assistant(final)`。
- 6 条实际存在文字差异的 Guqinizer 缺失阶段使用 6 并发重跑，6/6 成功；reasoning 脱敏 5 条由 GLM 完成、1 条由本地词汇兜底完成，最终 6/6 脱敏成功。合并候选为
  `ABC_J/agent_training/messages_batch_glm_40_complete_stops_v2/`，共 8,656 条，
  Fingering 4,328、Guqinizer 4,328；公开消息校验 `valid=true`。
- 完整多轮 SFT 导出为 `train/data/guqin_agent_sft_v2_complete_stops_v1/`，共 8,656 条，
  assistant turns 22,733、tool-call blocks 14,108；SFT 校验 `valid=true`。导出器现在保留工具 observation，且允许 no-op 样本无工具调用。
- 6 条缺失 Guqinizer 的 ID 为 `S3JNAahW-p0025`、`S4nhaTGN-p0019`、
  `SAyqwdky-p0006`、`SQzJKZZ8-p0069`、`SrbyFUb1-p0043`、`Sxq5d2F6-p0025`。
- 新增辅助脚本：`scripts/append_terminal_assistant.py`、
  `scripts/merge_noop_and_terminal_replies.py`、`scripts/repair_redaction_failures_local.py`。
  `train/scripts/validate_sft_export.py` 不再把普通“标注”一词本身当作私有泄露；仍保留对 GQS、教师提示和私有参考等来源词的检查。

## 7.106 空标注目标与全空段策略修正（2026-08-30）

- 旧 `infer_jianzi_text_patches` 会跳过 `jianzi_text=None`，且会按解析可信度跳过一部分空字符串；
  这会把“Base 写了具体减字、标注要求此行不显示”的情况误判为无差异。现统一把普通 `None`/`""`
  视作明确的空显示目标，允许 Guqinizer 用 `edit_plan.jianzi_rows=[[序号,""]]` 清除由前一吟、猱或
  复合多声减字覆盖的后续行。差异判断只看减字文字，不再用结构化取音字段决定空值是否有效。
- 再作物化行是例外：教师私有参考表现在明确显示
  `[无（由于是再作部分，省略）]`，不会显示成 `[空]`；当前待编辑段初始仍显示 `[减字待填写]`。
  教师私有规则明确说明该标记本身是合法减字文字，Base/Guqinizer 均可通过
  `edit_plan.jianzi_rows` 直接填入；它表示沿用已写出的再作动作，不等于空字符串，也不删除声音。
  小节线不显示 `[空]`。
- 新增 `is_all_empty_reference_phrase`：若整段没有任何非空标注，且不含再作物化省略行，则在任何
  Base/Guqinizer API 调用前整段排除。`prepare_guqinizer_retry.py` 同样跳过此类段。
- 对 `messages_batch_glm_40_complete_stops_v2` 审计：4,328 个 phrase 中有 50 个当前在册的全空段，
  已在新本地候选 `ABC_J/agent_training/messages_batch_glm_40_blank_policy_v1/` 成对剔除；新候选为
  8,556 条（Base/Fingering 4,278、Guqinizer 4,278），公开/私有 ID 对齐且消息校验 `valid=true`。
- 空目标精确审计保存在 `ABC_J/agent_training/blank_reference_audit_v1.json`：保留段中 2,800 个
  Base 稿在空标注行写了具体减字；已有 Guqinizer 修掉一部分，仍有 2,211 个最终稿存在此类差异。
  这些 2,211 个 ID 后续只需重跑 Guqinizer，再统一脱敏和覆盖合并；无需重跑 Base。当前尚未发送
  该批数据到 GLM。
- 新增辅助脚本：`scripts/audit_blank_reference_targets.py`、
  `scripts/prune_all_empty_trajectory_phrases.py`。相关单元测试 47/47 通过。
- 用户确认不重跑“Guqinizer 已运行但最终仍未按空标注清除”的 2,211 条；该统计仅保留作审计，
  不作为强制覆盖目标。实际重跑范围为 357 个再作私有提示受影响段（Base+Guqinizer）和 4 个由
  新空值比较首次识别出的旧 no-op（仅 Guqinizer）。357 段已用更新后的私有规则、GLM-4.5、
  6 并发启动于 `ABC_J/agent_training/messages_repeat_private_marker_rerun_v2_shards/`；此前 v1
  进程已停止，不能用于合并。当前测试为 47/47 通过。

## 7.107 再作语义纠正：起点与现有复现区（2026-08-31）

- 纠正 §5 中“把重复段复制进音高流”的旧表述：原始谱面已经包含复现音的音高与时值，省略的是复现区的减字文字；展开器不得插入合成音符，也不得改变序号或 phrase 边界。
- 语义固定为：`〔再作起点〕` 表示 `ㄱ` 起点，仅记录锚点；`从ㄱ再作` 从该锚点开始复现，复现标记行及其后连续空 `jianzi` 行，直到下一个非空减字，标记为 `notation_omitted=true`；单独的 `再作` 表示刚演奏部分的复现，从该行之后连续空减字行标记到下一个非空减字。小节线、延音和休止仍保留在音高/时值流中，不删除。
- 已修复 `agents/abc_to_jianzipu/repeat_materializer.py`：不再 append clone；保留原 index；对无法找到 `再作起点` 的 `从ㄱ再作` 只记录 `unpaired_markers`，不凭空猜测省略范围。新增 3 个回归测试，repeat/teacher 相关测试共 69/69 通过。
- 新推导输出为 `ABC_J/agent_training/inferred_v9_repeat_semantics/`。以旧 `inferred_v6` 对比，phrase 总数由 6,798 变为 6,741；1,086 个 trajectory 的当前行或输入范围发生变化，其中 566 个新增再作省略行，618 个输入音符范围发生变化。待重跑 ID 已保存至 `ABC_J/agent_training/repeat_semantics_rerun_ids_v1.txt`，差异审计为 `ABC_J/agent_training/repeat_semantics_diff_v2.json`。
- 具体核验：`Sx4mbdGp-p0012` 的 240 为单独 `再作`，其后 241–252 标为再作省略；253 为 `从ㄱ再作`，其后连续空减字至 278 标为再作省略；不再只标 253，也不把 253–278 生成成额外音符。教师轨迹需以 v9 推导重新生成后再合并，旧轨迹不能直接沿用。
- 已获用户明确授权将当前候选中受影响的 632 条发送到 GLM 重跑。6 路批量目录为 `ABC_J/agent_training/messages_repeat_semantics_rerun_v3_shards/`，使用 `glm-4.5`、每片段断点写入；先保留原始 reasoning，完成后再做统一脱敏。批量及失败重试现已结束：实际匹配到 627 个 source phrase，累计 accepted 1,125 条阶段记录（Fingering 621、Guqinizer 504），仍有 82 条 Guqinizer 因模型 JSON 包络/最终 JSON 协议失败，另有 11 条被离线音高过滤隔离。失败明细保存在各 shard 的 `generation_report.json` 与 `teacher_rejected_io.jsonl`；探针 `Sx4mbdGp-p0012` 已单条成功，不计入批量结果。该目录仍是未脱敏原始结果，尚未合并到正式训练集。

## 7.108 再作重跑失败样本诊断（2026-08-31）

- 计划 ID 清单为 632 条，但其中 5 条没有进入 API checkpoint：`S4nhaTGN-p0012`、`S4nhaTGN-p0020`、`S4nhaTGN-p0031`、`SaljUbT2-p0105`、`SqY0ZGAm-p0097`。它们在 v9 推导中均为整段标注减字为空（非空标注数为 0），被“全空段不生成两阶段轨迹”策略有意跳过，不是文件丢失；其中前三段分别为《大胡笳》节本，另两段为《秋鸿》和《大胡笳》。
- 为人工判断 82 条 Guqinizer 失败是否属于数据异常，生成私有诊断页：`C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/guqinizer-82-failures-with-annotation.html`。页面逐段显示 Guqinizer 收到的 Base 当前减字、私有标注和最终 5 次重试响应；该页含私有信息，不得用于训练。
- 82 条先重试 5 次后，又按新版提取器重试至每条最多 10 次；第二轮已结束。最终累计 accepted 1,155 条阶段记录（Fingering 621、Guqinizer 534），Guqinizer 从 504 增至 534，剩余失败从 82 降至 52，52 条仍为 Guqinizer 的 JSON 包络协议失败。6 路日志为各 shard 的 `retry10.log`，失败明细已刷新到 `generation_report.json` 和 `teacher_rejected_io.jsonl`。
- 高重试后的 52 条剩余失败诊断页为 `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/guqinizer-remaining-52-with-annotation.html`，包含 Base/标注逐序号对照及每条最终重试响应。

## 7.109 教师 JSON 标签回退修复（2026-08-31）

- 定位到一个提取器误判：`generate_one` 在解析前会把换行压成空格，而旧 `recover_labeled_teacher_envelope` 只接受行首换行形式的 `tool_calls：`；因此 `decision_summary：…… tool_calls：[...]` 会回退到首个调用对象并报顶层字段错误。
- 现已改为锚定 `decision_summary`、允许空白分隔的标签回退，并对合法 JSON 字符串形式的 `arguments` 对象做无损解码；任意损坏数组、额外字段和无 reasoning 的 calls-only 仍拒绝。协议测试 20/20 通过。此前 10 次重试产生的 52 条失败尚未用此修复重新发送。
- 已用修复后的提取器对剩余 52 条 Guqinizer 再重试一轮（6 路并发、每条最多 5 次），全部成功；当前目录汇总为 accepted 1,207 条阶段记录（Fingering 621、Guqinizer 586），rejected=0，另有 11 条离线音高过滤隔离。该结果仍保留原始 reasoning，尚未脱敏或合并。

## 7.110 重跑后工具调用格式审计（2026-08-31）

- 1,207 条 accepted 阶段记录中的 2,308 个工具调用均已规范化为 `assistant.tool_calls[]` 的 OpenAI 兼容结构：`id`、`type=function`、`function.name`、`function.arguments`；`arguments` 全部为 JSON 字符串，2,308 个 tool observation 均紧随对应调用且内容为字符串。
- 工具调用计数：`edit_plan` 1,584、`get_pitch_candidates` 723、`list_context` 1、`expand_context` 0；未发现旧式顶层 `tool_turn/tool_calls/decision_summary` 或非字符串 arguments。
- 消息序列仍有 24 条 Fingering 含相邻两个 assistant 回合（前一条是无工具的补充 reasoning，后一条才调用工具），不是工具调用字段不一致；42 条 Guqinizer 为正常 no-op（无工具调用）。如需严格交替的训练序列，导出前应把相邻 assistant 内容拼接，而不是修改工具调用对象。

## 7.111 no-op 标注来源复核与全空段清理（2026-08-31）

- 92 条 no-op 的私有审计记录没有携带参考 GQS；因此旧查看器中的“标注减字”列为空，不能据此判断真实标注为空。已改用修正再作语义后的 `inferred_v9_repeat_semantics` 逐段复核，缺失的 1 个 ID 用 `inferred_v6` 回退核对。
- 92 条中有 44 段真实参考减字全空；其中 28 段存在再作标记（`再作`、`从ㄱ再作`、`〔再作起点〕` 等）并保留，作为再作省略的合法空显示。其余 16 段既无非空参考也无再作标记，判定为无标注质量问题，已从 Base/Fingering 与 Guqinizer 成对移除：`S3rVwfH0-p0073`、`SQzJKZZ8-p0007`、`SZh5gUkP-p0005`、`SZkHNtAN-p0018`、`SaDRc7nJ-p0014`、`Sc9dDref-p0013`、`SddZqhN8-p0006`、`SfR5DDMK-p0031`、`Sh8BUAQh-p0011`、`Sh8BUAQh-p0018`、`ShSCyERN-p0002`、`Sj7n7ZM8-p0006`、`Sj7n7ZM8-p0020`、`SjgRh7RG-p0006`、`SjgRh7RG-p0058`、`Sxq5d2F6-p0011`。
- 复核明细保存于 `ABC_J/agent_training/noop_92_target_audit_v9.json`。另有 48 段真实参考非空，其中 32 段与 Base 存在文字差异；它们不是“全空段”，本次不删除，后续应作为普通 Guqinizer 质量候选单独处理。
- 清理后的原始候选为 `ABC_J/agent_training/messages_batch_glm_40_complete_stops_v3_filtered/`：8,624 条（Fingering 4,312、Guqinizer 4,312），公开/私有 ID 对齐，`validate_teacher_agent_messages.py` 为 `valid=true`。对应完整多轮 SFT 导出为 `train/data/guqin_agent_sft_v2_complete_stops_v2_filtered/`，同样 8,624 条，SFT 校验通过。原始 v2 与 v1 导出保留不覆盖，便于回溯。

## 7.112 no-op 数量相等不代表 Guqinizer 已补跑（2026-08-31）

- `Fingering=4,312、Guqinizer=4,312` 是阶段配平结果：此前缺失的 Guqinizer 阶段用 92 条 no-op 记录补齐，后续又补入 6 条，因此数量被人为配平，并不表示这些片段都经过了正常 Guqinizer 修改流程。
- 在保留下来的 no-op 中，48 段真实参考非空；其中 32 段与 v9 Base/参考存在差异，但当前 Guqinizer 仍是 `termination=no_changes`、无工具调用，**尚未补跑**。这些 32 段不能视为已完成的 Guqinizer 监督样本，应在正式训练前单独重跑或剔除。
- 因此 `messages_batch_glm_40_complete_stops_v3_filtered` 与 `guqin_agent_sft_v2_complete_stops_v2_filtered` 目前是清理后的候选，不应标记为最终训练集，直到这 32 段完成处理。

## 7.113 no-op 复核修正（2026-08-31）

- 先前“24 段真正 no-op”的结论和对应查看器不可信：`classify_noop_phrases_v9.py` 原先只比较 `set(target) ∩ set(base)`，参考表省略整段行时会把“参考缺行、Base 有整段文字”误判为相等。这正是查看器中出现成段“标注空/Base 有字”和“Base 空/标注有字”的根因；不是 24 段本身都应当 no-op。
- 已改为比较完整索引域（`notes_without_jianzi ∪ reference ∪ Base`），并把查看器的空值状态拆成“双方均为空”“仅一方有文字”“双方文字一致”。此外发现旧 Base 中部分 phrase 的事件范围仍是旧版分段（例如 `SywB9Jym-p0042` 为 887–910，而 v9 参考同 ID 为 911–935），因此复核时必须优先使用 `messages_repeat_semantics_rerun_v3_shards` 的重跑 Base；直接把旧 Base 与 v9 参考拼在一起会制造整段错位。
- 使用重跑 Base 覆盖旧记录后，92 条中只有 8 段可保留为真正 no-op；15 段为无再作标记的全空目标，应按数据质量策略剔除；其余均不能再按 no-op 处理。按旧 24 段清单重新对齐后的诊断统计为 366 行：99 行规范化一致、40 行双方均有文字但不同、138 行仅 Base 有文字、5 行仅标注有文字、84 行双方均为空。
- 24 段问题的对齐诊断查看器（逐行显示 Base/标注/简谱）为 `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/noop-24-aligned-diagnostic.html`；真正 8 段的查看器为 `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/true-noop-aligned-v9-with-annotation.html`。两者均使用 v9 参考表，缺失 ID 才回退 v6。
- 判定明细为 `ABC_J/agent_training/noop_92_target_audit_v9_overlay.json`；修正后的筛选清单为 `ABC_J/agent_training/true_noop_8_aligned_report.json`。`make_noop_jianzi_visualization.py` 现在会从 `notes_without_jianzi` 补回简谱，并将双方均为空的行明确标记，避免把省略行误显示为“Fingering 未提交”。

## 7.114 no-op final-reply 合并与末尾回合检查（2026-08-31）

- 使用 `scripts/merge_noop_and_terminal_replies.py`，将修正后的 8 条 no-op final-reply（来自 `messages_guqinizer_noop_public_v5_raw`，不含私有标注）合并到 `messages_batch_glm_40_complete_stops_v3_filtered`，输出 `ABC_J/agent_training/messages_batch_glm_40_complete_stops_v4_noop_aligned/`。合并结果 8,624 条，Fingering 4,312、Guqinizer 4,312；8 条 no-op 已替换为对应 final-reply，未额外追加 terminal assistant（原候选末尾已经是 assistant）。
- 对合并后的 `messages_train.jsonl` 逐条检查：8,624/8,624 条轨迹最后一条消息均为非空 assistant 文本；其中 Fingering 4,312/4,312、Guqinizer 4,312/4,312，末尾没有 tool observation、tool-call assistant 或空回复。
- 此前一次直接调用 GLM 的 `review_noop` 重试因 APIConnectionError 全部失败，输出目录 `messages_guqinizer_noop_reasoned_v3_aligned` 仅保留失败记录，不纳入合并；本次使用已经成功生成的 v5 no-op final-reply，避免重复消耗 API。

## 7.115 脱敏边界确认（2026-08-31）

- `messages_batch_glm_40_complete_stops_v4_noop_aligned/messages_train.jsonl` 是公开训练消息，自动泄露检查通过（`valid=true`）；assistant 文本中未发现 GQS、教师私有、最终标注、参考答案等禁用来源词。
- 同目录的 `teacher_trajectory_audit.jsonl` 是有意保留的私有审计文件，包含 `annotation_gqs`、原始 teacher I/O 和私有 reasoning，**不是脱敏文件，不能作为公开训练输入或对外分发**。因此“全部数据都脱敏”不成立：公开训练文件已脱敏，私有审计文件仍保留原始信息。

## 7.116 全量轨迹分层查看器（2026-08-31）

- 新增 `scripts/visualize_all_trajectories_hierarchical.py`，输入当前公开训练消息、私有审计和对齐后的 inferred source，按“score key → phrase ID”两级折叠；phrase 展开后显示 Fingering、Guqinizer、标注四列对比，匹配格为绿色，单方有字/双方不同/双方均为空分别区分，并支持曲谱 ID、phrase ID、曲名搜索。
- 覆盖 4,312 个 phrase、163 个曲谱 ID、100,514 行对比；标记 77 个 no-op phrase。查看器路径：`C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/all-trajectories-hierarchical-v1.html`。

## 7.117 最新 v4 训练候选与 4 卡 smoke（2026-08-31）

- 最新公开训练候选为 `ABC_J/agent_training/messages_batch_glm_40_complete_stops_v4_noop_aligned/`，共
  8,624 条（Fingering 4,312、Guqinizer 4,312），其 SFT 导出为
  `train/data/guqin_agent_sft_v4_noop_aligned/`；导出及结构校验均 `valid=true`，源消息 SHA-256 为
  `f1d98f366e7a2dd7c213fa67923ad28eadb6f31ee879968e507a4e3c698430ff`。
- 仅上传公开导出目录和必要脚本到服务器 `~/guqin-agent/`；私有 `teacher_trajectory_audit.jsonl`、原始
  教师 I/O 和 `.env` 未上传。服务器端校验再次通过，导出数据 SHA-256 为
  `3be95e35b88b55ba89d758ca035929007cd37fa285f1aecda29b11aef3360a2b`。
- 服务器缺失的 `guqin-sft` 环境已按 `train/server/bootstrap_guqin_sft_env.sh` 从 `cip` 克隆，版本检查为
  LLaMA-Factory 0.9.6.dev0、Transformers 5.8.0；`cip` 未修改。
- 原单独 smoke 任务 `J00009` 已取消，避免重复；随后发现再作省略语义缺失，4 卡串行任务 `J00010` 也已在
  尚未启动时取消，避免错误候选进入训练。

## 7.118 再作省略参考传播修复（2026-08-31）

- 根因：`infer_agent_trajectories.py` 为避免把再作复现误当成新的 attack，曾将所有
  `notation_omitted` 行从 `reference_plan` 过滤；这同时删掉了文本层的“应清空减字”目标。
  `Sx4mbdGp-p0012` 因此被错误归为 `review_noop`，Guqinizer 没有收到省略目标，保留了 Base
  的具体减字。`render_annotation_gqs()` 又只读取原始 `jianzi_text`，对 `null` 不恢复省略语义，
  所以旧全量查看器的标注列也没有“无（由于是再作部分，省略）”。
- 修复：保留再作起始标记；对其后 `notation_omitted` 且原始减字为空的行注入
  `无（由于是再作部分，省略）` 文本目标（小节线除外），使 Guqinizer 生成清空/省略目标；
  `render_annotation_gqs()` 与全量查看器同步恢复该显示标记。新增回归核验显示
  `Sx4mbdGp-p0012` 的 254–259、261–264、266–272 均出现该目标，253 保留“从ㄱ再作”标记。
- 旧 v4 训练候选已标记为不可直接训练；需要用修复后的推导和教师轨迹重新生成受影响段，再重新
  脱敏、导出、校验和提交 4 卡 smoke。旧全量查看器仍可追溯；修复后的查看器为
  `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/all-trajectories-hierarchical-v2-repeat-fixed.html`。
- 旧数据的可视化已重新生成：覆盖 4,312 个 phrase、163 个曲谱 ID；Sx4mbdGp-p0012 的标注列现
  能显示省略占位，但其旧 Guqinizer 列仍保留原错误 Base 字样，这正是需要重跑而不是静态修补的证据。
- 原拟提交的 4 卡串行任务 `J00010`（`guqin-sft-v4-smoke-full`）已在尚未启动时取消，避免把含有再作省略语义错误的旧候选送入训练；当前没有训练任务在运行。

## 7.119 再作修复单条两阶段探针（2026-08-31）

- 使用修正后的推导输入，仅重跑 `Sx4mbdGp-p0012` 两个阶段，输出目录为
  `ABC_J/agent_training/_repeat_fix_single_sx4mbdgp_p0012_escalated/`；GLM-4.5 两阶段均成功，`accepted=2`、`rejected=0`。
- Base/Fingering 重新生成了基础减字；Guqinizer 收到修正后的私有参考后，提交了 `253=从ㄱ再作`，并将
  `254–259、261–263、266–272` 置为空字符串，保留声音和时值，符合再作省略语义。
- 单条检查页为
  `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/sx4mbdGp-p0012-repeat-fix-check.html`。
  该探针通过后，受影响片段应按相同规则重跑 Base → Guqinizer；旧 v4 候选仍不可直接训练。

## 7.120 受影响片段正式重跑（2026-08-31）

- 已用 `inferred_v10_repeat_semantics_prompt_final/inferred_trajectories_train.jsonl` 和修正后的提示词启动 6 路重跑，目录为
  `ABC_J/agent_training/messages_repeat_semantics_rerun_v4_prompt_updated/`；训练集命中 638 条，分片按 107/107/106/106/106/106 条分配。
- 每个片段均按 Base/Fingering → Guqinizer 顺序生成，GLM-4.5、断点写入、每次请求间隔 1 秒；全空且无再作语义的片段由生成器自动跳过。
- 新生成的公开轨迹会使用更新后的 Guqinizer 公开提示；合并现有未受影响轨迹前，还需统一替换其旧公开提示文本，再做脱敏、校验和可视化。
- 全量公开提示统一脚本为 `scripts/rewrite_public_prompts.py`：只改每条公开轨迹的第一条 system，不触碰 assistant、tool result 或私有审计。

## 7.121 Guqinizer 保留再作省略标记（2026-08-31）

- 发现 Guqinizer 会把 Base 已填写的 `无（由于是再作部分，省略）` 改成空字符串；旧校验把两者按等价目标处理，因此这类结果曾被错误放行。
- 已撤回公开提示中的新增约束，保持公开提示原样；仅在 Guqinizer 私有规则中加入最小约束：初稿已有该文字且与私有参考一致时必须保留，不得改为空字符串；若私有参考要求其他文字，再按逐音分析决定。
- `validate_jianzi_only(..., toward_reference=True)` 现在仅在 Base 省略标记与私有参考一致时触发硬校验 `repeat_omission_marker_cleared`，阻止 Guqinizer 清除该标记；参考要求其他文字时仍允许修改，不影响普通空字符串或 Base 阶段。
- 新增回归测试覆盖提示词和硬校验。后续重跑受影响的 Guqinizer 片段时，需使用该版本代码；旧批次中清除标记的结果不能直接作为最终训练样本。

## 7.122 再作标记清除问题重跑结果（2026-09-01）

- 从 v4 审计中按“Base=私有参考=`无（由于是再作部分，省略）`、Guqinizer 最终为空字符串”精确筛出 40 个片段、141 个原受影响音位；未把 Base 不同或参考不同的片段混入。
- 首轮并发重跑中 22 条成功，18 条因公开 reasoning 泄露私有参考而失败；按既定流程允许原始私有 reasoning 后重试，18 条全部成功。结果分别保存在
  `ABC_J/agent_training/messages_repeat_marker_preserve_rerun_v3/` 和
  `ABC_J/agent_training/messages_repeat_marker_preserve_rerun_v4_leak_allowed/`。
- 合计 40/40 条 Guqinizer 通过回放与质量检查；重跑后的最终计划中，符合“Base 与私有参考一致”的行均未再被清空，`repeat_omission_marker_cleared=0`。v4 的 18 条原始 reasoning 仍需在公开训练合并前统一脱敏。

## 7.123 再作标记重跑结果脱敏与 4 卡训练入队（2026-09-01）

- 将 `messages_repeat_marker_preserve_rerun_v5_merged` 的 40 条重跑结果与当前公开候选
  `messages_batch_glm_40_complete_stops_v4_noop_aligned` 合并时，发现 `ShXKQjut-p0003` 只有
  Guqinizer 重跑记录、当前候选没有对应 Base/Fingering，故保留在私有重跑目录但不纳入配对训练集；
  合并原始候选为 `ABC_J/agent_training/messages_repeat_marker_preserve_final_raw_v2/`，共 8,624 条，
  Fingering/Guqinizer 各 4,312 条。
- 对合并候选进行 4 分片 reasoning 脱敏；前 3 个分片及最后 1 条失败样本均通过断点重试，最终
  `messages_repeat_marker_preserve_redacted_v1/` 写入 8,624/8,624 条，脱敏失败 0；公开消息校验
  `valid=true`，私有审计仍单独保留，未进入上传数据。
- 完整多轮 SFT 导出到 `train/data/guqin_agent_sft_v4_marker_preserve_redacted_v1/`：8,624 条，
  Fingering 4,312、Guqinizer 4,312，assistant turns 22,670；公开消息校验和 SFT 导出校验均通过。
- 仅上传该 SFT 目录到服务器 `~/guqin-agent/train/data/guqin_agent_sft_v4_marker_preserve_redacted_v1/`，
  主数据 SHA-256 为 `b02fde41ced0a7e958769ada435ba31c866bf2e394664eb62610088fd20cf8c3`；未上传私有审计、
  原始教师 I/O 或 `.env`。
- 已提交 Jobber 任务 `J00013`（4 GPU，`guqin-marker-preserve-smoke-full`）。任务命令先运行真实 2 条
  smoke，成功后自动执行 full；截至记录时任务仍在队列中等待 4 张卡，不会抢占其他任务。

## 7.124 再作语义修复合并、脱敏与最终训练入队（2026-09-01）

- 复核发现 7.120 的 638 段再作语义重跑尚未并入 8,624 条候选；现已将其与 7.122 的标记保留重跑
  一并合并。5 个当前候选中不存在配对 Base/Fingering 的片段（10 条阶段记录）继续排除，最终原始合并候选为
  `ABC_J/agent_training/messages_repeat_semantics_final_raw_v2/`，公开/私有各 8,624 行，阶段各 4,312。
- 4 路 reasoning 脱敏完成；模型重试及本地保守兜底后仍有 1 条 Fingering（`SaBzwJPS-p0049`）无法消除
  “系统提示”私有来源词，因此与其对应 Guqinizer 成对剔除。最终公开脱敏目录为
  `ABC_J/agent_training/messages_repeat_semantics_redacted_final_v2/`，共 8,622 条（Fingering/Guqinizer
  各 4,311），`validate_teacher_agent_messages.py` 为 `valid=true`。
- 完整多轮 SFT 导出为 `train/data/guqin_agent_sft_v4_repeat_semantics_marker_redacted_v1/`，共 8,622 条；
  `validate_sft_export.py` 为 `valid=true`，assistant turns 22,829，主数据 SHA-256 为
  `461756054cd7c72195a67d74d16fca14b4d08cba57f1629bf8a25dfe6904e9a2`。
- 已上传仅含公开 SFT 文件的目录到服务器
  `~/guqin-agent/train/data/guqin_agent_sft_v4_repeat_semantics_marker_redacted_v1/`，远端 SHA-256 与本地一致；
  私有审计、原始教师 I/O 和 `.env` 未上传。
- 已取消使用旧数据的 `J00013`，并提交 `J00014`（4 GPU，`guqin-repeat-semantics-marker-smoke-full`）。
  它先执行真实 2 条 smoke，成功后自动进入 full；截至记录时仍在 Jobber 队列等待 4 张卡，不会抢占其他任务。

## 7.125 最终脱敏数据 4 卡 smoke 通过并开始 full（2026-09-01）

- `J00014` 的真实 2 条 smoke 已在 4×RTX 3090 上完成，3 个优化步无 OOM，训练 loss=0.6868。
- 全量 preflight、Qwen3.5 模板渲染和标签检查随后通过；正式 full 已启动，使用
  `~/guqin-agent/train/data/guqin_agent_sft_v4_repeat_semantics_marker_redacted_v1/`、
  `CUTOFF_LEN=18432`、4 卡 DDP、QLoRA、3 epoch、有效 batch 32。
- 当前 Jobber 任务仍为 `J00014`，full 进程正常运行中；输出目录为
  `~/guqin-agent/train/output/guqin_sft_v4_repeat_semantics_marker_redacted/`，日志位于
  `~/code/training/runtime/gpu_queue/logs/J00014.log`。

## 7.126 “参考谱”来源词补充脱敏与 v5 训练入队（2026-09-01）

- 复核 `Sxq5d2F6-p0018-fingering_agent-teacher-tools` 时发现公开 reasoning 仍含“参考谱”；旧校验只覆盖
  “参考谱面”，因此 `private_leakage_passed=true` 属于词面漏检。现已将“参考谱”加入脱敏黑名单、改写提示和
  SFT 导出校验黑名单；按当前口径不再检查是否复述私有技法文字，只禁止暴露私有来源。
- 对 `messages_repeat_semantics_redacted_final_v2` 全量重新检查，命中 1,391 条轨迹、1,656 个 assistant
  reasoning 回合；6 分片 GLM 重写及失败尾部重试后得到
  `ABC_J/agent_training/messages_repeat_semantics_redacted_final_v3/`，共 8,622 条，失败 0，公开 assistant
  中“参考谱”及既有私有来源黑名单残留均为 0；公开/私有 ID 配对与消息校验通过。
- 新完整多轮 SFT 导出为 `train/data/guqin_agent_sft_v5_reference_score_redacted_v1/`：8,622 条，
  Fingering/Guqinizer 各 4,311 条，assistant turns 22,829；SFT 校验 `valid=true`。主数据 SHA-256 为
  `801d8af0349762c1794523b63f3a97be4bf3a56a83318b8a85ea25536fac124d`。
- 仅将公开 SFT 文件上传到服务器
  `~/guqin-agent/train/data/guqin_agent_sft_v5_reference_score_redacted_v1/`，远端 SHA-256 与本地一致；
  未上传私有审计或 `.env`。
- 已取消尚未启动、仍指向旧 v4 数据的 `J00017`，提交四卡任务 `J00018`
  （`guqin-v5-reference-score-redacted-ctx8192-trunc5-smoke-full`）。配置为 `CUTOFF_LEN=8192`、
  `MAX_TRUNCATION_RATE=0.05`，先 smoke 再 full；截至记录时 `J00016` 正占用四卡，`J00018` 在队列中等待。

## 7.127 v5 两卡 smoke 通过并进入 full（2026-09-01）

- 因 GPU 1、2 被其他用户任务占用，将 `J00018` 从 4 卡调整为 2 卡，Jobber 分配 GPU 0、3，未抢占其他进程。
- 两条样本 smoke 完成 3 步、无 OOM，`train_loss=0.6859`；全量 preflight 渲染通过，8,622 条中 16 条超过
  8,192（0.1856%，低于允许的 5%）；5 条标签抽检 `valid=true`。
- `J00018` 已从 smoke 进入正式 full，当前状态 `running`，日志为
  `~/code/training/runtime/gpu_queue/logs/J00018.log`，使用公开 v5 SFT 数据和 2 卡 DDP。

## 7.128 Base 公有提示补充再作标记（2026-09-01）

- 按确认后的最小措辞更新 `scripts/generate_teacher_tool_trajectories.py`：Base/Fingering 公有提示由
  “为当前段每个演奏事件填写基础减字”改为“为当前段每个演奏事件填写基础减字；也可以填写\"再作标记\"/\"再作\"/\"从ㄱ再作\"等表示省略的减字”。
- 仅补充 Base 公有提示，未改私有参考、Guqinizer 提示、工具 schema 或已生成训练数据；Python 语法检查通过。

## 7.129 训练后评估链路准备（2026-09-01）

- 新增 `train/scripts/build_public_eval_inputs.py`：从封存的
  `ABC_J/agent_training/exported/evaluation_pairs_validation.jsonl` 与
  `evaluation_pairs_test.jsonl` 生成不含 `reference`、`metrics`、`training_allowed`、`provenance`
  的公开输入 `train/eval_inputs_v1/{validation,test}.jsonl`。两份公开输入共 583 条
  （validation 166、test 417），不含私有标注文本。
- 新增 `train/scripts/eval_agent.py`：加载基础 Qwen3.5-9B 与完成的 LoRA 适配器，在公开输入上生成
  Base/Fingering 评估预测；保留原始模型输出，同时解析 `edit_plan.jianzi_rows`，兼容字符串型序号，
  便于统计协议稳定性。
- 新增 `train/scripts/score_agent_predictions.py`：只在本地读取封存标注，对服务器预测计算协议有效率、
  逐格文本一致率和非空减字召回；文本比较做 Unicode NFKC、括号/空白归一化及中文弦数字归一化。
  服务器不接触封存标注。
- 新增 `train/server/run_post_train_eval.sh`：训练适配器完成后由 Jobber 排队执行 validation/test
  推理，默认 4-bit、bf16、单卡、最大生成 1024 token，并把结果写入 `train/eval_outputs/`；任务完成后
  下载预测文件，在本地运行评分脚本得到最终报告。
- 当前已生成并完成本地语法检查；待 `J00018` 训练完成后，评估任务按 Jobber 队列提交，避免占用正在运行
  的训练卡。评估输入上传前再次检查不含私有字段，封存评估对永不上传。

## 7.130 训练后评估任务改为双卡并修复加载兼容性（2026-09-01）

- 首次评估任务 `J00020` 曾启动但在模型加载处失败：服务器 Transformers 版本不接受旧式
  `load_in_4bit` 参数（尚未读取任何评估样本）。已将 `eval_agent.py` 改为显式
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bf16)`，并在服务器环境验证配置可用。
- 按“训练占用两卡，完成后释放两卡”的要求，重新提交 `J00023`，申请 2 张 GPU；它排在现有队列后，
  不会抢占训练或其他任务。评估入口会自动选择训练输出目录下编号最大的 `checkpoint-*`。

## 7.131 J00018 当前有效 batch（2026-09-01）

- 实际配置 `per_device_train_batch_size=1`、`gradient_accumulation_steps=8`，Jobber 为该任务分配 2 张卡
  （GPU 0、3），因此全局有效 batch 为 `1 × 8 × 2 = 16` 条轨迹/optimizer update。
- 文档中原先的有效 batch 32 是四卡方案（`1 × 8 × 4`）；本次两卡运行并未自动把梯度累计翻倍，不能按 32
  解读。若后续要保持有效 batch 32，应改为四卡或将累计步数设为 16；当前运行保持不变。

## 7.132 最新全量轨迹两级查看器（2026-09-02）

- 用当前最终公开脱敏消息 `messages_repeat_semantics_redacted_final_v3/messages_train.jsonl`、对应私有审计
  和 `inferred_v10_repeat_semantics_prompt_final/inferred_trajectories_train.jsonl` 重新生成全量查看器。
- 查看器按“曲谱 ID → phrase ID”两级展开；phrase 内包含 Fingering、Guqinizer、标注三方对比表，
  文字归一化一致的格子标绿，并提供两阶段最后 assistant 回复、搜索和 no-op 标签。
- 生成结果：4,311 个 phrase、163 个曲谱 ID、94 个 no-op phrase；文件位于
  `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/all-trajectories-hierarchical-v2-final-public.html`。
- 同时移除查看器对可选 `langgraph` 运行时的导入依赖，避免仅做可视化时因未安装 agent 运行环境而失败。

## 7.133 查看器标注列覆盖 bug 修复（2026-09-02）

- `SK7bMnBG-p0009` 暴露出对比表显示全空的问题：Fingering 审计含完整 20 个非空标注，配对的
  Guqinizer `review_noop` 审计却带有一个全空 `annotation_gqs`；旧查看器用后者覆盖了前者。
- 已改为逐项合并标注：空的 Guqinizer GQS 不再覆盖已有非空标注；重新生成 v2 查看器后，该 phrase
  恢复 20 个非空标注，178、179、190、191 四个不可解析音位仍为空，属于数据本身的空参考，不是渲染错误。

## 7.134 J00018 训练完成与双卡评估修复（2026-09-02）

- `J00018` 已完成全部 1,617/1,617 step、3 epoch，Jobber 状态为 `succeeded`；最终
  `train_loss=0.3084`。末尾出现的 CUDA OOM 只发生在销毁 NCCL process group 时，训练已先完成，
  根目录及 `checkpoint-1617` 的 LoRA 权重、trainer state、训练指标均已保存，随后
  LLaMA-Factory 也成功生成 `training_loss.png`，不影响最终适配器。
- 损失日志共 323 个记录点：首点约 0.724、末点约 0.234，前 20 点均值约 0.497、后 20 点均值约
  0.250，曲线持续下降。0.3084 是全程平均训练损失，不是末步损失，也不能单独替代留出集效果评估。
- 旧评估 `J00023` 暴露两处问题：Qwen3.5 tokenizer 返回 `BatchEncoding`，不能作为单个位置参数传给
  `generate`；此外 LLaMA-Factory 保存的 LoRA 键位于 `model.language_model.*`，直接用 PEFT 加载到
  `AutoModelForCausalLM` 会静默跳过适配器。`eval_agent.py` 已改为关键字张量输入，并用 PEFT
  `key_mapping` 将 `model.language_model.*` 映射为 `model.*`。
- `run_post_train_eval.sh` 已改成 validation/test 各占一张卡并行生成。新任务 `J00026`
  （`guqin-v5-post-train-eval-2gpu-valid-lora`）正在 GPU 0、2 上运行；两张卡均已实际加载约 11.7 GiB
  模型，日志中 `missing adapter keys` 为 0。

## 7.135 评估批量化与截断结果重跑（2026-09-02）

- 检查 `J00026` 前 56 条输出后发现，虽然文件中已有完整曲谱 ID 的全部行，但只有 10 条实际生成了
  合法 `edit_plan.jianzi_rows`；其余大多在 `max_new_tokens=1024` 时停在逐音 reasoning，不能视为完成。
- `eval_agent.py` 已支持左侧 padding 的批量生成、`--resume` 断点续跑和 CUDA OOM 自动二分降级；断点只
  复用 `protocol_valid=true` 且含非空 `jianzi_rows` 的记录，旧的截断行会自动重算。

## 7.136 Guqinizer 初稿错配修复与最终合并（2026-09-02）

- 发现 no-op 补 reply 合并后，部分 Guqinizer user 仍携带旧 Base 初稿，而对应 Fingering 已被后续重跑结果替换；
  `SK7bMnBG-p0009` 是典型案例。问题根因是 no-op 合并没有校验 Guqinizer 输入与当前 Base accepted plan 的版本一致性，
  不是 terminal assistant 追加文本造成的。
- 新增 `scripts/prepare_stale_guqinizer_rerun.py`，逐行比较 Guqinizer user 当前段与配对 Base accepted plan，支持多行减字文本和空显示归一化；
  从最终 v3 数据筛出 123 条错配片段。另有 2 条整段空标注，按规则不生成两阶段轨迹，因此不进入重跑。
- 123 条中 122 条 Guqinizer 重跑成功并经 GLM 脱敏（脱敏 122/122、失败 0）；`SumLbkVi-p0008` 连续质量门禁失败，未伪造结果，
  已将其 Fingering/Guqinizer pair 从最终版本隔离。重跑原始结果、脱敏替换包和重试记录保存在
  `ABC_J/agent_training/stale_guqinizer_rerun_v1/`。
- 新增 `scripts/assemble_stale_guqinizer_replacements.py`、`scripts/merge_repaired_guqinizer_dataset.py`，将 122 条脱敏
  Guqinizer 替换回原数据并保留私有审计；最终公开目录为
  `ABC_J/agent_training/messages_repeat_semantics_repaired_final_v1/`，共 8,620 条，Fingering/Guqinizer 各 4,310 条。
- `validate_teacher_agent_messages.py` 校验 `valid=true`；重新运行 stale 对齐检查，`affected_count=0`、`diff_cell_count=0`，
  说明每条保留的 Guqinizer user 已与当前 Base accepted plan 对齐。后续任何 no-op 合并前必须先运行该检查，避免旧版本再次混入。
- 双卡仍按 validation/test 各一卡并行，每卡默认 batch 4，生成上限提高到 3,072；若显存不足，单批会
  自动从 4 降到 2/1。新任务 `J00028`（`guqin-v5-eval-batch4-gen3072-valid-resume`）已在 GPU 0、2
  启动。此前“已有 7 首完成”的说法只是输入/输出行数齐全，按协议有效性重新核对后暂时没有整首全部完成。
- 随后按要求将生成上限进一步提高到 `10,240`，取消 `J00028` 并从有效断点启动 `J00029`
  （`guqin-v5-eval-batch4-gen10240-valid-resume`）；batch 4 与 OOM 自动降级策略保持不变。

## 7.136 全量查看器增加逐轮完整轨迹（2026-09-02）

- 旧 `all-trajectories-hierarchical-v2-final-public.html` 只展示两个阶段最后一条 assistant 回复，不能审计
  多轮 SFT 结构。已更新 `scripts/visualize_all_trajectories_hierarchical.py`，为每个阶段保留并渲染全部
  public messages，包括 system、user、assistant reasoning/tool_calls、tool result 和 final assistant。
- 每条消息按原顺序标记轮次、角色、工具名；内容中的字面 `\\n` / `\\r\\n` 会转为真实换行，工具调用
  单独显示。原有“曲谱 ID → phrase ID”两级展开、no-op 标签和两阶段 vs 标注表格不变。
- 新全量文件为
  `C:/Users/30343/.codex/visualizations/2026/08/27/01a04324-2f64-76f1-9983-9acfa9c17711/all-trajectories-hierarchical-v3-full-turns.html`，
  包含 4,311 个 phrase、163 个曲谱、94 个 no-op，约 101 MB。

## 7.137 评估集划分口径复核（2026-09-02）

- validation/test 确实按 `leakage_group_id`（曲谱家族）整体分配，单个 `score_key` 不会跨 split；并非在全库中
  随机抽 phrase。此前将评估称为“按 phrase 抽样”不准确。
- `inferred/` 中 validation 共有 441 个 phrase、test 共有 927 个 phrase；导出的评估对分别为 166、417，
  数量恰好等于各 split 中 `quality.sft_eligible=true` 的 phrase 数。差额是质量门槛排除的不可可靠评分片段，
  不是 split 时抽掉的片段。
- 因此“某曲评估已完成”应解释为：该曲在该 split 内所有可评估（`sft_eligible`）phrase 都已得到有效结果；
  不要求把被质量规则排除的 phrase 也纳入指标。若用户要求查看整首谱面连贯输出，则仍需另做全 phrase 推理。

## 7.138 按当前文本协议重建曲级隔离评估集（2026-09-02）

- 旧评估来自 8 月 12 日的 `inferred/`（validation/test = 441/927），且额外要求旧结构化
  `quality.sft_eligible=true`，只剩 166/417；该门槛依赖弦、徽、左右手和音高结构解析，不再适用于
  当前只监督 `jianzi_text` 的协议。
- 新增 `train/scripts/build_text_protocol_eval_set.py`，以最新版
  `inferred_v10_repeat_semantics_prompt_final` 为源。当前源 phrase 数为 validation 652、test 1,398；
  相比 `inferred_v6`，质量过滤删除 2/7 条，同时 89/149 个同名 phrase 的事件范围已因后续切分与再作
  修复变化，故旧预测完全不复用。
- 新评估不再检查 `sft_eligible` 或 verified structured patch；封存目标为每行原始 `jianzi_text`。
  只应用既定源质量规则：整谱非空覆盖率低于 50% 时剔除、连续空尾截断、截断后仍全空的 phrase 剔除；
  `"无（由于是再作部分，省略）"` 是非空目标，正常保留。
- 最终新版评估为 validation 595、test 1,373，共 1,968 条。Validation 排除低覆盖整谱 46 条、空尾 4 条、
  真全空 7 条；Test 排除低覆盖整谱 20 条、真全空 5 条。封存标注位于
  `ABC_J/agent_training/evaluation_text_protocol_v2/`，公开无标注输入位于
  `train/eval_inputs_v2_text_protocol/`。
- `eval_agent.py` 已支持直接读取新版预渲染公开 prompt，并为输入写入 SHA-256；不过本轮按要求使用全新输出
  目录，不复用任何旧预测。旧任务 `J00029` 已取消，新双卡任务 `J00031`
  （`guqin-v5-text-protocol-v2-full-eval`）已提交，输出目录为
  `~/guqin-agent/train/eval_outputs_v2_text_protocol/`；记录时因仅 1 张卡空闲而在队列等待 2 张卡。

## 7.139 纯“撮＋中文数字”候选重跑（2026-09-02）

- 按完整减字文字严格匹配 `^撮[一二三四五六七]{2,3}$`，确认 222 个 phrase、900 个动作单元（
  `撮三六` 41 个、`撮三七三` 1 个；不包含带弦徽或“散撮”等更丰富文本）。
- 使用 GLM-4.5、6 并发、断点续跑和每条最多 6 次尝试，在
  `ABC_J/agent_training/pure_cuo_rerun_v1/` 保存 6 个原始分片及 `merged_raw/`；成功 320 条阶段记录，
  47 条因 API/协议重试耗尽失败，33 条因离线音高质量筛选隔离，未伪造结果。
- 成功结果已用 GLM-4.5 脱敏（320/320，含 1 条词汇校验失败的本地回退修复），脱敏合并包为
  `pure_cuo_rerun_v1/redacted_merged/`；替换回基础最终集后生成
  `ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v1/`。
- 新最终集 8,620 条，Fingering/Guqinizer 各 4,310 条；公开/私有 ID 对齐且验证脚本 `valid=true`。
  严格模式文本由原最终集 900 个动作单元降为 254 个（仍有 96 个样本保留该短文本，需人工/后续策略决定
  是否继续改写，不能把模型未改写误称为全部消除）。

## 7.140 纯“撮＋中文数字”尾部清零（2026-09-03）

- 新增 `scripts/prepare_pure_cuo_refresh.py`，按 phrase 和阶段审计最终 accepted plan。v1 尾部实际涉及
  70 个 phrase：49 条 Base/Fingering 命中，因此必须重跑 Base 后再以新 Base 重跑 Guqinizer；21 条
  仅 Guqinizer 命中，因此只重跑 Guqinizer。两阶段同时命中的 26 条只计一次 phrase。
- 教师私有规则新增一句短约束：普通撮弦减字必须写清参与的两弦及必要取音信息，不得把“撮”与二至三个
  中文数字直接拼成缩略写法。该规则同时用于 Base 和普通 Guqinizer，不改变公开 agent 提示。
- 70 条中有 4 条属于当前源数据的整段空标注，按既定质量策略不调用模型，并在最终集成对剔除。其余
  66 个 phrase 全部完成相应阶段重跑；Base 音高筛选尾部经过多轮定向重试后 47/47 可训练，Guqinizer
  结果亦补齐。新替换包位于 `ABC_J/agent_training/pure_cuo_refresh_v2/replacements_raw/`，共 113 个阶段
  样本，严格正则命中 0。
- 113 条 reasoning 已统一脱敏到 `pure_cuo_refresh_v2/replacements_redacted/`：写入 113、失败 0、实际
  API 脱敏调用 35。最终合并目录为
  `ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v2/`，共 8,612 条，Fingering 与
  Guqinizer 各 4,306 条；公开/私有 ID 对齐，消息验证 `valid=true`。最终 accepted plan 再次扫描
  `^撮[一二三四五六七]{2,3}$`：样本 0、动作 0。

## 7.141 训练前全量可视化与两卡训练提交（2026-09-03）

- 训练前全量复核页由 `scripts/visualize_all_trajectories_hierarchical.py` 生成，按曲谱 ID →
  phrase ID 两级展开，保留 Fingering/Guqinizer 完整公开轨迹，并提供与标注版本的逐行对比表。
  输出文件：`all-trajectories-final-v2-training-preflight.html`（8,612 条阶段记录合并为 4,306 个
  phrase、163 个曲谱、97,343 个对比行、129 个 no-op phrase）。
- SFT 导出目录为 `train/data/guqin_agent_sft_v5_pure_cuo_final_v2/`，本地两套校验均通过；训练数据
  主文件 SHA-256 为 `3ce5108535e2b09b97039847891765b36f260e5dd6d57dc18899881f7b295bb3`。
- 已上传至服务器 `~/guqin-agent/train/data/guqin_agent_sft_v5_pure_cuo_final_v2/` 并核对 SHA-256。
  通过 Jobber 提交完整训练：`J00054 guqin-sft-v5-pure-cuo-final-v2`。当前已将资源需求从 4 卡调整为
  2 卡；因 GPU 0、1 仍被已有任务占用，Jobber 会等待两张卡同时空闲后自动启动，未抢占现有任务。

## 7.142 Guqinizer 绰／注方向提示统一（2026-09-03）

- `scripts/generate_teacher_tool_trajectories.py` 已将 Guqinizer 的公开方向提示和私有教师规则统一为：
  “绰表示左手由低音位置滑向本位，注表示左手由高音位置滑向本位（如绰上七徽：由八徽方向滑至七徽；注下七徽：由六徽方向滑至七徽）”。
- 临时迁移脚本 `scripts/tmp_replace_guqinizer_direction_prompt.py` 已替换最终公开轨迹、私有审计和 SFT 导出中的历史提示文本；共替换 17,658 处（含 5 处历史 reasoning 引用片段），旧句残留为 0。
- 更新后的公开消息、SFT 导出和两级全量可视化均已复核；SFT 主文件 SHA-256 更新为
  `f197a3a1ba4648069057804780fbae341e23ed9069b87311daead499cc80db8c`。新的可视化文件为
  `all-trajectories-final-v2-direction-prompt.html`。J00054 尚未启动，已覆盖服务器训练目录，启动时会使用更新后的数据。

## 7.143 8192 两卡训练重提交（2026-09-03）

- 对最终 8,612 条 SFT 数据按服务器真实 Qwen3.5 tokenizer 统计：P50=3,357、P90=5,795.9、
  P95=6,150.45、P99=6,964.56、最大 10,017；区间计数为 ≤2,048: 209、2,049–4,096: 4,537、
  4,097–6,144: 3,432、6,145–8,192: 418、>8,192: 16（0.1858%）。Fingering 最大 10,017，
  Guqinizer 最大 6,762。
- 原两卡 `CUTOFF_LEN=18,432` 任务 `J00054` 在约 85/1,617 步因 Triton fused layer-norm backward
  OOM 失败，未产生可续训 checkpoint。两卡 8192 任务 `J00055` 随后已按用户要求取消；改用
  `CUTOFF_LEN=8,192`、`MAX_TRUNCATION_RATE=0.05` 从头提交四卡任务
  `J00057 guqin-sft-v5-pure-cuo-final-v2-ctx8192-4gpu`，当前排队等待四张卡同时空闲。

## 7.144 J00057 四卡训练完成（2026-09-04）

- J00057 已在 GPU 0、1、2、3 上完整跑完 810/810 steps（3 epoch），总耗时约 9 小时，最终
  `train_loss=0.3436`。结束阶段出现 NCCL 清理告警，但训练指标、adapter 和 `training_loss.png`
  均已正常写出，Jobber 状态为 `succeeded`；用户随后发出的停止请求到达时已无运行进程。
- 产物目录：`~/guqin-agent/train/output/guqin_sft_v5_pure_cuo_final_v2_ctx8192_4gpu/`。

## 7.144 评估调弦输入链路复核（2026-09-03）

- 逐条核对当前文本协议评估集：validation 595 条、test 1,373 条的 `public_prompt` 调弦数组均与
  `runtime_item.input.normalized_tuning.open_midi` 一致（1,968/1,968），且每条均包含当前段和音符表；
  评估 prompt 不含私有标注。
- `eval_two_stage_score.py` 通过 `render_public_prompt` 将 normalized tuning 传入 Base 与 Guqinizer；
  `eval_agent.py` 读取预渲染 `public_prompt` 时也会原样传递调弦。已修正无预渲染 prompt 时对 v2
  `runtime_item`、`normalized_tuning` 和 `notes_without_jianzi` 的兼容路径。
- `train/server/run_post_train_eval.sh` 默认评估集已从旧 `eval_inputs_v1` 切换为当前
  `eval_inputs_v2_text_protocol`；若需旧版对照，必须显式设置 `INPUT_ROOT`。
- 旧入口曾调用单阶段、单轮 `eval_agent.py`，该结果不能代表完整 Agent 能力。现已将默认入口切换为
  `eval_two_stage_score.py`：逐曲按 phrase 顺序执行 Base→Guqinizer，多轮开放生产版
  `get_pitch_candidates`、`expand_context` 与 `edit_plan`；评估输入若含 `reference` 或
  `reference_plan` 会直接拒绝运行。

## 7.145 评估框架与教师框架对齐（2026-09-03）

- 评估阶段直接复用教师轨迹生成器的公开 system prompt、公开 user prompt、工具 schema、
  `RealToolRuntime`、预览校验及基础阶段完整性检查；不使用教师私有 prompt，也不加载封存标注。
- 每个 phrase 严格按 Base→Guqinizer 完成后才进入下一 phrase；Guqinizer 的最终结果成为下一段的
  `confirmed_readonly` 前段，并用于计算泛音区间和 `expand_context` 历史。故级联上下文全部来自模型自身输出，
  不会混入标注前文。
- Base 可多轮分批编辑，Guqinizer 可编辑或明确 no-op；每轮工具调用与真实工具结果均写入评估轨迹。
  断点续跑只接受新两阶段 schema 的完整记录，旧单阶段结果不会被误判为已完成。

## 7.146 评估脚本入口与使用方法（2026-09-03）

- 主评估脚本为 `train/scripts/eval_two_stage_score.py`。它按曲谱、phrase 顺序运行
  `Base → Guqinizer → 下一 phrase`，不读取私有标注，并将模型自己生成的上一段 Guqinizer
  结果作为下一段的 `confirmed_readonly` 前文。
- 服务器入口为 `train/server/run_post_train_eval.sh`，默认读取
  `train/eval_inputs_v2_text_protocol/{validation,test}.jsonl`，写入
  `train/eval_outputs/{validation,test}_predictions.jsonl`；默认
  `MAX_NEW_TOKENS=10240`、`MAX_TOOL_ROUNDS=8`、`EVAL_ATTEMPTS=2`，带 `--resume` 断点续跑。
  设置 `CUDA_VISIBLE_DEVICES=0,1` 时 validation 与 test 各占一张卡并行运行。
- 若需重建公开评估输入，使用 `train/scripts/build_public_eval_inputs.py`；它从封存评估对中
  剥离标注后生成公开输入。服务器结果下载回本地后，使用
  `train/scripts/score_agent_predictions.py`，分别对封存的 validation/test 标注评分。
- 标准服务器命令：
  `CUDA_VISIBLE_DEVICES=0,1 ADAPTER=~/guqin-agent/train/output/<adapter> bash train/server/run_post_train_eval.sh`。
  评分命令示例：
  `python train/scripts/score_agent_predictions.py --reference agent_training/evaluation_pairs_validation.jsonl --predictions train/eval_outputs/validation_predictions.jsonl --output train/eval_outputs/validation_scores.json`。

## 7.147 no-op 目标不变量修复与受影响 Guqinizer 重跑（2026-09-03）

- 修复 `scripts/generate_teacher_tool_trajectories.py`：无工具调用只能在“当前 Base 完整且重新计算的文字目标为空”时接受为 no-op；只要存在实际减字差异，不能再被模型的 `no_changes` 绕过。`edit_plan` 预览中的固定文案也由“Agent 已填写”改为“已填写”。相关单元测试与协议测试共 76 项通过。
- 受影响目标 90 条已用 4 并发重跑：第一轮 73 条成功、17 条因 API 瞬时错误失败；第二轮成功 1 条；第三轮启用 `--allow-private-reasoning-leakage` 后剩余 16 条全部成功。原始结果保存在 `ABC_J/agent_training/noop_surface_target_rerun_v1/`，公开脱敏替换包为 `replacements_redacted_90/`。
- 合并后的最终公开轨迹为 `ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3/`，共 8,612 条（Fingering 4,306、Guqinizer 4,306），公开 reasoning 黑名单扫描通过，public/private ID 对齐。no-op 审计脚本已支持首个 source 权威、后续 legacy source 仅补缺行；使用旧版质量 source 补 `SywB9Jym-p0075` 后，审计结果为 39 个 no-op、0 个违规。
- 新 SFT 导出目录为 `train/data/guqin_agent_sft_v5_pure_cuo_final_v3/`，导出 8,612/8,612 条，assistant turns 22,782、tool-call turns 14,170；训练前需使用该 v3 目录替换服务器上旧 v2 训练目录。此前排队的旧数据训练任务不得直接作为本修复版结果。

## 7.148 训练阶段音高可解析/匹配统计（2026-09-03）

- 新增 `scripts/audit_training_phase_pitch_parse_stats.py`。脚本直接导入并复用
  `scripts/audit_jianpu_jianzi_pitch.py` 的 `parse_jianzi`、泛音/走手/复合技法上下文、
  弦徽位置计算及 MIDI 配对逻辑；不会用“出现数字/弦字”这种额外启发式替代旧审计。
- 用最终教师审计中的 Base/Fingering 与 Guqinizer `accepted_plan` 逐阶段重建减字，输入 source
  使用 v10 source，缺失的 90 个重跑片段由旧 quality source 补齐：
  `python scripts/audit_training_phase_pitch_parse_stats.py --source ABC_J/agent_training/inferred_v10_repeat_semantics_prompt_final/inferred_trajectories_train.jsonl --source ABC_J/agent_training/inferred_quality_filtered_v1/inferred_trajectories_train.jsonl --audit ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3/teacher_trajectory_audit.jsonl --output ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3/training_phase_pitch_parse_stats.json`
- 初版统计曾把“七弦、四弦”等中文弦号直接交给旧解析器，造成严重低估；脚本现只在解析前将紧邻“弦”的一至七中文数字归一化为阿拉伯数字，并支持明确的“散挑七”“大九勾四”“名十勾二”简写，不改变旧算法的语义判定。
- 修正后：8,612 个 phase 均成功配对；193 个（2.24%）没有可解析音高的减字，8,046 个（93.43%）有可解析减字且至少一音 MIDI 匹配，373 个（4.33%）有可解析减字但一个都不匹配。另有 26 个 phase 的减字文本全部为空（Base 5、Guqinizer 21）；剩余 167 个“无可解析”phase 有非空文本，主要是缺少必要弦/徽信息的其他简写，以及绰、注、泛音、复合动作等旧算法安全跳过的上下文动作，不能在统计层面擅自猜测。
- 报告中的 `summary.no_parseable_phase_reason_presence` 和各阶段同名字段记录了 phase 级原因覆盖数；`unparseable_reason_counts` 是逐音行计数，二者不能混用。包含小节线/延音的 `no_sounding_jianpu` 不代表减字质量问题。

## 7.149 音高审计两类轨迹导出与服务器同步（2026-09-03）

- 新增 `ABC_J/scripts/export_pitch_eligible_trajectories.py`。它读取音高统计报告，按 phrase 成对筛选：
  Fingering 与 Guqinizer 两阶段都必须属于“无可解析音高”或“可解析且至少一音匹配”；
  “可解析但完全不匹配”的阶段及其不完整 phrase 不进入筛选集。
- 完整公开轨迹本地上传包：`ABC_J/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3_public/`；
  筛选公开轨迹本地目录：`ABC_J/agent_training/messages_repeat_semantics_pitch_eligible_v1/`，含
  `messages_train.jsonl`、`pitch_eligible_phrase_ids.txt` 和 `selection_manifest.json`。筛选结果为 4,039 个 phrase、
  8,078 条阶段轨迹（两阶段各 4,039）。
- 服务器正确目录为：
  `~/guqin-agent/agent_training/messages_repeat_semantics_pure_cuo_repaired_final_v3_public/` 和
  `~/guqin-agent/agent_training/messages_repeat_semantics_pitch_eligible_v1/`；对应绝对路径为
  `/home/20223393ljw/guqin-agent/agent_training/...`。三份筛选文件及完整文件的 SHA-256 已与本地核对一致。

## 7.150 Qwen3.5 reasoning 分层与只读占位符修复（2026-09-04）

- 修改 `train/scripts/export_sft_dataset.py`：含工具调用的 assistant 公开 reasoning 现在序列化为
  `<think>...</think>`，随后保留 Qwen3.5 XML `tool_call`；no-op 的复核分析同样进入 `think`，标签外
  保留简短的“无需修改”最终答复。普通最终完成答复不凭空包成 reasoning。
- 新增 `scripts/repair_readonly_placeholders.py`。旧轨迹中只读段误用当前段占位符的 4,785 行已修复为
  `[空]`，涉及 831 条轨迹；当前段的 `[减字待填写]` 保持不变。修复副本为
  `ABC_J/agent_training/messages_repeat_semantics_pitch_eligible_v1_readonly_fixed/`。
- 新导出数据为 `train/data/guqin_agent_sft_pitch_eligible_v2_think/`，8,078 条轨迹、13,301 个工具回合；
  本地导出校验和服务器真实 LLaMA-Factory 100 条/268 个 assistant target 抽检均通过。
- 四卡训练已提交 Jobber `J00068`：`guqin-sft-pitch-think-ctx8192-accum1`，`cutoff_len=8192`、
  `gradient_accumulation_steps=1`、`MAX_TRUNCATION_RATE=0.05`。截至记录时因 GPU 0、2 被另一账户占用，任务仍为 queued，未误停他人进程。
- 只读/当前表格中的小节线统一显示为 `[小节线]`，不再误用 `[空]`；迁移脚本本次补写 53,272 行，
  并已覆盖 J00068 尚未启动时读取的服务器训练目录。

## 7.151 GQS 1.2 紧凑结构与连续音序协议（2026-09-04）

- `agents/abc_to_jianzipu/teacher_gqs.py` 已升级为 `teacher-gqs-1.2`：分节用独立的 `<一>` 等结构行，
  小节线用独立的 `小节线` 行，不占音序；音行首列使用连续的 agent-facing 音序，随后以紧凑 JSON
  数组保留简谱、ABC、时值、歌词和减字文字的空值语义。
- GQS 内含 `索引映射`（音序→内部 `source_index`、小节线及分节定义）作为机器侧无损回放信息；
  `parse_teacher_gqs` 同时兼容 `teacher-gqs-1.1`，新旧格式均有往返单元测试。
- `scripts/generate_teacher_tool_trajectories.py` 的公开工具参数已改为 `event_index/event_indices`，
  `edit_plan.jianzi_rows` 采用 `[音序, 减字文字]`；运行时自动转换为内部 source index，预览、候选来源和
  警告回执再显示音序。旧 `source_index/source_indices` 输入保留兼容路径，不写入新的公开 schema。
- 已生成全量规范 GQS：`ABC_J/agent_training/gqs_v12/`（241 曲）；已重新划分轨迹：
  `ABC_J/agent_training/inferred_gqs_v12/`，共 6,741 个 phrase（train 4,691、validation 652、test 1,398）。
  GQS 构建、解析和质量测试全部通过；该版本尚未替换正在排队的 J00068 训练数据，需明确重新导出后再训练。

## 7.152 GQS 1.2 音高合格集重跑（2026-09-05）

- `ABC_J/scripts/export_pitch_eligible_trajectories.py --inferred-dir ABC_J/agent_training/inferred_gqs_v12 --require-match`
  按新协议重新筛选出 5,715 个 phrase（总 6,741 个的 84.8%），全部属于“至少一音可解析且有匹配”；筛选清单为
  `ABC_J/agent_training/pitch_eligible_gqs_v12/pitch_eligible_phrase_ids.txt`，并保留 `selection_manifest.json`。
- 注意数量口径：旧版 8,612 是两阶段 messages 行数（4,306 phrase×2），新版 inferred_gqs_v12 的 6,741 是
  每 phrase 一行的规范输入，不是少了近 2,000 个 phrase；对应的 5,715 个筛选 phrase 后续会生成约 11,430 条两阶段消息。
  小节线仅不占连续音序，不会导致 phrase 被筛掉。脚本默认模式已修复为兼容新协议顶层分类字段；若要同时纳入“无可解析音高”类，省略 `--require-match`。
- 已抽查四个公开 prompt：只读前段与当前段分界正确，旧段不再出现 `[减字待填写]`，当前段保留该占位符；分节用 `<一>/<二>`
  等结构行，小节线为独立 `小节线` 行，音序连续且不把小节线计入音序。新 GQS 1.2 的 `event_index/event_indices` 工具协议可正常回放。
- 两条 smoke phrase（SMcokJse-p0062、SfRWCtNp-p0022）已用 GLM-4.5、`--allow-private-reasoning-leakage` 完成 Base+Guqinizer，4 个阶段回合全部通过，0 拒绝。
- 为支持 6 并发，生成器新增 `--score-shard-count/--score-shard-index`：每 worker 只保留所负责曲谱的完整历史，避免 689MB 输入被六次全量解析导致 `MemoryError`，不改变上下文语义。
- 全量重跑已启动（GLM-4.5、6 并发、`--include-guqinizer-no-op`、`--max-attempts 6`、`--min-interval 1`、允许原始 reasoning 泄露），当前分片输出放在
  `C:\Users\30343\.codex\visualizations\2026\09\03\01a065f1-9f9a-77a2-97e1-effc6361e996\gqs_v12_run1\messages_gqs_v12_run1_shard_0..5\`，各目录含 checkpoint，可断点续跑。
- 全量完成后应先用 `scripts/merge_teacher_retry_outputs.py` 合并六个分片，再运行 `scripts/redact_teacher_reasoning.py`（保留原始私有审计，公开副本脱敏），最后用验证脚本检查阶段成对、tool-call 可回放及黑名单。

## 7.153 GQS 1.2 可视化与运行暂停（2026-09-05）

- 已按要求暂停全量教师生成及监督器；C 盘工作区中的 checkpoint 和已完成分片均保留，未删除或覆盖。
- 修复 `scripts/visualize_all_trajectories_hierarchical.py`：对比表现在按源顺序显示段落标记和小节线，并将新 GQS 连续音序准确映射回含小节线的内部 `source_index`，避免标注与最终结果错位。
- 修复 `scripts/visualize_teacher_io_compare.py`：支持 `--source` 对齐新旧音序，并在教师 I/O 对比表中显示段落标记、小节线。
- `render_teacher_reference_gqs` / `render_annotation_gqs` 已补发 `<一>` 等段落标记；私有参考仍保留小节线，后续新生成的教师输入会同时包含两类结构行。
- 教师模型配置已统一为 `glm-5.3`，不再使用 `glm-5.3-flash`；已同步 `.env`、worker 与 supervisor 脚本（旧的 7.152 运行记录仍准确保留当时使用的 `glm-4.5`）。后续恢复运行前需重新确认并发和内存策略。

## 7.154 GQS 1.2 教师生成集限制为 train（2026-09-05）

- 复核发现 §7.152 的 5,715 个 phrase 来自 `train + validation + test`，原因是
  `export_pitch_eligible_trajectories.py --inferred-dir` 旧实现固定遍历三个 split，且 worker 又读取
  `inferred_trajectories_all.jsonl`。其中 train 3,927、validation 550、test 1,238；这批混合目标不得作为训练集。
- 新协议的曲谱/phrase 规模为：train 176 首、4,691 phrase；validation 21 首、652 phrase；test 44 首、
  1,398 phrase；合计 241 首、6,741 phrase。此前“241 首”是全量数据集，不是训练曲谱数。
- 筛选器新增 `--splits`，默认仅 `train`；显式需要评估集时才可传 `--splits validation test`。
  新训练专用清单为 `ABC_J/agent_training/pitch_eligible_gqs_v12_train/`，包含 3,927 个音高匹配 phrase、
  来自 169 首训练曲谱。
- worker/supervisor 已改为只读取 `inferred_trajectories_train.jsonl`，并写入全新的
  `gqs_v12_train_run1` 目录；旧 `gqs_v12_run1` 混合 checkpoint 保留作审计但不得续跑或合并进训练数据。

## 7.155 复杂技法知识与继承规则补充（2026-09-05）

- `complex_fingering_explanations.jsonl` 原本已有“撞”，但旧注入器过滤所有单字技法；“吟滑”原本确实缺失。
  现已补充“吟滑”解释，并允许“吟、猱、撞”三种明确的单字依附性左手动作进入私有知识注入；仍采用最长匹配，
  因此命中“吟滑”或“撞猱”时不会重复注入较短的“吟”“撞”“猱”。
- Base 与 Guqinizer 的公开系统提示共同增加三句局部继承规则：左手位置/右手指法按字段继承，“就”继承走手后的
  当前位置；吟、猱、撞、绰、注等续作承接前一按音，泛起—泛止继承泛音作用域；明确的新字段只更新相应状态，
  禁止机械复制上一音全部字段。私有教师提示的 `role` 使用同一公开系统文本，因此同步获得这些规则。
- 对 `S0FKGjFt-p0001` 静态检查：私有知识现同时注入“注下”“吟滑”“撞”；相关回归测试 3/3 通过。
- Base 与 Guqinizer 的公开提示新增普通“撮”规则：撮本身就是右手指法，括号内只保留两弦及必要的
  散/按音、左手按指和徽位信息，不再拆写“勾、挑、托”等右手动作；原有禁止“撮三六”式数字缩写的规则继续保留。

## 7.156 train-only GLM-5.3 十条 smoke（2026-09-05）

- 从 `pitch_eligible_gqs_v12_train/pitch_eligible_phrase_ids.txt` 取前 10 个训练 phrase，生成 Base 与
  Guqinizer 共 20 条阶段轨迹，accepted 20、rejected 0、quarantined 0；输出目录为
  `ABC_J/agent_training/messages_gqs_v12_train_smoke10_glm53/`。
- 审计中的 30 次 API 往返，其 request/response 模型元数据均为 `glm-5.3`；20/20 阶段私有参考均含
  当前段落标记，18/20 含小节线，另外 2 条源 phrase 本身没有小节线。
- 已生成 `trajectory_viewer.html`（两阶段最终结果与标注对比）及 `teacher_io_viewer.html`（教师 I/O）；
  两者均使用 train source 对齐连续音序，并显示段落标记与小节线。
- 首次查看 `S0FKGjFt-p0001` 暴露查看器自身的双重索引错误：accepted plan 的整数
  `source_index` 被存成字符串后又用整数查询，导致结构化源行全空并把同一批动作作为“额外行”重复追加；
  annotation-phrase-1.2 的连续音序还可能与同号小节线 source index 混淆。现已统一用整数
  `source_index` 做内部关联、用 `event_index` 做表面序号，并按 GQS 1.2 schema 强制音序映射；和弦
  `jianpu_alt` 也恢复到显示。10 个 smoke phrase 均验证音序连续且无重复。`p0001` 音序 3 现正确对齐
  Base、Guqinizer 与标注三列。

## 7.157 GLM 全量任务网络授权注意事项（2026-09-05）

- 外部 GLM API 任务必须在允许网络访问的授权环境中启动；普通沙箱内对
  `https://open.bigmodel.cn` 的 socket 访问会失败，并被生成器误分类为
  `transient API error`，随后反复重试而不产生任何轨迹。
- 2026-09-05 的 `messages_gqs_v12_pitch_eligible_full_parallel8_v5` 曾在普通沙箱中启动，
  8 个 worker 均连续失败，已停止；未产生有效 `messages_train.jsonl`。后续应使用全新输出目录，
  通过授权网络启动，不能续用该目录中仅含失败状态的 checkpoint。
- 启动前必须做一次只读 endpoint 连通性检查，并在日志中确认至少一个 worker 已写入成功轨迹；
  若所有 worker 只出现 `transient API error` 且输出 JSONL 仍为 0，应立即停止排查网络权限，
  不得让任务继续空转。

## 7.158 v6 全量运行磁盘空间阻塞（2026-09-05）

- 授权网络重启后的 `messages_gqs_v12_pitch_eligible_full_parallel8_v6` 已实际生成 Base 482 条、
  Guqinizer 469 条（共 951 条阶段记录），随后因 D 盘可用空间降为 0 而全部 worker 退出。
- 该目录中的部分 checkpoint 和阶段结果保留；磁盘清理前不得重启或合并。清理任何旧数据前必须先确认具体目录并取得明确授权。
