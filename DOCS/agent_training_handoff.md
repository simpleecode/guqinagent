# ABC → 减字谱 Agent：交付文档

更新时间：2026-10-01  
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
  --max-tool-rounds 24 --max-attempts 3 `
  --output-dir ABC_J\agent_training\messages_prompt_review_v24
```

pilot 生成（pilot ID 清单来自 `ABC_J\agent_training\pilot_sampling\pilot_ids.txt`；当前 Token Plan
先用单线程，且必须显式传 ID）：

```powershell
$pilotArgs = Get-Content ABC_J\agent_training\pilot_sampling\pilot_ids.txt |
  Where-Object { $_.Trim() } | ForEach-Object { '--trajectory-id'; $_.Trim() }

& 'D:\Program Files\miniconda\envs\guqin-agent\python.exe' `
  scripts\generate_teacher_tool_trajectories.py `
  --input ABC_J\agent_training\inferred_v6\inferred_trajectories_train.jsonl `
  --basic-intermediate --max-tool-rounds 24 --max-attempts 3 `
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

批量生成前，应先选定独立输出目录并小批量检查失败样本；不要覆盖已有验证目录。`--trajectory-id`
可用于分批与断点调查；已有 Fingering 中间体可通过 `--intermediate-input` 续跑
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

> 2026-08-24 至 2026-08-31 的历史日志（§7.19–§7.121）已归档至 [archive/agent_training_handoff_log_202608.md](archive/agent_training_handoff_log_202608.md)。

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
- 全量完成后应先用 `scripts/merge_teacher_retry_outputs.py` 合并六个分片，再运行 `ABC_J/scripts/redact_teacher_reasoning.py`（保留原始私有审计，公开副本脱敏），最后用验证脚本检查阶段成对、tool-call 可回放及黑名单。

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

## 7.159 音高资格与异常连续空减字筛选链路复核（2026-09-14）

- “标注减字有可解析音高、但所有可解析音均不匹配”这一类 phrase 现由
  `ABC_J/scripts/filter_training_data.py pitch-select` 筛选。它动态调用
  `scripts/audit_training_phase_pitch_parse_stats.py` 的 `build_pitch_audit()` / `classify()`，后者使用
  `scripts/audit_jianpu_jianzi_pitch.py` 的解析与审计逻辑；`--require-match` 仅保留
  `parseable_and_at_least_one_match`，剔除 `parseable_but_no_match`。两阶段按 phrase 成对选取，避免留下单阶段数据。
- 调弦双重偏移修复后，以 train 的 4,691 条 `inferred_gqs_v12` 和
  `inferred_gqs_v12_tuningfix_20260914` 重新执行同一 `--require-match` 口径：均为 3,969 条，新增 0、移除 0。
  因此修复确实影响带调弦 phrase 的音高数值/提示构造，但在“至少存在一个解析且命中音”的粗资格门槛下没有改变合格集合；不能据此把 993 条调弦重跑视为无效。
- “普通单发减字后至少连续 5 个未标注发音事件”的历史清理结果保存于
  `ABC_J/agent_training/exclusions/basic_attack_followed_by_5plus_blank_events_20260912.txt`（56 个 phrase）。
  该清单在 `pitch_eligible_gqs_v12_train/selection_manifest.json` 中作为 `quality_exclusions` 留痕；仓库中未找到当时生成该 5+ 连续空规则的独立脚本，说明该次检测代码没有持久化，未来不能假设它会自动随新源数据复扫。
- 相关空减字规则现由统一入口的 `source-audit`、`apply-source-audit` 与 `filter-corpus` 应用；
  `scripts/audit_blank_reference_targets.py` 仍仅负责审计全空/显式空目标。它们均不等价于上面的“基础单发 + 5 连空”规则。

## 7.160 训练数据筛选器统一入口（2026-09-14）

- 原有的音高资格、源谱覆盖率/连续空尾、全空 phrase 及按 ID 删除脚本已统一为
  `ABC_J/scripts/filter_training_data.py`：`pitch-select`、`source-audit`、`apply-source-audit`、
  `basic-blank-audit`、`filter-corpus` 五个子命令共享 JSONL、phrase ID 和报告格式。
- 删除旧入口 `export_pitch_eligible_trajectories.py`、`audit_mapped_jianzi_quality.py`、
  `merge_quality_filter_minimal.py`、`prune_all_empty_trajectory_phrases.py` 与
  `filter_teacher_trajectories_by_source_ids.py`；评估集构建脚本已改为从统一入口加载源谱审计函数。
- 新增的 `basic-blank-audit` 是保守的候选检测：跳过小节线、休止、再作省略和已知复合/延续技法，输出候选报告而不自动删数据。
  旧 56 条排除清单保留为人工审阅后的历史基线，不能与自动候选集混同。

## 7.161 继承弦位走手音高复核与 Guqinizer 重跑（2026-09-18）

- `scripts/audit_jianpu_jianzi_pitch.py` 现允许在已继承弦位时审计独立走手减字的明确徽位终点；没有可用继承弦位/有效终点时仍跳过，不猜弦。走手终点使用当前 phrase 的简谱音高比较。
- 新增 `scripts/audit_guqinizer_walk_pitch.py`，从源 phrase、历史 accepted Guqinizer plan 与 `pitch_audit_notes()` 回放每首曲子的历史状态，审计新增可计算终点的无右手取声走手行，并输出模式审核页及待处理 trajectory ID 清单。
- 初次统计的 2,020 个不匹配 phrase 是错误结果：历史回放音行被重复归入每个后续 phrase。脚本现按当前 phrase 的 `notes_without_jianzi.index` 过滤审计明细。初次修正后的基线为 7,434 个可核验行、455 个不匹配行、315 个完整 trajectory；`ShSCyERN-p0026` 单独复核使用正调 `open_midi=[48,50,53,55,57,60,62]`，其 `注下七徽` 与 C5 匹配，不在不匹配清单。
- 对上述 315 个完整 trajectory 仅重跑 Guqinizer，原 Finger 中间结果保留。GLM-5.3 8 路首轮接受 312 条；3 条因教师响应格式无效失败，随后单独重试均成功。最终接受 315/315，分片任务无失败。
- 新轨迹经 8 个 GLM-5.3 脱敏分片处理，315/315 脱敏成功、0 失败。合并替换后的完整数据目录为
  `ABC_J/agent_training/messages_glm_full_two_stage_harmonicstatefix_lipitchfix_p0063harmonicwarning_statewarning38_harmonicparsefix_walkpitchfix_20260918/`；报告显示替换 315 条、总计 7,694 条（Fingering 3,847、Guqinizer 3,847），公开/私有 sample ID 对齐。原版本保留未覆盖。
- 新版全量复审为 7,282 个可核验走手行，其中匹配 7,173、不匹配 109，涉及 71 个完整 trajectory。对本次重跑的 315 条子集，走手不匹配由 455 行/315 个 phrase 降至 107 行/69 个 phrase。新版审核页：
  `ABC_J/agent_training/guqinizer_walk_pitch_postaudit_20260918/review.html`；前版候选审核页及统计保留于
  `ABC_J/agent_training/guqinizer_walk_pitch_audit_20260918/`。审计口径仍只是标量音高筛查，复合动作及无明确终点的走手动作不据此判错。

## 7.162 四卡 A100 Qwen3.5-9B 训练产物归档（2026-09-17）

- 本地完整产物归档目录：`train/artifacts/qwen35_9b_lora_4xa100_bs2_acc1_ctx8192_20260917/`（约 103 MB）。训练使用服务器数据快照
  `/data/guqin-agent/data/guqin_agent_sft_glm_final_harmonicparsefix_20260915`，共 7,694 条样本；该快照早于 §7.161 的 315 条 Guqinizer 更新。
- 配置：Qwen3.5-9B，4 张 A100，4-bit QLoRA，Liger，bf16，`cutoff_len=8192`，每卡 batch 2，梯度累积 1，3 epochs；实际有效 batch 为 `4 × 2 × 1 = 8`。两份 YAML 均已归档于 `configs/`，其中注释“1 × 8 × 4 = 32”是过时内容，不能当作本次实际 batch 配置。
- 训练从 `checkpoint-1900` 续跑，最终 `global_step=2886`、epoch 3。最终报告 `train_loss=0.12697`；完成日志报告本次续跑 `train_runtime=15,833 秒（4:23:53）`，这是续跑进程所报运行时间，不代表含前段及暂停时间的总墙钟耗时。未配置 eval，因此没有 `eval_loss`。
- 主要产物：`final_adapter/adapter_model.safetensors`（约 83 MB，LoRA adapter）、`training_loss.png`、`trainer_state.json`、`trainer_log.jsonl`、`train_results.json`、`all_results.json`、`training_args.bin`；初次运行与从 step 1900 续跑的配置、日志均在 `configs/`、`logs/`。最终日志确认 2,886/2,886 步和 checkpoint 保存完成。
- 本段记录的是 SFT 训练产物与 loss 指标，不包含 held-out 评估结论；后续以 §7.161 更新后的训练数据重新训练时，应另建产物目录并与此 adapter 区分。

## 7.163 Guqinizer 走手徽位约束解码与实验室服务器代码同步（2026-09-18）

- 推理复核确认 Guqinizer 的系统性缺陷：`SCf7VJzZ` p0001（A100 最终 adapter，2026-09-17 评估）把
  `大指七徽三分挑六弦` 改写为 `注下七徽六分`（终点约 100 音分偏低）、`大指七徽挑四弦` 改为 `历五弦`、
  `七徽三分` 整体漂移为 `七徽六分`；Base 阶段音高全部正确，错音全部由 Guqinizer 润色引入。
  判断与用户一致：几千 token 的轨迹中徽位数字 token 在 SFT 中训练信号不足。
- 新增 `scripts/constrained_decoding/guqinizer_walk_constraint.py`：Guqinizer edit_plan 走手终点约束解码。规则＝终点徽位与
  Base 阶段相同，或在任意弦上（`string_scope=any`，可收紧为 `base`）音高正确（默认 ±50 音分，
  与审计口径一致）。实现为 trie 式 token 白名单 LogitsProcessor：仅在模型自身输出
  `<parameter=jianzi_rows>`/`"jianzi_rows"` 区域内激活（reasoning/`<think>` 不受限），检测
  绰/注/进/退/上/下/浒/淌/引上＋可选左手指后的徽位数字；除 span 内 trie 掩码外，还有
  trigger-pending 预掩码约束徽位整数本身，防止先写错整数再让 span 死锁。逐字符消费与逐 token
  掩码判定同构，含跨 token 合并（如"三分挑"）、UTF-8 字节级增量解码、空掩码安全阀。
  音高计算全部复用 `scripts/audit_jianpu_jianzi_pitch.py`（`parse_hui/position_pitch/parse_jianpu`
  及继承 context 重放），与离线审计口径一致；约束键为 GQS 1.2 连续音序（模型实际书写的行号）。
- `train/scripts/eval_two_stage_score.py` 新增 `--constrain-walk-hui`、`--walk-tolerance-cents`、
  `--walk-string-scope`；仅 Guqinizer 阶段生效，逐轮把 `constraint.stats`（spans/mask_steps/
  violations/fallbacks）与逐行允许集写入 trace 与 stdout 日志。本地测试
  `scripts/constrained_decoding/test_guqinizer_walk_constraint.py` 17 项通过（真实 SCf7VJzZ p0001 允许集、状态机、
  pending 掩码、合并 token、JSON 包络、安全阀）。
- 实验室服务器（219.216.65.119，4×RTX 3090，卡 2）代码同步：r1 失败根因是服务器为"半新半旧"
  快照——HEAD 版 `RealToolRuntime` 按音序解释行号，而旧版 `agents/abc_to_jianzipu/teacher_trajectory.py`
  仍渲染 source 序号（含小节线空档），交叉映射产生 `duplicate jianzi source_index`，Base 16 轮全败。
  已先备份（`~/guqin-agent-backup-20260918/`，210 文件）再将本地工作区的 `ABC_J/scripts/`、`scripts/`、
  `agents/`（rsync --delete，清除 12 个本地已删旧模块，现 25 文件）、`train/scripts/` 全量同步；
  顶层死代码 `~/guqin-agent/abc_to_jianzipu/` 移入备份。同步前服务器 `audit_jianpu_jianzi_pitch.py`
  缺调弦 semitone_offset 双重偏移修复与 §7.161 走手审计，现已对齐本地工作区。
- 卡 2 真实验证（r2，SCf7VJzZ 前 3 phrase，4-bit QLoRA adapter=`inference_artifacts/
  qwen35_9b_lora_final_20260917`）：3/3 phrase 协议有效；约束统计 8 个走手 span 全部完成、
  43 次掩码步、0 mid-token 违规、0 安全阀回退。`注下七徽六分` 类错误消失：p0001 第 2/17 音
  改写为 `注下七徽三分`；用项目音高口径复核 8 个走手终点全部 ≤3.8 音分（A100 基线同类音约
  100 音分）。`历五弦`（右手历弦换弦）与普通位置改写不在本约束范围，仍属已知缺口。
- r3（整曲 21 phrase，any 弦并集版）完成：21/21 协议有效、113 个走手 span 全部完成、
  571 次掩码步、0 违规 0 回退。但按 §7.161 的继承弦口径复核最终走手行：107 个可核验终点中
  37 个在实际发声弦上超差——any 并集允许"在别的弦上音高正确"的终点（如 `绰上七徽九分`
  在弦4 差 13 音分，但左手在弦5，实际差 213 音分），约束过松。
- 依用户决定重设计（同日）：CLI 仅保留 `--constrain-walk-hui` 一个开关，启用即固定配置
  （±50 音分）；允许集＝Base 位置锚点 ∪ **Base 弦**上音高正确 ∪ **当前有效弦**上音高正确。
  当前有效弦来自"当前计划"（Base＋本阶段已应用编辑）按音序的全量重放快照
  （`initial_live`），并在**同一工具调用内**、每行引号闭合时就地 `parse_jianzi` 重放更新
  （`rows_replayed`），不等 edit_plan 真正执行——行号按模型实际书写的连续音序。
  徽外/徽外半支持保留（允许集枚举标签、span 无数字入口、pending 掩码、外/半入终点字符集）。
- r4（base∪live 收紧版）已完成：21/21 phrase 协议有效；104 个走手 span 全部完成、518 次掩码步、
  0 违规 0 回退、187 行在工具调用内重放。双弦口径终验（独立走手按 base 自有弦∪继承弦、前置式按
  文本自带弦）：106 个可核验走手中 77 个正确、23 个为 Base 阶段自身音高错误被"=Base"条款忠实保留、
  6 个存疑中至少 3 个已查明为"写入时 live 弦正确、终稿重放弦漂移"（如 p0012 `进复六徽四分` 写入时
  live=弦4 差 21 音分）或 Base 已错延续（p0009 两行 guqinizer 未改）。
  对照 r3（any 并集）：真实改坏 12/113；基线（A100 无约束）为 `注下七徽六分` 系统性漂移。
- 口径修正与复核（同日，用户质疑后重验）：早期"A100 base 干净（0 warning）vs 3090 base 116
  warning"的对比不成立——warning 标记出自中间轮（终稿可能已自纠），且 A100 租赁机快照根本不产生
  base 音高 warning。以同一项目审计对 base 终稿重打分：A100 前 7 phrase 16/117（13.7%）超差、
  r3 同段 19/117、r4 同段 33/117——同一量级、A100 略优，不存在"0 错 vs 全错"；p0001 在全部
  5 次有效运行（A100/r3/r4/repeat×3 中的 2 次+repeat3）终稿 0 超差。LoRA 加载排除嫌疑：
  `load_adapter_checked` 逐张量严格校验（496 张量、248 lora_B 全非零）7 次运行全过，服务器
  adapter 与本地 A100 训练归档 md5 逐字节一致。可复现性：r4 与 repeat_p1_1/2/3 四次运行
  p0001 base 逐字一致（当前 3090+bnb4bit 环境下 greedy 确定）；r3 与 r4 的 6 行差异对应两轮间
  eval 脚本更新，非随机噪声。错误集中在后段 phrase 与高音区（`3̇` 系统性写成七徽六分而非五徽
  六分），级联上下文（模型自身前段输出）会放大尾部差距。结论：Base 音高错误的真因是"高置信
  音高不匹配 warning 不阻塞提交"（§7.18 设计）在两个环境都存在，下一步应把它升级为必须修正的
  反馈（复用 teacher runner 的 continue 机制）。
  可视化（`ABC_J/scripts/visualize_running_eval_score.py --skip-pull`；标注源
  `ABC_J/modern/SCf7VJzZ/mapped/jianpu_jianzi_readable.json` 按全局 index 对齐生成 references，
  因既有 `reference_trajectories_test.jsonl` 与 `evaluation_pairs_test.jsonl` 均不含该曲）：
  `train/eval_outputs_v3_two_stage/walk_constraint_debug/` 下 `SCf7VJzZ_baseline_a100_view.html`（7
  phrase 无约束）、`SCf7VJzZ_r3_anyunion_view.html`、`SCf7VJzZ_r4_baseflive_view.html`（各 21
  phrase）。剩余改进方向：Base 阶段高置信错音升级为阻塞反馈与跨轮 live 弦漂移（可把上一轮
  文本并入 initial_live 重放，已部分由每轮重建 current_text 覆盖）。
- 教师音高门槛移植进评估（同日，r5）：`eval_two_stage_score.py` 的 `run_stage` 原先"valid 即接受"，
  丢失了教师运行器的 pitch gate（`jianzi_pitch_mismatch` 挂起阻塞接受、直到该事件在看到警告的
  后续轮次被真实改写；教师版另要求先查过候选）。现已移植简化版：`pending_pitch`/`repaired_pitch`
  挂起集合、警告轮不接受、模型停止调用即采纳最后 valid 预览、轮次耗尽回退并记录
  `pitch_pending_at_stop`。更正一处旧记录：Guqinizer 阶段**有** `get_pitch_candidates`
  （`public_tools_for` 明确为修复音高警告而开，§7.20 的"不暴露"是 2026-08-25 旧状态）。
  r5（p0001–p0004，卡 2，同权重同环境）同口径终稿审计：r4 无门 base 13 错/guqinizer 14 错 →
  r5 有门 **8 个阶段终稿全部 0 错**（p0004 重灾区 base 4 轮、guqinizer 4 轮修净；pitch_gate 触发
  5 个警告轮，挂起全部清零后接受，无一靠回退收场；学生自发复现教师"警告→查候选→候选指导下改写"
  的修复模式，如 p0004 行 60 和弦撮按候选改为 `撮（大指十二徽三分二弦按音＋六弦七徽九分）`）。
  代价为轮次增加（约 2→4 轮/阶段）。待办：可把修复认定收紧为教师完整版"先查候选再改"；全曲
  21 phrase 的有门+约束完整跑待做。
- r6（整曲 21 phrase，音高门槛＋走手约束，卡 2）完成：21/21 协议有效。同口径终稿审计：
  r4（无门）base 87/323 错、guqinizer 99/283 错 → **r6 base 38/325 错、guqinizer 18/290 错**
  （走手约束同时统计：115 span 全部完成、535 次掩码步、0 违规 0 回退）。门槛活动：总 151 轮中
  82 轮带警告，仅 6 个阶段轮次耗尽回退且挂起未清（其余全部修净后接受）。剩余错误集中在标量
  解析器无法核验/修复的写法与高音区换弦重定位，后续可评估"候选指导下修复"认定收紧与 Base 侧
  候选引导。可视化新增暗色风格（#181818 系）：`SCf7VJzZ_r6_pitchgate_full_view.html` 及
  baseline/r3/r4/r5 各视图均已在 `train/eval_outputs_v3_two_stage/walk_constraint_debug/` 重生成。
  交接文档同步修正 §7.20/§7.22 的"Guqinizer 不暴露 get_pitch_candidates"旧记载。
- 约束的徽外支持（同日补充）：语料扫描确认 `徽外` 常见（train 1,169 处 / 561 phrase，
  test 537、validation 128），且存在 `注下徽外` 走手终点写法；`徽外半` 语料 0 处但解析支持。
  审计侧（本地工作区 `parse_hui` 字符串分支＋`position_pitch` 以 12.3 锚点减 1/2 半音，
  徽外=开弦+190 音分、徽外半=+90）已随全量同步上服务器。约束白名单侧新增：允许集枚举
  `徽外/徽外半` 标签、=Base 条款认标签锚点（`own_position` 返回 `str` 型徽位）、
  span 增加无数字入口（触发词后直接"徽"且允许集含徽外标签即进入）、pending 掩码要求
  "徽"开头 token 必须前缀匹配允许标签、"外/半"纳入终点字符集（防"徽外半"借退出子句偷渡、
  数字终点后的"半"本属畸形写法一并封禁）。测试 22/22 通过，已同步服务器。
- 允许集口径演进：初版 any 弦并集（每音 15–30 面）在 r3 上被证明过松（见上）；
  现为 base 弦 ∪ 当前有效弦（典型每弦 2–4 面）。弦位不写进谱面、且 guqinizer 可能已改写
  前文弦位，故不硬绑单一弦。

## 7.164 单谱字撮的"主音＋隐含伙伴"审计规则（2026-09-19）

- 实测推翻 §5 的旧印象"和弦音符 100% 双值"：train 标注撮 2,286 个中 813 个（35.6%）为单谱字；
  可解析的 615 个中 513 个同音加厚、45 个八度、仅 26 个主音不匹配——单谱字撮是"主音＋隐含
  伙伴"的合法省略记谱，而非"乱加声部"。
- `scripts/audit_jianpu_jianzi_pitch.py` 的 `audit()` 规则修改：`expected=1（单谱字）且
  actual>1` 时，任一发声成员与谱字 ≤50 音分即判 `matched`（reason=
  `single_symbol_multivoice_main_note`，pairs 记最优成员）；双谱字仍维持 2v2 最优配对；
  `expected>1 且 actual=1`（丢失声部）仍判 `pitch_count_mismatch`。回归测试 42/42 通过
  （含同音/八度/全不匹配/双谱字四例，均已同步服务器并复跑通过）。
- 影响量化：eval 侧 r6 终稿重审为 base 41 / guqinizer 21 错（本曲单谱字撮少，翻转仅 1 行）；
  训练筛选口径按用户决定放行——`filter_training_data.py pitch-select --require-match` 对
  `inferred_gqs_v12_tuningfix_20260914` train 重跑：**3,969 → 4,022（+53 phrase）**，
  新清单 `/tmp/pitch_select_newrule/pitch_eligible_phrase_ids.txt`（尚未用于任何教师生成或
  训练导出；正式采用时应写入 agent_training 固定目录并记录冻结口径变更）。

## 7.165 r7（新撮规则整曲复跑，2026-09-19）

- SCf7VJzZ 整曲 21 phrase、卡 2、同权重同环境，唯一变量为单谱字撮"主音＋隐含伙伴"审计规则：
  终稿错误（项目 `audit()` 口径）base 41→**16**、guqinizer 21→**16**；撮主音放行 1→6 行（与该曲
  6 处模型单谱字撮写法吻合）；警告轮 82→69；两阶段合计错误相对 r4（无门）从 218 降至 32（-85%）。
  走手约束全程 105 span、516 掩码步、0 违规 0 回退。视图：
  `train/eval_outputs_v3_two_stage/walk_constraint_debug/SCf7VJzZ_r7_courule_view.html`（暗色）。
  撮提示词与知识库改动经核验只作用于教师私有提示（`private_instruction["rules"]` 与私有知识注入），
  对评估链路零影响；r7 对比口径干净。

## 7.166 S2T7rDyJ 首次整曲评估与代码重组（2026-09-19/20）

- `ABC_J/final/S2T7rDyJ` 首次跑两阶段评估（19 phrase，卡 0，全栈：音高门＋走手约束＋单谱字撮
  主音规则，A100 最终 adapter）：19/19 协议有效；终稿 base **0/304** 错、guqinizer 4/253 错
  （p0003/p0012/p0019 各 1-2 行）；警告轮 17、挂起全部修净（无轮次耗尽回退）；走手约束 45 span、
  228 掩码步、0 违规 0 回退。视图（暗色）：
  `train/eval_outputs_v3_two_stage/walk_constraint_debug/S2T7rDyJ_full19_view.html`。
- 代码重组：约束解码模块移入 `scripts/constrained_decoding/`
  （`scripts/constrained_decoding/guqinizer_walk_constraint.py`＋测试，提交 `7ced238a`，本地与服务器已同步）；
  `audit_jianpu_jianzi_pitch.py` 保留在 `scripts/`——它是 12+ 调用方共享的音高审计底座
  （教师生成、训练筛选、离线审计、抽取器），非约束解码专属。
- GPU 备注：卡 2 曾被其他用户占用导致一次启动失败，评估改在卡 0 完成；启动前应先查卡。

## 7.167 约束解码 ASCII 逃逸修复与 S2T7rDyJ 复跑（2026-09-20）

- 用户在 S2T7rDyJ 首轮（§7.166）推理中发现 `绰上 eight`、`绰上 seven nine`、`退 ten`、`注下 nine`
  等英文混入——根因是约束解码的逃逸口：pending 掩码的"中止子句"与 span 自由退出均放行非数字
  开头 token（含前导空格的 " seven"），模型在被禁徽位后用英文拼数字绕过白名单（甚至允许集含
  `七徽九分` 时仍写 seven nine——序列内一旦建立该模式即被复制）。
- 修复（提交 `1678855c`）：无歧义触发头（绰上/注下/进复/退复/浒上/引上/绰/注/淌）后禁止一切
  中止、必须写允许集终点；歧义单字头（上/下/进/退/浒）保留中文中止（false-positive 退路）；
  全部位置（pending 中止、span 退出）封堵 ASCII 字母/数字/空格开头 token，JSON 结构符（引号/
  逗号/括号）仍放行。测试 26/26（含被禁整数时 greedy 端到端被迫选允许面的回归）。
- S2T7rDyJ r2 复跑（卡 0，同权重）：19/19 有效；**英文残留 7→0**（p0001 四行、p0008 三行全部
  变为正常减字，如 `名指七徽九分勾三弦`）；走手 span 45→73（模型更倾向写走手终点而非回避）、
  掩码步 228→317、0 违规 0 回退。终稿口径 guq 错 4→12：修复封闭了逃逸口后，原本"以垃圾文本
  逃过审计"的行转为真实减字（不可解析/超差计入显式错误），错误可见性上升而非质量下降；
  base 0→2 为轨迹波动。视图：`walk_constraint_debug/S2T7rDyJ_r2_escapefix_view.html`。

## 7.168 GLM 教师重跑的网络与限流排查（2026-09-23）

- 场景：重跑 `SaljUbT2` 全部 122 个 phrase 的 Fingering + Guqinizer，以使历史 `edit_plan` 审计
  与当前音高审计器一致。8 路、零间隔运行出现大量失败。
- 实测：在实际网络环境中，对 `.env` 的 `GLM_BASE_URL` 进行探测，**直连**（`curl --noproxy '*'`）
  与当前环境的**代理路径**均返回 HTTP 200，连接约 0.002 秒、总耗时约 0.10 秒。因此 GLM 网络
  可达，代理不是必要条件，也不是本次 API error 的根因。受限沙箱内曾出现直连 DNS 无法解析、默认
  代理 `127.0.0.1:7897` 连接失败；该现象不代表实际执行环境，不能据此判断生产网络不可用。
- 失败证据：8 路日志中的主因是 GLM `RateLimitError 429`、代码 `1302`（账户请求速率限制），不是
  `APIConnectionError`/DNS/timeout；另有少量模型输出未满足教师 envelope（缺 JSON、或
  `tool_calls` 结构错误）。每个 worker 各自退避，8 路同时恢复会形成重试尖峰，进一步触发 429。
- 推荐：教师批量生成默认显式**禁用代理**，即在启动命令中设置
  `HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= NO_PROXY=* no_proxy=*`；
  并使用全局受控的 2–3 路并发/节流。仅靠每 worker 的 `_RateLimitedMessages` 是进程内限速，不能
  限制多进程总 QPS；若以后仍需 7–8 路，应先实现跨进程共享限速器和 429 全局 cooldown，而不要仅
  把 `--min-interval` 设为 0。
- 当前运行：此前各轮累计已有 61 条两阶段成功产物保留；其余 61 条正在无代理 5 路任务
  `ABC_J/agent_training/pitch_eligible_two_or_half_SaljUbT2_auditrefresh_20260923_retry5/` 中断点重跑，
  screen 名 `saljubt2_auditrefresh_5way_20260923`。总控日志仅记录 worker 启停；应 tail
  `.parallel_workers/worker_*/worker.log` 查看每个 phrase 与 transient retry。生成器已补充重试日志：
  今后每条 transient 日志会携带 `cause=<异常类型和详情>`，不再只显示笼统标签。
- 全量启动（同日）：用户要求覆盖当前 `pitch_eligible_gqs_v12_train` 的**全部**曲谱；输入冻结为
  176 首、4,691 phrase。任务目录
  `ABC_J/agent_training/pitch_eligible_two_or_half_all_scores_auditrefresh_20260923_raw/`，screen
  `all_scores_auditrefresh_5way_20260923`。使用 5 个 worker（958/921/901/963/948 phrase），显式无代理。
  为避免每 worker 独立退避导致的 429 尖峰，生成器 `_RateLimitedMessages` 新增可选跨进程锁文件
  （`GLM_GLOBAL_RATE_LIMIT_STATE`）和全局请求启动间隔（当前 `GLM_GLOBAL_MIN_INTERVAL=0.35` 秒）；
  轨迹仍按 phrase 并发，只有 API 请求被全局错开，支持断点续跑。

## 7.169 J00124：GLM harmonic-parse-fix 终版 SFT（2026-09-24）

- Jobber `J00124` 已在实验室服务器 `219.216.65.119` 成功完成：4×RTX 3090，
  `cutoff_len=8192`、单卡 batch 1、梯度累积 2、有效 batch 8，3 epoch / 2,886 step。
  最终 `train_loss=0.4569`，耗时 9:25:59。
- 远端最终 LoRA 输出：
  `~/guqin-agent/train/output/guqin_sft_glm_final_harmonicparsefix_20260924_ctx8192_bs8_accum2/`。
  顶层为可推理产物；`checkpoint-*` 为中间恢复点。
- 已下载至本地：
  `train/artifacts/J00124_guqin_sft_glm_final_harmonicparsefix_20260924/`（116 MB）。其中包括
  `adapter_model.safetensors`（83 MB）、`adapter_config.json`、tokenizer/chat template、
  `training_loss.png`、`trainer_state.json` 和 `J00124_training.log`。

## 7.170 J00128：pitch-eligible two-or-half 全量 SFT（2026-09-25）

- Jobber `J00128` 已在实验室服务器 `219.216.65.119` 成功完成：4×RTX 3090，数据集
  `train/data/guqin_agent_sft_pitch_eligible_two_or_half_20260925_longrerun/`，`cutoff_len=8192`、
  单卡 batch 1、梯度累积 2、有效 batch 8、QLoRA 4-bit、bf16、Liger、3 epoch / 2,703 step。
- 训练参数：learning rate `2e-4`、AdamW、cosine scheduler、warmup ratio `0.03`；最终
  `train_loss=0.5697428`，训练耗时 `9:10:06.84`。loss 曲线由 Trainer 保存为 `training_loss.png`。
- 远端最终 adapter：
  `~/guqin-agent/train/output/guqin_sft_pitch_eligible_twohalf_longrerun_20260925_ctx8192_bs8/`；
  最后可恢复 checkpoint 为 `checkpoint-2703/`（含 LoRA、optimizer、scheduler、trainer state）。
  Jobber 日志：`~/code/training/runtime/gpu_queue/logs/J00128.log`。
- 本地归档：
  `train/artifacts/J00128_guqin_sft_pitch_eligible_twohalf_longrerun_20260925/`，含最终 adapter、
  tokenizer/config、`training_loss.png`、`trainer_log.jsonl`、`trainer_state.json` 和完整训练日志；
  下载时排除了中间 checkpoint。Adapter SHA-256：
  `492486b553178149ca7bc25abeb59ffd706d0c0aba548e496c3e4fc0e1610e4b`（与服务器一致）。
- 用户要求基于该训练继续 2 epoch：已准备独立续训配置
  `~/guqin-agent/train/runtime/guqin_sft_pitch_eligible_twohalf_longrerun_20260925_continue2/train_lora.yaml`，
  从 `checkpoint-2703` 恢复（包括 optimizer/scheduler/RNG），训练总 epoch 设为 5，因此续训预期约
  1,802 optimizer steps；仍用相同数据、8192 cutoff、batch 1×4 卡×累积 2、bf16、QLoRA 4-bit、
  Liger、learning rate `2e-4`、AdamW、cosine scheduler、warmup ratio `0.03`。scheduler 按 5 epoch
  总步数延展；续训的 loss 曲线和 adapter 写入独立目录
  `~/guqin-agent/train/output/guqin_sft_pitch_eligible_twohalf_longrerun_20260925_continue2_ctx8192_bs8/`，
  不覆盖 J00128 原产物。
- Jobber `J00130`（`guqin-sft-longrerun-continue2-20260925`）已提交，申请 4 卡，当前 `queued`；
  等待四卡空闲，不抢占其他任务。日志：
  `~/code/training/runtime/gpu_queue/logs/J00130.log`。

## 7.171 单阶段教师轨迹的过滤输入（2026-09-26）

- 单阶段训练/样例生成必须以
  `ABC_J/agent_training/pitch_eligible_two_or_half_v13_train/pitch_eligible_phrase_ids_train.txt`
  作为唯一 phrase 白名单；不得直接遍历完整
  `inferred_trajectories_train.jsonl`。
- 该清单已同时施加“每段至少两个音高匹配或匹配率至少一半”、全空标注排除，以及普通减字后连续五个以上
  空发音行排除。单阶段输出在脱敏、合并或 SFT 导出前必须再次按此清单 fail-closed 过滤。

## 7.172 测试 bench 的高可信音高曲谱白名单（2026-09-26）

- 测试源：`ABC_J/agent_training/inferred_gqs_v12_tuningfix_20260914/inferred_trajectories_test.jsonl`。
- 用当前 `scripts/audit_training_phase_pitch_parse_stats.py` 复用的 50 cents 音高审计，按完整曲谱聚合标注减字；分母仅包含审计可判为 `matched` 或 `mismatched` 的音，未解析行不计入分母。
- 严格条件为 `matched / parseable > 0.70`，44 首测试曲中保留 39 首。可直接给评测脚本使用的 ID 清单：
  `ABC_J/agent_training/test_pitch_match_over70_20260926_score_ids.txt`。
- 未入选：`S074gS1m`（51.69%）、`SdOedBkd`（47.62%）、`SWk1GWt2`（42.99%）、`SDcE3Geq`（31.42%），以及无可解析音的 `SCwngKTr`。
- 公开推理 bench 的 `train/eval_inputs_v2_text_protocol/test_runtime.jsonl` 只含其中 37 首：音高源中的 `S0ejf3Pz` 与 `SbgkbNBF` 并没有对应公开输入，不能在无标注评测时临时补入。实际 bench 清单为
  `train/eval_inputs_v2_text_protocol/test_pitch_match_over70_20260926_score_ids.txt`。

## 7.173 单阶段 v13 全量 SFT（2026-09-28）

- 数据：`train/data/guqin_agent_sft_single_stage_v13_20260928/guqin_agent_train.jsonl`；来自严格按
  `ABC_J/agent_training/pitch_eligible_two_or_half_v13_train/pitch_eligible_phrase_ids_train.txt`
  过滤后的 3,604 条脱敏单阶段教师轨迹。SFT 数据 SHA-256：
  `74408fbd1c4895c8ffe04cffb4bacd0b6606c63708a66fb133fcc038752032f5`。
- 配置：`train/configs/qwen35_9b_lora_4xa100_bs2_acc1_ctx8192_single_stage_v13_20260928.yaml`；Qwen3.5-9B，QLoRA 4-bit、bf16、Liger、`cutoff_len=8192`，4×A100 40GB，单卡 batch 2、梯度累积 1、有效 batch 8、3 epoch / 1,353 steps。
- 训练因云端余额中断后，从 `checkpoint-1300` 继续并完成至 `checkpoint-1353`。最终
  `train_loss=0.0181616`，总训练时间 `952.42s`（15:52），吞吐 `1.421 step/s`。
- 远端输出：`/data/guqin-agent/outputs/qwen35_9b_lora_4xa100_bs2_acc1_ctx8192_single_stage_v13_20260928/`；末尾可恢复 checkpoint 为 `checkpoint-1353/`。
- 本地归档：`train/artifacts/qwen35_9b_lora_4xa100_bs2_acc1_ctx8192_single_stage_v13_20260928/`，含最终 `adapter_model.safetensors`、adapter 配置、`trainer_state.json`、`training_loss.png`、`train_results.json` 和 `all_results.json`。adapter SHA-256：
  `b327d97a90477c5b4b393596781540738d9f1a16ce508febcfa22a2eac436502`。
- 新服务器的非交互 SSH 环境缺少 `/data/miniconda/envs/torch/bin`，会导致 LLaMA Factory 子进程找不到 `torchrun`。启动命令必须先设置：`export PATH=/data/miniconda/envs/torch/bin:$PATH`。

## 7.174 调弦源修复后的局部轨迹重生成（2026-10-01）

- 唯一正确源：`ABC_J/agent_training/inferred_gqs_v12_tuningfix_20260914/`；该源 4,691 个 phrase 的 `normalized_tuning` 与曲谱元数据 mismatch 为 0。
- 旧筛选产物 `pitch_eligible_gqs_v12_train` 和 `pitch_eligible_two_or_half_v13_train` 不得再作为轨迹输入；它们来自已删除的旧源，含错误固化的调弦字段。
- 从正确源按“每阶段至少匹配 2 个音高或匹配率至少一半”及普通前置减字后连续 5 个以上空音排除规则重新筛选：train 3,604、validation 522、test 1,153。
- 新 train 清单：`ABC_J/agent_training/pitch_eligible_two_or_half_v13_tuningfix_20261001/pitch_eligible_phrase_ids_train.txt`。
- 旧 train 清单与新清单均为 3,604 个相同 ID；逐条比较输入后，2,602 条完全相同，1,002 条受调弦或 phrase handoff 审计上下文变化影响，后者只重生成并替换。
- 1,002 条局部重生成原始输出目录：`ABC_J/agent_training/two_stage_v13_tuningfix_20261001_raw/`；生成完成后必须脱敏、按 `sample_id` 替换旧记录，并再次校验调弦 mismatch 为 0。未受影响的 2,602 条不得重复调用教师模型。
