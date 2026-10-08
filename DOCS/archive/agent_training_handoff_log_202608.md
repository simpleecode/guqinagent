# agent_training_handoff 历史日志归档（2026-08）

> 本文件自主文档 DOCS/agent_training_handoff.md 归档而来，包含 §7.19–§7.121
> （2026-08-24 至 2026-08-31）的逐日验证日志。小节编号沿袭当时记录，存在
> 跳号与重复（如 7.35/7.40/7.41 错序插入、7.58/7.59/7.136 重复），为保留
> 与正文交叉引用的一致性未作重排。当前工程状态以主文档为准。

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
- 公开/私有工具 schema 已统一，Fingering 均能看到 `get_pitch_candidates`。本节原记载
  “Guqinizer 不暴露该工具”为当时状态；后来为配合音高警告修复，Guqinizer 阶段同样暴露该
  只读工具（见 `public_tools_for` 注释与 §7.163）。
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
- 只读 `get_pitch_candidates` 返回方式、弦和徽位，因为这些是候选取音证据，不是可维护状态；
  该工具两个阶段均可见（Guqinizer 用于修复音高警告，2026-09 修正，见 §7.163）。
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

- 新增 `ABC_J/scripts/redact_teacher_reasoning.py`。用法：`python ABC_J/scripts/redact_teacher_reasoning.py --input-dir <教师轨迹目录> --output-dir <脱敏目录> --model <模型>`。
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

- `ABC_J/scripts/redact_teacher_reasoning.py` 现在逐样本追加写入 `messages_train.jsonl` 和 `reasoning_redaction_audit.jsonl`，不再等全部任务结束后一次性写文件；中断时已完成样本可直接复用，避免重复调用 API。
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
- 曾因生成器默认截断而产生的连接失败审计保留在
  `messages_quality_filter_minimal_retry_v1_fingering/`，不计入最终合并结果；生成器现已移除
  `--limit`，只按显式 ID、分片或输入池选择工作量。

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

