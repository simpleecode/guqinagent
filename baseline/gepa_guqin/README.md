# GEPA × Guqin Agent(single_stage prompt 进化基线)

把 `baseline/gepa`(GEPA 反思式提示词进化框架)接入本仓库的 ABC→减字谱
single_stage agent,作为 **prompt-only 基线**:GLM-5.3(API)+ 进化出的系统
提示词,与 SFT Qwen3.5-9B 对比。最终数字走现有密封测试集评测管线。

## 交付状态(2026-10-08,交接快照)

**已验证就绪**:数据协议(新划分 + 密封 test 导出)、候选自身 handoff 的
顺序 rollout(与生产同代码路径,泄漏审计 0)、私有评测器(完美预测精确
1.0)、固定 73 条目技法手册(教师同款匹配语义,`sha256 12dcc13b…`)、
cache/断点续跑、API/token 计数。

**最终配置下的三臂 pilot**(固定 dev split seed 42:train 145 / val 56,
证据在 `runs/pilot_final_20261008/`):

| 臂 | dev val 分数 | 成本(actor calls / tokens in+out) |
|---|---|---|
| vanilla ReAct(无手册) | 0.4699 | 160 / 213K + 95K |
| ReAct + Handbook(= GEPA 种子) | **0.8513** | 含于 GEPA 臂种子评测 |
| GEPA + Handbook(200 budget,3 提案) | **0.8513(未超越种子)** | 2480 / 4.03M + 1.23M |

结论:**手册增益确认(+0.381)**;GEPA 在此小尺度(56 val、3 次提案,
0.8428/0.8002/0.8317 均未过接受线)未再提升,不足以支撑 2000 预算决策。
另注意:242 次指标调用消耗 2480 次 actor 调用 —— 自身 handoff 语义下
minibatch 抽到长曲靠后短语会拖完整条前缀(前缀不占指标预算但占 API 成本)。

**建议的下一步(交接给学弟)**:
1. 先做 minibatch 采样优化(优先抽前缀已缓存/位置靠前的短语),把每次
   迭代的 API 成本从 ~10 calls/metric-call 压回 ~4;
2. 用更大 minibatch(6)与 600–800 预算在中尺度 dev split 上重试 GEPA,
   若仍无法超越 0.8513 则以 "ReAct + Handbook" 为论文的 prompt-only 基线,
   GEPA 作为负结果/消融报告;
3. 正式密封 test 数字用 `run_final_eval`(带/不带 `--prompt-json`)。

**数据依赖(不在 git 内)**:新划分数据在
`ABC_J/agent_training/renderfix_train_test_20261007/pitch_eligible_two_or_half/`
(~471MB,gitignore 挡住,需 U 盘/网盘传递);`train/eval_inputs_v2_text_protocol/renderfix_20261007/`
的密封协议文件可由 `export_eval_inputs.py` 从上述数据一键重导。`.env` 需要
`GLM_API_KEY/GLM_BASE_URL/GLM_MODEL`。


## 结构

| 文件 | 作用 |
|---|---|
| `data.py` | 从训练轨迹切 GEPA 优化 split(pitch-eligible 过滤、参考 handoff 注入、按曲对半切 train/val) |
| `export_eval_inputs.py` | 从新划分的 test 半边导出密封评测协议文件(公开 runtime 输入 + 参考对) |
| `glm_client.py` | Anthropic 兼容 GLM 客户端(限速/重试)、native tool_use 后端、gepa 反思 LM 适配 |
| `rollout.py` | single_stage agent 循环,镜像 `eval_two_stage_score.py::run_stage_steps` 的门控;系统提示词是参数 |
| `metric.py` | per-phrase 打分:参考前段(预测=参考)+ 当前段预测经 `structured_events_for_score` 重放状态后按现有 `metrics()` 汇总,加权组合分 + 确定性反馈 |
| `adapter.py` | `GEPAAdapter` 实现(evaluate / make_reflective_dataset) |
| `run_optimize.py` | GEPA 优化入口 |
| `run_final_eval.py` | 用胜出提示词在密封 test 上按生产顺序(自-handoff)出 predictions |

## 环境

```bash
python3 -m venv .venv-gepa
.venv-gepa/bin/pip install -r requirements-agent.txt -e baseline/gepa litellm
```

`.env` 需要 `GLM_API_KEY` / `GLM_BASE_URL` / `GLM_MODEL`(与 teacher 管线同
一套凭据)。默认按 0.5s 间隔调用 API,可用 `--min-interval` 调整。

## 固定技法手册(公平性设计)

`knowledge/complex_fingering_handbook.md` 是**所有 baseline 共享、GEPA 不可修改**
的固定上下文:由 `build_handbook.py` 用**教师私有注入同款的匹配语义**
(`matched_compound_gesture_knowledge`:全部斜杠别名的别名表 + 对参考渲染文本
`jianzi_text`/`text` 的最长匹配 + 子串吞并,如 掐撮三声 吞并 掐撮;一个别名可
带出多条目,如 撮 与 大撮/撮;按知识库 indexed_name 聚合计数)从
`agents/abc_to_jianzipu/knowledge/complex_fingering_explanations_v3_with_effects.jsonl`
中筛选**训练参考标注里实际出现的条目**(当前 73 个:上/绰上、掐起/滔起、注下、
猱、撞、罨、历、吟、撮、大撮/撮、抹挑、带起、就、勾剔、打、绰、泛止、爪起、
少息、拂、分开、如一、跪、至、轮、涓、滚、淌、进复、应合、不动、掐撮三声、
浒、打圆、泼剌、唤、推出、逗、双弹、泼、同声、急、往来、反撮、虚罨、急猱、
急吟、细吟、放合、飞吟、背锁、全扶、抹勾、长猱、半轮、剌、退复、吟滑、荡猱、
打摘、从头再作、长锁、短锁、圆搂、游吟、慢、连涓、弹、索铃、掐撮、撞猱、
双吟、三弹、拨剌)生成,只含定义/方向/前置条件/组合规则/效果等动作知识,
不含任何旋律触发启发式或样本级建议(伴随 `.manifest.json` 记录筛选明细与
每条目命中的段落数)。

系统提示词的组成始终是:

```text
system = [GEPA 候选策略提示词] + "\n\n" + [固定技法手册]
```

- `handbook.py::compose_system` 是唯一拼接点;GEPA 的 candidate 与反思数据集
  只含策略文本,handbook 不进入 mutation;
- `run_final_eval` 不带 `--prompt-json` 时 = "Strong ReAct + Handbook" 基线
  (生产种子策略 + 同一份手册);带 `--prompt-json` = "GEPA + Handbook"。
  两组对比回答的是:拥有相同长尾动作知识的前提下,prompt evolution 能否学会
  正确使用它们(对照 trajectory training 的 SFT)。

## 运行

数据划分使用 2026-10-07 的 `renderfix_train_test_20261007/pitch_eligible_two_or_half`
(train 4316 条/191 曲,test 1379 条/43 曲,曲级零重叠)。旧的
`evaluation_text_protocol_v2` 参考文件已被该划分取代,先从 test 半边导出
密封评测协议:

### 0. 导出密封 test 协议文件(一次性)

```bash
.venv-gepa/bin/python -m baseline.gepa_guqin.export_eval_inputs \
    --out-dir train/eval_inputs_v2_text_protocol/renderfix_20261007
```

产出 `test_runtime.jsonl`(公开输入,无参考)与
`evaluation_pairs_test.jsonl`(参考对,给 `evaluation.run_eval`)。

### 1. 切优化 split(只碰训练曲,不碰 test)

```bash
.venv-gepa/bin/python -m baseline.gepa_guqin.data \
    --out-dir baseline/gepa_guqin/data/gepa_split_v1 \
    --phrases 600            # 按曲采样,大约取到 ~700 条
```

默认源 = 新 train 半边;自动过滤 `pitch_eligible_phrase_ids_train.txt`
之外的短语,并校验与 test 半边无同曲泄漏;输出 `examples_train/val.jsonl` +
`historical.jsonl` + `manifest.json`。

### 2. GEPA 优化

```bash
.venv-gepa/bin/python -m baseline.gepa_guqin.run_optimize \
    --data-dir baseline/gepa_guqin/data/gepa_split_v1 \
    --run-dir   baseline/gepa_guqin/runs/opt_v1 \
    --max-metric-calls 2000
```

- 种子候选 = 生产提示词 `public_system_for("single_stage")`;组件名 `single_stage`。
- 组合分权重(默认 `0.45*pitch + 0.15*tone_type + 0.15*指法F1均值 + 0.10*ornament + 0.15*(1-违规率)`)
  可用 `--weights '{"pitch":0.45,...}'` 覆盖。
- 断点续跑:同 `--run-dir` 重跑即从 `gepa_state.bin` 恢复;随时
  `touch <run-dir>/gepa.stop` 优雅停止。
- 产物:`best_candidate.json`(最优提示词 + val 分)、`run_log.txt`、
  `adapter_log.jsonl`(每短语打分流水)。

预算参考:full_eval 策略下每次提案 = minibatch(默认 3)+ 全量 val;
val 约 150 条时,2000 次调用 ≈ 十几轮提案。单短语 rollout ≈ 3–8 次
API 调用。

### 3. 密封 test 最终评测

```bash
.venv-gepa/bin/python -m baseline.gepa_guqin.run_final_eval \
    --prompt-json baseline/gepa_guqin/runs/opt_v1/best_candidate.json \
    --output    baseline/gepa_guqin/runs/opt_v1/predictions_test.jsonl \
    --resume
```

(省略 `--prompt-json` 则评生产种子提示词。)输出兼容
`agent-eval-prediction-1.3`;随后走标准汇报:

```bash
.venv-gepa/bin/python -m evaluation.run_eval \
    --pred baseline/gepa_guqin/runs/opt_v1/predictions_test.jsonl \
    --reference train/eval_inputs_v2_text_protocol/renderfix_20261007/evaluation_pairs_test.jsonl \
    --experiment gepa_single_stage_v1 --model glm-5.3+gepa
```

## 协议要点(数据流边界)

- **actor 上下文只含候选自身产物**:优化期每个 rollout 的 user_prompt
  【只读前一段】、`expand_context`/`list_context` 可见历史、泛音区间标志,
  全部来自**同一候选在同一首曲上此前实际生成的输出**(`rollout.py::
  ScoreRunState`,与 `run_final_eval`/生产串行循环同一条代码路径)。某段
  协议失败即毒化该曲后续段(镜像生产 "break" 语义)。prefix rollout 按
  (候选文本, 短语) 缓存复用,不占 GEPA metric-call 预算(单独计入
  `adapter.prefix_rollouts`)。
- **reference 只存在于私有评测器**:`metric.py::phrase_report` 内部用参考
  全前缀重放(预测=参考的完美前缀)计算密封指标,并按 data.py 预计算的
  参考自审计剔除参考自身不达标的事件;其输出只有分数与确定性计数反馈。
  参考谱文本绝不进入 actor 的 prompt、工具结果或反思数据集。
- 反思数据集只包含:公开 user_prompt(含候选自身的 handoff)、模型自己的
  jianzi_rows 与最终预览、确定性反馈。`adapter._audit_leakage` 持续校验
  Feedback 中不出现当前段参考文本(feedback_leaks 必须为 0)。
- 门控与生产对齐:音高警告 follow-up、重复提交即收、轮次用尽取最后有效预
  览、attempts 重试(attempt 0 贪心、attempt 1 采样)。
