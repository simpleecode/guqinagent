# ABC → 减字谱 Agent 框架

这是一个可运行的 MVP。它把不会变化的音乐计算放进版本化 skill，把需要判断的
角色放进可替换的 agent backend，并用 LangGraph 管理状态、乐句 worker、汇合、
审计和有限修复。

## 当前流水线

```text
ABC
  → 确定性解析
  → 乐曲分析 agent（未配置模型时明确跳过）
  → 弦/徽候选
  → 每个 phrase 独立 worker：Top-K 路径
  → 全曲边界动态规划
  → PerformancePlan
  → 指法优化 agent（可主动查询候选和计算任意弦徽音高）
  → 确定性减字谱编译
  → 风格 critic agent（未配置模型时明确跳过）
  → 音高、覆盖率和编译审计
```

当前生成结果以“按音主干基线”为起点。指法优化 agent 可以把候选 Skill 当作参考，
也可以提出候选集外的徽位；候选不是白名单。所有实际写入 PerformancePlan 的方案
都会由独立音高函数重新验证。吟猱绰注、掐起、撞、完整泛音段等仍属于后续古琴化层。

## 环境与运行

Agent 依赖放在独立环境中，以免与采集环境的 Frida/mitmproxy 版本冲突：

```powershell
conda create -n guqin-agent python=3.11 pip -y
conda run -n guqin-agent python -m pip install -r requirements-agent.txt
conda run -n guqin-agent python -m agents.abc_to_jianzipu.cli input.abc
```

启用 `.env` 中配置的 MiniMax 角色 agent：

```powershell
conda run -n guqin-agent python -m agents.abc_to_jianzipu.cli input.abc `
  --model-backend minimax
```

只做一次不回显密钥的连通性测试：

```powershell
conda run -n guqin-agent python -m agents.abc_to_jianzipu.model_smoke
```

常用参数：

```powershell
conda run -n guqin-agent python -m agents.abc_to_jianzipu.cli input.abc `
  --out runs/abc_to_jianzipu `
  --tuning-name 正调 `
  --open-midi 48,50,53,55,57,60,62 `
  --top-k 5
```

每次任务会生成 `input.abc`、规范化事件、候选位置、乐句 Top-K、演奏计划、减字谱
IR、Markdown、JSON、审计报告和 provenance。默认使用内存 checkpoint；所有关键结果
同时原子写入任务目录。生产运行可以向 `AbcToJianzipuOrchestrator` 注入 SQLite/Postgres
checkpointer。

## 接入模型

实现 `backends.AgentBackend.run(task, prompt, tools)`，并把实例作为 `backend=` 传给
`AbcToJianzipuOrchestrator`。角色由 `RoleAgentFactory` 创建，输入使用 `TaskEnvelope`，
输出必须是 `TaskResult`；prompt 和预期 schema 都按角色固定。工具经角色白名单开放，
调用轨迹写入 `fingering_optimization.json`。模型可以不采用候选生成器的建议，但不能
绕过最终音高与结构审计。

## 测试

```powershell
conda run -n guqin-agent python -m unittest agents.abc_to_jianzipu.tests.test_framework -v
```
