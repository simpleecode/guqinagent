# 项目简介

本项目面向 AI for Guqin 研究，目标是将 ABC notation（机器可读简谱）自动转换为音高准确、可演奏、指法连贯且具有古琴“韵味”的减字谱，并形成以 WWW 为目标的论文（草稿见 `paper/main.tex`，系统名 GuqinAgent）。详细目标见 `DOCS/target.md`。

## 当前阶段

数据采集与解析管线已收官：授权谱面经 Gadget 版 App + Frida runtime hook 采集（优先读 `0x63f220` 的 `source_jab_event` 完整事件列表，WZa 窗口拼接仅作回退），由 `scripts/extract_score_runtime_windows.py` 重建、`scripts/extract_jianpu_jianzi.py` 输出映射 JSON / Markdown / ABC。当前主线是教师轨迹蒸馏：GLM 教师生产工具调用轨迹 → 学生 SFT（`train/configs/`）→ 事件级评估（`evaluation/`）→ 论文实验。

## 核心代码

- `agents/abc_to_jianzipu/` — ABC 解析、指法路由、patch 回放、减字渲染、参考逆向（inverse_*）、两阶段 prompt 与知识注入（`teacher_trajectory.py`、`knowledge/`）。
- `agents/ToolRuntime/runtime.py` — 公开 4 工具（`list_context` / `expand_context` / `get_pitch_candidates` / `edit_plan`）的唯一实现，含 `validate_jianzi_only`、`replay_patches` 与告警计算。
- `ABC_J/scripts/generate_teacher_tool_trajectories.py` — 教师轨迹生产：GLM 教师在私有参考 GQS 引导下真实执行工具（两阶段 fingering→guqinization / single_stage），经解析恢复、防泄漏与质量审计后导出公开训练数据 `messages_train.jsonl` 与私有 audit。
- `ABC_J/scripts/run_teacher_batch_parallel.py` — 教师轨迹并行协调器：不重叠分片、worker 目录隔离、按 ID 去重合并，支持 `--merge-only` / `--retry-failed`。
- `train/` — 学生 LoRA SFT 管线（qwen3.5-9b），configs 命名含模型/硬件/数据变体/日期。
- `evaluation/` — 事件级评估：PitchAcc@50c、指法/装饰音 P/R/F1、规则违规率。
- `paper/` — 论文草稿 `main.tex`（benchmark、warning-guided inference、两阶段 agent、逆向轨迹蒸馏）。

## Python 环境

- 统一使用 Conda 环境 `guqin`（Python 3.11）；不要默认依赖 `base` 或旧的 `sitong` 环境。
- 交互式终端先执行 `conda activate guqin`。Agent 和自动化脚本优先使用 `conda run -n guqin python ...`，避免依赖终端是否已经激活。
- 已安装项目依赖：`capstone`、`pyelftools`、`frida`/`frida-tools`、`mitmproxy`、`PyYAML`。
- 若需从零重建：

```powershell
conda create -n guqin python=3.11 pip -y
conda run -n guqin python -m pip install capstone pyelftools frida-tools mitmproxy PyYAML
```

- ABC→减字谱 Agent 编排层单独使用 `guqin-agent` 环境。LangGraph 的网络依赖与
  `guqin` 中 Frida/mitmproxy 的版本约束冲突，不要把两套依赖装进同一环境：

```powershell
conda create -n guqin-agent python=3.11 pip -y
conda run -n guqin-agent python -m pip install -r requirements-agent.txt
```

- 修改音高映射 skill 后运行：

```powershell
conda run -n guqin python -m unittest discover -s skills\guqin-pitch-mapper\scripts -p "test_*.py" -v
conda run -n guqin python C:\Users\30343\.codex\skills\.system\skill-creator\scripts\quick_validate.py skills\guqin-pitch-mapper
```

## 已完成

- 数据采集与简谱—减字谱解析管线（正调与调弦偏移解释，撮/撞/十徽/徽外及复合符号闭合，窗口边界缺音修复）。
- 教师轨迹生产与并行调度、无效工具调用清洗、数据切分脚本链。
- GEPA prompt 进化 baseline 与三臂试点（handbook +0.381；GEPA 试点规模下无增益）。

## 待解决

- 填充论文实验表格（Direct / single-stage / 两阶段对比，warning 消融），完成学生模型训练与评估迭代。

## 工作原则

- 论文主线优先；数据采集和逆向是基础设施，不应无限扩张。
- `raw_data.json` 作为原始证据不得静默改写；修正规则单独记录、可重放。
- 未知编码保留原值，结论区分“已验证 / 推断 / 未验证”。
- 修改提取逻辑后，用 `batch/SYuY17FF/out/raw_data.json` 回归检查输出。
- 完整采集说明见 `AGENT_REPRODUCTION_GUIDE.md`。

## 教师轨迹并发任务监控

`ABC_J/scripts/run_teacher_batch_parallel.py` 的 coordinator 日志只记录 worker 的启动、退出和最终汇总；实时 tqdm 进度条写入各 worker 的 `worker.log`。
每次运行这个代码时，记得给用户tail带实时进度条的命令，如
```
cd /Volumes/F/work/Gunqin-agent
tail -f ABC_J/agent_training/pitch_eligible_two_or_half_tuningfix_affected_20260924_guqinizer_raw/.parallel_workers/fast7_1000ms_*/worker.log
```
