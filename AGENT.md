# 项目简介

本项目面向 AI for Guqin 研究，目标是将 ABC notation 自动转换为音高准确、可演奏、指法连贯且具有古琴“韵味”的减字谱，并形成以 WWW 为目标的论文。详细目标见 `docs/target.md`。

## 当前阶段

目前重点是构建高质量 ABC—减字谱配对数据和可运行基线。授权谱面经 Gadget 版 App + Frida runtime hook 采集；优先从 `0x63f220` 的 `source_jab_event` 读取窗口切分前的完整事件列表，WZa 窗口拼接仅作回退。随后由 `scripts/extract_score_runtime_windows.py` 重建数据，再由 `scripts/extract_jianpu_jianzi.py --raw <file> --out-dir <dir>` 输出映射 JSON、Markdown 和 ABC。

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

- 已能采集并保存 `raw_data.json` / `data.json`，保留原始事件及来源。
- 已实现简谱—减字谱解析、ABC 输出、正调 `1=F` 和调弦偏移解释。
- 已处理撮、撞、十徽、徽外及复合符号闭合等已知解析问题。
- 已用完整 source event list 修复窗口边界缺音和全局去重误删；《不染》的“染”已恢复为 `q3 / 名指九徽勾5弦`。

## 待解决

- 建立可审计的数据纠错机制、生成基线和兼顾准确性、可演奏性、连贯性与韵味的评估体系。

## 工作原则

- 论文主线优先；数据采集和逆向是基础设施，不应无限扩张。
- `raw_data.json` 作为原始证据不得静默改写；修正规则单独记录、可重放。
- 未知编码保留原值，结论区分“已验证 / 推断 / 未验证”。
- 修改提取逻辑后，用 `batch/SYuY17FF/out/raw_data.json` 回归检查输出。
- 完整采集说明见 `AGENT_REPRODUCTION_GUIDE.md`；排查经验见 `docs/PARTIAL_OMISSION_POSTMORTEM.md`。
