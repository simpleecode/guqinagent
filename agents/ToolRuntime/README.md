# 工具运行时与告警

- `runtime.py`：四工具调度、检索与上下文、候选展示、patch 协议处理。保留原有告警函数的导出，兼容现有调用方。
- `warning_checks.py`：方案校验、再作与置空规则、历史音高审计上下文及告警汇总。
- `warning_rendering.py`：告警中文文案、edit_plan 预览和已有方案的公开告警展示。
- `warning_registry.py`：公开谱面告警的 code 集合及共享常量；不包含协议错误与私有参考差异。

音高和左手状态仍统一由 `scripts/audit_jianpu_jianzi_pitch.py` 维护，告警模块复用它的解析结果，不另建一套继承规则。

新增类别时同步维护注册表、检测和公开文案。教师生成、推理、可视化与离线重审继续使用同一校验入口 `validate_jianzi_only`。此次拆分没有调整规则、阈值或回执内容。
