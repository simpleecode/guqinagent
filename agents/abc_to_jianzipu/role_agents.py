from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from .backends import AgentBackend, NoModelBackend
from .models import TaskEnvelope, TaskResult
from .prompts import PROMPTS
from .skills import SkillRegistry
from .tool_broker import ToolBroker


ROLE_SCHEMAS = {
    "tuning_planner": "TuningDecision.v1",
    "score_analyst": "ScoreAnalysis.v1",
    "phrase_planner": "PhrasePlanReview.v1",
    "guqinization": "PerformancePlanPatch.v1",
    "fingering_agent": "FingeringOptimization.v1",
    "fingering_optimizer": "FingeringOptimization.v1",
    "style_critic": "StyleReview.v1",
}


@dataclass(slots=True)
class RoleAgent:
    role: str
    backend: AgentBackend
    prompt: str
    expected_schema: str
    tool_broker: ToolBroker

    def run(self, job_id: str, payload: dict[str, Any], *,
            skill_versions: dict[str, str], attempt: int = 1) -> TaskResult:
        task = TaskEnvelope(
            job_id=job_id,
            task_id=f"{self.role}-{uuid.uuid4().hex[:10]}",
            agent_role=self.role,
            payload=payload,
            expected_schema=self.expected_schema,
            skill_versions=skill_versions,
            attempt=attempt,
        )
        result = self.backend.run(task, self.prompt, self.tool_broker)
        if result.task_id != task.task_id or result.agent_role != self.role:
            raise ValueError("agent backend returned a result for the wrong task or role")
        trace = self.tool_broker.trace_for(task.task_id)
        if trace:
            result.trace.append({"type": "tool_broker_trace", "content": trace})
        return result


class RoleAgentFactory:
    """Creates narrowly scoped role agents with versioned prompt contracts."""

    def __init__(self, registry: SkillRegistry,
                 backend: AgentBackend | None = None) -> None:
        self.backend = backend or NoModelBackend()
        self.tool_broker = ToolBroker(registry)

    def create(self, role: str) -> RoleAgent:
        if role not in PROMPTS:
            raise KeyError(f"unknown agent role: {role}")
        return RoleAgent(
            role, self.backend, PROMPTS[role], ROLE_SCHEMAS[role], self.tool_broker
        )
