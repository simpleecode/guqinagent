from __future__ import annotations

import json
import os
import re
from typing import Protocol

from dotenv import load_dotenv

from .models import TaskEnvelope, TaskResult


class AgentBackend(Protocol):
    """Provider-neutral interface for an optional model-backed graph node."""

    def run(self, task: TaskEnvelope, prompt: str, tools=None) -> TaskResult:
        ...


class NoModelBackend:
    """Explicit placeholder used by the deterministic MVP."""

    def run(self, task: TaskEnvelope, prompt: str, tools=None) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            agent_role=task.agent_role,
            status="not_configured",
            output={},
            diagnostics=[{
                "code": "model_backend_not_configured",
                "message": "This node is optional in the deterministic baseline.",
            }],
        )


class AnthropicCompatibleBackend:
    """Model backend for Anthropic-compatible APIs, including MiniMax."""

    def __init__(self, *, model: str | None = None, max_tokens: int = 1400,
                 timeout: float = 90.0, max_tool_rounds: int | None = None) -> None:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        if not base_url:
            raise RuntimeError("ANTHROPIC_BASE_URL is not configured")
        from anthropic import Anthropic

        self.model = model or os.getenv("MINIMAX_MODEL", "MiniMax-M3")
        self.max_tokens = max_tokens
        configured_rounds = os.getenv("AGENT_MAX_TOOL_ROUNDS", "24")
        self.max_tool_rounds = (
            int(configured_rounds) if max_tool_rounds is None else max_tool_rounds
        )
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=1,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        cleaned = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
        if fenced:
            cleaned = fenced.group(1)
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                cleaned = cleaned[start:end + 1]
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise ValueError("model response must be a JSON object")
        return value

    def run(self, task: TaskEnvelope, prompt: str, tools=None) -> TaskResult:
        request_text = (
            "以下是 TaskEnvelope：\n"
            + json.dumps(task.to_dict(), ensure_ascii=False)
            + "\n只返回一个 JSON 对象，不要使用 Markdown。格式必须是："
              '{"status":"completed","output":{},"diagnostics":[]}'
        )
        required_tool = task.payload.get("_required_tool")
        if required_tool:
            request_text += f"\n在返回最终 JSON 前，必须先调用工具 {required_tool}。"
        try:
            trace: list[dict] = [
                {"type": "system", "content": prompt},
                {"type": "task", "content": task.to_dict()},
            ]
            messages = [{"role": "user", "content": request_text}]
            tool_definitions = tools.definitions_for(task.agent_role) if tools else []
            request_options = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 1.0,
                "system": prompt,
            }
            if tool_definitions:
                request_options["tools"] = tool_definitions
                available_names = {item["name"] for item in tool_definitions}
                if required_tool in available_names:
                    # MiniMax's compatibility layer reliably accepts the
                    # Anthropic `any` strategy; the prompt names the tool.
                    request_options["tool_choice"] = {"type": "any"}
            if self.model == "MiniMax-M3":
                request_options["thinking"] = {"type": "disabled"}
            text = ""
            for _ in range(self.max_tool_rounds + 1):
                response = self.client.messages.create(messages=messages, **request_options)
                observable_blocks = []
                for block in response.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        observable_blocks.append({"type": "text", "text": block.text})
                    elif block_type == "tool_use":
                        observable_blocks.append({
                            "type": "tool_call", "name": block.name,
                            "arguments": dict(block.input), "tool_use_id": block.id,
                        })
                trace.append({"type": "assistant", "content": observable_blocks})
                current_text = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                tool_uses = [
                    block for block in response.content
                    if getattr(block, "type", None) == "tool_use"
                ]
                if not tool_uses:
                    text = current_text
                    break
                if tools is None:
                    raise RuntimeError("model requested a tool but no tool broker is configured")
                messages.append({
                    "role": "assistant",
                    "content": [block.model_dump(exclude_none=True) for block in response.content],
                })
                tool_results = []
                for block in tool_uses:
                    result = tools.invoke(task, block.name, dict(block.input))
                    trace.append({
                        "type": "tool_result", "name": block.name,
                        "tool_use_id": block.id, "content": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                        "is_error": not result.get("ok", False),
                    })
                messages.append({"role": "user", "content": tool_results})
                # A required tool applies to the first round only; subsequent
                # rounds may call more tools or return the final JSON.
                request_options.pop("tool_choice", None)
            else:
                raise RuntimeError("model exceeded the maximum tool-call rounds")
            parsed = self._parse_json(text)
            output = parsed.get("output", parsed)
            diagnostics = parsed.get("diagnostics", [])
            if not isinstance(output, dict) or not isinstance(diagnostics, list):
                raise ValueError("model output or diagnostics has the wrong type")
            return TaskResult(
                task_id=task.task_id,
                agent_role=task.agent_role,
                status=str(parsed.get("status", "completed")),
                output=output,
                diagnostics=diagnostics,
                trace=trace,
            )
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                agent_role=task.agent_role,
                status="failed",
                output={},
                diagnostics=[{
                    "code": "model_backend_error",
                    "message": f"{type(exc).__name__}: {exc}",
                }],
                trace=locals().get("trace", []),
            )
