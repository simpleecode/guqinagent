"""Anthropic-compatible GLM client used by both the task rollouts and the
GEPA reflection model.

The teacher pipeline talks to GLM through the Anthropic wire protocol
(``GLM_API_KEY`` / ``GLM_BASE_URL`` / ``GLM_MODEL`` in ``.env``); this module
reuses that endpoint.  Rollouts drive native ``tool_use`` blocks with the
public tool schemas (already Anthropic-shaped), and the reflection LM adapts
gepa's ``LanguageModel`` protocol (str | chat messages -> str) onto the same
client.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic  # noqa: E402

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class GlmAnthropicClient:
    """Rate-limited, retrying wrapper around the Anthropic-compatible GLM API."""

    def __init__(self, *, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 240.0,
                 min_interval: float = 0.0, max_retries: int = 4,
                 max_tokens: int | None = None):
        self.model = model or os.getenv("GLM_MODEL", "glm-5.3")
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        self.base_url = (base_url or os.getenv("GLM_BASE_URL") or "").rstrip("/")
        if not self.api_key or not self.base_url:
            raise RuntimeError(
                "GLM API credentials are not configured; set GLM_API_KEY and "
                "GLM_BASE_URL (see .env)"
            )
        self.min_interval = max(0.0, float(min_interval))
        self.max_retries = int(max_retries)
        default_max_tokens = int(os.getenv("GLM_MAX_TOKENS", "8000"))
        self.max_tokens = int(max_tokens or default_max_tokens)
        self.thinking_mode = os.getenv("GLM_THINKING", "disabled").lower()
        self._client = Anthropic(api_key=self.api_key, base_url=self.base_url,
                                 timeout=timeout, max_retries=1)
        self._last_call = 0.0
        # Usage accounting: actor = task rollouts, reflection = GEPA proposer.
        self.actor_calls = 0
        self.actor_tokens_in = 0
        self.actor_tokens_out = 0
        self.reflection_calls = 0
        self.reflection_tokens_in = 0
        self.reflection_tokens_out = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "actor_calls": self.actor_calls,
            "actor_tokens_in": self.actor_tokens_in,
            "actor_tokens_out": self.actor_tokens_out,
            "reflection_calls": self.reflection_calls,
            "reflection_tokens_in": self.reflection_tokens_in,
            "reflection_tokens_out": self.reflection_tokens_out,
        }

    @staticmethod
    def _absorb(response: Any, target: str) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        return tokens_in, tokens_out

    def messages_create(self, **options: Any):
        """Call ``messages.create`` with spacing and bounded retries."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.min_interval:
                wait = self.min_interval - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                return self._client.messages.create(**options)
            except Exception as exc:  # anthropic.APIStatusError/APIConnectionError/…
                status = getattr(exc, "status_code", None)
                retryable = status is None or int(status) in RETRYABLE_STATUS
                last_error = exc
                if not retryable or attempt == self.max_retries:
                    raise
                time.sleep(min(60.0, 2.0 ** attempt) * (1.0 + 0.25 * attempt))
        raise last_error  # pragma: no cover - unreachable


class GlmBackend:
    """Task-LM backend: one Anthropic-native tool-call turn per generate call."""

    def __init__(self, client: GlmAnthropicClient):
        self.client = client

    def generate(self, *, system: str, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]], temperature: float = 0.0,
                 max_tokens: int | None = None):
        options: dict[str, Any] = {
            "model": self.client.model,
            "max_tokens": int(max_tokens or self.client.max_tokens),
            "temperature": temperature,
            "system": system,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "auto"},
        }
        if self.client.thinking_mode == "adaptive":
            options["thinking"] = {"type": "adaptive"}
        else:
            options["thinking"] = {"type": "disabled"}
        response = self.client.messages_create(**options)
        self.client.actor_calls += 1
        tokens_in, tokens_out = GlmAnthropicClient._absorb(response, "actor")
        self.client.actor_tokens_in += tokens_in
        self.client.actor_tokens_out += tokens_out
        return response


def visible_text(response: Any) -> str:
    """Concatenate the response's text blocks (reasoning stays invisible)."""
    parts = [block.text for block in (getattr(response, "content", None) or [])
             if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()


def truncated(response: Any) -> bool:
    return getattr(response, "stop_reason", None) == "max_tokens"


def tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract native tool_use blocks as ``{"name", "arguments", "id"}``."""
    calls = []
    for block in (getattr(response, "content", None) or []):
        if getattr(block, "type", None) == "tool_use":
            calls.append({"name": str(block.name), "id": str(block.id),
                          "arguments": dict(block.input or {})})
    return calls


def history_blocks(response: Any) -> list[dict[str, Any]]:
    """Assistant blocks to append to the conversation (text + tool_use)."""
    return [{"type": block.type, **({"text": block.text} if block.type == "text"
                                    else {"id": block.id, "name": block.name,
                                          "input": dict(block.input or {})})}
            for block in (getattr(response, "content", None) or [])
            if getattr(block, "type", None) in {"text", "tool_use"}]


class GlmReflectionLM:
    """Adapter from gepa's ``LanguageModel`` protocol onto the GLM endpoint.

    gepa calls ``prompt: str | list[dict]`` where the list form carries
    OpenAI-style ``{"role", "content"}`` chat messages.  System messages are
    folded into the Anthropic ``system`` parameter; the rest pass through
    unchanged.
    """

    def __init__(self, client: GlmAnthropicClient, *, temperature: float = 0.3):
        self.client = client
        self.temperature = temperature
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        system = ""
        messages: list[dict[str, Any]] = []
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            for row in prompt:
                role = str(row.get("role") or "user")
                content = row.get("content")
                content = content if isinstance(content, str) else json.dumps(
                    content, ensure_ascii=False)
                if role == "system":
                    system = (system + "\n\n" + content).strip() if system else content
                else:
                    messages.append({"role": role, "content": content})
        response = self.client.messages_create(
            model=self.client.model,
            max_tokens=self.client.max_tokens,
            temperature=self.temperature,
            system=system or None,
            messages=messages or [{"role": "user", "content": ""}],
            thinking={"type": "disabled"},
        )
        self.client.reflection_calls += 1
        self.client.reflection_tokens_in += int(
            getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)
        self.client.reflection_tokens_out += int(
            getattr(getattr(response, "usage", None), "output_tokens", 0) or 0)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
            self.total_tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        return visible_text(response)
