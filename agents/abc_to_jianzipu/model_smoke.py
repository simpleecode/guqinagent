from __future__ import annotations

import json

from .backends import AnthropicCompatibleBackend
from .abc_parser import parse_abc
from .role_agents import RoleAgentFactory
from .skills import default_skill_registry


def main() -> int:
    backend = AnthropicCompatibleBackend(max_tokens=500)
    registry = default_skill_registry()
    score = parse_abc("X:1\nT:工具测试\nM:4/4\nL:1/4\nK:C\nC D\n")
    result = RoleAgentFactory(registry, backend).create("fingering_optimizer").run(
        "minimax-smoke",
        {
            "_required_tool": "get_pitch_candidates",
            "canonical_score": score.to_dict(),
            "performance_plan": {
                "actions": [{
                    "action_id": "a00001", "kind": "pluck",
                    "source_event_ids": ["n00001"], "mode": "stopped",
                    "string": 7, "hui": 7.0, "left_finger": "大指",
                    "right_finger": "挑",
                }],
            },
        },
        skill_versions=registry.versions(),
    )
    # Never print environment variables or request headers.
    print(json.dumps({
        "status": result.status,
        "role": result.agent_role,
        "output_keys": sorted(result.output),
        "tool_calls": [
            block["name"]
            for step in result.trace if step.get("type") == "assistant"
            for block in step.get("content", []) if block.get("type") == "tool_call"
        ],
        "diagnostics": result.diagnostics,
    }, ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
