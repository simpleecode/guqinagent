"""ABC-to-jianzipu agent framework.

The package keeps deterministic music logic separate from model-backed agents.
The public entry point is :class:`orchestrator.AbcToJianzipuOrchestrator`.
"""

__all__ = ["AbcToJianzipuOrchestrator"]


def __getattr__(name):
    # Deterministic parsers/renderers are also used in lightweight evaluation
    # environments where LangGraph is intentionally not installed.
    if name == "AbcToJianzipuOrchestrator":
        from .orchestrator import AbcToJianzipuOrchestrator
        return AbcToJianzipuOrchestrator
    raise AttributeError(name)
