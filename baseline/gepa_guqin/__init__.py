"""GEPA adaptation for the Guqin ABC->jianzipu single-stage agent.

Wires the vendored GEPA framework (``baseline/gepa``) to the production
public agent protocol: per-phrase rollouts with the deterministic
``RealToolRuntime`` tool surface, scored by the sealed structured-event
metrics.  See ``README.md`` in this directory for the run book.
"""
from .adapter import GuqinAgentAdapter
from .metric import DEFAULT_WEIGHTS, phrase_report
from .rollout import GlmBackend, run_single_stage_phrase

__all__ = [
    "GuqinAgentAdapter",
    "GlmBackend",
    "run_single_stage_phrase",
    "phrase_report",
    "DEFAULT_WEIGHTS",
]
