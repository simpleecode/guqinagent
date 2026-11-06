"""GEPA callbacks: durable iteration identities and live HTML, without API calls."""
from __future__ import annotations

from typing import Any
import difflib
import json


class TeacherGepaAudit:
    def __init__(self, logger, adapter, run_dir, renderer):
        self.logger = logger
        self.adapter = adapter
        self.run_dir = run_dir
        self.renderer = renderer
        self.parent_text = ""
        self.proposal_text = ""

    def _save(self, name: str, event: dict[str, Any]) -> None:
        # Engine state contains runtime objects; only persist explicit fields.
        payload = {key: value for key, value in event.items()
                   if key not in {"state", "trainset_loader", "trajectories", "metadata"}}
        self.logger({"event": name, **payload})

    def on_iteration_start(self, event):
        self.adapter.audit_context = {"iteration": event["iteration"]}
        self._save("gepa_iteration_start", event)

    def on_minibatch_sampled(self, event):
        self._save("gepa_minibatch_sampled", event)

    def on_reflective_dataset_built(self, event):
        self.adapter.audit_context = {
            "iteration": event["iteration"], "iteration_id": event["iteration_id"],
        }
        self._save("gepa_reflective_dataset", event)

    def on_proposal_end(self, event):
        # Save rejected proposals too; candidates.json contains accepted ones only.
        manifest_path = self.run_dir / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        series = manifest.get("series")
        if series:
            event = {**event, "version": f"{series}{event['iteration']}"}
        self._save("gepa_proposal", event)
        self.proposal_text = event.get("new_instructions", {}).get("single_stage", "")
        if series and self.proposal_text:
            output = self.run_dir / "prompts"
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{event['version']}.txt").write_text(self.proposal_text + "\n", encoding="utf-8")

    def _refresh(self):
        manifest_path = self.run_dir / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if manifest.get("series") and (self.run_dir / "candidates.json").exists():
            from baseline.scripts.run_teacher_gepa_iteration import export_prompt_texts
            export_prompt_texts(self.run_dir)
        self.renderer(self.run_dir, self.run_dir / "optimization_report.html")

    def on_proposal_start(self, event):
        self.parent_text = event.get("parent_candidate", {}).get("single_stage", "")

    def _lesson(self, event, outcome):
        change = "\n".join(difflib.unified_diff(
            self.parent_text.splitlines(), self.proposal_text.splitlines(), lineterm=""
        ))[:1600]
        lesson = {"iteration": event.get("iteration"), "outcome": outcome,
                  "old_score": event.get("old_score"), "new_score": event.get("new_score"),
                  "reason": event.get("reason"), "prompt_change_excerpt": change,
                  "interpretation": "单次小批证据；保留可复用改动、避免重复失败，不将其当作因果定论或部署结论"}
        self.adapter.evolution_memory = getattr(self.adapter, "evolution_memory", []) + [lesson]
        self.logger({"event": "evolution_lesson", "lesson": lesson})

    def on_candidate_accepted(self, event):
        self._save("gepa_candidate_accepted", event)
        self._lesson(event, "accepted_for_pareto_evaluation")

    def on_candidate_rejected(self, event):
        self._save("gepa_candidate_rejected", event)
        self._lesson(event, "rejected_but_archived_for_learning")

    def on_state_saved(self, event):
        self._refresh()

    def on_iteration_end(self, event):
        self._save("gepa_iteration_end", event)
        self._refresh()
