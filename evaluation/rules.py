"""Final-state rule warnings replayed by the live edit_plan runtime."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.ToolRuntime.runtime import validate_jianzi_only

from .parser import final_actions


def replay_final_warnings(
    samples: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run final authored plans through edit_plan warning rules in score order."""
    ordered = sorted(samples, key=lambda pair: min(
        (int(note['index']) for note in pair[1]['runtime_item']['input']
         .get('notes_without_jianzi') or [] if note.get('index') is not None),
        default=-1,
    ))
    history: dict[tuple[str, str], dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for prediction, reference in ordered:
        item = deepcopy(reference['runtime_item'])
        score_key = str(item['score_key'])
        phrase_id = str(item['phrase_id'])
        actions = deepcopy(final_actions(prediction))
        if not actions:
            raise ValueError(f'no final plan for {score_key}-{phrase_id}')
        # Warning replay must depend only on public score input and the
        # model's authored plan, never on sealed annotation actions.
        item.pop('reference_plan', None)
        item['baseline_plan'] = {'actions': actions}
        _, report = validate_jianzi_only(
            item, [], require_complete=False, historical=history,
        )
        if report.get('pitch_audit', {}).get('unavailable'):
            raise RuntimeError(
                f'pitch warning replay unavailable for {score_key}-{phrase_id}: '
                f"{report['pitch_audit']['unavailable']}"
            )
        for warning in report['warnings']:
            result.append({
                'piece_id': score_key,
                'phrase_id': phrase_id,
                'event_id': warning.get('source_index'),
                'violation_type': warning['code'],
                'warning': deepcopy(warning),
                'definition': 'warning from final-plan edit_plan runtime replay',
            })
        item['reference_plan'] = deepcopy(item['baseline_plan'])
        history[(score_key, phrase_id)] = item
    return result
