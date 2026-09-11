from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .models import CanonicalScore, PerformanceAction, PerformancePlan, PositionCandidate, Route
from .pitch_candidates import MAPPER


@dataclass(slots=True)
class PlannerWeights:
    cents: float = 0.03
    tone_region: float = 1.0
    string_change: float = 0.75
    string_skip: float = 0.30
    horizontal_move: float = 8.0
    fast_move: float = 4.0
    direction_reversal: float = 0.35
    phrase_boundary: float = 1.0


@dataclass(slots=True)
class _State:
    cost: float
    node_cost: float
    transition_cost: float
    path: list[PositionCandidate]
    previous_direction: int = 0


@dataclass(slots=True)
class GlobalRouteMerge:
    selected_routes: list[Route]
    local_cost: float
    boundary_cost: float
    total_cost: float
    boundaries: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class _MergeState:
    total_cost: float
    local_cost: float
    boundary_cost: float
    path: list[Route]
    last_exit_candidate_id: str | None
    last_exit_phrase_id: str | None
    boundaries: list[dict]


def _node_cost(candidate: PositionCandidate, weights: PlannerWeights) -> float:
    return abs(candidate.cents_error) * weights.cents + candidate.tone_region_cost * weights.tone_region


def _coordinate(candidate: PositionCandidate) -> float:
    if candidate.hui is None:
        return 0.0
    return float(MAPPER.hui_coordinate(candidate.hui))


def transition_cost(previous: PositionCandidate, current: PositionCandidate,
                    available_beats: float, weights: PlannerWeights,
                    previous_direction: int = 0) -> tuple[float, int]:
    string_delta = abs(current.string - previous.string)
    horizontal = abs(_coordinate(current) - _coordinate(previous))
    cost = string_delta * weights.string_change
    if string_delta > 1:
        cost += (string_delta - 1) * weights.string_skip
    cost += horizontal * weights.horizontal_move
    if available_beats < 0.5:
        cost += horizontal * weights.fast_move / max(available_beats, 0.125)
    direction = 0
    if current.hui is not None and previous.hui is not None:
        direction = (current.hui > previous.hui) - (current.hui < previous.hui)
    if direction and previous_direction and direction != previous_direction:
        cost += weights.direction_reversal
    return cost, direction or previous_direction


def top_k_routes(score: CanonicalScore, phrase_id: str,
                 candidates: dict[str, list[PositionCandidate]], *,
                 top_k: int = 5,
                 states_per_candidate: int = 3,
                 weights: PlannerWeights | None = None) -> list[Route]:
    weights = weights or PlannerWeights()
    event_by_id = {event.id: event for event in score.events}
    phrase = next(item for item in score.phrases if item.id == phrase_id)
    pitched_events = [
        event_by_id[event_id] for event_id in phrase.event_ids
        if event_by_id[event_id].midi is not None and event_by_id[event_id].attack
    ]
    if not pitched_events:
        return [Route(phrase_id, 1, [], [], 0.0, 0.0, 0.0, None, None)]
    missing = [event.id for event in pitched_events if not candidates.get(event.id)]
    if missing:
        return []

    layers: dict[str, list[_State]] = {}
    first = pitched_events[0]
    for candidate in candidates[first.id]:
        node = _node_cost(candidate, weights)
        layers[candidate.candidate_id] = [_State(node, node, 0.0, [candidate])]

    for event_index, event in enumerate(pitched_events[1:], 1):
        previous_event = pitched_events[event_index - 1]
        available_beats = max(previous_event.duration_ticks / 960.0, 0.125)
        next_layers: dict[str, list[_State]] = {}
        for current in candidates[event.id]:
            proposals: list[_State] = []
            node = _node_cost(current, weights)
            for states in layers.values():
                for state in states:
                    trans, direction = transition_cost(
                        state.path[-1], current, available_beats,
                        weights, state.previous_direction,
                    )
                    proposals.append(_State(
                        cost=state.cost + node + trans,
                        node_cost=state.node_cost + node,
                        transition_cost=state.transition_cost + trans,
                        path=state.path + [current],
                        previous_direction=direction,
                    ))
            proposals.sort(key=lambda state: state.cost)
            next_layers[current.candidate_id] = proposals[:states_per_candidate]
        layers = next_layers

    final_states = [state for states in layers.values() for state in states]
    final_states.sort(key=lambda state: state.cost)
    routes: list[Route] = []
    seen_paths: set[tuple[str, ...]] = set()
    for state in final_states:
        candidate_ids = tuple(candidate.candidate_id for candidate in state.path)
        if candidate_ids in seen_paths:
            continue
        seen_paths.add(candidate_ids)
        routes.append(Route(
            phrase_id=phrase_id,
            rank=len(routes) + 1,
            candidate_ids=list(candidate_ids),
            event_ids=[candidate.event_id for candidate in state.path],
            total_cost=round(state.cost, 6),
            node_cost=round(state.node_cost, 6),
            transition_cost=round(state.transition_cost, 6),
            entry_candidate_id=candidate_ids[0] if candidate_ids else None,
            exit_candidate_id=candidate_ids[-1] if candidate_ids else None,
        ))
        if len(routes) >= top_k:
            break
    return routes


def _available_boundary_beats(score: CanonicalScore, previous_event_id: str,
                              current_event_id: str) -> float:
    event_index = {event.id: index for index, event in enumerate(score.events)}
    start = event_index[previous_event_id]
    end = event_index[current_event_id]
    if end <= start:
        return 0.125
    return max(
        sum(event.duration_ticks for event in score.events[start:end]) / 960.0,
        0.125,
    )


def merge_phrase_routes_report(
    score: CanonicalScore, routes_by_phrase: dict[str, list[Route]],
    candidate_index: dict[str, PositionCandidate], *,
    weights: PlannerWeights | None = None,
) -> GlobalRouteMerge:
    """Select the exact minimum-cost route combination across phrase boundaries."""
    weights = weights or PlannerWeights()
    phrase_order = [phrase.id for phrase in score.phrases]
    if not phrase_order:
        return GlobalRouteMerge([], 0.0, 0.0, 0.0)
    missing = [phrase_id for phrase_id in phrase_order if not routes_by_phrase.get(phrase_id)]
    if missing:
        raise ValueError(f"missing_phrase_routes:{','.join(missing)}")

    states = [
        _MergeState(
            route.total_cost, route.total_cost, 0.0, [route],
            route.exit_candidate_id,
            route.phrase_id if route.exit_candidate_id else None,
            [],
        )
        for route in routes_by_phrase[phrase_order[0]]
    ]
    for phrase_id in phrase_order[1:]:
        best_by_state: dict[tuple[int, str | None], _MergeState] = {}
        for state in states:
            for route_index, route in enumerate(routes_by_phrase[phrase_id]):
                boundary = 0.0
                boundary_record = None
                if state.last_exit_candidate_id and route.entry_candidate_id:
                    previous = candidate_index[state.last_exit_candidate_id]
                    current = candidate_index[route.entry_candidate_id]
                    available_beats = _available_boundary_beats(
                        score, previous.event_id, current.event_id,
                    )
                    boundary, _ = transition_cost(
                        previous, current, available_beats, weights,
                    )
                    boundary *= weights.phrase_boundary
                    boundary_record = {
                        "from_phrase": state.last_exit_phrase_id,
                        "to_phrase": phrase_id,
                        "from_candidate_id": state.last_exit_candidate_id,
                        "to_candidate_id": route.entry_candidate_id,
                        "available_beats": round(available_beats, 6),
                        "cost": round(boundary, 6),
                    }
                last_exit = route.exit_candidate_id or state.last_exit_candidate_id
                last_exit_phrase = (
                    route.phrase_id if route.exit_candidate_id else state.last_exit_phrase_id
                )
                proposal = _MergeState(
                    total_cost=state.total_cost + route.total_cost + boundary,
                    local_cost=state.local_cost + route.total_cost,
                    boundary_cost=state.boundary_cost + boundary,
                    path=state.path + [route],
                    last_exit_candidate_id=last_exit,
                    last_exit_phrase_id=last_exit_phrase,
                    boundaries=state.boundaries + ([boundary_record] if boundary_record else []),
                )
                key = (route_index, last_exit)
                incumbent = best_by_state.get(key)
                proposal_key = (
                    proposal.total_cost,
                    tuple(item.rank for item in proposal.path),
                    tuple(tuple(item.candidate_ids) for item in proposal.path),
                )
                incumbent_key = None if incumbent is None else (
                    incumbent.total_cost,
                    tuple(item.rank for item in incumbent.path),
                    tuple(tuple(item.candidate_ids) for item in incumbent.path),
                )
                if incumbent_key is None or proposal_key < incumbent_key:
                    best_by_state[key] = proposal
        states = list(best_by_state.values())

    best = min(
        states,
        key=lambda state: (
            state.total_cost,
            tuple(item.rank for item in state.path),
            tuple(tuple(item.candidate_ids) for item in state.path),
        ),
    )
    return GlobalRouteMerge(
        selected_routes=best.path,
        local_cost=round(best.local_cost, 6),
        boundary_cost=round(best.boundary_cost, 6),
        total_cost=round(best.total_cost, 6),
        boundaries=best.boundaries,
    )


def merge_phrase_routes(score: CanonicalScore, routes_by_phrase: dict[str, list[Route]],
                        candidate_index: dict[str, PositionCandidate], *,
                        weights: PlannerWeights | None = None) -> list[Route]:
    """Backward-compatible selected-route view of the global merge report."""
    return merge_phrase_routes_report(
        score, routes_by_phrase, candidate_index, weights=weights,
    ).selected_routes


def routes_to_plan(selected_routes: list[Route], candidate_index: dict[str, PositionCandidate]) -> PerformancePlan:
    actions: list[PerformanceAction] = []
    for route in selected_routes:
        for candidate_id in route.candidate_ids:
            candidate = candidate_index[candidate_id]
            actions.append(PerformanceAction(
                action_id=f"a{len(actions) + 1:05d}",
                kind="pluck",
                source_event_ids=[candidate.event_id],
                mode=candidate.mode,
                string=candidate.string,
                hui=candidate.hui,
                left_finger=None,
                right_finger=None,
                attack=True,
                confidence=candidate.confidence,
                rule_evidence=[
                    "BASELINE_POSITION_ROUTE", "PITCH_MAPPER_CANDIDATE",
                    "FINGERING_PENDING",
                ],
            ))
    return PerformancePlan(
        actions=actions,
        selected_routes=selected_routes,
        total_cost=round(sum(route.total_cost for route in selected_routes), 6),
        diagnostics=[
            "baseline_only: right-hand fingering and guqinization require a later agent pass"
        ],
    )
