"""Shared warning checks."""
from __future__ import annotations

import re
from copy import deepcopy

from agents.abc_to_jianzipu.inverse_baseline import AUDIT
from agents.abc_to_jianzipu.jianzi_renderer import hui_label
from agents.abc_to_jianzipu.pitch_candidates import abc_chord_midis, abc_chord_tones
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER
from agents.abc_to_jianzipu.score_structure import is_structural_note
from agents.abc_to_jianzipu.teacher_trajectory import chord_jianpu_label
from agents.abc_to_jianzipu.trajectory_replay import replay_patches

from agents.ToolRuntime.warning_registry import NUMBERED_REPEAT_PATTERN

def target_midis(item: dict, source_index: int) -> list[float]:
    """Every sounding target of a note: all ABC chord tones, else jianpu.

    Chord tokens are authoritative for 撮-style double stops; the jianpu field
    only carries one of the tones, so it is used solely as fallback.
    """
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    tonic = AUDIT.parse_tonic_midi(item["input"]["metadata"])
    # 和弦（撮等）的音高以存储双谱字为准：ABC 转换在降号调上存在半音偏差，
    # 记谱本身才是真相；无双谱字时回退到 ABC 和弦。
    if abc_chord_tones(str(note.get("abc") or "")):
        values = [AUDIT.parse_jianpu(note.get(field), tonic)
                  for field in ("jianpu", "jianpu_alt")]
        values = [float(value) for value in values if value is not None]
        if len(values) == 2:
            return values
        midis = abc_chord_midis(str(note.get("abc") or ""))
        if midis:
            return midis
    values = [AUDIT.parse_jianpu(note.get(field), tonic) for field in ("jianpu", "jianpu_alt")]
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError(f"source_index {source_index} has no sounding target")
    return [float(value) for value in values]

def pitch_label(item: dict, source_index: int) -> str:
    """Jianpu label of a note; chord notes show both stored digits."""
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    written = note.get("jianpu") or note.get("jianpu_alt") or "休止"
    return chord_jianpu_label(note.get("abc"), written, note.get("jianpu_alt"))

def event_index_for_source(item: dict, source_index: int) -> int | None:
    """Translate an internal sparse source index to the visible continuous 音序."""
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        if int(note.get("index")) == int(source_index):
            value = note.get("event_index")
            return int(value) if value is not None else None
    return None

def source_index_for_event(item: dict, event_index: int) -> int | None:
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        value = note.get("event_index")
        if value is not None and int(value) == int(event_index):
            return int(note["index"])
    return None

def harmonic_scope_after_phrases(
    phrases: list[tuple[list[dict], list[dict]]],
) -> bool:
    """Replay harmonic scope with the same bare-rest boundary as pitch audit."""
    active = False
    for notes, actions in phrases:
        by_index = {
            int(action["source_index"]): action
            for action in actions if action.get("source_index") is not None
        }
        # Older archived trajectory rows occasionally lack phrase notes.  They
        # cannot contribute a rest boundary, but their explicit markers still
        # remain meaningful state evidence.
        if not notes:
            for action in sorted(actions, key=lambda row: int(row.get("source_index", -1))):
                text = str(action.get("text") or action.get("jianzi_text") or "")
                for marker in re.findall(r"泛起|泛止", text):
                    active = marker == "泛起"
            continue
        for note in notes:
            source_index = note.get("index")
            action = by_index.get(int(source_index)) if source_index is not None else None
            text = str((action or {}).get("text") or (action or {}).get("jianzi_text") or "")
            active = AUDIT.replay_harmonic_scope(active, note, text)
    return active

def harmonic_region_at_phrase_start(
    item: dict, historical: dict[tuple[str, str], dict]
) -> bool:
    """Replay confirmed harmonic state before the current phrase."""
    def phrase_start(row: dict) -> int:
        """Accept both runtime and archived trajectory event-range schemas."""
        event_range = row["input"]["event_range"]
        return int(event_range.get("start", event_range.get("start_index")))

    current_start = phrase_start(item)
    earlier = sorted(
        (row for (score_key, _), row in historical.items()
         if score_key == item["score_key"]
         and phrase_start(row) < current_start),
        key=phrase_start,
    )
    phrases = [
        (
            list(row["input"].get("notes_without_jianzi")
                 or row["input"].get("phrase_notes") or []),
            list(row.get("reference_plan", {}).get("actions") or []),
        )
        for row in earlier
    ]
    return harmonic_scope_after_phrases(phrases)

def pitch_audit_notes(
    item: dict, actions_by_source: dict[int, dict],
    historical: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Build audit rows by replaying every confirmed earlier phrase."""
    handoff = (item.get("input") or {}).get("phrase_handoff") or {}
    previous = handoff.get("previous_phrase") or {}
    current_notes = list((item.get("input") or {}).get("notes_without_jianzi") or [])
    prefix_phrases: list[tuple[list[dict], list[dict]]] = []
    if historical:
        def phrase_start(row: dict) -> int:
            event_range = row["input"]["event_range"]
            return int(event_range.get("start", event_range.get("start_index")))

        current_start = phrase_start(item)
        earlier = sorted(
            (row for (score_key, _), row in historical.items()
             if score_key == item["score_key"]
             and phrase_start(row) < current_start),
            key=phrase_start,
        )
        prefix_phrases = [
            (list(row["input"].get("notes_without_jianzi") or row["input"].get("phrase_notes") or []),
             list(row.get("reference_plan", {}).get("actions") or []))
            for row in earlier
        ]
    if not prefix_phrases and previous:
        prefix_phrases = [(
            list(previous.get("notes") or []),
            list(previous.get("actions") or []),
        )]

    def render_rows(notes: list[dict], action_map: dict[int, dict]) -> list[dict]:
        rows = []
        for note in notes:
            row = deepcopy(note)
            action = action_map.get(int(note["index"]))
            if action is None:
                row["jianzi"] = ""
            else:
                value = action.get("jianzi_text")
                if value is None:
                    value = action.get("text") or ""
                row["jianzi"] = str(value)
            rows.append(row)
        return rows

    prefix = []
    # A phrase may begin in the middle of a harmonic passage.  The previous
    # phrase's structured action state then tells us the harmonic hui even if
    # its readable text omits ``泛起`` and the hui as an inherited field.
    all_prefix_actions = [action for _, actions in prefix_phrases for action in actions]
    seed_allowed = item.get("harmonic_region_at_start")
    if seed_allowed is None:
        seed_allowed = harmonic_scope_after_phrases(prefix_phrases)
    seed_hui = next(
        (action.get("hui") for action in reversed(all_prefix_actions)
         if seed_allowed and action.get("mode") == "harmonic"
         and action.get("hui") is not None),
        None,
    )
    if seed_hui is not None:
        prefix.append({
            "index": -1,
            "jianpu": None,
            "abc": "",
            "duration": "",
            # The pitch auditor parses numeric string designators.  Keep this
            # synthetic state seed in the same surface form as normal tool
            # input, otherwise “一弦” is treated as having no string and the
            # harmonic mode is never activated.
            "jianzi": f"泛起勾1弦{hui_label(seed_hui)}",
        })
    for notes, actions in prefix_phrases:
        action_map = {
            int(action["source_index"]): action
            for action in actions if action.get("source_index") is not None
        }
        prefix.extend(render_rows(notes, action_map))
    # ``harmonic_region_at_start`` is the canonical phrase-boundary replay
    # used for the model-facing prompt.  Historical reference actions can be
    # more permissive than that replay (for example, a stale ``泛起`` can
    # survive through an intervening empty/rest phrase).  Do not let such a
    # reference-only state manufacture harmonic-resonance warnings in a
    # phrase that is explicitly declared to start outside the region.
    # ``泛止`` clears only harmonic state; the preceding stopped-position
    # context remains available for ordinary shorthand inheritance.
    if prefix_phrases and seed_allowed is False:
        prefix.append({
            "index": -2,
            "jianpu": None,
            "abc": "",
            "duration": "",
            "jianzi": "泛止",
        })
    return prefix + render_rows(current_notes, actions_by_source)

def advance_pitch_warning_state(
    pending: set[int],
    repair_attempted: set[int],
    current_warnings: set[int],
    candidate_informed_edits: set[int],
) -> set[int]:
    """Record one edit-plan result and return warnings still requiring a turn.

    A warning produced by this very edit cannot also count as repaired by the
    same edit, even if its candidates were queried earlier.  The teacher must
    receive that warning and take at least one later turn before choosing
    whether to repair it or retain the notation.  A later candidate-informed
    edit may resolve the warning or leave it as an advisory result.
    """
    previously_pending = set(pending)
    pending.update(current_warnings)
    later_repairs = candidate_informed_edits & previously_pending
    repair_attempted.update(later_repairs)
    # If the later repair made the warning disappear, it no longer needs to
    # remain pending.  If it remains visible, the later attempted repair is
    # sufficient to let the model end on this advisory warning.
    pending.difference_update(later_repairs - current_warnings)
    return pending - repair_attempted

def normalize_patches(patches: list[dict], *, prefix: str = "teacher") -> list[dict]:
    normalized = []
    for number, patch in enumerate(patches, 1):
        item = deepcopy(patch)
        item.setdefault("patch_id", f"{prefix}-{number:04d}")
        item.setdefault("before", {})
        item.setdefault("after", {})
        normalized.append(item)
    return normalized

def canonical_jianzi_text(text: str | None) -> str | None:
    """Normalize harmless surface variants before private reference review."""
    if text is None:
        return None
    digits = str.maketrans("一二三四五六七", "1234567")
    return re.sub(r"\s+", "", str(text)).replace("弦", "弦").translate(digits)

def repeat_anchor_warnings(
    item: dict, actions_by_source: dict[int, dict], *,
    historical: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Check repeat anchors and the instruction preceding each omission run."""
    event_range = (item.get("input") or {}).get("event_range") or {}
    current_start = int(event_range.get("start", event_range.get("start_index", 0)))

    def phrase_start(row: dict) -> int:
        source_range = (row.get("input") or {}).get("event_range") or {}
        return int(source_range.get("start", source_range.get("start_index", 0)))

    item_score_key = item.get("score_key")
    earlier = sorted(
        (
            row for (history_score_key, _), row in (historical or {}).items()
            if history_score_key == item_score_key and phrase_start(row) < current_start
        ),
        key=phrase_start,
    ) if item_score_key else []
    phrase_rows: list[tuple[list[dict], dict[int, dict], bool]] = []
    for row in earlier:
        actions = {
            int(action["source_index"]): action
            for action in row.get("reference_plan", {}).get("actions", [])
            if action.get("source_index") is not None
        }
        phrase_rows.append((
            list(row["input"].get("notes_without_jianzi")
                 or row["input"].get("phrase_notes") or []),
            actions,
            False,
        ))
    phrase_rows.append((
        list(item["input"].get("notes_without_jianzi") or []),
        actions_by_source,
        True,
    ))

    anchor_seen = False
    previous_effective_text = ""
    omission_run = False
    warnings: list[dict] = []
    pending_anchor = None
    for notes, action_map, is_current in phrase_rows:
        for note in notes:
            source_index = note.get("index")
            if source_index is None:
                continue
            action = action_map.get(int(source_index))
            text = str((action or {}).get("jianzi_text")
                       or (action or {}).get("text") or "")
            # A standalone label is not an instruction that covers later
            # sounds. Require two blank sounding events; bars do not count.
            structural = is_structural_note(note)
            abc = str(note.get('abc') or '').strip()
            label = str(note.get('jianpu') or '')
            sounding = (not structural and not abc.startswith(('z','Z','x','X','-','{'))
                        and not any(t in label for t in ('休止','延音')) and label != '0')
            if pending_anchor is not None and not structural:
                blank = not text.strip() or text.strip() == OMITTED_PLACEHOLDER
                if blank and sounding:
                    pending_anchor['blanks'].append(int(source_index))
                    if (is_current and len(pending_anchor['blanks']) >= 2
                            and not pending_anchor['warned']):
                        warnings.append({'code':'repeat_anchor_followed_by_blanks',
                            'source_index':pending_anchor['source'] if pending_anchor['current'] else int(source_index),
                            'anchor_source_index':pending_anchor['source'],
                            'blank_source_indices':pending_anchor['blanks']})
                        pending_anchor['warned'] = True
                elif text.strip() or not sounding:
                    pending_anchor = None
            has_anchor = bool(re.search(r"再[作做](?:起点|标记)", text))
            anchor_remainder = re.sub(r'再[作做](?:起点|标记)\s*ㄱ?', '', text).strip(' []【】〔〕（）()')
            if has_anchor and not anchor_remainder:
                pending_anchor = {'source':int(source_index), 'current':is_current,
                                  'blanks':[], 'warned':False}
            instruction_text = re.sub(r"再[作做](?:起点|标记)", "", text)
            if is_current and has_anchor and re.search(r"再[作做]", instruction_text):
                warnings.append({
                    "source_index": int(source_index),
                    "code": "repeat_anchor_instruction_overlap",
                })
            if (is_current and "掐撮三声" in text
                    and re.search(r"再[作做]", instruction_text)):
                warnings.append({
                    "source_index": int(source_index),
                    "code": "compound_repeat_instruction_overlap",
                })
            # Test before setting the anchor so malformed text containing both
            # terms cannot satisfy its own prerequisite.
            if is_current and "从ㄱ再作" in text and not anchor_seen:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "missing_repeat_anchor",
                })
            if any(marker in text for marker in (
                "再作起点", "再作标记", "再做起点", "再做标记",
            )):
                anchor_seen = True
            if is_structural_note(note) or not text.strip():
                continue
            if text.strip() == OMITTED_PLACEHOLDER:
                # A start marker is not an instruction to repeat now.
                # 再作／再做 also covers 从ㄱ再作 and 从头再作／从头再做;
                # repeating from the beginning requires no authored anchor.
                instruction_text = re.sub(
                    r"再[作做](?:起点|标记)", "", previous_effective_text,
                )
                if (is_current and not omission_run
                        and not re.search(r"再[作做]", instruction_text)):
                    warnings.append({
                        "source_index": int(source_index),
                        "code": "missing_repeat_instruction",
                    })
                omission_run = True
            else:
                previous_effective_text = text
                omission_run = False
    return warnings

def repeat_content_warnings(
    item: dict, actions_by_source: dict[int, dict], *,
    historical: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Check public repeat tails: head repeats probe five notes, explicit
    anchored/numbered repeats compare the whole known omission interval.

    Candidate heads come from the score or explicit score sections, never a
    phrase boundary or sealed annotation. Sustain/rest timing is not compared.
    """
    def start(row):
        span = row.get('input', {}).get('event_range') or {}
        return int(span.get('start', span.get('start_index', 0)))

    prior = sorted((row for (score, _), row in (historical or {}).items()
                    if score == item.get('score_key') and start(row) < start(item)), key=start)
    entries = []
    for row in [*prior, item]:
        current = row is item
        action_map = actions_by_source if current else {
            int(a['source_index']): a for a in (row.get('reference_plan') or {}).get('actions', [])
            if a.get('source_index') is not None}
        notes = row.get('input', {}).get('notes_without_jianzi') or row.get('input', {}).get('phrase_notes') or []
        for note in notes:
            if note.get('index') is None:
                continue
            action = action_map.get(int(note['index'])) or {}
            text = str(action.get('jianzi_text') or action.get('text') or '').strip()
            section = note.get('section') or {}
            section_key = (section.get('number'), section.get('label'), section.get('title')) if section else None
            midi = None
            silent = (is_structural_note(note)
                      or any(s in str(note.get('jianpu')) for s in ('休止', '延音'))
                      or str(note.get('abc') or '').startswith(('z', '-')))
            if not silent:
                try:
                    midi = tuple(sorted(float(v) for v in target_midis(row, int(note['index']))))
                except (ValueError, TypeError, KeyError):
                    pass
            entries.append({'note': note, 'text': text, 'current': current,
                            'section': section_key, 'midi': midi,
                            'unresolved': not silent and midi is None})
    entries.sort(key=lambda entry: int(entry['note']['index']))
    if not entries:
        return []

    command_pattern = NUMBERED_REPEAT_PATTERN + r'|从头再[作做]|从ㄱ再[作做]|再[作做](?:[二三]声)?'

    def instruction(text):
        if text == OMITTED_PLACEHOLDER:
            return None, ''
        without_anchor = re.sub(r'再[作做](?:起点|标记)(?:ㄱ)?', '', text)
        match = re.search(command_pattern, without_anchor)
        return (match.group(), re.sub(command_pattern, '', without_anchor).strip('〔〕【】[]()（） 。；;：:、\t\n')) if match else (None, '')

    # A head is known only at source zero or a genuine section transition.
    section_heads = {}
    active_section = None
    active_head = None
    for pos, entry in enumerate(entries):
        note = entry['note']
        contiguous = pos > 0 and note['index'] == entries[pos-1]['note']['index'] + 1
        if pos == 0:
            active_head = pos if note['index'] == 0 or (note.get('section') or {}).get('start') else None
        elif (note.get('section') or {}).get('start') or entry['section'] != active_section:
            active_head = pos if contiguous or (note.get('section') or {}).get('start') else None
        elif not contiguous:
            active_head = None
        active_section = entry['section']
        section_heads[pos] = active_head

    warnings = []
    # A standalone endpoint cannot inherit through an intervening attack.
    for pos, entry in enumerate(entries):
        endpoint = re.fullmatch(r'[\[【]?至([一二三四五六七1-7])弦(?:[（(]散音[）)])?[\]】]?', entry['text'])
        if not endpoint or not entry['current']:
            continue
        origin = pos - 1
        def continuation(e):
            text = e['text'].strip('[]【】')
            return (is_structural_note(e['note']) or text in {'', OMITTED_PLACEHOLDER}
                    or bool(re.fullmatch(r'[一二三四五六七1-7]弦', text))
                    or (bool(re.match(r'浒|滸|淌|绰上|綽上|注下|上|下', text))
                        and not re.search(r'勾|剔|抹|挑|托|擘|打|摘|撮|掐|带起|帶起', text)))
        while origin >= 0 and continuation(entries[origin]):
            origin -= 1
        valid = origin >= 0 and bool(re.search(r'滚|拂|輪|轮|历|歷|雷|临|索铃|索鈴', entries[origin]['text']))
        segment = entries[max(origin, 0):pos+1]
        valid = valid and all(
            int(b['note']['index']) == int(a['note']['index'])+1
            for a,b in zip(segment,segment[1:]))
        valid = valid and not any(
            str(e['note'].get('abc') or '').startswith('z')
            or '休止' in str(e['note'].get('jianpu')) for e in segment[:-1])
        if not valid:
            warnings.append({'source_index': int(entry['note']['index']),
                             'code': 'zhi_missing_continuous_gesture'})
            continue
        # Only fully explicit open-string sweeps can be checked without
        # inventing stopped/harmonic positions or a fixed sweep direction.
        text = entries[origin]['text']
        # A control glyph may be attached to a rest rather than a sound.
        # It still needs a gesture, but cannot define a sounding endpoint.
        if entry['midi'] is None:
            continue
        # Intermediate left-hand/string changes establish valid continuity,
        # but a constant open-string pitch model cannot verify their pitches.
        has_modifiers = any(e['text'] and e['text'] != OMITTED_PLACEHOLDER
                            and not is_structural_note(e['note'])
                            for e in entries[origin+1:pos])
        begin = re.search(r'([一二三四五六七1-7])弦', text)
        tuning = item.get('input', {}).get('normalized_tuning', {}).get('open_midi')
        if not has_modifiers and begin and '散' in text and re.search(r'滚|拂|历|歷|雷', text) and isinstance(tuning, list) and len(tuning) == 7:
            number = lambda s: int(s) if s.isdigit() else '一二三四五六七'.index(s) + 1
            first, last = number(begin.group(1)), number(endpoint.group(1))
            step = 1 if last >= first else -1
            expected = [(float(tuning[s-1]),) for s in range(first, last+step, step)]
            segment = entries[origin:pos+1]
            actual = [e['midi'] for e in segment if e['midi'] is not None]
            # Some scores place 至 in the middle of a swept run. Consume
            # only blank continuation rows, never a new authored gesture.
            end = pos + 1
            while len(actual) < len(expected) and end < len(entries):
                later = entries[end]
                if (int(later['note']['index']) != int(segment[-1]['note']['index'])+1
                        or later['section'] != entry['section']
                        or (later['note'].get('section') or {}).get('start')
                        or str(later['note'].get('abc') or '').startswith('z')
                        or '休止' in str(later['note'].get('jianpu'))
                        or later['text'] not in {'', OMITTED_PLACEHOLDER}):
                    break
                segment.append(later)
                if later['midi'] is not None:
                    actual.append(later['midi'])
                end += 1
            # Packed grace/sweep rows are sequential notes, not one pitch.
            # Until expanded by a sequence-aware parser, do not use the
            # scalar jianpu parser's result (which may even be negative).
            if any(len(re.findall(r'[1-7]', str(e['note'].get('jianpu') or ''))) > 1
                   for e in segment):
                continue
            # A matching prefix at the current history boundary is unknown,
            # not a proven missing tail; a later phrase can supply it.
            if end == len(entries) and len(actual) < len(expected) and actual == expected[:len(actual)]:
                continue
            if (not any(e['unresolved'] for e in segment)
                    and all(int(b['note']['index']) == int(a['note']['index'])+1 for a,b in zip(segment,segment[1:]))
                    and actual != expected):
                warnings.append({'source_index': int(entry['note']['index']),
                                 'code': 'zhi_sweep_pitch_mismatch',
                                 'expected_midis': expected, 'actual_midis': actual})
    for pos, entry in enumerate(entries):
        command, ordinary = instruction(entry['text'])
        if not command:
            continue
        numbered = re.fullmatch(NUMBERED_REPEAT_PATTERN, command)
        tail = []
        if not ordinary and entry['midi'] is not None:
            tail.append(entry)
        incomplete_probe = entry['unresolved'] and not ordinary
        previous_source = int(entry['note']['index'])
        # Structural bars do not stop a repeat, but score sections do.
        for later in entries[pos+1:]:
            if later['note']['index'] != previous_source + 1:
                incomplete_probe = bool(numbered) or len(tail) < 5
                break
            previous_source = int(later['note']['index'])
            if ((later['note'].get('section') or {}).get('start')
                    or later['section'] != entry['section']):
                break
            if instruction(later['text'])[0]:
                break
            if is_structural_note(later['note']):
                continue
            if later['unresolved']:
                incomplete_probe = bool(numbered) or len(tail) < 5
                break
            if later['text'] and later['text'] != OMITTED_PLACEHOLDER:
                # Allow an explicit non-sounding control, e.g. 泛止, but an
                # authored attack ends the omitted repeat interval.
                if later['midi'] is not None:
                    break
            if later['midi'] is not None:
                tail.append(later)
        missing = [e for e in tail if e['current'] and not e['text']]
        if missing:
            warnings.append({'source_index': int(missing[0]['note']['index']),
                             'code': 'repeat_omission_marker_missing',
                             'command_source_index': int(entry['note']['index']),
                             'missing_source_indices': [int(e['note']['index']) for e in missing],
                             'missing_event_indices': [int(e['note']['event_index']) for e in missing
                                                       if e['note'].get('event_index') is not None]})
        if numbered and (entry['current'] or any(e['current'] for e in tail)):
            event_index = int(numbered.group(1))
            candidates = [p for p, e in enumerate(entries[:pos])
                          if e['note'].get('event_index') == event_index and e['midi'] is not None]
            location = entry if entry['current'] else next(e for e in tail if e['current'])
            warning = {'source_index': int(location['note']['index']),
                       'command_source_index': int(entry['note']['index']),
                       'target_event_index': event_index, 'compared_events': len(tail)}
            if not candidates:
                warning['code'] = 'repeat_numbered_start_missing'
                warnings.append(warning)
            else:
                pitches = []
                previous_index = None
                for candidate in entries[candidates[-1]:pos]:
                    index = int(candidate['note']['index'])
                    if (previous_index is not None and index != previous_index + 1) or candidate['unresolved']:
                        break
                    previous_index = index
                    if candidate['midi'] is not None:
                        pitches.append(candidate['midi'])
                    if len(pitches) == len(tail):
                        break
                target = [e['midi'] for e in tail]
                if not target or incomplete_probe or pitches != target:
                    warning.update(code='repeat_numbered_pitch_mismatch',
                                   target_midis=target, candidate_midis=pitches)
                    warnings.append(warning)
            continue
        if command in {'从ㄱ再作', '从ㄱ再做'}:
            anchors = [p for p, e in enumerate(entries[:pos])
                       if re.search(r'再[作做](?:起点|标记)(?:ㄱ)?', e['text'])]
            if anchors:
                candidates = []
                previous_index = None
                known = True
                for candidate in entries[anchors[-1]:pos]:
                    index = int(candidate['note']['index'])
                    if ((previous_index is not None and index != previous_index + 1)
                            or candidate['unresolved']):
                        known = False
                        break
                    previous_index = index
                    if candidate['midi'] is not None:
                        candidates.append(candidate)
                for offset, target in enumerate(tail):
                    if not target['current']:
                        continue
                    candidate = candidates[offset] if offset < len(candidates) else None
                    if ((candidate is not None and candidate['midi'] != target['midi'])
                            or (candidate is None and known)):
                        warnings.append({
                            'source_index': int(target['note']['index']),
                            'code': 'repeat_anchor_pitch_mismatch',
                            'command_source_index': int(entry['note']['index']),
                            'command_event_index': entry['note'].get('event_index'),
                            'history_event_index': candidate['note'].get('event_index') if candidate else None,
                            'target_midis': target['midi'],
                            'candidate_midis': candidate['midi'] if candidate else None,
                            'repeat_offset': offset + 1,
                        })
            continue
        if command in {'再作', '再做'} and not ordinary and len(tail) >= 2:
            # Bare 再作 has no named start: reject only when no fully known
            # preceding contiguous motif can explain the whole omission run.
            target = [e['midi'] for e in tail]
            candidates = []
            previous = []
            previous_index = int(entry['note']['index'])
            for earlier in reversed(entries[:pos]):
                if (int(earlier['note']['index']) != previous_index-1
                        or earlier['unresolved'] or earlier['section'] != entry['section']):
                    break
                previous_index = int(earlier['note']['index'])
                if earlier['midi'] is not None:
                    previous.insert(0, earlier['midi'])
                if len(previous) >= len(target):
                    break
            # A short immediately preceding motif can be repeated several
            # times; it need not already exist as one equally long sequence.
            for size in range(1, min(len(previous), len(target))+1):
                motif = previous[-size:]
                if all(pitch == motif[i % size] for i,pitch in enumerate(target)):
                    candidates.append(target)
            for begin in range(pos):
                segment = entries[begin:pos]
                if any(e['unresolved'] for e in segment):
                    continue
                if not all(int(b['note']['index']) == int(a['note']['index'])+1 for a,b in zip(segment,segment[1:])):
                    continue
                pitches = [e['midi'] for e in segment if e['midi'] is not None]
                if len(pitches) >= len(target):
                    candidates.append(pitches[:len(target)])
            if candidates and target not in candidates and any(e['current'] for e in tail):
                location = entry if entry['current'] else next(e for e in tail if e['current'])
                warnings.append({'source_index': int(location['note']['index']),
                                 'code': 'repeat_plain_pitch_mismatch',
                                 'command_source_index': int(entry['note']['index'])})
        probe = tail[:5]
        if (command not in {'从头再作', '从头再做'} or not probe or incomplete_probe
                or not any(e['current'] for e in probe)):
            continue
        candidates = []
        unknown = False
        heads = {section_heads.get(pos)}
        if entries[0]['note']['index'] == 0:
            heads.add(0)
        else:
            unknown = True
        for head in heads:
            if head is None:
                unknown = True
                continue
            pitches = []
            previous_index = None
            for candidate in entries[head:pos]:
                index = int(candidate['note']['index'])
                if previous_index is not None and index != previous_index + 1:
                    break
                previous_index = index
                if candidate['unresolved']:
                    break
                if candidate['midi'] is not None:
                    pitches.append(candidate['midi'])
                if len(pitches) == len(probe):
                    break
            if len(pitches) == len(probe):
                candidates.append(pitches)
            else:
                unknown = True
        target = [e['midi'] for e in probe]
        if not unknown and candidates and all(candidate != target for candidate in candidates):
            location = entry if entry['current'] else next(e for e in probe if e['current'])
            warnings.append({'source_index': int(location['note']['index']),
                             'code': 'repeat_from_head_pitch_mismatch',
                             'command_source_index': int(entry['note']['index']),
                             'compared_events': len(probe), 'target_midis': target,
                             'candidate_midis': candidates})
    # A malformed repeat is not a reason to force an omission run. Validate
    # its anchor/target/content first, then expose missing placeholders only
    # for commands that survived those checks.
    invalid_commands = {
        int(w.get('command_source_index', w['source_index']))
        for w in warnings + repeat_anchor_warnings(item, actions_by_source, historical=historical)
        if w['code'] not in {'repeat_omission_marker_missing', 'missing_repeat_instruction'}
    }
    return [w for w in warnings
            if w['code'] != 'repeat_omission_marker_missing'
            or w['command_source_index'] not in invalid_commands]

def finite_gesture_coverage_warnings(notes: list[dict]) -> list[dict]:
    """A finite pair/sweep cannot authorize an unbounded blank tail."""
    warnings=[]
    endpoint=re.compile(r'至[一二三四五六七1-7]弦(?:[（(]散音[）)])?')
    for pos,note in enumerate(notes):
        text=str(note.get('jianzi') or '').strip()
        if (is_structural_note(note) or any(x in str(note.get('jianpu')) for x in ('休止','延音'))
                or len(re.findall(r'[1-7]',str(note.get('jianpu') or ''))) != 1):
            continue
        strings=AUDIT.parse_li_string_sequence(text)
        count=len(strings)
        if not count and re.search(r'勾剔|抹挑|抹勾|挑勾|涓',text) and len(AUDIT.parse_string_numbers(text))==1:
            count=2
        if not count and re.search(r'滚|拂|历|歷',text):
            start=re.search(r'([一二三四五六七1-7])弦',text)
            end=re.search(r'至([一二三四五六七1-7])弦',text)
            if start and not end:
                for later in notes[pos+1:]:
                    t=str(later.get('jianzi') or '').strip()
                    if is_structural_note(later):continue
                    if endpoint.fullmatch(t):
                        end=re.search(r'至([一二三四五六七1-7])弦',t);break
                    if t or str(later.get('abc') or '').startswith(('z','-')):break
            if start and end:
                count=abs(AUDIT.string_number(start.group(1))-AUDIT.string_number(end.group(1)))+1
        if count < 2:continue
        used=1
        for later in notes[pos+1:]:
            if is_structural_note(later):continue
            t=str(later.get('jianzi') or '').strip()
            if str(later.get('abc') or '').startswith(('z','-')) or any(x in str(later.get('jianpu')) for x in ('休止','延音')):break
            if t and not endpoint.fullmatch(t):break
            # Packed sequential ornament rows need a separate expansion.
            if len(re.findall(r'[1-7]',str(later.get('jianpu') or ''))) != 1:break
            used+=1
            if used > count:
                warnings.append({'source_index':later['index'],
                    'origin_source_index':note['index'],
                    'code':'finite_gesture_uncovered_note','covered_count':count})
                break
    return warnings


def missing_attack_warnings(item: dict, actions_by_source: dict[int, dict]) -> list[dict]:
    """Conservative: only explicit blanks following a fully known simple attack.

    Unknown/compound notation, rests, ties and repeat omissions break the chain.
    Do not infer coverage from a private reference or reject unfinished rows.
    """
    simple = re.compile(
        r"(?:泛起|泛音|按音|散音|散)?"
        r"(?:(?:大指|食指|中指|名指|小指|跪指))?"
        r"(?:(?:[一二三四五六七八九十\d]+徽(?:[一二三四五六七八九十\d]+分)?|徽外))?"
        r"[抹挑勾剔擘托劈打摘搓]"
        r"[一二三四五六七1-7]弦(?:泛止)?"
    )
    previous_simple = None
    previous_pitch = None
    warnings = []
    notes = sorted(item.get('input', {}).get('notes_without_jianzi', []),
                   key=lambda n: int(n['index']))
    for note in notes:
        index = int(note['index'])
        if note.get('structure_token') or note.get('abc') in {'|', '||'}:
            previous_simple = None
            previous_pitch = None
            continue
        abc = str(note.get('abc') or '').strip()
        # Require an explicit pitched attack; rests, ties, ornaments and
        # ambiguous/legacy rows are outside this deliberately narrow check.
        sounding = bool(re.fullmatch(r"(?:\^{1,2}|_{1,2}|=)?[A-Ga-g][,']*\d*(?:/\d*)?", abc))
        if not sounding or note.get('notation_omitted'):
            previous_simple = None
            previous_pitch = None
            continue
        try:
            pitch = tuple(sorted(float(v) for v in target_midis(item, index))) or None
        except (ValueError, TypeError, KeyError, StopIteration):
            pitch = None
        text = actions_by_source.get(index, {}).get('jianzi_text')
        if text == '' and previous_simple is not None:
            # Only the immediately preceding sounding event can justify this
            # warning.  A blank (including a same-pitch blank) cannot remain
            # the "previous simple attack" for another row in a blank run.
            if pitch is not None and previous_pitch is not None and pitch != previous_pitch:
                warnings.append({'source_index': index, 'code': 'missing_independent_attack',
                                 'previous_attack_source_index': previous_simple})
            previous_simple = None
        elif isinstance(text, str) and simple.fullmatch(text.strip()):
            previous_simple = index
        else:
            previous_simple = None
        previous_pitch = pitch
    return warnings

def illegal_blank_after_walk_warnings(
    item: dict, actions_by_source: dict[int, dict],
) -> list[dict]:
    """A single directional walk cannot silently cover a different next pitch."""
    warnings = []
    previous_note = None
    previous_text = None
    notes = sorted(item.get("input", {}).get("notes_without_jianzi", []),
                   key=lambda note: int(note["index"]))
    for note in notes:
        if note.get("structure_token") or note.get("abc") in {"|", "||"}:
            continue
        abc = str(note.get("abc") or "").strip()
        sounding = bool(re.fullmatch(r"(?:\^{1,2}|_{1,2}|=)?[A-Ga-g][,']*\d*(?:/\d*)?", abc))
        if not sounding or note.get("notation_omitted"):
            previous_note = None
            previous_text = None
            continue
        index = int(note["index"])
        text = actions_by_source.get(index, {}).get("jianzi_text")
        if (text == "" and previous_note is not None
                and isinstance(previous_text, str)
                and (previous_text.startswith(("绰上", "注下"))
                     or re.fullmatch(r'(?:急|缓|少)?(?:上|下|进|退|淌|浒|引上)'
                                     r'(?:(?:[一二三四五六七八九十0-9]+徽'
                                     r'(?:[一二三四五六七八九0-9]+分)?|徽外半?))?',
                                     previous_text))
                and not any(t in previous_text for t in ('复', '往来', '进退', '退进'))
                and AUDIT.first_token(previous_text, AUDIT.RIGHT_HAND_TECHNIQUES) is None):
            try:
                before = target_midis(item, int(previous_note["index"]))
                after = target_midis(item, index)
            except (ValueError, TypeError, KeyError, StopIteration):
                pass
            else:
                if (len(before) == len(after) == 1
                        and abs(before[0] - after[0]) > 1e-6):
                    warnings.append({
                        "source_index": index,
                        "previous_walk_source_index": int(previous_note["index"]),
                        "code": "illegal_blank_after_walk",
                    })
        previous_note = note
        previous_text = text
    return warnings

def validate_jianzi_only(
    item: dict, patches: list[dict], *, toward_reference: bool = False,
    require_complete: bool = True,
    historical: dict[tuple[str, str], dict] | None = None,
) -> tuple[bool, dict]:
    """Validate the minimal text-only protocol.

    Syntax/coverage and safe reference omissions are hard conditions. Pitch is
    computed from the authored notation itself and is always advisory during
    the model conversation; the same report exposes a private eligibility bit
    for filtering training exports.
    """
    patches = normalize_patches(patches, prefix="jianzi")
    illegal = [patch for patch in patches
               if patch.get("patch_type") != "SET_JIANZI_TEXT"]
    problems = ([{"code": "non_jianzi_patch_in_minimal_protocol",
                  "patch_type": patch.get("patch_type")} for patch in illegal])
    replay = replay_patches(item["baseline_plan"], patches, strict_before=False)
    problems.extend(replay.errors)
    by_index = {int(action["source_index"]): action for action in replay.actions}
    if require_complete:
        for action in replay.actions:
            if action.get("jianzi_text") is None:
                problems.append({"source_index": int(action["source_index"]),
                                 "code": "pending_jianzi_text"})

    reference_review: list[dict] = []
    if toward_reference:
        # A Base/Fingering repeat-omission marker is an explicit display
        # value, not an ordinary empty target.  Guqinizer may leave it
        # unchanged, but must not erase it by submitting an empty string.
        baseline_by_index = {
            int(action["source_index"]): action.get("jianzi_text")
            for action in (item.get("baseline_plan", {}).get("actions") or [])
        }
        reference_by_index = {
            int(action["source_index"]): (
                action.get("jianzi_text")
                if action.get("jianzi_text") is not None
                else action.get("text")
            )
            for action in (item.get("reference_plan", {}).get("actions") or [])
        }
        for source_index, baseline_text in baseline_by_index.items():
            if (baseline_text == OMITTED_PLACEHOLDER
                    and reference_by_index.get(source_index) == OMITTED_PLACEHOLDER
                    and by_index.get(source_index, {}).get("jianzi_text") == ""):
                problems.append({
                    "source_index": source_index,
                    "code": "repeat_omission_marker_cleared",
                })
        for reference in item.get("reference_plan", {}).get("actions", []):
            if reference.get("confidence_class") not in {"verified", "weak"}:
                continue
            action = by_index.get(int(reference["source_index"]))
            if action is None:
                continue
            expected = reference.get("jianzi_text")
            actual = action.get("jianzi_text")
            equivalent = canonical_jianzi_text(actual) == canonical_jianzi_text(expected)
            if (reference.get("confidence_class") == "verified"
                    and expected is not None and not equivalent):
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "reference_jianzi_text_mismatch",
                    "expected": expected, "actual": actual,
                })
            elif bool(expected) and actual == "":
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "nonempty_reference_jianzi_dropped",
                    "expected": expected,
                })

    warnings: list[dict] = []
    warnings.extend(missing_attack_warnings(item, by_index))
    warnings.extend(illegal_blank_after_walk_warnings(item, by_index))
    warnings.extend(repeat_anchor_warnings(
        item, by_index, historical=historical,
    ))
    warnings.extend(repeat_content_warnings(item, by_index, historical=historical))
    pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                    "details": []}
    try:
        current_source_indices = {
            int(note["index"])
            for note in item["input"].get("notes_without_jianzi", [])
            if note.get("index") is not None
        }
        notes = pitch_audit_notes(item, by_index, historical=historical)
        warnings.extend(w for w in finite_gesture_coverage_warnings(notes)
                        if w['source_index'] in current_source_indices)
        from agents.ToolRuntime.hint_checks import repeated_pitch_hints
        hinted_indices={i for h in repeated_pitch_hints(item,list(by_index.values())) for i in h['source_indices']}
        for detail in AUDIT.ordinary_open_reattack_warnings({
                'metadata': item['input'].get('metadata') or {},
                'open_midi': (item['input'].get('normalized_tuning') or {}).get('open_midi'),
                'notes': notes}):
            if detail['index'] in current_source_indices and detail['index'] not in hinted_indices:
                warnings.append({'source_index': detail['index'],
                    'previous_source_index': detail['previous_index'],
                    'code': detail['code'], 'string': detail['string']})
        pitch_report = AUDIT.audit({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }, 50.0)
        for detail in pitch_report.get("details", []):
            if (int(detail.get("index")) in current_source_indices
                    and detail.get("status") == "mismatched"):
                # 历等多音减字写在第一行，但各发音依次覆盖后续音序。
                # 将单个失配成员报在其实际覆盖行，同时保留减字所在行供修复。
                covered = detail.get('covered_indices') or []
                pairs = detail.get('pairs') or []
                actual = detail.get('actual_midis') or []
                if (len(covered) > 1 and len(covered) == len(pairs) == len(actual)
                        and all(isinstance(pair, dict) and 'absolute_cents' in pair
                                for pair in pairs)):
                    mismatched = [position for position, pair in enumerate(pairs)
                                  if float(pair['absolute_cents']) > 50.0]
                    # Keep a wholly wrong sweep as one warning on its authored
                    # row; only a single wrong covered note can be pinpointed.
                    if len(mismatched) == 1:
                        position = mismatched[0]
                        covered_index = covered[position]
                        if covered_index is not None and int(covered_index) in current_source_indices:
                            pair = pairs[position]
                            sounded = pair['jianzi_midi']
                            resolved = next((sound for sound in detail.get('resolved_pitches') or []
                                             if abs(float(sound['midi']) - float(sounded)) < 1e-6), None)
                            warnings.append({
                                'source_index': int(covered_index),
                                'origin_source_index': int(detail['index']),
                                'origin_event_index': next(
                                    (note.get('event_index') for note in item['input']['notes_without_jianzi']
                                     if int(note['index']) == int(detail['index'])),
                                    int(detail['index']),
                                ),
                                'code': 'jianzi_pitch_mismatch',
                                'resolved_pitches': [deepcopy(resolved)] if resolved else [],
                                'expected_midis': [pair['jianpu_midi']],
                                'actual_midis': [sounded],
                                'pairs': [deepcopy(pair)],
                            })
                            continue
                warnings.append({
                    "source_index": detail.get("index"),
                    "code": "jianzi_pitch_mismatch",
                    "resolved_pitches": deepcopy(detail.get('resolved_pitches') or []),
                    "expected_midis": list(detail.get('expected_midis') or []),
                    "actual_midis": list(detail.get('actual_midis') or []),
                    "li_pitch_count_mismatch": detail.get('reason') == 'li_pitch_count_mismatch',
                    "pairs": detail.get("pairs") or [],
                    "possible_harmonic_state_mismatch": bool(
                        detail.get("possible_harmonic_state_mismatch")
                    ),
                })
        for detail in AUDIT.zhuang_direction_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if (source_index is not None and int(source_index) in current_source_indices
                    and not any(w.get("source_index") == int(source_index)
                                and w.get("code") == "jianzi_pitch_mismatch"
                                for w in warnings)):
                warnings.append({
                    "source_index": int(source_index),
                    "code": "jianzi_pitch_mismatch",
                    "reason": detail.get("reason"),
                })
        for detail in AUDIT.walk_span_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "walk_span_too_large",
                    "technique": detail.get("technique"),
                    "string": detail.get("string"),
                    "from_hui": detail.get("from_hui"),
                    "to_hui": detail.get("to_hui"),
                    "span_cents": detail.get("span_cents"),
                    "threshold_cents": detail.get("threshold_cents"),
                })
        for detail in AUDIT.walk_motion_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": detail.get("code"),
                    "technique": detail.get("technique"),
                    "string": detail.get("string"),
                    "from_hui": detail.get("from_hui"),
                    "to_hui": detail.get("to_hui"),
                    "suggested_technique": detail.get("suggested_technique"),
                })
        for detail in AUDIT.consecutive_walk_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "consecutive_walks_without_attack",
                    "count": detail.get("count"),
                })
        for detail in AUDIT.open_to_stopped_transition_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "open_to_stopped_same_string",
                    "strings": list(detail.get("strings") or []),
                })
        for detail in AUDIT.inherited_gesture_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy((item["input"].get("normalized_tuning") or {}).get("open_midi")),
            "notes": notes,
        }):
            source_index = detail.get('index')
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({"source_index": int(source_index),
                                 **{key: value for key, value in detail.items() if key != 'index'}})
        for detail in AUDIT.harmonic_continuity_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": detail.get("code"),
                    "technique": detail.get("technique"),
                    "strings": list(detail.get("strings") or []),
                    "same_pitch_different_hui": bool(detail.get("same_pitch_different_hui")),
                })
        for detail in AUDIT.invalid_string_warnings({'notes': notes}):
            source_index = detail.get('index')
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    'source_index': int(source_index),
                    'code': 'invalid_string_number',
                    'strings': detail['strings'],
                })
        for detail in AUDIT.nonadjacent_pluck_pair_warnings({"notes": notes}):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "nonadjacent_pluck_pair",
                    "technique": detail.get("technique"),
                    "strings": list(detail.get("strings") or []),
                })
        for detail in AUDIT.same_string_cuo_warnings({"notes": notes}):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "same_string_cuo_pair",
                    "strings": list(detail.get("strings") or []),
                })
        for detail in AUDIT.same_finger_reach_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "same_finger_reach_too_large",
                    "finger": detail.get("finger"),
                    "from_string": detail.get("from_string"),
                    "from_hui": detail.get("from_hui"),
                    "to_string": detail.get("to_string"),
                    "to_hui": detail.get("to_hui"),
                    "hui_delta": detail.get("hui_delta"),
                    "string_delta": detail.get("string_delta"),
                    "distance": detail.get("distance"),
                    "threshold": detail.get("threshold"),
                })
        for detail in AUDIT.low_hui_stopped_warnings({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }):
            source_index = detail.get("index")
            if source_index is not None and int(source_index) in current_source_indices:
                warnings.append({
                    "source_index": int(source_index),
                    "code": "low_hui_stopped_too_cramped",
                    "string": detail.get("string"),
                    "hui": detail.get("hui"),
                })
    except Exception as exc:
        # Tool/parser availability never blocks creative notation generation.
        pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                        "details": [], "unavailable": f"{type(exc).__name__}: {exc}"}

    current_pitch_details = [
        detail for detail in pitch_report.get("details", [])
        if detail.get("index") is not None
        and int(detail["index"]) in current_source_indices
    ]
    current_mismatch_count = sum(
        detail.get("status") == "mismatched" for detail in current_pitch_details
    )
    current_counts = {
        status: sum(detail.get("status") == status for detail in current_pitch_details)
        for status in ("matched", "mismatched", "skipped")
    }
    pitch_report["current_phrase_summary"] = {
        "total_notes": len(current_pitch_details),
        **current_counts,
        "match_rate": (
            round(current_counts["matched"] /
                  max(current_counts["matched"] + current_counts["mismatched"], 1), 6)
            if current_counts["matched"] + current_counts["mismatched"] else None
        ),
    }
    return not problems, {
        "valid": not problems,
        "problems": problems,
        "warnings": warnings,
        "private_reference_review": reference_review,
        "plan": {"actions": replay.actions},
        "patches": patches,
        "pitch_audit": pitch_report,
        "offline_quality": {
            # Pitch comparison is advisory.  A reliably parsed mismatch must
            # be shown to the teacher, but must not reject an otherwise valid
            # edit or silently remove the trajectory from training output.
            "training_eligible": not problems,
            "high_confidence_pitch_mismatches": current_mismatch_count,
        },
    }

def public_pitch_warning_source_indices(
    item: dict, *, historical: dict[tuple[str, str], dict] | None = None,
) -> set[int]:
    """Return current-phrase pitch warnings safe to expose in a Guqinizer prompt.

    This deliberately reuses the same audit path as ``edit_plan``.  It reads
    only the public Base plan, notation and prior model-generated history;
    sealed reference actions are never consulted.
    """
    _, report = validate_jianzi_only(
        item, [], toward_reference=False, require_complete=False,
        historical=historical,
    )
    return {
        int(warning["source_index"])
        for warning in report.get("warnings") or []
        if warning.get("code") == "jianzi_pitch_mismatch"
        and warning.get("source_index") is not None
    }
