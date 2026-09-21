#!/usr/bin/env python3
"""Constrained decoding for Guqinizer walk endpoints (edit_plan.jianzi_rows).

Motivation
----------
SFT trajectories are thousands of tokens long, and the handful of tokens that
spell a walk endpoint's hui position (e.g. the ``三分`` in ``注下七徽三分``)
receive little gradient.  At inference the Guqinizer therefore rewrites
pitch-correct Base rows into pitch-wrong walks (observed:
``大指七徽三分挑六弦`` -> ``注下七徽六分``, ~100 cents flat).

Rule (one fixed setting)
------------------------
While the Guqinizer is emitting ``edit_plan.jianzi_rows`` values, any walk
endpoint it writes (``绰上/注下/进/退/上/下/浒/淌/引上`` + optional left finger
+ hui position, or a 徽外-family label) must either

1. equal the Base (Fingering) stage's position for that note, or
2. be pitch-correct within ±50 cents (the project's audit standard) on the
   Base row's string or on the string the left hand is currently on.

The string is deliberately not pinned to a single value: a standalone walk
sounds on whatever string the surrounding context left the left hand on, and
the Guqinizer may have rewritten exactly that context (in earlier rounds, or
in rows already completed inside the current tool call — those are replayed
on the fly as each row's closing quote is generated).

Implementation
--------------
A trie-based token whitelist ``LogitsProcessor``.  It incrementally tracks
the generated text, activates only inside ``jianzi_rows`` regions (never
inside reasoning/``<think>``), detects a walk trigger immediately before a
hui numeral (or 徽), and masks every token that could not extend the
endpoint toward an allowed surface.  A trigger-pending mask additionally
constrains the hui integer itself.  Once a complete allowed endpoint is
reached, any token starting outside the endpoint charset resumes free
decoding, so the rest of the row (``挑六弦`` etc.) is untouched.

All pitch math is reused from ``scripts.audit_jianpu_jianzi_pitch`` so the
constraint agrees with the offline audits by construction.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jianpu_jianzi_pitch import (  # noqa: E402
    ZH_NUMBERS,
    new_context,
    parse_hui,
    parse_jianpu,
    parse_jianzi,
    parse_open_midi,
    parse_string_numbers,
    parse_tonic_midi,
    position_pitch,
    string_number,
)

# --- endpoint surface grammar ----------------------------------------------

ZH_HUI_INT = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    11: "十一", 12: "十二", 13: "十三",
}

# Characters that may legally continue a hui surface once the span began.
# 外 extends 徽 into the 徽外 label; 半 extends 徽外 into 徽外半 (its only
# legitimate use — numeric endpoints never take 半).
ENDPOINT_CHARSET = set("一二三四五六七八九十徽分外半")

# Position labels past the 13th hui; they carry no numeric hui.
OUTSIDE_HUI_LABELS = ("徽外", "徽外半")

# Fixed constraint setting: the project's standard audit tolerance.
WALK_TOLERANCE_CENTS = 50.0

# Walk heads mirror WALK_PREFIXES in scripts/audit_guqinizer_walk_pitch.py.
_WALK_HEAD = r"(?:绰上|注下|进复|退复|浒上|引上|绰|注|进|退|浒|淌|上|下)"
_LEFT_FINGER = r"(?:跪指|大指|名指|中指|食指)"
# Matches when the value text so far ENDS with a walk head (+optional finger);
# the next character is then the first numeral of the endpoint.
TRIGGER_SUFFIX_RE = re.compile(rf"(?P<head>{_WALK_HEAD})(?P<finger>{_LEFT_FINGER})?$")
# Single-character heads can be false positives inside ordinary jianzi text
# (e.g. 上/下 elsewhere); the multi-char or unambiguous heads always start a
# real walk, so after them the model MUST open an allowed endpoint.
AMBIGUOUS_HEADS = frozenset({"上", "下", "进", "退", "浒"})


def is_ascii_escape(char: str) -> bool:
    """ASCII letters/digits/whitespace spell forbidden numerals in disguise
    (observed: ``绰上 seven nine`` after ``七`` was banned or merely adjacent).
    Structural JSON punctuation (quote/comma/bracket) stays legitimate."""
    return char.isascii() and (char.isalnum() or char.isspace())
NUMERAL_CHARS = set("一二三四五六七八九十")
# Characters that may prolong a walk head or finger word inside one token.
TRIGGER_CONTINUATION_CHARS = set("绰注进退浒淌引上下大中食名跪指")

# ``撮（大指七徽五弦按音＋三弦散音）`` style rows: stopped partner position.
COMPOUND_POSITION_RE = re.compile(
    r"((?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
    r"(?:(?:一|二|三|四|五|六|七|八|九)分)?)([一二三四五六七1-7])弦按音"
)

# Region openers: the XML parameter form (Qwen3.5 tool calls) and the plain
# JSON-envelope form are both accepted by the eval parser.
REGION_OPENERS = (
    "<parameter=jianzi_rows>",
    '"jianzi_rows"',
)
_TAIL_WINDOW = max(len(opener) for opener in REGION_OPENERS) + 8


def render_hui(hui: float) -> str:
    """7.3 -> ``七徽三分``; 7.0 -> ``七徽`` (inverse of parse_hui)."""
    whole = int(hui)
    fen = int(round((hui - whole) * 10))
    if fen <= 0:
        return f"{ZH_HUI_INT[whole]}徽"
    return f"{ZH_HUI_INT[whole]}徽{ZH_HUI_INT[fen]}分"


def own_position(text: str | None) -> tuple[int, float | str] | None:
    """Resolve a row's own (string, hui); None when not resolvable.

    The hui may be a float (numeric 徽位) or one of the OUTSIDE_HUI_LABELS
    strings (徽外/徽外半) for positions beyond the 13th hui.
    """
    if not text:
        return None

    def _anchor(value: float | str | None) -> float | str | None:
        if value is None or isinstance(value, str):
            return value
        return float(value) if value <= 13.0 else None

    compound = COMPOUND_POSITION_RE.search(text)
    if compound:
        hui = _anchor(parse_hui(compound.group(1)))
        if hui is not None:
            return string_number(compound.group(2)), hui
    hui = _anchor(parse_hui(text))
    strings = parse_string_numbers(text)
    if hui is not None and len(strings) == 1:
        return strings[0], hui
    return None


class WalkConstraintTable:
    """Per-note allowed endpoint surfaces plus the live left-hand context.

    ``initial_live`` snapshots, for every event index, the string the left
    hand is on just before that note when the *current* plan (Base plus any
    edits applied in earlier rounds) is replayed in musical order.  The
    processor keeps the snapshot up to date by replaying each row the model
    completes inside the current tool call, before later rows open.
    """

    def __init__(
        self,
        item: dict,
        current_text: dict[int, str | None] | None = None,
        tolerance_cents: float = WALK_TOLERANCE_CENTS,
    ) -> None:
        source = item.get("input") or {}
        metadata = source.get("metadata") or {}
        self.open_midi = (source.get("normalized_tuning") or {}).get("open_midi")
        if not self.open_midi:
            self.open_midi = parse_open_midi(metadata)
        self.tonic = parse_tonic_midi(metadata)
        self.tolerance_cents = float(tolerance_cents)

        base_text = {
            int(action["source_index"]): str(action.get("jianzi_text") or "")
            for action in (item.get("baseline_plan") or {}).get("actions") or []
            if action.get("source_index") is not None
        }
        notes = sorted(
            (note for note in source.get("notes_without_jianzi") or []
             if note.get("index") is not None),
            key=lambda note: int(note["index"]),
        )

        # Replay the current plan in musical order to snapshot the inherited
        # left-hand string at every note.
        context = new_context()
        live_by_index: dict[int, int | None] = {}
        live_hui_by_index: dict[int, float | str | None] = {}
        for note in notes:
            live_by_index[int(note["index"])] = context.get("active_left_string")
            live_hui_by_index[int(note["index"])] = context.get("active_left_hui")
            text = (current_text or base_text).get(int(note["index"]))
            if text:
                try:
                    parse_jianzi(text, self.open_midi, context)
                except Exception:
                    pass  # constraint computation must never break generation

        self.static: dict[int, frozenset[str]] = {}
        self.base_string: dict[int, int | None] = {}
        self.initial_live: dict[int, int | None] = {}
        # The position immediately before this event is distinct from the
        # row's Base position.  A walk is a movement, so 绰上/注下 may not
        # merely restate this inherited endpoint.
        self.initial_live_hui: dict[int, float | str | None] = {}
        self.by_string: dict[int, dict[int, frozenset[str]]] = {}
        for note in notes:
            event = note.get("event_index")
            if event is None:
                continue
            event = int(event)
            index = int(note["index"])
            position = own_position(base_text.get(index))
            # Clause 1: identical to the Base stage's position.
            anchors: set[str] = set()
            if position is not None:
                hui = position[1]
                anchors.add(hui if isinstance(hui, str) else render_hui(hui))
            self.static[event] = frozenset(anchors)
            self.base_string[event] = position[0] if position else None
            self.initial_live[event] = live_by_index.get(index)
            self.initial_live_hui[event] = live_hui_by_index.get(index)
            # Clause 2: pitch-correct surfaces per string (looked up later
            # for the base and live strings only).
            target = parse_jianpu(note.get("jianpu"), self.tonic)
            by_string: dict[int, frozenset[str]] = {}
            if target is not None:
                for string in range(1, 8):
                    surfaces: set[str] = set()
                    for label in OUTSIDE_HUI_LABELS:
                        pitch, error = position_pitch(
                            string, label, self.open_midi, "stopped",
                        )
                        if error is None and abs(pitch - target) * 100 <= tolerance_cents:
                            surfaces.add(label)
                    for whole in range(1, 14):
                        # The guqin has exactly 13 hui; past 十三徽 the
                        # corpus uses the 徽外 labels, so fractional
                        # positions only exist below hui 13.
                        for fen in range(10) if whole < 13 else (0,):
                            hui = whole + fen / 10
                            pitch, error = position_pitch(
                                string, hui, self.open_midi, "stopped",
                            )
                            if error is None and abs(pitch - target) * 100 <= tolerance_cents:
                                surfaces.add(render_hui(hui))
                    by_string[string] = frozenset(surfaces)
            self.by_string[event] = by_string

    def allowed_for(self, event: int, live_string: int | None) -> frozenset[str]:
        parts = [self.static.get(event, frozenset())]
        for string in (self.base_string.get(event), live_string):
            if string is not None:
                parts.append(self.by_string.get(event, {}).get(int(string), frozenset()))
        result = frozenset().union(*parts)
        return result

    def preview_allowed(self) -> dict[int, list[str]]:
        """Maximal surfaces per event for trace logging (live overrides
        decided mid-generation are not included)."""
        return {
            event: sorted(self.allowed_for(event, self.initial_live.get(event)))
            for event in sorted(self.static)
        }

    def fresh_replay_context(self) -> dict:
        return new_context()


def build_walk_constraints(
    item: dict,
    current_text: dict[int, str | None] | None = None,
    tolerance_cents: float = WALK_TOLERANCE_CENTS,
) -> WalkConstraintTable:
    return WalkConstraintTable(item, current_text, tolerance_cents)


# --- byte-level vocab text --------------------------------------------------

def _byte_level_decoder() -> dict[str, int]:
    """GPT-2 style ``bytes_to_unicode`` inverse (char -> byte)."""
    bases = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    outputs = bases[:]
    extra = 0
    for byte in range(256):
        if byte not in bases:
            bases.append(byte)
            outputs.append(256 + extra)
            extra += 1
    return {chr(value): byte for value, byte in zip(outputs, bases)}


_BYTE_DECODER = _byte_level_decoder()


def build_id_texts(tokenizer: Any, probe_count: int = 300) -> dict[int, str]:
    """Map every vocab id to its decoded surface text.

    Byte-level BPE token strings are inverted locally (fast); a sample is
    cross-checked against ``tokenizer.decode`` so a different tokenizer
    family fails loudly instead of silently corrupting the constraint.
    """
    vocab: dict[str, int] = tokenizer.get_vocab()
    id_texts: dict[int, str] = {}
    for token, index in vocab.items():
        try:
            raw = bytes(_BYTE_DECODER[char] for char in token)
        except KeyError:
            raw = token.encode("utf-8")
        id_texts[int(index)] = raw.decode("utf-8", errors="replace")

    clean = sorted(index for index, text in id_texts.items() if "\ufffd" not in text)
    step = max(1, len(clean) // max(1, probe_count))
    mismatch = 0
    probed = 0
    for index in clean[::step][:probe_count]:
        expected = tokenizer.decode([index], skip_special_tokens=False)
        probed += 1
        if expected != id_texts[index]:
            mismatch += 1
    if mismatch:
        raise RuntimeError(
            f"tokenizer byte-level probe mismatch on {mismatch}/{probed}"
            " sampled ids; refusing to constrain with a wrong id->text map"
        )
    return id_texts


# --- streaming constraint state machine --------------------------------------

(
    ST_OUTSIDE,     # anywhere before a jianzi_rows region
    ST_ROWS,        # inside the rows array, between elements
    ST_ROW_INDEX,   # inside a row, reading the numeric index
    ST_ROW_AFTER,   # after the index comma, expecting the value quote
    ST_VALUE,       # inside the quoted jianzi text
    ST_SPAN,        # inside a constrained walk-endpoint span
) = range(6)


class _StaticTable:
    """Adapter so tests can pass a plain {event: surfaces} mapping."""

    def __init__(self, mapping: dict[int, frozenset[str]]) -> None:
        self.mapping = {int(key): frozenset(value) for key, value in mapping.items() if value}
        self.open_midi = None

    def allowed_for(self, event: int, live_string: int | None) -> frozenset[str]:
        return self.mapping.get(event, frozenset())

    def preview_allowed(self) -> dict[int, list[str]]:
        return {event: sorted(value) for event, value in self.mapping.items()}

    def fresh_replay_context(self) -> dict:
        return None


class WalkHuiConstraintProcessor:
    """Callable passed to ``model.generate(logits_processor=[...])``.

    ``prompt_length`` is the encoded prompt size so only newly generated
    tokens drive the state machine.  ``table`` is a :class:`WalkConstraintTable`
    (or a plain ``{event: surfaces}`` mapping for tests).  Rows whose resolved
    allowed set is empty stay unconstrained.
    """

    def __init__(
        self,
        id_texts: dict[int, str],
        table: WalkConstraintTable | dict[int, frozenset[str]],
        prompt_length: int,
    ) -> None:
        self.id_texts = dict(id_texts)
        self.table = table if not isinstance(table, dict) else _StaticTable(table)
        self.cursor = int(prompt_length)
        self.state = ST_OUTSIDE
        self.tail = ""
        self.row_index: int | None = None
        self._index_buffer = ""
        self._rows_depth = 0
        self.value_text = ""
        self._escaped = False
        self.span_prefix = ""
        self.span_allowed: frozenset[str] = frozenset()
        self._span_done = False
        self._prefix_cache: dict[str, frozenset[int]] = {}
        # Trigger-pending state: the value text currently ends with a walk
        # head (+optional finger), so the NEXT token may open the endpoint.
        # Masking there is what actually constrains the hui integer itself.
        self._pending_allowed: frozenset[str] | None = None
        self._pending_cache_by_allowed: dict[tuple[frozenset[str], bool], frozenset[int]] = {}
        self._pending_head_ambiguous = False
        # Live left-hand context: replayed from rows the model completes
        # inside this tool call, mirroring the audit's history replay.
        self._replay_context = self.table.fresh_replay_context()
        self._replay_last_index: int | None = None
        self._row_allowed: frozenset[str] = frozenset()
        self._pending_head: str | None = None
        # Lazy token indexes so each masked step only scans tokens whose
        # first character could matter, plus one precomputed "free exit" set.
        self._charset_first_tokens: list[tuple[int, str]] | None = None
        self._exit_ids: frozenset[int] | None = None
        self.stats = {
            "regions_entered": 0,
            "rows_seen": 0,
            "unconstrained_rows": 0,
            "rows_replayed": 0,
            "spans_entered": 0,
            "spans_completed": 0,
            "mask_steps": 0,
            "mid_token_span_violations": 0,
            "mask_fallbacks": 0,
        }

    # -- transformers LogitsProcessor protocol -------------------------------

    def __call__(self, input_ids: Any, scores: Any):
        import torch

        sequence = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        for token_id in sequence[self.cursor:]:
            self.cursor += 1
            self._feed_token(int(token_id))
        if self.state == ST_SPAN:
            allowed = self._allowed_ids_for_prefix(self.span_prefix)
        elif self.state == ST_VALUE and self._pending_allowed is not None:
            allowed = self._allowed_ids_for_pending()
        elif self.state == ST_VALUE:
            allowed = self._allowed_ids_preventing_noop_walk_head()
        else:
            return scores
        if allowed is None:
            # Safety valve: an empty mask would deadlock decoding.
            self.stats["mask_fallbacks"] += 1
            return scores
        banned = [index for index in range(scores.shape[-1]) if index not in allowed]
        if banned:
            scores.index_fill_(
                1, torch.tensor(banned, device=scores.device, dtype=torch.long),
                float("-inf"),
            )
            self.stats["mask_steps"] += 1
        return scores

    # -- token / char consumption ---------------------------------------------

    def _feed_token(self, token_id: int) -> None:
        text = self.id_texts.get(token_id)
        if text:
            self._feed_chars(text)

    def _feed_chars(self, text: str) -> None:
        for char in text:
            self.tail = (self.tail + char)[-_TAIL_WINDOW:]
            state = self.state
            if state == ST_OUTSIDE:
                if any(self.tail.endswith(opener) for opener in REGION_OPENERS):
                    self.state = ST_ROWS
                    self.stats["regions_entered"] += 1
                continue
            if state == ST_ROWS:
                if char == "[":
                    self._rows_depth += 1
                    if self._rows_depth >= 2:
                        self.state = ST_ROW_INDEX
                        self._index_buffer = ""
                elif char == "]":
                    self._rows_depth -= 1
                    if self._rows_depth <= 0:
                        self._exit_region()
                # commas / whitespace / colons are skipped
                continue
            if state == ST_ROW_INDEX:
                if char.isdigit() or char in ZH_NUMBERS:
                    self._index_buffer += char
                elif char == ",":
                    self.row_index = self._parse_index(self._index_buffer)
                    self.state = ST_ROW_AFTER
                elif char == "]":
                    self._rows_depth -= 1
                    if self._rows_depth <= 0:
                        self._exit_region()
                    else:
                        self.state = ST_ROWS
                elif not char.isspace():
                    self._exit_region()
                continue
            if state == ST_ROW_AFTER:
                if char == '"':
                    self.state = ST_VALUE
                    self.value_text = ""
                    self._escaped = False
                    self._pending_allowed = None
                    self._row_allowed = self.table.allowed_for(
                        self.row_index, self._live_string(),
                    ) if self.row_index is not None else frozenset()
                    self.stats["rows_seen"] += 1
                    if not self._row_allowed:
                        self.stats["unconstrained_rows"] += 1
                elif char == "]":
                    self._exit_region()
                elif not char.isspace():
                    self._exit_region()
                continue
            if state == ST_SPAN:
                combined = self.span_prefix + char
                if self._is_prefix_of_allowed(combined):
                    self.span_prefix = combined
                    self.value_text += char
                    if not self._span_done and self.span_prefix in self.span_allowed:
                        self._span_done = True
                        self.stats["spans_completed"] += 1
                    continue
                if (self.span_prefix in self.span_allowed
                        and char not in ENDPOINT_CHARSET
                        and not is_ascii_escape(char)):
                    self._exit_span()  # falls through to value handling
                else:
                    # Only reachable for tokens that entered the span
                    # mid-token without passing the mask.
                    self.stats["mid_token_span_violations"] += 1
                    self._exit_span()
                    self.value_text += char
                    continue
            # ST_VALUE
            if self._escaped:
                self._escaped = False
                self.value_text += char
                continue
            if char == "\\":
                self._escaped = True
                continue
            if char == '"':
                self._close_row()
                continue
            if char in NUMERAL_CHARS and self._at_walk_trigger():
                allowed = self._walk_allowed()
                if allowed:
                    self._row_allowed = allowed
                    self._enter_span(char)
                    continue
            elif char == "徽" and self._at_walk_trigger():
                # 徽外-family endpoints open the span on 徽 instead of a
                # numeral (they carry no numeric hui).
                allowed = self._walk_allowed()
                if allowed and any(
                    surface.startswith("徽") for surface in allowed
                ):
                    self._row_allowed = allowed
                    self._enter_span(char)
                    continue
            self.value_text += char
            # Arm the pre-endpoint mask when the value now ends with a walk
            # head, so the hui integer itself cannot dodge the whitelist.
            match = TRIGGER_SUFFIX_RE.search(self.value_text)
            self._pending_head = match.group("head") if match else None
            self._pending_allowed = self._walk_allowed() if match else None
            self._pending_head_ambiguous = (
                bool(match) and match.group("head") in AMBIGUOUS_HEADS
            )

    def _enter_span(self, char: str) -> None:
        self.state = ST_SPAN
        self.span_prefix = char
        self.span_allowed = self._row_allowed
        self._span_done = False
        self._prefix_cache = {}
        self.value_text += char
        self._pending_allowed = None
        self._pending_head = None
        self.stats["spans_entered"] += 1

    def _close_row(self) -> None:
        self._replay_row()
        self.state = ST_ROWS
        self._pending_allowed = None

    def _replay_row(self) -> None:
        """Mirror the audit's context replay for the row just completed.

        This is what lets a later row in the SAME tool call see the string
        context the Guqinizer has just established, before edit_plan has
        actually executed anything.
        """
        index = self.row_index
        if index is None:
            return
        if self._replay_context is not None and self.value_text:
            try:
                parse_jianzi(self.value_text, self.table.open_midi, self._replay_context)
                self.stats["rows_replayed"] += 1
            except Exception:
                pass
        if self._replay_last_index is None or index > self._replay_last_index:
            self._replay_last_index = index

    def _live_string(self) -> int | None:
        """Best estimate of the string a walk at the current row sounds on.

        Prefers the in-call replay state when the model has already
        completed an earlier row in this tool call; otherwise falls back to
        the table's snapshot of the current plan.
        """
        index = self.row_index
        if (self._replay_context is not None
                and self._replay_last_index is not None
                and index is not None
                and self._replay_last_index < index):
            value = self._replay_context.get("active_left_string")
            if value is not None:
                return int(value)
        if isinstance(self.table, WalkConstraintTable) and index is not None:
            return self.table.initial_live.get(index)
        return None

    def _live_hui(self) -> float | str | None:
        """Return the endpoint held immediately before the current row.

        Completed rows in this same ``edit_plan`` call take priority.  This
        makes ``…注下十徽八分 + 注下十徽八分`` and a following
        ``绰上十徽八分`` fail exactly like the offline stateful audit.
        """
        index = self.row_index
        if (self._replay_context is not None
                and self._replay_last_index is not None
                and index is not None
                and self._replay_last_index < index):
            value = self._replay_context.get("active_left_hui")
            if value is not None:
                return value
        if isinstance(self.table, WalkConstraintTable) and index is not None:
            return self.table.initial_live_hui.get(index)
        return None

    @staticmethod
    def _surface_for_hui(hui: float | str | None) -> str | None:
        if hui is None:
            return None
        return hui if isinstance(hui, str) else render_hui(float(hui))

    def _walk_allowed_for_head(self, head: str | None) -> frozenset[str]:
        """Endpoint whitelist after applying non-zero-motion walk semantics."""
        allowed = self._row_allowed
        # 绰上 and 注下 are endpoint-bearing slides.  Writing their endpoint
        # equal to the previous stopped position is a no-op, not a walk.
        if head in {"绰上", "注下"}:
            previous = self._surface_for_hui(self._live_hui())
            if previous is not None:
                allowed = frozenset(surface for surface in allowed if surface != previous)
        return allowed

    def _walk_allowed(self) -> frozenset[str]:
        match = TRIGGER_SUFFIX_RE.search(self.value_text)
        head = match.group("head") if match else self._pending_head
        return self._walk_allowed_for_head(head)

    def _allowed_ids_preventing_noop_walk_head(self) -> frozenset[int] | None:
        """Block a zero-motion 绰上/注下 *before* its head is emitted.

        Closing an already emitted head produced malformed ``注下`` rows and
        repeat loops.  If the current position has no legal different
        endpoint, mask the token that would complete the head instead.  This
        leaves ordinary phrasing and alternative non-walk edits available.
        """
        if not self._row_allowed or self._live_hui() is None:
            return None
        forbidden = {
            head for head in ("绰上", "注下")
            if not self._walk_allowed_for_head(head)
        }
        if not forbidden:
            return None
        prefix = self.value_text[-16:]
        allowed = frozenset(
            token_id for token_id, text in self.id_texts.items()
            if text and "\ufffd" not in text
            and not any(head in prefix + text for head in forbidden)
        )
        return allowed or None

    def _at_walk_trigger(self) -> bool:
        return bool(TRIGGER_SUFFIX_RE.search(self.value_text))

    def _parse_index(self, buffer: str) -> int | None:
        if not buffer:
            return None
        if buffer.isdigit():
            return int(buffer)
        value = ZH_NUMBERS.get(buffer)
        return int(value) if value is not None else None

    def _exit_region(self) -> None:
        self.state = ST_OUTSIDE
        self.row_index = None
        self._rows_depth = 0
        self.value_text = ""
        self._pending_allowed = None
        self._pending_head = None

    def _exit_span(self) -> None:
        self.state = ST_VALUE

    # -- span arithmetic --------------------------------------------------------

    def _is_prefix_of_allowed(self, prefix: str) -> bool:
        return any(surface.startswith(prefix) for surface in self.span_allowed)

    def _token_admissible(self, prefix: str, text: str) -> bool:
        """Could this token be generated from ``prefix`` inside the span?

        Mirrors the per-character consumer: every charset char must extend a
        prefix of an allowed surface; a non-charset char is legal once the
        current prefix is a complete allowed endpoint (it exits the span and
        the rest of the token is free text).
        """
        if not text or "\ufffd" in text:
            return False
        current = prefix
        for char in text:
            combined = current + char
            if char in ENDPOINT_CHARSET and self._is_prefix_of_allowed(combined):
                current = combined
                continue
            if (current in self.span_allowed and char not in ENDPOINT_CHARSET
                    and not is_ascii_escape(char)):
                return True
            return False
        return True

    def _allowed_ids_for_prefix(self, prefix: str) -> frozenset[int] | None:
        cached = self._prefix_cache.get(prefix)
        if cached is not None:
            return cached or None
        if self._charset_first_tokens is None:
            self._charset_first_tokens = [
                (token_id, text) for token_id, text in self.id_texts.items()
                if text and text[0] in ENDPOINT_CHARSET and "\ufffd" not in text
            ]
        extending = {
            token_id for token_id, text in self._charset_first_tokens
            if self._token_admissible(prefix, text)
        }
        allowed_ids = frozenset(extending)
        if prefix in self.span_allowed:
            # A complete endpoint may be followed by any token whose first
            # character cannot continue a hui surface (the span then ends).
            allowed_ids = frozenset(extending | self._free_exit_ids())
        self._prefix_cache[prefix] = allowed_ids
        return allowed_ids or None

    def _free_exit_ids(self) -> frozenset[int]:
        if self._exit_ids is None:
            self._exit_ids = frozenset(
                token_id for token_id, text in self.id_texts.items()
                if text and text[0] not in ENDPOINT_CHARSET
                and not is_ascii_escape(text[0]) and "\ufffd" not in text
            )
        return self._exit_ids

    # -- trigger-pending mask (constrains the hui integer itself) -------------

    def _allowed_ids_for_pending(self) -> frozenset[int] | None:
        allowed = self._pending_allowed
        if allowed is None:
            return None
        if not allowed:
            # This is normally unreachable: zero-motion heads are masked
            # before they complete. Keep the safety valve for malformed
            # multi-character tokens rather than deadlocking generation.
            return None
        key = (allowed, self._pending_head_ambiguous)
        cached = self._pending_cache_by_allowed.get(key)
        if cached is None:
            cached = frozenset(
                token_id for token_id, text in self.id_texts.items()
                if self._token_admissible_pending(
                    text, allowed, self._pending_head_ambiguous)
            )
            self._pending_cache_by_allowed[key] = cached
        return cached or None

    def _token_admissible_pending(
        self, text: str, allowed: frozenset[str], head_ambiguous: bool,
    ) -> bool:
        """Could this token follow a value that currently ends with a walk
        head?  Trigger/finger characters may prolong the head; the first
        numeral must open an allowed surface (then normal span rules apply
        for the rest of the token); a leading 徽 must open an allowed
        徽外-family label; any other character aborts the trigger and is
        free text.
        """
        if not text or "\ufffd" in text:
            return False
        index = 0
        while index < len(text) and text[index] in TRIGGER_CONTINUATION_CHARS:
            index += 1
        if index >= len(text):
            return True
        first = text[index]
        if first == "徽":
            if not any(surface.startswith("徽") for surface in allowed):
                return False
            prefix = ""
            for char in text[index:]:
                combined = prefix + char
                if any(surface.startswith(combined) for surface in allowed):
                    prefix = combined
                    continue
                if (prefix in allowed and char not in ENDPOINT_CHARSET
                        and not is_ascii_escape(char)):
                    return True
                return False
            return True
        if first not in NUMERAL_CHARS:
            # Abort = the trigger was a false positive.  Only single-character
            # heads may abort, and never via ASCII alnum/whitespace tokens:
            # the model evades a banned hui by spelling it in English
            # (``绰上 seven nine``) unless those tokens are blocked too.
            return head_ambiguous and not is_ascii_escape(first)
        if not any(surface.startswith(first) for surface in allowed):
            return False
        prefix = first
        for char in text[index + 1:]:
            combined = prefix + char
            if any(surface.startswith(combined) for surface in allowed):
                prefix = combined
                continue
            if (prefix in allowed and char not in ENDPOINT_CHARSET
                    and not is_ascii_escape(char)):
                return True
            return False
        return True


def build_processor_from_tokenizer(
    tokenizer: Any,
    item: dict,
    prompt_length: int,
    current_text: dict[int, str | None] | None = None,
) -> tuple[WalkHuiConstraintProcessor, WalkConstraintTable]:
    table = build_walk_constraints(item, current_text)
    processor = WalkHuiConstraintProcessor(
        build_id_texts(tokenizer), table, prompt_length,
    )
    return processor, table


__all__ = [
    "ENDPOINT_CHARSET",
    "WalkConstraintTable",
    "WalkHuiConstraintProcessor",
    "build_id_texts",
    "build_walk_constraints",
    "build_processor_from_tokenizer",
    "render_hui",
    "own_position",
]
