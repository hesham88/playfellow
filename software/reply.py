"""Reads the brain's answer and keeps only allowed actions. / يقرأ إجابة العقل ويحتفظ فقط بالأفعال المسموحة."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from controls import ControlsCard, normalize_name

MAX_TEXT = 500
GRID = 1000
FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass(frozen=True)
class Step:
    action: str
    x: int | None = None
    y: int | None = None
    x2: int | None = None  # a drag lets go here
    y2: int | None = None

    def label(self) -> str:
        if self.x2 is not None:
            return f"{self.action}({self.x},{self.y}->{self.x2},{self.y2})"
        return f"{self.action}({self.x},{self.y})" if self.x is not None else self.action


@dataclass
class Reply:
    thought: str = ""
    say: str = ""
    steps: list[Step] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    confused: bool = False
    mission_update: dict | None = None
    hud: dict[str, str | int] = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)  # T32-2: screen facts for knowledge base
    question: str = ""  # T32-2 §3: ambiguity question
    event: str = ""  # T32-2 §4: died/fell/hit events

    def action_names(self) -> list[str]:
        return [step.label() for step in self.steps]


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    cleaned = FENCE.sub("", text).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_reply(text: str, card: ControlsCard, max_actions: int) -> Reply:
    data = extract_json(text)
    if data is None:
        return Reply(confused=True)
    reply = Reply(
        thought=_text(data.get("thought")),
        say=_text(data.get("say")),
        mission_update=_parse_mission_update(data.get("mission_update")),
        hud=_parse_hud(data.get("hud")),
        facts=_parse_facts(data.get("facts")),
        question=_text(data.get("question")),
        event=_text(data.get("event")),
    )
    raw_actions = data.get("actions", [])
    if isinstance(raw_actions, (str, dict)):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        raw_actions = []
    for item in raw_actions:
        step, problem = _to_step(item, card)
        if step is None:
            reply.dropped.append(problem)
        elif len(reply.steps) >= max_actions:
            reply.dropped.append(f"{step.label()} (too many)")
        else:
            reply.steps.append(step)
    return reply


ALIAS_WORDS = ("MOVE", "GO", "PRESS", "HOLD", "KEY", "ARROW", "TAP", "PUSH")
DIRECTIONS = {"NORTH": "UP", "SOUTH": "DOWN", "EAST": "RIGHT", "WEST": "LEFT"}


def alias_on_card(name: str, card: ControlsCard) -> str:
    """A near miss for a button name: drop filler words (MOVE_RIGHT, PRESS_SPACE, ARROW_UP, UP_ARROW) and read compass
    words (NORTH). The answer is returned only when it IS a button on this card; otherwise the name comes back unchanged,
    so nothing that is not on the card can ever get through."""
    pieces = [DIRECTIONS.get(piece, piece) for piece in name.split("_") if piece]
    core = [piece for piece in pieces if piece not in ALIAS_WORDS] or pieces
    for candidate in ("_".join(core), "".join(core)):
        if candidate in card.actions:
            return candidate
    return name


def _to_step(item, card: ControlsCard) -> tuple[Step | None, str]:
    x = y = x2 = y2 = None
    if isinstance(item, dict):
        name = item.get("action") or item.get("name") or ""
        x, y = item.get("x"), item.get("y")
        x2, y2 = item.get("x2"), item.get("y2")
    else:
        name = item
    name = normalize_name(str(name))
    action = card.actions.get(name)
    if action is None:
        name = alias_on_card(name, card)  # MOVE_RIGHT -> RIGHT, but only onto a button that really exists
        action = card.actions.get(name)
    if action is None:
        return None, f"{name or '?'} (not on the controls card)"
    if action.kind == "drag":
        points = [_grid(value) for value in (x, y, x2, y2)]
        if None in points:
            return None, f"{name} (needs x, y, x2 and y2 from 0 to {GRID})"
        return Step(name, *points), ""
    if action.kind != "click":
        return Step(name), ""
    grid_x, grid_y = _grid(x), _grid(y)
    if grid_x is None or grid_y is None:
        return None, f"{name} (needs x and y from 0 to {GRID})"
    return Step(name, grid_x, grid_y), ""


def _grid(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= number <= GRID:
        return None
    return round(number)


def _text(value) -> str:
    return "" if value is None else str(value).strip()[:MAX_TEXT]


def _parse_mission_update(value) -> dict | None:
    """Validate mission_update: type in {push, complete, replace} + non-empty text for push/replace."""
    if not isinstance(value, dict):
        return None
    update_type = value.get("type")
    if update_type not in ("push", "complete", "replace"):
        return None
    # complete doesn't need text, but push and replace do
    if update_type in ("push", "replace"):
        text = value.get("text")
        if not text or not str(text).strip():
            return None
    return value


def _parse_hud(value) -> dict[str, str | int]:
    """Parse HUD readings from the brain's reply."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for name, val in value.items():
        if isinstance(val, (str, int)):
            result[str(name)] = val
    return result


def _parse_facts(value) -> list[dict]:
    """Parse facts list from the brain's reply (T32-2 §1, §2).

    Each fact: {kind: str, name: str, text: str}
    where kind in {controls, moves, goals, scoring, hazards, places}.
    """
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind", "").strip()
        name = item.get("name", "").strip()
        text = item.get("text", "").strip()
        if kind and name and text:
            # Validate kind
            if kind in ("controls", "moves", "goals", "scoring", "hazards", "places"):
                result.append({"kind": kind, "name": name, "text": text})
    return result
