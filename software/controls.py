"""Controls cards: the only buttons PlayFellow may press. / بطاقات التحكم: الأزرار الوحيدة التي يمكن لبلاي فيلو ضغطها."""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

KNOWN_KEYS = frozenset(
    list(string.ascii_lowercase)
    + list(string.digits)
    + [f"f{number}" for number in range(1, 12)]
    + ["up", "down", "left", "right", "space", "enter", "shift", "ctrl", "alt", "tab", "esc",
       "backspace", "pageup", "pagedown", "home", "end", "insert", "delete"]
    # punctuation keys used by real games (MK1 Kameo, AoE idle villager, DOOM strafe), as pydirectinput names them
    + [";", ".", ",", "/", "'", "[", "]", "-", "=", "`", "\\"]
)
STOP_KEY = "f12"
TAKEOVER_KEY = "f9"
# Keys the player always owns: a card can never press or hold them.
RESERVED_KEYS = {
    STOP_KEY: ("the emergency stop", "زر إيقاف الطوارئ"),
    TAKEOVER_KEY: ("the take-over key", "زر تولّي القيادة"),
}
BUTTONS = ("left", "right", "middle")


def _reserved(where: str, name: str, key: str) -> None:
    if key in RESERVED_KEYS:
        from messages import tr
        english, arabic = RESERVED_KEYS[key]
        raise CardError(tr(f"{where}: {name} can't use {key.upper()}, that's {english}",
                           f"{name} لا يمكنه استخدام {key.upper()} فهو {arabic}"))
NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class ControlAction:
    name: str
    kind: str
    keys: tuple[str, ...] = ()
    hold: float = 0.15
    dx: int = 0
    dy: int = 0
    button: str = "left"
    says: str = ""
    says_ar: str = ""
    hold_key: str = ""  # For hold_start/hold_stop: which key to hold/release
    release_all_flag: bool = False  # For release_all
    x1: int = 0  # swipe / drag: where the finger goes down and where it lets go, on the 0-1000 grid
    y1: int = 0
    x2: int = 0
    y2: int = 0
    ms: int = 300  # swipe / drag: how long the gesture takes


@dataclass(frozen=True)
class ControlsCard:
    name: str
    game_hint: str
    actions: dict[str, ControlAction]
    name_ar: str = ""
    game_hint_ar: str = ""
    window: str = ""  # §6: game window title match (substring, case-insensitive)
    default_goal: str = ""  # §6: default goal for this card
    default_goal_ar: str = ""  # §6: default goal (Arabic)
    genre: str = "other"  # T34: platformer, racing, puzzle, shooter, sports, rpg, strategy or other (the crew shares tricks by genre)

    @property
    def uses_clicks(self) -> bool:
        return any(action.kind in ("click", "drag") for action in self.actions.values())  # both need the 0-1000 grid


class CardError(Exception):
    """A mistake in a controls card, explained in both languages."""


def normalize_name(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().strip("[]").strip()).upper()


def list_cards(folder) -> list[str]:
    return sorted(path.stem for path in Path(folder).glob("*.yaml"))


def match_card_for_game(game_title: str, folder) -> str:
    """Find a card whose window field matches the game title (§6).

    Returns the card name (stem), or "any_game" if no match.
    Case-insensitive substring match.
    """
    if not game_title:
        return "any_game"

    game_lower = game_title.lower()
    for card_path in Path(folder).glob("*.yaml"):
        try:
            card = load_card(card_path)
            if card.window and card.window.lower() in game_lower:
                return card_path.stem
        except (CardError, Exception):
            continue

    return "any_game"


def load_card(path) -> ControlsCard:
    path = Path(path)
    where = path.name
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        from messages import tr
        raise CardError(tr(f"I can't find the controls card {where}", f"لا أجد بطاقة التحكم {where}")) from None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark else "?"
        from messages import tr
        raise CardError(tr(f"{where}: check line {line}", f"{where}: راجع السطر {line}")) from None
    if not isinstance(data, dict) or not isinstance(data.get("actions"), dict) or not data["actions"]:
        from messages import tr
        raise CardError(tr(f"{where}: needs an 'actions:' list", f"{where}: تحتاج إلى قائمة 'actions:'"))

    actions = {}
    for raw_name, spec in data["actions"].items():
        name = normalize_name(str(raw_name))
        if not NAME_PATTERN.match(name):
            from messages import tr
            raise CardError(
                tr(f"{where}: '{raw_name}' is not a good action name, use words like MOVE_LEFT",
                   f"{where}: '{raw_name}' ليس اسمًا مناسبًا، استخدم كلمات مثل MOVE_LEFT")
            )
        actions[name] = _parse_action(where, name, spec)
    return ControlsCard(
        name=str(data.get("name", path.stem)),
        game_hint=" ".join(str(data.get("game_hint", "")).split()),
        actions=actions,
        name_ar=str(data.get("name_ar", "")),
        game_hint_ar=" ".join(str(data.get("game_hint_ar", "")).split()),
        window=str(data.get("window", "")),
        default_goal=str(data.get("default_goal", "")),
        default_goal_ar=str(data.get("default_goal_ar", "")),
        genre=str(data.get("genre") or "other").strip().lower(),
    )


def _parse_action(where: str, name: str, spec) -> ControlAction:
    if not isinstance(spec, dict):
        from messages import tr
        raise CardError(tr(f"{where}: {name} should look like {{keys: up}}", f"{name} يجب أن يكون مثل {{keys: up}}"))
    says = str(spec.get("says", "")).strip()
    says_ar = str(spec.get("says_ar", "")).strip()
    if "keys" in spec:
        keys = spec["keys"]
        if isinstance(keys, str):
            keys = [keys]
        keys = tuple(str(key).strip().lower() for key in keys)
        if not keys:
            from messages import tr
            raise CardError(tr(f"{where}: {name} has no keys", f"{name} لا يحتوي على مفاتيح"))
        for key in keys:
            _reserved(where, name, key)
            if key not in KNOWN_KEYS:
                from messages import tr
                raise CardError(tr(f"{where}: {name} uses an unknown key '{key}'", f"{name} يستخدم مفتاحًا غير معروف '{key}'"))
        hold = _number(where, name, spec.get("hold", 0.15), 0.02, 3.0)
        return ControlAction(name, "keys", keys=keys, hold=hold, says=says, says_ar=says_ar)
    if "click" in spec:
        return ControlAction(name, "click", button=_button(where, name, spec["click"]), says=says, says_ar=says_ar)
    if "mouse_button" in spec:
        return ControlAction(name, "mouse_button", button=_button(where, name, spec["mouse_button"]), says=says, says_ar=says_ar)
    if "mouse_move" in spec:
        move = spec["mouse_move"]
        if not isinstance(move, list) or len(move) != 2:
            from messages import tr
            raise CardError(tr(f"{where}: {name} mouse_move needs [x, y]", f"{name} يحتاج mouse_move إلى [x, y]"))
        dx, dy = (int(_number(where, name, value, -2000, 2000)) for value in move)
        return ControlAction(name, "mouse_move", dx=dx, dy=dy, says=says, says_ar=says_ar)
    if "wait" in spec:
        return ControlAction(name, "wait", hold=_number(where, name, spec["wait"], 0.05, 5.0), says=says, says_ar=says_ar)
    if "hold_start" in spec:
        hold_key = str(spec["hold_start"]).strip().lower()
        if hold_key not in KNOWN_KEYS:
            from messages import tr
            raise CardError(tr(f"{where}: {name} hold_start uses unknown key '{hold_key}'", f"{name} hold_start يستخدم مفتاحًا غير معروف '{hold_key}'"))
        _reserved(where, name, hold_key)
        return ControlAction(name, "hold_start", hold_key=hold_key, says=says, says_ar=says_ar)
    if "hold_stop" in spec:
        hold_key = str(spec["hold_stop"]).strip().lower()
        if hold_key not in KNOWN_KEYS:
            from messages import tr
            raise CardError(tr(f"{where}: {name} hold_stop uses unknown key '{hold_key}'", f"{name} hold_stop يستخدم مفتاحًا غير معروف '{hold_key}'"))
        return ControlAction(name, "hold_stop", hold_key=hold_key, says=says, says_ar=says_ar)
    if "release_all" in spec and spec["release_all"]:
        return ControlAction(name, "release_all", release_all_flag=True, says=says, says_ar=says_ar)
    if "swipe" in spec:
        gesture = spec["swipe"]
        if not isinstance(gesture, dict) or "from" not in gesture or "to" not in gesture:
            from messages import tr
            raise CardError(tr(f"{where}: {name} swipe needs from: [x, y] and to: [x, y]",
                               f"{name} يحتاج swipe إلى from: [x, y] و to: [x, y]"))
        x1, y1 = _grid_point(where, name, gesture["from"])
        x2, y2 = _grid_point(where, name, gesture["to"])
        ms = int(_number(where, name, gesture.get("ms", 300), 50, 3000))
        return ControlAction(name, "swipe", x1=x1, y1=y1, x2=x2, y2=y2, ms=ms, says=says, says_ar=says_ar)
    if "drag" in spec:
        gesture = spec["drag"]
        ms = int(_number(where, name, gesture.get("ms", 400), 50, 3000)) if isinstance(gesture, dict) else 400
        return ControlAction(name, "drag", ms=ms, says=says, says_ar=says_ar)  # the AI gives the start and end each time
    from messages import tr
    raise CardError(
        tr(f"{where}: {name} needs keys, click, mouse_button, mouse_move, wait, hold_start, hold_stop, release_all, swipe or drag",
           f"{name} يحتاج إلى keys أو click أو mouse_button أو mouse_move أو wait أو hold_start أو hold_stop أو release_all أو swipe أو drag")
    )


def _grid_point(where: str, name: str, value) -> tuple[int, int]:
    """A [x, y] spot on the 0-1000 grid."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        from messages import tr
        raise CardError(tr(f"{where}: {name} needs a spot like [500, 300]", f"{name} يحتاج إلى نقطة مثل [500, 300]"))
    x, y = (int(_number(where, name, coordinate, 0, 1000)) for coordinate in value)
    return x, y


def _number(where: str, name: str, value, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        from messages import tr
        raise CardError(tr(f"{where}: {name} needs a number, not '{value}'", f"{name} يحتاج إلى رقم وليس '{value}'"))
    if not low <= value <= high:
        from messages import tr
        raise CardError(tr(f"{where}: {name} must be between {low} and {high}", f"{name} يجب أن يكون بين {low} و {high}"))
    return float(value)


def _button(where: str, name: str, value) -> str:
    button = str(value).strip().lower()
    if button not in BUTTONS:
        from messages import tr
        raise CardError(tr(f"{where}: {name} button must be left, right or middle", f"{name} الزر يجب أن يكون left أو right أو middle"))
    return button


def describe_card(card: ControlsCard) -> str:
    from messages import get_language, tr
    lines = []
    for action in card.actions.values():
        # Use says_ar when in Arabic and it's not empty, otherwise use says
        if get_language() == "arabic" and action.says_ar:
            what = action.says_ar or action.kind.replace("_", " ")
        else:
            what = action.says or action.kind.replace("_", " ")
        if action.kind == "click":
            what += tr(' (also give "x" and "y" from 0 to 1000 for where to click)',
                      ' (أعطِ أيضًا "x" و "y" من 0 إلى 1000 لمكان الضغط)')
        if action.kind == "drag":
            what += tr(' (also give "x","y" for where to press and "x2","y2" for where to let go, from 0 to 1000)',
                      ' (أعطِ أيضًا "x" و "y" لمكان الضغط و "x2" و "y2" لمكان الإفلات، من 0 إلى 1000)')
        lines.append(f"- {action.name}: {what}")
    return "\n".join(lines)
