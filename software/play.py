"""PlayFellow's play loop: look, mark, think, check, act. / حلقة اللعب: انظر، علّم، فكّر، تحقّق، تصرّف.

    python -m practice_arena.arena        first, in its own window: the small test game
    python play.py                        then play it with the built-in brain (no account, no key, no network)
    python play.py --brain gemini         the same with a cloud brain and your own key
    python play.py --save-key gemini      keep your key in Windows Credential Manager

Every turn it takes a picture of the one game window you chose (look), can put labels on it (mark), asks the brain
what to do (think), keeps only the actions on that game's controls card (check, reply.py), and presses them only
while that window is in front (act, hands.py). F12, or the mouse in the top-left corner, stops everything at once,
and the time and AI-call limits end a session. Launchers, stores, browsers and payment windows are never played.
"""
from __future__ import annotations

import argparse
import dataclasses
import getpass
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from brain import Brain, BrainError, make_brain
from brain.prompts import system_prompt, turn_text
from controls import ControlsCard, load_card
from eyes import Screen, add_grid, make_dpi_aware, shrink
from games import is_blocked, needs_online_warning
from grounding import Mark, draw_marks, label_marks, legend
from hands import CornerStop, Hands
from memory import Memory, Moment
from messages import msg, tr
from reply import Reply, parse_reply
from safety import Safety, listen_for_f12
from settings import KEY_NAMES, Settings, load_api_key, load_settings, save_api_key

SOFTWARE = Path(__file__).resolve().parent
MEMORY_IMAGE_WIDTH = 320  # the older pictures the brain sees again are small: they only remind it what happened
MAX_ERRORS_IN_ROW = 3
MAX_WAIT_SECONDS = 30.0
AGE_FILE = "age-18-confirmed.txt"


def _quiet(kind: str, data: Any) -> None:
    pass


@dataclass
class Parts:
    brain: Brain
    card: ControlsCard
    screen: Any  # eyes.Screen, or a pretend one in tests
    hands: Hands
    safety: Safety
    memory: Memory
    marker: Callable[[Image.Image], list[Mark]] | None = None  # labels for the picture: None = no labels yet


def system_for(parts: Parts, settings: Settings) -> str:
    return system_prompt(settings, parts.card)


def _look(parts: Parts, settings: Settings, image: Image.Image) -> tuple[Image.Image, str]:
    """The picture the brain sees, and the words that explain its labels."""
    if parts.card.uses_clicks:
        image = add_grid(image)
    if parts.marker is None:
        return image, ""
    marks = parts.marker(image)
    me = next((m for m in marks if m.kind == "me"), None)
    labelled = label_marks([m for m in marks if m.kind != "me"], me)
    if not labelled:
        return image, ""
    return draw_marks(image, labelled), legend(labelled)


def take_turn(parts: Parts, settings: Settings, system: str, on_event=_quiet) -> Reply | None:
    """One look-think-act turn. None when there was nothing to do (no window, or it is not in front)."""
    p, s = parts, settings
    box = p.screen.find(s.game_window)
    if box is None:
        on_event("status", msg("no_window", title=s.game_window))
        return None
    if not p.screen.in_front(s.game_window):
        on_event("status", msg("click_game"))  # no picture of a covered game, no AI call, no key
        return None
    image, labels = _look(p, s, shrink(p.screen.grab(box), s.image_width))
    text = turn_text(s.goal, p.memory.as_text(), [], None)
    if labels:
        text = f"{labels}\n\n{text}"
    p.safety.count_ai_call()
    raw = p.brain.ask(system, text, [moment.image for moment in p.memory.recent()] + [image])
    reply = parse_reply(raw, p.card, s.max_actions_per_step)
    on_event("thought", reply.thought)
    if reply.say:
        on_event("say", reply.say)
    if reply.dropped:
        on_event("dropped", reply.dropped)
    for step in reply.steps:
        if p.safety.stopped or not p.screen.in_front(s.game_window):
            break  # F12, the corner, a limit, or the player clicked away: nothing more is pressed
        p.hands.do(step, box)
    on_event("actions", reply.action_names())
    p.memory.add(Moment(shrink(image, MEMORY_IMAGE_WIDTH), reply.thought, tuple(reply.action_names()), reply.say))
    return reply


def play(parts: Parts, settings: Settings, on_event=_quiet, max_turns: int | None = None) -> str:
    """Take turns until something stops play. Returns the reason (a message key, e.g. "stopped_f12")."""
    p, s = parts, settings
    p.safety.start()
    p.hands.resume()
    system = system_for(p, s)
    errors = turns = 0
    try:
        while p.safety.check() is None:
            if max_turns is not None and turns >= max_turns:
                p.safety.stop("round_done")
                break
            try:
                take_turn(p, s, system, on_event)
            except CornerStop:
                p.safety.stop("stopped_corner")
                break
            except BrainError as error:
                errors += 1
                on_event("status", error.friendly())
                if not error.retry:
                    p.safety.stop(error.key)
                    break
                if errors >= MAX_ERRORS_IN_ROW:
                    p.safety.stop("too_many_errors")
                    break
                if p.safety.wait(min(error.wait_seconds * 2 ** (errors - 1), MAX_WAIT_SECONDS)):
                    break
                continue
            errors = 0
            turns += 1
            if p.safety.wait(s.look_every_seconds):
                break
    finally:
        p.hands.on_stop()  # every held key goes up, whatever ended the session
    return p.safety.reason or "stopped_button"


def check_window(title: str, online_ok: bool) -> str | None:
    """Why this window may not be played, or None. Online games need the player's explicit --online-ok."""
    if is_blocked(title):
        return msg("blocked_window", title=title)
    if needs_online_warning(title) and not online_ok:
        return msg("online_game_warning", title=title) + " " + tr(
            "To go ahead anyway, add --online-ok.", "للمتابعة رغم ذلك، أضف --online-ok.")
    return None


def build_parts(settings: Settings, software: Path = SOFTWARE, screen=None, backend=None) -> Parts:
    """The real parts for a session. Raises BrainError("no_key") when a cloud brain has no key yet."""
    s = settings
    card = load_card(software / "controls" / f"{s.controls_card}.yaml")
    key = load_api_key(s.brain) if s.brain in KEY_NAMES else None
    brain = make_brain(s.brain, s.brain_model(), key, card=card, language=s.language)
    screen = screen or Screen()
    hands = Hands(card, backend=backend, in_front=lambda: screen.in_front(s.game_window),
                  get_box=lambda: screen.find(s.game_window))
    safety = Safety(s.max_minutes, s.max_ai_calls)
    safety.add_stop_listener(hands.on_stop)  # F12 from any thread releases the keys at once
    return Parts(brain=brain, card=card, screen=screen, hands=hands, safety=safety, memory=Memory(s.memory_steps))


def _data_folder() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "PlayFellow"


def confirm_adult(ask: Callable[[str], str] = input, store: Path | None = None) -> bool:
    """PlayFellow is for adults. The first run asks once; a yes is remembered (date only), anything else is a no."""
    store = store or _data_folder() / AGE_FILE
    if store.is_file():
        return True
    answer = ask(tr("PlayFellow is for adults (18+). Are you 18 or older? Type yes to continue: ",
                    "بلاي فيلو للبالغين (18+). هل عمرك 18 سنة أو أكثر؟ اكتب yes للمتابعة: "))
    if answer.strip().lower() not in ("yes", "y", "نعم"):
        return False
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(f"confirmed 18+ on {datetime.now(timezone.utc).date().isoformat()}\n", encoding="utf-8")
    return True


def _printer():
    last = {"status": None}

    def on_event(kind: str, data: Any) -> None:
        if kind == "status":
            if data == last["status"]:
                return  # the same "waiting for the window" line once, not every second
            last["status"] = data
        elif kind == "actions":
            data = ", ".join(data) or tr("waits", "ينتظر")
        elif kind == "dropped":
            data = tr("not on the card: ", "ليست على البطاقة: ") + ", ".join(data)
        elif kind == "thought" and not data:
            return
        print(f"{time.strftime('%H:%M:%S')} {kind}: {data}", flush=True)

    return on_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PlayFellow: an AI that plays your game with you (research core)")
    parser.add_argument("--settings", type=Path, default=SOFTWARE / "settings.yaml")
    parser.add_argument("--brain", help="builtin, gemini, claude, openai or local")
    parser.add_argument("--model", help="a model name for that brain (default: settings.py MODEL_TIERS)")
    parser.add_argument("--window", help="the game window's title, or part of it")
    parser.add_argument("--card", help="the controls card (a file name in controls/, without .yaml)")
    parser.add_argument("--goal", help="what the buddy should do")
    parser.add_argument("--minutes", type=int, help="stop after this many minutes")
    parser.add_argument("--online-ok", action="store_true", help="play a game marked online/anti-cheat anyway")
    parser.add_argument("--save-key", metavar="BRAIN", help="store your key for gemini, claude or openai, then exit")
    args = parser.parse_args(argv)

    if not confirm_adult():
        print(tr("PlayFellow is only for adults (18+).", "بلاي فيلو للبالغين فقط (18+)."))
        return 2
    if args.save_key:
        save_api_key(args.save_key, getpass.getpass(f"{KEY_NAMES.get(args.save_key, args.save_key)}: "))
        print(tr("Saved in Windows Credential Manager.", "تم الحفظ في مدير بيانات اعتماد Windows."))
        return 0

    settings = load_settings(args.settings)
    overrides = {"brain": args.brain, "model": args.model, "game_window": args.window, "controls_card": args.card,
                 "goal": args.goal, "max_minutes": args.minutes}
    settings = dataclasses.replace(settings, **{k: v for k, v in overrides.items() if v is not None})
    if args.brain and not args.model:
        settings = dataclasses.replace(settings, model="")  # a model name belongs to one brain
    problem = check_window(settings.game_window, args.online_ok)
    if problem:
        print(problem)
        return 2

    make_dpi_aware()
    try:
        parts = build_parts(settings)
    except BrainError as error:
        print(error.friendly())
        return 2
    listen_for_f12(lambda: parts.safety.stop("stopped_f12"))
    print(tr(f"Playing '{settings.game_window}' with the {settings.brain} brain. F12 or the mouse in the top-left "
             "corner stops at once.",
             f"ألعب '{settings.game_window}' بعقل {settings.brain}. F12 أو الفأرة في الزاوية العلوية اليسرى يوقف فورًا."),
          flush=True)
    reason = play(parts, settings, on_event=_printer())
    print(msg(reason, minutes=settings.max_minutes, calls=settings.max_ai_calls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
