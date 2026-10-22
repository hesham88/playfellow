"""The built-in brain: plays with no AI account, no internet and no cost. / العقل المدمج: يلعب بلا حساب ولا إنترنت.

Like a puppy learning a new room: it presses a few keys, notices which thing on the screen moves when it does (that
is "me"), remembers where the walls are, what is worth collecting and where the door is, and then walks there. It is
the same learning that the rest of PlayFellow uses between thoughts (software/nav/), put behind the same `ask(...)`
doorway as Gemini, Claude, OpenAI and Ollama, so a player with no AI account can still play from the first day.

It answers in the usual JSON ({thought, say, actions}) using ONLY the actions on the controls card. Its thoughts are
in one language, English or Arabic, never both.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from brain.slots import SlotRegistry
from controls import ControlsCard
from nav.hud_bar import HudBar
from nav.planner import Navigator

STUCK_AFTER = 12  # the same picture this many times in a row, while I pressed keys: say I am stuck
# The loop uses the same brain for things that are not "make a move". These two are told apart by their own words
# (a test checks they still appear in loop.py), so such a call neither moves nor teaches the map anything.
HELP_MARK = "wants to ask the player for help"  # loop.HELP_SYSTEM: the engine words its own question when we say nothing
WATCH_MARK = "Send no actions"  # loop._say_what_i_saw: the player just played, these are THEIR pictures, not my steps
STOP_WORDS = {"english": ("stop", "wait", "hold on", "pause"), "arabic": ("قف", "توقف", "انتظر")}
EXPLORE_WORDS = {"english": ("explore", "look around"), "arabic": ("استكشف", "تجول")}

WORDS = {
    "english": {
        "waiting": "I am waiting for a picture of the game.",
        "probe": "I am testing my keys to find out which thing on the screen is me!",
        "reward": "I can see something to collect, so I am heading for it.",
        "goal": "I have what I need, so I am walking to the way out.",
        "explore": "I do not know this place yet, so I am looking around.",
        "avoid": "That looks dangerous, so I am taking a safer step.",
        "steer": "I keep moving by myself here, so I am turning before I hit something.",
        "wait": "I am staying still for a moment to see if I keep moving by myself.",
        "none": "I am thinking about where to go next.",
        "stopped": "Okay, I am stopping and waiting for you.",
        "found_me": "Found it! I am the one that moves when I press the keys.",
        "going_goal": "I think I can finish now, going to the exit!",
        "stuck": "I seem to be stuck. I will try something different.",
        "watched": "I was watching you play.",
        "watched_say": "I watched you play - got it, I will remember!",
    },
    "arabic": {
        "waiting": "أنتظر صورة من اللعبة.",
        "probe": "أجرب الأزرار لأعرف أي شيء على الشاشة هو أنا!",
        "reward": "أرى شيئًا يمكن جمعه، فأتجه نحوه.",
        "goal": "معي كل ما أحتاجه، فأمشي إلى باب الخروج.",
        "explore": "لا أعرف هذا المكان بعد، فأتجول لأرى.",
        "avoid": "هذا يبدو خطيرًا، فأخطو خطوة أكثر أمانًا.",
        "steer": "أستمر بالحركة وحدي هنا، فأستدير قبل أن أصطدم بشيء.",
        "wait": "أبقى ساكنًا لحظة لأرى هل أستمر بالحركة وحدي.",
        "none": "أفكر إلى أين أذهب بعد ذلك.",
        "stopped": "حسنًا، أتوقف وأنتظرك.",
        "found_me": "وجدتها! أنا الشخصية التي تتحرك عندما أضغط الأزرار.",
        "going_goal": "أظن أنني أستطيع الإنهاء الآن، أتجه إلى المخرج!",
        "stuck": "يبدو أنني عالق. سأجرب شيئًا مختلفًا.",
        "watched": "كنت أشاهدك تلعب.",
        "watched_say": "شاهدتك تلعب - فهمت، سأتذكر ذلك!",
    },
}


class BuiltinNavBrain:
    """Plays any game that moves with arrow-like keys, by learning it on the spot. No network is ever touched."""

    name = "builtin"
    model = "navigator"  # what the loop shows as the brain's model: "builtin (navigator)"
    last_used = "primary"

    def __init__(self, card: ControlsCard, language: str = "english", max_actions: int = 1):
        if language not in WORDS:
            raise ValueError(f"unknown language {language!r}")
        self.card = card
        self.language = language
        self.max_actions = max(1, max_actions)  # kept for the shared interface; one step at a time is what it learns from
        self.keys = sorted(name for name, action in card.actions.items() if action.kind == "keys")
        self.waits = [name for name, action in card.actions.items() if action.kind == "wait"]
        self.navigator = Navigator(self.keys)
        self.slots = SlotRegistry()  # learned parts for S1-S4; every socket is OFF until it passes its gate and is switched on
        self.navigator.planner.slots = self.slots
        self._prev: np.ndarray | None = None
        self._last: str | None = None
        self._same = 0  # pictures in a row that looked just like the one before
        self._said: set[str] = set()
        self._kind = ""
        self.hud_bar = HudBar()  # "did something happen?" from the top bar; belongs to whoever feeds the navigator
        # When the game loop's fast tick feeds the navigator (async play), it is the ONLY learner: this brain then
        # just reads the map and advises. Two learners on one navigator see pictures from two different moments
        # (before and after the key) and corrupt what it knows about "me".
        self.external_learner = False

    def use_navigator(self, navigator: Navigator, hud_bar: HudBar) -> None:
        """Hand over to the game loop's navigator and top-bar reader: from now on the loop feeds them and this brain
        only advises."""
        self.navigator, self.hud_bar, self.external_learner = navigator, hud_bar, True
        self.navigator.planner.slots = self.slots
        self._prev, self._last = None, None

    # ---- the shared brain doorway
    def ask(self, system: str, text: str, images: list[Image.Image]) -> str:
        words = WORDS[self.language]
        if HELP_MARK in system:
            return ""  # "ask the player for help": an empty answer makes the engine write its own question, in the right language
        if WATCH_MARK in text:
            return self._reply(words["watched"], words["watched_say"], [])  # words only; the map must not learn from THEIR play
        if not images or not self.keys:
            return self._reply(words["waiting"], "", [])
        frame = np.asarray(images[-1].convert("RGB"))
        if not self.external_learner:
            self._learn(frame)
        command = self._command(text)
        if command == "stop":
            self._last = None
            return self._reply(words["stopped"], "", self.waits[:1])
        advice = self.navigator.advise(frame, "explore" if command == "explore" else "all", 1)
        chosen = [a for a in advice.actions if a in self.keys][:1]
        key = self.slots.decide("S3_policy", chosen[0] if chosen else None, (frame, list(self.keys)), valid=lambda k: k in self.keys)
        chosen = [key] if key else []
        self._last = chosen[0] if chosen else None
        said = self._say(advice.kind, frame)
        return self._reply(words.get(advice.kind, words["none"]), said, chosen)

    # ---- learning from what I see
    def _learn(self, frame: np.ndarray) -> None:
        bar = self._bar(frame)
        if self._prev is None or self._prev.shape != frame.shape:
            self.navigator.start(frame, bar)
            self._same = 0
        else:
            self._same = self._same + 1 if np.array_equal(self._prev, frame) and self._last else 0
            self.navigator.learn(self._prev, self._last, frame, bar)
        self._prev = frame

    def _bar(self, frame: np.ndarray) -> dict:
        rule = self.hud_bar.read(frame)  # always read: the bar remembers the previous picture
        return self.slots.decide("S2_hud", rule, (frame,), valid=lambda d: isinstance(d, dict) and isinstance(d.get("bar"), int))

    @property
    def _bar_events(self) -> int:
        return self.hud_bar.events

    # ---- the player talking to me
    def _command(self, text: str) -> str:
        """'stop' or 'explore' if the player said so (the chat lines are the '- ' lines under 'The player says')."""
        if "The player says" not in text:
            return ""
        said = " ".join(line[2:].lower() for line in text.split("The player says", 1)[1].splitlines()
                        if line.startswith("- "))
        if any(word in said for word in STOP_WORDS[self.language]):
            return "stop"
        if any(word in said for word in EXPLORE_WORDS[self.language]):
            return "explore"
        return ""

    # ---- saying things (sparse: only when it is worth a sentence)
    def _say(self, kind: str, frame: np.ndarray) -> str:
        words = WORDS[self.language]
        said = ""
        if self.navigator.nav.me.known and "found_me" not in self._said:
            self._said.add("found_me")
            said = words["found_me"]
        elif kind == "goal" and self._kind != "goal":
            said = words["going_goal"]
        elif self._same >= STUCK_AFTER:
            self._same = 0
            said = words["stuck"]
        self._kind = kind
        return said

    def _reply(self, thought: str, say: str, actions: list[str]) -> str:
        return json.dumps({"thought": thought, "say": say, "actions": actions, "mission_update": {"type": "none"},
                           "hud": {}, "facts": []}, ensure_ascii=False)
