"""Pretend parts for tests: no real keys, screens or AI."""


class FailSafeException(Exception):
    """Same name as pydirectinput's corner-stop error."""


class FakeBackend:
    """Records what would have been pressed. After `corner_after` calls the mouse is 'in the corner'."""

    def __init__(self, corner_after=None):
        self.FAILSAFE = True
        self.calls = []
        self.corner_after = corner_after

    def _check(self):
        if self.FAILSAFE and self.corner_after is not None and len(self.calls) >= self.corner_after:
            raise FailSafeException("mouse in corner")

    def keyDown(self, key):
        self._check()
        self.calls.append(("down", key))

    def keyUp(self, key):
        self._check()
        self.calls.append(("up", key))

    def click(self, x=None, y=None, button="left"):
        self._check()
        self.calls.append(("click", x, y, button))

    def moveRel(self, xOffset=None, yOffset=None, relative=False):
        self._check()
        self.calls.append(("move", xOffset, yOffset, relative))


class ScriptedBrain:
    """A pretend brain that replays prepared answers (or raises prepared errors)."""

    def __init__(self, replies, is_helper=False):
        self.replies = replies
        self.calls = []
        self.is_helper = is_helper

    def ask(self, system, text, images):
        self.calls.append({"system": system, "text": text, "images": len(images)})
        if callable(self.replies):
            return self.replies(system, text, images)
        item = self.replies[min(len(self.calls), len(self.replies)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


from PIL import Image as _Image

from eyes import WindowBox as _WindowBox
from settings import Settings as _Settings


class FakeScreen:
    def __init__(self, image=None, box=_WindowBox(0, 0, 800, 600), in_front=True, active_title="PlayFellow Practice Arena"):
        self.image = image or _Image.new("RGB", (800, 600), "white")
        self.box = box
        self.front = in_front
        self.grabs = 0
        self._active_title = active_title
        self.bring_to_front_calls = []

    def list_windows(self):
        return ["PlayFellow Practice Arena"]

    def find(self, title):
        return self.box

    def in_front(self, title):
        return self.front

    def active_title(self):
        return self._active_title

    def grab(self, box):
        self.grabs += 1
        return self.image

    def bring_to_front(self, title):
        self.bring_to_front_calls.append(title)
        self.front = True


def quick_settings(**overrides):
    values = {
        "look_every_seconds": 0.0,
        "max_actions_per_step": 3,
        "memory_steps": 2,
        "max_ai_calls": 100,
    }
    values.update(overrides)
    return _Settings(**values)


class MemoryKeystore(dict):
    """Stand-in for Windows Credential Manager (keyring) in the sync tests (it lived in sync/engine.py until 2026-10-23)."""

    def set(self, name, value):
        self[name] = value
