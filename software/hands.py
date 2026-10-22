"""PlayFellow's hands: press keys and move the mouse, like a person. / يدا بلاي فيلو: تضغطان الأزرار وتحركان الفأرة مثل الإنسان."""
from __future__ import annotations

from ai_armor import armor

import threading
from contextlib import contextmanager

import time

from controls import ControlAction, ControlsCard
from eyes import GRID, WindowBox
from reply import Step


class CornerStop(Exception):
    """The mouse went to the top-left corner: emergency stop."""


def grid_to_screen(x: int, y: int, box: WindowBox) -> tuple[int, int]:
    """Turn a 0-1000 grid spot into a real screen pixel inside the window (out-of-range spots are clamped to its edge)."""
    x, y = min(max(x, 0), GRID), min(max(y, 0), GRID)
    return (
        box.left + round(x / GRID * (box.width - 1)),
        box.top + round(y / GRID * (box.height - 1)),
    )


class Hands:
    def __init__(self, card: ControlsCard, backend=None, sleep=time.sleep, in_front=None, get_box=None):
        if backend is None:
            import pydirectinput

            pydirectinput.FAILSAFE = True  # mouse to the top-left corner = stop
            pydirectinput.PAUSE = 0.02
            backend = pydirectinput
        self.card = card
        self.backend = backend
        self.sleep = sleep
        self._held: list[str] = []  # Temporary holds within a single action
        self._persistent_holds: dict[str, int] = {}  # Keys held across steps, with watchdog counter
        self._lock = threading.RLock()  # F12 releases keys from another thread
        self._stopped = False  # after a stop, no new key goes down until resume()
        self._mouse_down: str | None = None  # the mouse button a swipe / drag is holding right now
        # A gesture holds the mouse button down for a while, so it needs to look at the world again at every move:
        # in_front() says whether the game window is still the one in front, get_box() where the window is right now.
        # Without both, no gesture is ever started.
        self._in_front = in_front
        self._get_box = get_box
        self._failsafe_depth = 0  # how many releases are running at once; the corner check comes back at 0
        self._failsafe_saved = None

    def do(self, step: Step, box: WindowBox) -> None:
        action = self.card.actions[step.action]
        try:
            self._do(action, step, box)
        except Exception as exc:
            if type(exc).__name__ == "FailSafeException":
                self.release_all()
                self.release_persistent_holds()
                raise CornerStop() from exc
            raise

    def _do(self, action: ControlAction, step: Step, box: WindowBox) -> None:
        if action.kind == "keys":
            try:
                with self._lock:
                    if self._stopped:
                        return
                    for key in action.keys:
                        safe, _ = armor.validate_action(key)
                        if not safe:
                            continue
                        self.backend.keyDown(key)
                        self._held.append(key)
                self.sleep(action.hold)
            finally:
                self.release_all()
        elif action.kind == "click":
            x, y = grid_to_screen(step.x, step.y, box)
            with self._lock:
                if not self._stopped:  # F12 may land between the loop's check and here: no click after a stop
                    self.backend.click(x=x, y=y, button=action.button)
        elif action.kind == "swipe":
            self._gesture(box, action.x1, action.y1, action.x2, action.y2, action.ms, action.button)
        elif action.kind == "drag":
            self._gesture(box, step.x, step.y, step.x2, step.y2, action.ms, action.button)
        elif action.kind == "mouse_button":
            with self._lock:
                if not self._stopped:
                    self._pointer_into(box)
                    self.backend.click(button=action.button)
        elif action.kind == "mouse_move":
            with self._lock:
                if not self._stopped:
                    self.backend.moveRel(action.dx, action.dy, relative=True)
        elif action.kind == "wait":
            self.sleep(action.hold)
        elif action.kind == "hold_start":
            self._start_hold(action.hold_key)
        elif action.kind == "hold_stop":
            self._stop_hold(action.hold_key)
        elif action.kind == "release_all":
            self.release_persistent_holds()

    def _pointer_into(self, box: WindowBox) -> None:
        """A bare mouse press lands wherever the pointer is, so make sure the pointer is inside the game window first.
        If it already is, it is left alone (a game that holds the pointer in place must not be jolted)."""
        position = getattr(self.backend, "position", None)
        try:
            x, y = position() if callable(position) else (None, None)
        except Exception:
            x = y = None
        inside = x is not None and box.left <= x < box.left + box.width and box.top <= y < box.top + box.height
        move = getattr(self.backend, "moveTo", None)
        if not inside and callable(move):
            move(box.left + box.width // 2, box.top + box.height // 2)

    def set_box_source(self, get_box) -> None:
        """Tell the hands how to find the window's playable area right now (the engine may cut black bars off a stream)."""
        self._get_box = get_box

    @contextmanager
    def _corner_check_paused(self):
        """Pause the corner check while keys are let go, so a release can never be cut short by it.

        Several threads may release at once (F12, a lost focus, the main loop). All the pausing and restoring
        happens under the lock with a counter, and the check comes back only when the last release is done.
        """
        with self._lock:
            self._failsafe_depth += 1
            if self._failsafe_depth == 1:
                self._failsafe_saved = getattr(self.backend, "FAILSAFE", None)
                if self._failsafe_saved is not None:
                    self.backend.FAILSAFE = False
        try:
            yield
        finally:
            with self._lock:
                self._failsafe_depth -= 1
                if self._failsafe_depth == 0 and self._failsafe_saved is not None:
                    self.backend.FAILSAFE = self._failsafe_saved
                    self._failsafe_saved = None

    def _gesture_ok(self, box: WindowBox) -> bool:
        """Still safe to hold the mouse: the game is in front and its window has not moved or changed size."""
        if self._in_front is None or self._get_box is None:
            return False
        try:
            return bool(self._in_front()) and self._get_box() == box
        except Exception:
            return False

    def _gesture(self, box: WindowBox, x1: int, y1: int, x2: int, y2: int, ms: int, button: str = "left") -> None:
        """Press at one spot, glide to another and let go, like a finger on a screen or a mouse drag.

        Both spots are on the 0-1000 grid and are turned into pixels inside the game window. Before the press and
        before EVERY move the hands check that the game is still in front and the window is exactly where it was
        (a pop-up or a moved window ends the gesture on the spot). A stop (F12), a lost focus or the corner check
        also let go at once, even mid-glide. The stop check and the move happen under one lock, so nothing moves
        after F12 has let go.
        """
        start, end = grid_to_screen(x1, y1, box), grid_to_screen(x2, y2, box)
        steps = max(2, min(40, ms // 20))
        with self._lock:
            if self._stopped or not self._gesture_ok(box):
                return
            self.backend.moveTo(*start)
            self.backend.mouseDown(button=button)
            self._mouse_down = button
        try:
            for n in range(1, steps + 1):
                with self._lock:
                    if self._stopped or self._mouse_down is None or not self._gesture_ok(box):
                        break  # F12, a lost focus, a pop-up or a moved window: let go now
                    self.backend.moveTo(round(start[0] + (end[0] - start[0]) * n / steps),
                                        round(start[1] + (end[1] - start[1]) * n / steps))
                self.sleep(ms / 1000 / steps)
        finally:
            self._release_mouse()

    def _release_mouse(self) -> None:
        """Let go of a mouse button a gesture is holding. Safe to call twice and from any thread."""
        with self._lock:
            button, self._mouse_down = self._mouse_down, None
            if button is None:
                return
            with self._corner_check_paused():
                self.backend.mouseUp(button=button)

    def release_all(self) -> None:
        """Let go of every key and any held mouse button. The corner check is paused so keys can't get stuck down."""
        self._release_mouse()
        with self._corner_check_paused():
            with self._lock:
                while self._held:
                    self.backend.keyUp(self._held.pop())

    def _start_hold(self, key: str) -> None:
        """Start holding a key across steps."""
        if self._stopped:
            return  # after a stop, nothing new goes down until resume()
        # Release conflicting directions (left vs right)
        if key in ("left", "right"):
            for conflict in ("left", "right"):
                if conflict != key and conflict in self._persistent_holds:
                    self._stop_hold(conflict)

        # Check AI Armor before holding key
        safe, _ = armor.validate_action(key)
        if not safe:
            return

        # If not already held, press it down
        if key not in self._persistent_holds:
            self.backend.keyDown(key)

        # Reset watchdog counter to 0
        self._persistent_holds[key] = 0

    def _stop_hold(self, key: str) -> None:
        """Stop holding a specific key."""
        with self._lock:
            if key in self._persistent_holds:
                with self._corner_check_paused():
                    self.backend.keyUp(key)
                del self._persistent_holds[key]

    def release_persistent_holds(self) -> None:
        """Release all persistently held keys."""
        with self._corner_check_paused():
            with self._lock:
                for key in list(self._persistent_holds.keys()):
                    self.backend.keyUp(key)
                self._persistent_holds.clear()

    def tick_watchdog(self) -> None:
        """Increment watchdog counter and release keys not confirmed for 3 steps."""
        to_release = []
        for key, count in self._persistent_holds.items():
            if count >= 3:
                to_release.append(key)
            else:
                self._persistent_holds[key] = count + 1

        for key in to_release:
            self._stop_hold(key)

    def on_focus_lost(self) -> None:
        """Release all held keys when focus is lost."""
        self.release_all()
        self.release_persistent_holds()

    def on_stop(self) -> None:
        """Release all held keys on any stop condition, at once, and press nothing new until resume()."""
        with self._lock:
            self._stopped = True
        self.release_all()
        self.release_persistent_holds()

    def resume(self) -> None:
        """Allow key presses again (a new run started)."""
        with self._lock:
            self._stopped = False
