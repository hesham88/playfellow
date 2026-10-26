"""Safety rules: time limit, brain-question limit and emergency stops. / قواعد الأمان: حد الوقت وحد أسئلة العقل وإيقاف الطوارئ."""
from __future__ import annotations

import threading
import time
from typing import Callable


class Safety:
    def __init__(self, max_minutes: int, max_ai_calls: int, clock=time.monotonic):
        self.max_seconds = max_minutes * 60
        self.max_ai_calls = max_ai_calls
        self.clock = clock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.reason: str | None = None
        self.calls = 0
        self._started = clock()
        self._stop_listeners: list = []  # run at once on stop(), from whatever thread pressed F12

    def start(self) -> None:
        with self._lock:
            self._stop.clear()
            self.reason = None
            self.calls = 0
            self._started = self.clock()

    def add_stop_listener(self, listener) -> None:
        """Call listener() the moment anything stops play (F12, corner, limits, STOP)."""
        self._stop_listeners.append(listener)

    def stop(self, reason: str) -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
            self._stop.set()
        for listener in list(self._stop_listeners):
            try:
                listener()
            except Exception:
                pass  # releasing keys must never raise out of the F12 handler

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def count_ai_call(self) -> None:
        with self._lock:
            self.calls += 1

    def elapsed(self) -> float:
        """Seconds since start(): the round's clock, the one max_minutes is checked against."""
        return self.clock() - self._started

    def check(self) -> str | None:
        if not self.stopped:
            if self.clock() - self._started >= self.max_seconds:
                self.stop("stopped_time")
            elif self.calls >= self.max_ai_calls:
                self.stop("stopped_calls")
        return self.reason

    def wait(self, seconds: float) -> bool:
        """Sleep, but wake up at once if someone presses Stop."""
        return self._stop.wait(max(0.0, seconds))


def listen_for_key(key_name: str, callback: Callable[[], None]):
    """Watch the keyboard everywhere for one special key (e.g. "f12" stop, "f9" take-over) and call back.
    Only that key is looked at; every other key press is ignored on the spot and never kept."""
    from pynput import keyboard

    target = getattr(keyboard.Key, key_name.lower())

    def on_press(key):
        if key == target:
            callback()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def listen_for_f12(on_f12: Callable[[], None]):
    """Watch the keyboard everywhere: F12 means emergency stop."""
    return listen_for_key("f12", on_f12)
