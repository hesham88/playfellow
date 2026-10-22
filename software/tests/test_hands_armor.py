"""Unit tests verifying AI Armor action guardrails in hands.py. / اختبارات حماية الأيدي بدرع الذكاء الاصطناعي."""
from __future__ import annotations

import pytest
from controls import ControlAction, ControlsCard
from eyes import WindowBox
from reply import Step
from hands import Hands

BOX = WindowBox(100, 50, 801, 601)


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def keyDown(self, key):
        self.calls.append(("down", key))

    def keyUp(self, key):
        self.calls.append(("up", key))

    def click(self, x, y, button="left"):
        self.calls.append(("click", x, y, button))


def test_hands_blocks_alt_f4_and_allows_safe_keys():
    card = ControlsCard(
        name="test_card",
        game_hint="test",
        actions={
            "SAFE_JUMP": ControlAction(name="SAFE_JUMP", kind="keys", keys=("space",), hold=0.1),
            "HAZARDOUS_QUIT": ControlAction(name="HAZARDOUS_QUIT", kind="keys", keys=("alt+f4",), hold=0.1),
            "INJECTION": ControlAction(name="INJECTION", kind="keys", keys=("powershell.exe -c calc",), hold=0.1),
        },
    )
    backend = RecordingBackend()
    hands = Hands(card, backend=backend, in_front=lambda b: True, get_box=lambda: BOX)

    # 1. Safe action passes through
    hands.do(Step("SAFE_JUMP"), BOX)
    assert backend.calls == [("down", "space"), ("up", "space")]
    backend.calls.clear()

    # 2. Hazardous Alt+F4 is blocked by AI Armor
    hands.do(Step("HAZARDOUS_QUIT"), BOX)
    assert backend.calls == []

    # 3. Command injection is blocked by AI Armor
    hands.do(Step("INJECTION"), BOX)
    assert backend.calls == []
