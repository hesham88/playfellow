"""The public play loop: look, mark, think, check, act, with every safety rule (no real keys, screen or AI here)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import play
from brain import BrainError
from controls import load_card
from fakes import FakeBackend, FakeScreen, ScriptedBrain, quick_settings
from grounding import Mark
from hands import Hands
from memory import Memory
from safety import Safety

SOFTWARE = Path(__file__).resolve().parents[1]
CARD = load_card(SOFTWARE / "controls" / "practice_arena.yaml")


def answer(*actions, thought="t", say=""):
    return json.dumps({"thought": thought, "say": say, "actions": list(actions)})


def parts(replies, screen=None, backend=None, marker=None, **settings):
    s = quick_settings(**settings)
    screen = screen or FakeScreen()
    backend = backend or FakeBackend()
    hands = Hands(CARD, backend=backend, sleep=lambda seconds: None,
                  in_front=lambda: screen.in_front(s.game_window), get_box=lambda: screen.find(s.game_window))
    return play.Parts(brain=ScriptedBrain(replies), card=CARD, screen=screen, hands=hands,
                      safety=Safety(s.max_minutes, s.max_ai_calls), memory=Memory(s.memory_steps), marker=marker), s


def pressed(backend):
    return [key for kind, key, *rest in backend.calls if kind == "down"]


def test_a_turn_presses_only_the_card_actions_the_brain_chose():
    p, s = parts([answer("UP", "FLY", "LEFT")])
    reply = play.take_turn(p, s, play.system_for(p, s))
    assert len(reply.dropped) == 1 and reply.dropped[0].startswith("FLY")
    assert pressed(p.hands.backend) == ["up", "left"]


def test_nothing_is_asked_or_pressed_while_the_game_is_not_in_front():
    p, s = parts([answer("UP")], screen=FakeScreen(in_front=False))
    events = []
    assert play.take_turn(p, s, "system", on_event=lambda kind, data: events.append(kind)) is None
    assert p.brain.calls == [] and p.hands.backend.calls == []
    assert "status" in events


def test_a_stop_in_the_middle_of_a_turn_presses_nothing_more():
    backend = FakeBackend()
    p, s = parts([answer("UP", "DOWN", "LEFT")], backend=backend)
    original = backend.keyDown

    def key_down(key):
        original(key)
        p.safety.stop("stopped_f12")  # F12 arrives while the first key is down

    backend.keyDown = key_down
    play.take_turn(p, s, "system")
    assert pressed(backend) == ["up"]
    assert ("up", "up") in backend.calls  # and the key that was down went up again


def test_the_corner_stop_ends_the_session_at_once():
    p, s = parts([answer("UP")], backend=FakeBackend(corner_after=0))
    assert play.play(p, s) == "stopped_corner"


def test_the_session_ends_at_the_ai_call_limit():
    p, s = parts([answer("WAIT")], max_ai_calls=2)
    assert play.play(p, s) == "stopped_calls"
    assert len(p.brain.calls) == 2


def test_the_brain_sees_its_recent_moments_and_the_newest_picture_last():
    p, s = parts([answer("UP", thought="first"), answer("DOWN", thought="second")], memory_steps=2)
    play.play(p, s, max_turns=2)
    second = p.brain.calls[1]
    assert second["images"] == 2 and "you thought: first" in second["text"]


def test_marks_are_drawn_on_the_picture_and_explained_to_the_brain():
    seen = []

    def brain(system, text, images):
        seen.append((text, images[-1]))
        return answer("WAIT")

    marks = [Mark("me", 500, 500), Mark("reward", 700, 500)]
    plain, _ = parts(brain)
    play.take_turn(plain, quick_settings(), "system")
    marked, s = parts(brain, marker=lambda image: marks)
    play.take_turn(marked, s, "system")
    (plain_text, plain_image), (marked_text, marked_image) = seen
    assert "R1" in marked_text and "R1" not in plain_text
    assert marked_image.tobytes() != plain_image.tobytes()


def test_a_brain_that_keeps_failing_stops_the_session():
    p, s = parts([BrainError("busy", retry=True, wait_seconds=0)])
    assert play.play(p, s) == "too_many_errors"
    assert len(p.brain.calls) == play.MAX_ERRORS_IN_ROW


def test_a_brain_error_that_cannot_be_retried_stops_at_once():
    p, s = parts([BrainError("bad_key")])
    assert play.play(p, s) == "bad_key"
    assert len(p.brain.calls) == 1


def test_online_games_need_an_explicit_ok():
    assert play.check_window("PlayFellow Practice Arena", online_ok=False) is None
    assert "anti-cheat" in play.check_window("PUBG: BATTLEGROUNDS", online_ok=False)
    assert play.check_window("PUBG: BATTLEGROUNDS", online_ok=True) is None


def test_the_built_in_brain_needs_no_key_and_a_cloud_brain_does(monkeypatch):
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    built = play.build_parts(quick_settings(brain="builtin"), SOFTWARE, screen=FakeScreen(), backend=FakeBackend())
    assert built.brain.ask("system", "The goal: x", [Image.new("RGB", (64, 64))])  # answers with no network
    with pytest.raises(BrainError) as error:
        play.build_parts(quick_settings(brain="gemini"), SOFTWARE, screen=FakeScreen(), backend=FakeBackend())
    assert error.value.key == "no_key"


def test_the_first_run_asks_for_18_plus_and_remembers_the_yes(tmp_path):
    store = tmp_path / "age.txt"
    assert play.confirm_adult(ask=lambda prompt: "yes", store=store) is True

    def never(prompt):
        raise AssertionError("asked again after a yes")

    assert play.confirm_adult(ask=never, store=store) is True


def test_anything_but_yes_means_no_play(tmp_path):
    store = tmp_path / "age.txt"
    assert play.confirm_adult(ask=lambda prompt: "no", store=store) is False
    assert play.confirm_adult(ask=lambda prompt: "", store=store) is False
    assert not store.exists()


class ArenaScreen:
    """The real Practice Arena game behind a pretend screen: grab() is the game's own picture."""

    def __init__(self, game):
        from eyes import WindowBox
        self.game, self.box = game, WindowBox(0, 0, *game.pixel_size)

    def find(self, title):
        return self.box

    def in_front(self, title):
        return True

    def grab(self, box):
        return self.game.render()


class ArenaKeys(FakeBackend):
    """Keys go straight into the game, one square per press, like the real window."""

    def __init__(self, game):
        super().__init__()
        self.game = game

    def keyDown(self, key):
        super().keyDown(key)
        self.game.press(key)


def test_the_built_in_brain_wins_the_practice_arena_through_the_real_loop():
    """End to end with no network: real pictures of the real game, the real built-in brain, the card check and the
    hands. It must win, and nothing it presses may be off the card."""
    from practice_arena.game import ArenaGame

    game = ArenaGame()
    s = quick_settings(brain="builtin", max_ai_calls=1000)
    p = play.build_parts(s, SOFTWARE, screen=ArenaScreen(game), backend=ArenaKeys(game))
    p.hands.sleep = lambda seconds: None
    p.hands.resume()
    system = play.system_for(p, s)
    for _ in range(400):
        reply = play.take_turn(p, s, system)
        assert reply is not None and reply.dropped == []
        if game.won:
            break
    assert game.won, f"score {game.score}, coins left {game.coins_left}, key {game.has_key}"
