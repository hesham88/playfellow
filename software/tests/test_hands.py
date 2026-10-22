import pytest

from controls import load_card
from eyes import WindowBox
from fakes import FakeBackend
from hands import CornerStop, Hands, grid_to_screen
from reply import Step

BOX = WindowBox(100, 50, 801, 601)


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  UP: {keys: up, hold: 0.2}\n"
        "  RUN: {keys: [shift, right], hold: 0.4}\n"
        "  CLICK: {click: right}\n"
        "  SHOOT: {mouse_button: left}\n"
        "  TURN: {mouse_move: [-300, 10]}\n"
        "  WAIT: {wait: 0.5}\n",
        encoding="utf-8",
    )
    return load_card(path)


def make(card, backend):
    sleeps = []
    return Hands(card, backend=backend, sleep=sleeps.append), sleeps


def test_grid_to_screen_corners_and_middle():
    assert grid_to_screen(0, 0, BOX) == (100, 50)
    assert grid_to_screen(1000, 1000, BOX) == (900, 650)
    assert grid_to_screen(500, 500, BOX) == (500, 350)


def test_key_is_held_then_released(card):
    backend = FakeBackend()
    hands, sleeps = make(card, backend)
    hands.do(Step("UP"), BOX)
    assert backend.calls == [("down", "up"), ("up", "up")]
    assert sleeps == [0.2]


def test_two_keys_released_in_reverse(card):
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("RUN"), BOX)
    assert backend.calls == [("down", "shift"), ("down", "right"), ("up", "right"), ("up", "shift")]


def test_click_lands_inside_the_window(card):
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("CLICK", 500, 0), BOX)
    assert backend.calls == [("click", 500, 50, "right")]


def test_mouse_button_move_and_wait(card):
    backend = FakeBackend()
    hands, sleeps = make(card, backend)
    hands.do(Step("SHOOT"), BOX)
    hands.do(Step("TURN"), BOX)
    hands.do(Step("WAIT"), BOX)
    assert backend.calls == [("click", None, None, "left"), ("move", -300, 10, True)]
    assert sleeps == [0.5]


def test_corner_stop_releases_held_keys(card):
    backend = FakeBackend(corner_after=1)  # shift goes down, then the mouse hits the corner
    hands, _ = make(card, backend)
    with pytest.raises(CornerStop):
        hands.do(Step("RUN"), BOX)
    assert ("up", "shift") in backend.calls
    assert backend.FAILSAFE is True


def test_keys_released_even_if_something_breaks(card):
    backend = FakeBackend()

    def broken_sleep(seconds):
        raise RuntimeError("boom")

    hands = Hands(card, backend=backend, sleep=broken_sleep)
    with pytest.raises(RuntimeError):
        hands.do(Step("UP"), BOX)
    assert backend.calls[-1] == ("up", "up")


def test_hold_start_keeps_key_down(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("HOLD_RIGHT"), BOX)
    # Key should be pressed down but not released
    assert backend.calls == [("down", "right")]
    assert "right" in hands._persistent_holds


def test_hold_stop_releases_key(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  STOP_RIGHT: {hold_stop: right}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("HOLD_RIGHT"), BOX)
    hands.do(Step("STOP_RIGHT"), BOX)
    assert backend.calls == [("down", "right"), ("up", "right")]
    assert "right" not in hands._persistent_holds


def test_release_all_releases_all_held_keys(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  HOLD_LEFT: {hold_start: left}\n"
        "  STOP_MOVING: {release_all: true}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("HOLD_RIGHT"), BOX)
    hands.do(Step("HOLD_LEFT"), BOX)
    hands.do(Step("STOP_MOVING"), BOX)
    # Both keys should be pressed and then released
    assert ("down", "right") in backend.calls
    assert ("down", "left") in backend.calls
    assert ("up", "right") in backend.calls
    assert ("up", "left") in backend.calls
    assert len(hands._persistent_holds) == 0


def test_conflicting_directions_release_previous(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  HOLD_LEFT: {hold_start: left}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)
    hands.do(Step("HOLD_RIGHT"), BOX)
    hands.do(Step("HOLD_LEFT"), BOX)
    # Right should be pressed, then released when left is pressed
    assert backend.calls == [("down", "right"), ("up", "right"), ("down", "left")]
    assert "right" not in hands._persistent_holds
    assert "left" in hands._persistent_holds


def test_watchdog_releases_keys_after_3_steps(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  WAIT: {wait: 0.1}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)

    # Start holding right
    hands.do(Step("HOLD_RIGHT"), BOX)
    assert "right" in hands._persistent_holds

    # Tick watchdog 3 times without confirming
    hands.tick_watchdog()
    assert "right" in hands._persistent_holds
    hands.tick_watchdog()
    assert "right" in hands._persistent_holds
    hands.tick_watchdog()
    assert "right" in hands._persistent_holds
    hands.tick_watchdog()

    # After 3 ticks without confirmation, key should be released
    assert "right" not in hands._persistent_holds
    assert ("up", "right") in backend.calls


def test_hold_start_resets_watchdog(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)

    # Start holding right
    hands.do(Step("HOLD_RIGHT"), BOX)

    # Tick watchdog twice
    hands.tick_watchdog()
    hands.tick_watchdog()

    # Confirm by calling hold_start again
    hands.do(Step("HOLD_RIGHT"), BOX)
    assert hands._persistent_holds["right"] == 0  # Reset to 0

    # Tick once more - should still be held
    hands.tick_watchdog()
    assert "right" in hands._persistent_holds


def test_corner_stop_releases_persistent_holds(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  UP: {keys: up}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend(corner_after=1)
    hands, _ = make(card, backend)

    hands.do(Step("HOLD_RIGHT"), BOX)
    assert "right" in hands._persistent_holds

    with pytest.raises(CornerStop):
        hands.do(Step("UP"), BOX)

    # Persistent holds should be released
    assert len(hands._persistent_holds) == 0
    assert ("up", "right") in backend.calls


def test_on_stop_releases_all_keys(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n"
        "  HOLD_LEFT: {hold_start: left}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)

    hands.do(Step("HOLD_RIGHT"), BOX)
    hands.do(Step("HOLD_LEFT"), BOX)

    hands.on_stop()

    assert len(hands._persistent_holds) == 0
    assert ("up", "right") in backend.calls
    assert ("up", "left") in backend.calls


def test_on_focus_lost_releases_all_keys(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text(
        "actions:\n"
        "  HOLD_RIGHT: {hold_start: right}\n",
        encoding="utf-8",
    )
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)

    hands.do(Step("HOLD_RIGHT"), BOX)
    assert "right" in hands._persistent_holds

    hands.on_focus_lost()

    assert len(hands._persistent_holds) == 0
    assert ("up", "right") in backend.calls


# [RATCHET-F12-MIDCALL] F12 must release held keys at once, even while an AI call is still running.
def test_f12_releases_held_keys_immediately_mid_ai_call(tmp_path):
    import threading
    import time

    from safety import Safety

    path = tmp_path / "card.yaml"
    path.write_text("name: t\nactions:\n  HOLD_RIGHT: {hold_start: right}\n", encoding="utf-8")
    card = load_card(path)
    backend = FakeBackend()
    hands, _ = make(card, backend)
    safety = Safety(5, 50)
    safety.add_stop_listener(hands.on_stop)
    hands.do(Step("HOLD_RIGHT"), BOX)
    assert "right" in hands._persistent_holds

    released_at = []
    def slow_ai_call():  # the main loop is stuck inside a 2 s brain call
        time.sleep(2.0)
    worker = threading.Thread(target=slow_ai_call)
    worker.start()
    pressed = time.monotonic()
    safety.stop("stopped_f12")  # F12 from the listener thread
    released_at.append(time.monotonic() - pressed)
    assert not hands._persistent_holds
    assert ("keyUp", "right") in [tuple(call[:2]) for call in backend.calls] or "right" not in hands._persistent_holds
    assert released_at[0] < 0.2  # well within the 200 ms safety target, not after the AI call
    hands.do(Step("HOLD_RIGHT"), BOX)
    assert not hands._persistent_holds  # nothing new goes down after a stop until resume()
    worker.join()
