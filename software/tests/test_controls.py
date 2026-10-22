from pathlib import Path

import pytest

from controls import CardError, describe_card, list_cards, load_card, normalize_name

CARDS = Path(__file__).resolve().parent.parent / "controls"


def write(tmp_path, text):
    path = tmp_path / "card.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_all_shipped_cards_load():
    names = list_cards(CARDS)
    assert {"practice_arena", "platformer", "top_down", "first_person", "point_and_click", "racing"} <= set(names)
    for name in names:
        card = load_card(CARDS / f"{name}.yaml")
        assert card.actions
        assert card.game_hint


def test_practice_arena_card():
    card = load_card(CARDS / "practice_arena.yaml")
    assert set(card.actions) == {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
    assert card.actions["UP"].kind == "keys" and card.actions["UP"].keys == ("up",)
    assert card.uses_clicks is False


def test_every_kind_is_read(tmp_path):
    card = load_card(write(tmp_path, """
name: Test
game_hint: a test
actions:
  jump: {keys: space, hold: 0.3, says: jump}
  run right: {keys: [shift, right]}
  CLICK: {click: left}
  SHOOT: {mouse_button: left}
  TURN_LEFT: {mouse_move: [-200, 0]}
  WAIT: {wait: 0.5}
"""))
    assert card.actions["JUMP"].keys == ("space",) and card.actions["JUMP"].hold == 0.3
    assert card.actions["RUN_RIGHT"].keys == ("shift", "right")
    assert card.actions["CLICK"].kind == "click"
    assert card.actions["SHOOT"].kind == "mouse_button"
    assert (card.actions["TURN_LEFT"].dx, card.actions["TURN_LEFT"].dy) == (-200, 0)
    assert card.actions["WAIT"].kind == "wait" and card.actions["WAIT"].hold == 0.5
    assert card.uses_clicks is True


def test_f12_is_reserved_for_emergency_stop(tmp_path):
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  PANIC: {keys: f12}\n"))
    assert "F12" in str(error.value)


def test_f9_is_reserved_for_take_over(tmp_path):
    for text in ("actions:\n  SWAP: {keys: f9}\n", "actions:\n  SWAP: {keys: [shift, f9]}\n",
                 "actions:\n  SWAP: {hold_start: f9}\n"):
        with pytest.raises(CardError) as error:
            load_card(write(tmp_path, text))
        assert "F9" in str(error.value)


def test_unknown_key(tmp_path):
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  FLY: {keys: rocket}\n"))
    assert "rocket" in str(error.value)


def test_hold_must_be_a_number(tmp_path):
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  UP: {keys: up, hold: long}\n"))
    assert "number" in str(error.value)


def test_hold_range(tmp_path):
    with pytest.raises(CardError):
        load_card(write(tmp_path, "actions:\n  UP: {keys: up, hold: 10}\n"))


def test_bad_mouse_button(tmp_path):
    with pytest.raises(CardError):
        load_card(write(tmp_path, "actions:\n  CLICK: {click: thumb}\n"))


def test_action_without_a_kind(tmp_path):
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  DANCE: {says: dance}\n"))
    assert "DANCE" in str(error.value)


def test_card_needs_actions(tmp_path):
    with pytest.raises(CardError):
        load_card(write(tmp_path, "name: Empty\n"))


def test_bad_action_name(tmp_path):
    with pytest.raises(CardError):
        load_card(write(tmp_path, "actions:\n  '123': {keys: up}\n"))


def test_missing_card_file(tmp_path):
    with pytest.raises(CardError) as error:
        load_card(tmp_path / "ghost.yaml")
    assert "ghost.yaml" in str(error.value)


def test_normalize_name():
    assert normalize_name(" [move left] ") == "MOVE_LEFT"
    assert normalize_name("jump-high") == "JUMP_HIGH"


def test_describe_card_mentions_click_coordinates(tmp_path):
    card = load_card(write(tmp_path, "actions:\n  CLICK: {click: left, says: click a thing}\n  UP: {keys: up}\n"))
    text = describe_card(card)
    assert "- CLICK: click a thing" in text and "0 to 1000" in text
    assert "- UP:" in text


def test_shipped_cards_have_both_languages_separately():
    arabic = lambda text: any("؀" <= ch <= "ۿ" for ch in text)
    for name in list_cards(CARDS):
        card = load_card(CARDS / f"{name}.yaml")
        # game_hint and says should be English only (no Arabic letters)
        assert not arabic(card.game_hint), f"{name}: game_hint has Arabic"
        # game_hint_ar should contain Arabic
        assert arabic(card.game_hint_ar), f"{name}: game_hint_ar has no Arabic"
        # name_ar should contain Arabic
        assert arabic(card.name_ar), f"{name}: name_ar has no Arabic"
        for action in card.actions.values():
            assert not arabic(action.says), f"{name}: {action.name} says has Arabic"
            assert arabic(action.says_ar), f"{name}: {action.name} says_ar has no Arabic"


def test_any_game_card_loads_and_has_no_f9_or_f12():
    """Task 20 §0: any_game card loads, has no f9/f12, and every action is a single key."""
    card = load_card(CARDS / "any_game.yaml")
    assert card.name == "Any game (keyboard)"
    assert card.actions
    # Check no f9 or f12
    for action_name in card.actions:
        assert "F9" not in action_name.upper(), f"any_game has {action_name} but f9 should be excluded"
        assert "F12" not in action_name.upper(), f"any_game has {action_name} but f12 should be excluded"
    # Check every action is a single key (no multi-key combos)
    for action in card.actions.values():
        if action.kind == "keys":
            assert len(action.keys) == 1, f"{action.name} has {len(action.keys)} keys, should be 1"


def test_rayman_card_has_backspace():
    """Task 20 §0: verify Rayman card has KEY_BACKSPACE (added in Task 19)."""
    card = load_card(CARDS / "rayman_legends.yaml")
    assert "KEY_BACKSPACE" in card.actions, "Rayman card should have KEY_BACKSPACE"
    assert card.actions["KEY_BACKSPACE"].keys == ("backspace",)


def test_prompt_contains_key_hint_rule():
    """Task 20 §0: system prompt contains the key-hint rule."""
    from brain.prompts import system_prompt
    from settings import Settings

    settings = Settings()
    card = load_card(CARDS / "practice_arena.yaml")
    prompt = system_prompt(settings, card)
    assert "Menus and games often show key hints like [Space] or (ESC)" in prompt
    assert "read them first and press the matching KEY_ action" in prompt


def test_hold_start_action(tmp_path):
    """Task 31: hold_start action kind."""
    card = load_card(write(tmp_path, """
actions:
  HOLD_RIGHT: {hold_start: right, says: start holding right}
"""))
    assert card.actions["HOLD_RIGHT"].kind == "hold_start"
    assert card.actions["HOLD_RIGHT"].hold_key == "right"


def test_hold_stop_action(tmp_path):
    """Task 31: hold_stop action kind."""
    card = load_card(write(tmp_path, """
actions:
  STOP_RIGHT: {hold_stop: right, says: stop holding right}
"""))
    assert card.actions["STOP_RIGHT"].kind == "hold_stop"
    assert card.actions["STOP_RIGHT"].hold_key == "right"


def test_release_all_action(tmp_path):
    """Task 31: release_all action kind."""
    card = load_card(write(tmp_path, """
actions:
  STOP_MOVING: {release_all: true, says: stop all movement}
"""))
    assert card.actions["STOP_MOVING"].kind == "release_all"
    assert card.actions["STOP_MOVING"].release_all_flag is True


def test_hold_start_rejects_unknown_key(tmp_path):
    """Task 31: hold_start rejects unknown keys."""
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  HOLD_ROCKET: {hold_start: rocket}\n"))
    assert "rocket" in str(error.value)


def test_hold_start_rejects_f12(tmp_path):
    """Task 31: hold_start rejects F12."""
    with pytest.raises(CardError) as error:
        load_card(write(tmp_path, "actions:\n  HOLD_F12: {hold_start: f12}\n"))
    assert "F12" in str(error.value)


def test_any_game_has_hold_actions():
    """Task 31: any_game has HOLD_LEFT, HOLD_RIGHT, STOP_MOVING."""
    card = load_card(CARDS / "any_game.yaml")
    assert "HOLD_LEFT" in card.actions
    assert "HOLD_RIGHT" in card.actions
    assert "STOP_MOVING" in card.actions
    assert card.actions["HOLD_LEFT"].kind == "hold_start"
    assert card.actions["HOLD_RIGHT"].kind == "hold_start"
    assert card.actions["STOP_MOVING"].kind == "release_all"


def test_rayman_has_hold_and_glide_actions():
    """Task 31: rayman_legends has hold actions and gliding moves."""
    card = load_card(CARDS / "rayman_legends.yaml")
    # Hold actions
    assert "HOLD_LEFT" in card.actions
    assert "HOLD_RIGHT" in card.actions
    assert "STOP_MOVING" in card.actions
    # Gliding/flying actions
    assert "JUMP_HOLD" in card.actions
    assert "GLIDE_RIGHT" in card.actions
    assert "GLIDE_LEFT" in card.actions
    assert "AIR_ATTACK" in card.actions
    # Check kinds and timings
    assert card.actions["JUMP_HOLD"].kind == "keys"
    assert card.actions["JUMP_HOLD"].hold == 1.2
    assert card.actions["GLIDE_RIGHT"].kind == "keys"
    assert card.actions["GLIDE_RIGHT"].hold == 1.6
    assert card.actions["AIR_ATTACK"].hold == 0.4
    # Check game hint mentions floating cages
    assert "floating in the air" in card.game_hint or "floating cages" in card.game_hint


def test_punctuation_keys_are_allowed_on_cards():
    """Real games bind punctuation keys (owner's real-game cards, T45)."""
    from controls import KNOWN_KEYS
    for key in [";", ".", ",", "/", "'", "[", "]", "-", "=", "`", "\\"]:
        assert key in KNOWN_KEYS, key
