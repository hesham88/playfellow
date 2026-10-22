from pathlib import Path

import pytest

from controls import load_card
from reply import Step, extract_json, parse_reply


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text("actions:\n  UP: {keys: up}\n  LEFT: {keys: left}\n  CLICK: {click: left}\n", encoding="utf-8")
    return load_card(path)


def test_clean_json(card):
    reply = parse_reply('{"thought": "coin above", "say": "Going up!", "actions": ["UP"]}', card, 3)
    assert reply.thought == "coin above" and reply.say == "Going up!"
    assert reply.steps == [Step("UP")]
    assert not reply.confused and reply.dropped == []


def test_code_fences_and_extra_words(card):
    text = 'Sure! Here you go:\n```json\n{"thought": "t", "say": "", "actions": ["left"]}\n```\nHave fun'
    assert parse_reply(text, card, 3).steps == [Step("LEFT")]


def test_bracketed_and_lowercase_names(card):
    assert parse_reply('{"actions": ["[up]", " Left "]}', card, 3).steps == [Step("UP"), Step("LEFT")]


def test_unknown_actions_are_dropped_never_kept(card):
    reply = parse_reply('{"actions": ["UP", "DELETE_EVERYTHING", "JUMP"]}', card, 3)
    assert reply.steps == [Step("UP")]
    assert len(reply.dropped) == 2 and "DELETE_EVERYTHING" in reply.dropped[0]


def test_click_needs_coordinates_inside_the_grid(card):
    text = ('{"actions": [{"action": "CLICK", "x": 500, "y": 250.4}, {"action": "CLICK"}, '
            '{"action": "CLICK", "x": 1200, "y": 5}, {"action": "CLICK", "x": "abc", "y": 5}]}')
    reply = parse_reply(text, card, 8)
    assert reply.steps == [Step("CLICK", 500, 250)]
    assert len(reply.dropped) == 3


def test_too_many_actions(card):
    reply = parse_reply('{"actions": ["UP", "UP", "LEFT", "UP"]}', card, 2)
    assert reply.steps == [Step("UP"), Step("UP")]
    assert len(reply.dropped) == 2 and "too many" in reply.dropped[0]


def test_single_action_as_text(card):
    assert parse_reply('{"actions": "UP"}', card, 3).steps == [Step("UP")]


def test_not_json_is_confused(card):
    reply = parse_reply("I think I should go up.", card, 3)
    assert reply.confused and reply.steps == []


def test_empty_reply_is_confused(card):
    assert parse_reply("", card, 3).confused


def test_long_text_is_trimmed(card):
    reply = parse_reply('{"thought": "' + "a" * 2000 + '"}', card, 3)
    assert len(reply.thought) == 500


def test_labels():
    assert Step("CLICK", 1, 2).label() == "CLICK(1,2)"
    assert Step("UP").label() == "UP"


def test_extract_json_rejects_lists():
    assert extract_json("[1, 2]") is None


def test_mission_update_valid_push(card):
    """Task 20 §2: mission_update with type=push and text is accepted."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "push", "text": "Collect coins"}}', card, 3)
    assert reply.mission_update == {"type": "push", "text": "Collect coins"}


def test_mission_update_valid_complete(card):
    """Task 20 §2: mission_update with type=complete (no text needed) is accepted."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "complete"}}', card, 3)
    assert reply.mission_update == {"type": "complete"}


def test_mission_update_valid_replace(card):
    """Task 20 §2: mission_update with type=replace and text is accepted."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "replace", "text": "New goal"}}', card, 3)
    assert reply.mission_update == {"type": "replace", "text": "New goal"}


def test_mission_update_missing_becomes_none(card):
    """Task 20 §2: missing mission_update becomes None."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": []}', card, 3)
    assert reply.mission_update is None


def test_mission_update_invalid_type_becomes_none(card):
    """Task 20 §2: invalid type becomes None."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "invalid"}}', card, 3)
    assert reply.mission_update is None


def test_mission_update_push_without_text_becomes_none(card):
    """Task 20 §2: push without text becomes None."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "push"}}', card, 3)
    assert reply.mission_update is None


def test_mission_update_replace_with_empty_text_becomes_none(card):
    """Task 20 §2: replace with empty text becomes None."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": {"type": "replace", "text": "   "}}', card, 3)
    assert reply.mission_update is None


def test_mission_update_not_a_dict_becomes_none(card):
    """Task 20 §2: non-dict mission_update becomes None."""
    reply = parse_reply('{"thought": "t", "say": "", "actions": [], "mission_update": "push"}', card, 3)
    assert reply.mission_update is None
