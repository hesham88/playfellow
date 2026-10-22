from types import SimpleNamespace

import pytest
from PIL import Image

from brain import BrainError, classify_error, make_brain, to_jpeg_bytes
from brain.prompts import system_prompt, turn_text
from controls import load_card
from settings import Settings


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "card.yaml"
    path.write_text("game_hint: a maze\nactions:\n  UP: {keys: up, says: go up}\n", encoding="utf-8")
    return load_card(path)


@pytest.fixture
def click_card(tmp_path):
    path = tmp_path / "click.yaml"
    path.write_text("actions:\n  CLICK: {click: left}\n", encoding="utf-8")
    return load_card(path)


def error_with(**attributes):
    exc = Exception(attributes.pop("text", "something"))
    for name, value in attributes.items():
        setattr(exc, name, value)
    return exc


def test_wrong_key():
    assert classify_error(error_with(status_code=401)).key == "bad_key"
    assert classify_error(Exception("API key not valid")).key == "bad_key"


def test_rate_limit_waits_and_retries():
    error = classify_error(error_with(code=429))
    assert error.key == "busy" and error.retry and error.wait_seconds == 10
    assert classify_error(Exception("RESOURCE_EXHAUSTED quota")).key == "busy"


def test_server_trouble_retries():
    error = classify_error(error_with(status_code=503))
    assert error.key == "busy" and error.retry


def test_no_internet():
    error = classify_error(ConnectionError("getaddrinfo failed"))
    assert error.key == "offline" and error.retry


def test_timeout_is_busy_not_offline():
    # Timeouts and deadlines should be treated as "busy" (service overloaded) not "offline"
    error = classify_error(TimeoutError("request timed out"))
    assert error.key == "busy" and error.retry and error.wait_seconds == 3

    error = classify_error(Exception("DEADLINE_EXCEEDED after 60s"))
    assert error.key == "busy" and error.retry

    error = classify_error(error_with(status_code=504, text="Gateway Timeout"))
    assert error.key == "busy" and error.retry

    # Connection failures are still "offline"
    error = classify_error(ConnectionError("failed to connect"))
    assert error.key == "offline"


def test_bad_request_does_not_retry():
    error = classify_error(error_with(status_code=404, text="model not found"))
    assert error.key == "brain_trouble" and not error.retry and "model not found" in error.detail


def test_unknown_problem_retries():
    error = classify_error(RuntimeError("weird"))
    assert error.key == "brain_trouble" and error.retry


def test_status_given_as_words_is_ignored_safely():
    error = classify_error(error_with(status="RESOURCE_EXHAUSTED"))
    assert error.key == "busy"


def test_friendly_message_is_in_the_current_language():
    # English by default
    text = BrainError("busy", retry=True, wait_seconds=10).friendly()
    assert "10 seconds" in text

    # In Arabic
    from messages import set_language
    set_language("arabic")
    text = BrainError("busy", retry=True, wait_seconds=10).friendly()
    assert any("؀" <= ch <= "ۿ" for ch in text)


def test_make_brain_needs_a_key_except_local():
    with pytest.raises(BrainError) as error:
        make_brain("gemini", "m", None)
    assert error.value.key == "no_key"
    with pytest.raises(BrainError):
        make_brain("chatgpt", "m", "k")


def test_jpeg_bytes():
    data = to_jpeg_bytes(Image.new("RGBA", (10, 10)))
    assert data[:2] == b"\xff\xd8"


def test_system_prompt_lists_only_card_actions(card):
    text = system_prompt(Settings(buddy_name="Zappy", personality="silly", language="arabic"), card)
    assert "Zappy" in text and "silly" in text
    assert "- UP: go up" in text and "a maze" in text
    assert "Arabic" in text
    assert '"actions"' in text and "JSON" in text
    assert "grid" not in text


def test_system_prompt_explains_grid_for_click_cards(click_card):
    assert "0 to 1000" in system_prompt(Settings(), click_card)


def test_turn_text_has_goal_memory_chat_and_captain():
    text = turn_text("get the key", "1 step ago you did: UP", ["go left!"], "walk to the red door")
    assert "get the key" in text and "1 step ago" in text
    assert "- go left!" in text and "walk to the red door" in text
    assert text.strip().endswith("Reply with JSON only.")


def test_turn_text_skips_empty_parts():
    text = turn_text("goal", "", [], None)
    assert "captain" not in text and "player just wrote" not in text


def test_system_prompt_contains_missions_section(card):
    """Task 20 §2: system prompt includes missions section."""
    text = system_prompt(Settings(), card)
    assert "Missions:" in text or "missions" in text.lower()
    assert "mission_update" in text
    assert '"type": "push|complete|replace|none"' in text or '"type": "none"' in text


def test_system_prompt_json_shape_includes_mission_update(card):
    """Task 20 §2: JSON shape line includes mission_update."""
    text = system_prompt(Settings(), card)
    assert '"mission_update"' in text
    assert '"type": "none"' in text


def test_turn_text_with_missions_text():
    """Task 20 §2: turn_text accepts missions_text and puts it before the goal."""
    text = turn_text("get the key", "", [], None, missions_text="Main mission: Finish level → Sub-mission: Collect coins")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # missions_text should come before goal
    assert "Main mission: Finish level" in text
    assert "The goal: get the key" in text
    # Check order
    mission_pos = text.index("Main mission")
    goal_pos = text.index("The goal")
    assert mission_pos < goal_pos


def test_turn_text_without_missions_text():
    """Task 20 §2: turn_text works without missions_text (default empty)."""
    text = turn_text("goal", "", [], None)
    assert "goal" in text
    # No extra missions line
    assert text.count("Main mission") == 0


def test_quota_exhausted_gives_quota_error_not_busy():
    """Task 21 §0: Quota exhaustion is classified as 'quota', not 'busy'."""
    error = classify_error(Exception("429 RESOURCE_EXHAUSTED You exceeded your current quota"))
    assert error.key == "quota"
    assert not error.retry

    error = classify_error(Exception("insufficient_quota"))
    assert error.key == "quota"
    assert not error.retry

    error = classify_error(Exception("billing limit reached"))
    assert error.key == "quota"

    error = classify_error(Exception("quota exceeded"))
    assert error.key == "quota"


def test_plain_429_stays_busy():
    """Task 21 §0: A plain 429 without quota words stays 'busy'."""
    error = classify_error(error_with(code=429, text="rate limit hit"))
    assert error.key == "busy"
    assert error.retry


def test_fallback_falls_back_on_quota():
    """Task 21 §0: FallbackBrain treats quota like a retryable error for fallback purposes."""
    from brain import FallbackBrain

    class ScriptedBrain:
        def __init__(self):
            self.calls = []
            self.script = []

        def ask(self, system: str, text: str, images: list) -> str:
            self.calls.append((system, text))
            if self.script:
                item = self.script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            return '{"ok": true}'

    primary = ScriptedBrain()
    backup = ScriptedBrain()
    primary.script = [BrainError("quota")]
    backup.script = ['{"from": "backup"}']

    brain = FallbackBrain(primary, backup)
    result = brain.ask("sys", "text", [])

    assert result == '{"from": "backup"}'
    assert brain.last_used == "backup"
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1


def test_stale_goal_line_in_system_prompt(card):
    """Task 21 §0.3: System prompt tells the agent to say when a goal is already achieved."""
    text = system_prompt(Settings(), card)
    assert "already achieved" in text.lower() or "already done" in text.lower() or "step is already" in text.lower()
    assert "say so and move on" in text.lower() or "never repeat" in text.lower()


def test_turn_text_marks_player_messages(card):
    """Task 21 §1 (header renamed in Task 28): Player messages appear under 'The player says (answer them first)'."""
    text = turn_text("goal", "", ["go left", "jump!"], None)
    assert "The player says" in text
    assert "answer them first" in text.lower()
    assert "go left" in text
    assert "jump!" in text


def test_the_ai_is_told_it_plays_with_an_adult(card):
    """PlayFellow is 18+ (owner, 2026-10-22): the prompt must not tell the AI its player is a child."""
    for language in ("english", "arabic"):
        text = system_prompt(Settings(language=language), card)
        assert "12 to 14" not in text and "young player" not in text
        assert "adult" in text
