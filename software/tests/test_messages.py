from messages import MESSAGES, msg, set_language, tr
import pytest


def test_every_message_has_english_and_arabic():
    for key, (english, arabic) in MESSAGES.items():
        assert english.strip(), key
        assert arabic.strip(), key
        assert any("؀" <= ch <= "ۿ" for ch in arabic), f"{key} Arabic text has no Arabic letters"


def test_msg_fills_values_and_blanks_missing_ones():
    text = msg("no_window", title="Minecraft")
    assert "Minecraft" in text
    assert msg("brain_trouble").startswith("My brain had a problem: ")


def test_msg_returns_only_the_current_language():
    # English by default contains no Arabic letters
    text = msg("no_window", title="X")
    assert not any("؀" <= ch <= "ۿ" for ch in text)
    assert "\n" not in text

    # After set_language("arabic"), msg returns Arabic text with no newline
    set_language("arabic")
    text = msg("no_window", title="X")
    assert any("؀" <= ch <= "ۿ" for ch in text)
    assert "\n" not in text


def test_tr():
    # English by default
    assert tr("hello", "مرحبا") == "hello"

    # Arabic when set
    set_language("arabic")
    assert tr("hello", "مرحبا") == "مرحبا"


def test_unknown_language_rejected():
    with pytest.raises(ValueError) as error:
        set_language("french")
    assert "unknown language" in str(error.value)


def test_no_message_sends_the_player_to_a_grown_up():
    """PlayFellow is 18+ for everyone (owner, 2026-10-22): the player is the adult in charge, so no message tells them
    to go and fetch one."""
    for key, (english, arabic) in MESSAGES.items():
        assert "grown-up" not in english.lower() and "grown up" not in english.lower(), key
        assert "شخص بالغ" not in arabic, key
