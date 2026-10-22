"""D1: console and phone stream windows may be played (with the adult warning); stores, browsers and the Xbox app stay blocked."""
import pytest

from bridges import BRIDGES, detect, is_stream_window
from games import detect_games, is_blocked, needs_online_warning

STREAMS = ["PS Remote Play", "ps remote play", "Chiaki", "chiaki-ng", "Xbox Remote Play", "Xbox Cloud Gaming", "scrcpy",
           "iPhone Mirroring"]


@pytest.mark.parametrize("title", STREAMS)
def test_stream_windows_are_allowed_and_the_adult_is_warned(title):
    assert is_stream_window(title)
    assert not is_blocked(title)
    assert needs_online_warning(title)
    assert detect_games([title]) == [title]


@pytest.mark.parametrize("title", [
    "Xbox",  # the plain Xbox app (a store / launcher): blocked as an exact title
    "xbox",
    "Xbox - Microsoft Store",
    "Microsoft Store - Xbox Remote Play",  # a store page that merely mentions it
    "Xbox Cloud Gaming - Microsoft Edge",  # in a browser tab
    "xbox.com/play - Google Chrome",
    "PS Remote Play - Mozilla Firefox",
    "PlayStation Store",
    "PlayStation Store - Chiaki",
    "PS Remote Play checkout",
    "Chiaki payment",
    "Epic Games Launcher",
    "Steam",
])
def test_stores_browsers_payments_and_the_xbox_app_stay_blocked(title):
    assert is_blocked(title), title
    assert not is_stream_window(title) or is_blocked(title)


def test_a_money_word_stops_a_stream_from_counting_as_one():
    assert not is_stream_window("scrcpy - Upgrade to premium")
    assert not needs_online_warning("scrcpy - Upgrade to premium")


def test_the_narrow_door_does_not_open_for_ordinary_blocked_windows():
    for title in ("Discord", "Google Chrome", "Task Manager", "PayPal - Checkout"):
        assert is_blocked(title)
        assert detect(title) is None


def test_ordinary_games_do_not_get_the_stream_warning():
    assert not needs_online_warning("Celeste")
    assert not is_blocked("Celeste")


def test_each_bridge_names_a_card_and_has_patterns():
    assert {b.key for b in BRIDGES} == {"playstation", "xbox", "android", "ios"}
    for bridge in BRIDGES:
        assert bridge.patterns and bridge.card and bridge.name_ar
        assert detect(bridge.patterns[0]) is bridge
    phones = [b for b in BRIDGES if b.key in ("android", "ios")]
    assert all(b.aspect < 1 for b in phones) and all(b.aspect > 1.7 for b in BRIDGES if b not in phones)
