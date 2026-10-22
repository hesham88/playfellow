"""Controls cards for real famous games (T45): they load, are safe, are bilingual, cite a source, and get picked
for the right game window."""
import re
from pathlib import Path

import pytest
import yaml

from controls import load_card, match_card_for_game

CONTROLS = Path(__file__).resolve().parent.parent / "controls"
ARABIC = re.compile(r"[؀-ۿ]")

# card → a realistic window title for that game
REAL_GAMES = {
    "chess_com_bots": "Play Chess Online Against the Computer - Chess.com - Google Chrome",
    "subway_surfers": "Subway Surfers - Play Online for Free! | Poki - Google Chrome",  # poki.com <title>, 2026-10-16
    "asphalt": "Asphalt Legends Unite",
    "football_efootball": "eFootball™",
    "fc_football": "EA SPORTS FC™ 25",
    "mortal_kombat": "Mortal Kombat 1",
    "minecraft": "Minecraft 1.21.1 - Singleplayer",
    "hollow_knight": "Hollow Knight",
    "stardew_valley": "Stardew Valley",
    "celeste": "Celeste",
    "age_of_empires_ii": "Age of Empires II: Definitive Edition",
    "doom_classic": "DOOM",
}


def raw(name: str) -> dict:
    return yaml.safe_load((CONTROLS / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", REAL_GAMES)
def test_card_loads_with_genre_window_and_goals(name):
    card = load_card(CONTROLS / f"{name}.yaml")
    data = raw(name)
    assert data["genre"] and card.window
    assert card.default_goal and ARABIC.search(card.default_goal_ar)
    assert card.name_ar and ARABIC.search(card.game_hint_ar) and card.game_hint


@pytest.mark.parametrize("name", REAL_GAMES)
def test_card_cites_where_its_controls_come_from(name):
    head = (CONTROLS / f"{name}.yaml").read_text(encoding="utf-8").split("\nname:")[0]
    assert re.search(r"#.*https?://\S+", head), f"{name}: no source URL in the header comment"


@pytest.mark.parametrize("name", REAL_GAMES)
def test_no_reserved_keys_and_every_held_key_can_be_released(name):
    card = load_card(CONTROLS / f"{name}.yaml")
    used = {k for a in card.actions.values() for k in a.keys} | {a.hold_key for a in card.actions.values() if a.hold_key}
    assert not used & {"f9", "f12"}
    held = {a.hold_key for a in card.actions.values() if a.kind == "hold_start"}
    released = {a.hold_key for a in card.actions.values() if a.kind == "hold_stop"}
    has_release_all = any(a.kind == "release_all" for a in card.actions.values())
    assert held <= released or has_release_all
    if held:
        assert has_release_all, f"{name} holds keys but has no RELEASE_ALL"


@pytest.mark.parametrize("name", REAL_GAMES)
def test_every_action_says_what_it_does_in_both_languages(name):
    card = load_card(CONTROLS / f"{name}.yaml")
    for action in card.actions.values():
        assert action.says and ARABIC.search(action.says_ar), f"{name}.{action.name}"
        assert not ARABIC.search(action.says), f"{name}.{action.name}: Arabic in the English text"


@pytest.mark.parametrize("name,title", REAL_GAMES.items())
def test_the_game_window_picks_this_card(name, title):
    assert match_card_for_game(title, CONTROLS) == name


def test_chess_com_card_is_bots_only():
    data = raw("chess_com_bots")
    assert data["play_only_against"] == "computer_bots"
    assert "person" in data["stop_if"] and "rating" in data["stop_if"]
    assert "BOTS ONLY" in data["game_hint"]
    card = load_card(CONTROLS / "chess_com_bots.yaml")
    assert card.uses_clicks and not any(a.keys for a in card.actions.values())


def test_unverified_cards_say_so():
    for name in ("football_efootball", "fc_football"):
        head = (CONTROLS / f"{name}.yaml").read_text(encoding="utf-8").split("\nname:")[0]
        assert "VERIFY" in head
