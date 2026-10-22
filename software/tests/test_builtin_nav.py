"""T31-day1-brain: the built-in brain plays Day 1 (the Practice Arena) with no key, no network and no AI model."""
from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brain import BrainError, make_brain
from brain.builtin_nav import BuiltinNavBrain
from controls import load_card
from reply import parse_reply
from test_nav_worlds import ArenaWorld

SOFTWARE = Path(__file__).resolve().parent.parent
CARD = load_card(SOFTWARE / "controls" / "practice_arena.yaml")
ARABIC = re.compile("[؀-ۿ]")
LATIN = re.compile("[A-Za-z]")
# The question an engine sends when the buddy is stuck (the private engine's words); the brain spots
# HELP_MARK in it and answers with nothing, so the engine can word its own question.
HELP_SYSTEM = ("You are a friendly game buddy who is stuck and wants to ask the player for help. "
               "Write ONE short question (at most 25 words) in {language}.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open a connection fails the test: the built-in brain must work fully offline."""
    def refuse(*args, **kwargs):
        pytest.fail("the built-in brain tried to use the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def play(brain, max_steps=300, chat=""):
    """The Practice Arena, one frame per ask, doing exactly what the card says. Returns (world, replies)."""
    world = ArenaWorld()
    frame = world.reset(0)
    replies = []
    for _ in range(max_steps):
        text = "The goal: collect everything." + (f"\n\nThe player says (answer them first):\n- {chat}" if chat else "")
        raw = brain.ask("system prompt", text, [Image.fromarray(frame)])
        reply = parse_reply(raw, CARD, 4)
        replies.append((raw, reply))
        if world.game.won:
            break
        for step in reply.steps:
            frame, *_ = world.step(CARD.actions[step.action].keys[0])
    return world, replies


@pytest.mark.parametrize("language", ["english", "arabic"])
def test_it_beats_the_practice_arena_with_no_key_and_no_network(language):
    world, replies = play(BuiltinNavBrain(CARD, language))
    assert world.game.won and world.game.coins_left == 0
    assert len(replies) < 200  # it learns the maze on the spot: the first try, no practice runs


def test_it_returns_only_actions_on_the_card_and_nothing_is_dropped():
    _, replies = play(BuiltinNavBrain(CARD, "english"))
    for raw, reply in replies:
        data = json.loads(raw)
        assert set(data["actions"]) <= set(CARD.actions)
        assert reply.dropped == [] and not reply.confused
    assert any(reply.steps for _, reply in replies)


def test_the_reply_has_the_shape_every_other_brain_uses():
    raw = BuiltinNavBrain(CARD).ask("s", "t", [Image.new("RGB", (576, 504), "black")])
    data = json.loads(raw)
    assert {"thought", "say", "actions", "mission_update", "hud", "facts"} <= set(data)
    assert isinstance(data["actions"], list) and data["mission_update"] == {"type": "none"}


def test_thoughts_stay_in_one_language():
    _, english = play(BuiltinNavBrain(CARD, "english"), max_steps=80)
    _, arabic = play(BuiltinNavBrain(CARD, "arabic"), max_steps=80)
    for raw, _ in english:
        data = json.loads(raw)
        assert not ARABIC.search(data["thought"] + data["say"])
    for raw, _ in arabic:
        data = json.loads(raw)
        assert not LATIN.search(data["thought"] + data["say"]) and ARABIC.search(data["thought"])


def test_it_says_when_it_finds_itself_and_does_not_chatter():
    _, replies = play(BuiltinNavBrain(CARD, "english"))
    spoken = [r.say for _, r in replies if r.say]
    assert any("Found it" in line for line in spoken)
    assert len(spoken) <= len(replies) // 4


def test_a_stop_from_the_player_stops_it_moving():
    brain = BuiltinNavBrain(CARD, "english")
    world, _ = play(brain, max_steps=15)
    raw = brain.ask("s", "The goal: x\n\nThe player says (answer them first):\n- please stop", [Image.fromarray(ArenaWorld().reset(0))])
    data = json.loads(raw)
    assert data["actions"] in ([], ["WAIT"]) and "stopping" in data["thought"].lower()


def test_an_arabic_stop_works_too():
    brain = BuiltinNavBrain(CARD, "arabic")
    raw = brain.ask("s", "The goal: x\n\nThe player says (answer them first):\n- قف من فضلك", [Image.fromarray(ArenaWorld().reset(0))])
    assert json.loads(raw)["actions"] in ([], ["WAIT"])


def test_no_picture_means_wait_not_crash():
    for language in ("english", "arabic"):
        data = json.loads(BuiltinNavBrain(CARD, language).ask("s", "t", []))
        assert data["actions"] == [] and data["thought"]


def test_a_card_without_arrow_keys_gives_no_actions_instead_of_crashing():
    card = load_card(SOFTWARE / "controls" / "point_and_click.yaml")
    data = json.loads(BuiltinNavBrain(card).ask("s", "t", [Image.new("RGB", (200, 200), "grey")]))
    assert data["actions"] == []


def test_a_moving_message_in_the_top_bar_is_not_mistaken_for_a_pickup():
    brain = BuiltinNavBrain(CARD)
    world = ArenaWorld()
    frame = world.reset(0)
    for _ in range(40):  # walk back and forth into a wall: only the message line changes, never the score
        brain._bar(frame)
        frame, *_ = world.step("left")
    assert world.game.score == 0 and brain._bar_events == 0


def test_make_brain_builds_it_without_a_key_and_needs_the_card():
    brain = make_brain("builtin", "none", None, card=CARD, language="arabic")
    assert isinstance(brain, BuiltinNavBrain) and brain.language == "arabic"
    with pytest.raises(BrainError):
        make_brain("builtin", "none", None)


def test_other_brains_still_need_their_key():
    with pytest.raises(BrainError):
        make_brain("gemini", "m", None)


def test_seeing_pickups_in_the_top_bar_makes_it_much_faster(monkeypatch):
    """The planted test must be able to fail: blind it to the bar and the same maze takes far longer."""
    _, seeing = play(BuiltinNavBrain(CARD), max_steps=400)
    monkeypatch.setattr(BuiltinNavBrain, "_bar", lambda self, frame: {"bar": 0})
    _, blind = play(BuiltinNavBrain(CARD), max_steps=400)
    assert len(seeing) < 0.7 * len(blind)


# ---- the loop also asks this brain things that are not "make a move" --------------------------------------------


def test_a_help_question_gets_an_empty_answer_so_the_engine_words_its_own():
    brain = BuiltinNavBrain(CARD, "english")
    frame = Image.fromarray(ArenaWorld().reset(0))
    assert brain.ask(HELP_SYSTEM.format(language="english"), "The goal: x", [frame]) == ""


def test_the_take_over_summary_is_words_only_and_teaches_the_map_nothing():
    brain = BuiltinNavBrain(CARD, "english")
    world = ArenaWorld()
    frame = world.reset(0)
    brain.ask("system", "The goal: x", [Image.fromarray(frame)])
    before = brain.navigator.nav.episodes, brain._last, brain._prev.copy()
    theirs = [Image.fromarray(world.step("down")[0]) for _ in range(3)]  # the player's own moves
    raw = brain.ask("system", "The player just took over and played; these are frames you saw. Send no actions.", theirs)
    data = json.loads(raw)
    assert data["actions"] == [] and "watching" in data["thought"] and data["say"]
    assert (brain.navigator.nav.episodes, brain._last) == before[:2] and np.array_equal(brain._prev, before[2])


def test_those_extra_calls_have_arabic_words_too_and_never_mix_languages():
    brain = BuiltinNavBrain(CARD, "arabic")
    frame = Image.fromarray(ArenaWorld().reset(0))
    data = json.loads(brain.ask("s", "Send no actions.", [frame]))
    assert ARABIC.search(data["thought"] + data["say"]) and not LATIN.search(data["thought"] + data["say"])


def test_a_help_call_does_not_make_it_think_it_pressed_a_key():
    """Before the fix a help question set 'my last key' to a move that was never pressed, spoiling what it learned."""
    brain = BuiltinNavBrain(CARD, "english")
    frame = Image.fromarray(ArenaWorld().reset(0))
    brain.ask("system", "t", [frame])
    pressed = brain._last
    brain.ask(HELP_SYSTEM.format(language="english"), "t", [frame])
    assert brain._last == pressed


# ---- one learner: when the game loop's quick tick feeds the navigator, the brain only advises -------------------
def test_an_external_learner_means_ask_never_feeds_the_navigator_but_still_advises(monkeypatch):
    from nav.hud_bar import HudBar
    from nav.planner import Navigator

    brain = BuiltinNavBrain(CARD, "english")
    shared = Navigator(brain.keys)
    calls = []
    for name in ("start", "learn"):
        real = getattr(shared, name)
        monkeypatch.setattr(shared, name, lambda *a, _n=name, _r=real, **k: (calls.append(_n), _r(*a, **k))[1])
    advised = []
    real_advise = shared.advise
    monkeypatch.setattr(shared, "advise", lambda *a, **k: (advised.append(1), real_advise(*a, **k))[1])
    brain.use_navigator(shared, HudBar())
    world = ArenaWorld()
    frame = Image.fromarray(world.reset(0))
    for _ in range(3):
        brain.ask("system", "t", [frame])
    assert calls == [] and len(advised) == 3  # never start/learn from here; advice every time


def test_use_navigator_hands_over_the_map_and_the_top_bar_reader():
    from nav.hud_bar import HudBar
    from nav.planner import Navigator

    brain = BuiltinNavBrain(CARD, "english")
    own_navigator, own_bar = brain.navigator, brain.hud_bar
    assert not brain.external_learner
    nav, bar = Navigator(brain.keys), HudBar()
    brain.use_navigator(nav, bar)
    assert brain.navigator is nav and brain.hud_bar is bar and brain.external_learner
    assert brain.navigator is not own_navigator and brain.hud_bar is not own_bar


def test_on_its_own_the_brain_still_learns_from_what_it_asks():
    """The default (sync play, no quick tick) is unchanged: the brain is the only learner."""
    brain = BuiltinNavBrain(CARD, "english")
    world, _ = play(brain, max_steps=40)
    assert not brain.external_learner and brain.navigator.nav.me.known
