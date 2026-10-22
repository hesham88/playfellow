"""T32b-automover: the buddy learns that something keeps moving by itself (a key only steers) and turns before a wall."""
from __future__ import annotations

import numpy as np
import pytest

from nav.automover import AutoMover
from nav.planner import Navigator

GRID, TILE = 10, 20
KEYS = ("up", "down", "left", "right")
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ALIASES = {"w": "up", "s": "down", "a": "left", "d": "right"}


def floor():
    rng = np.random.default_rng(7)
    tiles = rng.integers(30, 70, size=(GRID, GRID, 3)).astype(np.uint8)
    return np.kron(tiles, np.ones((TILE, TILE, 1), dtype=np.uint8))


FLOOR = floor()


class Grid:
    """ORACLE (tests only). `rolling`: I keep moving every step and a key sets my heading (no straight-back turns).
    Otherwise a key moves me one tile. Leaving the grid ends the episode in the rolling game; a walker just stops."""

    def __init__(self, rolling: bool, aliases: bool = False):
        self.rolling, self.aliases = rolling, aliases
        self.keys = KEYS + (tuple(ALIASES) if aliases else ())
        self.reset()

    def reset(self):
        self.x, self.y, self.heading, self.dead = 2, 5, (1, 0), False
        return self.frame()

    def frame(self):
        img = FLOOR.copy()
        img[self.y * TILE + 2:self.y * TILE + TILE - 2, self.x * TILE + 2:self.x * TILE + TILE - 2] = (255, 0, 255)
        return img

    def step(self, key):
        key = ALIASES.get(key, key)
        if self.rolling:
            if key in DIRS and DIRS[key] != (-self.heading[0], -self.heading[1]):
                self.heading = DIRS[key]
            self.x, self.y = self.x + self.heading[0], self.y + self.heading[1]
            if not (0 <= self.x < GRID and 0 <= self.y < GRID):
                self.dead = True
        elif key in DIRS:
            dx, dy = DIRS[key]
            self.x, self.y = min(max(self.x + dx, 0), GRID - 1), min(max(self.y + dy, 0), GRID - 1)
        return self.frame()


def play(navigator, world, cap=80):
    """One episode of the NAVIGATE skill. Returns (steps survived, kinds of advice it gave)."""
    frame = world.reset()
    navigator.start(frame, {})
    kinds = []
    for used in range(1, cap + 1):
        advice = navigator.advise(frame)
        kinds.append(advice.kind)
        action = advice.actions[0] if advice.actions else None
        after = world.step(action)
        navigator.learn(frame, action, after, {}, world.dead)
        frame = after
        if world.dead:
            return used, kinds
    return cap, kinds


def trained_roller(episodes=4):
    navigator = Navigator(list(KEYS))
    for _ in range(episodes):
        play(navigator, Grid(rolling=True))
    return navigator


# ---- the counting model on its own ----------------------------------------------------------------------------
def test_pressing_nothing_and_still_moving_is_the_sign_of_rolling():
    auto = AutoMover()
    auto.observe("right", (20, 0))
    auto.observe(None, (20, 0))
    assert auto.rolling and auto.heading == (1, 0)


def test_pressing_nothing_and_stopping_is_the_sign_of_walking():
    auto = AutoMover()
    auto.observe("right", (20, 0))
    auto.observe(None, (0, 0))
    assert not auto.rolling and auto.heading is None


def test_a_key_that_turns_me_is_remembered_with_the_heading_it_worked_from():
    auto = AutoMover()
    auto.observe("right", (20, 0))
    auto.observe("up", (0, -20))  # heading right, up turned me up
    assert auto.turn_for("up", (1, 0)) == (0, -1)
    assert auto.turn_for("up", (-1, 0)) == (0, -1)  # a fixed way: also from other headings
    assert auto.turn_for("up", (0, 1)) == (0, 1)  # but straight back is ignored: I carry on
    assert auto.turn_for("left", (1, 0)) is None  # never seen


def test_a_key_that_changed_nothing_is_remembered_as_ignored():
    auto = AutoMover()
    auto.observe("right", (20, 0))
    auto.observe("down", (20, 0))  # pressed down, still going right
    assert auto.turn_for("down", (1, 0)) == (1, 0)


def test_it_saves_and_loads():
    auto = AutoMover()
    for key, move in [("right", (20, 0)), ("up", (0, -20)), ("up", (0, -20)), (None, (0, -20)), ("left", (-20, 0))]:
        auto.observe(key, move)
    again = AutoMover.from_dict(auto.to_dict())
    assert (again.kept_going, again.stopped, again.steered) == (auto.kept_going, auto.stopped, auto.steered)
    assert again.turn_for("up", (1, 0)) == auto.turn_for("up", (1, 0)) and again.ignored == auto.ignored


# ---- on a game ---------------------------------------------------------------------------------------------------
def test_a_roller_is_found_out_and_it_turns_before_the_wall():
    navigator = trained_roller()
    assert navigator.nav.auto.rolling
    survived, kinds = play(navigator, Grid(rolling=True), cap=60)
    assert survived == 60, f"it crashed after {survived} steps"
    assert "steer" in kinds


def test_without_steering_the_same_roller_crashes_early(monkeypatch):
    """The planted test must be able to fail: take the steering away and the roller drives into the wall."""
    from nav.planner import Planner
    monkeypatch.setattr(Planner, "_steer", lambda self, frame, advice: advice)
    navigator = trained_roller()
    survived, _ = play(navigator, Grid(rolling=True), cap=60)
    assert survived < 20


def test_a_walker_is_not_touched():
    navigator = Navigator(list(KEYS))
    steps, kinds = play(navigator, Grid(rolling=False), cap=60)
    assert steps == 60 and not navigator.nav.auto.rolling
    assert "steer" not in kinds and "wait" not in kinds


def test_a_walker_with_two_keys_for_each_way_is_not_taken_for_a_roller():
    navigator = Navigator(list(KEYS) + list(ALIASES))
    steps, kinds = play(navigator, Grid(rolling=False, aliases=True), cap=80)
    assert steps == 80 and not navigator.nav.auto.rolling
    assert "steer" not in kinds


def test_what_was_learned_about_rolling_carries_to_a_new_level():
    navigator = trained_roller()
    fresh = navigator.for_new_level()
    assert fresh.nav.auto.rolling and fresh.nav.auto.turns
