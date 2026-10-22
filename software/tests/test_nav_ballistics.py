"""T32b: Ballistics — the buddy learns gravity, drag, pushes and jump kicks from watching itself, then plans arcs."""
from __future__ import annotations

import numpy as np
import pytest

from nav.dynamics import Ballistics, Body

DT = 0.05
FLOOR_Y = 200.0
GRAVITY, DRAG, PUSH, KICK = 900.0, 0.85, 400.0, -420.0


def floor(x, y):
    return y >= FLOOR_Y


class Platformer:
    """ORACLE (tests only): a tiny platformer with known physics, written separately from the model."""

    def __init__(self, x=20.0, y=FLOOR_Y - 1):
        self.x, self.y, self.vx, self.vy = x, y, 0.0, 0.0

    @property
    def grounded(self):
        return self.y >= FLOOR_Y - 1 and self.vy >= 0

    def tick(self, key):
        on_ground = self.grounded
        self.vx = DRAG * self.vx + {"right": PUSH, "left": -PUSH}.get(key, 0.0) * DT
        self.vy += GRAVITY * DT
        if key == "jump" and on_ground:
            self.vy = KICK
        self.x += self.vx * DT
        self.y += self.vy * DT
        if self.y >= FLOOR_Y - 1:
            self.y, self.vy = FLOOR_Y - 1, 0.0
        return self.x, self.y


def watch(script, seed=0):
    game, model = Platformer(), Ballistics()
    model.observe(game.x, game.y, None, DT)
    for key in script:
        model.observe(*game.tick(key), key, DT)
    return game, model


SCRIPT = (["right"] * 6 + [None] * 4 + ["jump"] + [None] * 22 + ["left"] * 5 + ["jump", "right", "right"] + [None] * 22) * 3


@pytest.fixture(scope="module")
def learned():
    game, model = watch(SCRIPT)
    assert model.fit()
    return model


def test_too_little_to_say_returns_false():
    model = Ballistics()
    model.observe(0, 0, None, DT)
    model.observe(1, 0, "right", DT)
    assert model.fit() is False and model.known is False


def test_learns_gravity_and_drag(learned):
    assert learned.gravity == pytest.approx(GRAVITY, rel=0.1)
    assert learned.drag == pytest.approx(DRAG, abs=0.05)


def test_learns_which_key_pushes_which_way_and_which_key_jumps(learned):
    assert learned.push["right"] > 0 > learned.push["left"]
    assert learned.push["right"] == pytest.approx(PUSH, rel=0.15)
    assert learned.kick["jump"] == pytest.approx(KICK, rel=0.1)
    assert "jump" not in learned.push


def test_the_imagined_jump_matches_the_real_one(learned):
    real = Platformer()
    real.tick("jump")
    real_path = [(real.x, real.y)] + [real.tick("right") for _ in range(7)]
    start = Body(20.0, FLOOR_Y - 1, grounded=True)
    jump = learned.step(start, "jump", DT, floor)
    path = [jump] + learned.predict(jump, "right", 350, DT, floor)
    assert len(path) == len(real_path)
    for body, (x, y) in zip(path, real_path):
        assert abs(body.x - x) < 6 and abs(body.y - y) < 8


def test_a_floor_stops_a_fall_and_marks_me_grounded(learned):
    path = learned.predict(Body(50.0, 150.0), None, 2000, DT, floor)
    assert path[-1].grounded and path[-1].y == pytest.approx(FLOOR_Y - 1, abs=15) and path[-1].vy == 0.0


def test_the_jump_peaks_where_the_kick_and_gravity_say(learned):
    start = Body(20.0, FLOOR_Y - 1, grounded=True)
    peak = min(b.y for b in learned.predict(start, "jump", 50, DT, floor, after_ms=1500))
    assert FLOOR_Y - 1 - peak == pytest.approx(KICK ** 2 / (2 * GRAVITY), rel=0.15)


def test_plans_a_jump_over_a_spike_onto_a_ledge(learned):
    spike = lambda x, y: 70 <= x <= 90 and y >= FLOOR_Y - 12
    ledge_goal = (120.0, FLOOR_Y - 1)
    start = Body(20.0, FLOOR_Y - 1, grounded=True)
    plan = learned.plan_platform_trajectory(start, ledge_goal, reach=25, solid=floor, danger=spike)
    assert plan is not None and plan["jump"] == "jump" and plan["side"] == "right"
    # and the plan really works in the real game
    game = Platformer()
    for n in range(40):
        key = "jump" if n == 0 else ("right" if n <= round(plan["hold_ms"] / 1000 / DT) else None)
        x, y = game.tick(key)
        assert not spike(x, y), "the real jump landed on the spike"
    assert abs(game.x - ledge_goal[0]) < 25


def test_a_goal_it_cannot_reach_gives_no_plan(learned):
    start = Body(20.0, FLOOR_Y - 1, grounded=True)
    assert learned.plan_platform_trajectory(start, (900.0, 20.0), reach=10, solid=floor) is None


def test_a_danger_in_the_only_way_gives_no_plan(learned):
    wall_of_spikes = lambda x, y: 60 <= x <= 400  # spikes the whole way, up in the air too
    start = Body(20.0, FLOOR_Y - 1, grounded=True)
    assert learned.plan_platform_trajectory(start, (200.0, FLOOR_Y - 1), reach=25, solid=floor, danger=wall_of_spikes) is None


def test_saves_and_loads(learned):
    again = Ballistics.from_dict(learned.to_dict())
    assert again.gravity == learned.gravity and again.push == learned.push and again.kick == learned.kick


def test_a_maze_with_no_gravity_learns_no_gravity():
    model, rng = Ballistics(), np.random.default_rng(1)
    x = y = 100.0
    model.observe(x, y, None, DT)
    for _ in range(60):
        key, (dx, dy) = [("right", (8, 0)), ("left", (-8, 0)), ("up", (0, -8)), ("down", (0, 8))][rng.integers(4)]
        x, y = x + dx, y + dy
        model.observe(x, y, key, DT)
    model.fit()
    assert abs(model.gravity) < 200  # a maze has no pull; whatever is learned must not be a falling arc
    assert model.kick == {}


# ---- wired into the map: real-time frames in, numbers out -------------------------------------------------------
def _frames(script):
    from test_nav_scrolling import make_world
    ground = make_world(5)[:240, :320].copy()
    ground[int(FLOOR_Y) + 8:, :] = (30, 120, 30)

    def draw(game):
        img = ground.copy()
        x, y = int(round(game.x)) - 8, int(round(game.y)) - 15
        img[max(y, 0):y + 16, max(x, 0):x + 16] = (255, 0, 255)
        return img
    game = Platformer(x=60.0)
    yield game, draw(game), None
    for key in script:
        before = draw(game)
        game.tick(key)
        yield game, before, key, draw(game)


def test_the_map_learns_gravity_and_the_jump_from_real_time_frames():
    from nav.map import NavMap
    walk = (["right"] * 4 + [None] * 3 + ["left"] * 6 + [None] * 3 + ["jump"] + [None] * 22) * 4
    nav, first = NavMap(), True
    for item in _frames(walk):
        if first:
            nav.begin(item[1], {})
            first = False
            continue
        _, before, key, after = item
        nav.update(before, key, after, {}, False, dt=DT)
    nav.ballistics.fit()
    assert nav.me.known
    assert nav.ballistics.gravity == pytest.approx(GRAVITY, rel=0.25)
    assert nav.ballistics.kick.get("jump") == pytest.approx(KICK, rel=0.25)
    # it is saved with the map and carries over to a level never seen
    again = NavMap.from_dict(nav.to_dict())
    assert again.ballistics.gravity == nav.ballistics.gravity
    assert nav.transferable().ballistics.kick == nav.ballistics.kick


def test_without_a_clock_the_map_learns_no_ballistics():
    from nav.map import NavMap
    nav, first = NavMap(), True
    for item in _frames((["right", "left", "jump"] + [None] * 5) * 3):
        if first:
            nav.begin(item[1], {})
            first = False
            continue
        _, before, key, after = item
        nav.update(before, key, after, {}, False)  # no dt: the old turn-based way
    assert not nav.ballistics.known


# ---- the top of a jump is not the ground ------------------------------------------------------------------------
def _arc(dt, gravity=GRAVITY, kick=KICK):
    """y positions of a jump seen every `dt` seconds (ORACLE), starting on the floor."""
    y, vy, ys = FLOOR_Y - 1, kick, []
    for _ in range(int(2.2 * abs(kick) / gravity / dt)):
        vy += gravity * dt
        y += vy * dt
        if y >= FLOOR_Y - 1:
            y, vy = FLOOR_Y - 1, 0.0
        ys.append(y)
    return ys


@pytest.mark.parametrize("dt", [0.05, 0.02, 0.01])
def test_the_top_of_a_jump_is_never_marked_as_standing(dt):
    model = Ballistics()
    model.gravity = GRAVITY  # as if already learned
    for _ in range(4):
        model.observe(50.0, FLOOR_Y - 1, None, dt)  # resting on the floor first
    bodies = [model.observe(50.0, y, "jump" if n == 0 else None, dt) for n, y in enumerate(_arc(dt))]
    low = FLOOR_Y - 1 - 5  # anything clearly above the floor
    midair = [b for b, y in zip(bodies, _arc(dt)) if y < low]
    assert midair and not any(b.grounded for b in midair)
    after = [model.observe(50.0, FLOOR_Y - 1, None, dt) for _ in range(4)]
    assert after[-1].grounded  # and it is standing again once it has rested a few frames


def test_standing_still_for_a_few_frames_is_standing():
    model = Ballistics()
    bodies = [model.observe(50.0, FLOOR_Y - 1, None, DT) for _ in range(6)]
    assert [b.grounded for b in bodies] == [False, False, False, True, True, True]


def test_without_the_still_frames_rule_the_apex_would_be_marked_standing(monkeypatch):
    """The planted test must be able to fail: with the old one-frame rule the arc's top reads as the ground."""
    model = Ballistics()
    model.gravity = GRAVITY
    monkeypatch.setattr(Ballistics, "_standing", lambda self, dy, dt: abs(dy) < 0.5)
    ys = _arc(0.01)
    bodies = [model.observe(50.0, y, None, 0.01) for y in [FLOOR_Y - 1] + ys]
    assert any(b.grounded for b, y in zip(bodies[1:], ys) if y < FLOOR_Y - 6)
