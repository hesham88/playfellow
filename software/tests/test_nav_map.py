"""T32b §2: the learned map — places, arrows, reward places from HUD changes, the look memory, saving."""
from __future__ import annotations

import numpy as np
import pytest

from nav import map as navmap
from nav.map import NavMap
from practice_arena.game import BANNER, LEVEL, TILE
from test_nav_worlds import ARENA_KEYS, ArenaWorld, make_arena_level

MAX_STEPS = 400


def explore(world, nav, max_steps=MAX_STEPS):
    """Play one episode with exploration only (no planner): probe until "me" is known, then Go-Explore."""
    frame = world.reset(0)
    nav.begin(frame, world.hud())
    actions = list(ARENA_KEYS)
    for used in range(1, max_steps + 1):
        action = nav.explore_action(frame, actions) if nav.me.known else nav.me.probe_action(actions)
        assert action is not None, "exploration ran out of ideas"
        after, _, done, _, info = world.step(action)
        nav.update(frame, action, after, info["hud"], done)
        frame = after
        if done:
            return used
    return None


def level_cell(nav, square):
    """ORACLE (tests only): which level cell a learned square sits on."""
    cx, cy = nav.center_of(square)
    return int(cx // TILE), int((cy - BANNER) // TILE)


@pytest.fixture(scope="module")
def explored():
    world, nav = ArenaWorld(), NavMap()
    steps = explore(world, nav)
    return world, nav, steps


def test_exploration_alone_wins_the_arena(explored):
    world, _, steps = explored
    assert steps is not None and world.game.won


def test_reward_places_come_from_hud_changes_and_sit_on_coins_or_the_key(explored):
    _, nav, _ = explored
    rewarded = [nav.parse(k)[1] for k, p in nav.places.items() if p.rewards]
    assert len(rewarded) >= 3
    for square in rewarded:
        x, y = level_cell(nav, square)
        assert LEVEL[y][x] in "ck", (square, (x, y))
    goals = [nav.parse(k)[1] for k, p in nav.places.items() if p.goal]
    assert [LEVEL[y][x] for x, y in (level_cell(nav, s) for s in goals)] == ["D"]


def test_walls_are_blocked_arrows_learned_by_bumping(explored):
    _, nav, _ = explored
    blocked = [(k, a) for k, acts in nav.edges.items() for a, e in acts.items() if e.best()[0] == "blocked"]
    assert blocked
    for key, action in blocked:
        x, y = level_cell(nav, nav.target_square(key, action))
        assert LEVEL[y][x] in "#D"


def _accuracy(nav, frame, level):
    guesses = nav.predict(frame)
    right = total = 0
    for square, labels in guesses.items():
        x, y = level_cell(nav, square)
        if not (0 <= y < len(level) and 0 <= x < len(level[0])) or level[y][x] in "PD":
            continue
        truth = {"#": "wall", ".": "floor", "c": "reward", "k": "reward"}[level[y][x]]
        total += 1
        right += labels == {truth}
    return right, total


def test_the_look_memory_reads_the_whole_screen(explored):
    world, nav, _ = explored
    frame = world.reset(0)
    right, total = _accuracy(nav, frame, LEVEL)
    cells = sum(c != "P" and c != "D" for row in LEVEL for c in row)
    assert total >= 0.9 * cells and right >= 0.95 * total, (right, total, cells)


def test_a_never_seen_level_is_read_from_its_first_frame(explored):
    """Requirement 90: what things look like carries over to a level it has never played."""
    _, nav, _ = explored
    fresh = nav.transferable()
    assert not fresh.places and not fresh.edges
    for seed in (100, 101, 102):
        level = make_arena_level(seed)
        world = ArenaWorld(level)
        frame = world.reset(0)
        fresh.begin(frame, world.hud())
        assert fresh.here is not None, "did not find myself on the new level"
        right, total = _accuracy(fresh, frame, level)
        assert total >= 0.85 * sum(c not in "PD" for row in level for c in row)
        assert right >= 0.95 * total, (seed, right, total)


def test_reading_the_whole_screen_at_once_matches_square_by_square(explored):
    world, nav, _ = explored
    frame = world.reset(0)
    squares = nav.visible_squares(frame)
    centers = [nav.center_of(sq) for sq in squares]
    single = [navmap.look(frame, c, nav.step()) for c in centers]
    batch = navmap.looks_many(frame, centers, nav.step())
    for a, b in zip(single, batch):
        assert (a is None and b is None) or np.allclose(a, b, atol=1e-3)
    assert [nav.looks.labels(v) for v in single] == nav.looks.labels_many(batch)


def test_blocked_arrows_are_retried_once_the_hud_changes():
    nav = NavMap()
    nav.scene = 0
    edge = navmap.Edge(tries=1, outcomes={"blocked": 1}, blocked_hud=navmap.hud_signature({"key": "no"}))
    nav.edges["0:1:1"] = {"right": edge}
    nav.hud = {"key": "no"}
    assert nav.retryable("0:1:1") == []
    nav.hud = {"key": "yes"}
    assert nav.retryable("0:1:1") == ["right"]


def test_the_map_survives_saving_and_loading(explored, tmp_path):
    world, nav, _ = explored
    navmap.save(nav, tmp_path, "practice_arena")
    assert (tmp_path / "practice_arena.nav.json").exists()
    back = navmap.load(tmp_path, "practice_arena")
    assert back.places.keys() == nav.places.keys()
    assert {k: {a: e.outcomes for a, e in v.items()} for k, v in back.edges.items()} == \
        {k: {a: e.outcomes for a, e in v.items()} for k, v in nav.edges.items()}
    frame = world.reset(0)
    assert back.predict(frame) == nav.predict(frame)
    assert back.me.effect("down") == nav.me.effect("down")


def test_a_damaged_map_file_starts_fresh(tmp_path):
    (tmp_path / "game.nav.json").write_text("{not json", encoding="utf-8")
    nav = navmap.load(tmp_path, "game")
    assert not nav.places and not nav.me.known
