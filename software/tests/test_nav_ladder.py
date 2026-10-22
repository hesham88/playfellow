"""T32b §5: the same nav module, unchanged, on ladder levels 2 (Box Push) and 5 (Key Caves) and a side-scroller.

Box Push is a puzzle: deciding WHICH square to stand on and which way to push is the brain's job (T33 tools such
as a search skill), not navigation's. Here an ORACLE BRAIN (tests only; it reads the real game state) makes those
decisions, and the NAVIGATE skill must carry them out precisely, using only what it learned from pixels.
"""
from __future__ import annotations

import pytest

from arena_pack.base import BANNER
from arena_pack.box_push import LEVELS, TILE, BoxPush
from arena_pack.key_caves import KeyCaves
from nav.planner import Navigator
from nav.practice import PackWorld, play_episode
from test_nav_worlds import SIDE_KEYS, SideScroller, side_optimal_steps


def _box_square(game, nav, cell):
    """ORACLE: the learned grid square of a Box Push cell (via its pixel centre)."""
    level = LEVELS[game.level_no]
    ox = (game.board_pixels()[0] - max(len(row) for row in level) * TILE) // 2
    return nav.square((ox + cell[0] * TILE + TILE / 2, BANNER + cell[1] * TILE + TILE / 2))


def test_box_push_navigate_carries_out_the_brains_plan():
    game = BoxPush(seed=0)
    world = PackWorld(game)
    navigator = Navigator(list(game.keys))
    frame = world.reset(0)
    navigator.start(frame, world.hud())

    def press(action):
        nonlocal frame
        after, _, done, _, info = world.step(action)
        navigator.learn(frame, action, after, info["hud"], done)
        frame = after

    steps = 0
    while not navigator.nav.me.known or navigator.nav.here is None or navigator.nav.step() is None:
        advice = navigator.advise(frame)
        press(advice.actions[0] if advice.actions else navigator.nav.me.probe_action(list(game.keys)))
        steps += 1
        assert steps < 20, "never found myself"
    press("r")  # put the box back where it started
    # ORACLE BRAIN: "stand left of the box, then push right twice"
    (bx, by), = game.boxes
    target = _box_square(game, navigator.nav, (bx - 1, by))
    for _ in range(30):
        advice = navigator.advise(frame, f"to:{target[0]},{target[1]}")
        if advice.kind == "arrived":
            break
        press(advice.actions[0])
    else:
        raise AssertionError("NAVIGATE never arrived")
    assert game.player == (bx - 1, by)
    press("right")
    press("right")
    assert game.level_no == 1 and game.score == 100  # level 1 solved; the next level is a new scene
    assert navigator.nav.scene != 0


@pytest.mark.slow
def test_key_caves_rooms_are_told_apart_and_spikes_become_a_danger():
    """Measured limit (T32b report): it does NOT yet cross the spikes — that needs several steps of look-ahead on
    learned physics (a reflex/dynamics skill), not a map."""
    game = KeyCaves(seed=0)
    navigator = Navigator(list(game.keys))
    result = play_episode(PackWorld(game), navigator, cap=300)
    nav = navigator.nav
    assert nav.me.known and nav.step() == (40.0, 40.0)  # the tile size, learned
    assert len(nav.scenes) >= 2  # room 1 and room 2 are different places on the map
    if result.ended and not result.won:  # it fell on the spikes: that square is now a danger, never walked again
        assert any(p.hazard for p in nav.places.values())


@pytest.mark.slow
def test_side_scroller_cold_start_reaches_the_flag_and_then_takes_the_short_way():
    navigator = Navigator(list(SIDE_KEYS))
    first = play_episode(SideScroller(), navigator, cap=400)
    second = play_episode(SideScroller(), navigator, cap=400)
    assert first.won or second.won
    third = play_episode(SideScroller(), navigator, cap=400, oracle_steps=side_optimal_steps())
    # 0.85, not 1.0: places are positions only, and the same spot mid-jump moves differently rising or falling
    # (hidden speed), so the learned route takes ~2 extra steps (measured 17 vs the oracle's 15).
    assert third.won and third.efficiency >= 0.85
