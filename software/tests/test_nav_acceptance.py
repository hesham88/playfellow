"""T32b §5 acceptance on the Practice Arena (scripted, no AI, no network), with each kind of "new" kept apart (R90):

  1. SAME LEVEL (memorised): the brief's numbers — the level is replayed, so later episodes reuse a learned map;
  2. NEW LAYOUT of a game already played: held-out generated levels, carrying over only what things look like
     and how I move (no places);
  3. COLD START: nothing known at all (the baseline for 2).

The HUD numbers come from the game's own banner values here, not from reading the screen (live play reads them).
Route efficiency = oracle steps / steps taken; the oracle lives in tests only.
"""
from __future__ import annotations

import statistics

import pytest

from nav.planner import Navigator
from nav.practice import play_episode, practice
from practice_arena.game import COIN_POINTS, DOOR_POINTS, LEVEL
from test_nav_worlds import ARENA_KEYS, ArenaWorld, arena_after_first_episode, arena_optimal_steps, make_arena_level

HELD_OUT = tuple(range(200, 210))  # never used for anything but evaluation
QUICK = (200,)  # the default suite: one held-out layout, carried over and cold (all ten run with -m slow)
MAX_SCORE = COIN_POINTS * sum(row.count("c") for row in LEVEL) + DOOR_POINTS


@pytest.fixture(scope="module")
def memorised():
    navigator, first = arena_after_first_episode()  # episode 1 from a cold start (shared, trained once)
    later = practice(ArenaWorld(), navigator, episodes=4, oracle_steps=arena_optimal_steps(), save=False)
    return navigator, [first] + later


def test_same_level_memorised_meets_the_brief(memorised):
    _, episodes = memorised
    assert episodes[0].won  # episode 1: wins (all coins, the key, the door — the score proves all of it)
    assert episodes[0].score == MAX_SCORE
    assert max(e.efficiency for e in episodes[:3]) >= 0.8  # by episode 3
    assert episodes[4].efficiency >= 0.95 and episodes[4].score == MAX_SCORE  # by episode 5
    assert episodes[4].efficiency > episodes[0].efficiency  # the practice curve rises
    assert all(b.efficiency >= a.efficiency - 1e-9 for a, b in zip(episodes[1:], episodes[2:]))


def _first_game(level, navigator):
    world = ArenaWorld(level)
    result = play_episode(world, navigator, oracle_steps=arena_optimal_steps(level))
    return result, world.game.coins_left == 0


def test_new_layouts_are_won_first_time_and_beat_the_cold_baseline(memorised):
    trained, _ = memorised
    for seed in QUICK:
        level = make_arena_level(seed)
        carried, all_coins = _first_game(level, trained.for_new_level())
        assert carried.won and all_coins, seed
        assert carried.efficiency >= 0.75, (seed, carried.efficiency)
        if seed == QUICK[0]:
            cold, cold_coins = _first_game(level, Navigator(list(ARENA_KEYS)))
            assert cold.won and cold_coins, seed  # even with nothing known it gets there…
            assert carried.efficiency > cold.efficiency, (seed, carried.efficiency, cold.efficiency)  # …but slower


@pytest.mark.slow
def test_all_held_out_layouts(memorised):
    trained, _ = memorised
    carried, cold = [], []
    for seed in HELD_OUT:
        level = make_arena_level(seed)
        result, all_coins = _first_game(level, trained.for_new_level())
        assert result.won and all_coins, seed
        carried.append(result.efficiency)
        result, all_coins = _first_game(level, Navigator(list(ARENA_KEYS)))
        assert result.won and all_coins, seed
        cold.append(result.efficiency)
    assert statistics.mean(carried) >= 0.9
    assert statistics.mean(carried) >= 1.5 * statistics.mean(cold)
