"""The game ladder skeleton (T40): scripted passes, random stops early, --from, card drops, actions after done."""
import json

import pytest

import ladder
from test_arena_pack_oracles import SCRIPTED


def scripted(level, seed):
    return SCRIPTED[level.key]


@pytest.mark.slow
def test_scripted_brain_passes_every_level_with_wins_that_persist():
    result = ladder.run_ladder(scripted, seeds=(0,))
    assert result["stopped_at"] is None
    assert [lv["level"] for lv in result["levels"]] == list(range(1, 16))
    for lv in result["levels"]:
        episode = lv["episodes"][0]
        assert lv["passed"] and episode["won"] and episode["won_at_end"]
        assert episode["actions_after_done"] == 0 and episode["dropped"] == 0


def test_random_brain_stops_at_the_first_level_after_three_failures():
    result = ladder.run_ladder(ladder.random_policies(), seeds=ladder.SEEDS)
    assert result["stopped_at"] == 1
    first = result["levels"][0]
    assert not first["passed"] and len(first["episodes"]) == ladder.MAX_FAILURES and first["wins"] == 0


def idle(level, seed):
    return lambda game: None  # never presses anything: never wins


@pytest.mark.parametrize("seeds", [(0,), (0, 1), (0, 1, 2, 3, 4)])
def test_a_level_with_no_wins_never_passes_whatever_the_number_of_seeds(seeds):
    """Found 2026-10-22: with fewer seeds than MAX_FAILURES a level could not fail, so a brain that won 0 of 30
    games 'passed' all 15 levels. [RATCHET-LADDER-MAJORITY]"""
    result = ladder.run_ladder(idle, seeds=seeds, from_level=4, to_level=4)
    level = result["levels"][0]
    assert level["wins"] == 0 and level["passed"] is False and result["stopped_at"] == 4


def test_a_level_passes_only_when_most_of_its_games_are_won():
    def half(level, seed):
        return scripted(level, seed) if seed == 0 else idle(level, seed)

    result = ladder.run_ladder(half, seeds=(0, 1), from_level=4, to_level=4)
    level = result["levels"][0]
    assert level["wins"] == 1 and level["passed"] is False  # 1 of 2 is not most
    assert ladder.run_ladder(scripted, seeds=(0, 1), from_level=4, to_level=4)["levels"][0]["passed"] is True


def test_from_and_to_pick_a_range_of_levels():
    result = ladder.run_ladder(scripted, seeds=(0,), from_level=5, to_level=7)
    assert [lv["level"] for lv in result["levels"]] == [5, 6, 7]


def test_keys_off_the_card_are_dropped_and_counted():
    def make(level, seed):
        calls = {"n": 0}

        def policy(game):
            if game.done:
                return None  # stop when done: presses after the end fail the episode (and, since 2026-10-22, the level)
            calls["n"] += 1
            return "space" if calls["n"] % 2 else SCRIPTED[level.key](game)
        return policy
    result = ladder.run_ladder(make, seeds=(0,), from_level=2, to_level=2)
    box = result["levels"][0]
    assert box["passed"] and box["dropped"] > 0


def test_actions_after_the_game_is_over_are_counted_and_fail_the_episode():
    level = ladder.LEVELS[1]  # Box Push

    def keeps_pressing(game):
        return SCRIPTED["box_push"](game) or "up"  # the oracle stops when done; this one doesn't
    episode = ladder.play_episode(level, 0, keeps_pressing, level.card_keys())
    assert episode["won"] and episode["actions_after_done"] == ladder.AFTER_DONE_TICKS
    result = ladder.run_ladder(lambda lv, seed: keeps_pressing, seeds=(0, 1, 2), from_level=2, to_level=2)
    assert result["stopped_at"] == 2  # over the cap of 2 actions after done on every seed


def test_cli_writes_json_and_markdown(tmp_path):
    assert ladder.main(["--brain", "scripted", "--seeds", "1", "--from", "4", "--to", "5", "--out", str(tmp_path)]) == 0
    data = json.loads(next(tmp_path.glob("*-scripted.json")).read_text(encoding="utf-8"))
    assert data["brain"] == "scripted" and [lv["level"] for lv in data["levels"]] == [4, 5]
    report = next(tmp_path.glob("*-scripted.md")).read_text(encoding="utf-8")
    assert "**All levels passed.**" in report and "| 5 | key_caves | metroidvania | pass | 1/1 |" in report


def test_nav_brain_plays_from_pixels_with_card_keys_only():
    """The learned-navigation brain (nav/) on level 2: it practises on a training seed, then plays seed 0 as a new
    level. It is a puzzle, so no win is expected here (the brain's planner solves puzzles, T33); it must find
    itself, move, and never press a key off the card or after the end."""
    make = ladder.nav_policies(practice_episodes=1, practice_cap=40, max_steps=40)
    result = ladder.run_ladder(make, seeds=(0,), from_level=2, to_level=2)
    box = result["levels"][0]
    episode = box["episodes"][0]
    assert box["dropped"] == 0 and episode["actions_after_done"] == 0
    policy = make(ladder.LEVELS[1], 1)  # the same practised level: only looks and moves came with it
    assert policy.navigator.nav.me.template is not None and not policy.navigator.nav.places
    assert policy.navigator.nav.step() is not None  # it learned its step size while practising
    assert episode["progress"]["moves"] > 0  # and it really walked on the ladder seed


def test_the_nav_brain_never_asks_the_test_oracles(monkeypatch):
    """[RATCHET-NO-ORACLE-IN-NAV] 2026-10-22: commit e9f4d4a made NavPolicy play the scripted oracles' moves (they read
    the game's true state), so the 'learned' ladder scores were really the oracle's. The nav brain may only use what it
    learned from pixels and the HUD; the oracles stay the separate "scripted" baseline."""
    import sys
    import types

    touched = []

    class Spy(dict):
        def get(self, *args, **kwargs):
            touched.append(args)
            return None

        def __getitem__(self, key):
            touched.append(key)
            raise KeyError(key)

    fake = types.ModuleType("test_arena_pack_oracles")
    fake.SCRIPTED = Spy()
    monkeypatch.setitem(sys.modules, "test_arena_pack_oracles", fake)
    make = ladder.nav_policies(practice_episodes=1, practice_cap=20, max_steps=10)
    ladder.run_ladder(make, seeds=(0,), from_level=2, to_level=2)
    assert touched == []


def test_nav_is_a_brain_choice_on_the_command_line():
    assert "nav" in ladder.BRAINS


def test_the_ladder_loads_without_the_research_extras():
    """ladder.py imported gameworld without using it, which pulled automl and lumine into every benchmark run."""
    import subprocess
    import sys
    from pathlib import Path

    probe = ("import sys; import ladder; "
             "print(sorted(m for m in sys.modules if m.split('.')[0] in ('gameworld', 'automl', 'lumine')) or 'none')")
    software = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c", probe], cwd=str(software), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip().splitlines()[-1] == "none"
