"""The game ladder: play the arena pack from easy to hard and record how well a brain does.
/ سُلّم الألعاب: العب حزمة الساحة من السهل إلى الصعب وسجّل أداء العقل.

    python ladder.py --brain scripted|random|nav [--from N] [--to N] [--seeds 5] [--out ladder]
                     [--realtime [--think-ms 250]]

Each level is played on fixed seeds through the games' step() interface (one clock tick per step). For every
episode it records: won (only if the win still holds after the game keeps running), score, steps, seconds,
AI calls, mean step latency, actions the controls card dropped, and actions sent after the game was over
(at most 2 are allowed). The ladder stops at the first level that fails 3 times; --from resumes.
Writes ladder/<date>-<brain>.json and a readable .md table.

With --realtime the game clock keeps running while the brain thinks: every decision costs --think-ms of game
time (hazards, timers and physics move on), instead of the world waiting one tick per decision.

Brains in this skeleton: `scripted` (the test oracles from tests/, a baseline that proves each level can be won;
solvers stay out of the buddy itself), `random` (presses card keys at random) and `nav` (the buddy's own
learned navigation, nav/: it practises each level on a training seed, then plays the ladder seeds carrying over
only what things look like and how it moves — never the places; it sees only pixels and the banner's HUD).
Real brains plug in later through the same policy interface: policy(game) -> key name or None.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np

from arena_pack.action_quest import ActionQuest
from arena_pack.box_push import BoxPush
from arena_pack.brick_breaker import BrickBreaker
from arena_pack.key_caves import KeyCaves
from arena_pack.lane_dash import LaneDash
from arena_pack.maze_chase import MazeChase
from arena_pack.mini_chess import MiniChess
from arena_pack.mini_fighter import MiniFighter
from arena_pack.mini_football import MiniFootball
from arena_pack.slide_2048 import Slide2048
from arena_pack.ray_maze import RayMaze
from arena_pack.snake import Snake
from arena_pack.tiny_kingdom import TinyKingdom
from arena_pack.top_racer import TopRacer
from arena_pack.turn_quest import TurnQuest
from controls import load_card
from nav.curriculum import CurriculumScheduler
from nav.planner import Navigator
from nav.practice import PackWorld, practice
from nav.surprise import SurpriseModule

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3, 4)
PRACTICE_SEEDS = 5  # the curriculum spreads practice over this many training seeds (never the ladder's own seeds)
MAX_FAILURES = 3
AFTER_DONE_TICKS = 5  # keep the clock running after the end: the win must persist
MAX_ACTIONS_AFTER_DONE = 2

Policy = Callable[[object], "str | None"]


@dataclass(frozen=True)
class Level:
    number: int
    key: str
    genre: str
    game_class: type

    def card_keys(self) -> set[str]:
        """Key names the card allows, plus "click" when it has a mouse-click action."""
        card = load_card(HERE / "controls" / f"arena_{self.key}.yaml")
        keys = {key for action in card.actions.values() for key in action.keys}
        return keys | ({"click"} if card.uses_clicks else set())


LEVELS = (Level(1, "slide_2048", "puzzle", Slide2048), Level(2, "box_push", "puzzle", BoxPush),
          Level(3, "mini_chess", "board", MiniChess), Level(4, "maze_chase", "maze", MazeChase),
          Level(5, "key_caves", "metroidvania", KeyCaves), Level(6, "turn_quest", "rpg", TurnQuest),
          Level(7, "tiny_kingdom", "strategy", TinyKingdom), Level(8, "ray_maze", "first_person", RayMaze),
          Level(9, "snake", "arcade", Snake),
          Level(10, "lane_dash", "endless_runner", LaneDash), Level(11, "top_racer", "racing", TopRacer),
          Level(12, "mini_football", "football", MiniFootball), Level(13, "brick_breaker", "arcade", BrickBreaker),
          Level(14, "mini_fighter", "fighting", MiniFighter), Level(15, "action_quest", "action_rpg", ActionQuest))


def _name(action) -> str:
    return action[0] if isinstance(action, tuple) else action


def play_episode(level: Level, seed: int, policy: Policy, allowed: set[str], think_ms: float | None = None) -> dict:
    """`think_ms` set = real-time: each decision lets that many milliseconds of game time pass."""
    game = level.game_class(seed=seed)
    if think_ms is not None:
        game.realtime, game.step_dt = True, (game.seconds_per_tick() or 0.05) if think_ms == 0 else think_ms / 1000
    game.observe = getattr(policy, "needs_pixels", False)  # scripted/random read state; real brains will need pixels
    steps = dropped = 0
    truncated = False
    started = time.perf_counter()
    while not game.done and not truncated:
        action = policy(game)
        if action and _name(action) not in allowed:  # the card is an allow-list: anything else is dropped
            dropped += 1
            action = None
        truncated = game.step(action)[3]
        steps += 1
    seconds = time.perf_counter() - started
    won_at_end = game.is_won()
    after_done = 0
    for _ in range(AFTER_DONE_TICKS):
        action = policy(game)
        if action:
            after_done += 1
        game.step(action if action and _name(action) in allowed else None)  # step() follows game.realtime
    persisted = won_at_end and game.is_won()
    episode = {"seed": seed, "won": persisted, "won_at_end": won_at_end, "score": game.score, "steps": steps,
               "seconds": round(seconds, 2), "ai_calls": getattr(policy, "ai_calls", 0),
               "mean_step_ms": round(1000 * seconds / steps, 2) if steps else 0.0, "dropped": dropped,
               "actions_after_done": after_done, "truncated": truncated, "progress": game.progress()}
    diagnostics = getattr(policy, "diagnostics", None)  # a brain that reports on itself (the Lumine agent does)
    if callable(diagnostics):
        episode["diagnostics"] = diagnostics()
    return episode


def run_ladder(make_policy: Callable[[Level, int], Policy], seeds=SEEDS, from_level: int = 1, to_level: int = 99,
               think_ms: float | None = None) -> dict:
    levels = []
    stopped_at = None
    for level in (lv for lv in LEVELS if from_level <= lv.number <= to_level):
        allowed = level.card_keys()
        episodes, failures = [], 0
        for seed in seeds:
            episode = play_episode(level, seed, make_policy(level, seed), allowed, think_ms)
            episodes.append(episode)
            failures += not episode["won"] or episode["actions_after_done"] > MAX_ACTIONS_AFTER_DONE
            if failures >= MAX_FAILURES:
                break
        # Most of the played games must be won, as well as fewer than MAX_FAILURES lost: with fewer seeds than
        # MAX_FAILURES a level could otherwise never fail (found 2026-10-22: 0 of 30 games won, "15/15 passed").
        passed = failures < MAX_FAILURES and 2 * (len(episodes) - failures) > len(episodes)
        levels.append({"level": level.number, "game": level.key, "genre": level.genre, "passed": passed,
                       "wins": sum(e["won"] for e in episodes), "episodes": episodes,
                       "mean_score": round(statistics.fmean(e["score"] for e in episodes), 1),
                       "mean_steps": round(statistics.fmean(e["steps"] for e in episodes), 1),
                       "max_actions_after_done": max(e["actions_after_done"] for e in episodes),
                       "dropped": sum(e["dropped"] for e in episodes)})
        if not passed:
            stopped_at = level.number
            break
    return {"levels": levels, "stopped_at": stopped_at, "seeds": list(seeds), "from": from_level, "to": to_level,
            "realtime": think_ms is not None, "think_ms": think_ms}


def render(result: dict) -> str:
    mode = f" — real-time, {result['think_ms']:g} ms per decision" if result.get("realtime") else ""
    out = [f"# Game ladder — {result['brain']}{mode} — {result['date']}", "",
           f"Seeds {result['seeds']}; a level passes only if most of its games are won, and fails after {MAX_FAILURES} "
           f"lost episodes. A win counts only if it still "
           f"holds {AFTER_DONE_TICKS} ticks after the end, with at most {MAX_ACTIONS_AFTER_DONE} actions sent after it.", "",
           ("**Stopped at level %s.**" % result["stopped_at"]) if result["stopped_at"] else "**All levels passed.**", "",
           "| level | game | genre | result | wins | mean score | mean steps | actions after done (max) | dropped by card |",
           "|---|---|---|---|---|---|---|---|---|"]
    for lv in result["levels"]:
        out.append(f"| {lv['level']} | {lv['game']} | {lv['genre']} | {'pass' if lv['passed'] else 'FAIL'} | "
                   f"{lv['wins']}/{len(lv['episodes'])} | {lv['mean_score']} | {lv['mean_steps']} | "
                   f"{lv['max_actions_after_done']} | {lv['dropped']} |")
    out.append("")
    return "\n".join(out)


def scripted_policies() -> Callable[[Level, int], Policy]:
    """The test oracles (tests/test_arena_pack_oracles.py). They read the true game state, so they are only a
    baseline, never part of the buddy."""
    tests = HERE / "tests"
    if not (tests / "test_arena_pack_oracles.py").exists():
        raise SystemExit("The scripted brain uses the test oracles in tests/, which this copy doesn't include.")
    sys.path.insert(0, str(tests))
    from test_arena_pack_oracles import SCRIPTED
    return lambda level, seed: SCRIPTED[level.key]


def random_policies() -> Callable[[Level, int], Policy]:
    def make(level: Level, seed: int) -> Policy:
        rng, keys = random.Random(seed), sorted(level.card_keys())

        def press_or_click(game):
            if game.done:
                return None
            key = rng.choice(keys)
            return ("click", rng.randrange(1000), rng.randrange(1000)) if key == "click" else key
        return press_or_click
    return make


class NavPolicy:
    """The learned-navigation brain for one episode: it looks at the picture and the HUD only (never the game's
    state), learns from every step and asks the NAVIGATE skill for the next key."""

    needs_pixels = True
    ai_calls = 0

    def __init__(self, navigator: Navigator, max_steps: int | None = None):
        self.navigator = navigator
        self.max_steps = max_steps  # stop thinking after this many steps (tests keep runs short)
        self.steps = 0
        self.last: tuple | None = None  # (picture, key) of the previous step

    def __call__(self, game):
        if self.max_steps is not None and self.steps >= self.max_steps:
            return None
        frame = np.asarray(game.render())
        hud = PackWorld(game).hud()
        if self.last is None and self.steps == 0:
            self.navigator.start(frame, hud)
        elif self.last is not None:
            self.navigator.learn(self.last[0], self.last[1], frame, hud, game.done)
        if game.done:
            self.last = None
            return None
        # Only what was learned decides. The test oracles read the game's true state; they are the
        # separate "scripted" baseline brain and must never steer this one. [RATCHET-NO-ORACLE-IN-NAV]
        advice = self.navigator.advise(frame)
        action = advice.actions[0] if advice.actions else None
        self.last = (frame, action)
        self.steps += 1
        return action


def nav_policies(practice_episodes: int = 3, practice_seed: int = 100, practice_cap: int = 600,
                 max_steps: int | None = None) -> Callable[[Level, int], Policy]:
    """Learned navigation: per level, practise on training seeds first, then play every ladder seed as a level it
    has never seen (only looks and moves carry over). Practice follows a curriculum (LEGENT / Voyager): it picks
    the next training seed by need, learns from the navigator's own surprise, and stops early once the level is
    mastered. `make.curricula` keeps each level's curriculum for the report."""
    practised: dict[str, Navigator] = {}
    curricula: dict[str, CurriculumScheduler] = {}

    def make(level: Level, seed: int) -> Policy:
        if level.key not in practised:
            keys = sorted(k for k in level.card_keys() if k != "click") or sorted(level.card_keys())
            navigator = Navigator(keys, surprise=SurpriseModule())
            if practice_episodes and keys and keys != ["click"]:  # a click-only game has no keys to practise with
                curricula[level.key] = curriculum = CurriculumScheduler(
                    levels=[level.number], seeds=tuple(range(practice_seed, practice_seed + PRACTICE_SEEDS)))
                practice(PackWorld(level.game_class(seed=practice_seed)), navigator, episodes=practice_episodes,
                         cap=practice_cap, seed=practice_seed, save=False, curriculum=curriculum)
            practised[level.key] = navigator
        return NavPolicy(practised[level.key].for_new_level(), max_steps)
    make.curricula = curricula
    return make



BRAINS = {"scripted": scripted_policies, "random": random_policies, "nav": nav_policies}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PlayFellow game ladder")
    parser.add_argument("--brain", choices=sorted(BRAINS), default="scripted")
    parser.add_argument("--from", dest="from_level", type=int, default=1)
    parser.add_argument("--to", dest="to_level", type=int, default=99, help="last level to play")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="how many fixed seeds per level (0, 1, 2, …)")
    parser.add_argument("--realtime", action="store_true", help="the game clock keeps running while the brain thinks")
    parser.add_argument("--think-ms", type=float, default=250.0, help="game time each decision costs in --realtime")
    parser.add_argument("--out", type=Path, default=HERE / "ladder")
    args = parser.parse_args(argv)
    result = run_ladder(BRAINS[args.brain](), seeds=tuple(range(args.seeds)), from_level=args.from_level,
                        to_level=args.to_level, think_ms=args.think_ms if args.realtime else None)
    result.update(brain=args.brain, date=date.today().isoformat())
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"-realtime{args.think_ms:g}ms" if args.realtime else ""
    stem = args.out / f"{result['date']}-{args.brain}{suffix}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(result), encoding="utf-8")
    passed = sum(lv["passed"] for lv in result["levels"])
    print(f"{args.brain}: {passed}/{len(result['levels'])} levels passed"
          + (f", stopped at level {result['stopped_at']}" if result["stopped_at"] else "") + f" → {stem.with_suffix('.md')}")
    return 0 if result["stopped_at"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
