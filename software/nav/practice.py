"""Getting better with practice: replay the game, learn which ways pay off. / التحسّن بالتدريب.

Like practising a level again and again: each try adds to the map, and the buddy works out how good each place
is ("from here, how much treasure can I still get, minus the steps it costs?"). It does this by thinking over its
own map many times (value iteration on the learned model — "Dyna"), without pressing any more keys.

Every episode is measured: steps to win, score, and (in tests and the ladder evaluator only, which know the
true best route) route efficiency = best possible steps / steps taken.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nav.map import NavMap
from nav.planner import Navigator
from nav.curriculum import CurriculumScheduler
from nav.skills import SkillLibrary

STEP_COST = 0.05  # every step costs a little, so shorter ways are worth more
GAMMA = 0.97
TREASURE = 1.0  # value of a HUD change that wasn't the end
GOOD_END = 2.0
BAD_END = -2.0
SWEEPS = 80


def values(nav: NavMap, collected: set[str] | None = None, warm: dict[str, float] | None = None,
           sweeps: int = SWEEPS) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Value iteration on the learned map: V(place) and Q(place, action) from what each arrow did.

    Arrows keep all their outcomes with their shares (the same key can lead to different places when the game has
    things you can't see, like how fast you are falling). Treasure already taken this episode is worth nothing.
    `warm`: the values from last time (one step changes the map only a little, so a few sweeps are enough)."""
    collected = nav.collected if collected is None else collected
    model: dict[str, dict[str, list[tuple[str, float, float, bool]]]] = {}
    for place, acts in nav.edges.items():
        for action, edge in acts.items():
            total = sum(edge.outcomes.values())
            if not total:
                continue
            outcomes = []
            for where, count in edge.outcomes.items():
                share = count / total
                if where == "blocked":
                    outcomes.append((place, share, -STEP_COST, False))
                    continue
                gain = -STEP_COST
                treasure = nav.places.get(where)
                if treasure and treasure.rewards and where not in collected and where != place:
                    gain += TREASURE * min(1.0, treasure.rewards / max(edge.tries, 1) + 0.5)
                outcomes.append((where, share, gain, False))
            if edge.good_end:
                outcomes = [(w, s, g + GOOD_END * edge.good_end / edge.tries, True) for w, s, g, _ in outcomes]
            if edge.bad_end:
                outcomes = [(w, s, BAD_END, True) for w, s, _, _ in outcomes]
            model.setdefault(place, {})[action] = outcomes
    warm = warm or {}
    value = {p: warm.get(p, 0.0)
             for p in set(model) | {w for acts in model.values() for outs in acts.values() for w, *_ in outs}}
    q: dict[str, dict[str, float]] = {}
    for _ in range(sweeps):
        change = 0.0
        for place, acts in model.items():
            q[place] = {a: sum(s * (g + (0.0 if end else GAMMA * value.get(w, 0.0))) for w, s, g, end in outs)
                        for a, outs in acts.items()}
            best = max(q[place].values())
            change = max(change, abs(best - value[place]))
            value[place] = best
        if change < 1e-4:
            break
    return value, q


@dataclass
class Episode:
    steps: int
    won: bool
    ended: bool
    score: float
    events: int  # HUD changes seen (treasure taken, doors opened, rooms entered…)
    efficiency: float | None = None  # oracle steps / steps (only when an oracle is given)
    kinds: dict[str, int] = field(default_factory=dict)  # how often each kind of advice was followed
    surprise: float = 1.0  # how unpredictable the moves were, 0 (as expected) to 1 (all new): what the curriculum ranks by
    curiosity: float = 0.0  # summed intrinsic reward (surprise plus empowerment) the moves earned


def play_episode(world, navigator: Navigator, cap: int = 600, goal: str = "all", seed: int = 0,
                 oracle_steps: int | None = None) -> Episode:
    """One episode on any reset()/step() world whose info carries the HUD numbers ("hud")."""
    frame = world.reset(seed)
    hud = _hud(world, None)
    navigator.start(frame, hud)
    events, kinds, won, ended, score, steps = 0, {}, False, False, 0.0, 0
    for steps in range(1, cap + 1):
        advice = navigator.advise(frame, goal)
        if not advice.actions:
            advice.actions = [navigator.nav.me.probe_action(navigator.planner.actions)]
            advice.kind = "probe"
        action = advice.actions[0]
        kinds[advice.kind] = kinds.get(advice.kind, 0) + 1
        after, reward, done, truncated, info = world.step(action)
        new_hud = _hud(world, info)
        events += int(new_hud != hud)
        score += float(reward)
        navigator.learn(frame, action, after, new_hud, done)
        frame, hud = after, new_hud
        if done or truncated:
            ended = done
            won = bool(info.get("won", done))  # for the scorecard only; the buddy never sees it
            break
    efficiency = (oracle_steps / steps) if (oracle_steps and won) else (0.0 if oracle_steps else None)
    return Episode(steps, won, ended, score, events, efficiency, kinds,
                   surprise=getattr(navigator, "mean_surprise", 1.0), curiosity=getattr(navigator, "episode_curiosity", 0.0))


def practice(world, navigator: Navigator, episodes: int = 5, cap: int = 600, seed: int = 0,
             oracle_steps: int | None = None, save: bool = True,
             curriculum: CurriculumScheduler | None = None) -> list[Episode]:
    """Practise up to `episodes` times, learning all along; saves the map after each one.

    Without a curriculum every episode replays `seed`. With one (the LEGENT / Voyager idea), the curriculum picks each
    episode's seed by need (surprising and unwon first, least practised first), learns from the navigator's own
    surprise, and practice stops as soon as everything is mastered."""
    results = []
    for _ in range(episodes):
        level, play_seed = curriculum.next_task() if curriculum is not None else (None, seed)
        ep = play_episode(world, navigator, cap, seed=play_seed, oracle_steps=oracle_steps)
        results.append(ep)
        if curriculum is not None:
            curriculum.record_episode(level, play_seed, ep.won, surprise=ep.surprise, steps=ep.steps)
        if save:
            navigator.save()
        if curriculum is not None and curriculum.all_mastered():
            break
    return results


def _hud(world, info) -> dict:
    if info and "hud" in info:
        return dict(info["hud"])
    if hasattr(world, "hud"):
        return dict(world.hud())
    return {}


class PackWorld:
    """An arena-pack game as a practice world. Its HUD is what its banner shows: the score and the progress line
    (in live play the HUD comes from the brain's screen reading instead)."""

    def __init__(self, game):
        self.game = game

    def reset(self, seed: int = 0) -> np.ndarray:
        return self.game.reset(seed)

    def step(self, action):
        frame, reward, done, truncated, info = self.game.step(action)
        info = dict(info)
        info["hud"] = self.hud()
        return frame, reward, done, truncated, info

    def hud(self) -> dict:
        return {"score": self.game.score, "progress": self.game.progress_text()}  # exactly what the banner shows
