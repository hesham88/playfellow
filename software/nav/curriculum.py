"""Adaptive level/seed scheduler prioritizing levels with high surprise and low win rate.
/ جدولة تكيفية للمستويات تركز على المستويات ذات المفاجأة العالية ومعدل الفوز المنخفض."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class LevelStats:
    level: int
    episodes: int = 0
    wins: int = 0
    surprises: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    seed_counts: dict[int, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.episodes) if self.episodes > 0 else 0.0

    @property
    def mean_surprise(self) -> float:
        return (sum(self.surprises) / len(self.surprises)) if self.surprises else 1.0

    @property
    def mean_steps(self) -> float:
        return (sum(self.steps) / len(self.steps)) if self.steps else 0.0


class CurriculumScheduler:
    """Adaptive level and seed scheduler.
    
    Levels with high surprise (agent's dynamics model is still inaccurate)
    and low win rate (game not yet mastered) are prioritized for practice.
    Mastered levels (high win rate, low surprise) taper in priority.
    """

    def __init__(
        self,
        levels: Sequence[int] = tuple(range(1, 16)),
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        weight_surprise: float = 0.5,
        weight_failure: float = 0.4,
        weight_exploration: float = 0.2,
        seed: int = 0,
    ):
        self.levels = list(levels)
        self.seeds = list(seeds)
        self.weight_surprise = weight_surprise
        self.weight_failure = weight_failure
        self.weight_exploration = weight_exploration
        self.rng = random.Random(seed)
        self.stats: dict[int, LevelStats] = {lv: LevelStats(level=lv) for lv in self.levels}

    def record_episode(
        self,
        level: int,
        seed: int,
        won: bool,
        surprise: float = 0.0,
        steps: int = 0,
    ) -> None:
        """Records the outcome of one practice episode on (level, seed)."""
        if level not in self.stats:
            self.stats[level] = LevelStats(level=level)
        st = self.stats[level]
        st.episodes += 1
        if won:
            st.wins += 1
        st.surprises.append(float(surprise))
        if steps > 0:
            st.steps.append(steps)
        st.seed_counts[seed] = st.seed_counts.get(seed, 0) + 1

    def priority(self, level: int) -> float:
        """Calculates scheduling priority for a level. Higher = needs more practice."""
        st = self.stats.get(level)
        if st is None or st.episodes == 0:
            # Unplayed levels get maximal priority + initial exploration bonus
            return 2.0 + self.weight_exploration

        failure_rate = 1.0 - st.win_rate
        novelty = st.mean_surprise
        exploration = self.weight_exploration / math.sqrt(st.episodes + 1)

        return float(
            self.weight_surprise * novelty
            + self.weight_failure * failure_rate
            + exploration
        )

    def next_task(self) -> tuple[int, int]:
        """Chooses the next (level, seed) with highest priority."""
        if not self.levels:
            raise ValueError("No levels configured in curriculum")

        # Pick level with highest priority (break ties randomly using rng)
        ranked = sorted(
            self.levels,
            key=lambda lv: (self.priority(lv), self.rng.random()),
            reverse=True,
        )
        chosen_level = ranked[0]

        # Pick seed for that level (prefer seeds with least attempts)
        st = self.stats[chosen_level]
        ranked_seeds = sorted(
            self.seeds,
            key=lambda s: (st.seed_counts.get(s, 0), self.rng.random()),
        )
        chosen_seed = ranked_seeds[0] if ranked_seeds else 0

        return chosen_level, chosen_seed

    def is_mastered(
        self,
        level: int,
        win_rate_threshold: float = 0.8,
        surprise_threshold: float = 0.25,
        min_episodes: int = 3,
    ) -> bool:
        """True if level has enough attempts, high win rate, and low surprise."""
        st = self.stats.get(level)
        if not st or st.episodes < min_episodes:
            return False
        return (
            st.win_rate >= win_rate_threshold
            and st.mean_surprise <= surprise_threshold
        )

    def all_mastered(
        self,
        win_rate_threshold: float = 0.8,
        surprise_threshold: float = 0.25,
        min_episodes: int = 3,
    ) -> bool:
        return all(
            self.is_mastered(lv, win_rate_threshold, surprise_threshold, min_episodes)
            for lv in self.levels
        )

    def summary(self) -> dict[int, dict]:
        return {
            lv: {
                "episodes": st.episodes,
                "win_rate": round(st.win_rate, 3),
                "mean_surprise": round(st.mean_surprise, 3),
                "priority": round(self.priority(lv), 3),
                "mastered": self.is_mastered(lv),
            }
            for lv, st in self.stats.items()
        }
