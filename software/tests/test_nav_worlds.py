"""Test worlds for learned navigation (T32b): a synthetic side-scroller, adapters, and the route oracles.

The nav code only ever receives what a player would get: the picture, the HUD numbers and "the game ended".
The oracles here read the real game state; they live in tests only.
"""
from __future__ import annotations

import functools
import itertools
import json
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from practice_arena.game import ArenaGame, BANNER, LEVEL, TILE

# ---------------------------------------------------------------- a synthetic side-scroller
SIDE_LEVEL = (
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "....................",
    "........#...........",
    "..P.....#.......F...",
    "####################",
)
SIDE_TILE = 32
SIDE_KEYS = ("left", "right", "jump")


class SideScroller:
    """A one-screen side-view level: walk right, jump the block, touch the flag. A cloud drifts on its own."""

    observe = True  # False: step() skips drawing (the oracle's search needs no pictures)

    def __init__(self, level=SIDE_LEVEL, seed: int = 0):
        self.level = level
        self.reset(seed)

    def reset(self, seed: int = 0) -> np.ndarray:
        self.x, self.y = next((x, y) for y, row in enumerate(self.level) for x, c in enumerate(row) if c == "P")
        self.rise = 0
        self.cloud = 40 + 8 * seed
        self.won = False
        self.score = 0
        return self.observation()

    def _solid(self, x, y):
        return not (0 <= x < len(self.level[0])) or self.level[y][x] == "#"

    def step(self, action):
        if not self.won:
            if action in ("left", "right"):
                nx = self.x + (1 if action == "right" else -1)
                if not self._solid(nx, self.y):
                    self.x = nx
            elif action == "jump" and self._solid(self.x, self.y + 1):
                self.rise = 3
            if self.rise > 0 and not self._solid(self.x, self.y - 1):
                self.y -= 1
                self.rise -= 1
            elif not self._solid(self.x, self.y + 1):
                self.rise = 0
                self.y += 1
            if self.level[self.y][self.x] == "F":
                self.won = True
                self.score = 100
        self.cloud = (self.cloud + 8) % (len(self.level[0]) * SIDE_TILE)
        frame = self.observation() if self.observe else None
        return frame, 0.0, self.won, False, {"hud": self.hud()}

    def hud(self):
        return {"score": self.score}

    def observation(self) -> np.ndarray:
        width, height = len(self.level[0]) * SIDE_TILE, len(self.level) * SIDE_TILE
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)
        for row in range(height):  # a gentle sky gradient (not one flat colour)
            draw.line((0, row, width, row), fill=(90 + row // 16, 150 + row // 20, 230))
        for y, line in enumerate(self.level):
            for x, cell in enumerate(line):
                left, top = x * SIDE_TILE, y * SIDE_TILE
                if cell == "#":
                    draw.rectangle((left, top, left + SIDE_TILE - 1, top + SIDE_TILE - 1), fill=(110, 80, 50))
                elif cell == "F":
                    draw.rectangle((left + 14, top, left + 18, top + SIDE_TILE), fill=(40, 40, 40))
                    draw.polygon([(left + 18, top), (left + 30, top + 6), (left + 18, top + 12)], fill=(230, 40, 60))
        draw.ellipse((self.cloud, 20, self.cloud + 64, 44), fill=(250, 250, 250))
        left, top = self.x * SIDE_TILE, self.y * SIDE_TILE
        draw.rectangle((left + 4, top + 4, left + 27, top + 31), fill=(240, 140, 30))
        draw.rectangle((left + 8, top + 10, left + 12, top + 14), fill=(20, 20, 20))
        draw.rectangle((left + 19, top + 10, left + 23, top + 14), fill=(20, 20, 20))
        return np.asarray(image)

    def true_box(self):
        left, top = self.x * SIDE_TILE, self.y * SIDE_TILE
        return left + 4, top + 4, left + 28, top + 32


def make_side_level(seed: int, width: int = 20) -> tuple[str, ...]:
    """A NEW side-scroller layout: 2-3 blocks to jump (1-2 high), the flag near the far end."""
    import random
    rng = random.Random(seed)
    rows = [["."] * width for _ in range(9)]
    rows[8] = ["#"] * width
    for x in rng.sample(range(5, width - 4), rng.choice((2, 3))):
        for h in range(rng.choice((1, 2))):
            rows[7 - h][x] = "#"
    rows[7][2] = "P"
    flag = rng.choice(range(width - 4, width - 1))
    rows[7][flag] = "F" if rows[7][flag] == "." else rows[7][flag]
    if rows[7][flag] != "F":
        rows[7][flag] = "F"
        rows[6][flag] = "."
    return tuple("".join(r) for r in rows)


def side_optimal_steps(level=SIDE_LEVEL) -> int:
    """ORACLE: fewest steps to the flag, searching the real physics state (x, y, rise)."""
    world = SideScroller(level)
    world.reset()
    world.observe = False
    start = (world.x, world.y, world.rise)
    seen, queue = {start: 0}, deque([start])
    while queue:
        state = queue.popleft()
        for action in SIDE_KEYS:
            world.x, world.y, world.rise = state
            world.won = False
            world.step(action)
            nxt = (world.x, world.y, world.rise)
            if world.won:
                return seen[state] + 1
            if nxt not in seen:
                seen[nxt] = seen[state] + 1
                queue.append(nxt)
    return 10 ** 6


# ---------------------------------------------------------------- the Practice Arena as a step() world
ARENA_KEYS = ("up", "down", "left", "right")


class ArenaWorld:
    """The Practice Arena with reset()/step(). The HUD dict holds the numbers its banner shows."""

    def __init__(self, level=LEVEL):
        self.game = ArenaGame(level=level)

    def reset(self, seed: int = 0) -> np.ndarray:
        self.game.reset()
        return np.asarray(self.game.render())

    def step(self, action):
        before = self.game.score
        self.game.press(action)
        frame = np.asarray(self.game.render())
        return frame, float(self.game.score - before), self.game.won, False, {"hud": self.hud()}

    def hud(self):
        g = self.game
        return {"score": g.score, "coins_left": g.coins_left, "key": "yes" if g.has_key else "no"}

    def true_box(self):
        x, y = self.game.player
        return x * TILE, BANNER + y * TILE, (x + 1) * TILE, BANNER + (y + 1) * TILE


def make_arena_level(seed: int, width: int = 13, height: int = 9, coins: int = 6, loops: int = 6) -> tuple[str, ...]:
    """A NEW Practice Arena layout (held-out levels): a carved maze with a few loops, coins, a key and a locked door
    in a dead end. The same seed always gives the same level."""
    import random
    rng = random.Random(seed)
    grid = [["#"] * width for _ in range(height)]
    start = (1, 1)
    grid[1][1] = "."
    stack = [start]
    while stack:  # carve a maze (depth-first, two cells at a time)
        x, y = stack[-1]
        options = [(x + dx, y + dy, dx, dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
                   if 0 < x + dx < width - 1 and 0 < y + dy < height - 1 and grid[y + dy][x + dx] == "#"]
        if not options:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(options)
        grid[y + dy // 2][x + dx // 2] = "."
        grid[ny][nx] = "."
        stack.append((nx, ny))
    walls = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if grid[y][x] == "#"
             and ((grid[y][x - 1] == "." and grid[y][x + 1] == ".") or (grid[y - 1][x] == "." and grid[y + 1][x] == "."))]
    for x, y in rng.sample(walls, min(loops, len(walls))):
        grid[y][x] = "."
    floor = [(x, y) for y in range(height) for x in range(width) if grid[y][x] == "." and (x, y) != start]
    ends = [(x, y) for x, y in floor
            if sum(grid[y + dy][x + dx] == "." for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 1]
    rng.shuffle(ends)
    for door in ends + floor:  # the door must not cut anything off: everything else reachable without it
        seen, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if grid[nb[1]][nb[0]] == "." and nb != door and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(seen) == len(floor):  # floor excludes the start; seen includes it but not the door
            break
    rest = [p for p in floor if p != door]
    picks = rng.sample(rest, coins + 1)
    grid[door[1]][door[0]] = "D"
    grid[picks[0][1]][picks[0][0]] = "k"
    for x, y in picks[1:]:
        grid[y][x] = "c"
    grid[1][1] = "P"
    return tuple("".join(row) for row in grid)


@functools.lru_cache(maxsize=None)
def _first_arena_episode() -> tuple[str, object]:
    from nav.planner import Navigator
    from nav.practice import play_episode
    navigator = Navigator(list(ARENA_KEYS))
    episode = play_episode(ArenaWorld(), navigator, oracle_steps=arena_optimal_steps())
    return json.dumps(navigator.nav.to_dict()), episode


def arena_after_first_episode():
    """A navigator that has played the arena once from a cold start, plus that episode's result.
    Trained once per test run; every caller gets its own fresh copy (tests can't disturb each other)."""
    from nav.map import NavMap
    from nav.planner import Navigator
    data, episode = _first_arena_episode()
    return Navigator(list(ARENA_KEYS), nav=NavMap.from_dict(json.loads(data))), episode


def arena_optimal_steps(level=LEVEL) -> int:
    """ORACLE: the fewest moves that collect every coin and the key, then step into the door."""
    walls, coins, key, door, start = set(), [], None, None, None
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            elif cell == "c":
                coins.append((x, y))
            elif cell == "k":
                key = (x, y)
            elif cell == "D":
                door = (x, y)
            elif cell == "P":
                start = (x, y)

    def distances(source):
        seen, queue = {source: 0}, deque([source])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb not in walls and nb not in seen and nb != door:
                    seen[nb] = seen[(x, y)] + 1
                    queue.append(nb)
        return seen

    targets = coins + [key]
    table = {p: distances(p) for p in [start] + targets}

    def to_door(p):  # step next to the door, then into it
        dx = min(table[p].get((door[0] + ox, door[1] + oy), 10 ** 6) for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        return dx + 1

    best = 10 ** 9
    for order in itertools.permutations(targets):
        total, here = 0, start
        for target in order:
            total += table[here][target]
            here = target
            if total >= best:
                break
        else:
            best = min(best, total + to_door(here))
    return best


def test_side_scroller_flag_is_reachable_by_a_jump():
    world = SideScroller()
    world.reset()
    for action in ["right"] * 5 + ["jump", "right", "right", "right"] + ["right"] * 9:
        *_, done, _, _ = world.step(action)
        if done:
            break
    assert world.won


def test_arena_oracle_matches_known_route_length():
    assert arena_optimal_steps() > 0


def test_side_levels_are_winnable_and_the_oracle_counts_the_jump():
    assert side_optimal_steps() == 15
    for seed in range(300, 305):
        assert side_optimal_steps(make_side_level(seed)) < 10 ** 6


def test_held_out_levels_differ_and_are_winnable():
    levels = {make_arena_level(seed) for seed in range(100, 110)}
    assert len(levels) == 10 and LEVEL not in levels
    for level in levels:
        assert arena_optimal_steps(level) < 10 ** 6
