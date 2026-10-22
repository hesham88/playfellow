"""The learned roads as numbers on a grid, and a search that runs on them. / الطرق المتعلَّمة كأرقام على شبكة.

Think of a street map where every square has four little signs on it: "going up from here costs 1.0", "going right
costs 1.2", and so on, with a blank sign where there is no road. Such a map can hold ANY set of roads between
neighbouring squares: one-way streets, roads that cost different amounts, roads nobody has walked yet. (An older idea,
"every square has one price", does not fit a real learned map; I measured it on exploring mazes and it was never true.)

`EdgeGrid` keeps those signs up to date one road at a time, so asking for a route does not mean going through every
road again. `graph4_astar_py` is the search, written to be copied exactly by the native (Rust) version: the same
steps in the same order with the same float32 arithmetic and the same tie rule, so both return the same path.

The tie rule (the native code must follow it exactly):
  * the open list is a min-heap on (f, h, seq): smaller f first; equal f: smaller h first (that prefers the larger g);
    equal again: smaller seq first, where seq counts up by one on every push;
  * a cell is "closed" the first time it is popped, and stale pops are skipped;
  * the four directions are tried in the order up, right, down, left;
  * a neighbour is updated only when the new cost is STRICTLY smaller;
  * f = g + h and g = g + edge cost, both in float32; h = hmin * (|dr| + |dc|), hmin = the smallest road cost in the grid.
"""
from __future__ import annotations

import heapq
from contextlib import contextmanager
from typing import Callable

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
SIGNS = 4  # one price sign per direction on every square
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # (row, col) for up, right, down, left
F32 = np.float32


def graph4_astar_py(costs: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """The cheapest way from `start` to `goal` (both (row, col)) on a (rows, cols, 4) array of road costs, where
    costs[r, c, d] is the cost of leaving (r, c) in direction d and a value <= 0 or not finite means no road.

    Returns (path, cost): the (row, col) cells including both ends and the float32 total, or ([], 0.0) when there is no
    way, or None when the arguments are unusable (the same three answers the native function gives)."""
    costs = np.ascontiguousarray(costs, dtype=np.float32)
    if costs.ndim != 3 or costs.shape[2] != 4 or costs.shape[0] == 0 or costs.shape[1] == 0:
        return None
    rows, cols = costs.shape[:2]
    (sr, sc), (gr, gc) = (int(start[0]), int(start[1])), (int(goal[0]), int(goal[1]))
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return None
    if (sr, sc) == (gr, gc):
        return [(sr, sc)], 0.0
    roads = np.isfinite(costs) & (costs > 0)
    if not roads.any():
        return [], 0.0
    hmin = F32(costs[roads].min())

    def guess(r: int, c: int):
        return F32(hmin * F32(abs(r - gr) + abs(c - gc)))

    best = {(sr, sc): F32(0.0)}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    h0 = guess(sr, sc)
    heap = [(float(h0), float(h0), 0, (sr, sc))]
    seq = 1
    while heap:
        _, _, _, here = heapq.heappop(heap)
        if here in closed:
            continue
        closed.add(here)
        if here == (gr, gc):
            path = [here]
            while path[-1] != (sr, sc):
                path.append(came[path[-1]])
            return path[::-1], float(best[here])
        r, c = here
        g = best[here]
        for d, (dr, dc) in enumerate(DELTAS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in closed:
                continue
            step = costs[r, c, d]
            if not (np.isfinite(step) and step > 0):
                continue
            tentative = F32(g + step)
            if tentative < best.get((nr, nc), F32(np.inf)):
                best[(nr, nc)] = tentative
                came[(nr, nc)] = here
                h = guess(nr, nc)
                heapq.heappush(heap, (float(F32(tentative + h)), float(h), seq, (nr, nc)))
                seq += 1
    return [], 0.0


def direction_of(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    """Which of up/right/down/left takes (x, y) square `a` to its neighbour `b` (None if they are not neighbours).
    Squares are (x, y), so up is y - 1."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(0, -1): UP, (1, 0): RIGHT, (0, 1): DOWN, (-1, 0): LEFT}.get((dx, dy))


class _Scene:
    """The four-signs-per-square array of one room, grown as the map grows. Squares are (x, y); the array is
    indexed [y - oy, x - ox, direction]."""

    def __init__(self):
        self.ox = self.oy = 0
        self.costs = np.zeros((0, 0, SIGNS), np.float32)

    ROOM = 8  # squares of spare room added on the side that grows, so growing is rare

    def cover(self, x: int, y: int) -> None:
        """Make sure square (x, y) has a place in the array."""
        if self.costs.size == 0:
            self.ox, self.oy = x - self.ROOM, y - self.ROOM
            self.costs = np.zeros((2 * self.ROOM + 1, 2 * self.ROOM + 1, 4), np.float32)
            return
        rows, cols = self.costs.shape[:2]
        new_oy = y - self.ROOM if y < self.oy else self.oy
        new_ox = x - self.ROOM if x < self.ox else self.ox
        new_bottom = y + self.ROOM if y > self.oy + rows - 1 else self.oy + rows - 1
        new_right = x + self.ROOM if x > self.ox + cols - 1 else self.ox + cols - 1
        if (new_oy, new_ox, new_bottom, new_right) == (self.oy, self.ox, self.oy + rows - 1, self.ox + cols - 1):
            return
        grown = np.zeros((new_bottom - new_oy + 1, new_right - new_ox + 1, 4), np.float32)
        grown[self.oy - new_oy:self.oy - new_oy + rows, self.ox - new_ox:self.ox - new_ox + cols] = self.costs
        self.costs, self.oy, self.ox = grown, new_oy, new_ox

    def cell(self, x: int, y: int) -> tuple[int, int]:
        return y - self.oy, x - self.ox


class EdgeGrid:
    """All learned (and, for a moment, guessed) one-square roads of every room, as `_Scene` arrays.

    `mark(place, action)` says a road may have changed; `sync(...)` applies exactly those, so keeping the grid right costs
    one small update per step. Roads that do not fit (a jump to another room or far away: doors, teleports) are counted in
    `odd`; while any exist the grid is not an exact picture of the roads and `exact` is False."""

    def __init__(self, parse: Callable[[str], tuple[int, tuple[int, int]]]):
        self._parse = parse
        self.scenes: dict[int, _Scene] = {}
        self.slots: dict[tuple[int, int, int, int], dict[str, tuple[float, str, float]]] = {}  # (scene, x, y, d) -> action -> (cost, to, share)
        self._where: dict[tuple[str, str], tuple[int, int, int, int]] = {}  # (place, action) -> its slot
        self.odd: set[tuple[str, str]] = set()
        self.dirty: set[tuple[str, str]] = set()
        self.syncs = 0

    # ---- keeping it right
    def mark(self, place: str, action: str) -> None:
        self.dirty.add((place, action))

    def mark_all(self, edges) -> None:
        for place, acts in edges.items():
            for action in acts:
                self.dirty.add((place, action))

    def clear(self) -> None:
        self.scenes.clear()
        self.slots.clear()
        self._where.clear()
        self.odd.clear()
        self.dirty.clear()

    def sync(self, road_of: Callable[[str, str], tuple[str, float, float, bool] | None]) -> None:
        """Apply every road marked as changed. `road_of(place, action)` gives (to, cost, share, ends) or None."""
        while self.dirty:
            place, action = self.dirty.pop()
            self._apply(place, action, road_of(place, action))
            self.syncs += 1

    def _slot_array(self, slot) -> None:
        scene, x, y, d = slot
        sc = self.scenes[scene]
        entries = self.slots.get(slot)
        r, c = sc.cell(x, y)
        sc.costs[r, c, d] = min(cost for cost, _, _ in entries.values()) if entries else 0.0

    def _apply(self, place: str, action: str, road) -> None:
        key = (place, action)
        old = self._where.pop(key, None)
        if old is not None:
            entries = self.slots[old]
            entries.pop(action, None)
            if not entries:
                del self.slots[old]
            self._slot_array(old)
        self.odd.discard(key)
        if road is None or road[3]:  # no road, or one that ends the game (never walked through)
            return
        to, cost, share, _ = road
        scene, (x, y) = self._parse(place)
        scene2, (x2, y2) = self._parse(to)
        d = direction_of((x, y), (x2, y2)) if scene == scene2 else None
        if d is None or not (cost > 0 and np.isfinite(cost)):
            self.odd.add(key)
            return
        sc = self.scenes.setdefault(scene, _Scene())
        sc.cover(x, y)
        sc.cover(x2, y2)
        slot = (scene, x, y, d)
        self.slots.setdefault(slot, {})[action] = (float(cost), to, float(share))
        self._where[key] = slot
        self._slot_array(slot)

    @property
    def exact(self) -> bool:
        """True when every learned road is a one-square step in one room, so the arrays hold them all."""
        return not self.odd

    # ---- guessed roads, for the length of one question
    @contextmanager
    def overlay(self, scene: int, guessed: list[tuple[str, str, float, float]]):
        """Write guessed roads (place, to, cost, share, action) into the array for the length of the `with`, then put back
        what was there. The yielded object has `.added` ({slot: (cost, to, share, action)} for the guesses that were
        written) and `.skipped` (how many guesses were NOT one square in this room and so could not be written).

        A guess never replaces a cheaper road already in the slot (the Python search would take the cheaper one)."""
        sc = self.scenes.setdefault(scene, _Scene())
        undo, added, skipped = [], {}, 0
        for place, to, cost, share, action in guessed:
            s1, (x, y) = self._parse(place)
            s2, (x2, y2) = self._parse(to)
            d = direction_of((x, y), (x2, y2)) if s1 == s2 == scene else None
            if d is None or not (cost > 0 and np.isfinite(cost)):
                skipped += 1
                continue
            sc.cover(x, y)
            sc.cover(x2, y2)
            r, c = sc.cell(x, y)
            current = float(sc.costs[r, c, d])
            if current == 0.0 or cost < current:
                undo.append((r, c, d, sc.costs[r, c, d]))
                sc.costs[r, c, d] = cost
                added[(scene, x, y, d)] = (float(cost), to, float(share), action)
        view = type("Overlay", (), {})()
        view.added, view.skipped = added, skipped
        try:
            yield view
        finally:
            for r, c, d, value in reversed(undo):
                sc.costs[r, c, d] = value
