"""The Python reference of the native graph search pf_graph4_astar: its specification and its test oracle.

It was in nav/graph4.py until 2026-10-23 (D23), but only the tests used it: the planner asks the native search and
falls back to its own Python search over the road list (nav.planner.Planner.route). The tie rule it follows, which the
native code must follow exactly, is written at the top of nav/graph4.py."""
from __future__ import annotations

import heapq

import numpy as np

from nav.graph4 import DELTAS, F32


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
