"""Writes tests/data/graph4_vectors.npz: inputs and the exact answers of the reference search (nav.graph4.graph4_astar_py),
for the native (Rust) pf_graph4_astar to be compared against bit for bit. Run from the software folder:

    python tests/make_graph4_vectors.py

File layout (every array name ends in the case number i = 0 .. count-1):
    count                  int32 scalar
    costs_i                float32 (rows, cols, 4), index [r, c, d], d = 0 up, 1 right, 2 down, 3 left; <= 0 = no road
    start_i, goal_i        int32 (2,) = (row, col)
    status_i               int32 scalar: 0 = path found, 4 = no path
    path_i                 int32 (n, 2) = (row, col) cells including start and goal (empty when status is 4)
    cost_i                 float32 scalar: the total (0.0 for no path or start == goal)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import heapq  # noqa: E402

from nav import graph4  # noqa: E402
from nav.graph4 import DELTAS, graph4_astar_py  # noqa: E402

RIGHT = 1
OUT = Path(__file__).resolve().parent / "data" / "graph4_vectors.npz"


def lumpy(rng, rows, cols, wall=0.18, one_way=0.1, unit=False):
    costs = np.zeros((rows, cols, 4), np.float32)
    open_ = rng.random((rows, cols)) > wall
    for r in range(rows):
        for c in range(cols):
            for d, (dr, dc) in enumerate(DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and open_[r, c] and open_[nr, nc] and rng.random() > one_way:
                    costs[r, c, d] = 1.0 if unit else 1.0 + int(rng.integers(0, 5)) * 0.5
    return costs


class _LargerHFirst:
    """heapq as the search sees it, but with the h tie-break REVERSED (larger h first): the rule's mutant."""

    heappop = staticmethod(heapq.heappop)

    @staticmethod
    def heappush(heap, item):
        heapq.heappush(heap, (item[0], -item[1], item[2], item[3]))


def search_with_reversed_h(costs, start, goal):
    real, graph4.heapq = graph4.heapq, _LargerHFirst
    try:
        return graph4_astar_py(costs, start, goal)
    finally:
        graph4.heapq = real


def h_decides_cases(rng, wanted=3):
    """Small grids with whole-number costs (so ties are exact and common) where the (f, h, seq) rule's choice of h
    changes the path: found by running the real rule and its reversed-h mutant and keeping the cases where they pick
    different paths of the same cost. (Big random grids almost never do: 0 hits in 1,600 tries.)"""
    found, tries = [], 0
    while len(found) < wanted and tries < 200000:
        tries += 1
        rows, cols = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        costs = np.zeros((rows, cols, 4), np.float32)
        for r in range(rows):
            for c in range(cols):
                for d, (dr, dc) in enumerate(DELTAS):
                    if 0 <= r + dr < rows and 0 <= c + dc < cols and rng.random() > 0.12:
                        costs[r, c, d] = float(rng.integers(1, 4))
        start = (int(rng.integers(rows)), int(rng.integers(cols)))
        goal = (int(rng.integers(rows)), int(rng.integers(cols)))
        a, b = graph4_astar_py(costs, start, goal), search_with_reversed_h(costs, start, goal)
        if a[0] and b[0] and a[0] != b[0] and a[1] == b[1]:
            found.append((costs, start, goal))
    assert len(found) == wanted, f"only {len(found)} cases found in {tries} tries"
    return found


def cases():
    rng = np.random.default_rng(20261019)
    out = []
    for i in range(14):  # uneven costs, walls, one-way streets
        rows, cols = int(rng.integers(5, 16)), int(rng.integers(5, 16))
        out.append((lumpy(rng, rows, cols), (int(rng.integers(rows)), int(rng.integers(cols))),
                    (int(rng.integers(rows)), int(rng.integers(cols)))))
    for i in range(6):  # all costs equal: many tied ways, so the tie rule decides the path
        rows, cols = int(rng.integers(6, 14)), int(rng.integers(6, 14))
        out.append((lumpy(rng, rows, cols, wall=0.05, one_way=0.0, unit=True), (0, 0), (rows - 1, cols - 1)))
    open_grid = np.zeros((5, 5, 4), np.float32)
    for r in range(5):
        for c in range(5):
            for d, (dr, dc) in enumerate(DELTAS):
                if 0 <= r + dr < 5 and 0 <= c + dc < 5:
                    open_grid[r, c, d] = 1.0
    out.append((open_grid, (0, 0), (4, 4)))
    out.append((open_grid, (2, 2), (2, 2)))  # start == goal
    out.append((open_grid * 0.25, (0, 0), (4, 4)))  # every road below 1: hmin = 0.25
    blocked = open_grid.copy()
    blocked[:, 2, :] = 0.0  # nothing leaves column 2
    blocked[:, 1, RIGHT] = 0.0  # and nothing enters it from the west: no way east past column 1
    out.append((blocked, (0, 0), (4, 4)))  # unreachable
    out.append((np.zeros((4, 4, 4), np.float32), (0, 0), (3, 3)))  # no roads at all
    big = lumpy(rng, 40, 40, wall=0.1)
    out.append((big, (0, 0), (39, 39)))
    float_trap = np.zeros((1, 6, 4), np.float32)
    float_trap[0, :5, 1] = 0.1  # 0.1 is not exact in float32: the total depends on the order of the additions
    out.append((float_trap, (0, 0), (0, 5)))
    out.extend(h_decides_cases(rng))  # the last 3 cases: equal f, and only the h tie-break picks the path
    return out


def main() -> None:
    data = {}
    all_cases = cases()
    data["count"] = np.int32(len(all_cases))
    for i, (costs, start, goal) in enumerate(all_cases):
        found = graph4_astar_py(costs, start, goal)
        path, cost = found
        data[f"costs_{i}"] = costs.astype(np.float32)
        data[f"start_{i}"] = np.array(start, np.int32)
        data[f"goal_{i}"] = np.array(goal, np.int32)
        data[f"status_{i}"] = np.int32(0 if path else 4)
        data[f"path_{i}"] = np.array(path, np.int32).reshape(-1, 2)
        data[f"cost_{i}"] = np.float32(cost)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data)
    found = sum(int(data[f"status_{i}"]) == 0 for i in range(len(all_cases)))
    print(f"wrote {OUT} with {len(all_cases)} cases ({found} with a path, {len(all_cases) - found} without)")


if __name__ == "__main__":
    main()
