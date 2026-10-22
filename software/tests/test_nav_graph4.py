"""T38b: the learned roads as a (rows, cols, 4) array of costs, the reference search that runs on it, and the edge grid
that keeps the array right one road at a time."""
from __future__ import annotations

import heapq

import numpy as np
import pytest

from nav.graph4 import DELTAS, DOWN, LEFT, RIGHT, UP, EdgeGrid, direction_of, graph4_astar_py


def random_costs(seed, rows=9, cols=11, wall=0.18, one_way=0.1, lumpy=True):
    """A random road graph on a grid: walls (no road in or out), one-way streets, uneven costs (ties included)."""
    rng = np.random.default_rng(seed)
    costs = np.zeros((rows, cols, 4), np.float32)
    open_ = rng.random((rows, cols)) > wall
    for r in range(rows):
        for c in range(cols):
            for d, (dr, dc) in enumerate(DELTAS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and open_[r, c] and open_[nr, nc] and rng.random() > one_way:
                    costs[r, c, d] = (1.0 + int(rng.integers(0, 5)) * 0.5) if lumpy else 1.0
    return costs


def brute_force(costs, start, goal):
    """Plain Dijkstra on the same array: the cheapest total cost, or None."""
    rows, cols = costs.shape[:2]
    best, heap = {start: 0.0}, [(0.0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == goal:
            return d
        if d > best[(r, c)]:
            continue
        for k, (dr, dc) in enumerate(DELTAS):
            nr, nc = r + dr, c + dc
            step = float(costs[r, c, k])
            if 0 <= nr < rows and 0 <= nc < cols and step > 0 and d + step < best.get((nr, nc), 1e18):
                best[(nr, nc)] = d + step
                heapq.heappush(heap, (d + step, (nr, nc)))
    return None


def check_a_valid_walk(costs, path, start, goal):
    assert path[0] == start and path[-1] == goal
    total = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        d = DELTAS.index((r2 - r1, c2 - c1))
        assert costs[r1, c1, d] > 0, "every step must follow a road"
        total += float(costs[r1, c1, d])
    return total


# ---- the reference search ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(40))
def test_the_search_finds_the_cheapest_way_on_random_road_graphs(seed):
    costs = random_costs(seed)
    rng = np.random.default_rng(1000 + seed)
    start, goal = (int(rng.integers(9)), int(rng.integers(11))), (int(rng.integers(9)), int(rng.integers(11)))
    found = graph4_astar_py(costs, start, goal)
    want = brute_force(costs, start, goal)
    path, cost = found
    if want is None:
        assert path == [] and cost == 0.0
    else:
        assert cost == pytest.approx(want, rel=1e-5)
        assert check_a_valid_walk(costs, path, start, goal) == pytest.approx(want, rel=1e-5)


def test_one_way_streets_are_respected():
    costs = np.zeros((1, 3, 4), np.float32)
    costs[0, 0, RIGHT] = 1.0
    costs[0, 1, RIGHT] = 1.0  # only a way east; nothing leads back west
    assert graph4_astar_py(costs, (0, 0), (0, 2))[0] == [(0, 0), (0, 1), (0, 2)]
    assert graph4_astar_py(costs, (0, 2), (0, 0)) == ([], 0.0)


def test_a_cost_below_one_still_gives_the_cheapest_way_because_the_guess_uses_the_cheapest_road():
    costs = random_costs(5, lumpy=False) * 0.25  # every road costs 0.25
    path, cost = graph4_astar_py(costs, (0, 0), (8, 10))
    want = brute_force(costs, (0, 0), (8, 10))
    assert (want is None and path == []) or cost == pytest.approx(want, rel=1e-5)


def test_start_equal_to_goal_and_unusable_arguments():
    costs = random_costs(1)
    assert graph4_astar_py(costs, (3, 3), (3, 3)) == ([(3, 3)], 0.0)
    assert graph4_astar_py(costs, (-1, 0), (3, 3)) is None and graph4_astar_py(costs, (0, 0), (99, 3)) is None
    assert graph4_astar_py(np.zeros((0, 0, 4), np.float32), (0, 0), (0, 0)) is None
    assert graph4_astar_py(np.zeros((3, 3, 3), np.float32), (0, 0), (1, 1)) is None
    assert graph4_astar_py(np.zeros((3, 3, 4), np.float32), (0, 0), (2, 2)) == ([], 0.0)  # no roads at all


def test_nan_and_infinite_costs_are_not_roads():
    costs = np.zeros((1, 3, 4), np.float32)
    costs[0, 0, RIGHT] = np.nan
    costs[0, 1, RIGHT] = np.inf
    assert graph4_astar_py(costs, (0, 0), (0, 2)) == ([], 0.0)


def test_ties_are_broken_the_same_way_every_time_by_the_written_rule():
    """Open plain grid, many equal-cost ways: the rule (f, then h, then first-pushed; directions up, right, down, left)
    always picks the same one. The native version must produce exactly this path."""
    costs = np.zeros((4, 4, 4), np.float32)
    for r in range(4):
        for c in range(4):
            for d, (dr, dc) in enumerate(DELTAS):
                if 0 <= r + dr < 4 and 0 <= c + dc < 4:
                    costs[r, c, d] = 1.0
    first = graph4_astar_py(costs, (0, 0), (3, 3))
    assert first == graph4_astar_py(costs, (0, 0), (3, 3)) and first[1] == 6.0
    assert first[0] == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]  # recorded: right along the top, then down


def test_float32_arithmetic_is_used_so_the_total_is_a_float32_sum():
    costs = np.zeros((1, 4, 4), np.float32)
    for c in range(3):
        costs[0, c, RIGHT] = 0.1  # 0.1 is not exact in float32
    path, cost = graph4_astar_py(costs, (0, 0), (0, 3))
    total = np.float32(0.0)
    for _ in range(3):
        total = np.float32(total + np.float32(0.1))
    assert cost == float(total)


# ---- the edge grid ----------------------------------------------------------------------------------------
def parse(key):
    scene, x, y = (int(v) for v in key.split(":"))
    return scene, (x, y)


def make(roads):
    """An EdgeGrid whose roads are {(place, action): (to, cost, share, ends)}."""
    grid = EdgeGrid(parse)
    for key in roads:
        grid.mark(*key)
    grid.sync(lambda place, action: roads.get((place, action)))
    return grid


def slot_cost(grid, scene, x, y, d):
    sc = grid.scenes[scene]
    r, c = sc.cell(x, y)
    return float(sc.costs[r, c, d])


def test_a_road_lands_in_the_right_slot_with_its_cost():
    grid = make({("0:2:3", "right"): ("0:3:3", 1.0, 1.0, False), ("0:2:3", "up"): ("0:2:2", 1.25, 0.8, False)})
    assert slot_cost(grid, 0, 2, 3, RIGHT) == 1.0 and slot_cost(grid, 0, 2, 3, UP) == 1.25
    assert slot_cost(grid, 0, 2, 3, DOWN) == 0.0 and grid.exact


def test_changing_a_road_replaces_it_and_removing_it_clears_the_slot():
    roads = {("0:2:3", "right"): ("0:3:3", 1.0, 1.0, False)}
    grid = make(roads)
    roads[("0:2:3", "right")] = ("0:3:3", 2.0, 0.5, False)
    grid.mark("0:2:3", "right")
    grid.sync(lambda p, a: roads.get((p, a)))
    assert slot_cost(grid, 0, 2, 3, RIGHT) == 2.0
    roads[("0:2:3", "right")] = ("0:2:3", 1.0, 1.0, False)  # now it leads nowhere new: not a road at all? (same place)
    del roads[("0:2:3", "right")]
    grid.mark("0:2:3", "right")
    grid.sync(lambda p, a: roads.get((p, a)))
    assert slot_cost(grid, 0, 2, 3, RIGHT) == 0.0 and not grid.slots


def test_two_keys_for_the_same_direction_keep_the_cheaper_and_survive_removing_one():
    roads = {("0:1:1", "right"): ("0:2:1", 1.5, 1.0, False), ("0:1:1", "d"): ("0:2:1", 1.0, 1.0, False)}
    grid = make(roads)
    assert slot_cost(grid, 0, 1, 1, RIGHT) == 1.0
    del roads[("0:1:1", "d")]
    grid.mark("0:1:1", "d")
    grid.sync(lambda p, a: roads.get((p, a)))
    assert slot_cost(grid, 0, 1, 1, RIGHT) == 1.5  # the dearer key is still a road


def test_roads_that_end_the_game_are_not_in_the_grid_and_odd_roads_are_counted():
    grid = make({("0:1:1", "down"): ("0:1:2", 1.0, 1.0, True),  # ends the game: never walked through
                 ("0:1:1", "jump"): ("0:5:1", 1.0, 1.0, False),  # a jump of four squares
                 ("0:1:1", "door"): ("1:1:1", 1.0, 1.0, False)})  # another room
    assert not grid.slots and not grid.exact and grid.odd == {("0:1:1", "jump"), ("0:1:1", "door")}
    grid.mark("0:1:1", "jump")
    grid.mark("0:1:1", "door")
    grid.sync(lambda p, a: None)
    assert grid.exact  # once those roads are gone the grid is exact again


def test_the_array_grows_in_every_direction_and_keeps_what_it_had():
    grid = make({("0:0:0", "right"): ("0:1:0", 1.0, 1.0, False)})
    for place, to in (("0:-30:5", "0:-30:6"), ("0:40:-20", "0:41:-20"), ("0:3:60", "0:3:61")):
        grid.mark(place, "x")
    roads = {("0:-30:5", "x"): ("0:-30:6", 1.0, 1.0, False), ("0:40:-20", "x"): ("0:41:-20", 1.0, 1.0, False),
             ("0:3:60", "x"): ("0:3:61", 1.0, 1.0, False), ("0:0:0", "right"): ("0:1:0", 1.0, 1.0, False)}
    grid.mark("0:0:0", "right")
    grid.sync(lambda p, a: roads.get((p, a)))
    assert slot_cost(grid, 0, 0, 0, RIGHT) == 1.0 and slot_cost(grid, 0, -30, 5, DOWN) == 1.0
    assert slot_cost(grid, 0, 40, -20, RIGHT) == 1.0 and slot_cost(grid, 0, 3, 60, DOWN) == 1.0


def test_guesses_are_written_for_one_question_and_then_put_back():
    roads = {("0:1:1", "right"): ("0:2:1", 1.0, 1.0, False)}
    grid = make(roads)
    before = grid.scenes[0].costs.copy()
    with grid.overlay(0, [("0:2:1", "0:3:1", 1.2, 0.7, "right"), ("0:1:1", "0:1:2", 2.0, 0.4, "down"),
                          ("0:1:1", "0:9:9", 2.0, 0.4, "jump")]) as ov:
        assert slot_cost(grid, 0, 2, 1, RIGHT) == pytest.approx(1.2) and slot_cost(grid, 0, 1, 1, DOWN) == 2.0
        assert ov.skipped == 1 and len(ov.added) == 2
    assert np.array_equal(grid.scenes[0].costs[:before.shape[0], :before.shape[1]], before) or grid.scenes[0].costs.sum() == before.sum()
    assert slot_cost(grid, 0, 2, 1, RIGHT) == 0.0 and slot_cost(grid, 0, 1, 1, DOWN) == 0.0
    assert slot_cost(grid, 0, 1, 1, RIGHT) == 1.0  # the learned road is untouched


def test_a_guess_never_replaces_a_cheaper_learned_road():
    grid = make({("0:1:1", "right"): ("0:2:1", 1.0, 1.0, False)})
    with grid.overlay(0, [("0:1:1", "0:2:1", 2.0, 0.4, "d")]) as ov:
        assert slot_cost(grid, 0, 1, 1, RIGHT) == 1.0 and not ov.added


def test_directions_are_up_right_down_left_with_up_meaning_a_smaller_y():
    assert [direction_of((5, 5), n) for n in ((5, 4), (6, 5), (5, 6), (4, 5))] == [UP, RIGHT, DOWN, LEFT]
    assert direction_of((5, 5), (6, 6)) is None and direction_of((5, 5), (5, 5)) is None


# ---- the test vectors the native version is compared against -----------------------------------------------
def test_the_committed_vectors_still_match_the_reference_search():
    """tests/data/graph4_vectors.npz holds the reference's exact answers (python tests/make_graph4_vectors.py rewrites it).
    If this fails the file is stale, and the native parity test would be comparing against the wrong answers."""
    from pathlib import Path
    with np.load(Path(__file__).parent / "data" / "graph4_vectors.npz") as data:
        count = int(data["count"])
        assert count >= 20
        paths_found = 0
        for i in range(count):
            found = graph4_astar_py(data[f"costs_{i}"], tuple(data[f"start_{i}"]), tuple(data[f"goal_{i}"]))
            path, cost = found
            assert [tuple(p) for p in data[f"path_{i}"].tolist()] == path, f"case {i}"
            assert np.float32(cost) == data[f"cost_{i}"] and int(data[f"status_{i}"]) == (0 if path else 4)
            paths_found += bool(path)
        assert 0 < paths_found < count  # both found and unreachable cases are in the file


def test_the_last_vectors_are_cases_where_only_the_h_tie_break_decides_the_path():
    """Pins the middle of the rule (f, h, seq): with h reversed (larger first) these three come out different."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from make_graph4_vectors import search_with_reversed_h
    with np.load(Path(__file__).parent / "data" / "graph4_vectors.npz") as data:
        count = int(data["count"])
        for i in range(count - 3, count):
            costs, start, goal = data[f"costs_{i}"], tuple(data[f"start_{i}"]), tuple(data[f"goal_{i}"])
            real, mutant = graph4_astar_py(costs, start, goal), search_with_reversed_h(costs, start, goal)
            assert real[0] and mutant[0] and real[0] != mutant[0] and real[1] == mutant[1], f"case {i}"
