"""T38b step 2: Planner.route_frame answers "the cheapest way from here to there" on the map's road grid with the native
graph search, and always agrees with the Python search over the road list (Planner.route)."""
from __future__ import annotations

import random
import time
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import native_bridge
except ImportError:  # no native accelerator here: its tests skip
    native_bridge = None
from nav import planner as planner_module
from graph4_reference import graph4_astar_py
from nav.map import Edge, NavMap
from nav.planner import Planner, Road

ACTIONS = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
real = pytest.mark.skipif(native_bridge is None or not native_bridge.available()
                          or native_bridge.graph4_astar(np.ones((1, 2, 4), "float32"), (0, 0), (0, 1)) is None,
                          reason="the native library with graph4 is not built here")


@pytest.fixture(params=["python-reference", pytest.param("native", marks=real)])
def engine(request, monkeypatch):
    """Run the same tests against the native search and against its Python reference (same answers by contract)."""
    if request.param == "python-reference":
        monkeypatch.setattr(planner_module, "native_bridge", SimpleNamespace(graph4_astar=graph4_astar_py))
    return request.param


def key(x, y, scene=0):
    return f"{scene}:{x}:{y}"


def learn_grid(nav, width, height, walls=(), cost=lambda x, y: 1.0, scene=0):
    """Teach the map a whole grid: both ways between open neighbours; the cost belongs to the square entered
    (a cost of 2 is learned as an arrow that worked once in two tries, 3 as once in three)."""
    for x in range(width):
        for y in range(height):
            if (x, y) in walls:
                continue
            for (dx, dy), action in ACTIONS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                    c = int(cost(nx, ny))
                    to = key(nx, ny, scene)
                    outcomes = {to: 1} if c == 1 else {to: 1, "blocked": c - 1}
                    nav.edges.setdefault(key(x, y, scene), {})[action] = Edge(tries=c, outcomes=outcomes)


def make(width=8, height=8, **kw):
    nav = NavMap()
    learn_grid(nav, width, height, **kw)
    return Planner(nav, list(ACTIONS.values()))


def python_way(p, start, goal):
    return p.route(p.roads(None), start, goal)


def cost_of(way):
    return sum(r.cost for r in way)


def assert_a_valid_walk(p, way, start, goal):
    roads, here = p.roads(None), start
    for road in way:
        assert any(r.to == road.to and r.action == road.action for r in roads[here]), "every step must be a road leaving here"
        here = road.to
    assert here == goal


@pytest.mark.parametrize("seed", range(6))
def test_same_cost_as_the_python_search_on_random_grids(engine, seed):
    rng = random.Random(seed)
    walls = {(rng.randrange(14), rng.randrange(14)) for _ in range(40)} - {(0, 0), (13, 13)}
    heavy = {(x, y): 1 + rng.randrange(2) for x in range(14) for y in range(14)}
    p = make(14, 14, walls=walls, cost=lambda x, y: heavy[(x, y)])
    start, goal = key(0, 0), key(13, 13)
    way = p.route_frame(None, start, goal)
    used = p.last_route_engine
    want = python_way(p, start, goal)
    assert way is not Planner.NOT_ANSWERED
    if want is None:
        assert way is None
    else:
        assert used == "native"
        assert cost_of(way) == pytest.approx(cost_of(want), rel=1e-4)
        assert_a_valid_walk(p, way, start, goal)


def test_no_way_is_none_and_start_equal_to_goal_is_an_empty_way(engine):
    p = make(5, 5, walls={(2, y) for y in range(5)})  # a wall across the room
    assert p.route_frame(None, key(0, 0), key(4, 4)) is None
    assert p.route_frame(None, key(0, 0), key(0, 0)) == []


def test_a_square_the_map_has_never_seen_is_not_a_way(engine):
    p = make(4, 4)
    assert p.route_frame(None, key(0, 0), key(30, 30)) is None


def test_arrows_that_changed_are_used_straight_away_without_rebuilding(engine):
    p = make(6, 1)
    assert cost_of(p.route_frame(None, key(0, 0), key(5, 0))) == 5
    del p.nav.edges[key(2, 0)]["right"]  # a door slams shut
    assert p.route_frame(None, key(0, 0), key(5, 0)) is None
    p.nav.edges[key(2, 0)]["right"] = Edge(tries=2, outcomes={key(3, 0): 2})
    assert cost_of(p.route_frame(None, key(0, 0), key(5, 0))) == 5


def test_roads_that_end_the_game_and_bad_ends_are_never_walked_through(engine):
    p = make(5, 1)
    p.nav.edges[key(2, 0)]["right"].good_end = 1  # stepping from 2 to 3 ended the game well: only ever the last step
    assert p.route_frame(None, key(0, 0), key(4, 0)) is None
    assert python_way(p, key(0, 0), key(4, 0)) is None
    p.nav.sync_grid()
    p2 = make(5, 1)
    p2.nav.edges[key(1, 0)]["right"].bad_end = 1
    p2.nav.grid.mark(key(1, 0), "right")  # a count changed in place: the caller says so (NavMap.update does)
    assert p2.route_frame(None, key(0, 0), key(4, 0)) is None


def test_a_jump_or_teleport_road_means_the_grid_cannot_answer(engine):
    p = make(4, 4)
    p.nav.edges[key(0, 0)]["up"] = Edge(tries=2, outcomes={key(3, 3): 2})
    assert p.route_frame(None, key(0, 0), key(3, 3)) is Planner.NOT_ANSWERED
    assert python_way(p, key(0, 0), key(3, 3)) is not None  # the Python search still has the answer


def test_two_rooms_cannot_be_answered_on_one_grid(engine):
    p = make(3, 3)
    learn_grid(p.nav, 3, 3, scene=1)
    assert p.route_frame(None, key(0, 0, 0), key(2, 2, 1)) is Planner.NOT_ANSWERED


def test_guessed_roads_are_used_and_agree_with_the_python_search(engine, monkeypatch):
    p = make(4, 1)
    far = [(key(3, 0), Road("right", key(4, 0), 1.2, 0.7)), (key(4, 0), Road("right", key(5, 0), 2.0, 0.4))]
    monkeypatch.setattr(p, "_guessed", lambda frame: far)
    way = p.route_frame(None, key(0, 0), key(5, 0))
    roads = p.roads(None)
    assert [r.to for r in way] == [key(i, 0) for i in range(1, 6)]
    assert cost_of(way) == pytest.approx(cost_of(p.route(roads, key(0, 0), key(5, 0))))
    assert not any(s[1] >= 4 for s in p.nav.grid.slots)  # the guesses are not kept as learned roads
    assert float(p.nav.grid.scenes[0].costs[0 - p.nav.grid.scenes[0].oy, 3 - p.nav.grid.scenes[0].ox, 1]) == 0.0


def test_a_guess_that_ends_the_game_is_left_out_like_the_python_search(engine, monkeypatch):
    p = make(3, 1)
    monkeypatch.setattr(p, "_guessed", lambda frame: [(key(2, 0), Road("right", key(3, 0), 1.2, 0.7, ends=True))])
    assert p.route_frame(None, key(0, 0), key(3, 0)) is None
    assert python_way(p, key(0, 0), key(3, 0)) is None


def test_without_the_native_library_it_says_not_answered(monkeypatch):
    p = make(3, 3)
    monkeypatch.setattr(planner_module, "native_bridge", None)
    assert p.route_frame(None, key(0, 0), key(2, 2)) is Planner.NOT_ANSWERED
    monkeypatch.setattr(planner_module, "native_bridge", SimpleNamespace(graph4_astar=lambda *a: None))
    assert p.route_frame(None, key(0, 0), key(2, 2)) is Planner.NOT_ANSWERED  # the library refused the input


def test_go_to_uses_the_grid_and_falls_back_to_python(engine, monkeypatch):
    p = make(6, 1)
    p.nav.here, p.nav.scene = key(0, 0), 0
    advice = p._go_to(None, "5,0", 3)
    assert advice.kind == "to" and advice.actions == ["right"] * 3 and p.last_route_engine == "native"
    monkeypatch.setattr(planner_module, "native_bridge", None)
    advice = p._go_to(None, "5,0", 3)
    assert advice.kind == "to" and advice.actions == ["right"] * 3 and p.last_route_engine == "python"


@real
def test_the_grid_search_is_faster_than_the_python_search_and_the_numbers_are_reported(capsys):
    def best_of(fn, n=5):
        times = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        return min(times) * 1000

    lines = []
    for size in (40, 60, 100):
        p = make(size, size)
        p.nav.sync_grid()
        start, goal = key(0, 0), key(size - 1, size - 1)
        grid = best_of(lambda: p.route_frame(None, start, goal))
        python = best_of(lambda: python_way(p, start, goal))
        lines.append(f"{size}x{size}: python (roads + search) {python:.1f} ms, grid + native {grid:.1f} ms")
        assert grid < python, lines[-1]
        assert cost_of(p.route_frame(None, start, goal)) == pytest.approx(cost_of(python_way(p, start, goal)))
    with capsys.disabled():
        print("\n[T38b] route, one question: " + "; ".join(lines))
