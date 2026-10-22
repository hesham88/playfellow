"""T32b §3: planning on what was learned — shortest routes, treasure order, the NAVIGATE skill."""
from __future__ import annotations

import pytest

from nav.map import Edge, NavMap, Place
from nav.planner import Navigator, Planner
from practice_arena.game import COIN_POINTS, DOOR_POINTS, LEVEL
from test_nav_worlds import ARENA_KEYS, ArenaWorld, arena_after_first_episode, arena_optimal_steps


def play(navigator, world, cap=600, goal="all"):
    """One episode driven only by the NAVIGATE skill. Returns (steps, world)."""
    frame = world.reset(0)
    navigator.start(frame, world.hud())
    for used in range(1, cap + 1):
        advice = navigator.advise(frame, goal)
        assert advice.actions, f"no advice at step {used}"
        action = advice.actions[0]
        after, _, done, _, info = world.step(action)
        navigator.learn(frame, action, after, info["hud"], done)
        frame = after
        if done:
            return used, world
    return None, world


@pytest.fixture(scope="module")
def trained():
    navigator, first = arena_after_first_episode()
    assert first.won and first.score == COIN_POINTS * sum(row.count("c") for row in LEVEL) + DOOR_POINTS
    return navigator


def test_on_a_fully_explored_level_the_route_equals_the_oracle(trained):
    steps, world = play(trained, ArenaWorld())
    assert world.game.won and world.game.coins_left == 0
    assert steps == arena_optimal_steps()


def test_probe_first_while_i_am_unknown():
    navigator = Navigator(list(ARENA_KEYS))
    navigator.start(ArenaWorld().reset(0))
    advice = navigator.advise(ArenaWorld().reset(0))
    assert advice.kind == "probe" and advice.actions[0] in ARENA_KEYS


def test_the_brain_can_ask_for_the_nearest_treasure_or_the_goal_or_to_explore(trained):
    world = ArenaWorld()
    frame = world.reset(0)
    trained.start(frame, world.hud())
    nearest = trained.advise(frame, "reward")
    assert nearest.kind == "reward" and nearest.target is not None
    every = trained.advise(frame, "all")
    assert every.kind == "reward"
    assert len(nearest.route) <= len(every.route) or nearest.target == every.target
    goal = trained.advise(frame, "goal")
    assert goal.kind == "goal" and goal.route  # the remembered way out, all the way to the door
    explore = trained.advise(frame, "explore")
    assert explore.kind in ("explore", "none")
    assert trained.advise(frame, "nonsense").kind == "reward"  # unknown goals mean "all"


def _line_map(length=6):
    """A learned corridor 0..length-1 in scene 0 (right/left arrows), for planner unit tests."""
    nav = NavMap()
    nav.scene = 0
    for x in range(length):
        nav.places[f"0:{x}:0"] = Place(visits=1)
        acts = {}
        if x + 1 < length:
            acts["right"] = Edge(tries=1, outcomes={f"0:{x + 1}:0": 1})
        if x > 0:
            acts["left"] = Edge(tries=1, outcomes={f"0:{x - 1}:0": 1})
        nav.edges[f"0:{x}:0"] = acts
    return nav


def test_shortest_routes_use_only_learned_roads():
    nav = _line_map()
    planner = Planner(nav, ["left", "right"])
    roads = planner.roads(None)
    dist, back = planner.shortest(roads, "0:0:0")
    assert dist["0:5:0"] == 5
    assert [r.action for r in planner.walk(back, "0:0:0", "0:5:0")] == ["right"] * 5
    nav.edges["0:2:0"]["right"] = Edge(tries=3, outcomes={"blocked": 3})
    dist, _ = planner.shortest(planner.roads(None), "0:0:0")
    assert "0:5:0" not in dist  # learned wall: no road


def test_a_game_ending_move_is_only_ever_the_last_step():
    """The door sits in the middle of the corridor: reaching the treasure beyond it must not walk through it."""
    nav = _line_map()
    nav.edges["0:2:0"]["right"] = Edge(tries=1, outcomes={"0:3:0": 1}, good_end=1)
    nav.places["0:3:0"].goal = 1
    planner = Planner(nav, ["left", "right"])
    dist, _ = planner.shortest(planner.roads(None), "0:0:0")
    assert "0:4:0" not in dist  # beyond the ending move: not reachable without ending the game
    assert "0:2:0" in dist


def test_treasure_order_tries_every_order_and_2opt_beyond_seven():
    # points on a line: the best order from 0 is left-to-right, never zig-zag
    positions = {f"t{i}": i * 3 for i in range(1, 10)}
    positions["start"] = 0

    def table_for(names):
        return {a: {b: abs(positions[a] - positions[b]) for b in names} for a in names}

    planner = Planner(NavMap(), [])
    few = ["t3", "t1", "t2"]
    assert planner.order("start", few, table_for(["start"] + few), None) == ["t1", "t2", "t3"]
    many = [f"t{i}" for i in (9, 2, 7, 4, 1, 8, 3, 6, 5)]  # 9 > BRUTE_FORCE: greedy + 2-opt
    order = planner.order("start", many, table_for(["start"] + many), None)
    assert order == [f"t{i}" for i in range(1, 10)]
    # with a finish near the start, the far treasures come first
    finish = {name: positions[name] for name in ["start"] + few}
    assert planner.order("start", few, table_for(["start"] + few), finish) == ["t3", "t2", "t1"]


def test_the_navigator_saves_to_the_notebook_and_comes_back(trained, tmp_path):
    trained.folder, trained.card_stem = tmp_path, "practice_arena"
    path = trained.save()
    assert path.name == "practice_arena.nav.json"
    again = Navigator(list(ARENA_KEYS), tmp_path, "practice_arena")
    summary = again.summary()
    assert summary["places"] == trained.summary()["places"] and summary["blocked"] > 0
    steps, world = play(again, ArenaWorld())
    assert world.game.won and steps == arena_optimal_steps()


def test_the_safety_gate_never_steps_onto_a_known_danger():
    """The treasure lies past a square where the game once ended badly: no advice may step onto it."""
    nav = _line_map()
    nav.here = "0:2:0"
    nav.me.box = (0, 0, 1, 1)
    nav.me.effects = {"left": [(-10, 0)] * 3, "right": [(10, 0)] * 3}
    nav.phase = (0.0, 0.0)
    nav.places["0:3:0"].hazard = 1
    nav.places["0:5:0"].rewards = 1
    planner = Planner(nav, ["left", "right"])
    for goal in ("all", "reward", "to:5,0", "explore"):
        advice = planner.advise(None, goal)
        assert "right" not in advice.actions[:1], goal
    assert planner.advise(None, "all").kind in ("avoid", "none", "explore")
