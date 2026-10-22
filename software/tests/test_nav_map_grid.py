"""T38b step 2: the map keeps its road grid (nav/graph4.py) right as arrows are added, changed and removed, however the
caller touches `nav.edges`, and `learned_road` is the one rule for what road an arrow gives."""
from __future__ import annotations

import numpy as np

from nav.map import ActionBook, Edge, EdgeBook, NavMap
from test_nav_worlds import ARENA_KEYS, ArenaWorld, make_arena_level


def edge(to: str, tries: int = 4, **kw) -> Edge:
    return Edge(tries=tries, outcomes={to: tries}, **kw)


def rebuilt(nav: NavMap) -> NavMap:
    """A second map holding the same arrows whose grid was filled from scratch (the reference)."""
    other = NavMap()
    other.edges = {p: dict(a) for p, a in nav.edges.items()}
    other.sync_grid()
    return other


def same_grid(a: NavMap, b: NavMap) -> bool:
    if set(a.grid.scenes) != set(b.grid.scenes):
        return False
    for scene, sa in a.grid.scenes.items():
        sb = b.grid.scenes[scene]
        # the arrays can be different sizes (spare room); compare square by square
        for (s, x, y, d) in set(a.grid.slots) | set(b.grid.slots):
            if s != scene:
                continue
            ra, ca = sa.cell(x, y)
            rb, cb = sb.cell(x, y)
            va = sa.costs[ra, ca, d] if 0 <= ra < sa.costs.shape[0] and 0 <= ca < sa.costs.shape[1] else 0.0
            vb = sb.costs[rb, cb, d] if 0 <= rb < sb.costs.shape[0] and 0 <= cb < sb.costs.shape[1] else 0.0
            if va != vb:
                return False
    return a.grid.slots == b.grid.slots and a.grid.odd == b.grid.odd


def test_edges_are_a_book_and_plain_assignment_is_wrapped():
    nav = NavMap()
    assert isinstance(nav.edges, EdgeBook)
    nav.edges = {"0:1:1": {"right": edge("0:2:1")}}
    assert isinstance(nav.edges, EdgeBook) and isinstance(nav.edges["0:1:1"], ActionBook)
    nav.edges["0:2:1"] = {"left": edge("0:1:1")}
    assert isinstance(nav.edges["0:2:1"], ActionBook)
    nav.sync_grid()
    assert set(nav.grid.slots) == {(0, 1, 1, 1), (0, 2, 1, 3)}


def test_every_way_of_changing_an_arrow_marks_it():
    nav = NavMap()
    nav.edges["0:1:1"] = {}
    nav.edges["0:1:1"]["right"] = edge("0:2:1")
    nav.sync_grid()
    assert (0, 1, 1, 1) in nav.grid.slots
    nav.edges["0:1:1"].pop("right")
    nav.sync_grid()
    assert not nav.grid.slots
    nav.edges.setdefault("0:3:3", {}).setdefault("down", edge("0:3:4"))
    nav.sync_grid()
    assert (0, 3, 3, 2) in nav.grid.slots
    del nav.edges["0:3:3"]["down"]
    nav.sync_grid()
    assert not nav.grid.slots
    nav.edges["0:5:5"] = {"up": edge("0:5:4"), "down": edge("0:5:6")}
    nav.sync_grid()
    assert len(nav.grid.slots) == 2
    nav.edges.pop("0:5:5")  # a whole place goes
    nav.sync_grid()
    assert not nav.grid.slots
    nav.edges.update({"0:1:1": {"left": edge("0:0:1")}})
    nav.edges["0:1:1"].update(right=edge("0:2:1"))
    nav.sync_grid()
    assert len(nav.grid.slots) == 2
    nav.edges["0:1:1"].clear()
    nav.edges.clear()
    nav.sync_grid()
    assert not nav.grid.slots


def test_replacing_a_place_forgets_its_old_arrows():
    nav = NavMap()
    nav.edges["0:1:1"] = {"right": edge("0:2:1")}
    nav.sync_grid()
    nav.edges["0:1:1"] = {"left": edge("0:0:1")}
    nav.sync_grid()
    assert set(nav.grid.slots) == {(0, 1, 1, 3)}


def test_the_rule_for_what_road_an_arrow_gives():
    nav = NavMap()
    assert nav.learned_road("0:1:1", "right", edge("0:2:1")) == ("0:2:1", 1.0, 1.0, False)
    assert nav.learned_road("0:1:1", "right", Edge(tries=4, outcomes={"0:2:1": 3, "blocked": 1}))[1:3] == (1 / 0.75, 0.75)
    assert nav.learned_road("0:1:1", "right", Edge(tries=2, outcomes={"0:2:1": 1, "blocked": 1})) is not None  # exactly half
    assert nav.learned_road("0:1:1", "right", Edge(tries=3, outcomes={"0:2:1": 1, "blocked": 2})) is None  # mostly blocked
    assert nav.learned_road("0:1:1", "right", Edge(tries=3, outcomes={"blocked": 3})) is None
    assert nav.learned_road("0:1:1", "right", edge("0:1:1")) is None  # stayed put
    assert nav.learned_road("0:1:1", "right", edge("0:2:1", bad_end=1)) is None
    assert nav.learned_road("0:1:1", "right", edge("0:2:1", good_end=1)) == ("0:2:1", 1.0, 1.0, True)
    assert nav.learned_road("0:1:1", "right", Edge(good_end=1, outcomes={"blocked": 1})) is None


def test_the_grid_follows_real_learning_and_matches_a_grid_built_from_scratch():
    level = make_arena_level(3)
    world = ArenaWorld(level)
    nav = NavMap()
    frame = world.reset(0)
    nav.begin(frame, world.hud())
    actions = list(ARENA_KEYS)
    checked = 0
    for step in range(1, 160):
        action = nav.explore_action(frame, actions) if nav.me.known else nav.me.probe_action(actions)
        assert action is not None
        after, _, done, _, info = world.step(action)
        nav.update(frame, action, after, info["hud"], done)
        frame = after
        if step % 20 == 0:
            nav.sync_grid()
            assert same_grid(nav, rebuilt(nav)), f"after {step} steps"
            checked += 1
        if done:
            break
    assert checked >= 2 and nav.grid.slots, "nothing was learned"


def test_a_saved_and_loaded_map_has_the_same_grid():
    nav = NavMap()
    nav.edges = {"0:1:1": {"right": edge("0:2:1")}, "0:2:1": {"left": edge("0:1:1", tries=2), "down": edge("0:2:2", good_end=1)}}
    nav.sync_grid()
    again = NavMap.from_dict(nav.to_dict())
    again.sync_grid()
    assert again.grid.slots == nav.grid.slots and again.grid.odd == nav.grid.odd
    assert np.array_equal(again.grid.scenes[0].costs.sum(), nav.grid.scenes[0].costs.sum())
