"""T32b §1: "which thing is me?" — the self-model finds the player from pixels and key presses only."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from arena_pack.base import BANNER as PACK_BANNER
from arena_pack.box_push import BoxPush, TILE as BOX_TILE
from arena_pack.key_caves import KeyCaves, ROOM_W, TILE as CAVE_TILE
from nav.self_model import SelfModel, find_motions
from nav import frames
from test_nav_worlds import ARENA_KEYS, SIDE_KEYS, ArenaWorld, SideScroller

PROBES = 12


def _inside(point, box, slack=0):
    return box[0] - slack <= point[0] <= box[2] + slack and box[1] - slack <= point[1] <= box[3] + slack


def _probe(world, keys, model, limit=PROBES):
    frame = world.reset(0)
    view = None
    for used in range(1, limit + 1):
        action = model.probe_action(list(keys))
        after = world.step(action)[0]
        view = model.observe(frame, action, after)
        frame = after
        if model.known:
            return used, view, frame
    return None, view, frame


def test_finds_the_player_in_the_arena_within_twelve_probes():
    world, model = ArenaWorld(), SelfModel()
    used, view, _ = _probe(world, ARENA_KEYS, model)
    assert used is not None and used <= PROBES
    assert _inside(view.center, world.true_box())


def test_finds_the_player_in_a_side_scroller_and_ignores_the_drifting_cloud():
    world, model = SideScroller(), SelfModel()
    used, view, _ = _probe(world, SIDE_KEYS, model)
    assert used is not None
    assert _inside(view.center, world.true_box())
    # keep playing: the box follows the player, never the cloud
    frame = world.observation()
    for action in ["right", "right", "left", "jump", "right"]:
        after = world.step(action)[0]
        view = model.observe(frame, action, after)
        frame = after
        assert _inside(view.center, world.true_box(), slack=4)


def test_a_thing_that_moves_on_its_own_is_never_me():
    """Only the cloud moves (the player stands at a wall pressing a blocked key): nothing is identified."""
    world, model = SideScroller(level=("...", "...", "...", "...", "...", "...", "...", "#P#", "###")), SelfModel()
    world.level = tuple(row.ljust(20, ".") if i < 7 else row.ljust(20, "#") for i, row in enumerate(world.level))
    frame = world.reset(0)
    for _ in range(PROBES):
        after = world.step("left")[0]
        model.observe(frame, "left", after)
        frame = after
    assert not model.known


def test_finds_the_player_in_box_push_and_key_caves():
    box_game, model = BoxPush(seed=0), SelfModel()
    used, view, _ = _probe(box_game, box_game.keys, model)
    assert used is not None
    x, y = box_game.player
    ox = (box_game.board_pixels()[0] - 7 * BOX_TILE) // 2
    centre = (ox + x * BOX_TILE + BOX_TILE // 2, PACK_BANNER + y * BOX_TILE + BOX_TILE // 2)
    assert _inside(centre, view.box, slack=4)

    caves, model = KeyCaves(seed=0), SelfModel()
    used, view, _ = _probe(caves, caves.keys, model)
    assert used is not None
    centre = ((caves.x % ROOM_W) * CAVE_TILE + CAVE_TILE // 2, PACK_BANNER + caves.y * CAVE_TILE + CAVE_TILE // 2)
    assert _inside(centre, view.box, slack=4)


def test_learns_step_size_and_blocked_moves_in_the_arena():
    world, model = ArenaWorld(), SelfModel()
    used, _, frame = _probe(world, ARENA_KEYS, model)
    assert used is not None
    # walk the corridor along the left: down / up / down... then bump the top wall
    for action in ["up", "down", "up"]:
        after = world.step(action)[0]
        model.observe(frame, action, after)
        frame = after
    assert world.game.player == (1, 1)
    step = model.step_size()
    assert step is not None and abs(step[1] - 48) <= 4  # the tile size, never told to it
    after = world.step("up")[0]  # (1, 0) is a wall
    view = model.observe(frame, "up", after)
    assert view.blocked and view.moved == (0, 0)
    frame = after
    after = world.step("down")[0]  # free
    view = model.observe(frame, "down", after)
    assert not view.blocked and view.moved[1] > 0


def test_blocked_jump_in_the_side_scroller():
    """Jumps learned in the open rise; under a low ceiling the same jump goes nowhere: blocked."""
    level = ("....................",) * 6 + ("..#.................", "..P.................", "####################")
    world, model = SideScroller(level=level), SelfModel()
    used, _, frame = _probe(world, SIDE_KEYS, model, limit=24)
    assert used is not None

    def do(action):
        nonlocal frame
        after = world.step(action)[0]
        view = model.observe(frame, action, after)
        frame = after
        return view

    for _ in range(3):  # out in the open: walk right, jump, land (pushing left/right while falling)
        while world.x < 5 or world.y != 7:
            do("right" if world.x < 5 else "left")
        do("left")  # stand still for a beat: next press starts at rest
        do("right")
        do("jump")
    assert model.effect("jump") is not None and model.effect("jump")[1] < 0, "jump never learned as 'up'"
    while (world.x, world.y) != (2, 7):  # back under the ceiling block
        do("left" if world.x > 2 else "right")
    do("left")
    do("right")  # step out and back in, then jump under the block
    view = do("jump")
    assert view.blocked

def test_locate_finds_me_again_in_a_fresh_episode_and_round_trips():
    world, model = ArenaWorld(), SelfModel()
    assert _probe(world, ARENA_KEYS, model)[0] is not None
    copy = SelfModel.from_dict(model.to_dict())
    frame = world.reset(0)
    view = copy.locate(frame)
    assert _inside(view.center, world.true_box())
    assert copy.effect("down") == model.effect("down")


def test_motion_finder_reports_the_true_move():
    world = ArenaWorld()
    before = world.reset(0)
    after = world.step("down")[0]
    cell = frames.cell_size(after.shape)
    motions = find_motions(frames.shrink(before, cell), frames.shrink(after, cell), cell)
    assert [(m.dx, m.dy) for m in motions] == [(0, 48)]


def test_no_colour_or_position_constants_in_nav():
    """Navigation must be learned: no hex colours or (r, g, b)-like triples anywhere in software/nav/."""
    nav = Path(__file__).resolve().parents[1] / "nav"
    pattern = re.compile(r"#[0-9A-Fa-f]{6}\b|\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)")
    files = list(nav.glob("*.py"))
    assert files
    for path in files:
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name
