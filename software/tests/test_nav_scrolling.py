"""T32b: scrolling worlds — a following camera slides the background; "me" and the map must stay in WORLD terms."""
from __future__ import annotations

import numpy as np

from nav.map import NavMap
from nav.self_model import SelfModel

W, H = 320, 240
STEP = 16  # pixels "me" moves in the world per key press
SPRITE = 16
KEYS = {"left": (-STEP, 0), "right": (STEP, 0), "up": (0, -STEP), "down": (0, STEP)}


def make_world(seed=3):
    rng = np.random.default_rng(seed)
    tiles = rng.integers(40, 220, size=(60, 120, 3)).astype(np.uint8)
    return np.kron(tiles, np.ones((16, 16, 1), dtype=np.uint8))  # 960 × 1920 px of detailed ground


class Scroller:
    """A tiny scrolling game. The camera follows "me" (pinned mid-screen) or only once "me" nears an edge."""

    def __init__(self, follow="pinned", seed=3):
        self.ground = make_world(seed)
        self.follow = follow
        self.me = [400, 320]  # world pixels
        self.cam = [self.me[0] - 160, self.me[1] - 120]  # world position of the screen's top-left corner

    def frame(self):
        x, y = self.cam
        img = self.ground[y:y + H, x:x + W].copy()
        sx, sy = self.me[0] - x, self.me[1] - y
        img[sy:sy + SPRITE, sx:sx + SPRITE] = (255, 0, 255)
        img[:12, :80] = (255, 255, 255)  # a HUD that stays put
        return img

    def step(self, key):
        dx, dy = KEYS.get(key, (0, 0))
        self.me[0] += dx
        self.me[1] += dy
        sx, sy = self.me[0] - self.cam[0], self.me[1] - self.cam[1]
        if self.follow == "pinned":
            self.cam = [self.me[0] - 160, self.me[1] - 120]
        else:  # edge-follow: the camera only moves when "me" gets within 3 blocks of the screen edge
            if sx < 48:
                self.cam[0] -= STEP
            if sx > W - 48 - SPRITE:
                self.cam[0] += STEP
            if sy < 48:
                self.cam[1] -= STEP
            if sy > H - 48 - SPRITE:
                self.cam[1] += STEP
        return self.frame()


def probe(world, model, keys=("right", "left", "down", "up")):
    frame = world.frame()
    for key in keys:
        after = world.step(key)
        model.observe(frame, key, after)
        frame = after
    return frame


def test_a_pinned_camera_is_not_mistaken_for_a_new_scene():
    world, model = Scroller("pinned"), SelfModel()
    frame = world.frame()
    scene_changes = 0
    for key in ["right", "right", "down", "left", "up", "right"] * 3:
        after = world.step(key)
        view = model.observe(frame, key, after)
        scene_changes += view.scene_change
        frame = after
    assert scene_changes == 0


def test_the_camera_slide_is_reported_when_the_background_moves():
    world, model = Scroller("pinned"), SelfModel()
    frame = world.frame()
    after = world.step("right")  # me +16 in the world, camera +16, so the background slides left by 16
    view = model.observe(frame, "right", after)
    cell = model.cell
    assert cell is not None
    assert abs(view.camera[0] + STEP) <= cell and view.camera[1] == 0


def test_the_camera_offset_adds_up_over_a_walk():
    world, model = Scroller("pinned"), SelfModel()
    frame = world.frame()
    for key in ["right"] * 5 + ["down"] * 3:
        after = world.step(key)
        model.observe(frame, key, after)
        frame = after
    ox, oy = model.camera.offset
    assert abs(ox + 5 * STEP) <= 2 * model.cell and abs(oy + 3 * STEP) <= 2 * model.cell


def test_a_still_world_keeps_the_camera_at_zero():
    world, model = Scroller("pinned"), SelfModel()
    frame = world.frame()
    for _ in range(4):
        after = world.step("none")  # nothing pressed: nothing moves
        model.observe(frame, None, after)
        frame = after
    assert model.camera.offset == (0, 0)


def test_the_map_gives_one_world_square_the_same_address_after_the_screen_scrolled():
    world, nav = Scroller("pinned"), NavMap()
    frame = world.frame()
    nav.begin(frame, {})
    for key in ["right", "left", "down", "up", "right", "right", "right"]:
        after = world.step(key)
        nav.update(frame, key, after, {}, False)
        frame = after
    assert nav.me.known and nav.step() is not None, "the buddy should have found itself by now"
    # "me" is at world (400 + 48, 320). A world point 6 blocks to the right of "me" is a fixed square, whatever the camera did.
    sx = world.me[0] - world.cam[0] + SPRITE / 2
    sy = world.me[1] - world.cam[1] + SPRITE / 2
    here = nav.square((sx, sy))
    ahead = nav.square((sx + 6 * STEP, sy))
    assert ahead[0] - here[0] == round(6 * STEP / nav.step()[0])
    assert ahead[1] == here[1]
    # and going back from the address to the screen finds the same spot
    cx, cy = nav.center_of(ahead)
    assert abs(cx - (sx + 6 * STEP)) <= nav.step()[0] / 2 and abs(cy - sy) <= nav.step()[1] / 2


def test_the_same_world_spot_is_the_same_square_before_and_after_a_scroll():
    world, nav = Scroller("pinned"), NavMap()
    frame = world.frame()
    nav.begin(frame, {})
    for key in ["right", "left", "down", "up", "right", "left", "down", "up"]:
        after = world.step(key)
        nav.update(frame, key, after, {}, False)
        frame = after
    assert nav.me.known and nav.step() is not None, "the buddy should have found itself by now"
    spot_world = (world.me[0] + 5 * STEP + SPRITE / 2, world.me[1] + SPRITE / 2)  # a fixed point in the world
    sq_before = nav.square((spot_world[0] - world.cam[0], spot_world[1] - world.cam[1]))
    for key in ["right", "right", "right"]:
        after = world.step(key)
        nav.update(frame, key, after, {}, False)
        frame = after
    sq_after = nav.square((spot_world[0] - world.cam[0], spot_world[1] - world.cam[1]))
    assert sq_before == sq_after


def test_an_edge_following_camera_keeps_the_square_addresses_too():
    world, nav = Scroller("edge"), NavMap()
    frame = world.frame()
    nav.begin(frame, {})
    for key in ["right", "left", "down", "up"] + ["right"] * 10:
        after = world.step(key)
        nav.update(frame, key, after, {}, False)
        frame = after
    assert nav.me.known and nav.step() is not None, "the buddy should have found itself by now"
    spot_world = (400 + SPRITE / 2, 320 + SPRITE / 2)  # where "me" started
    here = nav.square((world.me[0] - world.cam[0] + SPRITE / 2, world.me[1] - world.cam[1] + SPRITE / 2))
    start = nav.square((spot_world[0] - world.cam[0], spot_world[1] - world.cam[1]))
    assert here[0] - start[0] == round(10 * STEP / nav.step()[0])
