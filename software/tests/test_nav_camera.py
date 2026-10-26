"""Camera ego-motion (T32b): the buddy tells the background scrolling apart from "me" moving."""
import numpy as np
import pytest

from nav import frames
from nav.camera import CameraTracker, align, estimate_grid_shift

W, H = 320, 240


def shift(prev_frame, curr_frame):
    """(dx, dy, confidence) in pixels from two whole pictures, the way the self-model measures it on its blocks."""
    cell = frames.cell_size(np.asarray(curr_frame).shape)
    dc, dr, confidence = estimate_grid_shift(frames.shrink(prev_frame, cell), frames.shrink(curr_frame, cell))
    return dc * cell, dr * cell, confidence


def world(seed=0):
    """A wide, detailed picture to scroll over (random coloured blocks, like tiles)."""
    rng = np.random.default_rng(seed)
    tiles = rng.integers(40, 220, size=(60, 120, 3)).astype(np.uint8)
    return np.kron(tiles, np.ones((8, 8, 1), dtype=np.uint8))  # 480 × 960 px


def view(big, x, y, sprite=(150, 110), hud=True):
    frame = big[y:y + H, x:x + W].copy()
    sx, sy = sprite
    frame[sy:sy + 16, sx:sx + 16] = (255, 0, 255)  # "me", pinned in the middle like a following camera
    if hud:
        frame[:12, :80] = (255, 255, 255)  # a still HUD strip
    return frame


@pytest.mark.parametrize("dx,dy", [(8, 0), (-8, 0), (0, 8), (0, -16), (16, 8), (-16, -8)])
def test_finds_the_scroll(dx, dy):
    big = world()
    before = view(big, 200, 120)
    after = view(big, 200 - dx, 120 - dy)  # the camera looked the other way: the background slid by (dx, dy)
    found_dx, found_dy, confidence = shift(before, after)
    cell = frames.cell_size(before.shape)
    assert abs(found_dx - dx) <= cell and abs(found_dy - dy) <= cell
    assert confidence >= 0.45


def test_a_still_camera_gives_zero_with_full_confidence():
    big = world()
    before = view(big, 200, 120, sprite=(150, 110))
    after = view(big, 200, 120, sprite=(158, 110))  # only the sprite moved
    assert shift(before, after) == (0, 0, 1.0)


def test_flat_background_says_nothing():
    flat = np.full((H, W, 3), 90, dtype=np.uint8)
    other = flat.copy()
    other[:, : W // 2] = (200, 200, 200)  # a big change but no detail to match
    assert shift(flat, other)[2] < 0.45


def test_the_sprite_and_hud_do_not_fool_it():
    big = world(1)
    before = view(big, 300, 100)
    after = view(big, 292, 100)
    dx, _, confidence = shift(before, after)
    assert dx == 8 // frames.cell_size(before.shape) * frames.cell_size(before.shape) or abs(dx - 8) <= 3
    assert confidence >= 0.45


def test_align_removes_the_background_change():
    big = world()
    before, after = view(big, 200, 120), view(big, 192, 120)
    cell = frames.cell_size(before.shape)
    a, b = frames.shrink(before, cell), frames.shrink(after, cell)
    dc, dr, conf = estimate_grid_shift(a, b)
    assert frames.changed(a, b).mean() > 0.25  # without help the scroll looks like a new scene
    assert frames.changed(align(a, b, dc, dr), b).mean() < 0.05


def test_tracker_adds_up_and_converts_to_world_squares():
    tracker = CameraTracker()
    assert tracker.update(8, 0, 0.9) == (8, 0)
    assert tracker.update(8, 0, 0.9) == (8, 0)
    assert tracker.update(24, 0, 0.2) == (0, 0)  # a shaky estimate is ignored
    assert tracker.offset == (16, 0)
    assert tracker.to_world((100, 50)) == (84, 50)
    tracker.reset()
    assert tracker.offset == (0, 0)


def test_mismatched_pictures_are_not_guessed():
    assert estimate_grid_shift(np.zeros((10, 10, 3)), np.zeros((12, 10, 3))) == (0, 0, 0.0)


def test_the_coarse_to_fine_search_is_fast_and_finds_nearly_every_slide():
    import time
    big = world()
    before = view(big, 200, 120)
    cell = frames.cell_size(before.shape)
    a = frames.shrink(before, cell)
    wrong = total = 0
    for dx in range(-20, 21, 4):
        for dy in range(-20, 21, 4):
            if (dx, dy) == (0, 0):
                continue
            b = frames.shrink(view(big, 200 - dx, 120 - dy), cell)
            dc, dr, _ = estimate_grid_shift(a, b)
            total += 1
            wrong += (dc * cell, dr * cell) != (dx, dy)
    assert wrong <= max(1, total // 50)  # a rare 'unsure' on a hard in-between slide is fine; a wrong guess is not
    b = frames.shrink(view(big, 192, 112), cell)
    started = time.perf_counter()
    for _ in range(10):
        estimate_grid_shift(a, b)
    assert (time.perf_counter() - started) / 10 < 0.020  # the old full search took 45 ms


def _all_slides(estimate):
    big = world()
    before = view(big, 200, 120)
    cell = frames.cell_size(before.shape)
    a = frames.shrink(before, cell)
    wrong = []
    for dx in range(-20, 21, 4):
        for dy in range(-20, 21, 4):
            if (dx, dy) == (0, 0):
                continue
            dc, dr, _ = estimate(a, frames.shrink(view(big, 200 - dx, 120 - dy), cell))
            if (dc * cell, dr * cell) != (dx, dy):
                wrong.append((dx, dy))
    return wrong


def test_without_the_native_helper_the_python_search_still_works(monkeypatch):
    from nav import camera
    monkeypatch.setattr(camera, "native_bridge", None)
    monkeypatch.setattr(frames, "native_bridge", None)
    assert len(_all_slides(estimate_grid_shift)) <= 2
