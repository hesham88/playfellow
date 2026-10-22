"""P0: the top-bar reading (nav/hud_bar.py) that lets the map learn pickups, and that the quick tick now feeds it."""
from __future__ import annotations

import time

import numpy as np
import pytest

from nav.hud_bar import WARMUP, HudBar

H, W = 360, 480


def frame_with_bar(score_marks=0, message=False, seed=0):
    """A game picture: a quiet ground, a top bar with `score_marks` little marks on its first line, and optionally a
    message on the second line of the bar."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 60, np.uint8)
    img[60:] = rng.integers(40, 200, (H - 60, W, 3), dtype=np.uint8)  # the playfield (the bar is the top ~12%)
    for i in range(score_marks):
        img[8:20, 10 + i * 20:22 + i * 20] = (250, 250, 0)
    if message:
        img[28:40, 10:150] = (255, 255, 255)
    return img


def warm_up(bar, n=WARMUP + 2, **kw):
    for _ in range(n):
        bar.read(frame_with_bar(**kw))


def test_a_quiet_bar_reads_zero_forever():
    bar = HudBar()
    assert all(bar.read(frame_with_bar())["bar"] == 0 for _ in range(30))


def test_the_count_goes_up_by_one_each_time_the_quiet_part_changes():
    bar = HudBar()
    warm_up(bar)
    assert bar.read(frame_with_bar(score_marks=1))["bar"] == 1
    assert bar.read(frame_with_bar(score_marks=1))["bar"] == 1  # same again: nothing happened
    assert bar.read(frame_with_bar(score_marks=2))["bar"] == 2


def test_nothing_counts_while_it_is_still_warming_up():
    bar = HudBar()
    for marks in range(WARMUP):
        assert bar.read(frame_with_bar(score_marks=marks))["bar"] == 0  # too early to know what flickers


def test_a_line_that_flickers_all_the_time_is_learned_as_noise_and_never_counts():
    bar = HudBar()
    for i in range(40):  # a message line that flips every picture, the quiet score line never changes
        bar.read(frame_with_bar(message=i % 2 == 0))
    assert bar.events <= 1  # at most the very first flips, before it was recognised as noise
    before = bar.events
    for i in range(20):
        bar.read(frame_with_bar(message=i % 2 == 0))
    assert bar.events == before  # and now it never counts
    assert bar.read(frame_with_bar(score_marks=3, message=True))["bar"] == before + 1  # but a real change still does


def test_a_different_picture_size_starts_over():
    bar = HudBar()
    warm_up(bar)
    bar.read(frame_with_bar(score_marks=1))
    assert bar.events == 1
    assert bar.read(np.full((200, 300, 3), 70, np.uint8))["bar"] == 0 and bar.events == 0


def test_the_same_pickup_reads_the_same_at_any_picture_size():
    def run(scale):
        bar = HudBar()
        for _ in range(WARMUP + 2):
            img = frame_with_bar()
            bar.read(np.kron(img, np.ones((scale, scale, 1), np.uint8)))
        img = frame_with_bar(score_marks=1)
        return bar.read(np.kron(img, np.ones((scale, scale, 1), np.uint8)))["bar"]
    assert run(1) == run(2) == run(3) == 1


def test_unusable_pictures_do_not_crash():
    bar = HudBar()
    assert bar.read(np.zeros((3, 3, 3), np.uint8))["bar"] == 0
    assert bar.read(np.zeros((50, 50), np.uint8))["bar"] == 0  # a grey picture without a colour axis


def test_reading_a_full_size_frame_is_quick_enough_for_every_tick():
    big = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bar = HudBar()
    bar.read(big)
    started = time.perf_counter()
    for _ in range(20):
        bar.read(big)
    per_read = (time.perf_counter() - started) / 20 * 1000
    assert per_read < 20.0, f"{per_read:.1f} ms per read"  # measured 5.9 ms; the quick tick runs about 20 times a second (50 ms each)


def test_the_whole_number_sums_give_the_same_blocks_as_the_float_average():
    """The speed-up must not change what the bar reads: compare with the original float computation."""
    from nav.hud_bar import BLOCK_COLS, BLOCK_ROWS, LEVELS, SHARE
    rng = np.random.default_rng(5)
    for shape in ((360, 480, 3), (721, 1283, 3), (240, 320, 4)):
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        rows = max(8, int(shape[0] * SHARE)) // BLOCK_ROWS * BLOCK_ROWS
        cols = shape[1] // BLOCK_COLS * BLOCK_COLS
        strip = img[:rows, :cols].astype(np.float32)
        want = (strip.reshape(rows // BLOCK_ROWS, BLOCK_ROWS, cols // BLOCK_COLS, BLOCK_COLS, shape[2]).mean(axis=(1, 3)) // LEVELS).astype(np.uint8)
        sums = np.add.reduceat(np.add.reduceat(img[:rows, :cols], np.arange(0, rows, BLOCK_ROWS), axis=0, dtype=np.uint16),
                               np.arange(0, cols, BLOCK_COLS), axis=1)
        assert np.array_equal((sums // (BLOCK_ROWS * BLOCK_COLS * LEVELS)).astype(np.uint8), want), shape
