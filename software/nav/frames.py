"""A small built-in picture encoder for navigation (until the T24 encoder lands). / مُرمِّز صور صغير للتنقل.

Like squinting at the screen: the picture is shrunk into little blocks (each block keeps its average colour), so
small details and noise disappear but things that move are still easy to see.
"""
from __future__ import annotations

import numpy as np

try:  # the optional fast helper (T38); every use below falls back to plain Python when it is not built
    import native_bridge
except ImportError:  # pragma: no cover
    native_bridge = None

GRID_TARGET = 160  # blocks across the longer side of the shrunk picture
CHANGE = 24  # a block "changed" when one colour channel moved by more than this (0-255)
THUMB_W, THUMB_H = 32, 24  # the tiny grey picture used to tell scenes (rooms, levels) apart


def cell_size(shape: tuple[int, ...]) -> int:
    """How many pixels one block covers, for a picture of this shape."""
    return max(1, round(max(shape[0], shape[1]) / GRID_TARGET))


def shrink(frame: np.ndarray, cell: int) -> np.ndarray:
    """Average colour per cell × cell block → float32 array (rows, cols, channels). Uses the native helper when it is
    built (measured 2.6x to 4.7x faster at every size from 320x240 to 1920x1080), else `shrink_numpy`."""
    frame = np.asarray(frame)
    fast = native_bridge.shrink(frame, cell) if native_bridge is not None else None
    return fast if fast is not None else shrink_numpy(frame, cell)


def shrink_numpy(frame: np.ndarray, cell: int) -> np.ndarray:
    """The same, in plain NumPy only: the reference the native version is checked and timed against.

    For ordinary 8-bit pictures the sums are whole numbers (at most 255 * cell * cell), so adding them as integers gives
    exactly the same numbers as adding them as floats, about 1.3x faster."""
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = frame[:, :, None]
    rows, cols = frame.shape[0] // cell, frame.shape[1] // cell
    frame = frame[: rows * cell, : cols * cell]
    if frame.dtype == np.uint8:
        sums = np.add.reduceat(np.add.reduceat(frame, np.arange(0, rows * cell, cell), axis=0, dtype=np.uint32),
                               np.arange(0, cols * cell, cell), axis=1).astype(np.float32)
    else:
        frame = frame.astype(np.float32)
        sums = np.add.reduceat(np.add.reduceat(frame, np.arange(0, rows * cell, cell), axis=0),
                               np.arange(0, cols * cell, cell), axis=1)
    return sums / (cell * cell)


def changed_share(before: np.ndarray, after: np.ndarray) -> float:
    """The share of blocks that changed (same as changed(...).mean(), which is the NumPy reference; the native helper
    measured about 10x faster)."""
    fast = native_bridge.changed_fraction(before, after, CHANGE) if native_bridge is not None else None
    return fast if fast is not None else float(changed(before, after).mean())


def changed(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """True for every block whose colour changed between two shrunk pictures."""
    return np.abs(after - before).max(axis=2) > CHANGE


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """A tiny grey version of the picture (THUMB_H × THUMB_W), for recognising the same scene again."""
    frame = np.asarray(frame, dtype=np.float32)
    grey = frame.mean(axis=2) if frame.ndim == 3 else frame
    rows = np.linspace(0, grey.shape[0], THUMB_H + 1).astype(int)
    cols = np.linspace(0, grey.shape[1], THUMB_W + 1).astype(int)
    return np.array([[grey[rows[r]:rows[r + 1], cols[c]:cols[c + 1]].mean() for c in range(THUMB_W)]
                     for r in range(THUMB_H)], dtype=np.float32)


def thumb_distance(a: np.ndarray, b: np.ndarray) -> float:
    """How different two thumbnails are (mean grey difference, 0-255)."""
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean())


def crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Cut a box (col0, row0, col1, row1 — end exclusive, in blocks) out of a shrunk picture."""
    c0, r0, c1, r1 = box
    return grid[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]


def match(grid: np.ndarray, template: np.ndarray, near: tuple[int, int] | None = None,
          reach: int | None = None) -> tuple[tuple[int, int], float]:
    """Find where `template` fits best in `grid` (top-left (col, row) in blocks) and how well (mean colour gap).

    With `near` and `reach`, only look within `reach` blocks of `near` (fast and avoids look-alikes far away)."""
    th, tw = template.shape[:2]
    gh, gw = grid.shape[:2]
    if th == 0 or tw == 0 or th > gh or tw > gw:
        return (0, 0), float("inf")
    c_lo, r_lo, c_hi, r_hi = 0, 0, gw - tw, gh - th
    if near is not None and reach is not None:
        c_lo, c_hi = max(c_lo, near[0] - reach), min(c_hi, near[0] + reach)
        r_lo, r_hi = max(r_lo, near[1] - reach), min(r_hi, near[1] + reach)
        if c_lo > c_hi or r_lo > r_hi:
            return (0, 0), float("inf")
    area = grid[r_lo:r_hi + th, c_lo:c_hi + tw]
    windows = np.lib.stride_tricks.sliding_window_view(area, (th, tw), axis=(0, 1))
    # windows: (rows, cols, channels, th, tw) → compare with the template laid out the same way
    gaps = np.abs(windows - np.moveaxis(template, 2, 0)[None, None]).mean(axis=tuple(range(2, windows.ndim)))
    r, c = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
    return (c_lo + int(c), r_lo + int(r)), float(gaps[r, c])
