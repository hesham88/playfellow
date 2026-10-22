"""Telling the world scrolling apart from "me" moving. / التمييز بين حركة الكاميرا وحركتي.

Like riding in a train: when the trees outside slide past, you know it is the TRAIN moving, not every tree at once.
In many games the camera follows the player, so after one step the whole background has slid sideways. The buddy
looks at many small patches of the picture; if most of the patches that have some detail moved by the same amount,
that shared slide is the camera. Taking it away leaves only what really moved in the world (me, enemies, coins).

Nothing about any game is written in: it only compares two pictures.
"""
from __future__ import annotations

import numpy as np

from nav import frames

try:  # the optional fast helper (T38): same answer as the search below, in native code
    import native_bridge
except ImportError:  # pragma: no cover
    native_bridge = None

MAX_SHIFT = 10  # the camera moves at most this many blocks in one step
PATCH = 8  # patches are PATCH × PATCH blocks
DETAIL = 6.0  # a patch needs this much variation in grey to be worth matching (flat sky tells nothing)
MATCH = 8.0  # a patch "found its place" when the mean grey gap there is below this
STATIC = 0.05  # fewer blocks than this share changed: the camera stood still (only a sprite moved)
SURE = 0.45  # at least this share of the detailed patches must agree before we believe in a camera move


def _grey(grid: np.ndarray) -> np.ndarray:
    grey = grid.mean(axis=2) if grid.ndim == 3 else grid
    return grey.astype(np.float32, copy=False)


def _patch_gaps(a: np.ndarray, b: np.ndarray, offsets: list[tuple[int, int]], reach: int, patch: int) -> tuple:
    """For every patch cut from the middle of `b`: how well it matches `a` slid by each offset (rows, cols).
    `reach` is how far an offset may go, so every candidate still lies inside `a`."""
    rows, cols = b.shape
    inner_rows, inner_cols = (rows - 2 * reach) // patch, (cols - 2 * reach) // patch
    if inner_rows < 1 or inner_cols < 1:
        return None
    r0, c0 = reach, reach
    r1, c1 = r0 + inner_rows * patch, c0 + inner_cols * patch
    now = b[r0:r1, c0:c1]
    detail = now.reshape(inner_rows, patch, inner_cols, patch).std(axis=(1, 3)) > DETAIL
    share = np.float32(1.0 / (patch * patch))
    gaps = np.empty((len(offsets), inner_rows, inner_cols), dtype=np.float32)
    for k, (dr, dc) in enumerate(offsets):
        was = a[r0 - dr:r1 - dr, c0 - dc:c1 - dc]  # what was here before the background slid by (dr, dc)
        gaps[k] = np.abs(now - was).reshape(inner_rows, patch, inner_cols, patch).sum(axis=3).sum(axis=1) * share
    return detail, gaps


def _vote(detail: np.ndarray, gaps: np.ndarray, offsets: list[tuple[int, int]],
          strict: bool = True) -> tuple[int, float] | None:
    """The offset most detailed patches agree on (index into `offsets`) and the share of patches that agree.
    `strict`: a patch only votes when it really found its place (the coarse looks just point the way)."""
    best = gaps.argmin(axis=0)
    voters = detail & (gaps.min(axis=0) < MATCH) if strict else detail
    if not voters.any():
        return None
    votes: dict[int, int] = {}
    for k in best[voters]:
        votes[int(k)] = votes.get(int(k), 0) + 1
    winner = max(votes, key=lambda k: (votes[k], -k))  # most votes; then the smaller move (offsets are sorted)
    return winner, votes[winner] / int(detail.sum())


def _smallest_first(offsets) -> list[tuple[int, int]]:
    return sorted(offsets, key=lambda o: (abs(o[0]) + abs(o[1]), o))  # ties go to "didn't move"


def _halve(grid: np.ndarray) -> np.ndarray:
    rows, cols = grid.shape[0] // 2 * 2, grid.shape[1] // 2 * 2
    return grid[:rows, :cols].reshape(rows // 2, 2, cols // 2, 2).mean(axis=(1, 3))


def estimate_grid_shift(before: np.ndarray, after: np.ndarray) -> tuple[int, int, float]:
    """(dc, dr, confidence): how far the background slid between two shrunk pictures, in blocks.

    A positive dc means the background moved right (the camera looked left); dr likewise downwards.
    Confidence is the share of detailed patches that agree; (0, 0, 1.0) when nothing but a sprite changed,
    (0, 0, 0.0) when there is too little detail to say.

    Coarse to fine, like finding a street on a map: a quarter-size look over the whole range, then a half-size and a
    full-size 3×3 look around the winner (about 7 times fewer comparisons than trying every offset at full size)."""
    if before.shape != after.shape:
        return 0, 0, 0.0
    if frames.changed_share(before, after) < STATIC:
        return 0, 0, 1.0
    a1, b1 = _grey(before), _grey(after)
    found = _pyramid_shift(a1, b1)
    if found[2] >= SURE or native_bridge is None:
        return found
    # The quick look was unsure (a slide that falls between the coarse blocks): the native helper can afford to
    # try every offset at full size. Without it, "unsure" simply means "treat the camera as still".
    slow = native_bridge.grid_shift(a1, b1, MAX_SHIFT, PATCH, DETAIL, MATCH)
    if slow is None:
        return found
    status, dc, dr, confidence = slow
    return (dc, dr, confidence) if status == native_bridge.PF_OK and confidence > found[2] else found


def _pyramid_shift(a1: np.ndarray, b1: np.ndarray) -> tuple[int, int, float]:
    """The quick coarse-to-fine look (Python), on two grey grids."""
    a2, b2 = _halve(a1), _halve(b1)
    a4, b4 = _halve(a2), _halve(b2)
    reach = -(-MAX_SHIFT // 4)  # the quarter-size picture covers the whole range in this many blocks
    guess = (0, 0)
    for scale, (x, y), patch in ((4, (a4, b4), PATCH // 4), (2, (a2, b2), PATCH // 2), (1, (a1, b1), PATCH)):
        span = reach if scale == 4 else (2 if scale == 2 else 1)  # a coarse miss of 1 block is 2 at the next size
        offsets = _smallest_first((guess[0] + r, guess[1] + c) for r in range(-span, span + 1)
                                  for c in range(-span, span + 1)
                                  if abs(guess[0] + r) * scale <= MAX_SHIFT and abs(guess[1] + c) * scale <= MAX_SHIFT)
        room = max(max(abs(r), abs(c)) for r, c in offsets)
        found = _patch_gaps(x, y, offsets, room, patch)
        if found is None:
            return 0, 0, 0.0
        picked = _vote(found[0], found[1], offsets, strict=scale == 1)
        if picked is None:
            return 0, 0, 0.0
        guess = offsets[picked[0]]
        if scale > 1:
            guess = (guess[0] * 2, guess[1] * 2)
    return int(guess[1]), int(guess[0]), float(picked[1])


def align(before: np.ndarray, after: np.ndarray, dc: int, dr: int) -> np.ndarray:
    """`before` slid by (dc, dr) blocks into the camera's new place. Blocks that slid in from outside the old
    picture are copied from `after`, so they never look like something that changed."""
    out = after.copy()
    rows, cols = before.shape[:2]
    src_r0, src_r1 = max(0, -dr), min(rows, rows - dr)
    src_c0, src_c1 = max(0, -dc), min(cols, cols - dc)
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[src_r0 + dr:src_r1 + dr, src_c0 + dc:src_c1 + dc] = before[src_r0:src_r1, src_c0:src_c1]
    return out


class CameraTracker:
    """Follows the camera over many steps: how far the background slid since the start (so a world square keeps its
    address while the screen scrolls)."""

    def __init__(self):
        self.offset: tuple[int, int] = (0, 0)  # total background slide since the start, in pixels
        self.last: tuple[int, int, float] = (0, 0, 1.0)  # (dx, dy, confidence) of the latest step, in pixels

    def estimate_shift(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> tuple[int, int, float]:
        """(dx, dy, confidence): how far the background slid between two pictures, in pixels."""
        cell = frames.cell_size(np.asarray(curr_frame).shape)
        dc, dr, confidence = estimate_grid_shift(frames.shrink(prev_frame, cell), frames.shrink(curr_frame, cell))
        return dc * cell, dr * cell, confidence

    def update(self, dx: int, dy: int, confidence: float) -> tuple[int, int]:
        """Take one step's estimate in; a shaky one (below SURE) is ignored. Returns the shift that was used."""
        used = (dx, dy) if confidence >= SURE else (0, 0)
        self.last = (used[0], used[1], confidence)
        self.offset = (self.offset[0] + used[0], self.offset[1] + used[1])
        return used

    def reset(self) -> None:
        self.offset = (0, 0)
        self.last = (0, 0, 1.0)

    def to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        """A screen point's address in the world (the screen point minus how far the background has slid)."""
        return point[0] - self.offset[0], point[1] - self.offset[1]

    def to_screen(self, point: tuple[float, float]) -> tuple[float, float]:
        return point[0] + self.offset[0], point[1] + self.offset[1]
