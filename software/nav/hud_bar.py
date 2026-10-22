"""Noticing that "something happened" from the top bar of the picture, with no AI reading the numbers. / ملاحظة حدوث شيء من الشريط العلوي.

Many games show the score, the coins left or the lives in a bar along the top. When a coin is picked up, that bar
changes. The buddy cannot read the numbers without an AI, but it can notice THAT the bar changed, and the map learns
from that: "the square I just stepped on made the bar change, so it was worth something".

Some parts of a bar change all the time (a message line that says "bump!", a timer). Those are learned as noise after a
few pictures and left out for good, so only rare changes count. The reading is a single number: how many times the
quiet part of the bar has changed. It goes up on a pickup, and stays still otherwise.

One HudBar belongs to whoever FEEDS the navigator, because it remembers the previous picture: if two readers shared
one, each would see the other's pictures and the count would be wrong.
"""
from __future__ import annotations

import numpy as np

SHARE = 0.12  # the top bar is this share of the picture's height
WARMUP = 6  # pictures to watch before trusting the bar: which parts of it flicker all the time?
WINDOW = 8  # steps looked at when deciding what flickers
NOISY = 3  # a block that changed this many times in those steps is flickering (a message, a timer)
BLOCK_ROWS, BLOCK_COLS, LEVELS = 4, 8, 40  # the bar is read as blocks of 4 x 8 pixels, each colour in coarse steps of 40


class HudBar:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget everything (a new picture size, a new game)."""
        self._prev: np.ndarray | None = None  # the bar as small coloured blocks, last time
        self._recent: list[np.ndarray] = []  # which blocks changed, in the last few steps
        self._ignored: np.ndarray | None = None  # lines of the bar that flickered: left out for good
        self._seen = 0
        self.events = 0  # how many times the quiet part of the bar changed

    def read(self, frame: np.ndarray) -> dict:
        """Look at one picture. Returns {"bar": events}: the number goes up when something in the quiet part changed.
        The count never goes up just because the list of noisy lines grew."""
        frame = np.asarray(frame)
        rows = max(8, int(frame.shape[0] * SHARE)) // BLOCK_ROWS * BLOCK_ROWS
        cols = frame.shape[1] // BLOCK_COLS * BLOCK_COLS
        if rows <= 0 or cols <= 0 or frame.ndim != 3:
            return {"bar": self.events}
        strip = frame[:rows, :cols]
        # Add up each 4 x 8 block as whole numbers (at most 255 * 32 = 8160, so 16 bits are enough): exactly the same
        # blocks as averaging in floats, in half the time (12.9 -> 5.9 ms on a 1920 x 1080 picture).
        if strip.dtype == np.uint8:
            sums = np.add.reduceat(np.add.reduceat(strip, np.arange(0, rows, BLOCK_ROWS), axis=0, dtype=np.uint16),
                                   np.arange(0, cols, BLOCK_COLS), axis=1)
            blocks = (sums // (BLOCK_ROWS * BLOCK_COLS * LEVELS)).astype(np.uint8)
        else:
            blocks = (strip.astype(np.float32).reshape(rows // BLOCK_ROWS, BLOCK_ROWS, cols // BLOCK_COLS, BLOCK_COLS, -1)
                      .mean(axis=(1, 3)) // LEVELS).astype(np.uint8)
        if self._prev is None or self._prev.shape != blocks.shape:
            self.reset()
            self._prev = blocks
            self._ignored = np.zeros(blocks.shape[0], bool)  # per line of blocks
            return {"bar": 0}
        moved = (blocks != self._prev).any(axis=2)
        self._prev = blocks
        self._seen += 1
        self._recent = (self._recent + [moved])[-WINDOW:]
        flickers = (np.sum(self._recent, axis=0) >= NOISY).any(axis=1)  # lines that keep changing
        self._ignored |= flickers | np.roll(flickers, 1) | np.roll(flickers, -1)  # and the lines touching them
        if self._seen > WARMUP and moved[~self._ignored].any():
            self.events += 1
        return {"bar": self.events}
