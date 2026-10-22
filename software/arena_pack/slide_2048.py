"""Arena level 1 (puzzle): Slide 2048. Slide the tiles with the arrows; equal tiles join. Reach 256 to win.
/ المستوى 1 (ألغاز): حرّك المربعات بالأسهم؛ المربعات المتساوية تندمج. صِل إلى 256 لتفوز."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, font, run_window

SIZE = 4
TILE = 96
GAP = 8
TARGET = 256
COLOURS = {0: (44, 44, 70), 2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121), 16: (245, 149, 99),
           32: (246, 124, 95), 64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97)}
DARK_TEXT, LIGHT_TEXT = (60, 58, 50), (255, 255, 255)


def slide_row(row: list[int]) -> tuple[list[int], int]:
    """Slide one row to the left and join equal neighbours once. Returns the new row and the points earned."""
    tiles = [value for value in row if value]
    out, points, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            points += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (len(row) - len(out)), points


class Slide2048(ArenaPackGame):
    name = "Slide 2048"
    name_ar = "انزلاق 2048"
    keys = ("up", "down", "left", "right")

    def new_game(self) -> None:
        self.board = [[0] * SIZE for _ in range(SIZE)]
        self.moves = 0
        self._spawn()
        self._spawn()

    def _spawn(self) -> None:
        empty = [(x, y) for y in range(SIZE) for x in range(SIZE) if not self.board[y][x]]
        if empty:
            x, y = self.rng.choice(empty)
            self.board[y][x] = 4 if self.rng.random() < 0.1 else 2

    def _moved(self, key: str) -> tuple[list[list[int]], int]:
        """The board after sliding towards `key`, and the points earned (the board itself is not changed)."""
        grid = [row[:] for row in self.board]
        if key in ("up", "down"):
            grid = [list(col) for col in zip(*grid)]  # columns become rows
        if key in ("right", "down"):
            grid = [row[::-1] for row in grid]
        points = 0
        for i, row in enumerate(grid):
            grid[i], earned = slide_row(row)
            points += earned
        if key in ("right", "down"):
            grid = [row[::-1] for row in grid]
        if key in ("up", "down"):
            grid = [list(col) for col in zip(*grid)]
        return grid, points

    def can_move(self, key: str) -> bool:
        return self._moved(key)[0] != self.board

    def on_key(self, key: str) -> None:
        grid, points = self._moved(key)
        if grid == self.board:
            return  # nothing slid: not a move, no new tile
        self.board = grid
        self.score += points
        self.moves += 1
        self._spawn()

    @property
    def best_tile(self) -> int:
        return max(max(row) for row in self.board)

    def is_won(self) -> bool:
        return self.best_tile >= TARGET

    def is_lost(self) -> bool:
        return not self.is_won() and not any(self.can_move(key) for key in self.keys)

    def progress(self) -> dict:
        return {"best_tile": self.best_tile, "target": TARGET, "moves": self.moves,
                "empty": sum(row.count(0) for row in self.board)}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"أكبر مربع: {self.best_tile} / {TARGET}"
        return f"Best tile: {self.best_tile} / {TARGET}"

    def board_pixels(self) -> tuple[int, int]:
        side = SIZE * TILE + (SIZE + 1) * GAP
        return side, side

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        tile_font = font(30)
        for y, row in enumerate(self.board):
            for x, value in enumerate(row):
                left, upper = GAP + x * (TILE + GAP), top + GAP + y * (TILE + GAP)
                draw.rounded_rectangle((left, upper, left + TILE, upper + TILE), radius=10, fill=COLOURS.get(value, (60, 58, 50)))
                if value:
                    text = str(value)
                    box = draw.textbbox((0, 0), text, font=tile_font)
                    draw.text((left + (TILE - box[2]) / 2, upper + (TILE - box[3]) / 2 - 4), text,
                              fill=DARK_TEXT if value <= 4 else LIGHT_TEXT, font=tile_font)


if __name__ == "__main__":
    run_window(Slide2048, "PlayFellow Arena: Slide 2048")
