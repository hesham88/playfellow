"""Arena level 9 (arcade): Snake. Steer with the arrows, eat the food, don't hit a wall or yourself.
Reach length 20 to win. The snake speeds up as it grows. / المستوى 9: وجّه الثعبان بالأسهم وكُل الطعام. الطول 20 يفوز."""
from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

GRID = 10
TILE = 44
TARGET = 20
FOOD_POINTS = 10
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
FLOOR, FLOOR_LINE = (30, 34, 58), (40, 46, 74)
BODY, HEAD, FOOD = (110, 220, 120), (182, 255, 59), (255, 61, 203)


@lru_cache(maxsize=1)
def _floor() -> Image.Image:
    """The empty grid, drawn once (redrawing 100 squares every tick made each step slow)."""
    image = Image.new("RGB", (GRID * TILE, GRID * TILE))
    draw = ImageDraw.Draw(image)
    for y in range(GRID):
        for x in range(GRID):
            draw.rectangle((x * TILE, y * TILE, x * TILE + TILE - 1, y * TILE + TILE - 1), fill=FLOOR, outline=FLOOR_LINE)
    return image


class Snake(ArenaPackGame):
    name = "Snake"
    name_ar = "الثعبان"
    keys = ("up", "down", "left", "right")
    max_ticks = 3000

    def new_game(self) -> None:
        middle = GRID // 2
        self.body = [(middle, middle), (middle - 1, middle), (middle - 2, middle)]  # head first
        self.direction = "right"
        self.turn = "right"  # the direction for the next tick (one turn per tick)
        self.crashed = False
        self.eaten = 0
        self._place_food()

    def _place_food(self) -> None:
        free = [(x, y) for y in range(GRID) for x in range(GRID) if (x, y) not in self.body]
        self.food = self.rng.choice(free) if free else None

    def on_key(self, key: str) -> None:
        dx, dy = DIRECTIONS[key]
        cx, cy = DIRECTIONS[self.direction]
        if (dx, dy) != (-cx, -cy):  # no turning straight back into yourself
            self.turn = key

    def tick(self) -> None:
        self.direction = self.turn
        dx, dy = DIRECTIONS[self.direction]
        head = (self.body[0][0] + dx, self.body[0][1] + dy)
        grows = head == self.food
        body = self.body if grows else self.body[:-1]  # the tail moves away this tick unless we grow
        if not (0 <= head[0] < GRID and 0 <= head[1] < GRID) or head in body:
            self.crashed = True
            return
        self.body = [head] + body
        if grows:
            self.eaten += 1
            self.score += FOOD_POINTS
            if not self.is_won():
                self._place_food()

    def seconds_per_tick(self) -> float:
        return max(0.08, 0.25 - 0.01 * len(self.body))

    def is_won(self) -> bool:
        return len(self.body) >= TARGET

    def is_lost(self) -> bool:
        return self.crashed

    def progress(self) -> dict:
        return {"length": len(self.body), "target": TARGET, "eaten": self.eaten, "direction": self.direction}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"الطول: {len(self.body)} / {TARGET}"
        return f"Length: {len(self.body)} / {TARGET}"

    def board_pixels(self) -> tuple[int, int]:
        return GRID * TILE, GRID * TILE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        image.paste(_floor(), (0, top))
        if self.food:
            x, y = self.food
            draw.ellipse((x * TILE + 9, top + y * TILE + 9, x * TILE + TILE - 9, top + y * TILE + TILE - 9), fill=FOOD)
        for i, (x, y) in enumerate(self.body):
            draw.rounded_rectangle((x * TILE + 3, top + y * TILE + 3, x * TILE + TILE - 3, top + y * TILE + TILE - 3),
                                   radius=8, fill=HEAD if i == 0 else BODY)


if __name__ == "__main__":
    run_window(Snake, "PlayFellow Arena: Snake")
