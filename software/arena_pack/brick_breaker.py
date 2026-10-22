"""Arena level 13 (arcade): Brick Breaker. Move the paddle with left/right, launch the ball with space, and break
every brick. Two levels (the second has tougher bricks). 3 lives. / المستوى 13: حرّك المضرب واكسر كل الطوب. مستويان و3 أرواح."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

W, H = 480, 400
PADDLE_W, PADDLE_H, PADDLE_Y, PADDLE_STEP = 80, 10, 372, 12
BALL_R, BALL_SPEED = 6, 6.0
BRICK_W, BRICK_H, BRICK_TOP = 56, 18, 50
LEVELS = (("11111111", "11111111", "11111111", "11111111"),
          ("22222222", "12111121", "11111111", "21111112", "11111111"))
LIVES = 3
BRICK_POINTS = 10
COLOURS = {1: (34, 228, 255), 2: (255, 61, 203)}
BACK, PADDLE, BALL = (20, 18, 40), (124, 92, 255), (255, 255, 255)


class BrickBreaker(ArenaPackGame):
    name = "Brick Breaker"
    name_ar = "كاسر الطوب"
    keys = ("left", "right", "space")
    max_ticks = 15000
    state_fields = ("paddle", "ball", "velocity", "launched", "bricks", "level_no", "lives", "cleared")

    def new_game(self) -> None:
        self.level_no = 0
        self.lives = LIVES
        self.cleared = False
        self.paddle = W / 2
        self._load()

    def _load(self) -> None:
        offset = (W - 8 * BRICK_W) / 2
        self.bricks = {(offset + c * BRICK_W, BRICK_TOP + r * BRICK_H): int(hp)
                       for r, row in enumerate(LEVELS[self.level_no]) for c, hp in enumerate(row) if hp != "0"}
        self._serve()

    def _serve(self) -> None:
        self.launched = False
        self.ball = [self.paddle, PADDLE_Y - BALL_R - 1]
        self.velocity = [0.0, 0.0]

    def on_key(self, key: str) -> None:
        if key in ("left", "right"):
            self.paddle = min(W - PADDLE_W / 2, max(PADDLE_W / 2, self.paddle + (PADDLE_STEP if key == "right" else -PADDLE_STEP)))
            if not self.launched:
                self.ball[0] = self.paddle
        elif key == "space" and not self.launched:
            angle = math.radians(self.rng.uniform(-35, 35))
            self.velocity = [BALL_SPEED * math.sin(angle), -BALL_SPEED * math.cos(angle)]
            self.launched = True

    def tick(self) -> None:
        if not self.launched:
            return
        for _ in range(3):  # small sub-steps so the ball never skips through a brick
            self._move(self.velocity[0] / 3, self.velocity[1] / 3)
            if not self.launched or self.done:
                return

    def _move(self, dx: float, dy: float) -> None:
        x, y = self.ball[0] + dx, self.ball[1] + dy
        if x < BALL_R or x > W - BALL_R:
            self.velocity[0] *= -1
            x = min(W - BALL_R, max(BALL_R, x))
        if y < BALL_R:
            self.velocity[1] *= -1
            y = BALL_R
        if self.velocity[1] > 0 and PADDLE_Y - BALL_R <= y <= PADDLE_Y + PADDLE_H and abs(x - self.paddle) <= PADDLE_W / 2 + BALL_R:
            offset = (x - self.paddle) / (PADDLE_W / 2)  # -1 left edge … 1 right edge
            angle = math.radians(60 * max(-1.0, min(1.0, offset)))
            self.velocity = [BALL_SPEED * math.sin(angle), -BALL_SPEED * math.cos(angle)]
            y = PADDLE_Y - BALL_R
        for (bx, by), hp in list(self.bricks.items()):
            if bx - BALL_R <= x <= bx + BRICK_W + BALL_R and by - BALL_R <= y <= by + BRICK_H + BALL_R:
                came_from_side = not (bx <= self.ball[0] <= bx + BRICK_W)
                self.velocity[0 if came_from_side else 1] *= -1
                if hp > 1:
                    self.bricks[(bx, by)] = hp - 1
                else:
                    del self.bricks[(bx, by)]
                    self.score += BRICK_POINTS
                x, y = self.ball[0], self.ball[1]  # bounce back from where we were
                break
        self.ball = [x, y]
        if y > H:
            self.lives -= 1
            if self.lives > 0:
                self._serve()
        elif not self.bricks:
            if self.level_no + 1 < len(LEVELS):
                self.level_no += 1
                self._load()
            else:
                self.cleared = True

    def seconds_per_tick(self) -> float:
        return 1 / 30

    def is_won(self) -> bool:
        return self.cleared

    def is_lost(self) -> bool:
        return self.lives <= 0

    def progress(self) -> dict:
        return {"level": self.level_no + 1, "levels": len(LEVELS), "bricks_left": len(self.bricks), "lives": self.lives,
                "launched": self.launched, "ball": (round(self.ball[0]), round(self.ball[1])), "paddle": round(self.paddle)}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"المستوى {self.level_no + 1}/{len(LEVELS)}    الطوب {len(self.bricks)}    الأرواح {self.lives}"
        return f"Level {self.level_no + 1}/{len(LEVELS)}    Bricks {len(self.bricks)}    Lives {self.lives}"

    def board_pixels(self) -> tuple[int, int]:
        return W, H

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        draw.rectangle((0, top, W, top + H), fill=BACK)
        for (bx, by), hp in self.bricks.items():
            draw.rectangle((bx + 2, top + by + 2, bx + BRICK_W - 2, top + by + BRICK_H - 2), fill=COLOURS[min(hp, 2)])
        draw.rounded_rectangle((self.paddle - PADDLE_W / 2, top + PADDLE_Y, self.paddle + PADDLE_W / 2, top + PADDLE_Y + PADDLE_H),
                               radius=5, fill=PADDLE)
        x, y = self.ball
        draw.ellipse((x - BALL_R, top + y - BALL_R, x + BALL_R, top + y + BALL_R), fill=BALL)


if __name__ == "__main__":
    run_window(BrickBreaker, "PlayFellow Arena: Brick Breaker")
