"""Arena level 10 (endless runner): Lane Dash. Run down three lanes for 60 seconds and collect 30 coins.
Left/right change lane, up jumps over low barriers, down slides under high ones. Trains block a whole lane.
/ المستوى 10: اركض في ثلاثة مسارات لمدة 60 ثانية واجمع 30 عملة. اقفز فوق الحواجز المنخفضة وانزلق تحت العالية."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

LANES = 3
HORIZON = 12  # rows ahead where new things appear
ROW_EVERY = 3  # a new row every 3 ticks
AIR = SLIDE = 4  # ticks a jump or a slide lasts
GOAL_TICKS = 600  # 60 seconds at 10 ticks a second
GOAL_COINS = 30
COIN_POINTS = 10
LANE_W, ROW_H = 120, 40
ROAD, LINE, PLAYER = (40, 40, 64), (90, 90, 130), (124, 92, 255)
COLOURS = {"train": (230, 70, 70), "low": (255, 160, 40), "high": (34, 228, 255), "coin": (255, 214, 60)}


def move(lane: int, air: int, slide: int, key: str | None) -> tuple[int, int, int]:
    """What one key press does to the runner (the rules, shared with anyone who wants to plan ahead)."""
    if key == "left":
        lane = max(0, lane - 1)
    elif key == "right":
        lane = min(LANES - 1, lane + 1)
    elif key == "up" and air == 0 and slide == 0:
        air = AIR
    elif key == "down" and air == 0:
        slide = SLIDE
    return lane, air, slide


def hit(kind: str, air: int, slide: int) -> bool:
    """Does meeting this thing end the run? Coins never do."""
    return kind == "train" or (kind == "low" and air == 0) or (kind == "high" and slide == 0)


class LaneDash(ArenaPackGame):
    name = "Lane Dash"
    name_ar = "سباق المسارات"
    keys = ("left", "right", "up", "down")
    max_ticks = GOAL_TICKS + 10
    state_fields = ("lane", "air", "slide", "items", "coins", "clock", "crashed")

    def new_game(self) -> None:
        self.lane, self.air, self.slide = 1, 0, 0
        self.items: list[list] = []  # [lane, rows ahead, kind]
        self.coins = 0
        self.clock = 0
        self.crashed = False

    def on_key(self, key: str) -> None:
        self.lane, self.air, self.slide = move(self.lane, self.air, self.slide, key)

    def tick(self) -> None:
        for item in self.items:
            item[1] -= 1
        for item in [i for i in self.items if i[1] == 0 and i[0] == self.lane]:
            if item[2] == "coin":
                self.coins += 1
                self.score += COIN_POINTS
                self.items.remove(item)
            elif hit(item[2], self.air, self.slide):
                self.crashed = True
                return
        self.items = [i for i in self.items if i[1] >= 0]
        self.air, self.slide = max(0, self.air - 1), max(0, self.slide - 1)
        self.clock += 1
        if self.clock % ROW_EVERY == 0:
            self._new_row()

    def _new_row(self) -> None:
        """Up to two obstacles, always leaving at least one lane completely free, plus maybe a coin."""
        lanes = list(range(LANES))
        self.rng.shuffle(lanes)
        free = lanes[0]
        for lane in lanes[1:]:
            roll = self.rng.random()
            if roll < 0.25:
                self.items.append([lane, HORIZON, "train"])
            elif roll < 0.4:
                self.items.append([lane, HORIZON, "low"])
            elif roll < 0.55:
                self.items.append([lane, HORIZON, "high"])
        if self.rng.random() < 0.7:
            self.items.append([free if self.rng.random() < 0.6 else self.rng.randrange(LANES), HORIZON, "coin"])
            if any(i[0] == self.items[-1][0] and i[1] == HORIZON and i[2] != "coin" for i in self.items[:-1]):
                self.items.pop()  # no coin inside an obstacle

    def seconds_per_tick(self) -> float:
        return 0.1

    def is_won(self) -> bool:
        return self.clock >= GOAL_TICKS and self.coins >= GOAL_COINS and not self.crashed

    def is_lost(self) -> bool:
        return self.crashed or (self.clock >= GOAL_TICKS and self.coins < GOAL_COINS)

    def progress(self) -> dict:
        return {"seconds": self.clock // 10, "goal_seconds": GOAL_TICKS // 10, "coins": self.coins,
                "goal_coins": GOAL_COINS, "lane": self.lane, "jumping": self.air > 0, "sliding": self.slide > 0}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"الوقت: {self.clock // 10}/{GOAL_TICKS // 10} ث    العملات: {self.coins}/{GOAL_COINS}"
        return f"Time: {self.clock // 10}/{GOAL_TICKS // 10} s    Coins: {self.coins}/{GOAL_COINS}"

    def board_pixels(self) -> tuple[int, int]:
        return LANES * LANE_W, (HORIZON + 1) * ROW_H

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        width, height = self.board_pixels()
        draw.rectangle((0, top, width, top + height), fill=ROAD)
        for lane in range(1, LANES):
            draw.line((lane * LANE_W, top, lane * LANE_W, top + height), fill=LINE, width=3)
        bottom = top + HORIZON * ROW_H  # the runner's row
        for lane, ahead, kind in self.items:
            left, upper = lane * LANE_W, bottom - ahead * ROW_H
            if kind == "coin":
                draw.ellipse((left + 46, upper + 8, left + 74, upper + 34), fill=COLOURS["coin"])
            elif kind == "train":
                draw.rounded_rectangle((left + 12, upper + 2, left + LANE_W - 12, upper + ROW_H - 2), radius=6, fill=COLOURS["train"])
            elif kind == "low":
                draw.rectangle((left + 10, upper + 26, left + LANE_W - 10, upper + ROW_H - 4), fill=COLOURS["low"])
            else:
                draw.rectangle((left + 10, upper + 2, left + LANE_W - 10, upper + 14), fill=COLOURS["high"])
        left = self.lane * LANE_W
        lift = 14 if self.air else 0
        tall = 14 if self.slide else 34
        draw.rounded_rectangle((left + 40, bottom + ROW_H - 4 - tall - lift, left + 80, bottom + ROW_H - 4 - lift),
                               radius=8, fill=PLAYER)


if __name__ == "__main__":
    run_window(LaneDash, "PlayFellow Arena: Lane Dash")
