"""Arena level 11 (racing): Top Racer. Race two computer cars for 3 laps around a square track, seen from above.
Up speeds up, down brakes (and ends a boost), left/right change lane, space boosts (3 per race). Too fast into a corner, or into a cone,
and you spin out. Finish first to win. / المستوى 11: سباق 3 لفات ضد سيارتين. فرمل قبل المنعطفات وتجنب الأقماع."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

COLS, ROWS = 14, 9  # the track runs around the edge of this grid, clockwise
CELLS = 2 * (COLS - 1) + 2 * (ROWS - 1)  # 42 cells a lap
UNITS = 4  # movement units in one cell
START = 3  # the start/finish line is on the top straight
LAPS = 3
TOP_SPEED, BOOST_SPEED, CORNER_SPEED = 4, 6, 2
BOOST_TICKS, BOOSTS, SPIN_TICKS = 4, 3, 5  # a boost covers 6 cells: shorter than a straight
CORNERS = (0, COLS - 1, COLS - 1 + ROWS - 1, 2 * (COLS - 1) + ROWS - 1)
CORNER_ZONE = {(c + d) % CELLS for c in CORNERS for d in (-1, 0, 1)}
TILE = 48
MARGIN = 14  # grass around the track so the outside lane is never cut off
LANE_GAP = 14
GRASS, ROAD, ZONE, LINE = (34, 70, 50), (60, 60, 80), (90, 70, 90), (230, 230, 240)
CAR_COLOURS = ((124, 92, 255), (230, 70, 70), (255, 160, 40))
CONE = (255, 120, 0)


def cell_xy(cell: int) -> tuple[int, int]:
    """Grid square of a track cell (clockwise from the top-left corner)."""
    cell %= CELLS
    if cell < COLS - 1:
        return cell, 0
    cell -= COLS - 1
    if cell < ROWS - 1:
        return COLS - 1, cell
    cell -= ROWS - 1
    if cell < COLS - 1:
        return COLS - 1 - cell, ROWS - 1
    return 0, ROWS - 1 - (cell - (COLS - 1))


def inward(cell: int) -> tuple[int, int]:
    """The direction towards the middle of the track at this cell (lanes are offset this way)."""
    x, y = cell_xy(cell)
    if y == 0 and x < COLS - 1:
        return 0, 1
    if x == COLS - 1 and y < ROWS - 1:
        return -1, 0
    if y == ROWS - 1 and x > 0:
        return 0, -1
    return 1, 0


class TopRacer(ArenaPackGame):
    name = "Top Racer"
    name_ar = "السبّاق"
    keys = ("up", "down", "left", "right", "space")
    max_ticks = 500
    state_fields = ("distance", "speed", "lane", "boosts", "boost_ticks", "spin", "spins", "cpu", "cones", "finished")

    def new_game(self) -> None:
        self.distance = 0
        self.speed = 0
        self.lane = 1  # 0 outside, 1 middle, 2 inside
        self.boosts = BOOSTS
        self.boost_ticks = 0
        self.spin = 0
        self.spins = 0
        self.cpu = [[0, 0], [0, 2]]  # [progress, lane] of the two computer cars
        straights = [c for c in range(CELLS) if c not in CORNER_ZONE and (c - START) % CELLS > 4]
        self.cones = sorted((self.rng.choice(straights), self.rng.randrange(3)) for _ in range(4))
        self.finished = None  # "you" or "cpu"

    @property
    def cell(self) -> int:
        return (START + self.distance // UNITS) % CELLS

    @property
    def lap(self) -> int:
        return min(LAPS, self.distance // (CELLS * UNITS) + 1)

    def on_key(self, key: str) -> None:
        if self.spin:
            return
        if key == "up":
            self.speed = min(TOP_SPEED, self.speed + 1)
        elif key == "down":  # braking also ends a boost
            self.boost_ticks = 0
            self.speed = max(0, min(self.speed, TOP_SPEED) - 2)
        elif key == "left":
            self.lane = max(0, self.lane - 1)  # the driver's left is the outside of a clockwise track
        elif key == "right":
            self.lane = min(2, self.lane + 1)
        elif key == "space" and self.boosts and not self.boost_ticks:
            self.boosts -= 1
            self.boost_ticks = BOOST_TICKS

    def tick(self) -> None:
        if self.spin:
            self.spin -= 1
            self.speed = 0
        else:
            if self.boost_ticks:
                self.boost_ticks -= 1
                self.speed = BOOST_SPEED if self.boost_ticks else TOP_SPEED
            self.distance += self.speed
            if (self.cell in CORNER_ZONE and self.speed > CORNER_SPEED) or (self.cell, self.lane) in self.cones:
                self.spin, self.speed, self.boost_ticks = SPIN_TICKS, 0, 0
                self.spins += 1
        for car in self.cpu:
            cell = (START + car[0] // UNITS) % CELLS
            ahead = (START + (car[0] + 2 * UNITS) // UNITS) % CELLS
            if cell in CORNER_ZONE or ahead in CORNER_ZONE:
                car[0] += CORNER_SPEED
            else:
                car[0] += 4 if self.rng.random() < 0.3 else 3
        goal = LAPS * CELLS * UNITS
        if self.distance >= goal:
            self.finished = "you"
            self.score += 300
        elif any(car[0] >= goal for car in self.cpu):
            self.finished = "cpu"

    def place(self) -> int:
        return 1 + sum(car[0] > self.distance for car in self.cpu)

    def is_won(self) -> bool:
        return self.finished == "you"

    def is_lost(self) -> bool:
        return self.finished == "cpu"


    def progress(self) -> dict:
        return {"lap": self.lap, "laps": LAPS, "place": self.place(), "speed": self.speed, "boosts": self.boosts,
                "spinning": bool(self.spin), "spins": self.spins}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"اللفة {self.lap}/{LAPS}    المركز {self.place()}    السرعة {self.speed}    التعزيز {self.boosts}"
        return f"Lap {self.lap}/{LAPS}    Place {self.place()}    Speed {self.speed}    Boosts {self.boosts}"

    def seconds_per_tick(self) -> float:
        return 0.1

    def board_pixels(self) -> tuple[int, int]:
        return COLS * TILE + 2 * MARGIN, ROWS * TILE + 2 * MARGIN

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        width, height = self.board_pixels()
        draw.rectangle((0, top, width, top + height), fill=GRASS)
        top += MARGIN
        for cell in range(CELLS):
            x, y = cell_xy(cell)
            draw.rectangle((MARGIN + x * TILE, top + y * TILE, MARGIN + x * TILE + TILE - 1, top + y * TILE + TILE - 1),
                           fill=ZONE if cell in CORNER_ZONE else ROAD)
        sx, sy = cell_xy(START)
        draw.line((MARGIN + sx * TILE, top + sy * TILE, MARGIN + sx * TILE, top + sy * TILE + TILE), fill=LINE, width=4)
        for cell, lane in self.cones:
            cx, cy = self._spot(cell, lane, top)
            draw.polygon([(cx - 7, cy + 7), (cx, cy - 8), (cx + 7, cy + 7)], fill=CONE)
        cars = [(self.distance, self.lane)] + [tuple(car) for car in self.cpu]
        for i, (distance, lane) in reversed(list(enumerate(cars))):
            cx, cy = self._spot((START + distance // UNITS) % CELLS, lane, top)
            draw.rounded_rectangle((cx - 9, cy - 9, cx + 9, cy + 9), radius=4, fill=CAR_COLOURS[i],
                                   outline=(255, 255, 255) if i == 0 else None, width=2)

    @staticmethod
    def _spot(cell: int, lane: int, top: int) -> tuple[int, int]:
        x, y = cell_xy(cell)
        dx, dy = inward(cell)
        return (MARGIN + x * TILE + TILE // 2 + dx * (lane - 1) * LANE_GAP,
                top + y * TILE + TILE // 2 + dy * (lane - 1) * LANE_GAP)


if __name__ == "__main__":
    run_window(TopRacer, "PlayFellow Arena: Top Racer")
