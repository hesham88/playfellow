"""Arena level 2 (puzzle): Box Push. Walk with the arrows and push every box onto a goal. R restarts the level.
Three levels; solve all three to win. / المستوى 2 (ألغاز): ادفع كل صندوق إلى هدف. R يعيد المستوى. ثلاثة مستويات."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

# '#' wall, '@' player, '$' box, '.' goal, '*' box on a goal, '+' player on a goal
LEVELS = (
    ("#######",
     "#     #",
     "# $ . #",
     "#  @  #",
     "#######"),
    ("#########",
     "#       #",
     "# $ $ ..#",
     "#   @   #",
     "#########"),
    ("#########",
     "#   #   #",
     "# $   $ #",
     "#   #   #",
     "##.###.##",
     "#   @   #",
     "#########"),
)
TILE = 56
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
WALL, FLOOR, GOAL = (62, 64, 86), (226, 232, 242), (120, 220, 150)
BOX, BOX_DONE, PLAYER = (196, 140, 60), (90, 190, 110), (40, 110, 255)
LEVEL_POINTS = 100


def parse(level: tuple[str, ...]) -> tuple[set, set, set, tuple[int, int]]:
    walls, boxes, goals, player = set(), set(), set(), (0, 0)
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            if cell in "$*":
                boxes.add((x, y))
            if cell in ".*+":
                goals.add((x, y))
            if cell in "@+":
                player = (x, y)
    return walls, boxes, goals, player


class BoxPush(ArenaPackGame):
    name = "Box Push"
    name_ar = "دفع الصناديق"
    keys = ("up", "down", "left", "right", "r")
    max_ticks = 600

    def new_game(self) -> None:
        self.level_no = 0
        self.moves = 0
        self.pushes = 0
        self._load()

    def _load(self) -> None:
        self.walls, self.boxes, self.goals, self.player = parse(LEVELS[self.level_no])

    def on_key(self, key: str) -> None:
        if key == "r":
            self._load()
            return
        dx, dy = MOVES[key]
        target = (self.player[0] + dx, self.player[1] + dy)
        if target in self.walls:
            return
        if target in self.boxes:
            beyond = (target[0] + dx, target[1] + dy)
            if beyond in self.walls or beyond in self.boxes:
                return
            self.boxes = (self.boxes - {target}) | {beyond}
            self.pushes += 1
        self.player = target
        self.moves += 1
        if self.boxes == self.goals:
            self.score += LEVEL_POINTS
            if self.level_no + 1 < len(LEVELS):
                self.level_no += 1
                self._load()
            else:
                self.level_no = len(LEVELS)  # all solved

    def is_won(self) -> bool:
        return self.level_no >= len(LEVELS)

    def progress(self) -> dict:
        return {"level": min(self.level_no + 1, len(LEVELS)), "levels": len(LEVELS),
                "boxes_on_goals": len(self.boxes & self.goals), "boxes": len(self.boxes),
                "moves": self.moves, "pushes": self.pushes}

    def progress_text(self) -> str:
        p = self.progress()
        if self.language == "arabic":
            return f"المستوى {p['level']}/{p['levels']}    صناديق على الأهداف: {p['boxes_on_goals']}/{p['boxes']}"
        return f"Level {p['level']}/{p['levels']}    Boxes on goals: {p['boxes_on_goals']}/{p['boxes']}"

    def board_pixels(self) -> tuple[int, int]:
        widest = max(len(row) for level in LEVELS for row in level)
        tallest = max(len(level) for level in LEVELS)
        return max(widest * TILE, 480), tallest * TILE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        if self.is_won():
            return
        level = LEVELS[self.level_no]
        ox = (self.board_pixels()[0] - max(len(row) for row in level) * TILE) // 2  # centre the level
        for y, row in enumerate(level):
            for x in range(len(row)):
                box = (ox + x * TILE, top + y * TILE, ox + x * TILE + TILE - 1, top + y * TILE + TILE - 1)
                draw.rectangle(box, fill=WALL if (x, y) in self.walls else FLOOR)
        for x, y in self.goals:
            left, upper = ox + x * TILE, top + y * TILE
            draw.ellipse((left + 18, upper + 18, left + TILE - 18, upper + TILE - 18), fill=GOAL)
        for x, y in self.boxes:
            left, upper = ox + x * TILE, top + y * TILE
            draw.rounded_rectangle((left + 6, upper + 6, left + TILE - 6, upper + TILE - 6), radius=6,
                                   fill=BOX_DONE if (x, y) in self.goals else BOX, outline=(70, 50, 20), width=3)
        x, y = self.player
        left, upper = ox + x * TILE, top + y * TILE
        draw.ellipse((left + 10, upper + 10, left + TILE - 10, upper + TILE - 10), fill=PLAYER, outline=(20, 60, 170), width=3)


if __name__ == "__main__":
    run_window(BoxPush, "PlayFellow Arena: Box Push")
