"""Arena level 5 (metroidvania): Key Caves. Three cave rooms. Find the jump boots, use the double jump to reach the
key on a high ledge, then go back to the locked door and open it (Z). Spikes end the game.
/ المستوى 5: ثلاث غرف كهوف. اعثر على حذاء القفز، اقفز مرتين لتصل إلى المفتاح، ثم عُد وافتح الباب (Z)."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

# '#' rock, '^' spikes, 'D' locked door, 'B' jump boots, 'K' key, 'P' start. Three rooms of 16 columns.
WORLD = (
    "################################################",
    "#..............................................#",
    "#..............................................#",
    "#........................................K.....#",
    "#.......................................####...#",
    "#..............................................#",
    "#..............................................#",
    "#..............................................#",
    "#D..P................^^^.....B.................#",
    "################################################",
)
ROOM_W = 16
ROOMS = len(WORLD[0]) // ROOM_W
JUMP = 3  # cells a jump rises
TILE = 40
POINTS = {"boots": 50, "key": 50, "door": 100}
SKY, ROCK, SPIKE = (24, 20, 44), (92, 74, 120), (230, 230, 240)
DOOR, DOOR_OPEN, KEY, BOOTS, PLAYER = (150, 90, 40), (90, 200, 120), (240, 190, 40), (34, 228, 255), (124, 92, 255)


def _find(char: str) -> tuple[int, int]:
    return next((x, y) for y, row in enumerate(WORLD) for x, c in enumerate(row) if c == char)


class KeyCaves(ArenaPackGame):
    name = "Key Caves"
    name_ar = "كهوف المفتاح"
    keys = ("left", "right", "space", "z")
    max_ticks = 1500
    state_fields = ("x", "y", "rise", "jumps", "has_boots", "has_key", "door_open", "dead")

    def new_game(self) -> None:
        self.x, self.y = _find("P")
        self.rise = 0
        self.jumps = 0
        self.has_boots = self.has_key = self.door_open = self.dead = False

    def _solid(self, x: int, y: int) -> bool:
        cell = WORLD[y][x]
        return cell == "#" or (cell == "D" and not self.door_open)

    def _on_ground(self) -> bool:
        return self._solid(self.x, self.y + 1)

    def on_key(self, key: str) -> None:
        if key in ("left", "right"):
            nx = self.x + (1 if key == "right" else -1)
            if not self._solid(nx, self.y):
                self.x = nx
                self._touch()
        elif key == "space":
            if self._on_ground():
                self.rise, self.jumps = JUMP, 1
            elif self.has_boots and self.jumps < 2:  # the double jump, once in the air
                self.rise, self.jumps = JUMP, 2
        elif key == "z" and self.has_key:
            if WORLD[self.y][self.x - 1] == "D" or WORLD[self.y][self.x + 1] == "D":
                self.door_open = True
                self.score += POINTS["door"]

    def tick(self) -> None:
        if self.rise > 0:
            if self._solid(self.x, self.y - 1):
                self.rise = 0  # bumped the ceiling
            else:
                self.y -= 1
                self.rise -= 1
        elif not self._on_ground():
            self.y += 1  # fall one cell per tick
        if self._on_ground() and self.rise == 0:
            self.jumps = 0
        self._touch()

    def seconds_per_tick(self) -> float:
        return 0.1  # gravity and the jump arc run on the clock, like the other real-time games

    def _touch(self) -> None:
        cell = WORLD[self.y][self.x]
        if cell == "^":
            self.dead = True
        elif cell == "B" and not self.has_boots:
            self.has_boots = True
            self.score += POINTS["boots"]
        elif cell == "K" and not self.has_key:
            self.has_key = True
            self.score += POINTS["key"]

    @property
    def room(self) -> int:
        return self.x // ROOM_W

    def is_won(self) -> bool:
        return self.door_open

    def is_lost(self) -> bool:
        return self.dead

    def progress(self) -> dict:
        return {"room": self.room + 1, "rooms": ROOMS, "has_boots": self.has_boots, "has_key": self.has_key,
                "door_open": self.door_open}

    def progress_text(self) -> str:
        if self.language == "arabic":
            yes = lambda flag: "نعم" if flag else "لا"  # noqa: E731
            return f"الغرفة {self.room + 1}/{ROOMS}    الحذاء: {yes(self.has_boots)}    المفتاح: {yes(self.has_key)}"
        yes = lambda flag: "yes" if flag else "no"  # noqa: E731
        return f"Room {self.room + 1}/{ROOMS}    Boots: {yes(self.has_boots)}    Key: {yes(self.has_key)}"

    def board_pixels(self) -> tuple[int, int]:
        return ROOM_W * TILE, len(WORLD) * TILE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        first = self.room * ROOM_W
        for y, row in enumerate(WORLD):
            for col in range(ROOM_W):
                x, cell = first + col, row[first + col]
                left, upper = col * TILE, top + y * TILE
                box = (left, upper, left + TILE - 1, upper + TILE - 1)
                draw.rectangle(box, fill=ROCK if cell == "#" else SKY)
                if cell == "^":
                    for i in range(3):
                        draw.polygon([(left + i * 13, upper + TILE), (left + i * 13 + 6, upper + 8), (left + i * 13 + 13, upper + TILE)], fill=SPIKE)
                elif cell == "D":
                    draw.rectangle((left + 6, upper + 2, left + TILE - 6, upper + TILE), fill=DOOR_OPEN if self.door_open else DOOR)
                elif cell == "B" and not self.has_boots:
                    draw.rectangle((left + 10, upper + 20, left + 30, upper + 34), fill=BOOTS)
                elif cell == "K" and not self.has_key:
                    draw.ellipse((left + 8, upper + 12, left + 22, upper + 26), outline=KEY, width=4)
                    draw.rectangle((left + 21, upper + 17, left + 34, upper + 21), fill=KEY)
        left, upper = (self.x - first) * TILE, top + self.y * TILE
        draw.rounded_rectangle((left + 8, upper + 4, left + TILE - 8, upper + TILE - 2), radius=8, fill=PLAYER)
        if self.has_boots:
            draw.rectangle((left + 8, upper + TILE - 8, left + TILE - 8, upper + TILE - 2), fill=BOOTS)


if __name__ == "__main__":
    run_window(KeyCaves, "PlayFellow Arena: Key Caves")
