"""The Practice Arena rules and drawing. / قواعد ساحة التدريب ورسمها."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from rtl import show

WINDOW_TITLE = "PlayFellow Practice Arena"
TILE = 48
BANNER = 72
LEVEL = (
    "############",
    "#P...#....c#",
    "#.##.#.##..#",
    "#c#..c..#..#",
    "#.#.###.#.##",
    "#...#k....D#",
    "#.###.###..#",
    "#c.......c.#",
    "############",
)
MOVES = {
    "up": (0, -1), "w": (0, -1),
    "down": (0, 1), "s": (0, 1),
    "left": (-1, 0), "a": (-1, 0),
    "right": (1, 0), "d": (1, 0),
}
REPEAT_MS = 150  # a held key moves the player again this often (key repeat)
COIN_POINTS = 10
DOOR_POINTS = 50

WALL = (62, 64, 86)
FLOOR = (226, 232, 242)
FLOOR_LINE = (204, 211, 224)
COIN = (255, 196, 0)
COIN_EDGE = (196, 140, 0)
KEY = (230, 170, 30)
DOOR = (140, 86, 40)
DOOR_OPEN = (60, 200, 110)
PLAYER = (40, 110, 255)
BACKGROUND = (22, 22, 34)

TEXTS = {
    "english": {
        "score": "Score",
        "coins_left": "Coins left",
        "key": "Key",
        "yes_no": "YES / no",
        "intro": "Collect the coins, find the key, open the door!",
        "bump": "Bump! That's a wall.",
        "locked": "The door is locked. Find the key!",
        "coin": "Coin! +10",
        "got_key": "You got the key! Now open the door.",
        "win": "YOU WIN! Press R to play again.",
    },
    "arabic": {
        "score": "النقاط",
        "coins_left": "العملات المتبقية",
        "key": "المفتاح",
        "yes_no": "نعم / لا",
        "intro": "اجمع العملات، جد المفتاح، افتح الباب!",
        "bump": "آخ! هذا جدار.",
        "locked": "الباب مقفل. ابحث عن المفتاح!",
        "coin": "عملة! ‎+10",
        "got_key": "حصلت على المفتاح! الآن افتح الباب.",
        "win": "فزت! اضغط R لتلعب مرة أخرى.",
    },
}


def _font(size: int, language: str = "english"):
    """Load a font with Arabic support for Arabic language."""
    if language == "arabic":
        for font_name in ("arial.ttf", "tahoma.ttf"):
            try:
                return ImageFont.truetype(font_name, size)
            except OSError:
                pass
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


class ArenaGame:
    def __init__(self, level: tuple[str, ...] = LEVEL, language: str = "english"):
        self.level = level
        self.language = language
        self.texts = TEXTS.get(language, TEXTS["english"])
        self.reset()

    def reset(self) -> None:
        self.walls: set[tuple[int, int]] = set()
        self.coins: set[tuple[int, int]] = set()
        self.key: tuple[int, int] | None = None
        self.door: tuple[int, int] = (0, 0)
        self.player: tuple[int, int] = (1, 1)
        for y, row in enumerate(self.level):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.walls.add((x, y))
                elif cell == "c":
                    self.coins.add((x, y))
                elif cell == "k":
                    self.key = (x, y)
                elif cell == "D":
                    self.door = (x, y)
                elif cell == "P":
                    self.player = (x, y)
        self.score = 0
        self.has_key = False
        self.won = False
        self.moves = 0
        self.bumps = 0
        self.game_seconds = 0.0  # the game's own clock: the sum of `dt` over step_realtime calls, never the wall clock
        self.message = self.texts["intro"]

    @property
    def coins_left(self) -> int:
        return len(self.coins)

    @property
    def pixel_size(self) -> tuple[int, int]:
        return len(self.level[0]) * TILE, len(self.level) * TILE + BANNER

    def press(self, key_name: str) -> None:
        move = MOVES.get(key_name.lower())
        if self.won or move is None:
            return
        target = (self.player[0] + move[0], self.player[1] + move[1])
        if target in self.walls:
            self.bumps += 1
            self.message = self.texts["bump"]
            return
        if target == self.door:
            if not self.has_key:
                self.message = self.texts["locked"]
                return
            self.player = target
            self.moves += 1
            self.score += DOOR_POINTS
            self.won = True
            self.message = self.texts["win"]
            return
        self.player = target
        self.moves += 1
        if target in self.coins:
            self.coins.remove(target)
            self.score += COIN_POINTS
            self.message = self.texts["coin"]
        elif target == self.key:
            self.key = None
            self.has_key = True
            self.message = self.texts["got_key"]
        else:
            self.message = ""

    realtime = False  # the arena is turn-based: nothing moves unless the player does, so the world always waits

    def step_realtime(self, key_name: str | None = None, dt: float = 0.05, duration_ms: int = 0) -> None:
        """The same call the arena pack games have. A key press moves one tile; holding it for `duration_ms`
        repeats the move every REPEAT_MS. `dt` only feeds the game clock (`game_seconds`): there are no hazards
        or timers to keep running. The clock grows by exactly `dt` on every call, with or without a key."""
        self.game_seconds += dt
        if not key_name:
            return
        self.press(key_name)
        for _ in range(duration_ms // REPEAT_MS):
            self.press(key_name)

    def render(self) -> Image.Image:
        image = Image.new("RGB", self.pixel_size, BACKGROUND)
        draw = ImageDraw.Draw(image)

        # Banner text with RTL support
        is_arabic = self.language == "arabic"
        banner_text = f"{self.texts['score']}: {self.score}    {self.texts['coins_left']}: {self.coins_left}    {self.texts['key']}: {self.texts['yes_no'].split(' / ')[0] if self.has_key else self.texts['yes_no'].split(' / ')[1]}"
        banner_text = show(banner_text)

        x_pos = 14
        if is_arabic:
            # Right-align for Arabic
            bbox = draw.textbbox((0, 0), banner_text, font=_font(22, self.language))
            width = bbox[2] - bbox[0]
            x_pos = self.pixel_size[0] - width - 14

        draw.text(
            (x_pos, 10),
            banner_text,
            fill=(255, 255, 255), font=_font(22, self.language),
        )

        # Message text with RTL support
        message_text = show(self.message)
        msg_x = 14
        if is_arabic:
            bbox = draw.textbbox((0, 0), message_text, font=_font(17, self.language))
            width = bbox[2] - bbox[0]
            msg_x = self.pixel_size[0] - width - 14

        draw.text((msg_x, 42), message_text, fill=(255, 214, 110), font=_font(17, self.language))

        for y, row in enumerate(self.level):
            for x, _ in enumerate(row):
                left, top = x * TILE, BANNER + y * TILE
                box = (left, top, left + TILE - 1, top + TILE - 1)
                if (x, y) in self.walls:
                    draw.rectangle(box, fill=WALL)
                else:
                    draw.rectangle(box, fill=FLOOR, outline=FLOOR_LINE)
        for x, y in self.coins:
            self._circle(draw, x, y, 12, COIN, COIN_EDGE)
        if self.key:
            left, top = self.key[0] * TILE, BANNER + self.key[1] * TILE
            draw.ellipse((left + 8, top + 14, left + 24, top + 30), outline=KEY, width=5)
            draw.rectangle((left + 23, top + 20, left + 40, top + 25), fill=KEY)
            draw.rectangle((left + 34, top + 25, left + 38, top + 32), fill=KEY)
        left, top = self.door[0] * TILE, BANNER + self.door[1] * TILE
        draw.rectangle((left + 6, top + 2, left + TILE - 7, top + TILE - 3), fill=DOOR,
                       outline=DOOR_OPEN if self.has_key else (90, 52, 20), width=4)
        draw.ellipse((left + TILE - 18, top + TILE // 2 - 3, left + TILE - 12, top + TILE // 2 + 3), fill=(250, 220, 120))
        self._circle(draw, *self.player, 18, PLAYER, (20, 60, 170))
        px, py = self.player[0] * TILE + TILE // 2, BANNER + self.player[1] * TILE + TILE // 2
        for eye in (-7, 7):
            draw.ellipse((px + eye - 4, py - 9, px + eye + 4, py - 1), fill=(255, 255, 255))
            draw.ellipse((px + eye - 2, py - 7, px + eye + 2, py - 3), fill=(10, 10, 30))
        return image

    @staticmethod
    def _circle(draw, x, y, radius, fill, edge) -> None:
        cx, cy = x * TILE + TILE // 2, BANNER + y * TILE + TILE // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=edge, width=3)
