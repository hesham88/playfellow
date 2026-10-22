"""Arena level 7 (strategy): Tiny Kingdom. Turn-based defence of your castle wall (the left edge).
Click a button (Farm, Tower, Soldier), then a grass square to build it there; Enter ends your turn. Farms earn gold,
towers shoot enemies within 2 squares, soldiers block a square and fight. Survive 10 waves to win.
/ المستوى 7: دافع عن سور قلعتك. انقر زرًا ثم مربعًا للبناء؛ Enter ينهي دورك. اصمد أمام 10 موجات."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, font, run_window

COLS, ROWS = 10, 7
TILE = 56
BAR = 64  # the button bar under the field
BUILD = {"farm": 10, "tower": 15, "soldier": 8}
BUTTONS = ("farm", "tower", "soldier")
LABELS = {"farm": ("Farm 10", "مزرعة 10"), "tower": ("Tower 15", "برج 15"), "soldier": ("Soldier 8", "جندي 8")}
WAVES, WAVE_EVERY = 10, 3
ENEMY_HP, SOLDIER_HP, CASTLE_HP = 2, 3, 5
TOWER_RANGE = 2
START_GOLD, INCOME, FARM_INCOME = 20, 3, 2
KILL_POINTS = 10
GRASS, WALL, FARM, TOWER, SOLDIER, ENEMY = (70, 140, 80), (150, 140, 170), (220, 190, 90), (124, 92, 255), (34, 228, 255), (230, 70, 70)


class TinyKingdom(ArenaPackGame):
    name = "Tiny Kingdom"
    name_ar = "المملكة الصغيرة"
    keys = ("enter",)
    max_ticks = 1500  # clicks and turn ends
    state_fields = ("gold", "castle", "turn", "waves_sent", "buildings", "enemies", "chosen")

    def new_game(self) -> None:
        self.gold = START_GOLD
        self.castle = CASTLE_HP
        self.turn = 0
        self.waves_sent = 0
        self.buildings: dict[tuple[int, int], list] = {}  # square -> [kind, hp]
        self.enemies: list[list[int]] = []  # [x, y, hp]
        self.chosen = None

    # ---- your actions
    def on_click(self, x: float, y: float) -> None:
        field_top = self.pixel_size[1] - ROWS * TILE - BAR
        bar_top = field_top + ROWS * TILE
        if y >= bar_top:
            index = int(x // (self.pixel_size[0] / len(BUTTONS)))
            if 0 <= index < len(BUTTONS):
                self.chosen = BUTTONS[index]
            return
        col, row = int(x // TILE), int((y - field_top) // TILE)
        if not (1 <= col < COLS and 0 <= row < ROWS) or self.chosen is None:
            return  # column 0 is the castle wall
        square = (col, row)
        cost = BUILD[self.chosen]
        if square in self.buildings or any((e[0], e[1]) == square for e in self.enemies) or self.gold < cost:
            return
        self.gold -= cost
        self.buildings[square] = [self.chosen, SOLDIER_HP]

    def on_key(self, key: str) -> None:
        if key == "enter":
            self._end_turn()

    # ---- the turn
    def _end_turn(self) -> None:
        self.turn += 1
        self.gold += INCOME + FARM_INCOME * sum(b[0] == "farm" for b in self.buildings.values())
        for (tx, ty), (kind, _) in list(self.buildings.items()):
            if kind == "tower":
                near = [e for e in self.enemies if max(abs(e[0] - tx), abs(e[1] - ty)) <= TOWER_RANGE]
                if near:
                    target = min(near, key=lambda e: e[0])  # the one closest to the wall
                    target[2] -= 1
        self.score += KILL_POINTS * sum(e[2] <= 0 for e in self.enemies)
        self.enemies = [e for e in self.enemies if e[2] > 0]
        moved = []
        for e in sorted(self.enemies, key=lambda e: e[0]):
            ahead = (e[0] - 1, e[1])
            guard = self.buildings.get(ahead)
            if guard and guard[0] == "soldier":  # a soldier blocks the way and they fight
                guard[1] -= 1
                e[2] -= 1
                if guard[1] <= 0:
                    del self.buildings[ahead]
            elif guard:  # farms and towers get trampled
                del self.buildings[ahead]
                e[0] -= 1
            else:
                e[0] -= 1
            if e[2] <= 0:
                self.score += KILL_POINTS
                continue
            if e[0] <= 0:
                self.castle -= 1
                continue
            moved.append(e)
        self.enemies = moved
        if self.turn % WAVE_EVERY == 0 and self.waves_sent < WAVES:
            self.waves_sent += 1
            for _ in range(self.waves_sent):
                self.enemies.append([COLS - 1, self.rng.randrange(ROWS), ENEMY_HP])

    # ---- results
    def is_won(self) -> bool:
        return self.castle > 0 and self.waves_sent >= WAVES and not self.enemies

    def is_lost(self) -> bool:
        return self.castle <= 0

    def progress(self) -> dict:
        kinds = [b[0] for b in self.buildings.values()]
        return {"turn": self.turn, "wave": self.waves_sent, "waves": WAVES, "castle": self.castle, "gold": self.gold,
                "enemies": len(self.enemies), "farms": kinds.count("farm"), "towers": kinds.count("tower"),
                "soldiers": kinds.count("soldier"), "chosen": self.chosen}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"الموجة {self.waves_sent}/{WAVES}    القلعة {self.castle}    الذهب {self.gold}    الأعداء {len(self.enemies)}"
        return f"Wave {self.waves_sent}/{WAVES}    Castle {self.castle}    Gold {self.gold}    Enemies {len(self.enemies)}"

    def board_pixels(self) -> tuple[int, int]:
        return COLS * TILE, ROWS * TILE + BAR

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        for row in range(ROWS):
            for col in range(COLS):
                left, upper = col * TILE, top + row * TILE
                draw.rectangle((left, upper, left + TILE - 1, upper + TILE - 1), fill=WALL if col == 0 else GRASS,
                               outline=(60, 120, 70))
        for (col, row), (kind, hp) in self.buildings.items():
            left, upper = col * TILE, top + row * TILE
            if kind == "farm":
                draw.rectangle((left + 8, upper + 14, left + TILE - 8, upper + TILE - 10), fill=FARM)
            elif kind == "tower":
                draw.rectangle((left + 16, upper + 6, left + TILE - 16, upper + TILE - 6), fill=TOWER)
            else:
                draw.ellipse((left + 12, upper + 12, left + TILE - 12, upper + TILE - 12), fill=SOLDIER)
        for x, y, hp in self.enemies:
            left, upper = x * TILE, top + y * TILE
            draw.polygon([(left + 10, upper + TILE // 2), (left + TILE - 10, upper + 10), (left + TILE - 10, upper + TILE - 10)], fill=ENEMY)
        bar_top = top + ROWS * TILE
        width = COLS * TILE // len(BUTTONS)
        label_font = font(18, self.language)
        from rtl import show
        for i, kind in enumerate(BUTTONS):
            left = i * width
            draw.rectangle((left + 4, bar_top + 8, left + width - 4, bar_top + BAR - 8),
                           fill=(124, 92, 255) if self.chosen == kind else (44, 40, 70), outline=(200, 200, 220))
            text = show(LABELS[kind][1 if self.language == "arabic" else 0])
            box = draw.textbbox((0, 0), text, font=label_font)
            draw.text((left + (width - box[2]) / 2, bar_top + (BAR - box[3]) / 2 - 3), text, fill=(255, 255, 255), font=label_font)


if __name__ == "__main__":
    run_window(TinyKingdom, "PlayFellow Arena: Tiny Kingdom")
