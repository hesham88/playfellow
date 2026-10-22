"""Arena level 15 (action RPG): Action Quest. A boss fight seen from above. W/A/S/D move, J light attack, K heavy
attack, L dodge-roll (you can't be hit while rolling), E drink a potion. The boss shows its attacks first: red squares
mean "about to be hit there" — get out, or roll. Attacks and rolls use stamina. Beat the boss to win.
/ المستوى 15: قتال زعيم من الأعلى. المربعات الحمراء تعني ضربة قادمة: ابتعد أو تدحرج. اهزم الزعيم."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

COLS, ROWS = 13, 9
TILE = 44
MOVES = {"w": (0, -1), "s": (0, 1), "a": (-1, 0), "d": (1, 0)}
HERO_HP, BOSS_HP, POTIONS, POTION_HEAL = 50, 120, 3, 20
STAMINA, REGEN = 100, 6
COST = {"j": 15, "k": 30, "l": 25}
LIGHT, HEAVY, HEAVY_WINDUP = 6, 15, 2
ROLL_TICKS, ROLL_DISTANCE = 3, 2
SLAM = {"windup": 4, "damage": 15, "radius": 2}
CHARGE = {"windup": 5, "damage": 20, "reach": 7}
BOSS_REST, BOSS_STEP_EVERY = 4, 2
FLOOR, LINE, DANGER, HERO, BOSS, ROLL = (40, 44, 58), (54, 58, 74), (200, 50, 60), (124, 92, 255), (230, 120, 40), (182, 255, 59)


class ActionQuest(ArenaPackGame):
    name = "Action Quest"
    name_ar = "مهمة الحركة"
    keys = ("w", "a", "s", "d", "j", "k", "l", "e")
    max_ticks = 1500
    state_fields = ("hero", "boss")

    def new_game(self) -> None:
        self.hero = {"x": 2, "y": 4, "hp": HERO_HP, "stamina": STAMINA, "potions": POTIONS, "roll": 0, "facing": (1, 0),
                     "heavy": 0}
        self.boss = {"x": 9, "y": 4, "hp": BOSS_HP, "attack": None, "windup": 0, "danger": [], "rest": 0, "clock": 0}

    # ---- you
    def _free(self, x: int, y: int) -> bool:
        return 0 <= x < COLS and 0 <= y < ROWS and (x, y) != (self.boss["x"], self.boss["y"])

    def on_key(self, key: str) -> None:
        h = self.hero
        if h["heavy"]:
            return  # winding up a heavy attack
        if key in MOVES:
            dx, dy = MOVES[key]
            h["facing"] = (dx, dy)
            if h["roll"] == 0 and self._free(h["x"] + dx, h["y"] + dy):
                h["x"], h["y"] = h["x"] + dx, h["y"] + dy
        elif key == "e" and h["potions"]:
            h["potions"] -= 1
            h["hp"] = min(HERO_HP, h["hp"] + POTION_HEAL)
        elif key in COST and h["stamina"] >= COST[key] and h["roll"] == 0:
            h["stamina"] -= COST[key]
            if key == "j":
                self._strike(LIGHT)
            elif key == "k":
                h["heavy"] = HEAVY_WINDUP
            else:
                h["roll"] = ROLL_TICKS
                dx, dy = h["facing"]
                for _ in range(ROLL_DISTANCE):
                    if self._free(h["x"] + dx, h["y"] + dy):
                        h["x"], h["y"] = h["x"] + dx, h["y"] + dy

    def _strike(self, damage: int) -> None:
        h, b = self.hero, self.boss
        if max(abs(h["x"] - b["x"]), abs(h["y"] - b["y"])) <= 1:
            b["hp"] -= damage
            self.score += damage

    # ---- the clock
    def tick(self) -> None:
        h, b = self.hero, self.boss
        b["clock"] += 1
        if h["heavy"]:
            h["heavy"] -= 1
            if h["heavy"] == 0:
                self._strike(HEAVY)
        if b["hp"] <= 0:
            return
        if b["attack"]:
            b["windup"] -= 1
            if b["windup"] == 0:
                self._boss_hits()
        elif b["rest"]:
            b["rest"] -= 1
        else:
            self._boss_decides()
        if h["roll"]:
            h["roll"] -= 1
        h["stamina"] = min(STAMINA, h["stamina"] + REGEN)

    def _boss_decides(self) -> None:
        h, b = self.hero, self.boss
        gap = max(abs(h["x"] - b["x"]), abs(h["y"] - b["y"]))
        if gap <= 1 or (gap <= SLAM["radius"] and self.rng.random() < 0.5):
            b["attack"], b["windup"] = "slam", SLAM["windup"]
            b["danger"] = [(x, y) for x in range(b["x"] - SLAM["radius"], b["x"] + SLAM["radius"] + 1)
                           for y in range(b["y"] - SLAM["radius"], b["y"] + SLAM["radius"] + 1)
                           if 0 <= x < COLS and 0 <= y < ROWS and (x, y) != (b["x"], b["y"])]
        elif (h["x"] == b["x"] or h["y"] == b["y"]) and gap <= CHARGE["reach"] and self.rng.random() < 0.6:
            dx = (h["x"] > b["x"]) - (h["x"] < b["x"])
            dy = (h["y"] > b["y"]) - (h["y"] < b["y"])
            lane, x, y = [], b["x"], b["y"]
            for _ in range(CHARGE["reach"]):
                x, y = x + dx, y + dy
                if not (0 <= x < COLS and 0 <= y < ROWS):
                    break
                lane.append((x, y))
            b["attack"], b["windup"], b["danger"] = "charge", CHARGE["windup"], lane
        elif b["clock"] % BOSS_STEP_EVERY == 0:
            dx = (h["x"] > b["x"]) - (h["x"] < b["x"])
            dy = (h["y"] > b["y"]) - (h["y"] < b["y"])
            step = (b["x"] + dx, b["y"]) if abs(h["x"] - b["x"]) >= abs(h["y"] - b["y"]) else (b["x"], b["y"] + dy)
            if step != (h["x"], h["y"]):
                b["x"], b["y"] = step

    def _boss_hits(self) -> None:
        h, b = self.hero, self.boss
        if (h["x"], h["y"]) in b["danger"] and h["roll"] == 0:
            h["hp"] -= SLAM["damage"] if b["attack"] == "slam" else CHARGE["damage"]
        if b["attack"] == "charge" and b["danger"]:
            end = b["danger"][-1]
            if end == (h["x"], h["y"]):  # stop just short of the hero
                end = b["danger"][-2] if len(b["danger"]) > 1 else (b["x"], b["y"])
            b["x"], b["y"] = end
        b["attack"], b["danger"], b["rest"] = None, [], BOSS_REST

    # ---- results
    def is_won(self) -> bool:
        return self.boss["hp"] <= 0

    def is_lost(self) -> bool:
        return self.hero["hp"] <= 0

    def progress(self) -> dict:
        h, b = self.hero, self.boss
        return {"hp": h["hp"], "stamina": h["stamina"], "potions": h["potions"], "boss_hp": max(0, b["hp"]),
                "boss_attack": b["attack"], "danger_here": (h["x"], h["y"]) in b["danger"], "rolling": h["roll"] > 0}

    def progress_text(self) -> str:
        h, b = self.hero, self.boss
        if self.language == "arabic":
            return f"صحتك {max(0, h['hp'])}    التحمل {h['stamina']}    الجرعات {h['potions']}    الزعيم {max(0, b['hp'])}"
        return f"HP {max(0, h['hp'])}    Stamina {h['stamina']}    Potions {h['potions']}    Boss {max(0, b['hp'])}"

    def seconds_per_tick(self) -> float:
        return 0.15

    def board_pixels(self) -> tuple[int, int]:
        return COLS * TILE, ROWS * TILE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        b, h = self.boss, self.hero
        for y in range(ROWS):
            for x in range(COLS):
                fill = DANGER if (x, y) in b["danger"] else FLOOR
                draw.rectangle((x * TILE, top + y * TILE, x * TILE + TILE - 1, top + y * TILE + TILE - 1), fill=fill, outline=LINE)
        bx, by = b["x"] * TILE, top + b["y"] * TILE
        draw.rounded_rectangle((bx + 3, by + 3, bx + TILE - 3, by + TILE - 3), radius=10, fill=BOSS,
                               outline=(255, 255, 255) if b["attack"] else None, width=3)
        share = max(0, b["hp"]) / BOSS_HP
        draw.rectangle((bx - 6, by - 8, bx - 6 + int((TILE + 12) * share), by - 4), fill=(230, 70, 70))
        hx, hy = h["x"] * TILE, top + h["y"] * TILE
        draw.ellipse((hx + 8, hy + 8, hx + TILE - 8, hy + TILE - 8), fill=ROLL if h["roll"] else HERO,
                     outline=(255, 214, 60) if h["heavy"] else None, width=3)
        fx, fy = h["facing"]
        cx, cy = hx + TILE / 2, hy + TILE / 2
        draw.line((cx, cy, cx + fx * 16, cy + fy * 16), fill=(255, 255, 255), width=3)


if __name__ == "__main__":
    run_window(ActionQuest, "PlayFellow Arena: Action Quest")
