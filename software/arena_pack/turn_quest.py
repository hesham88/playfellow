"""Arena level 6 (RPG): Turn Quest. Menu battles: fight slimes for experience and gold, level up, buy a sword and
potions, rest at the inn, then beat the dragon. Up/down choose, Enter does it, Esc goes back.
/ المستوى 6: معارك بالقوائم: قاتل الوحوش الهلامية، ارتقِ مستوى، اشترِ سيفًا وجرعات، ثم اهزم التنين."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, font, run_window

MENUS = {
    "town": [("Fight a slime", "قاتل وحشًا هلاميًا"), ("Shop", "المتجر"), ("Rest at the inn (10 gold)", "استرح في النُزُل (10 ذهب)"),
             ("Fight the dragon", "قاتل التنين")],
    "shop": [("Potion (10 gold)", "جرعة (10 ذهب)"), ("Sword +3 attack (40 gold)", "سيف +3 هجوم (40 ذهب)"), ("Back", "رجوع")],
    "battle": [("Attack", "هجوم"), ("Fire (5 MP)", "نار (5 طاقة)"), ("Potion", "جرعة"), ("Run away", "اهرب")],
}
ENEMIES = {"slime": {"name": ("Slime", "وحش هلامي"), "hp": 12, "atk": 5, "def": 1, "xp": 6, "gold": 8},
           "dragon": {"name": ("Dragon", "تنين"), "hp": 80, "atk": 12, "def": 4, "xp": 0, "gold": 0}}
POTION_HEAL, FIRE_DAMAGE, FIRE_MP = 20, 14, 5
PRICES = {"potion": 10, "sword": 40, "inn": 10}
DRAGON_POINTS = 1000
PANEL, TEXT, CURSOR, HP_BAR, MP_BAR = (34, 30, 60), (240, 240, 250), (124, 92, 255), (90, 220, 120), (34, 228, 255)


class TurnQuest(ArenaPackGame):
    name = "Turn Quest"
    name_ar = "مغامرة الأدوار"
    keys = ("up", "down", "enter", "esc")
    max_ticks = 3000
    state_fields = ("screen", "cursor", "hero", "enemy", "message", "outcome")

    def new_game(self) -> None:
        self.screen = "town"
        self.cursor = 0
        self.hero = {"level": 1, "xp": 0, "hp": 30, "max_hp": 30, "mp": 10, "max_mp": 10, "atk": 6, "def": 2,
                     "gold": 0, "potions": 1, "sword": False}
        self.enemy = None
        self.message = ("Choose with up/down, Enter to do it.", "اختر بالأعلى/الأسفل، وEnter للتنفيذ.")
        self.outcome = None

    # ---- menus
    def options(self) -> list[tuple[str, str]]:
        return MENUS.get(self.screen, [])

    def on_key(self, key: str) -> None:
        if key in ("up", "down"):
            self.cursor = (self.cursor + (1 if key == "down" else -1)) % len(self.options())
        elif key == "esc" and self.screen == "shop":
            self._go("town")
        elif key == "enter":
            {"town": self._town, "shop": self._shop, "battle": self._battle}[self.screen](self.cursor)

    def _go(self, screen: str) -> None:
        self.screen, self.cursor = screen, 0

    def _say(self, english: str, arabic: str) -> None:
        self.message = (english, arabic)

    def _town(self, choice: int) -> None:
        h = self.hero
        if choice in (0, 3):
            kind = "slime" if choice == 0 else "dragon"
            self.enemy = {"kind": kind, "hp": ENEMIES[kind]["hp"]}
            self._go("battle")
            self._say(f"A {ENEMIES[kind]['name'][0]} appears!", f"ظهر {ENEMIES[kind]['name'][1]}!")
        elif choice == 1:
            self._go("shop")
        elif choice == 2:
            if h["gold"] >= PRICES["inn"]:
                h["gold"] -= PRICES["inn"]
                h["hp"], h["mp"] = h["max_hp"], h["max_mp"]
                self._say("You rest. HP and MP are full.", "استرحت. الصحة والطاقة ممتلئتان.")
            else:
                self._say("Not enough gold for the inn.", "لا يكفي الذهب للنُزُل.")

    def _shop(self, choice: int) -> None:
        h = self.hero
        if choice == 2:
            self._go("town")
        elif choice == 0 and h["gold"] >= PRICES["potion"]:
            h["gold"] -= PRICES["potion"]
            h["potions"] += 1
            self._say("You bought a potion.", "اشتريت جرعة.")
        elif choice == 1 and not h["sword"] and h["gold"] >= PRICES["sword"]:
            h["gold"] -= PRICES["sword"]
            h["sword"] = True
            h["atk"] += 3
            self._say("You bought a sword: +3 attack.", "اشتريت سيفًا: +3 هجوم.")
        else:
            self._say("You can't buy that now.", "لا يمكنك شراء ذلك الآن.")

    def _battle(self, choice: int) -> None:
        h, e = self.hero, self.enemy
        foe = ENEMIES[e["kind"]]
        if choice == 0:
            hit = max(1, h["atk"] + self.rng.randint(0, 2) - foe["def"])
            e["hp"] -= hit
            self._say(f"You hit for {hit}.", f"ضربت بقوة {hit}.")
        elif choice == 1:
            if h["mp"] < FIRE_MP:
                self._say("Not enough MP.", "لا تكفي الطاقة.")
                return
            h["mp"] -= FIRE_MP
            e["hp"] -= FIRE_DAMAGE
            self._say(f"Fire burns for {FIRE_DAMAGE}!", f"النار تحرق بقوة {FIRE_DAMAGE}!")
        elif choice == 2:
            if not h["potions"]:
                self._say("No potions left.", "لا توجد جرعات.")
                return
            h["potions"] -= 1
            h["hp"] = min(h["max_hp"], h["hp"] + POTION_HEAL)
            self._say(f"You drink a potion (+{POTION_HEAL} HP).", f"شربت جرعة (+{POTION_HEAL} صحة).")
        elif choice == 3:
            if e["kind"] == "dragon":
                self._say("You can't run from the dragon!", "لا يمكنك الهروب من التنين!")
            else:
                self.enemy = None
                self._go("town")
                self._say("You ran back to town.", "هربت إلى البلدة.")
                return
        if e["hp"] <= 0:
            self._won_battle(foe)
            return
        bite = max(1, foe["atk"] + self.rng.randint(0, 2) - h["def"])
        h["hp"] -= bite
        if h["hp"] <= 0:
            h["hp"] = 0
            self.screen, self.outcome = "end", "lost"
            self._say(f"The {foe['name'][0]} wins...", f"فاز {foe['name'][1]}...")

    def _won_battle(self, foe: dict) -> None:
        h = self.hero
        if self.enemy["kind"] == "dragon":
            self.screen, self.outcome = "end", "won"
            self.score += DRAGON_POINTS
            self._say("You beat the dragon!", "هزمت التنين!")
            return
        h["xp"] += foe["xp"]
        h["gold"] += foe["gold"]
        self.score += foe["xp"]
        self.enemy = None
        self._go("town")
        self._say(f"You win! +{foe['xp']} XP, +{foe['gold']} gold.", f"فزت! +{foe['xp']} خبرة، +{foe['gold']} ذهب.")
        while h["xp"] >= 10 * h["level"]:
            h["xp"] -= 10 * h["level"]
            h["level"] += 1
            h["max_hp"] += 8
            h["max_mp"] += 3
            h["atk"] += 2
            h["def"] += 1
            h["hp"], h["mp"] = h["max_hp"], h["max_mp"]
            self._say(f"Level up! You are level {h['level']}.", f"ارتقيت! أنت الآن في المستوى {h['level']}.")

    # ---- results
    def is_won(self) -> bool:
        return self.outcome == "won"

    def is_lost(self) -> bool:
        return self.outcome == "lost"

    def progress(self) -> dict:
        h = self.hero
        return {"screen": self.screen, "level": h["level"], "hp": h["hp"], "max_hp": h["max_hp"], "mp": h["mp"],
                "gold": h["gold"], "potions": h["potions"], "sword": h["sword"],
                "enemy": self.enemy["kind"] if self.enemy else None, "enemy_hp": self.enemy["hp"] if self.enemy else None}

    def progress_text(self) -> str:
        h = self.hero
        if self.language == "arabic":
            return f"المستوى {h['level']}    الصحة {h['hp']}/{h['max_hp']}    الطاقة {h['mp']}    الذهب {h['gold']}    الجرعات {h['potions']}"
        return f"Level {h['level']}    HP {h['hp']}/{h['max_hp']}    MP {h['mp']}    Gold {h['gold']}    Potions {h['potions']}"

    def board_pixels(self) -> tuple[int, int]:
        return 560, 360

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        arabic = self.language == "arabic"
        pick = 1 if arabic else 0
        width = self.board_pixels()[0]
        draw.rectangle((0, top, width, top + 360), fill=PANEL)
        big, small = font(22, self.language), font(18, self.language)

        def write(x, y, text, f, fill=TEXT):
            from rtl import show
            text = show(text) if arabic else text
            if arabic:
                box = draw.textbbox((0, 0), text, font=f)
                x = width - x - (box[2] - box[0])
            draw.text((x, y), text, fill=fill, font=f)

        def bar(x, y, length, height, fill):  # bars grow from the right in Arabic, like the text
            left = width - x - length if arabic else x
            draw.rectangle((left, y, left + length, y + height), fill=fill)
        if self.enemy:
            foe = ENEMIES[self.enemy["kind"]]
            size = 70 if self.enemy["kind"] == "dragon" else 40
            colour = (230, 70, 70) if self.enemy["kind"] == "dragon" else (120, 220, 140)
            cx, cy = width // 2, top + 90
            draw.ellipse((cx - size, cy - size // 2, cx + size, cy + size // 2), fill=colour)
            write(20, top + 14, f"{foe['name'][pick]}  {max(0, self.enemy['hp'])}/{foe['hp']}", big)
            share = max(0, self.enemy["hp"]) / foe["hp"]
            bar(20, top + 44, int(200 * share), 10, (230, 70, 70))
        else:
            write(20, top + 14, "Town" if not arabic and self.screen == "town" else ("Shop" if not arabic else
                  ("البلدة" if self.screen == "town" else "المتجر")), big)
        h = self.hero
        bar(20, top + 160, int(200 * h["hp"] / h["max_hp"]), 10, HP_BAR)
        bar(20, top + 176, int(200 * h["mp"] / h["max_mp"]), 8, MP_BAR)
        for i, option in enumerate(self.options()):
            y = top + 200 + i * 34
            if i == self.cursor:
                draw.rectangle((14, y - 4, width - 14, y + 28), fill=CURSOR)
            write(26, y, option[pick], big)
        write(20, top + 336, self.message[pick], small, fill=(255, 214, 110))


if __name__ == "__main__":
    run_window(TurnQuest, "PlayFellow Arena: Turn Quest")
