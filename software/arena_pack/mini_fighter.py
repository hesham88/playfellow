"""Arena level 14 (fighting): Mini Fighter. One-on-one against the computer, seen from the side. Left/right walk,
A punch (quick, short), S kick (slower, longer, stronger), D block, F special (when the meter is full). Win 2 rounds.
/ المستوى 14: قتال فردي ضد الحاسوب. A لكمة، S ركلة، D صد، F حركة خاصة. افز بجولتين."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

STAGE = 24  # positions 0..24
MOVES = {"punch": {"range": 2, "windup": 1, "damage": 7, "recover": 2},
         "kick": {"range": 3, "windup": 2, "damage": 11, "recover": 4},
         "special": {"range": 4, "windup": 2, "damage": 28, "recover": 5}}
STUN, BLOCK_CHIP, BLOCK_PUNISH = 2, 1, 2
ROUND_TICKS, ROUNDS_TO_WIN, MAX_HP, METER_FULL = 300, 2, 100, 100
SCALE, FLOOR = 24, 260
YOU, CPU = (124, 92, 255), (230, 70, 70)


def fighter(x: int) -> dict:
    return {"x": x, "hp": MAX_HP, "meter": 0, "move": None, "timer": 0, "stun": 0, "block": False}


class MiniFighter(ArenaPackGame):
    name = "Mini Fighter"
    name_ar = "المقاتل الصغير"
    keys = ("left", "right", "a", "s", "d", "f")
    max_ticks = ROUND_TICKS * 3 + 30
    state_fields = ("you", "cpu", "round_clock", "wins")

    def new_game(self) -> None:
        self.wins = [0, 0]  # you, computer
        self._new_round()

    def _new_round(self) -> None:
        self.you, self.cpu = fighter(6), fighter(18)
        self.round_clock = 0

    # ---- shared rules for both fighters
    def _act(self, me: dict, other: dict, action: str | None, toward: int) -> None:
        me["block"] = False
        if me["stun"] or me["move"]:
            return  # busy: stunned, or in the middle of an attack
        if action == "block":
            me["block"] = True
        elif action == "forward":
            if abs(me["x"] + toward - other["x"]) >= 1:
                me["x"] = max(0, min(STAGE, me["x"] + toward))
        elif action == "back":
            me["x"] = max(0, min(STAGE, me["x"] - toward))
        elif action in MOVES and (action != "special" or me["meter"] >= METER_FULL):
            if action == "special":
                me["meter"] = 0
            me["move"], me["timer"] = action, MOVES[action]["windup"]

    def _resolve(self, me: dict, other: dict) -> None:
        """Advance my attack one tick. timer > 0: winding up (it lands when it reaches 0); timer < 0: recovering."""
        if me["stun"]:
            me["stun"] -= 1
            return
        if not me["move"]:
            return
        if me["timer"] < 0:
            me["timer"] += 1
            if me["timer"] == 0:
                me["move"] = None  # recovered: free to act again
            return
        me["timer"] -= 1
        if me["timer"] > 0:
            return
        move = MOVES[me["move"]]
        recover = move["recover"]
        if abs(me["x"] - other["x"]) <= move["range"]:
            if other["block"]:
                other["hp"] -= BLOCK_CHIP
                recover += BLOCK_PUNISH  # a blocked attack leaves you open a little longer
            else:
                other["hp"] -= move["damage"]
                other["stun"], other["move"] = STUN, None
                me["meter"] = min(METER_FULL, me["meter"] + move["damage"] * 2)
                other["meter"] = min(METER_FULL, other["meter"] + move["damage"])
        me["timer"] = -recover

    # ---- you and the computer
    def on_key(self, key: str) -> None:
        toward = 1 if self.cpu["x"] > self.you["x"] else -1
        action = {"a": "punch", "s": "kick", "d": "block", "f": "special"}.get(key)
        if key in ("left", "right"):
            action = "forward" if (key == "right") == (toward == 1) else "back"
        self._act(self.you, self.cpu, action, toward)

    def _cpu_choice(self) -> str | None:
        c, y = self.cpu, self.you
        gap = abs(c["x"] - y["x"])
        if y["move"] and y["timer"] > 0 and gap <= MOVES[y["move"]]["range"] and self.rng.random() < 0.4:
            return "block"
        if c["meter"] >= METER_FULL and gap <= MOVES["special"]["range"]:
            return "special"
        if gap <= MOVES["kick"]["range"] and self.rng.random() < 0.45:
            return "punch" if gap <= MOVES["punch"]["range"] and self.rng.random() < 0.5 else "kick"
        if gap > MOVES["punch"]["range"] and self.rng.random() < 0.7:
            return "forward"
        return None

    def tick(self) -> None:
        if self.you["move"] or self.you["stun"]:
            self.you["block"] = False
        self._act(self.cpu, self.you, self._cpu_choice(), 1 if self.you["x"] > self.cpu["x"] else -1)
        self._resolve(self.you, self.cpu)
        self._resolve(self.cpu, self.you)
        self.round_clock += 1
        if self.you["hp"] <= 0 or self.cpu["hp"] <= 0 or self.round_clock >= ROUND_TICKS:
            winner = 0 if self.you["hp"] > self.cpu["hp"] else 1
            self.wins[winner] += 1
            if winner == 0:
                self.score += 100
            if max(self.wins) < ROUNDS_TO_WIN:
                self._new_round()

    def seconds_per_tick(self) -> float:
        return 0.1

    def is_won(self) -> bool:
        return self.wins[0] >= ROUNDS_TO_WIN

    def is_lost(self) -> bool:
        return self.wins[1] >= ROUNDS_TO_WIN

    def progress(self) -> dict:
        return {"your_hp": self.you["hp"], "cpu_hp": self.cpu["hp"], "rounds": tuple(self.wins), "meter": self.you["meter"],
                "gap": abs(self.you["x"] - self.cpu["x"]), "cpu_attacking": self.cpu["move"], "time_left": ROUND_TICKS - self.round_clock}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"الجولات {self.wins[0]}-{self.wins[1]}    صحتك {max(0, self.you['hp'])}    الحاسوب {max(0, self.cpu['hp'])}    الطاقة {self.you['meter']}"
        return f"Rounds {self.wins[0]}-{self.wins[1]}    You {max(0, self.you['hp'])}    CPU {max(0, self.cpu['hp'])}    Meter {self.you['meter']}"

    def board_pixels(self) -> tuple[int, int]:
        return (STAGE + 2) * SCALE, 300

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        width = self.board_pixels()[0]
        draw.rectangle((0, top, width, top + 300), fill=(26, 22, 50))
        draw.rectangle((0, top + FLOOR, width, top + 300), fill=(60, 50, 80))
        for i, (who, colour, side) in enumerate(((self.you, YOU, 0), (self.cpu, CPU, 1))):
            share = max(0, who["hp"]) / MAX_HP
            bar_x = 12 if side == 0 else width - 12 - 200
            draw.rectangle((bar_x, top + 10, bar_x + 200, top + 24), outline=(220, 220, 230))
            fill_x = bar_x if side == 0 else bar_x + int(200 * (1 - share))
            draw.rectangle((fill_x, top + 10, fill_x + int(200 * share), top + 24), fill=colour)
            draw.rectangle((bar_x, top + 28, bar_x + int(200 * who["meter"] / METER_FULL), top + 32), fill=(255, 214, 60))
            cx = (who["x"] + 1) * SCALE
            body_top = top + FLOOR - (60 if who["block"] else 90)
            draw.rounded_rectangle((cx - 10, body_top, cx + 10, top + FLOOR), radius=6, fill=colour,
                                   outline=(255, 255, 255) if who["stun"] else None, width=3)
            draw.ellipse((cx - 10, body_top - 20, cx + 10, body_top), fill=colour)
            facing = 1 if (self.cpu if side == 0 else self.you)["x"] > who["x"] else -1
            if who["move"]:
                reach = MOVES[who["move"]]["range"] * SCALE if who["timer"] <= 0 else 18
                y = body_top + (20 if who["move"] == "punch" else 50)
                draw.line((cx, y, cx + facing * reach, y), fill=(255, 214, 60) if who["move"] == "special" else colour, width=8)


if __name__ == "__main__":
    run_window(MiniFighter, "PlayFellow Arena: Mini Fighter")
