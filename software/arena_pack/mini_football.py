"""Arena level 12 (football): Mini Football. Five against five, seen from above; you attack the goal on the right.
Arrows move (and dribble), space shoots straight ahead, Z passes forward. Win by 2 goals within 3 minutes.
/ المستوى 12: كرة قدم خماسية نراها من الأعلى؛ تهاجم المرمى الأيمن. الأسهم للحركة، المسافة للتسديد، Z للتمرير."""
from __future__ import annotations

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

W, H = 15, 9
GOAL_ROWS = (3, 4, 5)
MINUTES_TICKS = 1800  # 3 minutes at 10 ticks a second
WIN_LEAD = 2
SHOT_RANGE, PASS_RANGE = 9, 8
KEEPER_EVERY = 3  # a keeper moves one row every 3 ticks: a quick change of row near goal beats it
STEAL_CPU, STEAL_YOU = 0.3, 0.6  # chance per tick of winning the ball off the carrier next to you
GOAL_POINTS = 100
OURS = ((2, 2), (2, 6), (5, 4), (8, 2))  # home squares of your four outfield players
THEIRS = ((12, 2), (12, 6), (9, 4), (6, 6))
TILE = 40
GRASS, STRIPE, LINES = (40, 120, 60), (44, 130, 66), (230, 240, 230)
YOU, MATE, CPU, KEEPER, BALL = (124, 92, 255), (90, 120, 230), (230, 70, 70), (255, 214, 60), (250, 250, 250)


def toward(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """One step from a towards b (horizontal first when farther that way)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) >= abs(dy) and dx:
        return a[0] + (1 if dx > 0 else -1), a[1]
    if dy:
        return a[0], a[1] + (1 if dy > 0 else -1)
    return a


def near(a, b) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


class MiniFootball(ArenaPackGame):
    name = "Mini Football"
    name_ar = "كرة القدم المصغرة"
    keys = ("up", "down", "left", "right", "space", "z")
    max_ticks = MINUTES_TICKS + 5
    state_fields = ("ours", "theirs", "keepers", "ball", "owner", "flight", "goals", "clock")

    def new_game(self) -> None:
        self.goals = [0, 0]  # you, computer
        self.clock = 0
        self._kickoff()

    def _kickoff(self) -> None:
        self.ours = [list(p) for p in OURS]
        self.theirs = [list(p) for p in THEIRS]
        self.keepers = [4, 4]  # rows of your keeper (left goal) and theirs (right goal)
        self.ball = [W // 2, H // 2]
        self.owner = None  # ("you", i) or ("cpu", i) or ("keeper", side)
        self.flight = None  # [dx or teammate, dy, squares left, shot?] while the ball travels

    # ---- who is who
    def active(self) -> int:
        """Your player under control: the one with the ball, else the one nearest it."""
        if self.owner and self.owner[0] == "you":
            return self.owner[1]
        return min(range(4), key=lambda i: abs(self.ours[i][0] - self.ball[0]) + abs(self.ours[i][1] - self.ball[1]))

    @property
    def have_ball(self) -> bool:
        return bool(self.owner) and self.owner[0] == "you"

    # ---- your keys
    def on_key(self, key: str) -> None:
        me = self.ours[self.active()]
        if key in ("up", "down", "left", "right"):
            dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[key]
            nx, ny = me[0] + dx, me[1] + dy
            if 0 <= nx < W and 0 <= ny < H:
                me[0], me[1] = nx, ny
                if self.have_ball:
                    self.ball = list(me)
                elif self.owner and self.owner[0] == "cpu" and near(me, self.theirs[self.owner[1]]):
                    if self.rng.random() < STEAL_YOU:
                        self.owner, self.ball = ("you", self.active()), list(me)
        elif key == "space" and self.have_ball:
            self.owner, self.flight = None, [1, 0, SHOT_RANGE, True]
        elif key == "z" and self.have_ball:
            ahead = [i for i in range(4) if i != self.active() and self.ours[i][0] > me[0]]
            if ahead:
                mate = max(ahead, key=lambda i: self.ours[i][0])
                self.owner, self.flight = None, [mate, None, PASS_RANGE, False]

    # ---- the clock
    def tick(self) -> None:
        self.clock += 1
        if self.flight:
            self._fly()
        self._keepers()
        self._cpu()
        if not self.owner and not self.flight:
            for side, team in (("you", self.ours), ("cpu", self.theirs)):
                for i, p in enumerate(team):
                    if p == self.ball:
                        self.owner = (side, i)
                        return

    def _fly(self) -> None:
        dx, dy, left, shot = self.flight
        if shot:
            self.ball[0] += dx
        else:  # a pass heads for the teammate's square
            self.ball = list(toward(self.ball, tuple(self.ours[dx])))
        self.flight[2] -= 1
        x, y = self.ball
        if shot and x >= W - 1:
            if y in GOAL_ROWS and self.keepers[1] != y:
                self._goal(0)
            else:
                self.owner, self.flight, self.ball = ("keeper", 1), None, [W - 1, self.keepers[1]]
            return
        for i, p in enumerate(self.theirs):  # interception
            if p == self.ball:
                self.owner, self.flight = ("cpu", i), None
                return
        if not shot and self.ball == self.ours[dx]:
            self.owner, self.flight = ("you", dx), None
        elif self.flight[2] <= 0:
            self.flight = None

    def _keepers(self) -> None:
        if self.owner == ("keeper", 1):  # their keeper clears the ball to a defender
            self.owner, self.ball = ("cpu", 0), list(self.theirs[0])
        if self.owner == ("keeper", 0):
            self.owner, self.ball = ("you", 0), list(self.ours[0])
        if self.clock % KEEPER_EVERY == 0:
            target = min(max(self.ball[1], GOAL_ROWS[0]), GOAL_ROWS[-1])
            self.keepers[1] += (target > self.keepers[1]) - (target < self.keepers[1])  # keepers keep moving while a shot flies
            self.keepers[0] += (target > self.keepers[0]) - (target < self.keepers[0])

    def _cpu(self) -> None:
        if self.owner and self.owner[0] == "cpu":
            i = self.owner[1]
            carrier = self.theirs[i]
            if carrier[0] <= 3:  # shoot at your goal
                y = carrier[1]
                self.owner, self.ball = ("you", 0), list(self.ours[0])
                if y in GOAL_ROWS and self.keepers[0] != y:
                    self._goal(1)
                return
            if self.clock % 2 == 0:
                carrier[:] = toward(carrier, (0, 4))
                self.ball = list(carrier)
            return
        if self.clock % 2:
            return  # computer players move every other tick
        chasers = sorted(range(4), key=lambda i: abs(self.theirs[i][0] - self.ball[0]) + abs(self.theirs[i][1] - self.ball[1]))[:2]
        for i in chasers:
            self.theirs[i][:] = toward(self.theirs[i], tuple(self.ball))
            if self.have_ball and near(self.theirs[i], self.ours[self.owner[1]]) and self.rng.random() < STEAL_CPU:
                self.owner, self.ball = ("cpu", i), list(self.theirs[i])

    def _goal(self, side: int) -> None:
        self.goals[side] += 1
        if side == 0:
            self.score += GOAL_POINTS
        self._kickoff()

    # ---- results
    def is_won(self) -> bool:
        return self.goals[0] - self.goals[1] >= WIN_LEAD

    def is_lost(self) -> bool:
        return self.clock >= MINUTES_TICKS and not self.is_won()

    def seconds_per_tick(self) -> float:
        return 0.1

    def progress(self) -> dict:
        return {"you": self.goals[0], "computer": self.goals[1], "seconds_left": max(0, MINUTES_TICKS - self.clock) // 10,
                "have_ball": self.have_ball, "ball": tuple(self.ball)}

    def progress_text(self) -> str:
        left = max(0, MINUTES_TICKS - self.clock) // 10
        if self.language == "arabic":
            return f"أنت {self.goals[0]} - {self.goals[1]} الحاسوب    الوقت المتبقي {left // 60}:{left % 60:02d}"
        return f"You {self.goals[0]} - {self.goals[1]} Computer    Time left {left // 60}:{left % 60:02d}"

    def board_pixels(self) -> tuple[int, int]:
        return (W + 2) * TILE, H * TILE

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        for x in range(W + 2):
            draw.rectangle((x * TILE, top, x * TILE + TILE - 1, top + H * TILE), fill=STRIPE if x % 2 else GRASS)
        draw.rectangle((TILE, top, (W + 1) * TILE - 1, top + H * TILE - 1), outline=LINES, width=2)
        draw.line(((W // 2 + 1) * TILE + TILE // 2, top, (W // 2 + 1) * TILE + TILE // 2, top + H * TILE), fill=LINES, width=2)
        for gx in (0, W + 1):
            draw.rectangle((gx * TILE + 8, top + GOAL_ROWS[0] * TILE, gx * TILE + TILE - 8, top + (GOAL_ROWS[-1] + 1) * TILE),
                           outline=LINES, width=3)

        def dot(x, y, colour, size=14, ring=None):
            cx, cy = (x + 1) * TILE + TILE // 2, top + y * TILE + TILE // 2
            draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=colour, outline=ring, width=3)
        dot(-1, self.keepers[0], KEEPER, ring=MATE)
        dot(W, self.keepers[1], KEEPER, ring=CPU)
        for x, y in self.theirs:
            dot(x, y, CPU)
        active = self.active()
        for i, (x, y) in enumerate(self.ours):
            if i != active:
                dot(x, y, MATE)
        dot(*self.ours[active], YOU, ring=(255, 255, 255))  # drawn last: the player you control is always visible
        dot(self.ball[0], self.ball[1], BALL, size=6, ring=(30, 30, 30))


if __name__ == "__main__":
    run_window(MiniFootball, "PlayFellow Arena: Mini Football")
