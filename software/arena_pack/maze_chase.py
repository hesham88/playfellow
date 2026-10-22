"""Arena level 4 (maze): Maze Chase. Move through the maze with arrows, collect coins, grab the key,
evade the chaser, and unlock the exit door. / المستوى 4: مطاردة المتاهة."""
from __future__ import annotations

from collections import deque
from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

MAZE = (
    "############",
    "#P...#....c#",
    "#.##.#.##..#",
    "#c#..c..#..#",
    "#.#.###.#.##",
    "#...#k...G.#",
    "#.###.###..#",
    "#c.......c.D",
    "############",
)

TILE = 48
COIN_POINTS = 10
KEY_POINTS = 50
DOOR_POINTS = 100
LOSE_PENALTY = 50

WALL_COLOR = (62, 64, 86)
FLOOR_COLOR = (226, 232, 242)
FLOOR_LINE = (204, 211, 224)
COIN_COLOR = (255, 196, 0)
KEY_COLOR = (230, 170, 30)
DOOR_COLOR = (140, 86, 40)
DOOR_OPEN = (60, 200, 110)
PLAYER_COLOR = (40, 110, 255)
CHASER_COLOR = (230, 50, 50)

MOVES = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class MazeChase(ArenaPackGame):
    name = "Maze Chase"
    name_ar = "مطاردة المتاهة"
    keys = ("up", "down", "left", "right")
    max_ticks = 800
    state_fields = ("player", "chaser", "coins", "has_key", "won", "lost")

    def new_game(self) -> None:
        self.walls: set[tuple[int, int]] = set()
        self.coins: set[tuple[int, int]] = set()
        self.key_pos: tuple[int, int] | None = None
        self.door_pos: tuple[int, int] = (0, 0)
        self.player: tuple[int, int] = (1, 1)
        self.chaser: tuple[int, int] = (9, 5)

        for y, row in enumerate(MAZE):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.walls.add((x, y))
                elif cell == "c":
                    self.coins.add((x, y))
                elif cell == "k":
                    self.key_pos = (x, y)
                elif cell == "D":
                    self.door_pos = (x, y)
                elif cell == "P":
                    self.player = (x, y)
                elif cell == "G":
                    self.chaser = (x, y)

        self.has_key = False
        self.won = False
        self.lost = False

    def is_won(self) -> bool:
        return self.won

    def is_lost(self) -> bool:
        return self.lost

    def progress(self) -> dict:
        return {
            "coins_left": len(self.coins),
            "has_key": self.has_key,
            "won": self.won,
            "lost": self.lost,
        }

    def progress_text(self) -> str:
        if self.won:
            return "Escaped!"
        if self.lost:
            return "Caught!"
        key_str = "yes" if self.has_key else "no"
        return f"Coins: {len(self.coins)}  Key: {key_str}"

    def board_pixels(self) -> tuple[int, int]:
        return len(MAZE[0]) * TILE, len(MAZE) * TILE

    def _shortest_step(self, start: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
        """BFS step from start towards target avoiding walls."""
        if start == target:
            return start
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == target:
                return path[1] if len(path) > 1 else curr
            for dx, dy in MOVES.values():
                nxt = (curr[0] + dx, curr[1] + dy)
                if nxt not in self.walls and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return start

    def on_key(self, key: str) -> None:
        if self.done:
            return
        dx, dy = MOVES.get(key, (0, 0))
        target = (self.player[0] + dx, self.player[1] + dy)

        if target in self.walls:
            return

        if target == self.door_pos:
            if self.has_key:
                self.player = target
                self.score += DOOR_POINTS
                self.won = True
                return
            return

        self.player = target
        if target in self.coins:
            self.coins.remove(target)
            self.score += COIN_POINTS
        elif target == self.key_pos and not self.has_key:
            self.has_key = True
            self.key_pos = None
            self.score += KEY_POINTS

        if self.player == self.chaser:
            self.lost = True
            self.score -= LOSE_PENALTY

    def tick(self) -> None:
        if self.done:
            return
        # Chaser moves every 2 ticks
        if self.ticks % 2 == 0:
            next_chaser = self._shortest_step(self.chaser, self.player)
            self.chaser = next_chaser
            if self.chaser == self.player:
                self.lost = True
                self.score -= LOSE_PENALTY

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        # Draw background and floor grid
        w, h = self.board_pixels()
        draw.rectangle((0, top, w, top + h), fill=FLOOR_COLOR)

        for y in range(len(MAZE)):
            for x in range(len(MAZE[0])):
                rx, ry = x * TILE, top + y * TILE
                draw.rectangle((rx, ry, rx + TILE, ry + TILE), outline=FLOOR_LINE)

        # Draw walls
        for wx, wy in self.walls:
            rx, ry = wx * TILE, top + wy * TILE
            draw.rectangle((rx, ry, rx + TILE, ry + TILE), fill=WALL_COLOR)

        # Draw door
        dx, dy = self.door_pos
        drx, dry = dx * TILE, top + dy * TILE
        d_color = DOOR_OPEN if self.has_key else DOOR_COLOR
        draw.rectangle((drx + 4, dry + 4, drx + TILE - 4, dry + TILE - 4), fill=d_color)

        # Draw coins
        for cx, cy in self.coins:
            crx, cry = cx * TILE + TILE // 2, top + cy * TILE + TILE // 2
            r = TILE // 5
            draw.ellipse((crx - r, cry - r, crx + r, cry + r), fill=COIN_COLOR)

        # Draw key
        if self.key_pos is not None:
            kx, ky = self.key_pos
            krx, kry = kx * TILE + TILE // 2, top + ky * TILE + TILE // 2
            r = TILE // 4
            draw.rectangle((krx - r, kry - r // 2, krx + r, kry + r // 2), fill=KEY_COLOR)

        # Draw chaser
        gx, gy = self.chaser
        grx, gry = gx * TILE + TILE // 2, top + gy * TILE + TILE // 2
        r = TILE // 3
        draw.ellipse((grx - r, gry - r, grx + r, gry + r), fill=CHASER_COLOR)

        # Draw player
        px, py = self.player
        prx, pry = px * TILE + TILE // 2, top + py * TILE + TILE // 2
        r = TILE // 3
        draw.ellipse((prx - r, pry - r, prx + r, pry + r), fill=PLAYER_COLOR)


if __name__ == "__main__":
    run_window(MazeChase)
