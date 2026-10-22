"""Arena level 8 (first-person): Ray Maze. Walk a maze seen through your own eyes. W/S step forward/back, A/D step
sideways, left/right turn, space shoots straight ahead. Hit all 5 red targets, then walk through the green exit.
Ammo is limited: yellow boxes give more. / المستوى 8: متاهة بمنظور الشخص الأول. أصب 5 أهداف ثم اخرج من الباب الأخضر."""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from arena_pack.base import ArenaPackGame, run_window

# '#' wall, '.' floor, 'T' target, 'A' ammo, 'E' exit door (a wall you can walk through once all targets are hit), 'P' start
MAZE = (
    "##########",
    "#P..#..T.#",
    "#.#.#.##.#",
    "#.#...#..#",
    "#T#.#A#.##",
    "#...#...T#",
    "###.###.##",
    "#T....#..#",
    "#..A#.T..E",
    "##########",
)
FACINGS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # north, east, south, west
TARGETS_TO_HIT, START_AMMO, AMMO_BOX = 5, 3, 2
VIEW_W, VIEW_H, FOV = 480, 300, math.radians(66)
COLUMNS = 120
MAP_CELL = 9
SKY, FLOOR = (24, 22, 48), (52, 46, 64)
WALL_A, WALL_B, EXIT_OPEN, EXIT_SHUT = (124, 92, 255), (92, 68, 200), (90, 220, 120), (40, 110, 60)
TARGET, AMMO = (230, 60, 60), (255, 214, 60)
HIT_POINTS, EXIT_POINTS = 100, 300


class RayMaze(ArenaPackGame):
    name = "Ray Maze"
    name_ar = "متاهة الأشعة"
    keys = ("w", "s", "a", "d", "left", "right", "space")
    max_ticks = 800
    state_fields = ("x", "y", "facing", "targets", "ammo_boxes", "ammo", "hits", "escaped")

    def new_game(self) -> None:
        cells = [(x, y, c) for y, row in enumerate(MAZE) for x, c in enumerate(row)]
        self.x, self.y = next((x, y) for x, y, c in cells if c == "P")
        self.facing = 1
        self.targets = sorted((x, y) for x, y, c in cells if c == "T")
        self.ammo_boxes = sorted((x, y) for x, y, c in cells if c == "A")
        self.exit = next((x, y) for x, y, c in cells if c == "E")
        self.ammo = START_AMMO
        self.hits = 0
        self.escaped = False

    def _wall(self, x: int, y: int) -> bool:
        return MAZE[y][x] == "#" or (MAZE[y][x] == "E" and self.hits < TARGETS_TO_HIT)

    def on_key(self, key: str) -> None:
        if key in ("left", "right"):
            self.facing = (self.facing + (1 if key == "right" else -1)) % 4
            return
        if key == "space":
            self._shoot()
            return
        dx, dy = FACINGS[self.facing]
        if key == "s":
            dx, dy = -dx, -dy
        elif key in ("a", "d"):  # sideways: a is to your left
            dx, dy = (dy, -dx) if key == "a" else (-dy, dx)
        nx, ny = self.x + dx, self.y + dy
        if self._wall(nx, ny) or (nx, ny) in self.targets:
            return
        self.x, self.y = nx, ny
        if (nx, ny) in self.ammo_boxes:
            self.ammo_boxes.remove((nx, ny))
            self.ammo += AMMO_BOX
        if (nx, ny) == self.exit:
            self.escaped = True
            self.score += EXIT_POINTS

    def _shoot(self) -> None:
        if not self.ammo:
            return
        self.ammo -= 1
        dx, dy = FACINGS[self.facing]
        x, y = self.x + dx, self.y + dy
        while not self._wall(x, y):
            if (x, y) in self.targets:
                self.targets.remove((x, y))
                self.hits += 1
                self.score += HIT_POINTS
                return
            x, y = x + dx, y + dy

    def is_won(self) -> bool:
        return self.escaped

    def is_lost(self) -> bool:
        return not self.escaped and self.ammo == 0 and not self.ammo_boxes and self.hits < TARGETS_TO_HIT

    def progress(self) -> dict:
        return {"hits": self.hits, "targets": TARGETS_TO_HIT, "ammo": self.ammo, "ammo_boxes_left": len(self.ammo_boxes),
                "exit_open": self.hits >= TARGETS_TO_HIT, "facing": "NESW"[self.facing]}

    def progress_text(self) -> str:
        if self.language == "arabic":
            return f"الأهداف {self.hits}/{TARGETS_TO_HIT}    الذخيرة {self.ammo}    الاتجاه {'شمال شرق جنوب غرب'.split()[self.facing]}"
        return f"Targets {self.hits}/{TARGETS_TO_HIT}    Ammo {self.ammo}    Facing {'North East South West'.split()[self.facing]}"

    def board_pixels(self) -> tuple[int, int]:
        return VIEW_W, VIEW_H

    # ---- drawing: walls by ray casting, then targets/ammo as sprites behind a depth buffer
    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        draw.rectangle((0, top, VIEW_W, top + VIEW_H // 2), fill=SKY)
        draw.rectangle((0, top + VIEW_H // 2, VIEW_W, top + VIEW_H), fill=FLOOR)
        px, py = self.x + 0.5, self.y + 0.5
        heading = math.atan2(FACINGS[self.facing][1], FACINGS[self.facing][0])
        width = VIEW_W / COLUMNS
        depth = []
        for col in range(COLUMNS):
            angle = heading + (col / COLUMNS - 0.5) * FOV
            dist, side, cell = self._cast(px, py, math.cos(angle), math.sin(angle))
            dist *= math.cos(angle - heading)  # no fish-eye
            depth.append(dist)
            height = min(VIEW_H, VIEW_H / max(dist, 0.05))
            if MAZE[cell[1]][cell[0]] == "E":
                colour = EXIT_OPEN if self.hits >= TARGETS_TO_HIT else EXIT_SHUT
            else:
                colour = WALL_A if side == 0 else WALL_B
            shade = max(0.35, 1 - dist / 9)
            colour = tuple(int(c * shade) for c in colour)
            mid = top + VIEW_H / 2
            draw.rectangle((col * width, mid - height / 2, (col + 1) * width, mid + height / 2), fill=colour)
        things = [(t, TARGET, 0.32) for t in self.targets] + [(a, AMMO, 0.18) for a in self.ammo_boxes]
        for (tx, ty), colour, radius in sorted(things, key=lambda t: -math.hypot(t[0][0] + 0.5 - px, t[0][1] + 0.5 - py)):
            rx, ry = tx + 0.5 - px, ty + 0.5 - py
            ahead = rx * math.cos(heading) + ry * math.sin(heading)
            across = -rx * math.sin(heading) + ry * math.cos(heading)
            if ahead <= 0.2:
                continue
            col = (0.5 + math.atan2(across, ahead) / FOV) * COLUMNS
            if not 0 <= col < COLUMNS or depth[int(col)] < ahead:
                continue  # behind a wall or out of view
            size = VIEW_H * radius / ahead
            cx, cy = col * width, top + VIEW_H / 2 + (size * 0.6 if colour == AMMO else 0)
            draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=colour, outline=(255, 255, 255), width=2)
        # a small map in the corner, as old first-person games had
        for y, row in enumerate(MAZE):
            for x, c in enumerate(row):
                fill = (90, 90, 120) if c == "#" else (EXIT_OPEN if c == "E" else (30, 30, 50))
                draw.rectangle((6 + x * MAP_CELL, top + 6 + y * MAP_CELL, 6 + x * MAP_CELL + MAP_CELL - 1, top + 6 + y * MAP_CELL + MAP_CELL - 1), fill=fill)
        mx, my = 6 + self.x * MAP_CELL + MAP_CELL / 2, top + 6 + self.y * MAP_CELL + MAP_CELL / 2
        fx, fy = FACINGS[self.facing]
        draw.polygon([(mx + fx * 5, my + fy * 5), (mx - fy * 3 - fx * 3, my + fx * 3 - fy * 3), (mx + fy * 3 - fx * 3, my - fx * 3 - fy * 3)],
                     fill=(182, 255, 59))

    def _cast(self, px: float, py: float, dx: float, dy: float) -> tuple[float, int, tuple[int, int]]:
        """DDA ray march: distance to the first wall, which side it hit, and the wall cell."""
        mx, my = int(px), int(py)
        delta_x = abs(1 / dx) if dx else 1e9
        delta_y = abs(1 / dy) if dy else 1e9
        step_x, side_x = (-1, (px - mx) * delta_x) if dx < 0 else (1, (mx + 1 - px) * delta_x)
        step_y, side_y = (-1, (py - my) * delta_y) if dy < 0 else (1, (my + 1 - py) * delta_y)
        for _ in range(64):
            if side_x < side_y:
                side_x += delta_x
                mx += step_x
                side = 0
            else:
                side_y += delta_y
                my += step_y
                side = 1
            if MAZE[my][mx] in "#E":
                return (side_x - delta_x if side == 0 else side_y - delta_y), side, (mx, my)
        return 20.0, 0, (mx, my)


if __name__ == "__main__":
    run_window(RayMaze, "PlayFellow Arena: Ray Maze")
