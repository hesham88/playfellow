"""Grounding: putting sticky labels on the picture so the thinking part can point at things.
/ التثبيت: وضع ملصقات على الصورة ليتمكن الجزء المفكر من الإشارة إلى الأشياء.

Like a teacher circling things on a photo with a red pen ("this is you, this is a coin, this is a danger"), the
buddy draws [ME], [R1] (reward), [H1] (hazard) and [D1] (door) on the picture before the AI looks at it. The AI can
then answer "go to R1" instead of guessing pixel positions. Nothing here knows a particular game.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

GRID = 1000  # the same 0-1000 grid the mouse clicks use
KINDS = {"me": "ME", "reward": "R", "hazard": "H", "door": "D"}
COLOURS = {"me": (34, 228, 255), "reward": (182, 255, 59), "hazard": (255, 61, 203), "door": (124, 92, 255)}
MAX_PER_KIND = 6  # more labels than this just clutters the picture
HUD_CHANGE = 12.0  # average colour change (0-255) in a HUD patch that counts as "the number changed"
AUTOPSY_SECONDS = 3.0


@dataclass(frozen=True)
class Mark:
    """One thing of interest, with its centre on the 0-1000 grid."""

    kind: str  # "me", "reward", "hazard" or "door"
    x: int
    y: int
    tag: str = ""  # filled in by label_marks(): ME, R1, H2, ...


def mark_from_box(kind: str, box: tuple[int, int, int, int], size: tuple[int, int]) -> Mark:
    """Turn a pixel box (left, top, right, bottom) into a mark on the 0-1000 grid."""
    width, height = size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return Mark(kind, round(cx / max(width, 1) * GRID), round(cy / max(height, 1) * GRID))


def label_marks(marks: list[Mark], me: Mark | None = None) -> list[Mark]:
    """Give every mark its tag. ME comes first; the others are numbered in the order given, nearest-to-ME first."""
    out: list[Mark] = []
    if me is not None:
        out.append(Mark("me", me.x, me.y, "ME"))
    for kind in ("reward", "hazard", "door"):
        same = [m for m in marks if m.kind == kind]
        if me is not None:
            same.sort(key=lambda m: math.hypot(m.x - me.x, m.y - me.y))
        for number, mark in enumerate(same[:MAX_PER_KIND], start=1):
            out.append(Mark(kind, mark.x, mark.y, f"{KINDS[kind]}{number}"))
    return out


def draw_marks(image: Image.Image, marks: list[Mark], arrows: bool = True) -> Image.Image:
    """Return a copy of the picture with a ring and a tag on every mark and a compass arrow from ME to each."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    radius = max(6, min(width, height) // 40)
    me = next((m for m in marks if m.kind == "me"), None)
    for mark in marks:
        px, py = mark.x / GRID * width, mark.y / GRID * height
        colour = COLOURS.get(mark.kind, (255, 255, 255))
        if arrows and me is not None and mark is not me:
            draw_arrow(draw, (me.x / GRID * width, me.y / GRID * height), (px, py), colour, radius)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=colour, width=2)
        text = f"[{mark.tag or KINDS.get(mark.kind, '?')}]"
        left, top, right, bottom = draw.textbbox((px + radius + 2, py - radius), text)
        draw.rectangle((left - 1, top - 1, right + 1, bottom + 1), fill=(11, 11, 26))
        draw.text((left, top), text, fill=colour)
    return canvas


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float],
               colour: tuple[int, int, int], gap: float) -> None:
    """A thin line from start to end that stops short of both rings, with a small arrowhead."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= gap * 3:
        return
    ux, uy = dx / length, dy / length
    a = (start[0] + ux * gap, start[1] + uy * gap)
    b = (end[0] - ux * gap, end[1] - uy * gap)
    draw.line((*a, *b), fill=colour, width=1)
    head = gap * 0.8
    for side in (-0.5, 0.5):
        angle = math.atan2(-uy, -ux) + side
        draw.line((*b, b[0] + math.cos(angle) * head, b[1] + math.sin(angle) * head), fill=colour, width=1)


def compass(me: Mark, target: Mark) -> str:
    """Plain words for where the target is from ME: 'up-right, far'. Screen y grows downwards."""
    dx, dy = target.x - me.x, me.y - target.y
    if abs(dx) < 40 and abs(dy) < 40:
        return "right here"
    vertical = "up" if dy > 0 else "down"
    horizontal = "right" if dx > 0 else "left"
    if abs(dx) < abs(dy) * 0.4:
        way = vertical
    elif abs(dy) < abs(dx) * 0.4:
        way = horizontal
    else:
        way = f"{vertical}-{horizontal}"
    distance = math.hypot(dx, dy)
    return f"{way}, {'far' if distance > 450 else 'near' if distance < 180 else 'a bit away'}"


def legend(marks: list[Mark]) -> str:
    """The text that goes next to the marked picture so the AI knows what the labels mean."""
    me = next((m for m in marks if m.kind == "me"), None)
    names = {"reward": "something good to get", "hazard": "danger, avoid", "door": "a way through"}
    lines = ["Labels drawn on the picture:"]
    for mark in marks:
        if mark.kind == "me":
            lines.append("[ME] = you")
        else:
            where = f" ({compass(me, mark)})" if me is not None else ""
            lines.append(f"[{mark.tag}] = {names.get(mark.kind, 'thing')}{where}")
    return "\n".join(lines)


class HudWatcher:
    """Notices when a number on the screen changes (score, lives) without asking the AI. Costs no tokens."""

    def __init__(self, patches: dict[str, tuple[float, float, float, float]], threshold: float = HUD_CHANGE):
        self.patches = patches  # name -> (left, top, right, bottom) as fractions of the picture
        self.threshold = threshold
        self._last: dict[str, np.ndarray] = {}

    def _crop(self, image: Image.Image, frac: tuple[float, float, float, float]) -> np.ndarray:
        width, height = image.size
        box = (int(frac[0] * width), int(frac[1] * height), max(int(frac[2] * width), int(frac[0] * width) + 1),
               max(int(frac[3] * height), int(frac[1] * height) + 1))
        return np.asarray(image.convert("L").crop(box), dtype=np.int16)

    def update(self, image: Image.Image) -> list[str]:
        """Names of the patches that changed since the last look. The first look never reports a change."""
        changed = []
        for name, frac in self.patches.items():
            now = self._crop(image, frac)
            before = self._last.get(name)
            if before is not None and (before.shape != now.shape or float(np.abs(now - before).mean()) > self.threshold):
                changed.append(name)
            self._last[name] = now
        return changed


@dataclass
class Autopsy:
    """What happened in the seconds before something went wrong."""

    frames: list[Image.Image]
    seconds: float
    note: str = ""


@dataclass
class FailureRecorder:
    """Keeps the last few seconds of small pictures so that after a death we can look back at what led to it."""

    clock: Callable[[], float]
    window: float = AUTOPSY_SECONDS
    keep: int = 4  # pictures handed to the AI, spread evenly over the window
    _frames: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)  # the tick adds, the planner reads

    def add(self, image: Image.Image, thumb: int = 320) -> None:
        now = self.clock()
        small = image.convert("RGB").copy()
        small.thumbnail((thumb, thumb))
        with self._lock:
            self._frames.append((now, small))
            while self._frames and now - self._frames[0][0] > self.window:
                self._frames.popleft()

    def autopsy(self, note: str = "") -> Autopsy | None:
        with self._lock:
            items = list(self._frames)
        if not items:
            return None
        if len(items) > self.keep:
            picks = [items[round(i * (len(items) - 1) / (self.keep - 1))] for i in range(self.keep)]
        else:
            picks = items
        return Autopsy([p[1] for p in picks], round(items[-1][0] - items[0][0], 2), note)
