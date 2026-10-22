""""Which thing is me?" — the buddy learns its own body from its own key presses. / "أيّ شيء هو أنا؟"

Like waving your hand in front of a mirror: the thing that moves every time you move is you. Things that move on
their own (clouds, enemies) are the world. For every key press we compare the picture before and after:
  - find what moved and by how much (by matching moved content between the two pictures);
  - the thing whose moves follow the keys (LEFT moves it left every time, RIGHT moves it right…) is "me";
  - each key's usual move (pixels per press) is learned, and a press that should move "me" but doesn't is
    "blocked" — that is how walls are learned.
No colours or positions are given: everything comes from the buddy's own pictures and presses.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nav import camera as cam
from nav import frames

MIN_CELLS = 4  # a moved thing must cover at least this many blocks
MAX_POINTS = 1500  # more changed blocks than this in one step: the whole scene changed
SCENE_CHANGE = 0.25  # this share of the picture changed at once: a new scene (room, level)
CONSISTENT = 0.5  # a move agrees with a key's usual direction when their directions' dot product is above this
NEEDED_MOVES = 3  # "me" needs at least this many key-following moves…
NEEDED_DIRECTIONS = 2  # …in at least this many different directions
BLOCKED_SHARE = 0.3  # moving less than this share of the usual step along the key's direction = blocked
TELEPORT_STEPS = 3  # appearing further away than this many usual steps = a scene change, not a walk
GOOD_MATCH = 18.0  # mean colour gap for "that's still me" when matching my picture
FORGET_AFTER = 30  # a candidate not seen moving for this many steps is dropped


@dataclass
class Motion:
    """Something that moved between two pictures: by (dx, dy) pixels, now inside `box` (x0, y0, x1, y1)."""
    dx: int
    dy: int
    box: tuple[int, int, int, int]
    cells: int

    @property
    def old_box(self) -> tuple[int, int, int, int]:
        return shift(self.box, -self.dx, -self.dy)


@dataclass
class SelfView:
    """What the self-model says about the newest picture."""
    box: tuple[int, int, int, int] | None  # where "me" is now, in pixels (x0, y0, x1, y1); None = unknown yet
    confidence: float  # 0-1
    moved: tuple[int, int] | None = None  # how "me" moved this step, in pixels
    blocked: bool = False  # the key should have moved "me" but didn't
    scene_change: bool = False  # the whole scene changed (new room, level or screen)
    camera: tuple[int, int] = (0, 0)  # how far the background slid this step, in pixels (a following camera)

    @property
    def center(self) -> tuple[float, float] | None:
        if self.box is None:
            return None
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class _Candidate:
    box: tuple[int, int, int, int]
    moves: list[tuple[str, int, int]] = field(default_factory=list)  # (action, dx, dy)
    last_seen: int = 0


def shift(box, dx, dy):
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def overlap(a, b) -> int:
    """Overlapping area of two boxes (0 if apart)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _unit(dx: float, dy: float) -> np.ndarray:
    v = np.array([dx, dy], dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def _groups(points: np.ndarray, gap: int = 2) -> list[np.ndarray]:
    """Split block positions (row, col) into touching groups (blocks up to `gap` apart touch)."""
    left = {tuple(p) for p in points.tolist()}
    groups = []
    while left:
        start = left.pop()
        queue, group = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for dr in range(-gap, gap + 1):
                for dc in range(-gap, gap + 1):
                    nb = (r + dr, c + dc)
                    if nb in left:
                        left.remove(nb)
                        queue.append(nb)
                        group.append(nb)
        groups.append(np.array(group))
    return groups


def find_motions(before: np.ndarray, after: np.ndarray, cell: int,
                 region: tuple[int, int, int, int] | None = None) -> list[Motion] | None:
    """What moved between two shrunk pictures. None when so much changed that the scene itself changed.
    `region` (col0, row0, col1, row1 in blocks): only look for moves there (around "me" — much faster).

    For every changed block in `after`, look for a changed block in `before` with the same colour: the offset
    between them is a vote for "something moved by this much". The offset with the most votes is a real move."""
    mask = frames.changed(before, after)
    if mask.mean() > SCENE_CHANGE:
        return None
    if region is not None:
        c0, r0, c1, r1 = region
        inside = np.zeros_like(mask)
        inside[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True
        mask = mask & inside
    points = np.argwhere(mask)
    if len(points) == 0:
        return []
    if len(points) > MAX_POINTS:
        return None
    was = before[points[:, 0], points[:, 1]]
    now = after[points[:, 0], points[:, 1]]
    pairs_new, pairs_old = [], []
    for start in range(0, len(points), 400):  # in slices, so memory stays small
        block = now[start:start + 400]
        same = np.abs(block[:, None, 0] - was[None, :, 0]) < frames.CHANGE
        for channel in range(1, was.shape[1]):  # one colour channel at a time: no big 3-D array
            same &= np.abs(block[:, None, channel] - was[None, :, channel]) < frames.CHANGE
        i, j = np.nonzero(same)
        pairs_new.append(i + start)
        pairs_old.append(j)
    new_i, old_j = np.concatenate(pairs_new), np.concatenate(pairs_old)
    offsets = points[new_i] - points[old_j]
    moving = (offsets != 0).any(axis=1)
    new_i, old_j, offsets = new_i[moving], old_j[moving], offsets[moving]
    if len(offsets) == 0:
        return []
    found, votes = np.unique(offsets, axis=0, return_counts=True)
    order = np.argsort(-votes)
    ring = _ring_colours(mask, after)
    chosen: list[tuple[np.ndarray, np.ndarray]] = []  # (offset, new block positions)
    used = np.zeros(len(points), dtype=bool)
    rejected: list[np.ndarray] = []  # mirrors already judged to be the background
    floor = max(MIN_CELLS, votes[order[0]] // 10)  # far weaker peaks are chance colour matches
    for k in order[:24]:
        if votes[k] < floor:
            break
        offset = found[k]
        if any(np.abs(offset - o).max() <= 1 or np.abs(offset + o).max() <= 1 for o, _ in chosen):
            continue  # the same move again (blocks don't line up exactly), or its mirror
        if any(np.abs(offset - o).max() <= 1 for o in rejected):
            continue
        hit = (offsets == offset).all(axis=1)
        news = np.unique(new_i[hit])
        news = news[~used[news]]
        if len(news) < MIN_CELLS:
            continue
        mirror = (found == -offset).all(axis=1)
        if mirror.any():
            # A plain background makes "it moved from A to B" look like "the background moved from B to A".
            # The real mover is the one whose content looks unlike its unchanged surroundings.
            back = (offsets == -offset).all(axis=1)
            mirror_news = np.unique(new_i[back])
            if _distinct(now[mirror_news], ring) > _distinct(now[news], ring):
                rejected.append(offset)
                continue  # the mirror is the real move; it comes up in its own turn
        used[news] = True
        chosen.append((offset, points[news]))
    motions = []
    for offset, news in chosen:
        dr, dc = int(offset[0]), int(offset[1])
        for group in _groups(news):
            if len(group) < MIN_CELLS:
                continue
            olds = group - np.array([dr, dc])  # where those blocks were before, moved forward again
            rows = np.concatenate([group[:, 0], olds[:, 0] + dr])
            cols = np.concatenate([group[:, 1], olds[:, 1] + dc])
            box = (int(cols.min()) * cell, int(rows.min()) * cell, (int(cols.max()) + 1) * cell, (int(rows.max()) + 1) * cell)
            motions.append(Motion(dc * cell, dr * cell, box, len(group)))
    return motions


def _ring_colours(mask: np.ndarray, grid: np.ndarray, width: int = 2) -> np.ndarray:
    """Colours of unchanged blocks just around the changed ones (what the surroundings look like)."""
    grown = mask.copy()
    for _ in range(width):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    colours = grid[grown & ~mask]
    if len(colours) > 256:
        colours = colours[np.linspace(0, len(colours) - 1, 256).astype(int)]
    return colours


def _distinct(content: np.ndarray, ring: np.ndarray) -> float:
    """How unlike the surroundings some content is (mean distance to the closest surrounding colour)."""
    if len(content) == 0:
        return 0.0
    if len(ring) == 0:
        return 1.0
    return float(np.abs(content[:, None, :] - ring[None, :, :]).max(axis=2).min(axis=1).mean())


class SelfModel:
    """Learns which thing on screen is "me", how each key moves it, and when a key is blocked."""

    def __init__(self):
        self.cell: int | None = None
        self.box: tuple[int, int, int, int] | None = None
        self.template: np.ndarray | None = None  # what "me" looks like, in shrunk blocks
        self.effects: dict[str, list[tuple[int, int]]] = {}  # action → learned moves (pixels)
        # Presses made while "me" stood still show a key's own effect, without falling or sliding mixed in.
        self.rest_effects: dict[str, list[tuple[int, int]]] = {}
        self.rest_presses: dict[str, int] = {}
        self._last_moved: tuple[int, int] | None = None
        # The part of the last move the key did not explain (falling, sliding): it usually carries on next step.
        self.drift: tuple[float, float] = (0.0, 0.0)
        self.blocked_counts: dict[str, int] = {}
        self.candidates: list[_Candidate] = []
        self.steps = 0
        self._probe = 0
        self._shrunk: tuple | None = None
        self.camera = cam.CameraTracker()  # how far the background has slid (so scrolling is not "me" moving)
        self._cache: dict = {}  # learned effects only change in observe(): remember them between steps

    @property
    def known(self) -> bool:
        return self.box is not None

    # ---- what was learned
    def effect(self, action: str) -> tuple[float, float] | None:
        """The usual move of `action` (pixels), when it is a key that moves "me"; else None."""
        if ("effect", action) not in self._cache:
            self._cache[("effect", action)] = self._effect(action)
        return self._cache[("effect", action)]

    def _effect(self, action: str) -> tuple[float, float] | None:
        moves = self.rest_effects.get(action, [])
        if len(moves) < 2:
            if self.rest_presses.get(action, 0) >= 3 and not moves:
                return None  # pressed often while standing still and never moved "me": not a moving key
            moves = self.effects.get(action, [])
        if len(moves) < 2:
            return None
        arr = np.array(moves, dtype=np.float64)
        mean = arr.mean(axis=0)
        if np.linalg.norm(mean) == 0:
            return None
        direction = _unit(*mean)
        agree = sum(float(_unit(*m) @ direction) > CONSISTENT for m in moves)
        if agree < 0.6 * len(moves):
            return None  # this key does not move "me" the same way each time
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    @property
    def last_move(self) -> tuple[int, int] | None:
        """How "me" moved in the last step (pixels), if known."""
        return self._last_moved

    def moving_actions(self) -> list[str]:
        return [a for a in self.effects if self.effect(a) is not None]

    def step_size(self) -> tuple[float, float] | None:
        """The learned size of one step across (x) and down (y), in pixels."""
        if "step" not in self._cache:
            self._cache["step"] = self._step_size()
        return self._cache["step"]

    def _step_size(self) -> tuple[float, float] | None:
        xs, ys = [], []
        for action in self.moving_actions():
            dx, dy = self.effect(action)
            if abs(dx) >= abs(dy) and dx:
                xs.append(abs(dx))
            elif dy:
                ys.append(abs(dy))
        if not xs and not ys:
            return None
        sx = float(np.median(xs)) if xs else float(np.median(ys))
        sy = float(np.median(ys)) if ys else sx
        return sx, sy

    def probe_action(self, actions: list[str]) -> str | None:
        """The next key to try while "me" is still unknown: take turns through the keys."""
        if not actions:
            return None
        action = actions[self._probe % len(actions)]
        self._probe += 1
        return action

    # ---- learning from one step
    def observe(self, before: np.ndarray, action: str | None, after: np.ndarray) -> SelfView:
        """Learn from one (picture, key, next picture) step and say where "me" is now."""
        self.steps += 1
        self._cache = {}
        view = self._observe(before, action, after)
        view.camera = self.camera.last[:2]  # (0, 0) when the camera stood still or the estimate was shaky
        if view.scene_change:
            self.camera.reset()  # a new room or level: its world starts here
        self._cache = {}  # what was learned this step changes the effects
        return view

    def _observe(self, before: np.ndarray, action: str | None, after: np.ndarray) -> SelfView:
        if self.cell is None:
            self.cell = frames.cell_size(np.asarray(after).shape)
        if self._shrunk is not None and self._shrunk[0] is before:
            a = self._shrunk[1]  # the last step's "after" is this step's "before"
        else:
            a = frames.shrink(before, self.cell)
        b = frames.shrink(after, self.cell)
        self._shrunk = (after, b)
        # A following camera slides the whole background each step: take that away first, so only things that
        # really moved in the world are left (and a pan is not mistaken for a new scene).
        dc, dr, sureness = cam.estimate_grid_shift(a, b)
        slid = self.camera.update(dc * self.cell, dr * self.cell, sureness)
        if slid != (0, 0):
            a = cam.align(a, b, slid[0] // self.cell, slid[1] // self.cell)
            if self.box is not None:
                self.box = shift(self.box, *slid)  # where I would be if I had stayed put in the world
            for candidate in self.candidates:
                candidate.box = shift(candidate.box, *slid)  # same for things I am still guessing might be me
        motions = find_motions(a, b, self.cell, self._search_region())
        if not self.known:
            if motions:
                self._learn_candidates(action, motions)
            if self.known:
                self.template = frames.crop(b, self._blocks(self.box))
                return SelfView(self.box, 0.6)
            return SelfView(None, 0.0)
        if motions is None:
            return self._reacquire(b, scene_change=True)
        mine = self._my_motion(motions)
        from_rest = self._last_moved == (0, 0)
        if action and from_rest:
            self.rest_presses[action] = self.rest_presses.get(action, 0) + 1
        self._last_moved = (0, 0) if mine is None else (mine.dx, mine.dy)
        if mine is None:
            self.drift = (0.0, 0.0)
        else:
            own = self.effect(action) if action else None
            own = own or (0.0, 0.0)
            self.drift = (mine.dx - own[0], mine.dy - own[1])
        if mine is not None:
            far = self._far(mine.dx, mine.dy)
            self.box = shift(self.box, mine.dx, mine.dy)
            self._refine(b)
            if far:
                return SelfView(self.box, 0.5, scene_change=True)
            if action:
                self.effects.setdefault(action, []).append((mine.dx, mine.dy))
                self.effects[action] = self.effects[action][-40:]
                if from_rest:
                    self.rest_effects.setdefault(action, []).append((mine.dx, mine.dy))
                    self.rest_effects[action] = self.rest_effects[action][-40:]
            return SelfView(self.box, 0.9, moved=(mine.dx, mine.dy), blocked=self._is_blocked(action, mine.dx, mine.dy))
        # nothing of "me" moved: am I still where I was?
        _, gap = frames.match(b, self.template, near=self._blocks(self.box)[:2], reach=1)
        if gap > GOOD_MATCH:
            return self._reacquire(b, scene_change=False)
        blocked = self._is_blocked(action, 0, 0)
        if blocked:
            self.blocked_counts[action] = self.blocked_counts.get(action, 0) + 1
        return SelfView(self.box, 0.9, moved=(0, 0), blocked=blocked)

    def _learn_candidates(self, action: str | None, motions: list[Motion]) -> None:
        for motion in motions:
            match = max(self.candidates, key=lambda c: overlap(c.box, motion.old_box), default=None)
            if match is None or overlap(match.box, motion.old_box) == 0:
                match = _Candidate(motion.old_box)
                self.candidates.append(match)
            match.box = motion.box
            match.last_seen = self.steps
            if action:
                match.moves.append((action, motion.dx, motion.dy))
        self.candidates = [c for c in self.candidates if self.steps - c.last_seen <= FORGET_AFTER]
        best, best_score = None, 0
        for candidate in self.candidates:
            score = self._key_following(candidate.moves)
            if score > best_score:
                best, best_score = candidate, score
        if best is not None:
            self.box = best.box
            for action, dx, dy in best.moves:
                self.effects.setdefault(action, []).append((dx, dy))
            self.candidates = []

    @staticmethod
    def _key_following(moves: list[tuple[str, int, int]]) -> int:
        """How many moves followed their key, if "me"-like (enough of them, in enough directions); else 0."""
        by_action: dict[str, list[np.ndarray]] = {}
        for action, dx, dy in moves:
            by_action.setdefault(action, []).append(_unit(dx, dy))
        directions, following = [], 0
        for units in by_action.values():
            mean = _unit(*np.mean(units, axis=0))
            agree = sum(float(u @ mean) > CONSISTENT for u in units)
            if len(units) >= 1 and agree == len(units):
                following += agree
                if all(float(mean @ d) < CONSISTENT for d in directions):
                    directions.append(mean)
        if following >= NEEDED_MOVES and len(directions) >= NEEDED_DIRECTIONS:
            return following
        return 0

    def _my_motion(self, motions: list[Motion]) -> Motion | None:
        best = max(motions, key=lambda m: overlap(m.old_box, self.box), default=None)
        if best is None or overlap(best.old_box, self.box) == 0:
            return None
        return best

    def _far(self, dx: int, dy: int) -> bool:
        step = self.step_size()
        if step is None:
            return False
        return abs(dx) > TELEPORT_STEPS * step[0] or abs(dy) > TELEPORT_STEPS * step[1]

    def _is_blocked(self, action: str | None, dx: float, dy: float) -> bool:
        expected = self.effect(action) if action else None
        if expected is None:
            return False
        size = float(np.hypot(*expected))
        along = float(np.array([dx, dy]) @ _unit(*expected))
        return along < BLOCKED_SHARE * size

    def _search_region(self) -> tuple[int, int, int, int] | None:
        """Where to look for my next move: around me, a few steps (or a few of my sizes) each way."""
        if self.box is None:
            return None
        step = self.step_size()
        reach_x = int(3 * max(step[0] if step else 0, self.box[2] - self.box[0]))
        reach_y = int(3 * max(step[1] if step else 0, self.box[3] - self.box[1]))
        return self._blocks((self.box[0] - reach_x, self.box[1] - reach_y, self.box[2] + reach_x, self.box[3] + reach_y))

    def _blocks(self, box) -> tuple[int, int, int, int]:
        c = self.cell
        return int(box[0] // c), int(box[1] // c), int(box[2] // c), int(box[3] // c)

    def _refine(self, grid: np.ndarray) -> None:
        """Nudge the box onto the best fit of "me" nearby (stops small errors adding up)."""
        if self.template is None:
            return
        c0, r0, c1, r1 = self._blocks(self.box)
        (c, r), gap = frames.match(grid, self.template, near=(c0, r0), reach=1)
        if gap <= GOOD_MATCH and (c, r) != (c0, r0):
            self.box = shift(self.box, (c - c0) * self.cell, (r - r0) * self.cell)

    def locate(self, frame: np.ndarray) -> SelfView:
        """Find "me" in a fresh picture by my looks alone (a new episode, level or game session)."""
        if self.template is None or self.cell is None:
            return SelfView(None, 0.0)
        self._last_moved = None
        self.camera.reset()
        grid = frames.shrink(frame, self.cell)
        (c, r), gap = frames.match(grid, self.template)
        if gap > GOOD_MATCH:
            return SelfView(None, 0.0)
        th, tw = self.template.shape[:2]
        self.box = (c * self.cell, r * self.cell, (c + tw) * self.cell, (r + th) * self.cell)
        return SelfView(self.box, 0.7)

    def _reacquire(self, grid: np.ndarray, scene_change: bool) -> SelfView:
        """Look for "me" anywhere (after a new scene, or when lost)."""
        self._last_moved = None
        (c, r), gap = frames.match(grid, self.template)
        if gap > GOOD_MATCH:
            return SelfView(self.box, 0.1, scene_change=scene_change)
        c0, r0, _, _ = self._blocks(self.box)
        dx, dy = (c - c0) * self.cell, (r - r0) * self.cell
        far = self._far(dx, dy)
        self.box = shift(self.box, dx, dy)
        return SelfView(self.box, 0.7, scene_change=scene_change or far)

    # ---- saving
    def to_dict(self) -> dict:
        return {"cell": self.cell, "box": list(self.box) if self.box else None,
                "template": self.template.round(1).tolist() if self.template is not None else None,
                "effects": {a: [list(m) for m in moves] for a, moves in self.effects.items()},
                "rest_effects": {a: [list(m) for m in moves] for a, moves in self.rest_effects.items()},
                "rest_presses": dict(self.rest_presses), "blocked_counts": dict(self.blocked_counts)}

    @classmethod
    def from_dict(cls, data: dict) -> "SelfModel":
        model = cls()
        model.cell = data.get("cell")
        box = data.get("box")
        model.box = tuple(box) if box else None
        template = data.get("template")
        model.template = np.array(template, dtype=np.float32) if template is not None else None
        model.effects = {a: [tuple(m) for m in moves] for a, moves in data.get("effects", {}).items()}
        model.rest_effects = {a: [tuple(m) for m in moves] for a, moves in data.get("rest_effects", {}).items()}
        model.rest_presses = dict(data.get("rest_presses", {}))
        model.blocked_counts = dict(data.get("blocked_counts", {}))
        return model
