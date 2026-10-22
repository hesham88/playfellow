"""The learned map: places the buddy has been, what each key did there, and what things look like.
/ الخريطة المتعلَّمة: الأماكن التي زارها الرفيق، وماذا فعل كل مفتاح هناك، وكيف تبدو الأشياء.

Like drawing a treasure map while you walk: every square you stand on becomes a place; every key you press there
becomes an arrow ("RIGHT from here → that square", or "UP from here → bump, a wall"). Squares where the score went up
are marked as treasure.

It also remembers what things LOOK like, learned from what happened:
  - the square I bumped into looked like THIS → walls look like this;
  - the square I walked onto looked like THIS → floor looks like this;
  - stepping on THIS made the score change → treasure looks like this;
  - stepping on THIS ended the game well (or badly) → goals (or dangers) look like this.
So on a level it has never seen, it can read the whole screen and guess the map before taking a step
(then check each guess by moving). Nothing about any game's colours or layout is written in the code.
"""
from __future__ import annotations

import functools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nav import frames
from nav.automover import AutoMover
from nav.dynamics import MAX_HISTORY, Ballistics, Dynamics
from nav.graph4 import EdgeGrid
from nav.self_model import SelfModel, SelfView

PATCH = 6  # a square's look is shrunk to PATCH × PATCH colours
LOOK_MATCH = 20.0  # mean colour gap under which two looks count as the same kind of thing
SAME_LOOK = 2.0  # closer than this: the same sample again, don't store it twice
KEEP_PER_LABEL = 64
SAME_SCENE = 8.0  # thumbnails closer than this (mean grey gap) are the same scene
LABELS = ("wall", "floor", "reward", "goal", "hazard")


def hud_signature(hud: dict | None) -> str:
    return json.dumps(hud or {}, sort_keys=True, default=str)


def look(frame: np.ndarray, center: tuple[float, float], step: tuple[float, float]) -> np.ndarray | None:
    """What one square (one step wide, centred on `center`) looks like, shrunk to PATCH × PATCH colours."""
    frame = np.asarray(frame)
    half_w, half_h = step[0] / 2, step[1] / 2
    x0, y0 = int(round(center[0] - half_w)), int(round(center[1] - half_h))
    x1, y1 = int(round(center[0] + half_w)), int(round(center[1] + half_h))
    if x0 < 0 or y0 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0] or x1 - x0 < PATCH or y1 - y0 < PATCH:
        return None
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    starts_r, starts_c, counts = _bins(patch.shape[0], patch.shape[1])
    sums = np.add.reduceat(np.add.reduceat(patch, starts_r, axis=0), starts_c, axis=1)
    return (sums / counts).reshape(-1)


def looks_many(frame: np.ndarray, centers: list[tuple[float, float]], step: tuple[float, float]) -> list:
    """look() for many squares at once (the same numbers), using one running total of the picture."""
    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    # exact whole-number running totals (a 4K picture of 255s still fits in 32 bits)
    total = np.zeros((height + 1, width + 1, frame.shape[2]), dtype=np.uint32)
    total[1:, 1:] = frame.cumsum(axis=0, dtype=np.uint32).cumsum(axis=1, dtype=np.uint32)
    out: list = [None] * len(centers)
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for i, (cx, cy) in enumerate(centers):
        x0, y0 = int(round(cx - step[0] / 2)), int(round(cy - step[1] / 2))
        x1, y1 = int(round(cx + step[0] / 2)), int(round(cy + step[1] / 2))
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 - x0 < PATCH or y1 - y0 < PATCH:
            continue
        groups.setdefault((y1 - y0, x1 - x0), []).append((i, x0, y0))
    for (h, w), members in groups.items():  # squares of one size share their block layout
        rows = np.linspace(0, h, PATCH + 1).astype(int)
        cols = np.linspace(0, w, PATCH + 1).astype(int)
        index = np.array([m[0] for m in members])
        r = np.array([m[2] for m in members])[:, None] + rows[None, :]
        c = np.array([m[1] for m in members])[:, None] + cols[None, :]
        corner = lambda rr, cc: total[rr, cc].astype(np.int64)  # noqa: E731
        sums = (corner(r[:, 1:, None], c[:, None, 1:]) - corner(r[:, :-1, None], c[:, None, 1:])
                - corner(r[:, 1:, None], c[:, None, :-1]) + corner(r[:, :-1, None], c[:, None, :-1]))
        counts = np.outer(np.diff(rows), np.diff(cols))[None, :, :, None]
        vecs = (sums / counts).astype(np.float32).reshape(len(members), -1)
        for k, i in enumerate(index):
            out[i] = vecs[k]
    return out


@functools.lru_cache(maxsize=64)
def _bins(height: int, width: int):
    """Where each of the PATCH × PATCH blocks of a height × width patch starts, and how many pixels it holds."""
    rows = np.linspace(0, height, PATCH + 1).astype(int)
    cols = np.linspace(0, width, PATCH + 1).astype(int)
    return rows[:-1], cols[:-1], np.outer(np.diff(rows), np.diff(cols))[:, :, None]


class Looks:
    """The appearance memory: example looks for each kind of square, learned from experience, with a vote count
    (how often that look was seen as that kind)."""

    def __init__(self):
        self.samples: dict[str, list[np.ndarray]] = {label: [] for label in LABELS}
        self.votes: dict[str, list[int]] = {label: [] for label in LABELS}
        self.version = 0  # goes up with every new example (so old screen readings are redone)

    def add(self, label: str, vec: np.ndarray | None) -> None:
        if vec is None:
            return
        kept, votes = self.samples[label], self.votes[label]
        for i, old in enumerate(kept):
            if float(np.abs(old - vec).mean()) < SAME_LOOK:
                votes[i] += 1
                return
        kept.append(vec)
        votes.append(1)
        if len(kept) > KEEP_PER_LABEL:
            weakest = int(np.argmin(votes))
            del kept[weakest], votes[weakest]
        self.version += 1

    def labels(self, vec: np.ndarray | None) -> set[str]:
        """The kinds of square this look matches (empty = never seen anything like it).

        When a look matched both "wall" and "floor" (a jump that went nowhere because I was falling also feels
        like a bump), the kind with more votes wins."""
        if vec is None:
            return set()
        best: dict[str, tuple[float, int]] = {}
        for label, stacked in self._stacked().items():
            if len(stacked):
                gaps = np.abs(stacked - vec).mean(axis=1)
                i = int(np.argmin(gaps))
                best[label] = (float(gaps[i]), self.votes[label][i])
        if not best:
            return set()
        return self._decide(best)

    def labels_many(self, vecs: list) -> list[set[str]]:
        """labels() for many looks at once (the same answers, one array sum instead of a loop)."""
        known = [i for i, v in enumerate(vecs) if v is not None]
        out: list[set[str]] = [set() for _ in vecs]
        stacked = {label: s for label, s in self._stacked().items() if len(s)}
        if not known or not stacked:
            return out
        vv = np.stack([vecs[i] for i in known])
        best = {}
        for label, s in stacked.items():
            gaps = np.abs(vv[:, None, :] - s[None, :, :]).mean(axis=2)
            pick = gaps.argmin(axis=1)
            best[label] = (gaps[np.arange(len(known)), pick], np.array(self.votes[label])[pick])
        for row, i in enumerate(known):
            per = {label: (float(g[row]), int(v[row])) for label, (g, v) in best.items()}
            out[i] = self._decide(per)
        return out

    def _decide(self, best: dict[str, tuple[float, int]]) -> set[str]:
        nearest = min(gap for gap, _ in best.values())
        if nearest > LOOK_MATCH:
            return set()
        found = {label for label, (gap, _) in best.items() if gap <= min(nearest + SAME_LOOK * 2, LOOK_MATCH)}
        if {"wall", "floor"} <= found:
            found.discard("floor" if best["wall"][1] > best["floor"][1] else "wall")
        return found

    def _stacked(self) -> dict[str, np.ndarray]:
        """The examples of each kind as one array (rebuilt only when an example was added or voted)."""
        stamp = (self.version, tuple(len(k) for k in self.samples.values()))
        if getattr(self, "_stack_stamp", None) != stamp:
            self._stack = {label: np.stack(kept) if kept else np.empty((0, 0), dtype=np.float32)
                           for label, kept in self.samples.items()}
            self._stack_stamp = stamp
        return self._stack

    @property
    def empty(self) -> bool:
        return not any(self.samples.values())

    def to_dict(self) -> dict:
        return {label: {"looks": [v.round(1).tolist() for v in kept], "votes": list(self.votes[label])}
                for label, kept in self.samples.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Looks":
        looks = cls()
        for label, kept in data.items():
            if label in looks.samples:
                looks.samples[label] = [np.array(v, dtype=np.float32) for v in kept["looks"]]
                looks.votes[label] = [int(n) for n in kept["votes"]]
        return looks


@dataclass
class Edge:
    """What one key did from one place, counted over every try."""
    tries: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)  # place key or "blocked" → count
    events: int = 0  # HUD changes right after this move
    good_end: int = 0  # the game ended well right after it
    bad_end: int = 0  # the game ended badly right after it
    blocked_hud: str | None = None  # the HUD when it was last blocked (a change may unblock it: a key opens a door)

    def best(self) -> tuple[str | None, float]:
        """The most common outcome and how often it happened (0-1)."""
        if not self.outcomes:
            return None, 0.0
        where, count = max(self.outcomes.items(), key=lambda item: item[1])
        return where, count / max(sum(self.outcomes.values()), 1)


class ActionBook(dict):
    """One place's arrows (action -> Edge). Every time an arrow is added, replaced or removed it tells the road grid, so
    the grid never has to look through the whole map to find out what changed."""

    def __init__(self, owner: "NavMap", place: str, items=()):
        super().__init__()
        self._owner, self._place = owner, place
        for action, edge in dict(items).items():
            self[action] = edge

    def __setitem__(self, action, edge):
        super().__setitem__(action, edge)
        self._owner.grid.mark(self._place, action)

    def __delitem__(self, action):
        super().__delitem__(action)
        self._owner.grid.mark(self._place, action)

    def pop(self, action, *default):
        had = action in self
        value = super().pop(action, *default)
        if had:
            self._owner.grid.mark(self._place, action)
        return value

    def setdefault(self, action, default=None):
        if action not in self:
            self[action] = default
        return self[action]

    def update(self, *args, **kwargs):
        for action, edge in dict(*args, **kwargs).items():
            self[action] = edge

    def clear(self):
        for action in list(self):
            del self[action]

    def popitem(self):
        action, edge = super().popitem()
        self._owner.grid.mark(self._place, action)
        return action, edge


class EdgeBook(dict):
    """All arrows of the map: place -> ActionBook. A plain dict of dicts can still be assigned into it; it is wrapped."""

    def __init__(self, owner: "NavMap"):
        super().__init__()
        self._owner = owner

    def __setitem__(self, place, actions):
        for action in self.get(place, {}):  # whatever was there is gone (unless it is put back below)
            self._owner.grid.mark(place, action)
        book = actions if isinstance(actions, ActionBook) and actions._owner is self._owner and actions._place == place \
            else ActionBook(self._owner, place, actions)
        super().__setitem__(place, book)
        for action in book:
            self._owner.grid.mark(place, action)

    def __delitem__(self, place):
        for action in self.get(place, {}):
            self._owner.grid.mark(place, action)
        super().__delitem__(place)

    def pop(self, place, *default):
        for action in self.get(place, {}):
            self._owner.grid.mark(place, action)
        return super().pop(place, *default)

    def setdefault(self, place, default=None):
        if place not in self:
            self[place] = {} if default is None else default
        return self[place]

    def update(self, *args, **kwargs):
        for place, actions in dict(*args, **kwargs).items():
            self[place] = actions

    def clear(self):
        for place in list(self):
            del self[place]


@dataclass
class Place:
    visits: int = 0
    rewards: int = 0  # HUD changes when arriving here
    goal: int = 0
    hazard: int = 0


class NavMap:
    """The buddy's map of one game: places, arrows between them, rewards, and the look memory."""

    def __init__(self):
        self.me = SelfModel()
        self.looks = Looks()
        self.scenes: list[np.ndarray] = []  # thumbnails of the scenes (rooms, levels) seen so far
        self.places: dict[str, Place] = {}
        self.grid = EdgeGrid(self.parse)  # the learned roads as numbers on a grid (nav/graph4.py), kept right as edges change
        self._edges = EdgeBook(self)  # place → action → edge
        self.phase: tuple[float, float] | None = None  # where squares line up on screen (learned, not told)
        self.scene: int | None = None
        self.here: str | None = None
        self.hud: dict = {}
        self.collected: set[str] = set()  # reward places already taken in this episode
        self._end_watch_state: tuple[str, str, str] | None = None  # (place, key, square I reached) of a step that may have ended the game
        self._end_steps = 0  # and how many steps of that
        self._end_keys: set[str] = set()  # the different keys pressed at the frozen picture so far
        self._restart_pending = False  # an end was just noticed: the next picture that moves is a new game
        self.ended: list[tuple[str, str]] = []  # (place, key) arrows just marked "ends the game"; the loop takes them with take_ended()
        self.episodes = 0
        self._predicted: tuple | None = None  # the last screen read (the planner asks several times per step)
        self.dynamics = Dynamics()  # how my moves work (learned physics; carries over to new levels)
        self.auto = AutoMover()  # do I keep moving by myself (a key only steers)? learned from watching
        self.ballistics = Ballistics()  # the same physics as numbers (speed, gravity, jump kick), from real-time play
        self.moves: tuple[tuple[int, int], ...] = ()  # my last few moves, in squares
        self.exits: dict[int, set[str]] = {}  # scene → screen sides I have already left it by
        self._camera_then: tuple[int, int] | None = None  # while reading the picture from BEFORE the step

    @property
    def edges(self) -> EdgeBook:
        return self._edges

    @edges.setter
    def edges(self, value) -> None:
        self.grid.clear()
        self._edges = EdgeBook(self)
        for place, actions in dict(value).items():
            self._edges[place] = actions

    END_STEPS = 3  # this many steps with the picture not moving at all, after a step that moved me AND changed the HUD,
    END_KEYS = 2  # pressed with at least this many DIFFERENT keys, mean the game ended. One key pressed over and over at a
    # frozen picture is just a wall: a coin followed by three bumps must not look like a win. The planner rotates to
    # untried keys at a frozen picture (frozen_key) so the second key does get pressed.

    def frozen_key(self, actions: list[str]) -> str | None:
        """While a picture stays frozen after a step that may have ended the game: a key not pressed yet at this frozen
        picture, so the 'different keys' test can be answered. None when not watching or every key has been tried."""
        if self._end_watch_state is None or self._end_steps < 1:
            return None
        return next((a for a in actions if a not in self._end_keys), None)

    def _watch_for_the_end(self, before: np.ndarray, after: np.ndarray, action: str, here: str, new: str,
                           reward_step: bool) -> None:
        """Learn-only guess that a step ENDED the game (a win nobody told us about). The signs: I moved onto a new square,
        the HUD changed, and from then on the picture does not change AT ALL however many different keys are pressed.
        A menu or a pause screen does not fit (nothing moved me onto a square before it), and a picture that changes even
        once cancels the guess. A wrong guess costs only a square wrongly marked as 'ends the game'. Never stops anything."""
        frozen = self._same_picture(before, after)
        if self._end_watch_state is not None:
            if not frozen:
                self._end_watch_state, self._end_steps, self._end_keys = None, 0, set()
            else:
                self._end_steps += 1
                self._end_keys.add(action)
                if self._end_steps >= self.END_STEPS and len(self._end_keys) >= self.END_KEYS:
                    self._mark_end(*self._end_watch_state)
                    self._end_watch_state, self._end_steps, self._end_keys = None, 0, set()
            return
        if reward_step and not frozen and new != here:
            # I moved (or vanished: the door square may hide me) and the HUD changed. Where did the key take me?
            target = new
            if target is None:
                square = self.target_square(here, action)
                target = None if square is None else self.key(square)
            if target is not None and target != here:
                self._end_watch_state, self._end_steps, self._end_keys = (here, action, target), 0, set()

    @staticmethod
    def _same_picture(before: np.ndarray, after: np.ndarray) -> bool:
        return before.shape == after.shape and bool(np.array_equal(before, after))

    def _mark_end(self, place: str, action: str, reached: str) -> None:
        edge = self._edges.get(place, {}).get(action)
        if edge is None:  # the step that ended it left no arrow (I could not see myself after it): make one
            edge = self._edges.setdefault(place, {}).setdefault(action, Edge(tries=1, outcomes={reached: 1}))
        if edge.good_end:
            return
        edge.good_end += 1
        place_reached = self.places.setdefault(reached, Place())
        place_reached.goal += 1
        place_reached.rewards = max(0, place_reached.rewards - 1)  # it was not a pickup after all: the game ended there
        self.grid.mark(place, action)
        self.ended.append((place, action))
        self._restart_pending = True

    def take_ended(self) -> list[tuple[str, str]]:
        """The arrows marked 'ends the game' since the last call (the loop logs them as one event each)."""
        out, self.ended = self.ended, []
        return out

    def learned_road(self, place: str, action: str, edge: Edge):
        """The road this learned arrow gives: (to, cost, how sure, ends the game) or None if it gives none.

        The ONE place that decides this, so the planner's list of roads and the road grid can never disagree."""
        if edge.bad_end:
            return None  # it ended the game badly once: never again
        if edge.good_end:
            where = next((w for w in edge.outcomes if w != "blocked"), None)
            return (where, 1.0, 1.0, True) if where else None
        where, share = edge.best()
        if where in (None, "blocked", place) or share < 0.5:
            return None
        return where, 1.0 / share, share, False

    def sync_grid(self) -> None:
        """Bring the road grid up to date with the arrows that changed since the last time."""
        def road_of(place: str, action: str):
            edge = self._edges.get(place, {}).get(action)
            return None if edge is None else self.learned_road(place, action, edge)

        self.grid.sync(road_of)

    # ---- squares and places
    def step(self) -> tuple[float, float] | None:
        return self.me.step_size()

    @property
    def camera(self) -> tuple[int, int]:
        """How far the background has slid on screen since this scene began (0, 0) when the camera never moves."""
        return self._camera_then or self.me.camera.offset

    def square(self, center: tuple[float, float]) -> tuple[int, int] | None:
        """The square at a screen point. Squares are addresses in the WORLD: a scrolling screen keeps them."""
        step = self.step()
        if step is None or center is None:
            return None
        center = (center[0] - self.camera[0], center[1] - self.camera[1])
        if self.phase is None:
            self.phase = (center[0] % step[0], center[1] % step[1])
        return round((center[0] - self.phase[0]) / step[0]), round((center[1] - self.phase[1]) / step[1])

    def center_of(self, square: tuple[int, int]) -> tuple[float, float]:
        """Where a square is on the screen right now."""
        step = self.step()
        return (self.phase[0] + square[0] * step[0] + self.camera[0],
                self.phase[1] + square[1] * step[1] + self.camera[1])

    def key(self, square: tuple[int, int], scene: int | None = None) -> str:
        return f"{self.scene if scene is None else scene}:{square[0]}:{square[1]}"

    @staticmethod
    def parse(key: str) -> tuple[int, tuple[int, int]]:
        scene, x, y = key.split(":")
        return int(scene), (int(x), int(y))

    def place_of(self, view: SelfView) -> str | None:
        if view.box is None or self.scene is None:
            return None
        square = self.square(view.center)
        return None if square is None else self.key(square)

    def recognise(self, frame: np.ndarray, came_from: np.ndarray | None = None) -> int:
        """Which scene is this? A known one if the thumbnail matches, else a new one.

        Right after the scene changed (`came_from` = the picture before), a known scene must be much closer than the
        jump we just saw: two rooms can look alike, but coming back to a room looks almost exactly the same."""
        thumb = frames.thumbnail(frame)
        gaps = [frames.thumb_distance(thumb, known) for known in self.scenes]
        limit = SAME_SCENE
        if came_from is not None:
            limit = min(limit, frames.thumb_distance(thumb, frames.thumbnail(came_from)) / 2)
        if gaps and min(gaps) <= limit:
            return int(np.argmin(gaps))
        self.scenes.append(thumb)
        return len(self.scenes) - 1

    # ---- living through steps
    def begin(self, frame: np.ndarray, hud: dict | None = None) -> None:
        """A new episode (or a new level): find myself and the scene again; nothing collected yet."""
        self.scene = self.recognise(frame)
        self.collected = set()
        self._end_watch_state, self._end_steps, self._end_keys = None, 0, set()
        self._restart_pending = False
        self.hud = dict(hud or {})
        self.episodes += 1
        self.moves = ()
        self.auto.new_episode()
        self.ballistics.reset_track()
        view = self.me.locate(frame) if self.me.template is not None else SelfView(None, 0.0)
        self.here = self.place_of(view)
        if self.here:
            self.places.setdefault(self.here, Place()).visits += 1

    def update(self, before: np.ndarray, action: str | None, after: np.ndarray, hud: dict | None = None,
               done: bool = False, dt: float | None = None) -> SelfView:
        """Learn from one step: where I went, what the key did, what the HUD said, what things look like.
        `dt`: seconds of game time the step took (real-time play only): then my speed and arcs are learned too."""
        hud = dict(hud or {})
        if self._restart_pending and not self._same_picture(before, after):
            # The game had ended (the picture stood still), and now it moves again: a new game. Start a new episode.
            self._restart_pending = False
            self.begin(after, hud)
            return self.me.locate(after) if self.me.template is not None else SelfView(None, 0.0)
        events = hud_signature(hud) != hud_signature(self.hud)
        old_center = None if self.me.box is None else ((self.me.box[0] + self.me.box[2]) / 2, (self.me.box[1] + self.me.box[3]) / 2)
        was_known = self.me.known
        camera_then = self.camera
        view = self.me.observe(before, action, after)
        self._learn_ballistics(view, action, dt)
        self._learn_auto(view, action)
        if self.scene is None:
            self.scene = self.recognise(before)
        if view.scene_change:
            if old_center is not None and self.scene is not None:
                self.exits.setdefault(self.scene, set()).add(_nearest_side(old_center, np.asarray(before).shape))
            self.scene = self.recognise(after, came_from=before)
        if not was_known and self.me.known:
            self.here = None  # just found myself: start the map from here
        new = self.place_of(view)
        good_end = done and events
        bad_end = done and not events
        self._camera_then = camera_then  # `before` was drawn with the old camera: read it that way
        try:
            self._learn_physics(before, action, view, new)
        finally:
            self._camera_then = None
        if self.here is not None and action and new is not None:
            edge = self.edges.setdefault(self.here, {}).setdefault(action, Edge())
            edge.tries += 1
            outcome = "blocked" if view.blocked else new
            edge.outcomes[outcome] = edge.outcomes.get(outcome, 0) + 1
            if view.blocked:
                edge.blocked_hud = hud_signature(self.hud)
            edge.events += int(events and not done)
            edge.good_end += int(good_end)
            edge.bad_end += int(bad_end)
            self.grid.mark(self.here, action)  # this arrow changed: its road may have too
            self._camera_then = camera_then
            try:
                self._learn_look(before, action, old_center, view, events, good_end, bad_end)
            finally:
                self._camera_then = None
        if action and self.here is not None:
            self._watch_for_the_end(before, after, action, self.here, new, events and not done)
        if new is not None:
            place = self.places.setdefault(new, Place())
            if new != self.here:
                place.visits += 1
            if events and not done and new != self.here:
                place.rewards += 1
                self.collected.add(new)
            place.goal += int(good_end and new != self.here)
            place.hazard += int(bad_end)
        self.here = new
        self.hud = hud
        return view

    def solid_around(self, frame: np.ndarray, square: tuple[int, int]) -> tuple[bool, bool, bool, bool]:
        """Which of the four squares around (up, right, down, left) look like walls."""
        guesses = self.predict(frame)
        x, y = square
        return tuple("wall" in guesses.get(sq, set()) for sq in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)))

    def _learn_auto(self, view, action) -> None:
        """Did I keep moving on my own, and which way does each key steer me? (counted by nav.automover)"""
        if view.scene_change or view.moved is None:
            self.auto.lost()
            return
        self.auto.observe(action, (int(view.moved[0]), int(view.moved[1])))

    BALLISTICS_FIT_EVERY = 25

    def _learn_ballistics(self, view, action, dt) -> None:
        """Real-time play: watch where I am in the WORLD (screen position minus the camera slide) every tick."""
        if dt is None or dt <= 0:
            return
        center = view.center
        if view.scene_change or center is None:
            self.ballistics.reset_track()
            return
        cx, cy = self.me.camera.to_world(center)
        self.ballistics.observe(cx, cy, action, dt)
        if len(self.ballistics._rows) % self.BALLISTICS_FIT_EVERY == 0:
            self.ballistics.fit()

    def _learn_physics(self, before, action, view, new) -> None:
        """One more example of how my moves work: (recent moves, walls around, key) → the move that happened."""
        step = self.step()
        if view.scene_change or new is None or self.here is None or step is None or view.moved is None:
            self.moves = ()
            return
        scene, square = self.parse(self.here)
        if scene != self.scene:
            self.moves = ()
            return
        move = tuple(int(np.clip(round(d / s), -3, 3)) for d, s in zip(view.moved, step))
        if action:
            self.dynamics.add(self.moves, self.solid_around(before, square), action, move)
        self.moves = (self.moves + (move,))[-MAX_HISTORY:]

    def _learn_look(self, before, action, old_center, view, events, good_end, bad_end) -> None:
        effect = self.me.effect(action)
        step = self.step()
        if step is None or old_center is None or view.scene_change:
            return
        if view.moved and view.moved != (0, 0):
            target = (old_center[0] + view.moved[0], old_center[1] + view.moved[1])  # where I really went
        elif effect is not None:
            target = (old_center[0] + effect[0], old_center[1] + effect[1])  # where the key should have taken me
        else:
            return
        vec = look(before, target, step)
        if bad_end:
            self.looks.add("hazard", vec)
        elif good_end:
            self.looks.add("goal", vec)
        elif view.blocked:
            self.looks.add("wall", vec)
        elif view.moved and view.moved != (0, 0):
            self.looks.add("reward" if events else "floor", vec)

    # ---- reading the map
    def neighbours(self, key: str) -> dict[str, str]:
        """Known arrows out of a place: action → where it usually leads (blocked moves left out)."""
        out = {}
        for action, edge in self.edges.get(key, {}).items():
            where, share = edge.best()
            if where and where != "blocked" and where != key and share >= 0.5 and not edge.bad_end:
                out[action] = where
        return out

    def untried(self, key: str, actions: list[str]) -> list[str]:
        tried = self.edges.get(key, {})
        return [a for a in actions if a not in tried]

    def retryable(self, key: str) -> list[str]:
        """Blocked arrows worth one more try: the HUD has changed since they were blocked (keys open doors)."""
        now = hud_signature(self.hud)
        return [a for a, e in self.edges.get(key, {}).items()
                if e.best()[0] == "blocked" and e.blocked_hud is not None and e.blocked_hud != now]

    def target_square(self, key: str, action: str) -> tuple[int, int] | None:
        """The square `action` should lead to from place `key`, by the key's learned move."""
        effect, step = self.me.effect(action), self.step()
        if effect is None or step is None:
            return None
        _, (qx, qy) = self.parse(key)
        return qx + round(effect[0] / step[0]), qy + round(effect[1] / step[1])

    def path(self, start: str, wanted, safe=None) -> list[str] | None:
        """The fewest known arrows from `start` to the nearest place where wanted(place) is true
        (skipping arrows where safe(place, action) is False)."""
        if wanted(start):
            return []
        seen, frontier = {start: []}, [start]
        while frontier:
            nxt = []
            for here in frontier:
                for action, there in self.neighbours(here).items():
                    if safe is not None and not safe(here, action):
                        continue
                    if there not in seen:
                        seen[there] = seen[here] + [action]
                        if wanted(there):
                            return seen[there]
                        nxt.append(there)
            frontier = nxt
        return None

    def explore_action(self, frame: np.ndarray, actions: list[str]) -> str | None:
        """Go-Explore: try an untried arrow here (squares that look like walls last); else walk known arrows to the
        nearest place that still has untried arrows; else retry a blocked arrow the HUD may have opened."""
        if self.here is None:
            return None
        guesses = self.predict(frame)

        def safe(key: str, action: str) -> bool:
            return not self.dangerous(key, action, guesses)

        def fresh(key: str) -> list[str]:
            options = [a for a in self.untried(key, actions) if not self.dangerous(key, a, guesses)]
            looks_open = [a for a in options if "wall" not in guesses.get(self.target_square(key, a), set())]
            return looks_open

        options = fresh(self.here)
        if options:
            unknown = [a for a in options if self.target_square(self.here, a) not in guesses]
            return (unknown or options)[0]
        route = self.path(self.here, lambda k: self.parse(k)[0] == self.scene and bool(fresh(k)), safe)
        if route:
            return route[0]
        retry = self.retryable(self.here)
        if retry:
            return retry[0]
        route = self.path(self.here, lambda k: self.parse(k)[0] == self.scene and bool(self.retryable(k)), safe)
        if route:
            return route[0]
        def leftovers(key: str) -> list[str]:  # only wall-looking arrows are left: check them too (never dangers)
            return [a for a in self.untried(key, actions) if not self.dangerous(key, a, guesses)]

        if leftovers(self.here):
            return leftovers(self.here)[0]
        route = self.path(self.here, lambda k: self.parse(k)[0] == self.scene and bool(leftovers(k)), safe)
        if route:
            return route[0]
        return self.least_tried(actions, guesses)

    def dangerous(self, key: str, action: str, guesses: dict | None = None) -> bool:
        """Would this arrow lead onto a known danger, or onto a square that looks like one? From where I stand now,
        also count the drift I'm in (still falling: the key's move plus the fall)."""
        target = self.target_square(key, action)
        if target is None:
            return False
        targets = [target]
        step = self.step()
        if key == self.here and step is not None:
            # where I might really land: the key's move plus my drift, or plus my whole last move (at most one
            # square each way — a safety check looks at every likely landing square, not one best guess)
            for carry in (self.me.drift, self.me.last_move or (0, 0)):
                dx, dy = (int(np.clip(round(d / s), -1, 1)) for d, s in zip(carry, step))
                if (dx or dy) and (target[0] + dx, target[1] + dy) not in targets:
                    targets.append((target[0] + dx, target[1] + dy))
        scene, _ = self.parse(key)
        for square in targets:
            place = self.places.get(self.key(square, scene))
            if place is not None and place.hazard:
                return True
            if guesses is not None and scene == self.scene and "hazard" in guesses.get(square, set()):
                return True
        return False

    def least_tried(self, actions: list[str], guesses: dict | None = None) -> str | None:
        """Everything here has been tried: try again the arrow tried least (counting the walk to it). Games with
        things the map can't see (where the boxes are, how fast I'm falling) give new results on a second try."""
        best, best_score = None, float("inf")
        seen, frontier, depth = {self.here: []}, [self.here], 0
        while frontier and depth <= 12:
            nxt = []
            for here in frontier:
                for action in actions:
                    edge = self.edges.get(here, {}).get(action)
                    if edge is None or edge.bad_end or edge.good_end or self.dangerous(here, action, guesses):
                        continue
                    score = edge.tries + 0.5 * depth
                    if score < best_score:
                        best, best_score = (seen[here] + [action])[0], score
                for action, there in self.neighbours(here).items():
                    if there not in seen:
                        seen[there] = seen[here] + [action]
                        nxt.append(there)
            frontier, depth = nxt, depth + 1
        return best

    def visible_squares(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Every square that fits wholly on screen (in the learned grid)."""
        step = self.step()
        if step is None or self.phase is None or frame is None:
            return []
        height, width = np.asarray(frame).shape[:2]
        ox, oy = self.phase[0] + self.camera[0], self.phase[1] + self.camera[1]  # square (0, 0) on screen
        qx0 = int(np.ceil((step[0] / 2 - ox) / step[0]))
        qy0 = int(np.ceil((step[1] / 2 - oy) / step[1]))
        qx1 = int(np.floor((width - step[0] / 2 - ox) / step[0]))
        qy1 = int(np.floor((height - step[1] / 2 - oy) / step[1]))
        return [(qx, qy) for qx in range(qx0, qx1 + 1) for qy in range(qy0, qy1 + 1)]

    def predict(self, frame: np.ndarray) -> dict[tuple[int, int], set[str]]:
        """Read the whole screen with the look memory: each visible square → what it probably is."""
        step = self.step()
        if step is None or self.phase is None or self.looks.empty or frame is None:
            return {}
        cache_key = (id(frame), self.looks.version)
        if self._predicted is not None and self._predicted[0] == cache_key and self._predicted[1] is frame:
            return self._predicted[2]
        guesses = {}
        squares = self.visible_squares(frame)
        vecs = looks_many(frame, [self.center_of(sq) for sq in squares], step)
        for square, labels in zip(squares, self.looks.labels_many(vecs)):
            if labels:
                guesses[square] = labels
        self._predicted = (cache_key, frame, guesses)
        return guesses

    # ---- saving (one file per crew member and game: notebook/<card>.nav.json)
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "me": self.me.to_dict(),
            "looks": self.looks.to_dict(),
            "scenes": [s.round(1).tolist() for s in self.scenes],
            "places": {k: asdict(p) for k, p in self.places.items()},
            "edges": {k: {a: asdict(e) for a, e in acts.items()} for k, acts in self.edges.items()},
            "phase": list(self.phase) if self.phase else None,
            "episodes": self.episodes,
            "dynamics": self.dynamics.to_dict(),
            "ballistics": self.ballistics.to_dict(),
            "auto": self.auto.to_dict(),
            "exits": {str(k): sorted(v) for k, v in self.exits.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NavMap":
        nav = cls()
        nav.me = SelfModel.from_dict(data.get("me", {}))
        nav.looks = Looks.from_dict(data.get("looks", {}))
        nav.scenes = [np.array(s, dtype=np.float32) for s in data.get("scenes", [])]
        nav.places = {k: Place(**p) for k, p in data.get("places", {}).items()}
        nav.edges = {k: {a: Edge(**e) for a, e in acts.items()} for k, acts in data.get("edges", {}).items()}
        phase = data.get("phase")
        nav.phase = tuple(phase) if phase else None
        nav.episodes = data.get("episodes", 0)
        nav.dynamics = Dynamics.from_dict(data.get("dynamics", {}))
        nav.ballistics = Ballistics.from_dict(data.get("ballistics", {}))
        nav.auto = AutoMover.from_dict(data.get("auto", {}))
        nav.exits = {int(k): set(v) for k, v in data.get("exits", {}).items()}
        return nav

    def transferable(self) -> "NavMap":
        """Only what carries over to a level never seen: my looks and moves, the look memory and the learned physics
        (no places)."""
        nav = NavMap()
        nav.me = SelfModel.from_dict(self.me.to_dict())
        nav.me.box = None
        nav.looks = Looks.from_dict(self.looks.to_dict())
        nav.phase = self.phase
        nav.dynamics = Dynamics.from_dict(self.dynamics.to_dict())  # physics is the same on every level
        nav.ballistics = Ballistics.from_dict(self.ballistics.to_dict())
        nav.auto = AutoMover.from_dict(self.auto.to_dict())  # how I move is the same on every level
        return nav


def _nearest_side(center: tuple[float, float], shape: tuple[int, ...]) -> str:
    """Which screen edge a point is closest to (the way I left a scene)."""
    x, y = center
    height, width = shape[:2]
    gaps = {"left": x, "right": width - x, "up": y, "down": height - y}
    return min(gaps, key=gaps.get)


def path_for(folder: Path, card_stem: str) -> Path:
    return Path(folder) / f"{card_stem}.nav.json"


def save(nav: NavMap, folder: Path, card_stem: str) -> Path:
    path = path_for(folder, card_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(nav.to_dict()), encoding="utf-8")
    tmp.replace(path)
    return path


def load(folder: Path, card_stem: str) -> NavMap:
    """The saved map, or a fresh one (a missing or damaged file never stops play)."""
    path = path_for(folder, card_stem)
    try:
        return NavMap.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return NavMap()
