"""How my moves work — the buddy learns the game's physics from its own key presses. / كيف تعمل حركاتي.

Like learning to ride a bike: after a few tries you know that if you're already rolling and you pedal, you go
further; if you jump while running, you fly in an arc; if there's a wall in front, you stop. The buddy learns the
same kind of thing from every step it takes:

    (my last few moves, which squares around me are solid, the key I pressed)  →  how I moved next

Nothing about gravity, jumping or walls is written in: it all comes from the buddy's own experience. It also
learns HOW MUCH it needs to remember: in a maze the last move doesn't matter (no momentum), but in a jumping game
the last few moves tell it whether it is rising or falling.

With that, it can imagine several moves ahead without pressing anything ("if I jump here, then hold right, I land
past the spikes") — the look-ahead that jumping games need. Physics is the same on every level of a game, so it
carries over to levels it has never seen.
"""
from __future__ import annotations

from collections import deque

MAX_HISTORY = 4  # remember at most this many of my last moves
MAX_SAMPLES = 4000
RECHOOSE_EVERY = 40  # look again at how much history matters after this many new samples
SURE_ENOUGH = 0.95  # a history length is "enough" when it explains this share of what happened
RISKY = 0.2  # an outcome at least this likely counts when checking for danger

Move = tuple[int, int]  # in squares
Solid = tuple[bool, bool, bool, bool]  # up, right, down, left


class Dynamics:
    """A counting model of my moves with back-off (like guessing the next word from the last few words)."""

    def __init__(self):
        self.samples: list[tuple[tuple[Move, ...], Solid, str, Move]] = []
        self.history = 0  # how many recent moves matter (learned)
        self._tables: dict = {}
        self._built_at = -1
        self._chosen_at = -1

    # ---- learning
    def add(self, history: tuple[Move, ...], solid: Solid, action: str, move: Move) -> None:
        self.samples.append((tuple(history[-MAX_HISTORY:]), solid, action, move))
        del self.samples[:-MAX_SAMPLES]
        if len(self.samples) - self._chosen_at >= RECHOOSE_EVERY:
            self._choose_history()

    @property
    def physics(self) -> bool:
        """True when my past moves change what a key does (momentum, gravity): plan by imagining moves ahead."""
        return self.history > 0 and len(self.samples) >= RECHOOSE_EVERY

    def _choose_history(self) -> None:
        """The shortest memory that explains (almost) as much as the longest one."""
        scores = []
        for k in range(MAX_HISTORY + 1):
            groups: dict = {}
            for history, solid, action, move in self.samples:
                key = (_tail(history, k), solid, action)
                groups.setdefault(key, {}).setdefault(move, 0)
                groups[key][move] += 1
            explained = sum(max(c.values()) for c in groups.values())
            scores.append(explained / max(len(self.samples), 1))
        best = max(scores)
        self.history = next(k for k, s in enumerate(scores) if s >= min(best, SURE_ENOUGH) - 1e-9)
        self._chosen_at = len(self.samples)
        self._built_at = -1

    def _build(self) -> None:
        if self._built_at == len(self.samples):
            return
        tables: dict = {}
        for history, solid, action, move in self.samples:
            for k in range(self.history, -1, -1):
                for key in ((k, _tail(history, k), solid, action), (k, _tail(history, k), None, action)):
                    counts = tables.setdefault(key, {})
                    counts[move] = counts.get(move, 0) + 1
        self._tables = tables
        self._built_at = len(self.samples)

    # ---- predicting
    def predict(self, history: tuple[Move, ...], solid: Solid, action: str) -> list[tuple[Move, float]] | None:
        """How likely each next move is (most likely first); None if nothing like this was ever tried.
        Backs off to a shorter memory, then ignores the walls around, when this exact situation is new."""
        self._build()
        for k in range(self.history, -1, -1):
            for key in ((k, _tail(history, k), solid, action), (k, _tail(history, k), None, action)):
                counts = self._tables.get(key)
                if counts:
                    total = sum(counts.values())
                    return sorted(((m, c / total) for m, c in counts.items()), key=lambda mc: -mc[1])
        return None

    # ---- saving
    def to_dict(self) -> dict:
        return {"history": self.history,
                "samples": [[list(map(list, h)), list(s), a, list(m)] for h, s, a, m in self.samples]}

    @classmethod
    def from_dict(cls, data: dict) -> "Dynamics":
        model = cls()
        model.samples = [(tuple(tuple(x) for x in h), tuple(s), a, tuple(m)) for h, s, a, m in data.get("samples", [])]
        model.history = data.get("history", 0)
        model._chosen_at = len(model.samples)
        return model


def _tail(history: tuple[Move, ...], k: int) -> tuple[Move, ...]:
    if k == 0:
        return ()
    padded = ((0, 0),) * MAX_HISTORY + tuple(history)
    return padded[-k:]


def imagine(model: Dynamics, start: tuple[int, int], history: tuple[Move, ...], actions: list[str],
            solid, danger, wanted, inside, fallback, limit: int = 6000) -> list[str] | None:
    """Imagine moves ahead (breadth first) until a square where wanted(square) is true; the keys to get there.

    solid(square) / danger(square): what the screen reading says; inside(square): still on screen.
    fallback(action): a key's usual move when the model has never seen this situation.
    A move is skipped when any reasonably likely outcome lands on a danger."""
    def neighbours(square) -> Solid:
        x, y = square
        return solid((x, y - 1)), solid((x + 1, y)), solid((x, y + 1)), solid((x - 1, y))

    k = max(model.history, 1)
    first = (start, _tail(history, k))
    seen = {first: None}
    queue = deque([first])
    while queue and len(seen) < limit:
        state = queue.popleft()
        square, past = state
        around = neighbours(square)
        for action in actions:
            outcomes = model.predict(past, around, action)
            if outcomes is None:
                guess = fallback(action)
                if guess is None:
                    continue
                outcomes = [(guess, 1.0)]
            if any(p >= RISKY and danger((square[0] + m[0], square[1] + m[1])) for m, p in outcomes):
                continue
            move = outcomes[0][0]
            target = (square[0] + move[0], square[1] + move[1])
            if solid(target):  # never through a wall: slide along it, or stay
                for option in ((move[0], 0), (0, move[1]), (0, 0)):
                    target = (square[0] + option[0], square[1] + option[1])
                    if not solid(target):
                        move = option
                        break
            if danger(target):
                continue
            nxt = (target, _tail(past + (move,), k))
            if nxt in seen:
                continue
            seen[nxt] = (state, action)
            if wanted(target):
                keys = []
                while seen[nxt] is not None:
                    nxt, key = seen[nxt]
                    keys.append(key)
                return keys[::-1]
            if inside(target):
                queue.append(nxt)
    return None


# ---------------------------------------------------------------------------------------------------------------
# Ballistics (T32b): the same idea as above, but with numbers, for games where you keep rolling and fall in arcs.
# It learns a handful of values (how fast a key speeds me up, how much I slow down by myself, how hard gravity pulls,
# how hard a jump kicks) from the pixel positions the buddy sees, and can then play a jump out in its head:
# "if I jump now and hold right for 300 ms, do I land on that ledge?"
# ---------------------------------------------------------------------------------------------------------------
import numpy as np  # noqa: E402  (kept next to the code that uses it)

REST_FRAMES = 3  # standing on something takes this many still frames in a row (the top of a jump is still for 1-2)
MIN_PULL = 100.0  # pixels/s²: less pull than this is no gravity
STILL = 0.5  # pixels: moving less than this in a tick counts as standing still


class Body:
    """Where I am and how fast I am going, in pixels and pixels per second."""
    __slots__ = ("x", "y", "vx", "vy", "grounded")

    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, grounded: bool = False):
        self.x, self.y, self.vx, self.vy, self.grounded = float(x), float(y), float(vx), float(vy), bool(grounded)

    def copy(self) -> "Body":
        return Body(self.x, self.y, self.vx, self.vy, self.grounded)

    def __repr__(self) -> str:
        return f"Body(x={self.x:.1f}, y={self.y:.1f}, vx={self.vx:.1f}, vy={self.vy:.1f}, grounded={self.grounded})"


class Ballistics:
    """Learned physics with numbers: drag, gravity, a push per key and a jump kick per key.

    Per tick of `dt` seconds:  vx = drag·vx + push[key]·dt ;  vy += gravity·dt ;  a jump key pressed on the ground
    sets vy = kick[key].  The numbers are fitted by least squares from what the buddy watched (`observe`)."""

    MIN_SAMPLES = 8

    def __init__(self):
        self.gravity = 0.0  # pixels/s², positive = down the screen
        self.drag = 1.0  # share of sideways speed kept per tick (1 = nothing slows me)
        self.push: dict[str, float] = {}  # key → sideways speed added per second while held
        self.kick: dict[str, float] = {}  # key → vertical speed given when pressed on the ground (negative = up)
        self._rows: list[tuple[float, float, float, float, bool, str | None, float]] = []
        self._last: Body | None = None
        self._rest = 0  # still frames in a row
        self._drops: list[float] = []  # the last few vertical changes per tick, in pixels

    # ---- learning
    def reset_track(self) -> None:
        """Forget where I just was (a new level or a teleport): the next sighting is a fresh start."""
        self._last = None
        self._rest, self._drops = 0, []

    def observe(self, x: float, y: float, action: str | None, dt: float) -> Body:
        """One sighting of me, `dt` seconds after the last one, with `action` held during that time."""
        now = Body(x, y)
        prev = self._last
        if prev is not None and dt > 0:
            now.vx, now.vy = (x - prev.x) / dt, (y - prev.y) / dt
            now.grounded = self._standing(y - prev.y, dt)
            self._rows.append((prev.vx, prev.vy, now.vx, now.vy, prev.grounded, action, dt))
            del self._rows[:-3000]
        self._last = now
        return now

    def _standing(self, dy: float, dt: float) -> bool:
        """Am I resting on something? Not just 'barely moving': the top of a jump is barely moving too, for a frame
        or two. So I must be still for several frames in a row, and not be speeding up by what gravity says."""
        self._rest = self._rest + 1 if abs(dy) < STILL else 0
        self._drops = (self._drops + [dy])[-REST_FRAMES:]
        if self._rest < REST_FRAMES:
            return False
        pull = self.gravity * dt * dt  # how much each tick's change grows in free fall
        if self.gravity > MIN_PULL and len(self._drops) == REST_FRAMES:
            growth = [b - a for a, b in zip(self._drops, self._drops[1:])]
            if all(abs(g - pull) <= 0.35 * pull for g in growth):
                return False  # still 'dropping' by gravity's amount each tick: that is the top of an arc
        return True

    def fit(self) -> bool:
        """Work the numbers out from everything watched so far. True when there was enough to say."""
        rows = self._rows
        if len(rows) < self.MIN_SAMPLES:
            return False
        # Gravity: in the air, with no key that jumps, the typical change in vertical speed per second.
        # Only ticks where I was clearly moving up or down both before and after (not standing, not taking off or landing).
        air = [(vy1 - vy0) / dt for _vx0, vy0, _vx1, vy1, _g, _a, dt in rows
               if abs(vy0) * dt >= STILL and abs(vy1) * dt >= STILL]
        if air:
            self.gravity = float(np.median(air))
        # Sideways: vx1 = drag·vx0 + push[key]·dt — one drag for everything, one push per key.
        keys = sorted({r[5] for r in rows if r[5]})
        if keys:
            column = {k: 1 + i for i, k in enumerate(keys)}
            design = np.zeros((len(rows), 1 + len(keys)))
            target = np.zeros(len(rows))
            for n, (vx0, _vy0, vx1, _vy1, _g, action, dt) in enumerate(rows):
                design[n, 0] = vx0
                if action in column:
                    design[n, column[action]] = dt
                target[n] = vx1
            solved, *_ = np.linalg.lstsq(design, target, rcond=None)
            self.drag = float(min(max(solved[0], 0.0), 1.0))
            self.push = {k: float(solved[c]) for k, c in column.items() if abs(solved[c]) > 1.0}
        # Jump kick: on the ground, a key that suddenly gives upward speed.
        kicks: dict[str, list[float]] = {}
        for _vx0, _vy0, _vx1, vy1, grounded, action, dt in rows:
            if grounded and action and vy1 * dt < -STILL:
                kicks.setdefault(action, []).append(vy1)
        # A launch only matters where something pulls me back down; in a maze "up" is just a walk.
        self.kick = {k: float(np.median(v)) for k, v in kicks.items() if len(v) >= 2 and self.gravity > MIN_PULL}
        for key in self.kick:
            self.push.pop(key, None)  # a jump key is not also a sideways push
        return True

    @property
    def known(self) -> bool:
        return self.gravity != 0.0 or bool(self.push) or bool(self.kick)

    # ---- imagining
    def step(self, body: Body, action: str | None, dt: float, solid=None) -> Body:
        """Play one tick in my head. `solid(x, y)` says whether a pixel is wall or floor (None = nothing is)."""
        solid = solid or (lambda x, y: False)
        nxt = body.copy()
        nxt.vx = self.drag * body.vx + self.push.get(action, 0.0) * dt
        nxt.vy = body.vy + self.gravity * dt
        if body.grounded and action in self.kick:
            nxt.vy = self.kick[action]
        x, y = body.x + nxt.vx * dt, body.y + nxt.vy * dt
        if solid(x, body.y):  # a wall in the way sideways
            x, nxt.vx = body.x, 0.0
        nxt.grounded = False
        if solid(x, y):  # floor below, or a ceiling above
            nxt.grounded = nxt.vy >= 0
            nxt.vy, y = 0.0, body.y
        nxt.x, nxt.y = x, y
        return nxt

    def predict(self, body: Body, action: str | None, duration_ms: int, dt: float = 0.05, solid=None,
                after_ms: int = 0) -> list[Body]:
        """The path I would take holding `action` for `duration_ms`, then letting go for `after_ms` (falling)."""
        path, current = [], body.copy()
        for n in range(int(round(duration_ms / 1000 / dt)) + int(round(after_ms / 1000 / dt))):
            held = action if n < int(round(duration_ms / 1000 / dt)) else None
            current = self.step(current, held, dt, solid)
            path.append(current)
        return path

    def plan_platform_trajectory(self, start: Body, goal: tuple[float, float], reach: float, solid=None,
                                 danger=None, dt: float = 0.05, max_ms: int = 2000,
                                 holds: tuple[int, ...] = (0, 100, 200, 300, 450, 600, 800)):
        """Find the keys that put me within `reach` pixels of `goal` without touching a danger.

        The plan is: press the jump key for one tick (if I stand on something), hold a sideways key for `hold_ms`,
        then let go and fall. Returns {"jump", "side", "hold_ms", "gap"} for the closest plan, or None."""
        jumps = [None] + list(self.kick) if start.grounded else [None]
        sides = [None] + list(self.push)
        best = None
        for jump in jumps:
            for side in sides:
                for hold in holds:
                    gap = self._gap(start, jump, side, hold, goal, solid, danger, dt, max_ms)
                    if gap is not None and gap <= reach and (best is None or gap < best["gap"]):
                        best = {"jump": jump, "side": side, "hold_ms": hold, "gap": gap}
        return best

    def _gap(self, start, jump, side, hold, goal, solid, danger, dt, max_ms):
        """How close this plan gets to `goal` (None if it touches a danger on the way)."""
        current, closest = start.copy(), float("inf")
        held_ticks = int(round(hold / 1000 / dt))
        for n in range(int(max_ms / 1000 / dt)):
            action = side if n < held_ticks else None
            if n == 0 and jump:
                action = jump
            current = self.step(current, action, dt, solid)
            if danger and danger(current.x, current.y):
                return None
            closest = min(closest, float(np.hypot(current.x - goal[0], current.y - goal[1])))
            if n > held_ticks and current.grounded:
                break
        return closest

    # ---- saving
    def to_dict(self) -> dict:
        return {"gravity": self.gravity, "drag": self.drag, "push": self.push, "kick": self.kick}

    @classmethod
    def from_dict(cls, data: dict) -> "Ballistics":
        model = cls()
        model.gravity = float(data.get("gravity", 0.0))
        model.drag = float(data.get("drag", 1.0))
        model.push = dict(data.get("push", {}))
        model.kick = dict(data.get("kick", {}))
        return model
