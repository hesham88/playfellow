"""Things that keep moving by themselves — the buddy finds this out from watching, then steers instead of stepping.

Like a bike: in a maze one key press is one step and I stop when I stop pressing. On a bike I keep rolling whatever
I do, and the handlebars only change WHERE I roll. The buddy tells these apart from two kinds of evidence:

    * a step with no key where I still moved exactly as far as last time (rolling),
    * a step where the key changed but I carried on the same way (the key did not move me, it only steers).

Once it is sure it is rolling, `turn_for` says which way each key steers (learned from the times a key turned me),
and the planner uses that to turn away from a wall before it arrives. Nothing here knows any game: it is all counts.
"""
from __future__ import annotations

Move = tuple[int, int]  # how far I moved in one step, in pixels or squares (only sizes relative to each other matter)
MOVING = 4  # moving less than this each way counts as standing still (picture jitter)
SAME = 0.35  # two moves are "the same" when they point the same way and differ in length by less than this share

SUSPECT = 2  # this many "carried on after a different key" sightings make me test it by pressing nothing
STEER_SURE = 4  # without an idle test: this many "carried on after a different key" sightings before I trust it
KEEP_TURNS = 8  # remember this many sightings per key


def _still(move: Move) -> bool:
    return max(abs(move[0]), abs(move[1])) < MOVING


def _same(a: Move, b: Move) -> bool:
    if _still(a) or _still(b) or _unit(a) != _unit(b):
        return False
    return abs(max(map(abs, a)) - max(map(abs, b))) <= SAME * max(map(abs, b))


def _unit(move: Move) -> Move:
    """The direction of a move: (+1, 0), (0, -1) and so on (the bigger part wins)."""
    x, y = move
    if abs(x) >= abs(y):
        return (1 if x > 0 else -1, 0) if x else (0, 0)
    return (0, 1 if y > 0 else -1)


class AutoMover:
    def __init__(self):
        self.kept_going = 0  # pressing nothing, I moved on exactly as before (the strong sign)
        self.stopped = 0  # pressing nothing, I stood still after a move (so a key moves me)
        self.steered = 0  # a different key, yet I carried on the same way (weak: two keys can do the same thing)
        self.turns: dict[str, list[tuple[Move, Move]]] = {}  # key → (heading before, heading after) when it turned me
        self.ignored: set[tuple[str, Move]] = set()  # (key, heading) pairs where the key changed nothing
        self.last_move: Move = (0, 0)  # this episode: my last move in squares
        self.last_key: str | None = None

    # ---- learning
    def new_episode(self) -> None:
        """A new try: where I was heading no longer applies (what was learned stays)."""
        self.last_move, self.last_key = (0, 0), None

    def lost(self) -> None:
        """I could not see myself this step (a new room, a teleport): forget the heading."""
        self.new_episode()

    def observe(self, action: str | None, move: Move) -> None:
        """One step: the key pressed (None = nothing) and how I moved."""
        before = self.last_move
        if action is None:
            if not _still(before):
                if _same(move, before):
                    self.kept_going += 1
                elif _still(move):
                    self.stopped += 1
        elif not _still(before) and _same(move, before):
            self.ignored.add((action, _unit(before)))
            if action != self.last_key:
                self.steered += 1
        elif not _still(before) and not _still(move) and _unit(move) != _unit(before):
            sightings = self.turns.setdefault(action, [])
            sightings.append((_unit(before), _unit(move)))
            del sightings[:-KEEP_TURNS]
        self.last_move, self.last_key = move, action

    # ---- knowing
    @property
    def rolling(self) -> bool:
        """True when I keep moving by myself. A step pressing nothing decides it; without one, a pile of
        "carried on after a different key" sightings is enough."""
        if self.kept_going or self.stopped:
            return self.kept_going > self.stopped
        return self.steered >= STEER_SURE

    @property
    def heading(self) -> Move | None:
        """The way I am rolling now (a unit direction), if I am."""
        return _unit(self.last_move) if self.rolling and not _still(self.last_move) else None

    def wants_idle_test(self, known: bool) -> bool:
        """Spend one step pressing nothing? Only when something looks off (a key changed yet I carried on, a couple
        of times), I know myself, I am moving, and I have not tested it yet. A plain walker never wastes a step."""
        return known and not (self.kept_going or self.stopped) and not _still(self.last_move) and self.steered >= SUSPECT

    def turn_for(self, key: str, heading: Move) -> Move | None:
        """The heading this key leaves me with when I am heading `heading`; None if I never saw it turn me.
        Seen from this same heading: what happened then. Otherwise the key points a fixed way: that way, unless
        it is straight back (then the game ignored it and I carry on)."""
        if (key, heading) in self.ignored:
            return heading
        sightings = self.turns.get(key)
        if not sightings:
            return None
        same = [after for before, after in sightings if before == heading]
        if same:
            return max(set(same[-4:]), key=same[-4:].count)
        way = max(set(a for _, a in sightings[-4:]), key=[a for _, a in sightings[-4:]].count)
        return heading if way == (-heading[0], -heading[1]) else way

    # ---- saving
    def to_dict(self) -> dict:
        return {"kept_going": self.kept_going, "stopped": self.stopped, "steered": self.steered,
                "turns": {k: [[list(b), list(a)] for b, a in v] for k, v in self.turns.items()},
                "ignored": [[k, list(h)] for k, h in sorted(self.ignored)]}

    @classmethod
    def from_dict(cls, data: dict) -> "AutoMover":
        model = cls()
        model.kept_going = int(data.get("kept_going", 0))
        model.stopped = int(data.get("stopped", 0))
        model.turns = {k: [(tuple(b), tuple(a)) for b, a in v] for k, v in data.get("turns", {}).items()}
        model.steered = int(data.get("steered", 0))
        model.ignored = {(k, tuple(h)) for k, h in data.get("ignored", [])}
        return model
