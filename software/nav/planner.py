"""Planning on what was learned: the shortest way to every treasure, then the goal. / التخطيط بما تعلّمه الرفيق.

Like planning a trip with your own hand-drawn map: roads you have walked are sure; roads you can only see from
a distance are probably fine; places you have never looked at closely might be blocked. The planner:
  - finds the shortest way between places (Dijkstra) using only the buddy's own map and look memory;
  - picks the order of treasures that needs the fewest steps (tries every order for up to 7 treasures, otherwise
    nearest-first then fixes crossings with 2-opt), and keeps the goal for last;
  - never uses a move that ENDS the game except as the very last step to the goal.
This is the NAVIGATE skill: the brain chooses where to go ("nearest treasure", "the goal", "explore", or "all"),
the skill moves precisely, the HUD checks the result.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nav import map as navmap
from nav.dynamics import imagine
from nav.graph4 import direction_of
from nav.map import NavMap

try:  # the optional fast helper (T38b): the native graph search, used by Planner.route_frame
    import native_bridge
except ImportError:  # pragma: no cover
    native_bridge = None

GUESSED_COST = 1.2  # a road seen but never walked
UNKNOWN_COST = 2.0  # a road onto a square that looks like nothing known yet
SURE, GUESSED, UNKNOWN = 1.0, 0.7, 0.4  # how sure we are of each kind of road
BRUTE_FORCE = 7  # up to this many treasures, try every order
GOALS = ("all", "reward", "goal", "explore")  # or "to:X,Y" — go to one square of the learned grid


@dataclass
class Road:
    action: str
    to: str
    cost: float
    sure: float
    ends: bool = False  # this move ends the game (only ever the last step, into the goal)


@dataclass
class Advice:
    """What the NAVIGATE skill suggests: the next keys, how sure it is, where it is heading and why."""
    actions: list[str]
    confidence: float
    target: str | None
    kind: str  # "probe", "reward", "goal", "explore" or "none"
    route: list[str] = field(default_factory=list)  # every key to the target (actions is the first part)


class Planner:
    def __init__(self, nav: NavMap, actions: list[str]):
        self.nav = nav
        self.actions = list(actions)
        self._warm: dict[str, float] | None = None  # last practice values (a warm start for the next step)
        self.slots = None  # brain.slots.SlotRegistry, set by the built-in brain; None = rules only

    # ---- the roads
    def roads(self, frame: np.ndarray) -> dict[str, list[Road]]:
        nav = self.nav
        out: dict[str, list[Road]] = {}
        for place, acts in nav.edges.items():
            for action, edge in acts.items():
                road = nav.learned_road(place, action, edge)
                if road is not None:
                    where, cost, sure, ends = road
                    out.setdefault(place, []).append(Road(action, where, cost, SURE if ends else sure, ends=ends))
        for key, road in self._guessed(frame):
            out.setdefault(key, []).append(road)
        return out

    def _guessed(self, frame: np.ndarray) -> list[tuple[str, Road]]:
        """Roads nobody has walked yet but the picture suggests, as (from place, road). Never for an arrow already tried."""
        nav = self.nav
        found: list[tuple[str, Road]] = []
        if nav.scene is None or nav.step() is None or nav.phase is None:
            return found
        guesses = nav.predict(frame)
        visible = set(nav.visible_squares(frame))
        moving = [a for a in self.actions if nav.me.effect(a) is not None]
        for square in visible:
            key = nav.key(square)
            if nav.places.get(key) and nav.places[key].hazard:
                continue
            tried = nav.edges.get(key, {})
            for action in moving:
                if action in tried:
                    continue  # experience beats guessing
                target = nav.target_square(key, action)
                if target not in visible:
                    continue
                labels = guesses.get(target, set())
                if "hazard" in labels:
                    continue
                if "goal" in labels:
                    found.append((key, Road(action, nav.key(target), GUESSED_COST, GUESSED, ends=True)))
                elif "wall" in labels:
                    continue
                elif labels:
                    found.append((key, Road(action, nav.key(target), GUESSED_COST, GUESSED)))
                else:
                    found.append((key, Road(action, nav.key(target), UNKNOWN_COST, UNKNOWN)))
        return found


    @staticmethod
    def shortest(roads: dict[str, list[Road]], start: str) -> tuple[dict[str, float], dict[str, tuple[str, Road]]]:
        """Dijkstra from `start` over roads that don't end the game."""
        dist, back = {start: 0.0}, {}
        heap = [(0.0, start)]
        while heap:
            d, here = heapq.heappop(heap)
            if d > dist.get(here, float("inf")):
                continue
            for road in roads.get(here, []):
                if road.ends:
                    continue
                nd = d + road.cost
                if nd < dist.get(road.to, float("inf")):
                    dist[road.to], back[road.to] = nd, (here, road)
                    heapq.heappush(heap, (nd, road.to))
        return dist, back

    # ---- one way from here to one place
    last_route_engine = ""

    def route(self, roads: dict[str, list[Road]], start: str, goal: str) -> list[Road] | None:
        """The roads of the cheapest way from `start` to `goal`, or None when there is no way (the Python search)."""
        self.last_route_engine = "python"
        dist, back = self.shortest(roads, start)
        return self.walk(back, start, goal) if goal in dist else None

    NOT_ANSWERED = object()  # "the road grid cannot answer this exactly": ask the Python search instead

    def route_frame(self, frame: np.ndarray, start: str, goal: str):
        """Same answer as `route(self.roads(frame), start, goal)` (a cheapest way, as a list of Road, or None for no way),
        worked out on the map's road grid with the native graph search when it is built (`last_route_engine` says who).
        Returns `NOT_ANSWERED` when the grid cannot do it exactly (no native search, a road that is not a one-square step
        in one room, start and goal in different rooms); the caller then uses the Python search."""
        nav = self.nav
        if native_bridge is None:  # graph4_astar itself says None when the library lacks the search
            return self.NOT_ANSWERED
        nav.sync_grid()
        grid = nav.grid
        try:
            scene, start_sq = nav.parse(start)
            scene2, goal_sq = nav.parse(goal)
        except ValueError:
            return self.NOT_ANSWERED
        if scene != scene2 or not grid.exact:
            return self.NOT_ANSWERED
        guessed = [(key, road.to, road.cost, road.sure, road.action) for key, road in self._guessed(frame) if not road.ends]
        with grid.overlay(scene, guessed) as extra:
            if extra.skipped:
                return self.NOT_ANSWERED
            sc = grid.scenes.get(scene)
            if sc is None:
                return self.NOT_ANSWERED
            sc.cover(*start_sq)
            sc.cover(*goal_sq)
            found = native_bridge.graph4_astar(sc.costs, sc.cell(*start_sq), sc.cell(*goal_sq))
            if found is None:
                return self.NOT_ANSWERED
            path, _ = found
            if not path:
                return [] if start == goal else None
            steps: list[Road] = []
            for (r1, c1), (r2, c2) in zip(path, path[1:]):
                a = (c1 + sc.ox, r1 + sc.oy)
                d = direction_of(a, (c2 + sc.ox, r2 + sc.oy))
                slot = (scene, a[0], a[1], d)
                options = [(cost, action, to, share) for action, (cost, to, share) in grid.slots.get(slot, {}).items()]
                if slot in extra.added:
                    cost, to, share, action = extra.added[slot]
                    options.append((cost, action, to, share))
                if not options:
                    return self.NOT_ANSWERED
                cost, action, to, share = min(options, key=lambda o: o[0])
                steps.append(Road(action, to, cost, share))
        self.last_route_engine = "native"
        return steps

    @staticmethod
    def walk(back: dict[str, tuple[str, Road]], start: str, end: str) -> list[Road]:
        path = []
        while end != start:
            here, road = back[end]
            path.append(road)
            end = here
        return path[::-1]

    # ---- targets
    def rewards(self, frame: np.ndarray) -> list[str]:
        """Treasures still to take: reward-looking squares on screen, plus remembered ones out of sight."""
        nav = self.nav
        found: set[str] = set()
        guesses = nav.predict(frame)
        visible = set(nav.visible_squares(frame))
        for square, labels in guesses.items():
            if "reward" in labels and "goal" not in labels:
                found.add(nav.key(square))
        for key, place in nav.places.items():
            if not place.rewards or key in nav.collected:
                continue
            scene, square = nav.parse(key)
            if scene == nav.scene and square in visible:
                continue  # on screen: trust what it looks like now (taken treasure is gone)
            found.add(key)
        found.discard(nav.here)
        # A square where the game ended is a way to FINISH, never a treasure, however much its picture looked like a pickup
        # (it changed the HUD too). The end is taken last, after the pickups.
        found -= {key for key, place in nav.places.items() if place.goal}
        return sorted(found)

    def goal_roads(self, roads: dict[str, list[Road]]) -> dict[str, list[tuple[str, Road]]]:
        """Every known way to finish: goal place → [(from place, ending road)]."""
        out: dict[str, list[tuple[str, Road]]] = {}
        for place, rs in roads.items():
            for road in rs:
                if road.ends:
                    out.setdefault(road.to, []).append((place, road))
        return out

    # ---- choosing a route
    def order(self, start: str, targets: list[str], table: dict[str, dict[str, float]],
              finish: dict[str, float] | None) -> list[str]:
        """The visiting order with the fewest steps (finish: cost to end the game from each place, or None)."""
        def length(seq):
            here, total = start, 0.0
            for t in seq:
                total += table[here].get(t, float("inf"))
                here = t
            return total + (finish.get(here, float("inf")) if finish is not None else 0.0)

        if len(targets) <= BRUTE_FORCE:
            return list(min(itertools.permutations(targets), key=length, default=()))
        seq, here, left = [], start, set(targets)
        while left:  # nearest first…
            nxt = min(left, key=lambda t: table[here].get(t, float("inf")))
            seq.append(nxt)
            left.remove(nxt)
            here = nxt
        improved = True
        while improved:  # …then undo crossings (2-opt)
            improved = False
            for i, j in itertools.combinations(range(len(seq)), 2):
                trial = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                if length(trial) < length(seq) - 1e-9:
                    seq, improved = trial, True
        return seq

    def advise(self, frame: np.ndarray, goal: str = "all", max_actions: int = 1) -> Advice:
        """The next keys toward `goal`. Whatever chose them, a first move that looks deadly from here is swapped
        for a safe exploring move (never walk onto a danger I already know, by place or by look)."""
        nav = self.nav
        if nav.auto.wants_idle_test(nav.me.known):
            return Advice([], 0.3, None, "wait")  # press nothing for one step: do I keep moving by myself?
        advice = self._advise(frame, goal, max_actions)
        if advice.actions and nav.here and advice.kind != "probe" and                 nav.dangerous(nav.here, advice.actions[0], nav.predict(frame)):
            action = nav.explore_action(frame, self.actions)
            if action is None or nav.dangerous(nav.here, action, nav.predict(frame)):
                advice = Advice([], 0.0, None, "none")
            else:
                advice = Advice([action], 0.5, None, "avoid", [action])
        return self._steer(frame, advice)

    LOOK = 2  # squares ahead I check when I am rolling: closer than this to trouble and I turn
    ROOM = 4  # a turn is good when it leaves this many safe squares

    def _steer(self, frame: np.ndarray, advice: Advice) -> Advice:
        """If I keep rolling by myself and the way I would be heading runs into the screen edge, a wall or a
        danger within a couple of squares, turn away now (a key moves a roller's heading, not its position)."""
        nav, auto = self.nav, self.nav.auto
        heading = auto.heading
        if heading is None or nav.here is None or frame is None or nav.step() is None:
            return advice
        scene, here = nav.parse(nav.here)
        if scene != nav.scene:
            return advice
        guesses = nav.predict(frame)
        visible = set(nav.visible_squares(frame))
        edge_kills = nav.me.camera.offset == (0, 0)  # a roller cannot stop: off a still screen is the end

        def run(direction) -> int:
            """How many squares ahead (up to 4) are safe going this way."""
            for ahead in range(1, 5):
                square = (here[0] + direction[0] * ahead, here[1] + direction[1] * ahead)
                labels = guesses.get(square, set())
                place = nav.places.get(nav.key(square))
                if (edge_kills and square not in visible) or ("wall" in labels and "goal" not in labels)                         or "hazard" in labels or bool(place and place.hazard):
                    return ahead - 1
            return 4

        def going(key: str | None) -> tuple[int, int] | None:
            """The heading this key leaves me with (None = I do not know)."""
            if key is None:
                return heading
            turn = auto.turn_for(key, heading)
            if turn is not None:
                return turn
            return None

        chosen = advice.actions[0] if advice.actions else None
        now = going(chosen)
        if now is not None and run(now) >= self.LOOK:
            return advice
        options = [(run(d), key) for key in [None] + self.actions for d in [going(key)]
                   if d is not None and d != (-heading[0], -heading[1])]
        safe = [o for o in options if o[0] >= self.ROOM]
        if safe:  # a turn I know works, and I can see room beyond it: best room first, my own choice on a tie
            best_run, best_key = max(safe, key=lambda o: (o[0], o[1] == chosen, o[1] is None))
        else:  # nothing I know is safe: a key I have never seen turn me is a better bet than a known crash
            untried = [k for k in self.actions if auto.turn_for(k, heading) is None and k != chosen]
            if untried:
                return Advice([untried[0]], 0.3, None, "steer", [untried[0]])
            if not options:
                return advice
            best_run, best_key = max(options, key=lambda o: (o[0], o[1] == chosen, o[1] is None))
        if now is not None and best_run <= run(now):
            return advice
        if best_key is None:
            return Advice([], 0.6, None, "steer")
        return Advice([best_key], 0.6, None, "steer", [best_key])

    def _advise(self, frame: np.ndarray, goal: str, max_actions: int) -> Advice:
        nav = self.nav
        if not nav.me.known:
            return Advice([nav.me.probe_action(self.actions)], 0.2, None, "probe")
        if nav.here is None:
            if nav.step() is None:  # I know what I look like but not yet how far one press takes me: keep trying keys
                return Advice([nav.me.probe_action(self.actions)], 0.2, None, "probe")
            return Advice([], 0.0, None, "none")
        rotate = nav.frozen_key(self.actions)
        if rotate is not None:  # the picture froze after a step that may have ended the game: try a key not pressed yet
            return Advice([rotate], 0.3, None, "untried", [rotate])
        if goal.startswith("to:"):
            return self._go_to(frame, goal[3:], max_actions)
        if nav.dynamics.physics:
            imagined = self._imagine(frame, goal, max_actions)
            if imagined is not None:
                return imagined
        if goal != "explore":
            roads = self.roads(frame)
            dist, back = self.shortest(roads, nav.here)
            treasures = [t for t in self.rewards(frame) if t in dist] if goal in ("all", "reward") else []
            finishes = self.goal_roads(roads) if goal in ("all", "goal") else {}
            if goal == "reward" and treasures:
                treasures = [min(treasures, key=lambda t: dist[t])]
            if treasures:
                table = {nav.here: dist}
                for t in treasures:
                    table[t] = self.shortest(roads, t)[0]
                finish = self._finish_costs(roads, finishes, [nav.here] + treasures) if finishes else None
                first = self.order(nav.here, treasures, table, finish)[0]
                path = self.walk(back, nav.here, first)
                return self._advice(path, first, "reward", max_actions)
            if finishes:
                best = None
                for target, entries in finishes.items():
                    for place, road in entries:
                        if place in dist and (best is None or dist[place] + road.cost < best[0]):
                            best = (dist[place] + road.cost, place, road, target)
                if best is not None:
                    _, place, road, target = best
                    return self._advice(self.walk(back, nav.here, place) + [road], target, "goal", max_actions)
            valued = self._valued()
            if valued is not None:
                return valued
        action = nav.explore_action(frame, self.actions)
        if action is None:
            return Advice([], 0.0, None, "none")
        return Advice([action], 0.5, None, "explore", [action])

    def _imagine(self, frame: np.ndarray, goal: str, max_actions: int) -> Advice | None:
        """Jumping games: imagine moves ahead with the learned physics on the screen as read, to the nearest
        treasure, then the goal, else somewhere new — never through walls or onto anything that looks deadly."""
        nav = self.nav
        step = nav.step()
        if frame is None or step is None or nav.here is None:
            return None
        guesses = nav.predict(frame)
        visible = set(nav.visible_squares(frame))
        scene, here = nav.parse(nav.here)
        if scene != nav.scene:
            return None

        def solid(square):
            labels = guesses.get(square, set())
            return "wall" in labels and "goal" not in labels

        def danger(square):
            place = nav.places.get(nav.key(square))
            return "hazard" in guesses.get(square, set()) or bool(place and place.hazard)

        def fallback(action):
            effect = nav.me.effect(action)
            return None if effect is None else (round(effect[0] / step[0]), round(effect[1] / step[1]))

        def go(wanted, kind, then: list[str] | None = None) -> Advice | None:
            keys = imagine(nav.dynamics, here, nav.moves, self.actions, solid, danger, wanted,
                           lambda sq: sq in visible, fallback)
            if keys is None and then and wanted(here):
                keys = []
            if keys is None:
                return None
            keys = keys + (then or [])
            if not keys:
                return None
            return Advice(keys[:max(1, max_actions)], 0.7, None, kind, keys)

        if goal in ("all", "reward"):
            treasures = {nav.parse(k)[1] for k in self.rewards(frame) if nav.parse(k)[0] == nav.scene}
            if treasures:
                advice = go(lambda sq: sq in treasures, "reward")
                if advice:
                    return advice
        if goal in ("all", "goal"):
            goals = {sq for sq, labels in guesses.items() if "goal" in labels}
            goals |= {nav.parse(k)[1] for k, p in nav.places.items() if p.goal and nav.parse(k)[0] == nav.scene}
            if goals:
                advice = go(lambda sq: sq in goals, "goal")
                if advice:
                    return advice
            for place, acts in nav.edges.items():  # finishing keys that don't move me (open a door with Z)
                for action, edge in acts.items():
                    where, square = nav.parse(place)
                    if edge.good_end and where == nav.scene and nav.me.effect(action) is None:
                        advice = go(lambda sq, sq0=square: sq == sq0, "goal", then=[action])
                        if advice:
                            return advice
        if goal in ("all", "explore"):
            xs, ys = [sq[0] for sq in visible], [sq[1] for sq in visible]
            used = nav.exits.get(nav.scene, set())

            def new(square):
                if square in visible:
                    return nav.key(square) not in nav.places  # a square I have never stood on
                side = ("left" if square[0] < min(xs) else "right" if square[0] > max(xs)
                        else "up" if square[1] < min(ys) else "down")
                return side not in used  # a way off screen I haven't taken yet
            return go(new, "explore")
        return None

    def _go_to(self, frame: np.ndarray, where: str, max_actions: int) -> Advice:
        """Walk to one square the brain asked for (in this scene), by the learned roads."""
        nav = self.nav
        try:
            x, y = (int(v) for v in where.split(","))
        except ValueError:
            return Advice([], 0.0, None, "none")
        target = nav.key((x, y))
        if target == nav.here:
            return Advice([], 1.0, target, "arrived")
        way = self.route_frame(frame, nav.here, target)
        if way is self.NOT_ANSWERED:
            way = self.route(self.roads(frame), nav.here, target)
        if way is None:
            action = nav.explore_action(frame, self.actions)
            return Advice([action] if action else [], 0.3, target, "explore", [action] if action else [])
        return self._advice(way, target, "to", max_actions)

    def _valued(self) -> Advice | None:
        """No sure road to treasure or the goal: follow the practice values (they also count roads whose outcome
        varies, like moves in mid-air), if any action is expected to pay off."""
        from nav.practice import STEP_COST, values  # practice builds on the planner; imported when needed
        self._warm, q = values(self.nav, warm=self._warm, sweeps=12 if self._warm else 80)
        options = q.get(self.nav.here, {})
        if self.slots is not None:  # S4: a learned value part, only when it passed its gate (brain/slots.py)
            options = self.slots.decide("S4_value", options, (self.nav.here, dict(options)),
                                        valid=lambda o: isinstance(o, dict) and bool(o) and all(isinstance(v, (int, float)) for v in o.values()))
        if not options:
            return None
        action, worth = max(options.items(), key=lambda item: item[1])
        if worth <= STEP_COST:
            return None
        return Advice([action], min(1.0, worth), None, "value", [action])

    def _finish_costs(self, roads, finishes, places) -> dict[str, float]:
        costs = {}
        for place in places:
            dist = self.shortest(roads, place)[0]
            costs[place] = min((dist.get(src, float("inf")) + road.cost
                                for entries in finishes.values() for src, road in entries), default=float("inf"))
        return costs

    @staticmethod
    def _advice(path: list[Road], target: str, kind: str, max_actions: int) -> Advice:
        first = path[:max(1, max_actions)]
        confidence = float(np.prod([road.sure for road in first])) if first else 0.0
        return Advice([road.action for road in first], confidence, target, kind, [road.action for road in path])


class Navigator:
    """The NAVIGATE skill for one game: the learned map + the planner, saved in the notebook between runs."""

    def __init__(self, actions: list[str], folder: Path | None = None, card_stem: str | None = None,
                 nav: NavMap | None = None, surprise: Any = None, skills: Any = None):
        self.folder, self.card_stem = folder, card_stem
        if nav is None:
            nav = navmap.load(folder, card_stem) if folder and card_stem else NavMap()
        self.nav = nav
        self.planner = Planner(self.nav, actions)
        self.surprise = surprise
        self.skills = skills
        self.episode_errors: list[float] = []  # this episode's prediction errors (with a surprise module)
        self.episode_curiosity = 0.0  # this episode's summed intrinsic reward (surprise plus empowerment)

    def for_new_level(self) -> "Navigator":
        """A navigator for a level never played: it keeps what things look like and how I move, not the places."""
        return Navigator(self.planner.actions, nav=self.nav.transferable(), surprise=self.surprise, skills=self.skills)

    def start(self, frame: np.ndarray, hud: dict | None = None) -> None:
        self.episode_errors, self.episode_curiosity = [], 0.0
        self.nav.begin(frame, hud)

    @property
    def mean_surprise(self) -> float:
        """How unpredictable this episode's moves were: 0 = all as expected, 1 = all new (or nothing measured yet)."""
        return sum(self.episode_errors) / len(self.episode_errors) if self.episode_errors else 1.0

    def advise(self, frame: np.ndarray, goal: str = "all", max_actions: int = 1) -> Advice:
        if self.skills is not None:
            context = {"here": self.nav.here, "actions": self.planner.actions}
            action, skill_name = self.skills.step(context)
            if action:
                return Advice([action], 0.8, None, f"skill:{skill_name}", [action])
        goal = goal if goal in GOALS or goal.startswith("to:") else "all"
        return self.planner.advise(frame, goal, max_actions)

    def learn(self, before: np.ndarray, action: str | None, after: np.ndarray, hud: dict | None = None,
              done: bool = False):
        before_place = self.nav.here
        res = self.nav.update(before, action, after, hud, done)
        if self.surprise is not None and action and before_place is not None:
            after_place = self.nav.here or "unknown"
            # curiosity is scored before the move is learned (afterwards it would no longer be a surprise)
            self.episode_curiosity += self.surprise.intrinsic_reward(before_place, action, after_place,
                                                                     actions=self.planner.actions)
            self.episode_errors.append(self.surprise.observe(before_place, action, after_place))
        return res

    def save(self) -> Path | None:
        if self.folder and self.card_stem:
            return navmap.save(self.nav, self.folder, self.card_stem)
        return None

    def summary(self) -> dict:
        """A small picture of what was learned (for the blackboard, the diary and the Mind view)."""
        nav = self.nav
        blocked = sum(e.best()[0] == "blocked" for acts in nav.edges.values() for e in acts.values())
        return {"places": len(nav.places), "blocked": blocked,
                "rewards": sorted(k for k, p in nav.places.items() if p.rewards),
                "goals": sorted(k for k, p in nav.places.items() if p.goal),
                "dangers": sorted(k for k, p in nav.places.items() if p.hazard),
                "here": nav.here, "scenes": len(nav.scenes), "episodes": nav.episodes,
                "knows_me": nav.me.known}
