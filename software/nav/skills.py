"""Reusable temporal macro-actions: clear_corridor, dodge_hazard, fetch_key, push_align.
/ مهارات وإجراءات كبرى قابلة لإعادة الاستخدام: عبور الممر، تفادي الخطر، جلب المفتاح، محاذاة الدفع."""
from __future__ import annotations

from collections import deque
from typing import Any, Callable, Sequence

MOVES = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _bfs(start: tuple[int, int], target: tuple[int, int],
         walls: set[tuple[int, int]]) -> list[str]:
    """Finds shortest sequence of moves from start to target."""
    if start == target:
        return []
    queue = deque([(start, [])])
    visited = {start}
    while queue:
        curr, path = queue.popleft()
        if curr == target:
            return path
        for k, (dx, dy) in MOVES.items():
            nxt = (curr[0] + dx, curr[1] + dy)
            if nxt not in walls and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [k]))
    return []


class Skill:
    """Base class for temporal macro-actions (options)."""

    name: str = "base"

    def can_apply(self, context: dict[str, Any]) -> bool:
        raise NotImplementedError

    def step(self, context: dict[str, Any]) -> tuple[str | None, bool]:
        """Returns (action, done). When done is True, the skill has completed."""
        raise NotImplementedError

    def reset(self) -> None:
        pass


class ClearCorridor(Skill):
    """Macro-action: traverses an unobstructed straight corridor until reaching
    a junction, obstacle, or open room."""

    name = "clear_corridor"

    def __init__(self):
        self.last_move: str | None = None

    def reset(self) -> None:
        self.last_move = None

    def can_apply(self, context: dict[str, Any]) -> bool:
        player = context.get("player")
        walls = context.get("walls", set())
        if not player or not isinstance(walls, (set, frozenset)):
            return False

        px, py = player
        open_dirs = [k for k, (dx, dy) in MOVES.items() if (px + dx, py + dy) not in walls]

        # In a corridor if exactly 2 directions are open (one forward, one backward)
        # or if we are already traversing one direction
        if len(open_dirs) == 2:
            d1, d2 = open_dirs
            return OPPOSITE.get(d1) == d2
        return False

    def step(self, context: dict[str, Any]) -> tuple[str | None, bool]:
        player = context.get("player")
        walls = context.get("walls", set())
        if not player:
            return None, True

        px, py = player
        open_dirs = [k for k, (dx, dy) in MOVES.items() if (px + dx, py + dy) not in walls]

        # Don't turn back unless it's the only option
        if self.last_move and len(open_dirs) == 2:
            forward = [k for k in open_dirs if k != OPPOSITE.get(self.last_move)]
            if forward:
                move = forward[0]
                self.last_move = move
                # Check if next step is still a corridor
                nx, ny = px + MOVES[move][0], py + MOVES[move][1]
                next_open = [k for k, (dx, dy) in MOVES.items() if (nx + dx, ny + dy) not in walls]
                is_still_corridor = len(next_open) == 2 and OPPOSITE.get(next_open[0]) == next_open[1]
                return move, not is_still_corridor

        if open_dirs:
            move = open_dirs[0]
            self.last_move = move
            return move, True

        return None, True


class DodgeHazard(Skill):
    """Emergency reactive macro-action: evades nearby hazards or chasers
    by moving towards the safest adjacent tile."""

    name = "dodge_hazard"

    def can_apply(self, context: dict[str, Any]) -> bool:
        player = context.get("player")
        hazard = context.get("chaser") or context.get("hazard")
        if not player or not hazard:
            return False
        # Danger zone: Manhattan distance <= 2
        dist = abs(player[0] - hazard[0]) + abs(player[1] - hazard[1])
        return dist <= 2

    def step(self, context: dict[str, Any]) -> tuple[str | None, bool]:
        player = context.get("player")
        hazard = context.get("chaser") or context.get("hazard")
        walls = context.get("walls", set())
        if not player or not hazard:
            return None, True

        px, py = player
        hx, hy = hazard

        best_move = None
        best_dist = -1

        for k, (dx, dy) in MOVES.items():
            nx, ny = px + dx, py + dy
            if (nx, ny) not in walls:
                dist = abs(nx - hx) + abs(ny - hy)
                if dist > best_dist:
                    best_dist = dist
                    best_move = k

        done = best_dist >= 3 or best_move is None
        return best_move, done


class FetchKey(Skill):
    """Macro-action: finds and executes the shortest route to the key."""

    name = "fetch_key"

    def __init__(self):
        self.plan: list[str] = []

    def reset(self) -> None:
        self.plan.clear()

    def can_apply(self, context: dict[str, Any]) -> bool:
        has_key = context.get("has_key", False)
        key_pos = context.get("key") or context.get("key_pos")
        return (not has_key) and (key_pos is not None)

    def step(self, context: dict[str, Any]) -> tuple[str | None, bool]:
        player = context.get("player")
        key_pos = context.get("key") or context.get("key_pos")
        walls = context.get("walls", set())

        if not player or not key_pos:
            return None, True

        if player == key_pos or context.get("has_key"):
            return None, True

        if not self.plan:
            self.plan = _bfs(player, key_pos, walls)

        if not self.plan:
            return None, True

        move = self.plan.pop(0)
        done = len(self.plan) == 0
        return move, done


class PushAlign(Skill):
    """Macro-action: maneuvers player behind a target box/object in order to
    push it in the intended direction."""

    name = "push_align"

    def __init__(self):
        self.plan: list[str] = []

    def reset(self) -> None:
        self.plan.clear()

    def can_apply(self, context: dict[str, Any]) -> bool:
        box = context.get("target_box")
        direction = context.get("push_direction")
        return box is not None and direction in MOVES

    def step(self, context: dict[str, Any]) -> tuple[str | None, bool]:
        player = context.get("player")
        box = context.get("target_box")
        direction = context.get("push_direction")
        walls = context.get("walls", set())

        if not player or not box or direction not in MOVES:
            return None, True

        bx, by = box
        pdx, pdy = MOVES[direction]
        # Required stand position is opposite to push direction
        stand_pos = (bx - pdx, by - pdy)

        if player == stand_pos:
            # At position! Execute the push move
            return direction, True

        # Navigate to stand position
        if not self.plan:
            # Temporarily treat box as a wall when routing behind it
            box_walls = set(walls) | {box}
            self.plan = _bfs(player, stand_pos, box_walls)
            if not self.plan:
                # If path blocked with box as wall, try general BFS
                self.plan = _bfs(player, stand_pos, walls)

        if self.plan:
            move = self.plan.pop(0)
            return move, False

        return None, True


class SkillLibrary:
    """Manages and arbitrates macro-actions based on context priority."""

    def __init__(self, skills: Sequence[Skill] | None = None):
        if skills is None:
            self.skills: list[Skill] = [
                DodgeHazard(),  # Emergency priority
                PushAlign(),    # Object manipulation
                FetchKey(),     # Goal acquisition
                ClearCorridor(),  # Navigation efficiency
            ]
        else:
            self.skills = list(skills)
        self.active_skill: Skill | None = None

    def reset(self) -> None:
        self.active_skill = None
        for s in self.skills:
            s.reset()

    def select_skill(self, context: dict[str, Any]) -> Skill | None:
        """Finds highest-priority applicable skill."""
        # If active skill is still valid, continue
        if self.active_skill and self.active_skill.can_apply(context):
            return self.active_skill

        for s in self.skills:
            if s.can_apply(context):
                self.active_skill = s
                return s

        self.active_skill = None
        return None

    def step(self, context: dict[str, Any]) -> tuple[str | None, str | None]:
        """Returns (action, skill_name) or (None, None)."""
        skill = self.select_skill(context)
        if not skill:
            return None, None
        action, done = skill.step(context)
        skill_name = skill.name
        if done:
            self.active_skill = None
            skill.reset()
        return action, skill_name
