"""Test oracles for the arena pack: scripted policies that read the true game state.

They live in tests/ on purpose: hand-written solvers and route finders are allowed only as
test oracles, never as the buddy's own skills. The ladder uses them as its `scripted` brain, the baseline
that proves every level is winnable. Each oracle is also checked here.
"""
from __future__ import annotations

import random
from collections import deque

from arena_pack import brick_breaker as bb
from arena_pack.action_quest import ActionQuest
from arena_pack.box_push import LEVELS, MOVES, BoxPush, parse
from arena_pack.key_caves import KeyCaves
from arena_pack import chess_rules
from arena_pack.lane_dash import LaneDash, hit, move
from arena_pack.maze_chase import MazeChase, MOVES as MAZE_CHASE_MOVES
from arena_pack.mini_chess import SQUARE, MiniChess
from arena_pack.mini_fighter import MiniFighter
from arena_pack.mini_football import H, W, MiniFootball, toward
from arena_pack.slide_2048 import SIZE, Slide2048, slide_row
from arena_pack.ray_maze import FACINGS, RayMaze
from arena_pack.snake import GRID, Snake
from arena_pack import tiny_kingdom as tk
from arena_pack.tiny_kingdom import TinyKingdom
from arena_pack.top_racer import TopRacer
from arena_pack.turn_quest import PRICES, TurnQuest

# --------------------------------------------------------------------------- Slide 2048: small expectimax

ORDER = ("down", "left", "right", "up")
WEIGHTS = [[2 ** w for w in row] for row in ([1, 2, 3, 4], [8, 7, 6, 5], [9, 10, 11, 12], [16, 15, 14, 13])]


def _slid(board, key):
    grid = [row[:] for row in board]
    if key in ("up", "down"):
        grid = [list(col) for col in zip(*grid)]
    if key in ("right", "down"):
        grid = [row[::-1] for row in grid]
    grid = [slide_row(row)[0] for row in grid]
    if key in ("right", "down"):
        grid = [row[::-1] for row in grid]
    if key in ("up", "down"):
        grid = [list(col) for col in zip(*grid)]
    return grid


def _value(board) -> float:
    """Big tiles along a snake path ending in the bottom-left corner score highest."""
    return sum(board[y][x] * WEIGHTS[y][x] for y in range(SIZE) for x in range(SIZE))


def _after_spawn(board) -> float:
    """Average, over where the next 2 (or 4) appears, of the best move's value."""
    empty = [(x, y) for y in range(SIZE) for x in range(SIZE) if not board[y][x]]
    if not empty:
        return _value(board)
    total = 0.0
    for x, y in empty:
        for tile, chance in ((2, 0.9), (4, 0.1)):
            board[y][x] = tile
            moves = [_slid(board, k) for k in ORDER]
            total += chance * max((_value(b) for b in moves if b != board), default=-1.0)
            board[y][x] = 0
    return total / len(empty)


def policy_2048(game: Slide2048) -> str | None:
    if game.done:
        return None  # a good player stops once the game is over (the ladder counts actions after done)
    best = None
    for key in ORDER:
        board = _slid(game.board, key)
        if board != game.board:
            score = _after_spawn(board)
            if best is None or score > best[0]:
                best = (score, key)
    return best[1] if best else None


# --------------------------------------------------------------------------- Box Push: breadth-first solver

def solve_level(level) -> list[str]:
    """Shortest list of moves that puts every box on a goal (BFS over player + box positions)."""
    walls, boxes, goals, player = parse(level)
    start = (player, frozenset(boxes))
    came = {start: None}
    frontier = deque([start])
    while frontier:
        state = frontier.popleft()
        here, boxes_now = state
        if boxes_now == goals:
            plan = []
            while came[state] is not None:
                state, key = came[state]
                plan.append(key)
            return plan[::-1]
        for key, (dx, dy) in MOVES.items():
            step = (here[0] + dx, here[1] + dy)
            if step in walls:
                continue
            moved = boxes_now
            if step in boxes_now:
                beyond = (step[0] + dx, step[1] + dy)
                if beyond in walls or beyond in boxes_now:
                    continue
                moved = (boxes_now - {step}) | {beyond}
            nxt = (step, moved)
            if nxt not in came:
                came[nxt] = (state, key)
                frontier.append(nxt)
    return []


_PLANS = [solve_level(level) for level in LEVELS]


def policy_box_push(game: BoxPush) -> str | None:
    """Replays the solved plan for the current level (restarting it first if the player has wandered off)."""
    if game.is_won():
        return None
    plan = _PLANS[game.level_no]
    walls, boxes, goals, start = parse(LEVELS[game.level_no])
    done = getattr(game, "_oracle_steps", None)
    if done is None or done[0] != game.level_no or (done[1] == 0 and (game.player, game.boxes) != (start, boxes)):
        game._oracle_steps = [game.level_no, 0]
        if (game.player, game.boxes) != (start, boxes):
            return "r"
    level, i = game._oracle_steps
    game._oracle_steps = [level, i + 1]
    return plan[i] if i < len(plan) else "r"


# --------------------------------------------------------------------------- Snake: follow a loop through every cell

def hamiltonian_cycle(size: int = GRID) -> list[tuple[int, int]]:
    """A closed path through every cell: snake back and forth over columns 1..n-1, return up column 0."""
    path = []
    for y in range(size):
        xs = range(1, size) if y % 2 == 0 else range(size - 1, 0, -1)
        path += [(x, y) for x in xs]
    path += [(0, y) for y in range(size - 1, -1, -1)]
    return path


_CYCLE = hamiltonian_cycle()
_INDEX = {cell: i for i, cell in enumerate(_CYCLE)}
_TURN = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}


def policy_snake(game: Snake) -> str | None:
    """Walk the loop backwards (the snake starts heading that way). It can never hit itself or a wall."""
    if game.done:
        return None
    head = game.body[0]
    nxt = _CYCLE[(_INDEX[head] - 1) % len(_CYCLE)]
    return _TURN[(nxt[0] - head[0], nxt[1] - head[1])]


# --------------------------------------------------------------------------- any game: breadth-first search over states

def search_plan(game, actions, max_states: int = 200_000) -> list:
    """Shortest action list from the game's current state to a win, trying moves on snapshots.
    The real game is restored afterwards. States that lose are not explored."""
    def key():
        return tuple(getattr(game, name) for name in game.state_fields)
    real = game.snapshot()
    start = key()
    came, snaps, frontier = {start: None}, {start: real}, deque([start])
    goal = None
    while frontier and len(came) < max_states:
        state = frontier.popleft()
        game.restore(snaps[state])
        if game.is_won():
            goal = state
            break
        for action in actions:
            game.restore(snaps[state])
            if action:
                game.press(action)
            game.tick()
            if game.is_lost():
                continue
            nxt = key()
            if nxt not in came:
                came[nxt] = (state, action)
                snaps[nxt] = game.snapshot()
                frontier.append(nxt)
    game.restore(real)
    plan = []
    while goal is not None and came[goal] is not None:
        goal, action = came[goal]
        plan.append(action)
    return plan[::-1]


def replaying(actions):
    """A policy that follows search_plan, re-planning whenever the game is not where the plan expects."""
    memory = {"plan": [], "expect": None}

    def policy(game):
        if game.done:
            return None
        here = tuple(getattr(game, name) for name in game.state_fields)
        if not memory["plan"] or memory["expect"] != here:
            memory["plan"] = search_plan(game, actions)
        if not memory["plan"]:
            return None
        action = memory["plan"].pop(0)
        snap = game.snapshot()  # note where this action should lead
        if action:
            game.press(action)
        game.tick()
        memory["expect"] = tuple(getattr(game, name) for name in game.state_fields)
        game.restore(snap)
        return action
    return policy


def policy_key_caves(game: KeyCaves):
    if not hasattr(game, "_oracle"):
        game._oracle = replaying((None, "left", "right", "space", "z"))
    return game._oracle(game)


# --------------------------------------------------------------------------- Lane Dash: exact lookahead

def policy_lane_dash(game: LaneDash, horizon: int = 8) -> str | None:
    """Plan the next `horizon` ticks with the game's own move()/hit() rules: never crash, then most coins.
    Everything that can arrive within the horizon is already on the road (rows appear 12 ahead)."""
    if game.done:
        return None
    ahead = {}
    for lane, rows, kind in game.items:
        ahead.setdefault((lane, rows), []).append(kind)
    memo = {}

    def value(k, lane, air, slide):
        if k > horizon:
            return 0.0, None
        if (k, lane, air, slide) in memo:
            return memo[(k, lane, air, slide)]
        best = (float("-inf"), None)
        for key in (None, "left", "right", "up", "down"):
            nl, na, ns = move(lane, air, slide, key)
            things = ahead.get((nl, k), [])
            if any(hit(kind, na, ns) for kind in things):
                continue
            total = things.count("coin") + value(k + 1, nl, max(0, na - 1), max(0, ns - 1))[0]
            if total > best[0]:
                best = (total, key)
        memo[(k, lane, air, slide)] = best
        return best

    return value(1, game.lane, game.air, game.slide)[1]


# --------------------------------------------------------------------------- any game: short lookahead on snapshots

def lookahead(game, actions, depth, value):
    """The first action of the best `depth`-move sequence, trying every sequence on snapshots (the real game is
    restored). A sequence that loses or makes `value` return None is never chosen."""
    real = game.snapshot()

    def best(level):
        if level == 0 or game.done:
            return value(game)
        top = None
        here = game.snapshot()
        for action in actions:
            game.restore(here)
            if action:
                game.press(action)
            game.tick()
            score = None if game.is_lost() else best(level - 1)
            if score is not None and (top is None or score > top):
                top = score
        game.restore(here)
        return top

    choice, top = None, None
    for action in actions:
        game.restore(real)
        if action:
            game.press(action)
        game.tick()
        score = None if game.is_lost() else best(depth - 1)
        if score is not None and (top is None or score > top):
            choice, top = action, score
    game.restore(real)
    return choice


def policy_top_racer(game: TopRacer):
    """Two moves ahead: go as far as possible without ever spinning out (spins are what lose races).
    Two is enough because one brake press always gets you down to corner speed."""
    if game.done:
        return None
    spins = game.spins

    def value(g):
        return None if g.spins > spins else g.distance
    return lookahead(game, (None, "up", "down", "left", "right", "space"), 2, value)


# --------------------------------------------------------------------------- Mini Football: dribble, then beat the keeper

_STEP_KEY = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}


def policy_mini_football(game: MiniFootball) -> str | None:
    """Without the ball: go and win it. With it: dribble right around the red players; at the far end, get onto
    the goal row the keeper is furthest from (keepers are slow) and shoot from close range."""
    if game.done:
        return None
    me = game.ours[game.active()]
    if not game.have_ball:
        target = game.theirs[game.owner[1]] if game.owner and game.owner[0] == "cpu" else game.ball
        nxt = toward(tuple(me), tuple(target))
        return _STEP_KEY.get((nxt[0] - me[0], nxt[1] - me[1]))
    keeper = game.keepers[1]
    if me[0] >= W - 2:
        row = 3 if keeper >= 4 else 5
        if me[1] == row and abs(keeper - row) >= 2:
            return "space"
        if me[1] != row:
            return "up" if row < me[1] else "down"
        return "down" if row == 3 else "up"  # the keeper caught up: switch to the other side

    def danger(cell):
        return min(max(abs(cell[0] - t[0]), abs(cell[1] - t[1])) for t in game.theirs)
    options = []
    for (dx, dy), key in (((1, 0), "right"), ((0, -1), "up"), ((0, 1), "down")):
        cell = (me[0] + dx, me[1] + dy)
        if 0 <= cell[0] < W and 0 <= cell[1] < H:
            options.append((min(danger(cell), 2), key == "right", key))
    return max(options)[2]


# --------------------------------------------------------------------------- Mini Chess: look one move deeper than the bot

def square_click(game: MiniChess, sq: int) -> tuple:
    """A click on the centre of a board square, on the 0-1000 window grid."""
    width, height = game.pixel_size
    row, col = divmod(sq, 8)
    x, y = col * SQUARE + SQUARE / 2, height - 8 * SQUARE + row * SQUARE + SQUARE / 2
    return ("click", x / width * 1000, y / height * 1000)


def policy_mini_chess(game: MiniChess, depth: int = 3):
    """Search 3 moves ahead (the bot looks 2), then click the piece and its target square."""
    if game.done:
        return None
    plan = getattr(game, "_oracle_move", None)
    if plan and game.picked == plan[0]:
        game._oracle_move = None
        return square_click(game, plan[1])
    best = sorted(chess_rules.best_moves(game.pos, depth))
    if not best:
        return None
    game._oracle_move = best[0]
    return square_click(game, best[0][0])


# --------------------------------------------------------------------------- Turn Quest: grind, gear up, then the dragon

def _menu(game: TurnQuest, index: int) -> str:
    """Move the menu cursor to `index` (the short way round), then Enter."""
    if game.cursor == index:
        return "enter"
    size = len(game.options())
    return "down" if (index - game.cursor) % size <= size // 2 else "up"


def policy_turn_quest(game: TurnQuest) -> str | None:
    if game.done:
        return None
    h = game.hero
    if game.screen == "town":
        if h["hp"] < 0.6 * h["max_hp"] and h["gold"] >= PRICES["inn"]:
            return _menu(game, 2)  # rest
        if not h["sword"] and h["gold"] >= PRICES["sword"]:
            return _menu(game, 1)  # shop: sword
        if h["sword"] and h["potions"] < 3 and h["gold"] >= PRICES["potion"] + PRICES["inn"]:
            return _menu(game, 1)  # shop: potions
        if h["level"] >= 4 and h["sword"] and h["potions"] >= 2 and h["hp"] == h["max_hp"]:
            return _menu(game, 3)  # the dragon
        return _menu(game, 0)  # a slime
    if game.screen == "shop":
        if not h["sword"] and h["gold"] >= PRICES["sword"]:
            return _menu(game, 1)
        if h["sword"] and h["potions"] < 3 and h["gold"] >= PRICES["potion"]:
            return _menu(game, 0)
        return _menu(game, 2)
    if h["hp"] < 0.4 * h["max_hp"] and h["potions"]:
        return _menu(game, 2)
    if game.enemy["kind"] == "dragon" and h["mp"] >= 5:
        return _menu(game, 1)
    return _menu(game, 0)


# --------------------------------------------------------------------------- Tiny Kingdom: economy first, then a defence line

def _grid_click(game, px: float, py: float) -> tuple:
    width, height = game.pixel_size
    return ("click", px / width * 1000, py / height * 1000)


def _kingdom_button(game: TinyKingdom, kind: str) -> tuple:
    width = tk.COLS * tk.TILE / len(tk.BUTTONS)
    return _grid_click(game, tk.BUTTONS.index(kind) * width + width / 2, game.pixel_size[1] - tk.BAR / 2)


def _kingdom_square(game: TinyKingdom, col: int, row: int) -> tuple:
    top = game.pixel_size[1] - tk.BAR - tk.ROWS * tk.TILE
    return _grid_click(game, col * tk.TILE + tk.TILE / 2, top + row * tk.TILE + tk.TILE / 2)


KINGDOM_PLAN = ([("farm", (1, r)) for r in (1, 3, 5)] + [("tower", (2, r)) for r in (1, 3, 5)]
                + [("farm", (1, r)) for r in (0, 2, 4, 6)] + [("soldier", (3, r)) for r in range(7)]
                + [("tower", (2, r)) for r in (0, 2, 4, 6)] + [("tower", (4, r)) for r in (1, 3, 5)])


def policy_tiny_kingdom(game: TinyKingdom):
    """Farms by the wall, towers behind a line of soldiers; rebuild anything lost. End the turn when broke."""
    if game.done:
        return None
    busy = {(e[0], e[1]) for e in game.enemies}
    for kind, square in KINGDOM_PLAN:
        if square in game.buildings or square in busy:
            continue
        if game.gold < tk.BUILD[kind]:
            return "enter"
        if game.chosen != kind:
            return _kingdom_button(game, kind)
        return _kingdom_square(game, *square)
    return "enter"


# --------------------------------------------------------------------------- Ray Maze: line up a target, shoot, refill, exit

def _target_ahead(game: RayMaze) -> bool:
    dx, dy = FACINGS[game.facing]
    x, y = game.x + dx, game.y + dy
    while not game._wall(x, y):
        if (x, y) in game.targets:
            return True
        x, y = x + dx, y + dy
    return False


def _walk_to(game: RayMaze, goal) -> list:
    """Shortest walk (BFS over square + facing, on snapshots) to a state where goal(game) is true."""
    real = game.snapshot()
    key = lambda: (game.x, game.y, game.facing)  # noqa: E731
    came, snaps, frontier = {key(): None}, {key(): real}, deque([key()])
    found = None
    while frontier:
        state = frontier.popleft()
        game.restore(snaps[state])
        if goal(game):
            found = state
            break
        for action in ("w", "s", "a", "d", "left", "right"):
            game.restore(snaps[state])
            game.press(action)
            nxt = key()
            if nxt not in came:
                came[nxt], snaps[nxt] = (state, action), game.snapshot()
                frontier.append(nxt)
    game.restore(real)
    plan = []
    while found is not None and came[found] is not None:
        found, action = came[found]
        plan.append(action)
    return plan[::-1]


def policy_ray_maze(game: RayMaze):
    if game.done:
        return None
    if game.hits >= 5:
        plan = _walk_to(game, lambda g: g.escaped)
    elif game.ammo == 0:
        boxes = len(game.ammo_boxes)
        plan = _walk_to(game, lambda g: len(g.ammo_boxes) < boxes)
    elif _target_ahead(game):
        return "space"
    else:
        plan = _walk_to(game, _target_ahead)
    return plan[0] if plan else None


# --------------------------------------------------------------------------- Brick Breaker: be where the ball comes down

def _landing_x(game) -> float | None:
    """Where the falling ball crosses the paddle line, bouncing off the side walls (bricks ignored)."""
    (x, y), (vx, vy) = game.ball, game.velocity
    if vy <= 0:
        return None
    x = (x + vx * (bb.PADDLE_Y - y) / vy) % (2 * bb.W)
    return 2 * bb.W - x if x > bb.W else x


def policy_brick_breaker(game):
    """Launch, then move the paddle under the ball's landing spot, a little off-centre towards the remaining bricks
    (so the ball doesn't get stuck going straight up and down)."""
    if game.done:
        return None
    if not game.launched:
        return "space"
    target = _landing_x(game)
    target = game.ball[0] if target is None else target
    if game.bricks:
        middle = sum(b[0] for b in game.bricks) / len(game.bricks) + bb.BRICK_W / 2
        target += 12 if middle < target else -12
    if abs(game.paddle - target) <= bb.PADDLE_STEP / 2:
        return None
    return "right" if target > game.paddle else "left"


# --------------------------------------------------------------------------- Mini Fighter: two moves ahead

def policy_mini_fighter(game: MiniFighter):
    """Two moves ahead on snapshots: rounds count most, then damage dealt minus (a little more) damage taken."""
    if game.done:
        return None

    def value(g):
        return 1000 * (g.wins[0] - g.wins[1]) + (100 - g.cpu["hp"]) - 1.3 * (100 - g.you["hp"])
    return lookahead(game, (None, "left", "right", "a", "s", "d", "f"), 2, value)


# --------------------------------------------------------------------------- Action Quest: two moves ahead (sees the red squares)

def policy_action_quest(game: ActionQuest):
    """Two moves ahead on snapshots: hurt the boss, avoid being hurt (worth more), keep potions and some stamina."""
    if game.done:
        return None

    def value(g):
        return (3 * (120 - max(0, g.boss["hp"])) - 4 * (50 - g.hero["hp"]) + 10 * g.hero["potions"]
                + 0.05 * g.hero["stamina"])
    return lookahead(game, (None, "w", "a", "s", "d", "j", "k", "l", "e"), 2, value)



# --------------------------------------------------------------------------- Maze Chase: BFS path to coins, key, and exit

def policy_maze_chase(game: MazeChase) -> str | None:
    if game.done:
        return None
    if game.coins:
        targets = set(game.coins)
    elif not game.has_key and game.key_pos:
        targets = {game.key_pos}
    else:
        targets = {game.door_pos}

    queue = deque([(game.player, [])])
    visited = {game.player}
    best_path = None
    while queue:
        curr, path = queue.popleft()
        if curr in targets:
            best_path = path
            break
        for k, (dx, dy) in MAZE_CHASE_MOVES.items():
            nxt = (curr[0] + dx, curr[1] + dy)
            if nxt not in game.walls and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [k]))
    if best_path:
        return best_path[0]
    return "up"

SCRIPTED = {"slide_2048": policy_2048, "box_push": policy_box_push, "maze_chase": policy_maze_chase, "snake": policy_snake, "key_caves": policy_key_caves,
            "lane_dash": policy_lane_dash, "top_racer": policy_top_racer,
            "mini_football": policy_mini_football, "mini_chess": policy_mini_chess,
            "turn_quest": policy_turn_quest, "tiny_kingdom": policy_tiny_kingdom,
            "ray_maze": policy_ray_maze, "brick_breaker": policy_brick_breaker,
            "mini_fighter": policy_mini_fighter, "action_quest": policy_action_quest}


def random_policy(keys: tuple[str, ...], seed: int):
    """Random key presses; for a mouse-only game (no keys), random clicks on the 0-1000 grid."""
    rng = random.Random(seed)
    if not keys:
        return lambda game: ("click", rng.randrange(1000), rng.randrange(1000))
    return lambda game: rng.choice(keys)


# --------------------------------------------------------------------------- the oracles themselves are checked

def test_box_push_solver_solves_every_level_and_the_plan_works_in_the_game():
    game = BoxPush()
    for number, plan in enumerate(_PLANS):
        assert plan, f"level {number + 1} has no solution"
        for key in plan:
            game.press(key)
    assert game.is_won() and game.score == 300


def test_hamiltonian_cycle_visits_every_cell_once_and_closes():
    cycle = hamiltonian_cycle()
    assert len(cycle) == len(set(cycle)) == GRID * GRID
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_maze_chase_oracle_wins():
    for seed in (0, 1, 2, 3, 4):
        game = MazeChase(seed=seed)
        game.observe = False
        while not game.done and game.ticks < 400:
            action = policy_maze_chase(game)
            if action:
                game.step(action)
        assert game.is_won(), f"Maze chase failed on seed {seed}"
        assert not game.is_lost()
