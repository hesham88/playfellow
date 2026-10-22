"""The arena pack games (T40): shared base, rules, headless runs, scripted wins, random does worse, card keys only."""
import numpy as np
import pytest

from arena_pack.action_quest import ActionQuest
from arena_pack.base import TITLE_PREFIX, run_window
from arena_pack.box_push import BoxPush
from arena_pack.brick_breaker import BrickBreaker
from arena_pack.key_caves import KeyCaves
from arena_pack.lane_dash import LaneDash
from arena_pack.maze_chase import MazeChase
from arena_pack.mini_chess import MiniChess
from arena_pack.mini_fighter import MiniFighter
from arena_pack.mini_football import MiniFootball
from arena_pack.slide_2048 import Slide2048, slide_row
from arena_pack.ray_maze import RayMaze
from arena_pack.snake import Snake
from arena_pack.tiny_kingdom import TinyKingdom
from arena_pack.top_racer import TopRacer
from arena_pack.turn_quest import TurnQuest
from controls import load_card
from games import detect_games, needs_online_warning
from test_arena_pack_oracles import SCRIPTED, random_policy

GAMES = {"slide_2048": Slide2048, "box_push": BoxPush, "maze_chase": MazeChase, "snake": Snake, "key_caves": KeyCaves, "lane_dash": LaneDash, "top_racer": TopRacer, "mini_football": MiniFootball, "mini_chess": MiniChess, "turn_quest": TurnQuest, "tiny_kingdom": TinyKingdom, "ray_maze": RayMaze, "brick_breaker": BrickBreaker, "mini_fighter": MiniFighter, "action_quest": ActionQuest}
CONTROLS = __import__("pathlib").Path(__file__).resolve().parent.parent / "controls"


def play(game, policy, max_ticks=None):
    game.observe = False  # the policies read the game state; skip drawing every tick
    truncated = False
    limit = max_ticks or game.max_ticks
    while not game.done and not truncated and game.ticks < limit:
        _, _, _, truncated, _ = game.step(policy(game))
    return game


@pytest.fixture
def headless(monkeypatch):
    for name, value in {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}.items():
        monkeypatch.setenv(name, value)


# --------------------------------------------------------------------------- the shared base

@pytest.mark.parametrize("name", GAMES)
def test_title_has_the_arena_prefix_and_games_py_accepts_it(name):
    title = GAMES[name]().title
    assert title.startswith(TITLE_PREFIX)
    assert detect_games([title]) == [title] and not needs_online_warning(title)


@pytest.mark.parametrize("name", GAMES)
def test_gym_style_reset_and_step_return_pixels_and_progress(name):
    game = GAMES[name](seed=3)
    obs = game.reset(3)
    width, height = game.pixel_size
    assert obs.shape == (height, width, 3) and obs.dtype == np.uint8
    obs2, reward, terminated, truncated, info = game.step(game.keys[0] if game.keys else ("click", 500, 500))
    assert obs2.shape == obs.shape and isinstance(reward, float)
    assert terminated is False and truncated is False
    assert info["ticks"] == 1 and isinstance(info["progress"], dict) and info["progress"]


def test_observe_off_skips_the_picture_but_not_the_game():
    fast, full = Snake(seed=4), Snake(seed=4)
    fast.observe = False
    for action in ("up", None, "left", None):
        a, b = fast.step(action), full.step(action)
        assert a[0] is None and b[0] is not None and a[1:] == b[1:]


@pytest.mark.parametrize("name", GAMES)
def test_same_seed_same_game(name):
    actions = ["up", "left", None, "down", "right", "left"]
    runs = []
    for _ in range(2):
        game = GAMES[name](seed=7)
        frames = [game.step(a)[0] for a in actions]
        runs.append((frames, game.score, game.progress()))
    assert all(np.array_equal(a, b) for a, b in zip(runs[0][0], runs[1][0]))
    assert runs[0][1:] == runs[1][1:]


@pytest.mark.parametrize("name", GAMES)
def test_runs_headless_in_a_real_window(name, headless):
    game = run_window(GAMES[name], "test", argv=["--seed", "1", "--speed", "50", "--lang", "arabic"], frames=3)
    assert game.language == "arabic" and game.seed == 1
    if name == "snake":
        assert game.ticks > 0  # the real-time clock ran (×50 speed)


@pytest.mark.parametrize("name", GAMES)
def test_card_has_genre_and_exactly_the_game_keys(name):
    card = load_card(CONTROLS / f"arena_{name}.yaml")
    import yaml
    assert yaml.safe_load((CONTROLS / f"arena_{name}.yaml").read_text(encoding="utf-8"))["genre"]
    card_keys = {key for action in card.actions.values() for key in action.keys}
    assert card_keys == set(GAMES[name].keys)
    assert all(action.says_ar for action in card.actions.values()) and card.name_ar and card.game_hint_ar


@pytest.mark.parametrize("name", GAMES)
def test_keys_not_on_the_card_do_nothing(name):
    game = GAMES[name](seed=2)
    before = (game.observation().tobytes(), game.score, game.progress())
    for key in ("space", "f12", "escape", "a", "enter", "q", "tab"):
        if key not in game.keys:
            game.press(key)
    assert (game.observation().tobytes(), game.score, game.progress()) == before


# --------------------------------------------------------------------------- scripted wins, random does worse

# Fewer seeds where a run is slow (the 2048 lookahead, Snake's long safe loop); the ladder plays 5 seeds each.
@pytest.mark.parametrize("name", GAMES)
def test_scripted_policy_wins(name):
    game = play(GAMES[name](seed=0), SCRIPTED[name])
    assert game.is_won(), f"{name}: {game.progress()}"


@pytest.mark.slow
@pytest.mark.parametrize("name", GAMES)
def test_scripted_policy_wins_on_every_ladder_seed(name):
    for seed in range(1, 5):
        game = play(GAMES[name](seed=seed), SCRIPTED[name])
        assert game.is_won(), f"{name} seed {seed}: {game.progress()}"


# How far each game is from its win (the scripted policy reaches 0).
SHORT_OF_WIN = {"slide_2048": lambda g: 256 - g.best_tile, "box_push": lambda g: 3 - g.score // 100, "maze_chase": lambda g: 0 if g.is_won() else len(g.coins) + (0 if g.has_key else 1) + 1,
                "snake": lambda g: 20 - len(g.body),
                "key_caves": lambda g: 3 - g.has_boots - g.has_key - g.door_open,
                "lane_dash": lambda g: max(600 - g.clock, 30 - g.coins),
                "top_racer": lambda g: 0 if g.is_won() else 3 * 42 * 4 - g.distance,
                "mini_football": lambda g: 2 - (g.goals[0] - g.goals[1]),
                "mini_chess": lambda g: 0 if g.is_won() else 30 - g.moves_made,
                "turn_quest": lambda g: 0 if g.is_won() else 1,
                "tiny_kingdom": lambda g: 0 if g.is_won() else 10 - g.waves_sent + len(g.enemies),
                "ray_maze": lambda g: 0 if g.is_won() else 6 - g.hits,
                "brick_breaker": lambda g: 0 if g.is_won() else 2 - g.level_no + len(g.bricks),
                "mini_fighter": lambda g: 2 - g.wins[0],
                "action_quest": lambda g: max(0, g.boss["hp"])}


@pytest.mark.parametrize("name", GAMES)
def test_random_policy_does_worse_than_scripted(name):
    game = play(GAMES[name](seed=0), random_policy(GAMES[name].keys, 0), max_ticks=150)
    assert not game.is_won() and SHORT_OF_WIN[name](game) > 0


@pytest.mark.slow
@pytest.mark.parametrize("name", GAMES)
def test_random_policy_never_wins_on_more_seeds(name):
    for seed in range(1, 5):
        game = play(GAMES[name](seed=seed), random_policy(GAMES[name].keys, seed), max_ticks=600)
        assert not game.is_won() and SHORT_OF_WIN[name](game) > 0


# --------------------------------------------------------------------------- rules

def test_2048_slide_joins_each_pair_once():
    assert slide_row([2, 2, 2, 2]) == ([4, 4, 0, 0], 8)
    assert slide_row([2, 0, 2, 4]) == ([4, 4, 0, 0], 4)
    assert slide_row([4, 4, 8, 0]) == ([8, 8, 0, 0], 8)


def test_2048_a_move_that_changes_nothing_adds_no_tile():
    game = Slide2048()
    game.board = [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    game.on_key("left")
    assert game.board == [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] and game.moves == 0
    game.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert game.is_lost() and not game.is_won()
    game.board[0][0] = 256
    assert game.is_won() and not game.is_lost()


def test_box_push_rules_walls_two_boxes_and_restart():
    from arena_pack.box_push import parse
    game = BoxPush()  # level 1: player (3,3), box (2,2), goal (4,2)
    start = (game.player, set(game.boxes))
    for key in ("left", "left", "left"):  # the third press walks into the wall at x=0
        game.press(key)
    assert game.player == (1, 3) and game.moves == 2
    game.press("up"), game.press("right")
    assert game.pushes == 1 and (3, 2) in game.boxes  # pushed one square right
    game.press("r")
    assert (game.player, set(game.boxes)) == start
    game.walls, game.boxes, game.goals, game.player = parse(("#####", "#@$$.", "#####"))
    boxes = set(game.boxes)
    game.press("right")  # two boxes in a row can't be pushed
    assert game.player == (1, 1) and game.boxes == boxes


def test_snake_cannot_reverse_grows_on_food_and_crashes_on_a_wall():
    game = Snake()
    head = game.body[0]
    game.step("left")  # straight back is ignored: it keeps going right
    assert game.body[0] == (head[0] + 1, head[1])
    game.food = (game.body[0][0] + 1, game.body[0][1])
    game.step(None)
    assert len(game.body) == 4 and game.score == 10
    for _ in range(10):
        game.step(None)
    assert game.is_lost() and not game.is_won()
    frozen = game.step("up")
    assert frozen[2] is True  # terminated: nothing moves after the end
