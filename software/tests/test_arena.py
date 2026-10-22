from collections import deque

from practice_arena.game import BANNER, COIN_POINTS, DOOR_POINTS, TILE, ArenaGame


def reachable(game, start):
    seen, todo = {start}, deque([start])
    while todo:
        x, y = todo.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (x + dx, y + dy)
            if nxt not in seen and nxt not in game.walls:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def test_level_is_solvable():
    game = ArenaGame()
    cells = reachable(game, game.player)
    assert game.coins <= cells and game.key in cells and game.door in cells
    assert game.coins_left == 5


def test_walls_bump():
    game = ArenaGame()
    game.press("up")  # wall above the start
    assert game.player == (1, 1) and game.bumps == 1 and "wall" in game.message.lower()


def test_wasd_and_arrows_both_move():
    game = ArenaGame()
    game.press("right")
    game.press("d")
    assert game.player == (3, 1) and game.moves == 2


def test_coin_key_and_door():
    game = ArenaGame()
    game.coins = {(2, 1)}
    game.press("right")
    assert game.score == COIN_POINTS and game.coins_left == 0
    game.player = (8, 5)
    game.press("right")  # to (9,5)
    game.press("right")  # door, locked
    assert not game.won and game.player == (9, 5) and "locked" in game.message.lower()
    game.player = (6, 5)
    game.press("left")  # key at (5,5)
    assert game.has_key and game.key is None
    game.player = (9, 5)
    game.press("right")
    assert game.won and game.score == COIN_POINTS + DOOR_POINTS
    game.press("left")
    assert game.player == game.door  # no moving after winning


def test_reset():
    game = ArenaGame()
    game.press("right")
    game.reset()
    assert game.player == (1, 1) and game.moves == 0 and game.score == 0


def test_render_size_and_colours():
    game = ArenaGame()
    image = game.render()
    assert image.mode == "RGB" and image.size == game.pixel_size
    assert game.pixel_size == (12 * TILE, 9 * TILE + BANNER)
    px, py = game.player
    centre = (px * TILE + TILE // 2, BANNER + py * TILE + TILE // 2 + 10)
    r, g, b = image.getpixel(centre)
    assert b > 200 and r < 120  # the player is blue


def test_arabic_render():
    game_en = ArenaGame(language="english")
    game_ar = ArenaGame(language="arabic")
    image_en = game_en.render()
    image_ar = game_ar.render()
    # Same size
    assert image_en.size == image_ar.size
    # Banner regions differ (different text)
    banner_en = image_en.crop((0, 0, image_en.width, BANNER))
    banner_ar = image_ar.crop((0, 0, image_ar.width, BANNER))
    assert banner_en.tobytes() != banner_ar.tobytes()
