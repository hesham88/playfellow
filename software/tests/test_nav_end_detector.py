"""P0 follow-up (A): the map guesses, from the pictures alone, that a step ENDED the game, so the next run on the same map
treats that arrow as 'ends the game' (never walked through before the pickups). It is learn-only: it changes the map and
nothing else."""
from __future__ import annotations

import numpy as np

from nav.map import Edge, NavMap

H, W = 40, 60


def pic(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def nav_with_arrow():
    nav = NavMap()
    nav.edges["0:1:1"] = {"right": Edge(tries=1, outcomes={"0:2:1": 1})}
    return nav


def step(nav, before, after, action, new, reward=True, here="0:1:1"):
    nav._watch_for_the_end(before, after, action, here, new, reward)


def test_a_moving_step_with_a_hud_change_then_a_frozen_picture_marks_the_end():
    nav = nav_with_arrow()
    a, b = pic(1), pic(2)
    step(nav, a, b, "right", "0:2:1")  # moved, the HUD changed
    for key in ("up", "down"):
        step(nav, b, b.copy(), key, "0:2:1", reward=False, here="0:2:1")
        assert nav.ended == []  # not enough frozen steps yet
    step(nav, b, b.copy(), "left", "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == [("0:1:1", "right")] and nav._restart_pending
    edge = nav.edges["0:1:1"]["right"]
    assert edge.good_end == 1 and nav.places["0:2:1"].goal == 1
    assert nav.learned_road("0:1:1", "right", edge) == ("0:2:1", 1.0, 1.0, True)
    assert nav.take_ended() == [("0:1:1", "right")] and nav.take_ended() == []


def test_two_frozen_steps_are_not_enough():
    nav = nav_with_arrow()
    a, b = pic(1), pic(2)
    step(nav, a, b, "right", "0:2:1")
    for _ in range(2):
        step(nav, b, b.copy(), "up", "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == [] and not nav.edges["0:1:1"]["right"].good_end


def test_one_key_pressed_again_and_again_is_a_wall_not_the_end():
    nav = nav_with_arrow()
    a, b = pic(1), pic(2)
    step(nav, a, b, "right", "0:2:1")  # a coin: moved, the HUD changed
    for _ in range(6):
        step(nav, b, b.copy(), "left", "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == [] and not nav.edges["0:1:1"]["right"].good_end


def test_the_frozen_picture_asks_for_a_key_not_pressed_yet_and_two_keys_mark_the_end():
    nav = nav_with_arrow()
    keys = ["up", "down", "left", "right"]
    a, b = pic(1), pic(2)
    assert nav.frozen_key(keys) is None  # not watching yet
    step(nav, a, b, "right", "0:2:1")
    assert nav.frozen_key(keys) is None  # no frozen step yet
    step(nav, b, b.copy(), "left", "0:2:1", reward=False, here="0:2:1")
    assert nav.frozen_key(keys) == "up"  # the first key not pressed at this picture
    step(nav, b, b.copy(), "left", "0:2:1", reward=False, here="0:2:1")
    step(nav, b, b.copy(), "left", "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == []  # three steps but one key
    step(nav, b, b.copy(), nav.frozen_key(keys), "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == [("0:1:1", "right")]
    assert nav.frozen_key(keys) is None


def test_one_change_of_the_picture_cancels_the_guess():
    nav = nav_with_arrow()
    a, b, c = pic(1), pic(2), pic(3)
    step(nav, a, b, "right", "0:2:1")
    step(nav, b, b.copy(), "up", "0:2:1", reward=False, here="0:2:1")
    step(nav, b, c, "down", "0:2:1", reward=False, here="0:2:1")  # something moved: the game is still going
    for key in ("left", "up", "down", "right"):
        step(nav, c, c.copy(), key, "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == [] and not nav.edges["0:1:1"]["right"].good_end


def test_a_menu_or_pause_screen_is_not_the_end():
    """A pause menu freezes the picture after a key, but no step moved me onto a new square with a HUD change before it."""
    nav = nav_with_arrow()
    game_frame, menu = pic(1), pic(9)
    step(nav, game_frame, menu, "escape", None, reward=False)  # the menu opened: I did not move
    for key in ("up", "down", "left", "right", "space"):
        step(nav, menu, menu.copy(), key, None, reward=False)
    assert nav.ended == []
    # and a step that stayed on the same square never starts a guess, even with a HUD change (a score ticking up)
    step(nav, game_frame, menu, "right", "0:1:1", reward=True)
    for key in ("up", "down", "left"):
        step(nav, menu, menu.copy(), key, "0:1:1", reward=False)
    assert nav.ended == []


def test_a_pickup_followed_by_normal_play_is_not_the_end():
    nav = nav_with_arrow()
    frames = [pic(i) for i in range(10, 16)]
    step(nav, frames[0], frames[1], "right", "0:2:1")
    for i, key in enumerate(("up", "down", "left", "right")):
        step(nav, frames[1 + i], frames[2 + i], key, "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == []


def test_a_new_episode_forgets_a_half_finished_guess():
    nav = nav_with_arrow()
    a, b = pic(1), pic(2)
    step(nav, a, b, "right", "0:2:1")
    step(nav, b, b.copy(), "up", "0:2:1", reward=False, here="0:2:1")
    nav._end_watch_state, nav._end_steps, nav._end_keys = None, 0, set()  # what begin() does
    for key in ("down", "left"):
        step(nav, b, b.copy(), key, "0:2:1", reward=False, here="0:2:1")
    assert nav.ended == []


def test_the_marked_arrow_is_in_the_road_grid_as_no_walkable_road():
    nav = nav_with_arrow()
    nav.sync_grid()
    assert nav.grid.slots
    a, b = pic(1), pic(2)
    step(nav, a, b, "right", "0:2:1")
    for key in ("up", "down", "left"):
        step(nav, b, b.copy(), key, "0:2:1", reward=False, here="0:2:1")
    nav.sync_grid()
    assert not nav.grid.slots  # an ends-the-game road is never walked through, so it is not in the walking grid


def test_a_picture_that_moves_again_after_an_end_is_a_new_game():
    nav = nav_with_arrow()
    a, b, c = pic(1), pic(2), pic(3)
    step(nav, a, b, "right", "0:2:1")
    for key in ("up", "down", "left"):
        step(nav, b, b.copy(), key, "0:2:1", reward=False, here="0:2:1")
    assert nav._restart_pending
    nav.collected = {"0:1:1"}
    episodes = nav.episodes
    nav.update(b, "up", b.copy(), {})  # still frozen: still the same ended game
    assert nav.episodes == episodes and nav.collected
    nav.update(b, "up", c, {})  # the picture moves: a new game began
    assert nav.episodes == episodes + 1 and nav.collected == set() and not nav._restart_pending
    assert nav.edges["0:1:1"]["right"].good_end == 1  # what it learned stays


def test_the_end_can_be_noticed_when_the_step_left_no_arrow_and_hid_me():
    nav = NavMap()
    nav.edges["0:1:1"] = {}
    a, b = pic(1), pic(2)
    nav.scene = 0
    nav.target_square = lambda here, action: (2, 1)  # where the key should take me
    step(nav, a, b, "right", None)  # the door square hid me: no square seen after the step
    for key in ("up", "down", "left"):
        step(nav, b, b.copy(), key, None, reward=False, here="0:1:1")
    assert nav.ended == [("0:1:1", "right")]
    assert nav.learned_road("0:1:1", "right", nav.edges["0:1:1"]["right"]) == ("0:2:1", 1.0, 1.0, True)
