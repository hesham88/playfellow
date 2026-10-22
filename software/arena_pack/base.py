"""The shared base for the arena pack games. / الأساس المشترك لألعاب حزمة الساحة.

Every game is pure rules plus a picture (like the Practice Arena), so it runs with or without a window:
  - a window title "PlayFellow Arena · <name>" and a HUD with the score and progress;
  - a fixed seed (the same seed always gives the same game);
  - is_won() / is_lost();
  - a Gymnasium-style reset() / step(action): each step presses at most one key, then moves the game clock
    exactly one tick, and returns the screen as pixels. Slow or fast players face the same game;
  - real-time mode (realtime=True): the clock keeps running while the player thinks. step_realtime(action, dt,
    duration_ms) lets `dt` seconds of game time pass (hazards, timers, physics keep ticking) and can hold a key
    down for `duration_ms`. Turn-based games have no clock, so they simply wait for the player in either mode;
  - run_window(): the pygame window, with --seed, --speed (for tests) and --lang.
"""
from __future__ import annotations

import argparse
import copy
import random
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from practice_arena.game import _font as _load_font
from rtl import show

TITLE_PREFIX = "PlayFellow Arena · "
KEY_ALIASES = {"return": "enter", "escape": "esc"}  # pygame key names → the names controls cards use
font = lru_cache(maxsize=None)(_load_font)  # loading a font file every frame made each render ~10 ms
BANNER = 64
BACKGROUND = (16, 16, 32)
HUD_TEXT = (255, 255, 255)
HUD_NOTE = (255, 214, 110)
WIN = (120, 230, 140)
LOSE = (255, 120, 120)
LABELS = {
    "score": ("Score", "النقاط"),
    "won": ("You win!", "فزت!"),
    "lost": ("Game over", "انتهت اللعبة"),
}


class ArenaPackGame:
    """Subclasses set name/keys and write new_game, on_key, draw_board, board_pixels, progress, is_won, is_lost."""

    name = "Game"
    name_ar = "لعبة"
    keys: tuple[str, ...] = ()
    max_ticks = 2000  # step() reports `truncated` after this many ticks
    realtime = False  # True: step() lets `step_dt` seconds pass instead of exactly one tick (the world doesn't wait)
    step_dt = 0.05  # seconds of game time per step() while realtime (how long the player "thinks" per decision)
    observe = True  # False: step() skips drawing the picture (fast headless runs; policies that read state)

    def __init__(self, seed: int = 0, language: str = "english"):
        self.seed = seed
        self.language = language
        self.reset(seed)

    # ---- the game, in the subclass
    def new_game(self) -> None:
        raise NotImplementedError

    def on_key(self, key: str) -> None:
        raise NotImplementedError

    def on_click(self, x: float, y: float) -> None:
        """A mouse click at window pixel (x, y). Games without the mouse ignore it."""

    def tick(self) -> None:
        """One clock tick. Turn-based games do nothing here."""

    def board_pixels(self) -> tuple[int, int]:
        raise NotImplementedError

    def draw_board(self, image: Image.Image, draw: ImageDraw.ImageDraw, top: int) -> None:
        raise NotImplementedError

    def progress(self) -> dict:
        raise NotImplementedError

    def progress_text(self) -> str:
        return ""

    def is_won(self) -> bool:
        raise NotImplementedError

    def is_lost(self) -> bool:
        return False

    def seconds_per_tick(self) -> float | None:
        """Real-time games return how long one tick lasts in the window; turn-based games return None."""
        return None

    # ---- shared
    @property
    def title(self) -> str:
        return TITLE_PREFIX + self.name

    @property
    def done(self) -> bool:
        return self.is_won() or self.is_lost()

    def text(self, key: str) -> str:
        english, arabic = LABELS[key]
        return arabic if self.language == "arabic" else english

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.score = 0
        self.ticks = 0
        self._clock_debt = 0.0  # real-time: seconds that have passed but have not made a whole tick yet
        self._hold: tuple[str, float] | None = None  # real-time: (key, seconds still held down)
        self.new_game()
        return self.observation()

    state_fields: tuple[str, ...] = ()  # the attributes that make up the game state (for snapshot/restore)

    def snapshot(self) -> tuple:
        """A cheap copy of the whole game state, without pictures (tests use it to try moves ahead)."""
        fields = ("score", "ticks") + self.state_fields
        return copy.deepcopy(tuple(getattr(self, name) for name in fields)), self.rng.getstate()

    def restore(self, snap: tuple) -> None:
        values, rng_state = snap
        for name, value in zip(("score", "ticks") + self.state_fields, copy.deepcopy(values)):
            setattr(self, name, value)
        self.rng.setstate(rng_state)

    def press(self, key: str) -> None:
        key = KEY_ALIASES.get(key.lower(), key.lower())
        if key in self.keys and not self.done:
            self.on_key(key)

    def click(self, gx: float, gy: float) -> None:
        """A click on the 0-1000 grid that covers the whole window (the buddy's way of pointing)."""
        if not self.done:
            width, height = self.pixel_size
            self.on_click(gx / 1000 * width, gy / 1000 * height)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Do `action` (a key name, ("click", x, y) on the 0-1000 grid, or None to wait), then advance one tick."""
        if self.realtime:
            return self.step_realtime(action, self.step_dt)
        before = self.score
        if isinstance(action, tuple) and action[0] == "click":
            self.click(action[1], action[2])
        elif action:
            self.press(action)
        if not self.done:
            self.tick()
        self.ticks += 1
        info = {"progress": self.progress(), "won": self.is_won(), "lost": self.is_lost(), "ticks": self.ticks}
        frame = self.observation() if self.observe else None
        return frame, float(self.score - before), self.done, self.ticks >= self.max_ticks, info

    def step_realtime(self, action=None, dt: float = 0.05, duration_ms: int = 0) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Do `action`, then let `dt` seconds of game time pass: as many whole clock ticks as fit (a leftover
        carries over to the next call), so hazards and timers keep moving while the player thinks.
        `duration_ms` holds a key action down: it is pressed again on every tick until the time is up (like
        key repeat). Turn-based games have no clock: the action is applied and the world waits."""
        before = self.score
        if isinstance(action, tuple) and action[0] == "click":
            self.click(action[1], action[2])
        elif action:
            self.press(action)
            self._hold = (action, duration_ms / 1000) if duration_ms > 0 else None
        per_tick = self.seconds_per_tick()
        if per_tick is None:
            self.ticks += 1
        else:
            self._clock_debt += dt
            while self._clock_debt >= per_tick - 1e-9 and not self.done:
                self._clock_debt = max(0.0, self._clock_debt - per_tick)
                self.tick()
                self.ticks += 1
                if self._hold:
                    key, left = self._hold
                    left -= per_tick
                    self._hold = (key, left) if left > 1e-9 else None
                    if self._hold:
                        self.press(key)
        info = {"progress": self.progress(), "won": self.is_won(), "lost": self.is_lost(), "ticks": self.ticks}
        frame = self.observation() if self.observe else None
        return frame, float(self.score - before), self.done, self.ticks >= self.max_ticks, info

    @property
    def pixel_size(self) -> tuple[int, int]:
        width, height = self.board_pixels()
        return width, height + BANNER

    def observation(self) -> np.ndarray:
        return np.asarray(self.render())

    def render(self) -> Image.Image:
        image = Image.new("RGB", self.pixel_size, BACKGROUND)
        draw = ImageDraw.Draw(image)
        arabic = self.language == "arabic"
        name = self.name_ar if arabic else self.name
        lines = [(f"{name}    {self.text('score')}: {self.score}", 20, HUD_TEXT, 8),
                 (self.progress_text(), 16, HUD_NOTE, 36)]
        if self.done:
            lines[1] = (self.text("won") if self.is_won() else self.text("lost"), 18, WIN if self.is_won() else LOSE, 36)
        for text, size, colour, y in lines:
            text = show(text)
            text_font = font(size, self.language)
            x = 12
            if arabic:
                box = draw.textbbox((0, 0), text, font=text_font)
                x = self.pixel_size[0] - (box[2] - box[0]) - 12
            draw.text((x, y), text, fill=colour, font=text_font)
        self.draw_board(image, draw, BANNER)
        return image


def run_window(game_class: type[ArenaPackGame], description: str, argv: list[str] | None = None,
               frames: int | None = None) -> ArenaPackGame:
    """Open the game in a window. Keys go to the game; real-time games tick on their own clock (× --speed).
    `frames` stops after that many frames (tests run it headless with the SDL dummy driver)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="clock speed for real-time games (tests use more)")
    parser.add_argument("--lang", choices=["english", "arabic"], default="english")
    args = parser.parse_args(argv)

    import pygame
    pygame.display.init()  # the games draw with PIL: only the window is needed (full init also starts audio)
    game = game_class(seed=args.seed, language=args.lang)
    window = pygame.display.set_mode(game.pixel_size)
    pygame.display.set_caption(game.title)
    clock = pygame.time.Clock()
    waited = 0.0
    running = True
    while running and frames != 0:
        frames = None if frames is None else frames - 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                game.press(pygame.key.name(event.key))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game.done:
                game.on_click(*event.pos)
        waited += clock.tick(60) / 1000
        per_tick = game.seconds_per_tick()
        if per_tick is not None and not game.done and waited >= per_tick / max(args.speed, 0.01):
            game.tick()
            game.ticks += 1
            waited = 0.0
        frame = game.render()
        window.blit(pygame.image.frombytes(frame.tobytes(), frame.size, "RGB"), (0, 0))
        pygame.display.flip()
    pygame.quit()
    return game
