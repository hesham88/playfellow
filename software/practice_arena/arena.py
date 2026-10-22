"""Opens the Practice Arena window. Play with the arrow keys, R to restart. / يفتح نافذة ساحة التدريب. العب بالأسهم، R لإعادة البدء."""
import argparse
from pathlib import Path

import pygame

from practice_arena.game import WINDOW_TITLE, ArenaGame


def main() -> None:
    parser = argparse.ArgumentParser(description="PlayFellow Practice Arena")
    parser.add_argument("--lang", choices=["english", "arabic"], help="Language (english or arabic)")
    args = parser.parse_args()

    # Try to load language from settings.yaml next to software folder
    language = "english"
    if args.lang:
        language = args.lang
    else:
        try:
            here = Path(__file__).resolve().parent.parent
            settings_path = here / "settings.yaml"
            if settings_path.exists():
                from settings import load_settings
                settings = load_settings(settings_path)
                language = settings.language
        except Exception:
            pass  # Fall back to English if settings can't be loaded

    pygame.init()
    game = ArenaGame(language=language)
    window = pygame.display.set_mode(game.pixel_size)
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                name = pygame.key.name(event.key)
                if name == "r":
                    game.reset()
                else:
                    game.press(name)
        frame = game.render()
        window.blit(pygame.image.frombytes(frame.tobytes(), frame.size, "RGB"), (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
