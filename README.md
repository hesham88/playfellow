# PlayFellow

**An AI that plays your PC game with you: the open research core.**

PlayFellow watches one game window, thinks about what it sees, and presses only the keys that game's controls card allows. It is built to play *with* people: a co-op partner you can hand the controls to, an accessibility aid for players who can't use every key, and an open testbed for research on game-playing agents that learn from pixels.

> **Status: early research.** The safety layer, the controls cards, the Practice Arena and 15 small test games, the brain interface and the built-in learning navigator work and are covered by 695 automated tests (2026-10-22). The built-in brain wins the Practice Arena through the real play loop (`tests/test_play.py`). On our 15-game benchmark ladder the learned navigator has **not passed a level yet**, and in our latest live run a cloud brain (Gemini) took the key and all but one coin of the Practice Arena, but did not finish inside the two-minute limit. Numbers here are measured, not hoped for.

## How it works

Every turn: **look → mark → think → check → act**, with the safety rules around all of it.

| Step | Where | What it does |
|---|---|---|
| look | `software/eyes.py`, `games.py` | Finds the chosen game window and takes its picture. Launchers, stores, browsers and payment pages are never looked at or played. |
| mark | `software/grounding.py` | Draws labels on the picture (ME, R1 = reward, H1 = hazard, D1 = door) so a brain can say "go to R1" instead of guessing pixels. |
| think | `software/brain/` | One interface, `ask(system, text, images) -> str`, for the built-in navigator (no key, no network), Gemini, Claude, OpenAI or a local Ollama model. |
| check | `software/reply.py`, `controls/*.yaml` | The brain answers in JSON; any action that is not on the game's controls card is dropped. |
| act | `software/hands.py` | Presses the keys, and only while the game window is in front. |
| stop | `software/safety.py` | **F12**, or the mouse in the top-left corner, stops everything at once and releases held keys; time and AI-call limits end a session. |

Around the loop: `nav/` is the built-in navigator (it works out which thing on screen is "me", maps walls, rewards and doors, and plans routes), `practice_arena/` and `arena_pack/` are the test games, `ladder.py` is the benchmark and `play.py` ties the loop together.

## Quick start (Windows 10/11, Python 3.13)

```
git clone https://github.com/hesham88/playfellow.git
cd playfellow\software
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In one terminal, start the test game:

```
python -m practice_arena.arena
```

In a second terminal, let PlayFellow play it with the built-in brain (no account, no key, no network):

```
python play.py
```

Click the arena window so it is in front: PlayFellow presses keys only while the game is in front. Press **F12**, or move the mouse to the top-left corner, to stop at once. The first run asks you to confirm that you are 18 or older.

### With a cloud brain

Use your own key. It is kept in Windows Credential Manager, never in a file:

```
python play.py --save-key gemini
python play.py --brain gemini
```

`claude`, `openai` and `local` (Ollama) work the same way. Model names live in one place, `software/settings.py` (`MODEL_TIERS`). Using a provider means accepting that provider's terms; several, including Meta's and Google's, are for adults only.

### Other games

Choose the window and its controls card: `python play.py --window "My Game" --card my_game`. A card is a short YAML file in `software/controls/` that lists the only actions allowed, with a hint about the game. Online games played against real people are refused unless you add `--online-ok`, and many of them forbid automated play: read their rules first ([RESPONSIBLE_USE.md](RESPONSIBLE_USE.md)).

## Tests and the benchmark

```
cd software
python -m pytest -q -W error          # every test: no real keys, screen, network or credential store
python ladder.py --brain scripted     # the baseline: proves every level can be won
python ladder.py --brain nav          # the learned navigator, from pixels only
```

**Learning, not hard-coding.** The hand-written solvers ("oracles") read a game's true state, so they live only in `tests/` and serve as the `scripted` baseline. The buddy itself sees only pixels and what a human player could read on screen, and a test fails if the learned navigator ever consults an oracle.

## Roadmap

During the Meta Global AI Developer Hackathon (26 October to 4 November 2026) this repository adds, in dated commits:

- **SAM 3.1 for seeing:** find "me", rewards and hazards in any game's picture, as real marks for the mark step;
- **Muse for thinking:** a Muse brain behind the same `ask(...)` interface;
- **hand-off co-op:** you and PlayFellow pass the controls back and forth during play.

## Responsible use and licence

See [LICENSE](LICENSE) and [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md); also [TRADEMARKS.md](TRADEMARKS.md), [NOTICE.md](NOTICE.md) and [SECURITY.md](SECURITY.md).

## Acknowledgements

Inspired by research such as Google DeepMind's SIMA 2 and NVIDIA's NitroGen. No code, weights or data from either is used, and there is no affiliation or endorsement.

## Author

Hesham Aboul-Fetouh ([@hesham88](https://github.com/hesham88)). Issues are welcome; pull requests are not accepted ([CONTRIBUTING.md](CONTRIBUTING.md)).
