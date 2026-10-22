import pytest

from games import detect_games, is_blocked


def test_is_blocked_blocks_launchers_and_stores():
    assert is_blocked("Ubisoft Connect")
    assert is_blocked("Epic Games Launcher")
    assert is_blocked("Battle.net")


def test_is_blocked_blocks_browsers():
    assert is_blocked("New Tab - Google Chrome")
    assert is_blocked("Microsoft Edge")


def test_is_blocked_exact_matches():
    # Exact matches only when the title is exactly this
    assert is_blocked("Steam")
    assert is_blocked("Origin")
    assert is_blocked("Edge")


def test_is_blocked_does_not_block_real_games():
    # These are real games that should NOT be blocked
    assert not is_blocked("Rayman Legends")
    assert not is_blocked("Mirror's Edge")
    assert not is_blocked("Rayman Origins")
    assert not is_blocked("SteamWorld Dig")
    assert not is_blocked("Code Vein")
    assert not is_blocked("Ori and the Will of the Wisps")


def test_is_blocked_case_insensitive():
    assert is_blocked("UBISOFT CONNECT")
    assert is_blocked("google chrome")
    assert is_blocked("steam")


def test_is_blocked_removes_zero_width_spaces():
    # Windows sometimes adds zero-width spaces (U+200B)
    assert is_blocked("Ubisoft​Connect")
    assert is_blocked("Google​ Chrome")


def test_detect_games_filters_blocked_titles():
    titles = [
        "Rayman Legends",
        "Ubisoft Connect",
        "Minecraft",
        "Google Chrome",
        "Mirror's Edge",
    ]
    result = detect_games(titles)
    assert result == ["Rayman Legends", "Minecraft", "Mirror's Edge"]


def test_detect_games_removes_blanks():
    titles = ["", "  ", "Minecraft", ""]
    assert detect_games(titles) == ["Minecraft"]


def test_detect_games_deduplicates():
    titles = ["Minecraft", "Roblox", "Minecraft", "Terraria", "Roblox"]
    assert detect_games(titles) == ["Minecraft", "Roblox", "Terraria"]


def test_detect_games_keeps_order():
    titles = ["Game C", "Game A", "Game B"]
    assert detect_games(titles) == ["Game C", "Game A", "Game B"]


# [RATCHET-ONLINE-GAMES-WARN] Owner decision 2026-10-16: online / anti-cheat games are NOT blocked; the adult is warned once.
def test_online_games_warn_but_are_never_blocked():
    from games import detect_games, is_blocked, needs_online_warning
    from messages import msg

    for title in ["PUBG: BATTLEGROUNDS", "Fortnite", "VALORANT  ", "Apex Legends", "Call of Duty® HQ", "Roblox"]:
        assert needs_online_warning(title), title
        assert not is_blocked(title), title
    assert detect_games(["PUBG: BATTLEGROUNDS", "Rayman Legends"]) == ["PUBG: BATTLEGROUNDS", "Rayman Legends"]
    assert not needs_online_warning("Rayman Legends")
    assert "anti-cheat" in msg("online_game_warning", title="PUBG")


# [RATCHET-CHESSCOM-TAB] The Chess.com tab must be lockable (owner's bots-only chess test) while other browser tabs stay blocked.
def test_chess_com_browser_tab_is_allowed_and_warned():
    from games import is_blocked, needs_online_warning

    title = "Play Chess vs Computer - Chess.com - Google Chrome"
    assert not is_blocked(title)
    assert needs_online_warning(title)
    assert is_blocked("YouTube - Google Chrome")
    assert is_blocked("PayPal Checkout - Microsoft Edge")


def test_chess_com_payment_pages_stay_blocked():
    from games import is_blocked

    for title in ["Checkout - Chess.com - Google Chrome", "Upgrade Membership - Chess.com - Microsoft Edge"]:
        assert is_blocked(title), title


def test_chess_com_allow_is_anchored_not_substring():
    """Security review 2026-10-16: 'chess.com' anywhere in a title must not unlock a browser tab."""
    from games import is_blocked

    for title in ["chess.com tips and tricks - YouTube - Google Chrome",
                  "Totally chess.com - Evil Site - Microsoft Edge",
                  "chess.com - Google Chrome"]:
        assert is_blocked(title), title
    assert not is_blocked("Play Chess vs Computer - Chess.com - Google Chrome")


def test_poki_subway_surfers_tab_allowed_and_anchored():
    from games import is_blocked, needs_online_warning

    for title in ["Subway Surfers - Play Online for Free! | Poki - Google Chrome",
                  "Subway Surfers - Play Online for Free! | Poki - Microsoft Edge",
                  "Subway Surfers - Play Online for Free! | Poki - Personal - Microsoft Edge"]:
        assert not is_blocked(title), title
    for title in ["poki tips - YouTube - Google Chrome",
                  "Poki - Evil Site - Microsoft Edge",
                  "Checkout | Poki - Google Chrome",
                  "chess.com - Evil - Personal - Microsoft Edge"]:
        assert is_blocked(title), title


# [RATCHET-ONLINE-GAMES-WARN] Real-game cards with anti-cheat / always-online parts must warn (review 2026-10-16).
def test_risky_real_game_cards_warn():
    from games import needs_online_warning

    for title in ["EA SPORTS FC™ 26", "Asphalt Legends Unite", "eFootball™", "Mortal Kombat 1"]:
        assert needs_online_warning(title), title
    for title in ["Hollow Knight", "Celeste", "Stardew Valley", "Rayman Legends"]:
        assert not needs_online_warning(title), title
