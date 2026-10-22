"""Game window detection and safety blocks. / كشف نوافذ الألعاب والحماية من النوافذ الممنوعة."""
from __future__ import annotations

# Blocked substrings: unambiguous phrases that can't be part of a game's name
BLOCKED_PHRASES = [
    # Launchers and stores
    "ubisoft connect",
    "epic games launcher",
    "ea app",
    "battle.net",
    "gog galaxy",
    "microsoft store",
    "playstation store",
    "ps store",
    "rockstar games launcher",
    "riot client",
    # Browsers
    "google chrome",
    "microsoft edge",
    "mozilla firefox",
    "opera gx",
    "brave",
    "safari",
    # Chat and mail
    "discord",
    "whatsapp",
    "telegram",
    "microsoft teams",
    "zoom workplace",
    "slack",
    "outlook",
    "messenger",
    # System and tools
    "file explorer",
    "task manager",
    "control panel",
    "program manager",
    "visual studio",
    "windows powershell",
    "powershell",
    "command prompt",
    "cmd.exe",
    "pwsh",
    "mingw64",
    "terminal",
    "registry editor",
    "task scheduler",
    "event viewer",
    "windows security",
    "chromium",
    "tor browser",
    "yandex",
    "vivaldi",
    # Money
    "paypal",
    "checkout",
    "payment",
    # Itself
    "rivel playfellow ai",
]

# Blocked exact matches: whole titles that are blocked only when the title is exactly this
BLOCKED_EXACT = [
    "steam",
    "playfellow",
    "services",
    "arc",
    "origin",
    "ea",
    "xbox",
    "settings",
    "mail",
    "opera",
    "claude",
    "edge",
]


# Online games against real people / anti-cheat protected. Owner decision 2026-10-16:
# NOT blocked — the adult sees a warning once and may continue. A restriction list may come later.
ONLINE_GAMES = [
    # Real-game cards with anti-cheat or always-online parts (side-agent review, 2026-10-16)
    "ea sports fc",
    "asphalt",
    "efootball",
    "mortal kombat",
    "chess.com",
    "pubg",
    "battlegrounds",
    "fortnite",
    "valorant",
    "apex legends",
    "call of duty",
    "warzone",
    "counter-strike",
    "rainbow six",
    "league of legends",
    "dota 2",
    "overwatch",
    "rocket league",
    "roblox",
]


from titles import title_variants  # noqa: E402
from bridges import is_stream_window  # noqa: E402  (the narrow door for console and phone streams)


BLOCKED_SUFFIXES = (" - arc",)  # the Arc browser puts its name last ("New Tab - Arc")


def _normalize(title: str) -> str:
    return title_variants(title)[0]  # look-alike letters, fullwidth forms and invisible marks are read as plain letters


def _blocked_reading(text: str) -> bool:
    return (any(phrase in text for phrase in BLOCKED_PHRASES) or text in BLOCKED_EXACT
            or any(text.endswith(suffix) for suffix in BLOCKED_SUFFIXES))


# Browser-tab games allowed even though browsers are blocked (owner: Chess.com Play Bots, 2026-10-15).
BROWSER_GAMES_ALLOWED = ["chess.com", "poki"]  # Poki: Subway Surfers etc. (owner, 2026-10-16)


BROWSER_NAMES = ("google chrome", "microsoft edge", "mozilla firefox", "firefox", "brave", "opera", "vivaldi", "chrome", "edge")
BROWSER_PROFILES = ("personal", "work", "school", "guest")
MONEY_WORDS = ("checkout", "payment", "paypal", "membership", "upgrade", "subscribe", "billing", "purchase", "buy ")


def _is_allowed_browser_game(normalized: str) -> bool:
    """True only when the SITE itself is an allowed game site.

    Titles look like "<page> - <site> - <browser>". The site must be the last segment before the
    browser name — so "chess.com tips - YouTube - Google Chrome" is NOT allowed (anchored, not substring).
    Pages with money words always stay blocked.
    """
    if any(word in normalized for word in MONEY_WORDS):
        return False
    parts = [part.strip() for part in normalized.replace("—", "-").replace("–", "-").split(" - ")]
    if parts and parts[-1] in BROWSER_NAMES:
        parts = parts[:-1]
        if parts and parts[-1] in BROWSER_PROFILES:  # Edge adds "Personal" / "Work" etc.
            parts = parts[:-1]
    if not parts:
        return False
    site = parts[-1]
    if "|" in site:  # "Play Online for Free! | Poki"
        site = site.split("|")[-1].strip()
    elif len(parts) < 2:  # a bare "chess.com" tab title is not a game page
        return False
    return site in BROWSER_GAMES_ALLOWED


def needs_online_warning(title: str) -> bool:
    """True for online / anti-cheat games: warn the adult before playing (never blocked)."""
    normalized = _normalize(title)
    if is_stream_window(title):
        return True  # a console or phone stream may be an online game with real people: warn the adult once
    return any(name in normalized for name in ONLINE_GAMES)


def is_blocked(title: str) -> bool:
    """True if the title is blocked: a blocked phrase anywhere in it, or exactly a blocked name.

    The title is read every way an invisible mark could hide a word (marks removed, and marks as spaces), with look-alike
    letters and fullwidth forms turned into plain ones. A blocked word in EITHER reading blocks the window. Console and phone
    streams get no shortcut: every title goes through here first.
    """
    removed, spaced = title_variants(title)
    # Browser games the owner allows (warned as online, never blocked): e.g. Chess.com "Play Bots"
    if _is_allowed_browser_game(removed):
        return False
    return _blocked_reading(removed) or _blocked_reading(spaced)


def detect_games(titles: list[str]) -> list[str]:
    """Return non-blank titles that aren't blocked, de-duplicated, keeping order."""
    seen = set()
    result = []
    for title in titles:
        title = title.strip()
        if not title:
            continue
        if is_blocked(title):
            continue
        if title in seen:
            continue
        seen.add(title)
        result.append(title)
    return result
