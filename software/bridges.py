"""Console and phone streams: windows on the PC that show a PlayStation, an Xbox or a phone. / بث الأجهزة: نوافذ تعرض بلايستيشن أو إكس بوكس أو هاتفًا.

Like a window onto another room: the console or phone is somewhere else, and its picture appears in a window on
this PC. The buddy plays by pressing keys or dragging in that window, the same way it plays any game window.
Recognising a stream window is all this file decides; it never touches the console, the phone or the network.
"""
from __future__ import annotations

from dataclasses import dataclass

from titles import normalize_title

# Titles that are never a stream, even when they contain a streaming name: browsers, stores, launchers, payments.
# (games.py keeps its own full blocklist; this is the extra guard on the narrow door that lets streams through.)
NOT_A_STREAM = ("google chrome", "microsoft edge", "mozilla firefox", "firefox", "brave", "opera", "vivaldi", "safari",
                "arc", "chromium", "tor browser", "yandex", "duckduckgo",
                "microsoft store", "playstation store", "ps store", "xbox store", "app store", "play store",
                "steam", "epic games", "launcher", "discord", "checkout", "payment", "paypal", "membership",
                "upgrade", "subscribe", "billing", "purchase", "buy ")


@dataclass(frozen=True)
class Bridge:
    key: str
    name: str
    name_ar: str
    patterns: tuple[str, ...]  # the stream window's whole title, lower case (nothing else around it)
    card: str  # the controls card for this kind of stream
    aspect: float  # shape of the picture inside the window (width / height); phones are tall


BRIDGES = (
    Bridge("playstation", "PlayStation Remote Play", "بلايستيشن ريموت بلاي",
           ("ps remote play", "chiaki", "chiaki-ng"), "ps_remote_play", 16 / 9),
    Bridge("xbox", "Xbox Remote Play", "إكس بوكس ريموت بلاي",
           ("xbox remote play", "xbox cloud gaming"), "xbox_remote_play", 16 / 9),
    Bridge("android", "Android (scrcpy)", "أندرويد (scrcpy)", ("scrcpy",), "android_scrcpy", 9 / 19.5),
    Bridge("ios", "iPhone Mirroring", "عرض الآيفون", ("iphone mirroring",), "iphone_mirror", 9 / 19.5),
)


def _normalize(title: str) -> str:
    return normalize_title(title)


def detect(title: str) -> Bridge | None:
    """Which stream this window title is, or None. A browser tab, a store or a payment page is never a stream."""
    text = _normalize(title)
    if not text or any(word in text for word in NOT_A_STREAM):
        return None
    for bridge in BRIDGES:
        if text in bridge.patterns:  # the title IS the stream app's title, not just a window that mentions it
            return bridge
    return None


def is_stream_window(title: str) -> bool:
    return detect(title) is not None


def bridge_for_window(game_window: str) -> Bridge | None:
    """The stream a settings 'game_window' value names (it may hold several titles separated by |)."""
    for part in str(game_window).split("|"):
        found = detect(part.strip())
        if found is not None:
            return found
    return None


def crop_video(image, aspect: float = 16 / 9, tolerance: float = 0.04, dark: int = 24):
    """Where the real picture is inside a stream window that has black bars (a 16:9 video in a resized window).

    Returns (left, top, right, bottom) in the image's pixels. It only cuts when the bars are really black ALL the way
    across AND what is left has the shape of the stream; otherwise the whole image is kept, so a dark game scene is
    never cut by mistake.
    """
    import numpy as np

    arr = np.asarray(image.convert("L"))
    height, width = arr.shape
    whole = (0, 0, width, height)
    rows = np.where(arr.max(axis=1) > dark)[0]
    cols = np.where(arr.max(axis=0) > dark)[0]
    if rows.size == 0 or cols.size == 0:
        return whole
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    inner_w, inner_h = right - left, bottom - top
    if inner_w * inner_h < 0.4 * width * height or abs(inner_w / inner_h / aspect - 1) > tolerance:
        return whole
    return left, top, right, bottom
