import sys
from types import SimpleNamespace

from PIL import Image

from eyes import APP_TITLE, Screen, WindowBox, add_grid, box_is_usable, shrink


def test_box_is_usable():
    assert box_is_usable(WindowBox(0, 0, 800, 600))
    assert not box_is_usable(WindowBox(-32000, -32000, 160, 28))  # minimised window
    assert not box_is_usable(WindowBox(10, 10, 50, 600))


def test_shrink_keeps_shape_and_never_enlarges():
    big = Image.new("RGB", (1600, 900))
    small = shrink(big, 800)
    assert small.size == (800, 450)
    tiny = Image.new("RGB", (300, 200))
    assert shrink(tiny, 800).size == (300, 200)


def test_add_grid_draws_on_a_copy():
    image = Image.new("RGB", (1001, 501), "white")
    gridded = add_grid(image)
    assert gridded.size == image.size
    assert gridded.getpixel((500, 400)) != (255, 255, 255)  # the x=500 line
    assert image.getpixel((500, 400)) == (255, 255, 255)


def window(title, left=0, top=0, width=800, height=600, minimized=False):
    return SimpleNamespace(title=title, left=left, top=top, width=width, height=height, isMinimized=minimized)


def test_find_skips_playmate_itself_and_minimised_windows(monkeypatch):
    windows = [
        window(APP_TITLE),
        window("PlayFellow Practice Arena", minimized=True),
        window("PlayFellow Practice Arena", left=100, top=50),
    ]
    fake = SimpleNamespace(
        getWindowsWithTitle=lambda title: windows,
        getActiveWindow=lambda: None
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().find("PlayFellow") == WindowBox(100, 50, 800, 600)


def test_find_returns_none_when_missing(monkeypatch):
    fake = SimpleNamespace(
        getWindowsWithTitle=lambda title: [],
        getActiveWindow=lambda: None
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().find("Minecraft") is None


def test_in_front_matches_part_of_the_title(monkeypatch):
    fake = SimpleNamespace(getActiveWindow=lambda: window("Minecraft 1.21 - Singleplayer"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().in_front("minecraft")
    assert not Screen().in_front("Roblox")


def test_list_windows_hides_playmate_and_blanks(monkeypatch):
    fake = SimpleNamespace(getAllTitles=lambda: ["", " Game A ", APP_TITLE, "Game A", "Game B"])
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().list_windows() == ["Game A", "Game B"]


def test_find_prefers_the_window_in_front_among_several_titles(monkeypatch):
    """When several titles are given (A | B), prefer the one in front."""
    windows_rayman = [window("Rayman Legends", left=100, top=100)]
    windows_minecraft = [window("Minecraft", left=200, top=200)]

    def fake_get_windows(title):
        if "Rayman" in title:
            return windows_rayman
        elif "Minecraft" in title:
            return windows_minecraft
        return []

    # Minecraft is in front
    fake = SimpleNamespace(
        getWindowsWithTitle=fake_get_windows,
        getActiveWindow=lambda: window("Minecraft", left=200, top=200)
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    result = Screen().find("Rayman Legends | Minecraft")
    assert result == WindowBox(200, 200, 800, 600)  # Minecraft's position


def test_find_falls_back_to_title_order(monkeypatch):
    """When none of the listed titles is in front, use the first match."""
    windows_rayman = [window("Rayman Legends", left=100, top=100)]
    windows_minecraft = [window("Minecraft", left=200, top=200)]

    def fake_get_windows(title):
        if "Rayman" in title:
            return windows_rayman
        elif "Minecraft" in title:
            return windows_minecraft
        return []

    # Neither is in front (some other window is active)
    fake = SimpleNamespace(
        getWindowsWithTitle=fake_get_windows,
        getActiveWindow=lambda: window("Notepad", left=300, top=300)
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    result = Screen().find("Rayman Legends | Minecraft")
    assert result == WindowBox(100, 100, 800, 600)  # Rayman's position (first in list)


def test_in_front_matches_any_listed_title(monkeypatch):
    """When the title contains |, match any part."""
    fake = SimpleNamespace(getActiveWindow=lambda: window("Rayman Legends - Level 3"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().in_front("Rayman Legends | Minecraft")

    fake2 = SimpleNamespace(getActiveWindow=lambda: window("Minecraft - Survival"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake2)
    assert Screen().in_front("Rayman Legends | Minecraft")

    fake3 = SimpleNamespace(getActiveWindow=lambda: window("Terraria"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake3)
    assert not Screen().in_front("Rayman Legends | Minecraft")


def test_find_skips_blocked_windows(monkeypatch):
    """find() never returns a blocked window, even if it's listed and in front."""
    windows = [
        window("Ubisoft Connect", left=100, top=100),
        window("Rayman Legends", left=200, top=200),
    ]

    def fake_get_windows(title):
        return [w for w in windows if title.lower() in w.title.lower()]

    # Ubisoft Connect is in front but blocked
    fake = SimpleNamespace(
        getWindowsWithTitle=fake_get_windows,
        getActiveWindow=lambda: window("Ubisoft Connect", left=100, top=100)
    )
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)

    # Should skip Ubisoft Connect and return Rayman Legends
    result = Screen().find("Rayman Legends | Ubisoft Connect")
    assert result == WindowBox(200, 200, 800, 600)


def test_in_front_is_false_for_blocked_active_window(monkeypatch):
    """in_front() returns False when the active window is blocked."""
    # Google Chrome is blocked
    fake = SimpleNamespace(getActiveWindow=lambda: window("Google Chrome"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert not Screen().in_front("Google Chrome")
    assert not Screen().in_front("Minecraft")  # Game isn't in front either


def test_active_title_returns_active_window_title(monkeypatch):
    """active_title() returns the title of the active window."""
    fake = SimpleNamespace(getActiveWindow=lambda: window("Minecraft"))
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().active_title() == "Minecraft"


def test_active_title_returns_empty_when_no_active_window(monkeypatch):
    """active_title() returns empty string when there's no active window."""
    fake = SimpleNamespace(getActiveWindow=lambda: None)
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    assert Screen().active_title() == ""


def test_find_leaves_out_the_invisible_border(monkeypatch):
    """Windows' invisible resize border shows the window behind the game; find() uses the visible edges (fake DWM)."""
    import eyes
    arena = SimpleNamespace(title="PlayFellow Practice Arena", left=93, top=50, width=814, height=607,
                            isMinimized=False, _hWnd=4242)
    fake = SimpleNamespace(getWindowsWithTitle=lambda title: [arena], getActiveWindow=lambda: None)
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    asked = []
    monkeypatch.setattr(eyes, "frame_bounds", lambda hwnd: asked.append(hwnd) or (100, 50, 900, 650))
    assert Screen().find("PlayFellow") == WindowBox(100, 50, 800, 600)
    assert asked == [4242]


def test_find_keeps_the_old_box_when_windows_gives_an_odd_answer(monkeypatch):
    import eyes
    arena = SimpleNamespace(title="PlayFellow Practice Arena", left=93, top=50, width=814, height=607,
                            isMinimized=False, _hWnd=4242)
    fake = SimpleNamespace(getWindowsWithTitle=lambda title: [arena], getActiveWindow=lambda: None)
    monkeypatch.setitem(sys.modules, "pygetwindow", fake)
    for answer in (None, (0, 0, 50, 50), (2000, 2000, 2900, 2600)):  # unknown, too small, somewhere else
        monkeypatch.setattr(eyes, "frame_bounds", lambda hwnd, answer=answer: answer)
        assert Screen().find("PlayFellow") == WindowBox(93, 50, 814, 607)
