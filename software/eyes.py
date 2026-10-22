"""PlayFellow's eyes: find the game window and take pictures of it. / عينا بلاي فيلو: تجد نافذة اللعبة وتلتقط صورًا لها."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

APP_TITLE = "Rivel PlayFellow AI"
GRID = 1000
GRID_COLOUR = (255, 0, 255)


@dataclass(frozen=True)
class WindowBox:
    left: int
    top: int
    width: int
    height: int


def box_is_usable(box: WindowBox) -> bool:
    """Minimised windows sit far off-screen (about -32000) and are tiny."""
    return box.width >= 100 and box.height >= 100 and box.left > -10000 and box.top > -10000


def shrink(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def add_grid(image: Image.Image) -> Image.Image:
    """Draw a 0-1000 grid so the brain can say where to click."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    for mark in range(100, GRID, 100):
        x = round(mark / GRID * (out.width - 1))
        y = round(mark / GRID * (out.height - 1))
        draw.line([(x, 0), (x, out.height)], fill=GRID_COLOUR, width=1)
        draw.line([(0, y), (out.width, y)], fill=GRID_COLOUR, width=1)
        draw.text((x + 2, 2), str(mark), fill=GRID_COLOUR)
        draw.text((2, y + 2), str(mark), fill=GRID_COLOUR)
    return out


def make_dpi_aware() -> None:
    """Stop Windows display scaling (125%, 150%) from shifting pictures and clicks."""
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def frame_bounds(hwnd) -> tuple[int, int, int, int] | None:
    """The window's visible edges (left, top, right, bottom) from Windows, or None if unknown."""
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                                      ctypes.byref(rect), ctypes.sizeof(rect)) != 0:
            return None
        return rect.left, rect.top, rect.right, rect.bottom
    except (AttributeError, OSError):
        return None


def client_bounds(hwnd) -> tuple[int, int, int, int] | None:
    """The part of the window the game draws in, in screen pixels (no title bar, no border), or None if unknown."""
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        rect, corner = wintypes.RECT(), wintypes.POINT(0, 0)
        if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(corner)):
            return None
        return corner.x, corner.y, corner.x + (rect.right - rect.left), corner.y + (rect.bottom - rect.top)
    except (AttributeError, OSError):
        return None


def visible_box(window, box: WindowBox) -> WindowBox:
    """Windows counts an invisible resize border as part of the window; the window behind shows through it.
    Use the visible edges instead, so pictures hold only the game.

    Then cut away the title bar and the borders when Windows can say where the game really draws (the client area).
    Pictures and the 0-1000 click grid both use this box, so a click at the top edge of the grid can never land on
    the title bar or the close button.
    """
    hwnd = getattr(window, "_hWnd", None)
    bounds = frame_bounds(hwnd)
    if bounds:
        left, top, right, bottom = bounds
        inside = box.left <= left < right <= box.left + box.width and box.top <= top < bottom <= box.top + box.height
        if inside and (right - left) >= box.width // 2 and (bottom - top) >= box.height // 2:
            box = WindowBox(left, top, right - left, bottom - top)  # an odd answer keeps the old box instead
    client = client_bounds(hwnd)
    if client:
        left, top, right, bottom = client
        slack = 2  # pixels of disagreement between the two Windows answers that we forgive
        within = (box.left - slack <= left < right <= box.left + box.width + slack
                  and box.top - slack <= top < bottom <= box.top + box.height + slack)
        if within and (right - left) >= box.width // 2 and (bottom - top) >= box.height // 2:
            return WindowBox(left, top, right - left, bottom - top)
    return box


class Screen:
    """The real screen. Tests use a pretend screen with the same methods."""

    def list_windows(self) -> list[str]:
        import pygetwindow as gw

        titles = {title.strip() for title in gw.getAllTitles() if title.strip()}
        titles.discard(APP_TITLE)
        return sorted(titles)

    def active_title(self) -> str:
        """Return the title of the active window, or empty string if none."""
        import pygetwindow as gw

        active = gw.getActiveWindow()
        return active.title if active else ""

    def find(self, title: str) -> WindowBox | None:
        import pygetwindow as gw
        from games import is_blocked

        # Split on | and strip each part
        title_parts = [part.strip() for part in title.split("|")]

        # Collect all usable windows for any title part
        candidates = []
        for part in title_parts:
            for window in gw.getWindowsWithTitle(part):
                if window.title.strip() == APP_TITLE or getattr(window, "isMinimized", False):
                    continue
                # Never return a blocked window
                if is_blocked(window.title):
                    continue
                box = visible_box(window, WindowBox(window.left, window.top, window.width, window.height))
                if box_is_usable(box):
                    candidates.append((part, window, box))

        if not candidates:
            return None

        # Check if any candidate is the active window
        active = gw.getActiveWindow()
        if active:
            for part, window, box in candidates:
                if window.title == active.title:
                    return box

        # Otherwise return the first match in title order
        return candidates[0][2]

    def in_front(self, title: str) -> bool:
        import pygetwindow as gw
        from games import is_blocked

        active = gw.getActiveWindow()
        if not active:
            return False

        # Return False if the active window is blocked
        if is_blocked(active.title):
            return False

        # Split on | and check if any part matches
        title_parts = [part.strip() for part in title.split("|")]
        active_title = (active.title or "").lower()
        return any(part.lower() in active_title for part in title_parts)

    def grab(self, box: WindowBox) -> Image.Image:
        import mss

        with mss.mss() as screen:
            shot = screen.grab({"left": box.left, "top": box.top, "width": box.width, "height": box.height})
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def bring_to_front(self, title: str) -> None:
        """Activate the window (restore if minimized). Swallows pygetwindow's spurious error messages."""
        import pygetwindow as gw
        from games import is_blocked

        # Never bring a blocked window to front
        if is_blocked(title):
            return

        # Split on | and find the first matching window
        title_parts = [part.strip() for part in title.split("|")]
        for part in title_parts:
            for window in gw.getWindowsWithTitle(part):
                if is_blocked(window.title):
                    continue
                try:
                    if getattr(window, "isMinimized", False):
                        window.restore()
                    window.activate()
                    return
                except Exception:
                    # pygetwindow sometimes prints "Error code from Windows: 0"
                    # which is spurious - swallow it
                    pass
