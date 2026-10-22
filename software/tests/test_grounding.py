"""Tests for Grounding: Marks, HUD Watcher and Failure Recorder."""
from __future__ import annotations

from PIL import Image

from grounding import (
    compass,
    draw_marks,
    FailureRecorder,
    HudWatcher,
    label_marks,
    legend,
    Mark,
    mark_from_box,
)


# ---------------------------------------------------------------------------
# Legacy Grounding Tests
# ---------------------------------------------------------------------------

def test_mark_from_box_uses_the_0_1000_grid():
    mark = mark_from_box("reward", (100, 50, 140, 90), (400, 200))
    assert (mark.x, mark.y) == (300, 350)


def test_label_marks_numbers_nearest_first_and_puts_me_first():
    me = Mark("me", 500, 500)
    far, near = Mark("reward", 900, 900), Mark("reward", 520, 500)
    out = label_marks([far, near, Mark("hazard", 100, 100)], me)
    assert [m.tag for m in out] == ["ME", "R1", "R2", "H1"]
    assert out[1].x == 520  # the nearest coin is R1


def test_label_marks_caps_clutter():
    out = label_marks([Mark("reward", i * 10, 0) for i in range(20)])
    assert len(out) == 6


def test_draw_marks_changes_pixels_and_keeps_the_original():
    image = Image.new("RGB", (200, 200), (0, 0, 0))
    marks = label_marks([Mark("reward", 800, 200)], Mark("me", 200, 800))
    out = draw_marks(image, marks)
    assert out.size == image.size
    assert image.getbbox() is None  # original untouched (still all black)
    assert out.getbbox() is not None  # the copy has rings, tags and an arrow


def test_compass_words():
    me = Mark("me", 500, 500)
    assert compass(me, Mark("reward", 520, 510)) == "right here"
    assert compass(me, Mark("reward", 500, 100)).startswith("up,")
    assert compass(me, Mark("reward", 900, 900)).startswith("down-right")
    assert compass(me, Mark("reward", 100, 500)).startswith("left,")


def test_legend_explains_every_label():
    text = legend(label_marks([Mark("hazard", 900, 500)], Mark("me", 500, 500)))
    assert "[ME] = you" in text
    assert "[H1] = danger, avoid (right," in text


def test_hud_watcher_reports_only_real_changes():
    watcher = HudWatcher({"score": (0.0, 0.0, 0.5, 0.2)})
    page = Image.new("RGB", (100, 100), (10, 10, 10))
    assert watcher.update(page) == []  # first look is never a change
    assert watcher.update(page) == []
    changed = page.copy()
    changed.paste((250, 250, 250), (0, 0, 50, 20))
    assert watcher.update(changed) == ["score"]
    elsewhere = changed.copy()
    elsewhere.paste((0, 0, 0), (60, 60, 100, 100))
    assert watcher.update(elsewhere) == []  # change outside the patch is ignored


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_failure_recorder_keeps_only_the_last_three_seconds():
    clock = Clock()
    rec = FailureRecorder(clock=clock)
    for i in range(10):
        clock.now = i * 1.0
        rec.add(Image.new("RGB", (640, 480), (i, i, i)))
    result = rec.autopsy("fell in a pit")
    assert result.seconds == 3.0
    assert len(result.frames) == 4
    assert max(result.frames[0].size) == 320  # shrunk to save tokens
    assert result.frames[0].getpixel((0, 0))[0] == 6  # oldest kept frame is 3 s back
    assert result.note == "fell in a pit"


def test_failure_recorder_with_nothing_recorded():
    assert FailureRecorder(clock=Clock()).autopsy() is None


# ---------------------------------------------------------------------------
# GEA Hybrid Perceptual Anchors & GroundingBuffer Tests
# ---------------------------------------------------------------------------
