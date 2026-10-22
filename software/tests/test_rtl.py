"""Tests for RTL Arabic display. / اختبارات عرض العربية من اليمين إلى اليسار."""
import re
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from rtl import show


def test_show_leaves_english_unchanged():
    assert show("Start") == "Start"


def test_show_empty_string():
    assert show("") == ""


def test_show_arabic_reversed_and_reshaped():
    """Multi-word Arabic: visual order puts the last logical word on the left."""
    result = show("لا شيء بعد")
    # No characters in the basic Arabic block (all are presentation forms)
    assert not re.search(r'[ء-ي]', result)
    # First visible word from the left is the shaped form of "بعد" (the last logical word)
    expected_first = get_display(reshape("بعد"))
    assert result.strip().startswith(expected_first)


def test_show_multiline():
    """First line unchanged, second line transformed."""
    result = show("Ready!\nجاهز! اضغط ابدأ.")
    lines = result.split("\n")
    assert lines[0] == "Ready!"
    # Second line has Arabic
    assert re.search(r'[ﹰ-﻿ﭐ-﷿]', lines[1])


def test_show_mixed_line():
    """Mixed English/Arabic: English prefix unchanged, Arabic suffix transformed."""
    result = show("Start / ابدأ")
    assert result.startswith("Start / ")
    expected_suffix = get_display(reshape("ابدأ"))
    assert result.endswith(expected_suffix)
