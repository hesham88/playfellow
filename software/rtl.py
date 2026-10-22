"""Shows Arabic text the right way round on screen. / يعرض النص العربي بالترتيب الصحيح على الشاشة."""
import re
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# Arabic Unicode ranges
ARABIC_PATTERN = re.compile(r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]')


def show(text: str) -> str:
    """Returns the text ready for Tk to display.

    Each line is processed separately. Lines with no Arabic letters are returned
    unchanged. Lines with Arabic become presentation forms in visual order, so Tk
    draws them correctly without reshaping them again.
    """
    if not text:
        return text

    lines = text.split('\n')
    result_lines = []

    for line in lines:
        if ARABIC_PATTERN.search(line):
            # Line contains Arabic: reshape then reorder for visual display
            result_lines.append(get_display(reshape(line)))
        else:
            # No Arabic: pass through unchanged
            result_lines.append(line)

    return '\n'.join(result_lines)
