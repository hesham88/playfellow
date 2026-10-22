"""Window titles, made comparable. / عناوين النوافذ، موحَّدة للمقارنة.

A window title can be dressed up so a plain text match misses it: a Cyrillic "о" that looks like a Latin "o", a fullwidth
"Ｃ", invisible characters in the middle of a word, odd kinds of spaces. Both the blocklist (games.py) and the stream
recogniser (bridges.py) read titles through here, so they cannot disagree.
"""
from __future__ import annotations

import unicodedata

# Letters from other alphabets that look like Latin ones (lower case, after NFKC and casefold).
CONFUSABLES = str.maketrans({
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ё": "e", "з": "3", "и": "i", "і": "i", "ї": "i", "ј": "j", "к": "k", "м": "m", "н": "h",
    "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ү": "y", "ѕ": "s", "ԁ": "d", "һ": "h", "ԛ": "q", "ԝ": "w",
    "ь": "b", "ӏ": "l", "ο": "o",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "γ": "y", "μ": "u",
    "ο": "o", "ϲ": "c", "ј": "j",
})
_INVISIBLE = ("Cf", "Cc", "Cs", "Co", "Cn")  # format marks (zero-width), controls, unassigned: nothing a person can see
_SPACES = ("Zs", "Zl", "Zp")


def title_variants(title: str) -> tuple[str, str]:
    """(invisible characters removed, invisible characters turned into spaces), both lower case with the look-alike
    letters mapped to Latin and the spaces tidied. A blocked word is matched against both, so "Comm<ZWSP>and Prompt" and
    "Ubisoft<ZWSP>Connect" are each caught."""
    base = unicodedata.normalize("NFKC", str(title))
    removed, spaced = [], []
    for ch in base:
        kind = unicodedata.category(ch)
        if kind in _INVISIBLE and ch not in "\t\n\r":
            spaced.append(" ")
            continue
        char = " " if kind in _SPACES or ch in "\t\n\r" else ch
        removed.append(char)
        spaced.append(char)

    def tidy(chars: list[str]) -> str:
        text = "".join(chars).casefold().translate(CONFUSABLES)
        return " ".join(text.split())

    return tidy(removed), tidy(spaced)


def normalize_title(title: str) -> str:
    return title_variants(title)[0]
