"""Canonical Hebrew text cleaner - one implementation, so every author is cleaned identically and
preprocessing cannot become a cross-author confound.

`strip_nikud` removes only Unicode *combining* marks, which PRESERVES U+05BE maqaf (the
word-joining hyphen). A character-range regex deletes maqaf instead and fuses words
(מלחמת־ששת־הימים → מלחמתששתהימים), so do not reintroduce one.
"""
import re
import unicodedata

# Ben-Yehuda's editorial apparatus. Cleaning lives here, not in run_dicta, so every consumer of the
# stripped text sees the same document Dicta saw - a partial reimplementation skewed punctuation
# rates by author.
FOOTER = "את הטקסט"
DROP_PREFIX = ("נדפס", "עי'", "[")


def clean_paragraphs(raw: str) -> list:
    """Drop the footer, title line, bare section numbers, footnote lines and stray entities.
    Returns the paragraphs Dicta actually analyses."""
    raw = raw.split(FOOTER)[0]
    paras = []
    for ln in raw.splitlines()[1:]:
        s = ln.strip()
        if not s or re.fullmatch(r"\d+", s):
            continue
        if s.startswith(DROP_PREFIX):
            continue
        s = s.replace("&nbsp;", " ").replace("↩", "")
        paras.append(s.strip())
    return paras


def clean_text(raw: str) -> str:
    """The cleaned document as one string - what raw-text features must be computed over."""
    return "\n".join(clean_paragraphs(raw))


def strip_nikud(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.combining(ch))
