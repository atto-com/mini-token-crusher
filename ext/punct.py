"""Stage 1.5: punctuation/symbol identity rework (normalization-invariant).

Why: after stage 1 nearly every *word* id is rerolled (tight commas strip the
leading space), yet ~60% of the original punctuation ids survive -- and
those still carry the watermark's green excess (measured on B_dev: unigram
z stays ~2.4 with only ~22% of original token ids preserved; substituting
punctuation alone drops it to ~1.0). Swapping surviving marks for
identity-different ones kills those links/memberships for free:

- the word stream (hence unigram-F1 sim) never counts punctuation or
  symbols, so substitution costs ZERO sim (only non-word, non-space units
  are touched; apostrophes *inside* words live in the word unit and stay);
- every replacement is NFKC-stable (NFKC(x) == x), casefold-stable and
  never whitespace, so `--normalize` cannot undo it;
- unmapped non-word chars (CJK, replacement chars, zero-width space,
  soft hyphen...) fall back to a generic stable-symbol pool.

Runs AFTER ext/surface (case/NBSP levers, undone by --normalize) and BEFORE
ext/funcword, so the commas stage 1 inserts stay fresh-rerolled random ids
and are not wasted on substitution.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

# --- tunables -------------------------------------------------------------
RATE = 1.00            # P(substitute) per punctuation/symbol character

# Alternatives per source char: all verified NFKC-stable + casefold-stable.
ALTS: dict[str, tuple[str, ...]] = {
    ".": ("·", ";", "?"),
    ",": ("·", ";", ":"),
    ";": (":", ",", "·"),
    ":": (";", ",", "·"),
    "?": ("!", "."),
    "!": ("?", "."),
    "(": ("[", "{"),
    ")": ("]", "}"),
    "[": ("(", "{"),
    "]": (")", "}"),
    "-": ("·", "–"),
    "–": ("—", "-"),
    "—": ("–", "-"),
    '"': ("`", "«"),
    "'": ('"', "`"),
    "`": ('"', "«"),
    "/": ("\\", "|"),
    "\\": ("/", "|"),
    "|": ("¦", "/"),
    "&": ("§", "+"),
    "=": ("≈", "~"),
    "*": ("×", "†"),
    "+": ("×", "±"),
    "_": ("–", "~"),
    "~": ("≈", "-"),
    "#": ("§", "¶"),
    "@": ("¶", "§"),
    "^": ("ˇ", "~"),
    "$": ("€", "¤"),
    "<": ("«", "‹"),
    ">": ("»", "›"),
    "§": ("¶", "#"),
    "•": ("·", "*"),
    "®": ("©", "*"),
    "«": ("»", "<"),
    "»": ("«", ">"),
    "−": ("-", "–"),
    "“": ('"', "`"),
    "”": ('"', "`"),
    "‘": ('"', "`"),
    "’": ('"', "`"),
    "​": ("·", "-"),     # zero-width space
    "­": ("·", "-"),     # soft hyphen
    "…": (".", "·"),         # NFKC(...) -> "..." (length change is fine)
}

# fallback pool for any other non-word, non-space char (CJK, U+FFFD, ...)
_GENERIC = ("·", "•", "§", "|", "~", "†", "¶", "*")

_NONWORD_RE = re.compile(r"\s+|[A-Za-z0-9']+|[^\sA-Za-z0-9']+")


def _alt(ch: str, rng: np.random.Generator) -> str:
    pool = ALTS.get(ch)
    if pool is None:
        pool = _GENERIC
    return pool[int(rng.integers(len(pool)))]


def rework(text: str, rng: np.random.Generator) -> str:
    rate = RATE
    out: list[str] = []
    for u in _NONWORD_RE.findall(text):
        if u.isspace() or re.fullmatch(r"[A-Za-z0-9']+", u):
            out.append(u)          # keep words (incl. in-word apostrophes)
            continue
        buf = []
        for ch in u:
            if rng.random() < rate:
                buf.append(_alt(ch, rng))
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "".join(out)


def _check_stability() -> None:
    for pool in list(ALTS.values()) + [_GENERIC]:
        for t in pool:
            assert unicodedata.normalize("NFKC", t) == t and t.casefold() == t, t
            # replacements must never enter the unigram-F1 word stream
            assert not re.fullmatch(r"[A-Za-z0-9']+", t), t
            assert not t.isspace(), t


_check_stability()
