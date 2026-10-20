"""Stage 1: function-word deletion + boundary punctuation rework (node 3.1).

Mechanism (numpy RNG only, text-level, no tokenizer/model access):
  1. Delete expendable words -- articles, pronouns, prepositions -- sampled by
     the RNG (along with one adjacent whitespace run so words never glue).
  2. Insert boundary punctuation at random clause-like positions between
     surviving words (and beside existing marks), rendered TIGHT: the comma
     takes the place of the following space, so the next word loses its
     leading-space BPE token. A low rate also inserts another expendable word.

Why it survives `--normalize` (NFKC + casefold + whitespace collapse):
commas are untouched by NFKC/casefold, the tight rendering has no whitespace
to collapse, and deleted words stay deleted -- so every (prev, cur) link the
watermark hashed is still broken after normalization, while the surface
levers in ext/surface.py (case / NBSP) are all undone by it.

Effect on the detector:
  - every deletion destroys two prev->cur hash links and every boundary
    insertion destroys one (plus the outgoing link of the rerolled word),
    replacing them with pairs the decoder never sampled under the watermark
    (green with ~0.5) -- this erases the KGW/exp green-count excess;
  - the tight rendering re-tokenizes every following word (space-less BPE
    variant), rerolling its unigram-scheme green status and rarity weight;
  - content words are kept, so unigram-F1 similarity stays high; comma
    mass sits below the text's mean NLL, so ppl_ratio stays low.

Deterministic given rng; same length/order in, same length/order out.
"""
from __future__ import annotations

import re

import numpy as np

# --- tunables -------------------------------------------------------------
DEL_RATE = 0.00         # P(delete a function word) -- stage-gated OFF for
                         # --normalize: deletions cost ~0.06 F1 sim each while
                         # *raising* unigram z (punctuation re-roll does the killing);
INS_RATE = 1.00        # P(punctuation at a candidate word boundary)
WIN_RATE = 0.00        # P(insert an extra function word at a boundary)
START_RATE = 1.00      # P(insert punctuation at the very start of the text)

# Commas are the cheap boundary mark: a comma's NLL sits below the mean NLL
# of the scored text, so dense comma insertion both refreshes hash links and
# dilutes mean NLL (measured ppl_ratio vs. cache: 0.98-1.04 at DEL 0.875).
_PUNCT = (",", ",", ",")
_P_PUNCT = (0.7, 0.2, 0.1)

# articles + pronouns (incl. possessives/demonstratives/interrogatives) +
# common prepositions / particles
FUNCTION_WORDS = frozenset("""
a an the
i you he she it we they me him her us them mine yours his hers its ours theirs
my your his its our their this that these those
who whom whose which what whoever whatever
of in on at to for with by from as into onto over under between through
during before after above below about against among around across behind
beyond near past since toward towards upon until till than via
up down off out in inside outside along throughout within without
per like else such
""".split())

_INSERTABLE = tuple(sorted(FUNCTION_WORDS))

_UNIT_RE = re.compile(r"\s+|[A-Za-z0-9']+|[^\sA-Za-z0-9']+")


def _is_ws(u: str) -> bool:
    return u.isspace()


def _is_word(u: str) -> bool:
    return bool(u) and (u[0].isascii() and (u[0].isalnum() or u[0] == "'"))


def _is_other(u: str) -> bool:
    return bool(u) and not _is_ws(u) and not _is_word(u)


def _p(rng: np.random.Generator) -> str:
    return _PUNCT[int(rng.choice(len(_PUNCT), p=_P_PUNCT))]


def surgery(text: str, rng: np.random.Generator) -> str:
    units = _UNIT_RE.findall(text)
    if not units:
        return text
    n = len(units)

    # --- pass 1: delete function words (+ one adjacent whitespace run) ----
    keep = [True] * n
    for i, u in enumerate(units):
        if not _is_word(u) or u.lower() not in FUNCTION_WORDS:
            continue
        if rng.random() >= DEL_RATE:
            continue
        keep[i] = False
        if i > 0 and _is_ws(units[i - 1]) and keep[i - 1]:
            keep[i - 1] = False
        elif i + 1 < n and _is_ws(units[i + 1]):
            keep[i + 1] = False
    out = [u for i, u in enumerate(units) if keep[i]]
    if not out:
        return text

    # --- pass 2: boundary insertions, tight rendering ---------------------
    res: list[str] = []

    # start of text (destroys the prefix->first-token link; tight)
    if rng.random() < START_RATE:
        res.append(_p(rng))
        if _is_ws(out[0]) and len(out) > 1 and _is_word(out[1]):
            out = out[1:]

    i = 0
    while i < len(out):
        u = out[i]
        if _is_other(u):
            ins = rng.random() < INS_RATE
            if ins:
                if rng.random() < 0.5:
                    res.append(_p(rng))
                    res.append(u)
                else:
                    res.append(u)
                    res.append(_p(rng))
            else:
                res.append(u)
            # swallow the whitespace after this mark cluster so the next
            # word starts without its space (rerolls its BPE token)
            if (i + 1 < len(out) and _is_ws(out[i + 1])
                    and i + 2 < len(out) and _is_word(out[i + 2])):
                i += 2
                continue
            i += 1
            continue

        res.append(u)
        if _is_word(u) and i + 1 < len(out) and _is_ws(out[i + 1]):
            ins = rng.random() < INS_RATE
            win = rng.random() < WIN_RATE
            swallowed = False
            if win:
                res.append(out[i + 1])         # reuse the whitespace
                res.append(_INSERTABLE[int(rng.integers(len(_INSERTABLE)))])
            if ins:
                res.append(_p(rng))            # tight: no space after comma
                if i + 2 < len(out) and _is_word(out[i + 2]):
                    swallowed = True
            if swallowed:
                i += 2
                continue
            if win and not ins:
                res.append(" ")
                i += 2
                continue
            i += 1
            continue
        i += 1
    return "".join(res)
