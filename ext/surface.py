"""Surface-form perturbation (node 1.1).

Key-blind attack: rewrite only the *surface* of each text -- letter casing,
inter-word whitespace (NBSP / double space), and punctuation/segment spacing
(NBSP inserted at char-class boundaries inside non-space runs) -- so that
GPT-2 BPE re-tokenization yields different token ids (case- and
space-sensitive merges) while the alphanumeric word stream (hence
unigram-F1 sim) is unchanged.

Levers and why they matter for each detector family in wm/schemes.py:
- case flips: GPT-2 stores case variants as distinct BPE tokens, so every
  flipped word changes its id -> breaks kgw/exp *links* (H(prev, token)) and
  unigram *memberships* (H(token)).  UPPERCASE is used because it fragments
  words into short capital pieces that are *cheap* under GPT-2 perplexity
  (measured: all-upper halves ppl vs the watermarked original), whereas
  title-casing/lowercasing raises it.
- NBSP (U+00A0) between words: not a regular space in BPE, so the following
  word loses its leading-"Ġ" merge and gets a different standalone id;
  additionally inserts an NBSP token, breaking neighbor links.
- double space: inserts a separate " " token before the next word; the word
  keeps its "Ġword" id but every neighboring prev-link shifts.
- NBSP inserted at word/non-word boundaries *inside* a run (punctuation
  runs, glued zero-width runs, CJK segments): re-chunks punctuation and
  symbol BPE merges (quote walls, "[276]", "a​b" glue) without ever
  splitting an [A-Za-z0-9']+ word, so unigram-F1 similarity is untouched.
  This is the only lever that changes token ids in digit/punctuation/CJK
  heavy degenerate texts where case flips are no-ops.

All choices are drawn from an rng stream (key-blind: no watermark key, no
reimplementation of the detection hash is used here).
"""
from __future__ import annotations

import re

NBSP = " "

# Variant probabilities (tuned on B_dev; see results/1.1-surface-form-perturbation/).
P_LOWER = 0.00      # word -> lowercase (expensive PPL; rarely needed)
P_TITLE = 0.00      # word -> Capitalized (expensive PPL)
P_UPPER = 1.00      # word -> UPPERCASE (cheap: fragments into predictable pieces)
P_GAP_NBSP = 0.85   # whitespace gap -> NBSP (breaks the leading-Ġ merge)
P_GAP_DOUBLE = 1.00 # whitespace gap -> double space (shifts every prev link)
P_INS_BOUND = 0.55  # insert NBSP at word<->non-word boundary inside a run
P_INS_PUNCT = 0.85  # insert NBSP inside a pure non-word (punct/symbol) run

_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")

_RUN_RE = re.compile(r"\s+|\S+")


def _case_variant(run: str, rng) -> str:
    if not any(ch.isalpha() for ch in run):
        return run
    r = rng.random()
    if r < P_LOWER:
        return run.lower()
    if r < P_TITLE:
        out = []
        seen = False
        for ch in run:
            if ch.isalpha() and not seen:
                out.append(ch.upper())
                seen = True
            elif ch.isalpha():
                out.append(ch.lower())
            else:
                out.append(ch)
        return "".join(out)
    if r < P_UPPER:
        return run.upper()
    return run


def _gap_variant(ws: str, rng) -> str:
    # keep structural newlines/tabs intact; only pure-space gaps are attacked
    if ws.strip(" ") != "":
        return ws
    r = rng.random()
    if r < P_GAP_NBSP:
        return NBSP
    if r < P_GAP_DOUBLE:
        return "  "
    return " "


def _split_run(run: str, rng) -> str:
    """Insert NBSP at char-class boundaries inside `run` (never inside an
    [A-Za-z0-9']+ word), re-chunking punctuation/symbol/glued BPE merges."""
    if len(run) < 2:
        return run
    out = []
    for i, ch in enumerate(run):
        if i > 0:
            a, b = run[i - 1], ch
            wa, wb = a in _WORD_CHARS, b in _WORD_CHARS
            if wa != wb:
                if rng.random() < P_INS_BOUND:
                    out.append(NBSP)
            elif not wa and not wb:
                if rng.random() < P_INS_PUNCT:
                    out.append(NBSP)
        out.append(ch)
    return "".join(out)


def perturb(text: str, rng) -> str:
    parts = []
    for m in _RUN_RE.finditer(text):
        s = m.group()
        if s[0].isspace():
            parts.append(_gap_variant(s, rng))
        else:
            parts.append(_case_variant(_split_run(s, rng), rng))
    return "".join(parts)
