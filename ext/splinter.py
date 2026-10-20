"""Stage 2: hyphen splinters on content words (normalization-invariant).

Why hyphens: NFKC leaves ASCII "-" alone, casefold does not touch it, and
whitespace collapse only rewrites runs of whitespace -- so a hyphen we plant
survives `--normalize` exactly, while every surface lever in ext/surface.py
(case, NBSP) is undone. The affected GPT-2 ids therefore stay changed on the
normalized, re-tokenized text the detector actually scores.

Two renderings (MODE):

- "lead" (word-initial): ``cat`` -> ``-cat`` written into the gap as
  ``prev -cat``.  GPT-2's pre-tokenizer takes `` -`` as one piece, so the
  word itself is re-tokenized *without* its leading ``Ġ`` -> its id (and the
  prev->cur link into it) rerolls, yet the unigram-F1 word stream still sees
  exactly ``cat``: **zero sim cost**.  This buys far more rerolls per sim
  point than internal splitting.

- "internal" (word-internal): ``become`` -> ``be-come``. The word splits
  into two unmatched word tokens, so it costs ~1 unigram-F1 match per
  splinter -- keep its rate low against the sim budget.

Split points are verified with ext/gpt2_bpe (pure-python GPT-2 BPE, no
transformers): a candidate is only used if its id sequence differs from the
original word's. Key-blind: no watermark key, no hash reimplementation.
"""
from __future__ import annotations

import re

import numpy as np

from ext.gpt2_bpe import load_bpe
from ext.funcword import FUNCTION_WORDS

# --- tunables -------------------------------------------------------------
MODE = "internal"      # "internal" (be-come) | "lead" (word-initial; tested
                        # on B_dev and REJECTED: "," -> ",-" raised unigram z)
RATE = 0.12            # P(splinter) per eligible content-word occurrence
MIN_LEN = 4            # shortest word we may prefix with "-"
INTERNAL_MIN_LEN = 6   # shortest word we may split internally (both parts >= 3)
MIN_TOKENS = 1         # only splinter words that BPE-split into >= N tokens
                        # (multi-subword words are the rare, high unigram-w
                        # ones, so each splinter cuts more z per F1 point)

_UNIT_RE = re.compile(r"\s+|[A-Za-z0-9']+|[^\sA-Za-z0-9']+")


def _word_ids(word: str) -> tuple[int, ...]:
    return load_bpe().word_ids(word)


def _tokenization_changed(orig: str, cand: str) -> bool:
    return _word_ids(cand) != _word_ids(orig)


def _internal_split(word: str, rng: np.random.Generator) -> str:
    """Word-internal hyphen at an rng-chosen point that changes BPE ids."""
    lo, hi = 3, len(word) - 3            # both parts at least 3 chars
    if hi < lo:
        return word
    ks = list(range(lo, hi + 1))
    rng.shuffle(ks)
    for k in ks:
        cand = word[:k] + "-" + word[k:]
        if _tokenization_changed(word, cand):
            return cand
    return word


def splinter(text: str, rng: np.random.Generator) -> str:
    mode = MODE
    rate = RATE
    out: list[str] = []
    prev_nonws = ""
    for u in _UNIT_RE.findall(text):
        is_ws = u.isspace()
        eligible = (
            not is_ws
            and u.isascii() and u.isalpha()
            and len(u) >= (MIN_LEN if mode == "lead" else INTERNAL_MIN_LEN)
            and u.lower() not in FUNCTION_WORDS
            and (MIN_TOKENS <= 1 or len(_word_ids(u)) >= MIN_TOKENS)
            and rng.random() < rate
        )
        if eligible:
            if mode == "lead":
                # don't stack onto a hyphen that is already there
                if prev_nonws != "-" and _tokenization_changed(u, "-" + u):
                    out.append("-" + u)
                    prev_nonws = u
                    continue
            else:
                cand = _internal_split(u, rng)
                if cand != u:
                    out.append(cand)
                    prev_nonws = u
                    continue
        out.append(u)
        if not is_ws:
            prev_nonws = u
    return "".join(out)
