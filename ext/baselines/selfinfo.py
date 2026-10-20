"""Baseline: self-information-weighted rewrite (arXiv 2505.05190).

"Self-Information Rewrite Attacks" found that rewriting the words a text
finds *most surprising* (highest self-information) damages watermark
detection most per unit of quality loss: informative tokens carry a larger
share of the detector's evidence than predictable ones.  Numpy-only
approximation of that scoring:

  self-info(word) = -log( mean GPT-2 unigram weight of the word's BPE ids )
                    (resources/unigram_w.npy; lower corpus weight = more
                    surprising = higher self-information)

Attack: rank all eligible content-word occurrences by self-information,
rewrite the TOP_FRAC most informative via the frequency-matched synonym
table (ext/synswap's precomputed frequency-tied max-churn candidates;
frequency-matched fallback keeps PPL on-manifold), SKIP any word with no
legal synonym, spend the rest of the F1 budget on nothing.  Top-si words
are content words by construction (function words are the most predictable
in the corpus), so no explicit stoplist is needed.  No other stages.
"""
from __future__ import annotations

import math

import numpy as np

from ext import synswap

TOP_FRAC = 0.70    # fraction of word occurrences to rewrite, by self-info rank
                   # (max unigram-F1-legal: 0.70 -> sim 0.832, 0.85 -> 0.789
                   #  below the 0.80 floor; measured on B_dev, 4x80 texts)

_W = synswap._WORD_RE
_MIN_LEN = synswap.MIN_WORD_LEN


def self_info(word: str) -> float:
    """-log corpus unigram weight (cached per lowercase word)."""
    return -math.log(synswap.word_weight(word.lower()))


def _rewrite(text: str, rng: np.random.Generator, top_frac: float) -> str:
    table = synswap.candidate_table()
    units = _W.findall(text)
    # rank eligible occurrences by self-information (ties: leftmost first)
    ranked: list[tuple[float, int]] = []
    for i, u in enumerate(units):
        prev = units[i - 1] if i > 0 else ""
        nxt = units[i + 1] if i + 1 < len(units) else ""
        if (u.isascii() and u.isalpha() and len(u) >= _MIN_LEN
                and prev != "-" and nxt != "-"):
            ranked.append((self_info(u), i))
    k = int(round(top_frac * len(ranked)))
    if k <= 0:
        return text
    ranked.sort(key=lambda p: (-p[0], p[1]))
    targets = {i for _, i in ranked[:k]}
    parts: list[str] = []
    for i, u in enumerate(units):
        if i in targets:
            tied = table.get(u.lower())
            if tied:
                syn = tied[int(rng.integers(len(tied)))]
                parts.append(synswap._match_case(syn, u))
                continue
        parts.append(u)          # no legal synonym -> skip (no budget spent)
    return "".join(parts)


def attack_text(texts: list[str], rng: np.random.Generator) -> list[str]:
    return [_rewrite(t, rng, TOP_FRAC) for t in texts]
