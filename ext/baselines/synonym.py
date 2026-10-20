"""Baseline: frequency-matched synonym substitution (WaterPark, 2411.13425).

WaterPark's edit-suite family attacks watermarks with lexical substitution
while holding the text on the natural-frequency manifold.  This baseline is
ext/synswap's logic STANDALONE (no surface, punct, funcword, splinter or
crush stages) at its maximum unigram-F1-legal rate: it is a baseline, so it
spends the whole similarity budget (RATE pushed toward the F1 0.80 floor).

Selection mirrors ext/synswap exactly: candidates from resources/
synonyms.json, frequency-band eligibility vs the GPT-2 unigram weight, the
frequency-tied max-token-id-churn pick (import-time precomputed table).
"""
from __future__ import annotations

import numpy as np

from ext import synswap

# Max unigram-F1-legal rate (F1 floor 0.80; each swap costs one F1 match,
# eligibility gates keep the realized sim above the floor -- measured on
# B_dev: 0.75 -> sim 0.821, 0.85 -> 0.797 below the floor).
RATE = 0.75

_W = synswap._WORD_RE
_MIN_LEN = synswap.MIN_WORD_LEN


def _swap(text: str, rng: np.random.Generator, rate: float) -> str:
    table = synswap.candidate_table()
    parts: list[str] = []
    units = _W.findall(text)
    prev = ""
    for i, u in enumerate(units):
        nxt = units[i + 1] if i + 1 < len(units) else ""
        if (u.isascii() and u.isalpha() and len(u) >= _MIN_LEN
                and prev != "-" and nxt != "-" and rng.random() < rate):
            tied = table.get(u.lower())
            if tied:
                syn = tied[int(rng.integers(len(tied)))]
                parts.append(synswap._match_case(syn, u))
                prev = u
                continue
        parts.append(u)
        if not u.isspace():
            prev = u
    return "".join(parts)


def attack_text(texts: list[str], rng: np.random.Generator) -> list[str]:
    return [_swap(t, rng, RATE) for t in texts]
