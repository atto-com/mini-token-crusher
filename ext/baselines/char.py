"""Baseline: character-level perturbation (arXiv 2509.09112).

"Character-Level Perturbations Disrupt LLM Watermarks": perturb only the
*character surface* of the text -- letter casing, inter-word whitespace
(NBSP / double space) and punctuation identity -- so GPT-2 BPE re-tokenizes
into different ids while the alphanumeric word stream (unigram-F1 sim) is
unchanged.  This is exactly the trunk's stage-0 + stage-1 levers
(ext/surface.py + ext/punct.py) running in ISOLATION: no funcword, no
splinter, no synswap, no crush.  Expected profile: strong raw kill (every
word id rerolled by case/NBSP), collapsed robust kill (NFKC+casefold+
whitespace collapse undoes every lever except punctuation substitution).
"""
from __future__ import annotations

import numpy as np

from ext import punct, surface


def attack_text(texts: list[str], rng: np.random.Generator) -> list[str]:
    # surface first, then punctuation: same order as the trunk pipeline, so
    # punctuation substitutions never land next to a freshly inserted NBSP.
    return [punct.rework(surface.perturb(t, rng), rng) for t in texts]
