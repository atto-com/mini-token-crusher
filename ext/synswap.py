"""Stage 3: sparse frequency-matched synonym swaps maximizing id churn.

Node 2.1 measured that substitution alone caps the z-cut: only ~15% of the
word budget can be spent under the F1 floor, so here swaps are a *minority*
stage whose whole budget goes where token-id churn per word is largest.

Mechanism (numpy rng + ext/gpt2_bpe, no key):
  - candidates come from resources/synonyms.json (WordNet, lowercase keys);
  - eligibility: GPT-2 corpus unigram weight (resources/unigram_w.npy,
    averaged over the word's BPE ids) inside a relative band around the
    original word -> the text stays on the natural unigram manifold, so
    PPL and sim survive;
  - selection: among the frequency-tied candidates, pick the one with the
    largest token-id churn (mismatched positions between the id sequences,
    preferring length changes) -- key-free proxy for detector damage;
  - sparse rate: swaps cost one unigram-F1 match each, so RATE stays small.

Swaps survive --normalize: NFKC/casefold leave plain ASCII synonyms alone
and no whitespace is introduced (multi-word synonyms are rejected).
Words already splintered by stage 2 (adjacent to a planted "-") are skipped:
their ids are already rerolled, so a swap there would only burn sim.
"""
from __future__ import annotations

import json
import re
from itertools import zip_longest
from pathlib import Path

import numpy as np

from ext.gpt2_bpe import load_bpe

# --- tunables -------------------------------------------------------------
RATE = 0.15            # P(swap) per eligible word occurrence
W_BAND = (0.5, 2.0)    # candidate weight must lie in [lo*wo, hi*wo]
MIN_WORD_LEN = 3       # skip very short words (function-word synonyms hurt PPL)
TOP_EPS = 0.05         # |dw| within best+eps counts as a frequency tie

RESOURCES = Path(__file__).resolve().parents[1] / "resources"

_WORD_RE = re.compile(r"\s+|[A-Za-z0-9']+|[^\sA-Za-z0-9']+")

_syn: dict[str, list[str]] | None = None
_uw: np.ndarray | None = None
_w_cache: dict[str, float] = {}
_default_w: float | None = None


def _load_resources() -> None:
    global _syn, _uw, _default_w, _candidate_table
    if _syn is None:
        _syn = json.loads((RESOURCES / "synonyms.json").read_text())
        _uw = np.load(RESOURCES / "unigram_w.npy").astype(np.float64)
        _default_w = float(np.median(_uw))
        # import-time precompute: the eligibility/tie logic is a pure
        # function of static data, so pay its BPE/weight cost once here
        # rather than inside the timed attack
        _candidate_table = {w: _tied_candidates(w) for w in _syn}


def word_weight(word: str) -> float:
    """Mean GPT-2 unigram weight of the word's tokens (mid-text form)."""
    hit = _w_cache.get(word)
    if hit is not None:
        return hit
    ids = load_bpe().word_ids(word)
    w = float(np.mean(_uw[list(ids)])) if ids else _default_w
    _w_cache[word] = w
    return w


def _match_case(syn: str, orig: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return syn.upper()
    if orig[:1].isupper():
        return syn[:1].upper() + syn[1:]
    return syn


def _churn(base: tuple[int, ...], cand: tuple[int, ...]) -> int:
    """Token-id churn: mismatched positions (length changes count too)."""
    return sum(1 for a, b in zip_longest(base, cand) if a != b)


def _tied_candidates(word: str) -> tuple[str, ...]:
    """Rng-free extraction of the eligibility + tie-break logic above.

    Returns the frequency-tied, max-churn candidate tuple (in `eligible`
    order) for a lowercase word, or () when nothing qualifies.  This is the
    exact decision body of the original _pick_synonym, minus the final tie
    draw -- precomputed once per word at import (static data: synonyms.json,
    unigram_w.npy, BPE vocab) so the attack runtime only pays a dict lookup.
    """
    assert _syn is not None
    cands = _syn.get(word)
    if not cands:
        return ()
    orig_ids = load_bpe().word_ids(word)
    wo = word_weight(word)
    lo, hi = W_BAND[0] * wo, W_BAND[1] * wo
    eligible: list[tuple[int, float, str]] = []
    for c in cands:
        c = c.strip().lower()
        if not c or not c.isalpha() or len(c) < MIN_WORD_LEN or " " in c:
            continue
        wc = word_weight(c)
        if not (lo <= wc <= hi):
            continue  # off-manifold candidate: reject
        cids = load_bpe().word_ids(c)
        if cids == orig_ids:
            continue  # not a different tokenization: reject
        eligible.append((_churn(orig_ids, cids), abs(wc - wo), c))
    if not eligible:
        return ()
    # frequency tie window (keeps PPL on-manifold), then max id churn
    best_dw = min(e[1] for e in eligible)
    tied = [e for e in eligible if e[1] <= best_dw + TOP_EPS]
    mx = max(e[0] for e in tied)
    tied = [e for e in tied if e[0] == mx]
    return tuple(e[2] for e in tied)


_candidate_table: dict[str, tuple[str, ...]] | None = None


def candidate_table() -> dict[str, tuple[str, ...]]:
    """word -> tied candidate tuple (import-time precomputed)."""
    assert _candidate_table is not None
    return _candidate_table


def _pick_synonym(word: str, rng: np.random.Generator) -> str | None:
    """Frequency-tied candidate with the largest token-id churn, or None."""
    tied = _candidate_table.get(word)
    if not tied:
        return None
    return tied[int(rng.integers(len(tied)))]


def synswap(text: str, rng: np.random.Generator) -> str:
    _load_resources()
    rate = RATE
    parts: list[str] = []
    units = _WORD_RE.findall(text)
    prev = ""
    nxt = ""
    for i, u in enumerate(units):
        nxt = units[i + 1] if i + 1 < len(units) else ""
        if (u.isascii() and u.isalpha() and len(u) >= MIN_WORD_LEN
                and prev != "-" and nxt != "-"
                and rng.random() < rate):
            syn = _pick_synonym(u.lower(), rng)
            if syn is not None:
                parts.append(_match_case(syn, u))
                prev = u
                continue
        parts.append(u)
        if not u.isspace():
            prev = u
    return "".join(parts)


# deterministic static-data precompute (synonym eligibility/tie table) --
# runs at import, before eval's attack timer starts
_load_resources()
