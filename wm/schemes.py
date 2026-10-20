"""Token-level decoding watermark schemes and detectors (numpy only).

Hash-based green-list decoders with distinct hash contexts and bias styles,
chosen so a successful attack must be scheme-agnostic:

- kgw:      green membership H(prev, token);  constant bias +/- delta.
- unigram:  green membership H(token);        green boost scaled by rarity weight.
- exp:      green membership H'(prev, token); bias scaled by per-step Exp(1) noise.
- synthid:  SynthID-Text-style hash tournament over the model's top-k
            candidate tokens, keyed by a hash of the whole context; the
            bracket winner's logit is boosted before sampling.

Hashing is a keyed vectorized integer hash (numpy uint64), fast over the full
vocab per decoding step. Detectors return a scalar z-score; thresholds are
calibrated empirically on null (unwatermarked) texts, so only monotonicity in
watermark strength matters.

The synthid detector is a z-score over per-position bracket-winner
indicators: at each position the same tournament is replayed from the
teacher-forced model logits, and the null expectation of "observed token ==
winner" is the model's own probability of that winner. `synthid_score` takes
logits as an argument (computed by the model owner in eval/prepare_cache),
so this module stays numpy-only.
"""
from __future__ import annotations

import hashlib
import numpy as np

WM_KEY = b"arbor-wm-attack-2026"
GAMMA = 0.5
DELTA_KGW = 2.0
DELTA_UNIGRAM = 1.5
DELTA_EXP = 2.5
DELTA_SYNTHID = 6.0
SYNTHID_TOPK = 16
SCHEMES = ("kgw", "unigram", "exp", "synthid")

_seed = np.frombuffer(hashlib.blake2b(WM_KEY, digest_size=48).digest(),
                      dtype=np.uint64)
_A = np.uint64(_seed[0] | np.uint64(1))
_B = np.uint64(_seed[1] | np.uint64(1))
_C = np.uint64(_seed[2] | np.uint64(1))
_D = np.uint64(_seed[3] | np.uint64(1))
_E = np.uint64(_seed[4] | np.uint64(1))
_F = np.uint64(_seed[5] | np.uint64(1))

# Separate salt/keys for the synthid tournament (own key material).
_SYN_SEED = np.frombuffer(
    hashlib.blake2b(WM_KEY + b"|synthid-tourney", digest_size=48).digest(),
    dtype=np.uint64)
_T_A = np.uint64(_SYN_SEED[0] | np.uint64(1))
_T_B = np.uint64(_SYN_SEED[1] | np.uint64(1))
_T_C = np.uint64(_SYN_SEED[2] | np.uint64(1))
_T_D = np.uint64(_SYN_SEED[3] | np.uint64(1))
_T_R = np.uint64(_SYN_SEED[4] | np.uint64(1))
_SYN_KEY = WM_KEY + b"|synthid-seed"


def _hash_green(t: np.ndarray, prev, salt: np.uint64) -> np.ndarray:
    """1 where token(s) are green. t: int64 array; prev: scalar or array."""
    t64 = t.astype(np.uint64)
    p64 = np.asarray(prev, dtype=np.uint64)
    x = t64 * _A + p64 * _B + np.uint64(salt)
    x ^= x >> np.uint64(31)
    x = x * _C
    x ^= x >> np.uint64(17)
    x = x * _D
    x ^= x >> np.uint64(23)
    return (x & np.uint64(1)).astype(np.int8)


def _exp_noise(prev: int, step: int) -> float:
    m = hashlib.blake2b(digest_size=8, key=WM_KEY + b"|expn")
    m.update(np.asarray([prev, step], dtype=np.uint64).tobytes())
    u = (int.from_bytes(m.digest(), "little") + 1) / 2**64
    return float(-np.log(u))


def vocab_green_unigram(vocab_size: int) -> np.ndarray:
    T = np.arange(vocab_size, dtype=np.int64)
    return _hash_green(T, np.uint64(0), np.uint64(0x756E))


def synthid_seed(context) -> np.uint64:
    """Keyed hash of the full context (tokens seen so far) -> tournament seed."""
    m = hashlib.blake2b(digest_size=8, key=_SYN_KEY)
    m.update(np.asarray(list(context), dtype=np.uint64).tobytes())
    return np.uint64(int.from_bytes(m.digest(), "little") | np.uint64(1))


def _tourney_round(players: np.ndarray, seeds: np.ndarray, rnd: int) -> np.ndarray:
    """One single-elimination round: hash-based pairwise matches. (T, 2m) -> (T, m)."""
    a = players[:, 0::2]
    b = players[:, 1::2]
    r_salt = np.uint64((rnd * int(_T_R)) & ((1 << 64) - 1))
    x = a * _T_A + b * _T_B + seeds + r_salt
    x ^= x >> np.uint64(31)
    x = x * _T_C
    x ^= x >> np.uint64(17)
    x = x * _T_D
    x ^= x >> np.uint64(23)
    return np.where((x & np.uint64(1)) == np.uint64(1), b, a)


def synthid_winners(cand: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Bracket winners. cand: (T, k) sorted candidate ids (k a power of 2);
    seeds: (T,) uint64 tournament seeds. Returns (T,) winner ids."""
    players = np.asarray(cand, dtype=np.uint64)
    seeds = np.asarray(seeds, dtype=np.uint64).reshape(-1, 1)
    rnd = 0
    while players.shape[1] > 1:
        players = _tourney_round(players, seeds, rnd)
        rnd += 1
    return players[:, 0]


def synthid_topk(logits: np.ndarray) -> np.ndarray:
    """Top-k candidate ids per row, ordered by (-logit, token id). logits: (T, V)."""
    logits = np.asarray(logits)
    k = min(SYNTHID_TOPK, logits.shape[-1])
    part = np.argpartition(logits, -k, axis=-1)[..., -k:]
    pl = np.take_along_axis(logits, part, axis=-1)
    # per-row stable order: primary -logit, secondary token id (axis-aware lexsort)
    order = np.lexsort((part, -pl), axis=-1)
    return np.take_along_axis(part, order, axis=-1)


def apply_bias(logits: np.ndarray, prev: int, step: int, scheme: str,
               unigram_green: np.ndarray | None = None,
               unigram_w: np.ndarray | None = None,
               context: list[int] | None = None) -> np.ndarray:
    """Biased logits for one decoding step. logits: (V,) float.

    `context` (tokens seen so far, including `prev`) is only used by synthid;
    it falls back to [prev] when omitted.
    """
    v = logits.shape[0]
    out = logits.astype(np.float64, copy=True)
    if scheme == "kgw":
        g = _hash_green(np.arange(v, dtype=np.int64), np.uint64(prev),
                        np.uint64(0x6B67))
        out += DELTA_KGW * (2 * g.astype(np.float64) - 1.0)
    elif scheme == "unigram":
        if unigram_green is None:
            unigram_green = vocab_green_unigram(v)
        w = np.ones(v) if unigram_w is None else np.clip(unigram_w, 0.0, 5.0)
        out += DELTA_UNIGRAM * unigram_green.astype(np.float64) * w
    elif scheme == "exp":
        g = _hash_green(np.arange(v, dtype=np.int64), np.uint64(prev),
                        np.uint64(0x6578))
        out += DELTA_EXP * _exp_noise(prev, step) * (2 * g.astype(np.float64) - 1.0)
    elif scheme == "synthid":
        top = synthid_topk(out[None, :])[0]
        seed = synthid_seed(context if context is not None else [prev])
        winner = int(synthid_winners(top[None, :], seed[None])[0])
        out[winner] += DELTA_SYNTHID
    else:
        raise ValueError(scheme)
    return out


def sample_from(logits: np.ndarray, rng: np.random.Generator) -> int:
    logits = logits - logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(logits.shape[0], p=p))


def score_tokens(tokens: np.ndarray, scheme: str, vocab_size: int,
                 unigram_w: np.ndarray | None = None) -> float:
    """Watermark z-score for one full sequence (prefix + continuation)."""
    toks = np.asarray(tokens, dtype=np.int64)
    if toks.size < 8:
        return 0.0
    prev, cur = toks[:-1], toks[1:]
    if scheme == "kgw":
        green = np.empty(cur.size, dtype=np.float64)
        for i, (p, c) in enumerate(zip(prev, cur)):
            green[i] = _hash_green(np.array([c], dtype=np.int64),
                                   np.uint64(int(p)), np.uint64(0x6B67))[0]
        w = np.ones_like(green)
    elif scheme == "unigram":
        green = _hash_green(cur, np.uint64(0), np.uint64(0x756E)).astype(np.float64)
        if unigram_w is None:
            unigram_w = np.ones(vocab_size, dtype=np.float64)
        w = np.clip(unigram_w[cur], 0.0, 5.0).astype(np.float64)
    elif scheme == "exp":
        green = np.empty(cur.size, dtype=np.float64)
        for i, (p, c) in enumerate(zip(prev, cur)):
            green[i] = _hash_green(np.array([c], dtype=np.int64),
                                   np.uint64(int(p)), np.uint64(0x6578))[0]
        w = np.ones_like(green)
    else:
        raise ValueError(scheme)
    g = float((green * w).sum())
    mean = GAMMA * float(w.sum())
    var = GAMMA * (1.0 - GAMMA) * float((w ** 2).sum())
    if var <= 1e-12:
        return 0.0
    return (g - mean) / np.sqrt(var)


def fast_score_batch(seqs, scheme: str, vocab_size: int,
                     unigram_w: np.ndarray | None = None) -> np.ndarray:
    """Vectorized-ish scores for a list of token-id arrays."""
    return np.array([score_tokens(s, scheme, vocab_size, unigram_w) for s in seqs],
                    dtype=np.float64)


def synthid_score(logits: np.ndarray, tokens: np.ndarray,
                  row_start: int) -> tuple[float, float]:
    """SynthID-Text tournament z-score for one sequence.

    logits: (L-1, V) teacher-forced logits where row j predicts tokens[j+1]
    (fewer rows are tolerated: scoring stops at the last available row).
    tokens: (L,) the sequence's token ids (prefix + continuation).
    row_start: first scored row; callers pass prefix_len - 1 so only the
    continuation positions (whose contexts the watermarking controlled) count.

    Per position the same top-k tournament is replayed from the logits; the
    indicator is 1 when the observed token is the bracket winner, and the
    null expectation is the model's own probability of that winner (tokens
    were sampled from the model under H0), giving a z-like aggregate.
    Returns (z, indicator_rate).
    """
    toks = np.asarray(tokens, dtype=np.int64)
    n_rows = int(np.asarray(logits).shape[0])
    L = min(toks.size, n_rows + 1)
    row_start = max(int(row_start), 0)
    if L - 1 - row_start < 4:
        return 0.0, 0.0
    rows = np.asarray(logits[row_start:L - 1], dtype=np.float64)  # (T, V)
    T = rows.shape[0]
    cand = synthid_topk(rows)  # (T, k)
    seeds = np.array([synthid_seed(toks[:j + 1])
                      for j in range(row_start, L - 1)], dtype=np.uint64)
    winners = synthid_winners(cand, seeds)  # (T,)
    obs = toks[row_start + 1:L]  # observed tokens predicted by those rows
    ind = (winners.astype(np.int64) == obs)
    m = rows.max(axis=1, keepdims=True)
    e = np.exp(rows - m)
    p0 = e[np.arange(T), winners.astype(np.int64)] / e.sum(axis=1)
    var = float(np.sum(p0 * (1.0 - p0)))
    z = (float(ind.sum()) - float(p0.sum())) / np.sqrt(var) if var > 1e-12 else 0.0
    return float(z), float(ind.mean())
