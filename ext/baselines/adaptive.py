"""Baseline: adaptive white-box ORACLE ceiling (arXiv 2410.02440 attack model).

"Optimizing Adaptive Attacks against Content Watermarks" studies attackers
that adapt to the *detector statistic* instead of attacking blind.  This
module is that attacker: the ORACLE row of the benchmark.  It starts from the
key-blind trunk pipeline's output (solution.attack_text) and then runs a
greedy local search over the pipeline's own cheap edit levers, using the
detector's z-score as the acceptance feedback, so its final TPR is an upper
bound for any attack that may only edit text under the same quality gates.

THREAT MODEL / FEEDBACK BOUNDARY (important):
  - ALLOWED: querying the detector through its public scoring interface --
    wm.schemes.score_tokens / fast_score_batch and wm.model.synthid_batch_scores
    -- exactly the calls eval.py makes, on exactly the token ids eval.py
    scores (prefix + tokenizer.encode(normalize(text))).  The watermark key
    is therefore available only *implicitly* through the score, which is the
    definition of the white-box/adaptive attacker in 2410.02440.  The
    operating threshold is the same null-calibration quantile eval computes
    (public calibration, not key material).
  - FORBIDDEN: reading the key's per-token decisions -- no _hash_green call,
    no green/red list inspection, no WM_KEY access, no per-token membership
    oracle.  Candidate edits are proposed by generic text levers and accepted
    or rejected purely by the aggregate z-score.  The score function is the
    only interface to the watermark.

MECHANISM:
  1. run the trunk pipeline (byte-identical base, same rng stream);
  2. score every attacked text under all 4 schemes; texts whose threshold-
     rectified z ("detection risk" J = sum_s w_s * max(z_s - thr_s, 0)) is
    positive are the only ones that can still cost TPR points, so only they
    are searched (the rest are already null-like);
  3. per round, propose cheap single-edit variants (punctuation identity,
    whitespace/case surface flips in raw mode, hyphen splinters, tied
    synonym alternates, boundary-comma tighten/untighten), cheap-score them
    under kgw/unigram/exp, verify the best few under the (expensive) synthid
    tournament, and accept the best flip that strictly lowers J;
  4. stop at ROUNDS accepted flips per text, at the sim floor (0.82 per
    text, mirroring eval's 0.80 mean gate with margin) and at MAX_PAID
    sim-costing flips -- so the oracle cannot buy z with text quality.

The attack is scheme-DEPLOYED-aware in the sense of 2410.02440: the batch's
watermark scheme is inferred from score signatures alone (the original texts
score z >> 1 only under the scheme that generated them), and the objective
weights that scheme 1.0 and the other three 0.25 (ADAPTIVE_XW; 1.0 = plain
joint-sum objective, ADAPTIVE_OBJECTIVE=joint).

Budget knobs (env): ADAPTIVE_ROUNDS (accepted flips per text; 0 = pure
trunk = budget point 0 of the trade curve), ADAPTIVE_CANDS (candidates
scored per text per round), ADAPTIVE_VERIFY(_SYN) (candidates per text per
round verified under synthid), ADAPTIVE_SECS (wall-clock cap per call).
"""
from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np

import solution
from ext import funcword, punct, splinter, synswap
from wm import model, schemes

SCHEMES = ("kgw", "unigram", "exp", "synthid")
CHEAP = ("kgw", "unigram", "exp")

ROUNDS = int(os.environ.get("ADAPTIVE_ROUNDS", "4"))
CANDS = int(os.environ.get("ADAPTIVE_CANDS", "20"))
VERIFY = int(os.environ.get("ADAPTIVE_VERIFY", "2"))
VERIFY_SYN = int(os.environ.get("ADAPTIVE_VERIFY_SYN", "3"))
XW = float(os.environ.get("ADAPTIVE_XW", "0.25"))
MAX_PAID = int(os.environ.get("ADAPTIVE_MAX_PAID", "3"))
SIM_FLOOR = float(os.environ.get("ADAPTIVE_SIM_FLOOR", "0.82"))
MAX_SECS = float(os.environ.get("ADAPTIVE_SECS", "45"))
OBJECTIVE = os.environ.get("ADAPTIVE_OBJECTIVE", "deployed")
EPS = 1e-9

NBSP = "\u00a0"
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_UNIT_RE = re.compile(r"\s+|[A-Za-z0-9']+|[^\sA-Za-z0-9']+")

_CTX: dict | None = None
_THRESH: dict[str, float] | None = None
STATS: dict = {}


def bind_eval_context(ctx: dict) -> None:
    """Wire in the harness-side context (eval.py, node-9 additive loader)."""
    global _CTX
    _CTX = ctx


# --- scoring: the ONLY watermark interface this module may use -------------

def _norm(text: str) -> str:
    return _CTX["normalize_text"](text) if _CTX["normalize"] else text


def _ids(i: int, text: str) -> list[int]:
    return list(_CTX["prefixes"][i]) + _CTX["tokenizer"].encode(_norm(text))


def _z_cheap(ids: list[int]) -> dict[str, float]:
    c = _CTX
    return {s: float(schemes.score_tokens(ids, s, c["vocab_size"], c["unigram_w"]))
            for s in CHEAP}


def _syn(ids_list: list[list[int]]) -> np.ndarray:
    """Synthid z-scores (length-grouped, memory-bounded teacher-forced passes).

    The trunk pipeline occasionally emits a degenerate ~700-token text; if it
    shared a padded chunk with normal ~150-token ones, every chunk would pad
    to its width and the logits tensor alone would be ~2 GB -- which OOMs the
    MPS pool after a handful of calls.  So sequences are scored in ascending-
    length groups (splitting when a group fills up or the width would >1.5x)
    with a smaller forward batch for wide groups.
    """
    n_seq = len(ids_list)
    out = np.zeros(n_seq, dtype=np.float64)
    if n_seq == 0:
        return out
    lens = [len(s) for s in ids_list]
    order = sorted(range(n_seq), key=lambda i: lens[i])
    group: list[int] = []
    gmax = 0

    def flush(g: list[int]) -> None:
        seqs = [ids_list[i] for i in g]
        b = 8 if max(len(s) for s in seqs) > 256 else 16
        out[g] = model.synthid_batch_scores(seqs, _CTX["plen"], batch=b)

    for i in order:
        L = lens[i]
        if group and (len(group) >= 16 or L > 1.5 * gmax):
            flush(group)
            group, gmax = [], 0
        group.append(i)
        gmax = max(gmax, L)
    flush(group)
    try:
        model.torch.mps.empty_cache()   # synthid forwards fragment the MPS pool
    except Exception:
        pass
    return out


def _ensure_thresholds() -> dict[str, float]:
    """Reproduce eval's null calibration exactly (quantile 0.99 per scheme)."""
    global _THRESH
    if _THRESH is not None:
        return _THRESH
    c = _CTX
    null = [list(map(int, s)) for s in c["null_ids"]]
    if c["normalize"]:
        nm = c["normalize_text"]
        null = [c["prefixes"][i] + c["tokenizer"].encode(nm(model.decode(s[c["plen"]:])))
                for i, s in enumerate(null)]
    thr = {s: float(np.quantile(
        schemes.fast_score_batch(null, s, c["vocab_size"], c["unigram_w"]), 0.99))
        for s in CHEAP}
    thr["synthid"] = float(np.quantile(_syn(null), 0.99))
    _THRESH = thr
    return thr


def _detect(texts: list[str]) -> str:
    """Infer the batch's deployed scheme from score signatures alone.

    On watermarked originals only the generating scheme scores high
    (measured B_dev: matching mean z 5.2-7.5, others |z| <= 0.45), so the
    argmax of the cheap-scheme mean z with a 3.0 floor identifies it without
    ever touching the key.
    """
    ids = [_ids(i, t) for i, t in enumerate(texts)]
    best, bz = None, 0.0
    for s in CHEAP:
        m = float(np.mean(schemes.fast_score_batch(
            ids, s, _CTX["vocab_size"], _CTX["unigram_w"])))
        if m > bz:
            best, bz = s, m
    return best if (best is not None and bz >= 3.0) else "synthid"


def _w(scheme: str, match: str | None) -> float:
    return 1.0 if (match is None or scheme == match) else XW


def _J(z: dict[str, float], match: str | None) -> float:
    """Threshold-rectified detection risk = the score this oracle minimizes."""
    tot = 0.0
    for s in SCHEMES:
        ex = z[s] - _THRESH[s]
        if ex > 0.0:
            tot += _w(s, match) * ex
    return tot


def _J_cheap(z: dict[str, float], syn_z: float, match: str | None) -> float:
    """Cheap-scheme ranking proxy: same J with the synthid term held fixed."""
    tot = max(syn_z - _THRESH["synthid"], 0.0) * _w("synthid", match)
    for s in CHEAP:
        ex = z[s] - _THRESH[s]
        if ex > 0.0:
            tot += _w(s, match) * ex
    return tot


# --- quality mirror (eval.py's unigram_f1, verbatim semantics) -------------

def _f1(a: str, b: str) -> float:
    ca, cb = Counter(_WORD_RE.findall(a.lower())), Counter(_WORD_RE.findall(b.lower()))
    overlap = sum((ca & cb).values())
    na, nb = sum(ca.values()), sum(cb.values())
    if na == 0 or nb == 0:
        return 0.0
    p, r = overlap / nb, overlap / na
    return 2 * p * r / (p + r) if (p + r) else 0.0


# --- candidate levers (text-level, no key material) ------------------------

def _candidates(text: str, surface: bool) -> list[tuple[str, bool]]:
    """Single-edit variants of `text`; flag = costs unigram-F1 (paid).

    `surface` is False under --normalize (case/whitespace flips are undone
    there, so proposing them would only waste candidate slots).
    """
    spans = [(m.start(), m.end(), m.group()) for m in _UNIT_RE.finditer(text)]
    n = len(spans)
    out: list[tuple[str, bool]] = []
    for k, (s, e, u) in enumerate(spans):
        prev_u = spans[k - 1][2] if k else ""
        next_u = spans[k + 1][2] if k + 1 < n else ""
        if u.isspace():
            if surface and u.strip(" ") == "" and len(u) <= 2:
                for g in (NBSP, "  "):
                    if g != u:
                        out.append((text[:s] + g + text[e:], False))
                # drop the space after a mark -> the next word loses its
                # leading-space merge (tight rendering), zero sim cost
                if u == " " and prev_u and not _WORD_RE.fullmatch(prev_u) \
                        and _WORD_RE.fullmatch(next_u or ""):
                    out.append((text[:s] + text[e:], False))
            if u == " " and k == 0:
                out.append((text[:s] + text[e:], False))
            continue
        if _WORD_RE.fullmatch(u):
            low = u.lower()
            if surface and u.isalpha():
                out.append((text[:s] + (u.lower() if u.isupper() else u.upper())
                            + text[e:], False))
            if u.isascii() and u.isalpha() and len(u) >= 6 \
                    and low not in funcword.FUNCTION_WORDS \
                    and prev_u != "-" and next_u != "-":
                for mid in sorted({3, len(u) // 2, len(u) - 3}):
                    if 3 <= mid <= len(u) - 3:
                        w2 = u[:mid] + "-" + u[mid:]
                        if splinter._tokenization_changed(u, w2):
                            out.append((text[:s] + w2 + text[e:], True))
            if "-" in u:
                parts = u.split("-")
                if len(parts) == 2 and all(len(p) >= 3 for p in parts):
                    out.append((text[:s] + parts[0] + parts[1] + text[e:], True))
            if len(u) >= 3 and prev_u != "-" and next_u != "-":
                for c in synswap.candidate_table().get(low, ())[:4]:
                    c2 = synswap._match_case(c, u)
                    if c2 != u:
                        out.append((text[:s] + c2 + text[e:], True))
            # tighten a spaced boundary into a comma (next word re-tokenizes)
            if next_u == " " and k + 2 < n and _WORD_RE.fullmatch(spans[k + 2][2]):
                out.append((text[:e] + "," + text[e + 1:], False))
            continue
        # punctuation / symbol run: identity alternates (NFKC-stable, free)
        for j in range(s, e):
            ch = text[j]
            if ch.isspace():
                continue
            for alt in punct.ALTS.get(ch, punct._GENERIC):
                if alt != ch:
                    out.append((text[:j] + alt + text[j + 1:], False))
        # untighten: insert a space between a mark cluster and the next word
        if _WORD_RE.fullmatch(next_u or ""):
            out.append((text[:e] + " " + text[e:], False))
    return out


# --- the oracle ------------------------------------------------------------

def attack_text(texts: list[str], rng: np.random.Generator) -> list[str]:
    if _CTX is None:
        raise SystemExit("adaptive baseline: eval must call bind_eval_context()")
    orig = list(texts)
    base = solution.attack_text(orig, rng)      # trunk pipeline, byte-identical
    _ensure_thresholds()
    match = _detect(orig) if OBJECTIVE == "deployed" else None
    allow_surface = not _CTX["normalize"]

    all_ids = [_ids(i, t) for i, t in enumerate(base)]
    syn_z = list(_syn(all_ids))                 # one batched gate pass
    states = [{"i": i, "text": t, "flips": 0, "paid": 0,
               "z": {**_z_cheap(all_ids[i]), "synthid": float(syn_z[i])}}
              for i, t in enumerate(base)]

    verify_n = VERIFY_SYN if match == "synthid" else VERIFY
    hot = [st for st in states if _J(st["z"], match) > EPS]
    n_hot0, n_flips, n_paid = len(hot), 0, 0
    deadline = os.times().elapsed + MAX_SECS
    rnd = 0
    while hot and rnd < ROUNDS and os.times().elapsed < deadline:
        rnd += 1
        pool: list[tuple] = []                  # (hi, j_cheap, text, paid, sim, z, ids)
        for hi, st in enumerate(hot):
            cands = _candidates(st["text"], allow_surface)
            rng.shuffle(cands)
            seen = {st["text"]}
            tried, best = 0, []
            for ct, paid in cands:
                if tried >= CANDS:
                    break
                if ct in seen:
                    continue
                seen.add(ct)
                if paid and st["paid"] >= MAX_PAID:
                    continue
                s2 = _f1(orig[st["i"]], _norm(ct))
                if s2 < SIM_FLOOR:
                    continue
                tried += 1
                ids = _ids(st["i"], ct)
                z2 = _z_cheap(ids)
                best.append((_J_cheap(z2, st["z"]["synthid"], match),
                             ct, paid, s2, z2, ids))
            best.sort(key=lambda b: b[0])
            pool.extend((hi,) + b for b in best[:verify_n])
        if not pool:
            break
        zs = list(_syn([p[6] for p in pool]))   # one batched verification pass
        accept: dict[int, tuple] = {}
        for p, zsyn in zip(pool, zs):
            hi, j2, ct, paid, s2, z2, ids = p
            jfull = _J({**z2, "synthid": float(zsyn)}, match)
            if jfull < _J(hot[hi]["z"], match) - EPS and \
                    (hi not in accept or jfull < accept[hi][0]):
                accept[hi] = (jfull, ct, paid, s2, {**z2, "synthid": float(zsyn)})
        for hi, (jfull, ct, paid, s2, z) in accept.items():
            st = hot[hi]
            st.update(text=ct, z=z, flips=st["flips"] + 1,
                      paid=st["paid"] + (1 if paid else 0))
            n_flips += 1
            n_paid += 1 if paid else 0
        hot = [st for st in hot
               if _J(st["z"], match) > EPS and st["flips"] < ROUNDS
               and st["paid"] <= MAX_PAID]

    STATS.setdefault("hot0", 0)
    STATS["hot0"] += n_hot0
    STATS["flips"] = STATS.get("flips", 0) + n_flips
    STATS["paid"] = STATS.get("paid", 0) + n_paid
    STATS["match"] = STATS.get("match", []) + [match]
    STATS["rounds"] = rnd
    return [str(st["text"]) for st in states]
