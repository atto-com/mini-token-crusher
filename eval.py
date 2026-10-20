"""Attack benchmark evaluation.

Usage:
  python eval.py --split dev [--limit N] [--node-id ID] [--noop] [--normalize]
                 [--baseline {char,synonym,selfinfo,adaptive}]

Pipeline per split (uses prebuilt cache from scripts/prepare_cache.py):
  1. decode watermarked continuation texts
  2. apply solution.attack_text (numpy-only attack surface, enforced);
     --noop skips the attack (identity transform) for baseline measurement
  3. re-tokenize, score with each scheme detector vs null-calibrated
     thresholds -> TPR@1%FPR; also AUC
     (--normalize: NFKC + casefold + whitespace collapse applied to BOTH
      attacked and null continuation texts before tokenization; the primary
      `score:` then IS this normalized-mode score, also reported as
      `robust_score:`)
  4. quality: GPT-2 PPL ratio and unigram-F1 similarity vs originals
  5. score = clip(1 - mean over 4 schemes TPR - penalties, 0, 1);
     prints `score: X`

Protected harness: executors may only edit solution.py and ext/**.
(Additive exception, node 9 / user-approved like node 8: `--baseline
adaptive` loads ext/baselines/adaptive.py -- an ORACLE attack-model
comparator that needs the detector scorer wired in, so the loader binds an
eval context to that module only.  Every other code path is unchanged.)
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wm import data as Wdata  # noqa: E402
from wm import model, schemes  # noqa: E402

FPR = 0.01
PPL_MAX = 1.25
SIM_MIN = 0.80
ALLOWED_IMPORTS = {"numpy", "math", "json", "re", "random", "collections",
                   "itertools", "functools", "pathlib", "os", "sys", "typing",
                   "dataclasses", "heapq", "bisect", "string", "hashlib",
                   "unicodedata", "ext", "__future__"}

_word_re = re.compile(r"[A-Za-z0-9']+")


def check_numpy_only(paths: list[Path], extra: frozenset | set = frozenset()) -> None:
    """AST import check over the attack surface (`extra` widens it for one file)."""
    allowed = ALLOWED_IMPORTS | set(extra)
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m not in allowed:
                    raise SystemExit(
                        f"FORBIDDEN import '{m}' in {path.name}: attack surface "
                        f"is numpy+stdlib only ({sorted(allowed)})")


def load_solution():
    ext_dir = ROOT / "ext"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if ext_dir.exists() and str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ROOT))  # `import ext.x` needs ROOT on path
    paths = [ROOT / "solution.py"]
    if ext_dir.exists():
        # node 9: ext/baselines/adaptive.py is excluded from this sweep -- it
        # is never imported on the trunk path and is checked once, with its
        # own widened import set, by load_baseline("adaptive").  Every other
        # file is checked exactly as before.
        paths += sorted(p for p in ext_dir.rglob("*.py")
                        if not (p.parent.name == "baselines"
                                and p.name == "adaptive.py"))
    check_numpy_only(paths)
    spec = importlib.util.spec_from_file_location("solution", ROOT / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "attack_text"):
        raise SystemExit("solution.py must define attack_text(texts, rng)")
    return mod


def load_baseline(name: str):
    """Load a literature-baseline attack from ext/baselines/ (node 8).

    Same contract as load_solution's module: must expose
    attack_text(texts, rng) -> list[str].  The numpy-only AST check covers
    ext/baselines/** exactly as it covers every other ext file.

    Exception (node 9): the `adaptive` oracle may additionally import the
    detector/model scorer (`wm`) and the trunk pipeline (`solution`), since
    its threat model IS "attacker can query the detector statistic".
    """
    extra: frozenset | set = {"wm", "solution"} if name == "adaptive" else frozenset()
    path = ROOT / "ext" / "baselines" / f"{name}.py"
    check_numpy_only([path], extra)
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "attack_text"):
        raise SystemExit(f"ext/baselines/{name}.py must define attack_text")
    return mod


def unigram_f1(a: str, b: str) -> float:
    ca, cb = Counter(_word_re.findall(a.lower())), Counter(_word_re.findall(b.lower()))
    overlap = sum((ca & cb).values())
    na, nb = sum(ca.values()), sum(cb.values())
    if na == 0 or nb == 0:
        return 0.0
    p, r = overlap / nb, overlap / na
    return 2 * p * r / (p + r) if (p + r) else 0.0


def normalize_text(text: str) -> str:
    """Diagnostic surface normalization: NFKC + casefold + whitespace collapse."""
    t = unicodedata.normalize("NFKC", text)
    t = t.casefold()
    return re.sub(r"\s+", " ", t).strip()


def auc_pos_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    all_s = np.concatenate([pos, neg])
    order = all_s.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_s) + 1)
    r_pos = ranks[:len(pos)].sum()
    n, m = len(pos), len(neg)
    return float((r_pos - n * (n + 1) / 2) / (n * m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--node-id", default="trunk")
    ap.add_argument("--noop", action="store_true",
                    help="identity transform: skip the attack (baseline)")
    ap.add_argument("--baseline", choices=["char", "synonym", "selfinfo",
                                           "adaptive"],
                    default=None,
                    help="run a literature-baseline attack (ext/baselines/) "
                         "instead of solution.attack_text; composes with "
                         "--normalize; --noop takes precedence. `adaptive` "
                         "is the ORACLE/white-box ceiling row (node 9): it "
                         "may query the detector statistic, so it is an "
                         "attack-model comparator, never a competing attack")
    ap.add_argument("--normalize", action="store_true",
                    help="NFKC+casefold+whitespace-collapse attacked and null "
                         "continuation texts before tokenization; the primary "
                         "score then IS the normalized-mode score "
                         "(also reported as robust_score)")
    args = ap.parse_args()

    t0 = time.time()
    cache_file = Wdata.cache_path(args.split)
    if not cache_file.exists():
        raise SystemExit(f"missing {cache_file}; run scripts/prepare_cache.py")
    cache = np.load(cache_file, allow_pickle=True)
    prefixes = [list(map(int, p)) for p in cache["prefix_ids"]]
    n = len(prefixes) if args.limit == 0 else min(args.limit, len(prefixes))
    prefixes = prefixes[:n]
    plen = len(prefixes[0])

    tok, gpt = model.load()
    V = gpt.config.vocab_size
    uw = np.load(Wdata.RESOURCES / "unigram_w.npy").astype(np.float64)

    if args.noop:
        sol = None
    elif args.baseline:
        sol = load_baseline(args.baseline)
        if args.baseline == "adaptive":
            # node 9 (additive, oracle row only): hand the detector-side
            # context to the adaptive baseline so it can QUERY the score
            # exactly as this harness computes it (prefix ids, tokenizer,
            # the null-calibration material, and the normalization route).
            if not hasattr(sol, "bind_eval_context"):
                raise SystemExit(
                    "ext/baselines/adaptive.py must define bind_eval_context(ctx)")
            sol.bind_eval_context({
                "prefixes": prefixes,
                "plen": plen,
                "tokenizer": tok,
                "vocab_size": V,
                "unigram_w": uw,
                "normalize": args.normalize,
                "normalize_text": normalize_text,
                "null_ids": [list(map(int, s)) for s in cache["null_tokens"]][:n],
            })
    else:
        sol = load_solution()
    rng = np.random.default_rng(2026)

    orig_texts, attacked_texts = {}, {}
    per_scheme = {}
    for scheme in schemes.SCHEMES:
        seqs = [list(map(int, s)) for s in cache[f"{scheme}_tokens"]][:n]
        orig_texts[scheme] = [model.decode(s[len(prefixes[i]):])
                              for i, s in enumerate(seqs)]
        per_scheme[scheme] = seqs

    # attack once per scheme's texts (same rng stream, deterministic)
    t_attack = time.time()
    for scheme in schemes.SCHEMES:
        if args.noop:
            out = list(orig_texts[scheme])
        else:
            out = sol.attack_text(list(orig_texts[scheme]), rng)
            if len(out) != n:
                raise SystemExit(
                    f"attack_text returned {len(out)} texts, expected {n}")
        attacked_texts[scheme] = [str(t) for t in out]
    attack_secs = time.time() - t_attack

    # null sequences (normalization routes them through decode -> norm -> encode)
    null_ids = [list(map(int, s)) for s in cache["null_tokens"]][:n]
    if args.normalize:
        null_seqs = [prefixes[i] + tok.encode(
                         normalize_text(model.decode(null_ids[i][plen:])))
                     for i in range(n)]
    else:
        null_seqs = null_ids

    null_scores = {}
    for scheme in schemes.SCHEMES:
        if scheme == "synthid":
            null_scores[scheme] = model.synthid_batch_scores(null_seqs, plen)
        else:
            null_scores[scheme] = schemes.fast_score_batch(null_seqs, scheme,
                                                           V, uw)

    diag: dict = {"split": args.split, "n": n,
                  "noop": args.noop, "normalize": args.normalize,
                  "baseline": args.baseline,
                  "schemes": {}}
    tprs, sims, ppl_ratios = [], [], []
    for scheme in schemes.SCHEMES:
        att_texts, att_ids = [], []
        for i in range(n):
            txt = attacked_texts[scheme][i]
            if args.normalize:
                txt = normalize_text(txt)
            att_texts.append(txt)
            att_ids.append(prefixes[i] + tok.encode(txt))
        if scheme == "synthid":
            att_scores = model.synthid_batch_scores(att_ids, plen)
        else:
            att_scores = schemes.fast_score_batch(att_ids, scheme, V, uw)
        null_s = null_scores[scheme]
        thr = float(np.quantile(null_s, 1.0 - FPR))
        tpr = float((att_scores > thr).mean())
        auc = auc_pos_neg(att_scores, null_s)
        ppl_att = model.ppl_batch(att_ids)
        ppl_orig = np.asarray(cache[f"ppl_{scheme}"][:n], dtype=np.float64)
        if args.normalize:
            orig_norm_ids = [prefixes[i] + tok.encode(
                normalize_text(orig_texts[scheme][i])) for i in range(n)]
            ppl_orig = model.ppl_batch(orig_norm_ids)
        ppl_ratio = float(ppl_att.mean() / max(ppl_orig.mean(), 1e-9))
        sim = float(np.mean([unigram_f1(orig_texts[scheme][i], att_texts[i])
                             for i in range(n)]))
        tprs.append(tpr)
        sims.append(sim)
        ppl_ratios.append(ppl_ratio)
        diag["schemes"][scheme] = {
            "tpr": round(tpr, 4), "auc": round(auc, 4), "thr": round(thr, 3),
            "mean_z_attacked": round(float(att_scores.mean()), 3),
            "mean_z_null": round(float(null_s.mean()), 3),
            "ppl_ratio": round(ppl_ratio, 4), "sim": round(sim, 4),
        }

    raw = 1.0 - float(np.mean(tprs))
    pen_ppl = max(0.0, float(np.mean(ppl_ratios)) - PPL_MAX) * 1.0
    pen_sim = max(0.0, SIM_MIN - float(np.mean(sims))) * 2.0
    score = max(0.0, raw - pen_ppl - pen_sim)
    diag.update({
        "attack_score_raw": round(raw, 4),
        "penalty_ppl": round(pen_ppl, 4), "penalty_sim": round(pen_sim, 4),
        "score": round(score, 4),
        "mean_tpr": round(float(np.mean(tprs)), 4),
        "mean_ppl_ratio": round(float(np.mean(ppl_ratios)), 4),
        "mean_sim": round(float(np.mean(sims)), 4),
        "attack_secs": round(attack_secs, 2),
        "total_secs": round(time.time() - t0, 1),
    })
    if args.normalize:
        diag["robust_score"] = round(score, 4)
    out_path = Wdata.RESULTS / f"{args.split}_{args.node_id}.json"
    Wdata.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps(diag, indent=2))
    print(f"score: {score:.4f}")
    if args.normalize:
        print(f"robust_score: {score:.4f}")


if __name__ == "__main__":
    main()
