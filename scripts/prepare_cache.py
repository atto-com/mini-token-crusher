"""Generate watermarked + null generation caches (run once; slow).

Outputs cache/{dev,test}.npz with:
  prefix_ids:   list[int] per prompt
  null_tokens:  list[list[int]] plain continuations (full seq = prefix+cont)
  {scheme}_tokens: watermarked full sequences
  ppl_null, ppl_{scheme}: cached original perplexities

Runs are merge-safe: an existing cache is loaded, only missing entries are
generated (use --force to overwrite), so keys already on disk -- e.g. the
original three schemes -- are never regenerated with different seeds.
`--schemes synthid` appends the SynthID-Text-style scheme to the existing
cache in place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wm import data as Wdata  # noqa: E402
from wm import model, schemes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--schemes", default=",".join(schemes.SCHEMES))
    ap.add_argument("--length", type=int, default=model.CONT_LEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="regenerate entries even if present in the cache")
    args = ap.parse_args()

    model.load()
    uw = np.load(Wdata.RESOURCES / "unigram_w.npy")
    ug = schemes.vocab_green_unigram(model.load()[1].config.vocab_size)
    V = model.load()[1].config.vocab_size

    for split in args.splits.split(","):
        prompts = Wdata.load_prompts(split)
        prefixes = [p["prompt_ids"] for p in prompts]
        path = Wdata.cache_path(split)
        store: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path, allow_pickle=True) as z:
                store = {k: z[k] for k in z.files}
            print(f"[{split}] loading existing cache ({len(store)} keys) "
                  f"from {path}")
        store["prefix_ids"] = np.array(prefixes, dtype=object)
        store["ids"] = np.array([p["id"] for p in prompts], dtype=object)

        if "null_tokens" not in store or args.force:
            print(f"[{split}] null generation ({len(prefixes)} prompts)...")
            null = model.generate_batched(prefixes, scheme=None,
                                          length=args.length, seed=args.seed)
            store["null_tokens"] = np.array(null, dtype=object)
            store["ppl_null"] = model.ppl_batch(null)
        else:
            print(f"[{split}] null_tokens present, skipping")

        for si, scheme in enumerate(args.schemes.split(",")):
            if f"{scheme}_tokens" in store and not args.force:
                print(f"[{split}] {scheme}_tokens present, skipping")
                continue
            print(f"[{split}] {scheme} generation...")
            wm_seqs = model.generate_batched(
                prefixes, scheme=scheme, length=args.length,
                seed=args.seed + 1000 + si, unigram_w=uw, unigram_green=ug)
            store[f"{scheme}_tokens"] = np.array(wm_seqs, dtype=object)
            store[f"ppl_{scheme}"] = model.ppl_batch(wm_seqs)
            if scheme == "synthid":
                z = model.synthid_batch_scores(wm_seqs, len(prefixes[0]))
            else:
                z = schemes.fast_score_batch(wm_seqs, scheme, V, uw)
            print(f"[{split}] {scheme} mean z (watermarked): {z.mean():.2f}")
        Wdata.CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **store)
        print(f"[{split}] saved {path} ({len(store)} keys)")


if __name__ == "__main__":
    main()
