"""Normalization-invariant attack combo (node 5.1).

Two objectives, one pipeline: maximize the `--normalize` dev score (NFKC +
casefold + whitespace collapse applied to attacked AND null texts before
tokenization) while keeping raw-mode TPR crushed and both quality gates
(sim >= 0.80 aim 0.85, ppl_ratio <= 1.25) green in both modes.

Measured on B_dev (4-scheme metric): raw 0.9531, robust 0.9281 -- versus
trunk 0.9500 raw / 0.3375 robust (surface levers undone by --normalize)
and noop 0.2625 robust.

Stages (all key-blind, numpy-rng only; no watermark key, no hash use):

  0. ext/surface (node 1.1): UPPERCASE + NBSP / double-space surface levers.
     Cheap, huge raw-mode kill -- runs FIRST so its in-run NBSP insertion
     never lands next to commas / hyphens the later stages plant (an NBSP
     there would restore the leading space after normalization and un-reroll
     the word). Under --normalize these levers are all undone: harmless.
  1a. ext/punct: punctuation/symbol identity rework (RATE=1.0). After stage 1
     nearly every *word* id is rerolled by tight commas, yet ~60% of the
     ORIGINAL punctuation ids survived and carried the green excess
     (unigram z ~2.4 at 22% token-id preservation). Substituting every
     non-word char for NFKC-stable alternates cuts it to z ~0.5 at ZERO
     sim cost (the word stream never counts punctuation).
  1. ext/funcword (node 3.1): tight boundary punctuation (INS=1.0) rerolls
     every following word's id (leading space swallowed) and breaks its
     prev-link -- kills kgw/exp/synthid under --normalize as well as raw.
     Function-word DELETION is stage-gated OFF (DEL=0): measured net-negative
     (each deletion costs ~0.06 unigram-F1 sim while *raising* unigram z;
     punctuation re-roll is what does the killing).
  2. ext/splinter: word-internal hyphen splinters on 12% of content words
     ("become" -> "be-come"). Hyphens survive NFKC/casefold/whitespace
     collapse, so ids stay changed post-normalization; split points are
     verified against ext/gpt2_bpe (pure-python BPE) to actually change the
     tokenization. The word-initial rendering ("-word") was tested and
     rejected: it re-rolled no word id once commas were present and pushed
     unigram z UP by +1.1.
  3. ext/synswap (node 2.1 mechanism): sparse (15%) frequency-matched
     synonym swaps, picking the frequency-tied candidate with the largest
     token-id churn per word.
  4. ext/crush (node 3 mechanism, runs last on the near-final text):
     bias-inversion frequency crusher. The unigram scheme samples its green
     list in proportion to a RARITY weight (wm/schemes.py: green boost =
     DELTA*w, w = median_freq/p clipped to 5) and scores with that same
     weight as leverage, so the residual z after stages 0-3 is carried by the
     rare ids that SURVIVE the attack (99.5% green, +2.5 each: inherited
     word subtokens + inherited symbol ids). Crush shifts every surviving
     token ANTI-correlated with that green distribution, cheapest lever
     first: re-substitute inherited symbol ids (free -- symbols never enter
     unigram-F1), flip each word's leading space so its first subtoken id is
     fresh rather than inherited (free -- F1 ignores spaces), then demote the
     words whose inherited INTERIOR ids cannot be rerolled any other way:
     lowest-unigram-weight synonym where the table allows (common-ward
     demotion, i.e. the direction the scheme samples green in, reversed),
     else deletion -- on a fixed global budget so mean F1 sim stays >= 0.81.

Stage order matters: surface before punctuation so NBSP cannot undo tight
rendering after --normalize; punct before funcword so freshly inserted
commas are not substituted (their fixed id hashes red-ish here, which
helps); splinter before synswap so stage 3 skips splinter-adjacent words
(already rerolled) and spends its sim budget on intact words; crush last so
it sees the final tokenization (its symbol fix also covers commas inserted
downstream of ext/punct).
"""
from __future__ import annotations

import numpy as np

from ext import crush, fused

# Which stages run (tuning knob; all five are active in the final combo).
STAGES = ("surface", "punct", "funcword", "splinter", "synswap")


def attack_text(texts: list[str], rng: np.random.Generator) -> list[str]:
    """Five stages fused (node 2) + frequency crusher (node 3), integrated.

    ext/fused.py applies the same surface/punct/funcword/splinter/synswap
    decisions per position over ONE shared tokenization scan per text (with
    batched rng draws and import-time synonym precomputation), consuming the
    rng stream in the same order as the sequential stage loop -- so for a
    given seed the five-stage output is byte-identical to running the stage
    functions one after another, only much faster.  `STAGES` stays live as
    the stage-selection knob (subset / reorder supported).

    ext/crush then runs on that identical near-final text; it never draws
    from `rng` (deterministic given texts + originals), so fusing the five
    stages leaves BOTH the rng stream position and crush's input untouched
    -> the integrated pipeline is byte-identical to the sequential trunk
    while inheriting the fused engine's ~4x wall-clock win (trunk's crush
    pass was the whole of the residual 1.19s attack time).
    """
    orig = list(texts)          # pristine watermarked text (ext/crush needs it)
    out = fused.attack(texts, rng, stages=STAGES)   # stages 0-3 (fused replay)
    return crush.crush(out, orig, rng)              # stage 4: frequency crusher
