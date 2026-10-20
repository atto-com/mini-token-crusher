"""Literature-grade baseline attacks (node 8, harness comparator).

Three standalone comparators, each exposing `attack_text(texts, rng)` with
the same contract as solution.attack_text, so `eval.py --baseline <name>`
swaps them in with no other harness change:

- char:      character-level perturbation (arXiv 2509.09112) -- surface
             case/whitespace levers + punctuation identity substitution in
             isolation (ext/surface + ext/punct only).
- synonym:   frequency-matched synonym substitution a la WaterPark
             (arXiv 2411.13425) edit suite -- ext/synswap logic standalone
             at its max unigram-F1-legal rate.
- selfinfo:  self-information-weighted rewrite (arXiv 2505.05190) --
             synonym-rewrite only the TOP-fraction most informative words
             (self-information proxy = -log GPT-2 unigram weight).
- adaptive:  ORACLE / adaptive white-box ceiling (node 9, arXiv 2410.02440
             attack model).  Unlike the other three it MAY query the
             detector statistic (wm.schemes / wm.model score interface), so
             eval wires an extra context into it (bind_eval_context); it is
             an attack-model comparator row, never a competing attack.

numpy + stdlib + ext imports only (eval's AST check covers this package;
the adaptive oracle is additionally allowed to import wm + solution).
"""
