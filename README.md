# mini-token-crusher

my token crusher:
Pipeline (5 stages, each breaks its own type of signal):

surface — case-flips, NBSP/double-space: shifts all (prev, id) pairs down the chain

punct — punctuation alternation: finishes off the context-hash in kgw/exp

funcword — function-word substitutions (sim-preserving, F1 treats them as "noise")

splinter — hyphen-splits, verified to change BPE-ids

synswap + crush — tied synonyms + bias-inversion: rare inherited green ids → common substitutes (finishes off unigram/synthid)
