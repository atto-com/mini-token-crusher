"""GPT-2 small wrapper: watermarked generation and perplexity (torch allowed here)."""
from __future__ import annotations

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from . import schemes as S

MODEL_NAME = "gpt2"
CONT_LEN = 64
BATCH = 8
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_tok: GPT2TokenizerFast | None = None
_model: GPT2LMHeadModel | None = None


def load():
    global _tok, _model
    if _model is None:
        _tok = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _tok.pad_token = _tok.eos_token
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return _tok, _model


def encode(text: str) -> list[int]:
    tok, _ = load()
    return tok.encode(text)


def decode(ids) -> str:
    tok, _ = load()
    return tok.decode(list(ids), skip_special_tokens=True)


CHUNK = 16


@torch.no_grad()
def generate_batched(prefixes: list[list[int]], scheme: str | None,
                     length: int = CONT_LEN, seed: int = 0,
                     unigram_w: np.ndarray | None = None,
                     unigram_green: np.ndarray | None = None) -> list[list[int]]:
    """Generate `length` continuation tokens for each prefix.

    scheme=None -> plain sampling (null texts). Returns full sequences
    (prefix + continuation). Forward passes run in fixed-shape chunks of
    CHUNK sequences to keep the MPS allocator stable.
    """
    tok, model = load()
    rng = np.random.default_rng(seed)
    batch = [list(p) for p in prefixes]
    n = len(batch)
    if n == 0:
        return []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 50256
    total_len = max(len(b) for b in batch) + length
    for step in range(length):
        for s0 in range(0, n, CHUNK):
            part = batch[s0:s0 + CHUNK]
            ids_cpu = torch.full((len(part), total_len), pad_id, dtype=torch.long)
            attn_cpu = torch.zeros(len(part), total_len, dtype=torch.long)
            for i, b in enumerate(part):
                ids_cpu[i, :len(b)] = torch.tensor(b, dtype=torch.long)
                attn_cpu[i, :len(b)] = 1
            logits = model(input_ids=ids_cpu.to(DEVICE),
                           attention_mask=attn_cpu.to(DEVICE)).logits
            for i, b in enumerate(part):
                row = logits[i, len(b) - 1].float().cpu().numpy()
                prev = b[-1]
                if scheme == "synthid":
                    # raw logits (lifted to float64) so the top-k tournament at
                    # detection time sees exactly the same ordering
                    biased = S.apply_bias(row.astype(np.float64), prev, step,
                                          scheme, unigram_green, unigram_w,
                                          context=b)
                    b.append(int(S.sample_from(biased, rng)))
                    continue
                row = row - row.max()
                p = np.exp(row)
                p /= p.sum()
                if scheme is not None:
                    p = S.apply_bias(np.log(p + 1e-12), prev, step, scheme,
                                     unigram_green, unigram_w)
                    nxt = S.sample_from(p, rng)
                else:
                    nxt = int(rng.choice(p.shape[0], p=p))
                b.append(int(nxt))
        if (step + 1) % 8 == 0:
            print(f"    gen step {step + 1}/{length}", flush=True)
    return batch


@torch.no_grad()
def synthid_batch_scores(seqs: list[list[int]], prefix_len: int,
                         batch: int = CHUNK) -> np.ndarray:
    """SynthID tournament z-scores via teacher-forced forwards (torch allowed).

    Sequences are fed in fixed-shape chunks padded to a single global width,
    mirroring generate_batched, so for unmodified sequences the per-position
    logits (and hence top-k tournaments) reproduce the generation-time logits.
    Positions scored are the continuation ones (row >= prefix_len - 1).
    """
    tok, model = load()
    out = np.zeros(len(seqs), dtype=np.float64)
    if not seqs:
        return out
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 50256
    total_len = max(len(s) for s in seqs)
    for s0 in range(0, len(seqs), batch):
        part = seqs[s0:s0 + batch]
        ids_cpu = torch.full((len(part), total_len), pad_id, dtype=torch.long)
        attn_cpu = torch.zeros(len(part), total_len, dtype=torch.long)
        for i, b in enumerate(part):
            ids_cpu[i, :len(b)] = torch.tensor(b, dtype=torch.long)
            attn_cpu[i, :len(b)] = 1
        logits = model(input_ids=ids_cpu.to(DEVICE),
                       attention_mask=attn_cpu.to(DEVICE)).logits
        lg = logits.float().cpu().numpy()
        for i, b in enumerate(part):
            rows = lg[i, :len(b) - 1]
            z, _rate = S.synthid_score(rows, np.asarray(b, dtype=np.int64),
                                       prefix_len - 1)
            out[s0 + i] = z
        del logits
    return out


@torch.no_grad()
def ppl_batch(seqs: list[list[int]], max_tokens: int = 256) -> np.ndarray:
    """Perplexity of each sequence under plain GPT-2 (truncated to max_tokens)."""
    tok, model = load()
    out = np.zeros(len(seqs), dtype=np.float64)
    for i, s in enumerate(seqs):
        ids = torch.tensor([s[:max_tokens]], dtype=torch.long, device=DEVICE)
        logits = model(input_ids=ids).logits[0, :-1].float()
        target = ids[0, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, target.unsqueeze(1)).mean().item()
        out[i] = float(np.exp(nll))
    return out
