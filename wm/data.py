"""Prompt splits: data/prompts_{dev,test}.jsonl with token-id prefixes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
RESOURCES = ROOT / "resources"

PROMPT_TOKENS = 16  # prefix length fed to the model


def load_prompts(split: str) -> list[dict]:
    path = DATA / f"prompts_{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cache_path(split: str) -> Path:
    return CACHE / f"{split}.npz"
