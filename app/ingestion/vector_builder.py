"""Baseline vector store: embeds context_text only (no provenance, no grouping)."""
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.config import get_settings
from app.utils.text import content_tokens

DIM = 1024


def _local_embed(texts: List[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = content_tokens(t)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            out[i, int(hashlib.md5(g.encode()).hexdigest(), 16) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def embed_texts(texts: List[str]) -> np.ndarray:
    s = get_settings()
    if s.embedding_backend == "openai" and s.llm_enabled:
        from app.utils.llm import embed_openai
        m = np.array(embed_openai(texts), dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    return _local_embed(texts)


class VectorStore:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        self.matrix = embed_texts(chunks)

    def search(self, question: str, k: int = 3) -> List[Tuple[str, float]]:
        sims = self.matrix @ embed_texts([question])[0]
        idx = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


def build_from_dataset(path: Path) -> VectorStore:
    facts = json.loads(Path(path).read_text(encoding="utf-8"))
    return VectorStore([f["context_text"] for f in facts])
