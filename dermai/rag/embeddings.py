"""Embedding backend for the FAISS index.

Prefers local HuggingFace sentence-transformers embeddings (no API key, matches
the published pipeline). If sentence-transformers / torch is unavailable in the
environment, it degrades gracefully to a deterministic hashing embedding so the
FAISS retrieval path still works for a demo. In both cases the vectors plug into
the same LangChain ``Embeddings`` interface.
"""

from __future__ import annotations

import hashlib
import math
import os

# sentence-transformers runs on PyTorch; tell transformers to skip its TensorFlow
# integration, which otherwise errors under Keras 3. Must be set before import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from langchain_core.embeddings import Embeddings

from dermai.config import EMBEDDING_MODEL


def get_embeddings():
    """Return a LangChain ``Embeddings`` instance (HF if available, else hashing)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return _HashingEmbeddings()


class _HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free fallback embedding (bag-of-hashed-tokens).

    Not as semantically strong as sentence-transformers, but it keeps FAISS
    retrieval functional offline without torch. Implements the LangChain
    ``Embeddings`` interface so it plugs into FAISS like any other backend.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
