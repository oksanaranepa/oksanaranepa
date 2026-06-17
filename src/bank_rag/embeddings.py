from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
EMBEDDING_RATIONALE = (
    "intfloat/multilingual-e5-large выбран для production-сценария, потому что модель "
    "обучена для retrieval-задач, поддерживает русский и английский языки, хорошо работает "
    "с форматом query/passsage и устойчива к банковской терминологии в многоязычной среде."
)

_TOKEN_RE = re.compile(r"[a-zа-яё0-9%]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower().replace("ё", "е") for token in _TOKEN_RE.findall(text)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass
class HashingEmbeddingModel(EmbeddingModel):
    """Small deterministic embedding model for offline tests and examples."""

    dimension: int = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()


def create_embedding_model(
    backend: str | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingModel:
    selected_backend = backend or os.getenv("BANK_RAG_EMBEDDING_BACKEND", "hash")
    if selected_backend == "sentence-transformers":
        return SentenceTransformerEmbeddingModel(model_name=model_name)
    if selected_backend == "hash":
        return HashingEmbeddingModel()
    raise ValueError(
        "Unknown embedding backend. Use 'hash' for offline runs or "
        "'sentence-transformers' for the multilingual E5 model."
    )
