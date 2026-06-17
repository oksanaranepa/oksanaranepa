"""Lightweight vector-style factuality validation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


@dataclass(frozen=True)
class Evidence:
    """Retrieved knowledge-base fragment."""

    text: str
    similarity: float


@dataclass(frozen=True)
class FactValidationResult:
    """Post-generation factuality verdict."""

    is_supported: bool
    score: float
    evidence: list[Evidence]
    skipped: bool = False


class InMemoryVectorFactValidator:
    """Small cosine-similarity validator over a local knowledge base."""

    def __init__(
        self,
        documents: list[str] | None = None,
        threshold: float = 0.12,
        top_k: int = 3,
    ) -> None:
        self._documents = [doc.strip() for doc in documents or [] if doc.strip()]
        self._vectors = [self._vectorize(doc) for doc in self._documents]
        self.threshold = threshold
        self.top_k = top_k

    @classmethod
    def from_file(
        cls,
        path: str | Path | None,
        threshold: float = 0.12,
        top_k: int = 3,
    ) -> "InMemoryVectorFactValidator":
        if path is None:
            return cls(threshold=threshold, top_k=top_k)

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {source}")

        text = source.read_text(encoding="utf-8")
        documents = [chunk.strip() for chunk in re.split(r"\n\s*\n", text)]
        if len(documents) == 1:
            documents = [line.strip() for line in text.splitlines()]
        return cls(documents=documents, threshold=threshold, top_k=top_k)

    def validate(self, answer: str) -> FactValidationResult:
        """Return whether answer is supported by at least one KB fragment."""

        if not self._documents:
            return FactValidationResult(
                is_supported=True,
                score=1.0,
                evidence=[],
                skipped=True,
            )

        answer_vector = self._vectorize(answer)
        ranked = sorted(
            (
                Evidence(text=document, similarity=self._cosine(answer_vector, vector))
                for document, vector in zip(self._documents, self._vectors, strict=True)
            ),
            key=lambda item: item.similarity,
            reverse=True,
        )
        evidence = ranked[: self.top_k]
        score = evidence[0].similarity if evidence else 0.0
        return FactValidationResult(
            is_supported=score >= self.threshold,
            score=score,
            evidence=evidence,
        )

    @staticmethod
    def _vectorize(text: str) -> Counter[str]:
        return Counter(token.lower() for token in _TOKEN_PATTERN.findall(text))

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        common = set(left) & set(right)
        dot_product = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)
