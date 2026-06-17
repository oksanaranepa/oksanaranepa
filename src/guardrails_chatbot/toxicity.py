"""Toxicity detection adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToxicityResult:
    """Normalized toxicity detector output."""

    is_toxic: bool
    score: float
    labels: dict[str, float]
    source: str


class ToxicityDetector(Protocol):
    """Detector protocol used by the guardrail engine."""

    def score(self, text: str) -> ToxicityResult:
        """Return toxicity decision for text."""


class HeuristicToxicityDetector:
    """Small fallback detector for offline demos and tests."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._terms = {
            "hate": 0.7,
            "kill yourself": 0.95,
            "idiot": 0.6,
            "stupid": 0.55,
            "дурак": 0.6,
            "ненавижу": 0.65,
            "убей себя": 0.95,
            "тупой": 0.6,
        }

    def score(self, text: str) -> ToxicityResult:
        lowered = text.lower()
        labels = {
            term: weight for term, weight in self._terms.items() if term in lowered
        }
        score = max(labels.values(), default=0.0)
        return ToxicityResult(
            is_toxic=score >= self.threshold,
            score=score,
            labels=labels,
            source="heuristic",
        )


class DetoxifyToxicityDetector:
    """Adapter around Detoxify's multilingual toxicity model."""

    def __init__(self, threshold: float = 0.7, model_name: str = "multilingual") -> None:
        try:
            from detoxify import Detoxify
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "Detoxify is not installed. Install with `pip install -e '.[local-ml]'`."
            ) from exc

        self.threshold = threshold
        self._model = Detoxify(model_name)

    def score(self, text: str) -> ToxicityResult:
        raw_scores = self._model.predict(text)
        labels = {label: float(value) for label, value in raw_scores.items()}
        score = max(labels.values(), default=0.0)
        return ToxicityResult(
            is_toxic=score >= self.threshold,
            score=score,
            labels=labels,
            source="detoxify",
        )


def build_toxicity_detector(
    prefer_detoxify: bool = True,
    threshold: float = 0.7,
) -> ToxicityDetector:
    """Build the strongest available toxicity detector."""

    if prefer_detoxify:
        try:
            return DetoxifyToxicityDetector(threshold=threshold)
        except RuntimeError:
            pass
    return HeuristicToxicityDetector(threshold=min(threshold, 0.5))
