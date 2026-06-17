"""Example NeMo Guardrails actions backed by the Python guardrail modules."""

from __future__ import annotations

import os

from guardrails_chatbot.fact_validator import InMemoryVectorFactValidator
from guardrails_chatbot.pii import PIIDetector
from guardrails_chatbot.toxicity import build_toxicity_detector

try:  # pragma: no cover - NeMo is optional for the base test environment.
    from nemoguardrails.actions import action
except ImportError:  # pragma: no cover

    def action(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


_pii_detector = PIIDetector()
_toxicity_detector = build_toxicity_detector()
_fact_validator = InMemoryVectorFactValidator.from_file(
    os.getenv("GUARDRAILS_KNOWLEDGE_BASE")
)


@action(name="detect_pii")
async def detect_pii(text: str) -> bool:
    return _pii_detector.has_pii(text)


@action(name="detect_toxicity")
async def detect_toxicity(text: str) -> bool:
    return _toxicity_detector.score(text).is_toxic


@action(name="validate_facts")
async def validate_facts(text: str) -> bool:
    return _fact_validator.validate(text).is_supported
