"""Guardrail orchestration for the chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field

from guardrails_chatbot.fact_validator import (
    FactValidationResult,
    InMemoryVectorFactValidator,
)
from guardrails_chatbot.model import ChatModel
from guardrails_chatbot.pii import PIIDetector, PIIMatch
from guardrails_chatbot.toxicity import ToxicityDetector, ToxicityResult


@dataclass(frozen=True)
class GuardrailDecision:
    """Final result returned by the guardrail engine."""

    allowed: bool
    stage: str
    response: str
    reasons: list[str] = field(default_factory=list)
    masked_prompt: str | None = None
    raw_response: str | None = None
    pii_matches: list[PIIMatch] = field(default_factory=list)
    toxicity: ToxicityResult | None = None
    fact_validation: FactValidationResult | None = None


class GuardrailEngine:
    """Safety and factuality layer around a local chat model."""

    def __init__(
        self,
        model: ChatModel,
        pii_detector: PIIDetector,
        toxicity_detector: ToxicityDetector,
        fact_validator: InMemoryVectorFactValidator | None = None,
    ) -> None:
        self._model = model
        self._pii_detector = pii_detector
        self._toxicity_detector = toxicity_detector
        self._fact_validator = fact_validator or InMemoryVectorFactValidator()

    def chat(self, prompt: str) -> GuardrailDecision:
        """Run pre-generation, generation, and post-generation checks."""

        input_pii = self._pii_detector.find(prompt)
        masked_prompt = self._pii_detector.mask(prompt, input_pii)
        if input_pii:
            kinds = sorted({match.kind for match in input_pii})
            return GuardrailDecision(
                allowed=False,
                stage="input",
                response=(
                    "Я не могу обработать запрос с персональными данными. "
                    "Удалите или замаскируйте: " + ", ".join(kinds) + "."
                ),
                reasons=["pii_detected"],
                masked_prompt=masked_prompt,
                pii_matches=input_pii,
            )

        input_toxicity = self._toxicity_detector.score(prompt)
        if input_toxicity.is_toxic:
            return GuardrailDecision(
                allowed=False,
                stage="input",
                response=(
                    "Я не могу продолжить с токсичным или оскорбительным запросом. "
                    "Переформулируйте его нейтрально."
                ),
                reasons=["toxicity_detected"],
                masked_prompt=masked_prompt,
                toxicity=input_toxicity,
            )

        raw_response = self._model.generate(prompt)

        output_pii = self._pii_detector.find(raw_response)
        if output_pii:
            kinds = sorted({match.kind for match in output_pii})
            return GuardrailDecision(
                allowed=False,
                stage="output",
                response=(
                    "Ответ заблокирован, потому что модель попыталась вывести "
                    "персональные данные: " + ", ".join(kinds) + "."
                ),
                reasons=["pii_leak_detected"],
                masked_prompt=masked_prompt,
                raw_response=self._pii_detector.mask(raw_response, output_pii),
                pii_matches=output_pii,
            )

        output_toxicity = self._toxicity_detector.score(raw_response)
        if output_toxicity.is_toxic:
            return GuardrailDecision(
                allowed=False,
                stage="output",
                response="Ответ заблокирован из-за токсичного содержания.",
                reasons=["toxic_output_detected"],
                masked_prompt=masked_prompt,
                raw_response=raw_response,
                toxicity=output_toxicity,
            )

        fact_validation = self._fact_validator.validate(raw_response)
        if not fact_validation.is_supported:
            return GuardrailDecision(
                allowed=False,
                stage="output",
                response=(
                    "Я не могу подтвердить сгенерированный ответ по локальной "
                    "базе знаний, поэтому не буду выдавать его как факт."
                ),
                reasons=["unsupported_by_knowledge_base"],
                masked_prompt=masked_prompt,
                raw_response=raw_response,
                fact_validation=fact_validation,
            )

        return GuardrailDecision(
            allowed=True,
            stage="complete",
            response=raw_response,
            masked_prompt=masked_prompt,
            raw_response=raw_response,
            toxicity=output_toxicity,
            fact_validation=fact_validation,
        )
