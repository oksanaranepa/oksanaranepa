from guardrails_chatbot.fact_validator import InMemoryVectorFactValidator


def test_validator_returns_supporting_evidence() -> None:
    validator = InMemoryVectorFactValidator(
        documents=[
            "Guardrails проверяют пользовательский ввод до генерации.",
            "Локальная модель генерирует ответ после pre-checks.",
        ]
    )

    result = validator.validate("Guardrails проверяют ввод пользователя.")

    assert result.is_supported
    assert result.evidence
    assert "Guardrails" in result.evidence[0].text


def test_validator_skips_when_no_knowledge_base_configured() -> None:
    validator = InMemoryVectorFactValidator()

    result = validator.validate("Любой ответ")

    assert result.is_supported
    assert result.skipped
