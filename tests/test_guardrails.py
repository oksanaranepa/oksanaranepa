from guardrails_chatbot.fact_validator import InMemoryVectorFactValidator
from guardrails_chatbot.guardrails import GuardrailEngine
from guardrails_chatbot.pii import PIIDetector
from guardrails_chatbot.toxicity import HeuristicToxicityDetector


class RecordingModel:
    def __init__(self, response: str = "Безопасный ответ") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


def build_engine(
    model: RecordingModel,
    facts: list[str] | None = None,
) -> GuardrailEngine:
    return GuardrailEngine(
        model=model,
        pii_detector=PIIDetector(),
        toxicity_detector=HeuristicToxicityDetector(),
        fact_validator=InMemoryVectorFactValidator(documents=facts),
    )


def test_blocks_pii_before_generation() -> None:
    model = RecordingModel()
    engine = build_engine(model)

    decision = engine.chat("Мой email user@example.com, помоги.")

    assert not decision.allowed
    assert decision.stage == "input"
    assert decision.reasons == ["pii_detected"]
    assert model.calls == []
    assert decision.masked_prompt == "Мой email [EMAIL], помоги."


def test_blocks_toxic_prompt_before_generation() -> None:
    model = RecordingModel()
    engine = build_engine(model)

    decision = engine.chat("Ты дурак")

    assert not decision.allowed
    assert decision.stage == "input"
    assert decision.reasons == ["toxicity_detected"]
    assert model.calls == []


def test_blocks_pii_in_model_output() -> None:
    model = RecordingModel("Пишите на admin@example.com")
    engine = build_engine(model)

    decision = engine.chat("Расскажи о правилах")

    assert not decision.allowed
    assert decision.stage == "output"
    assert decision.reasons == ["pii_leak_detected"]
    assert decision.raw_response == "Пишите на [EMAIL]"


def test_blocks_unsupported_facts_when_knowledge_base_is_present() -> None:
    model = RecordingModel("Python использует snake_case для имен функций.")
    engine = build_engine(model, facts=["Столица Франции - Париж."])

    decision = engine.chat("Какая столица Франции?")

    assert not decision.allowed
    assert decision.reasons == ["unsupported_by_knowledge_base"]


def test_allows_supported_safe_response() -> None:
    model = RecordingModel("Guardrails проверяют PII и токсичность.")
    engine = build_engine(model, facts=["Guardrails проверяют PII и токсичность."])

    decision = engine.chat("Какие проверки выполняются?")

    assert decision.allowed
    assert decision.stage == "complete"
    assert decision.response == "Guardrails проверяют PII и токсичность."
