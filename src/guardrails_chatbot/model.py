"""Local language model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    """Minimal chat model protocol."""

    def generate(self, prompt: str) -> str:
        """Generate a response for prompt."""


@dataclass
class StubChatModel:
    """Deterministic model used for demos, tests, and offline smoke checks."""

    system_hint: str = (
        "Я локальный демо-ассистент с guardrails. "
        "Подключите --model-id, чтобы использовать реальную модель."
    )

    def generate(self, prompt: str) -> str:
        return (
            f"{self.system_hint} Получен безопасный запрос: "
            f"{prompt.strip()[:240]}"
        )


class TransformersChatModel:
    """Text-generation adapter for local Hugging Face models."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        device: int | str | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "transformers is not installed. Install with `pip install -e '.[local-ml]'`."
            ) from exc

        kwargs: dict[str, object] = {"model": model_id}
        if device is not None:
            kwargs["device"] = device

        self._generator = pipeline("text-generation", **kwargs)
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def generate(self, prompt: str) -> str:
        outputs = self._generator(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=self._temperature > 0,
            temperature=self._temperature,
            return_full_text=False,
        )
        if not outputs:
            return ""
        return str(outputs[0].get("generated_text", "")).strip()


def build_chat_model(model_id: str | None) -> ChatModel:
    """Return a local chat model for the requested backend."""

    if model_id:
        return TransformersChatModel(model_id=model_id)
    return StubChatModel()
