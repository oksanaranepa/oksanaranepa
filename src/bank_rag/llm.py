from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from bank_rag.embeddings import tokenize
from bank_rag.retrieval import RetrievedContext


SYSTEM_PROMPT = """Вы банковский консультант. Отвечайте кратко, вежливо и только на основе
переданного контекста из базы знаний банка. Если в контексте нет ответа, скажите, что
нужно уточнить информацию у специалиста банка. Не выдумывайте тарифы, ставки, сроки и
требования. В конце ответа укажите источники в формате [SOURCE / SECTION]."""


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 500


class BankLLM(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        raise NotImplementedError


class RuleBasedBankLLM(BankLLM):
    """Deterministic offline model that extracts answer sentences from retrieved context."""

    def generate(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        if not contexts:
            return (
                "Я не нашел релевантную информацию в базе знаний банка. "
                "Пожалуйста, уточните вопрос или обратитесь к специалисту банка."
            )

        question_tokens = set(tokenize(question))
        selected_sentences: list[str] = []
        citations: list[str] = []
        for context in contexts:
            sentences = _split_sentences(context.text)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(question_tokens & set(tokenize(sentence))),
                reverse=True,
            )
            for sentence in ranked[:2]:
                if sentence and sentence not in selected_sentences:
                    selected_sentences.append(sentence)
            citation = f"[{context.citation}]"
            if citation not in citations:
                citations.append(citation)
            if len(selected_sentences) >= 4:
                break

        answer = " ".join(selected_sentences[:4]).strip()
        if not answer:
            answer = "В найденных документах есть связанная информация, но точный ответ требует уточнения у специалиста банка."
        return f"{answer}\n\nИсточники: {' '.join(citations[:3])}"


class OpenAIChatLLM(BankLLM):
    def __init__(self, model: str = "gpt-4o-mini", config: GenerationConfig | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.config = config or GenerationConfig()

    def generate(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        messages = _build_chat_messages(question, contexts, history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""


class OllamaLLM(BankLLM):
    def __init__(
        self,
        model: str = "llama3.1",
        endpoint: str = "http://localhost:11434/api/chat",
        config: GenerationConfig | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.config = config or GenerationConfig()

    def generate(
        self,
        question: str,
        contexts: list[RetrievedContext],
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": _build_chat_messages(question, contexts, history),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = requests.post(self.endpoint, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]


def create_llm(backend: str | None = None, config: GenerationConfig | None = None) -> BankLLM:
    selected = backend or os.getenv("BANK_RAG_LLM_BACKEND", "rule-based")
    if selected == "rule-based":
        return RuleBasedBankLLM()
    if selected == "openai":
        return OpenAIChatLLM(config=config)
    if selected == "ollama":
        return OllamaLLM(config=config)
    raise ValueError("Unknown LLM backend. Use 'rule-based', 'openai', or 'ollama'.")


def validate_citations(answer: str, contexts: Iterable[RetrievedContext]) -> bool:
    allowed = {f"{context.citation}" for context in contexts}
    citations = re.findall(r"\[([^\]]+)\]", answer)
    return all(citation in allowed for citation in citations)


def _build_chat_messages(
    question: str,
    contexts: list[RetrievedContext],
    history: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    context_text = "\n\n".join(
        f"[{context.citation}]\n{context.text}"
        for context in contexts
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_message, assistant_message in history or []:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append(
        {
            "role": "user",
            "content": f"Контекст:\n{context_text}\n\nВопрос клиента: {question}",
        }
    )
    return messages


def _split_sentences(text: str) -> list[str]:
    normalized = text.replace("\n", " ")
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]
