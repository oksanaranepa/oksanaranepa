from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bank_rag.chunking import ChunkStrategy
from bank_rag.indexing import BankRAGIndex, build_local_index
from bank_rag.llm import BankLLM, GenerationConfig, create_llm, validate_citations
from bank_rag.retrieval import RetrievedContext, SearchMode


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    latency_ms: float
    metadata: dict[str, Any]


class RAGPipeline:
    def __init__(
        self,
        index: BankRAGIndex,
        llm: BankLLM,
        cache_enabled: bool = True,
        min_relevance_score: float = 0.12,
    ):
        self.index = index
        self.llm = llm
        self.cache_enabled = cache_enabled
        self.min_relevance_score = min_relevance_score
        self._answer_cache: dict[tuple[str, str | None, str], RAGResponse] = {}

    @classmethod
    def from_defaults(
        cls,
        strategy: ChunkStrategy = "recursive",
        embedding_backend: str = "hash",
        llm_backend: str = "rule-based",
        cache_enabled: bool = True,
        generation_config: GenerationConfig | None = None,
    ) -> "RAGPipeline":
        index = build_local_index(strategy=strategy, embedding_backend=embedding_backend)
        llm = create_llm(backend=llm_backend, config=generation_config)
        return cls(index=index, llm=llm, cache_enabled=cache_enabled)

    def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        product_type: str | None = None,
        k: int = 5,
        mode: SearchMode = "hybrid",
    ) -> RAGResponse:
        cache_key = (question, product_type, mode)
        if self.cache_enabled and cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        started = time.perf_counter()
        filters = {"product_type": product_type} if product_type else None
        contexts = self.index.retriever.retrieve(
            query=_with_history(question, history),
            k=k,
            mode=mode,
            filters=filters,
            use_self_query=product_type is None,
            use_multi_query=True,
            compress=True,
        )
        if not contexts or contexts[0].score < self.min_relevance_score:
            answer = (
                "Я не нашел релевантную информацию в актуальной базе знаний банка. "
                "Пожалуйста, уточните продукт или обратитесь к специалисту банка."
            )
        else:
            answer = self.llm.generate(question=question, contexts=contexts, history=history)
            if not validate_citations(answer, contexts):
                citations = " ".join(f"[{context.citation}]" for context in contexts[:3])
                answer = f"{answer}\n\nПроверенные источники: {citations}"

        latency_ms = (time.perf_counter() - started) * 1000
        response = RAGResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            latency_ms=round(latency_ms, 2),
            metadata={
                "mode": mode,
                "product_type_filter": product_type,
                "cache_enabled": self.cache_enabled,
                "citation_validation": validate_citations(answer, contexts) if contexts else True,
            },
        )
        if self.cache_enabled:
            self._answer_cache[cache_key] = response
        return response


def _with_history(question: str, history: list[tuple[str, str]] | None) -> str:
    if not history:
        return question
    recent = " ".join(f"Клиент: {user} Консультант: {assistant}" for user, assistant in history[-2:])
    return f"{recent} Новый вопрос: {question}"
