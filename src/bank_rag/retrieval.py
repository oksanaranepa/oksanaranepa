from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

from bank_rag.chunking import Chunk
from bank_rag.embeddings import EmbeddingModel, tokenize
from bank_rag.vector_store import LocalVectorStore, SearchResult

SearchMode = Literal["similarity", "mmr", "hybrid"]


PRODUCT_KEYWORDS = {
    "loan": {"кредит", "кредита", "кредиту", "потребительский", "удобный", "платеж", "погасить"},
    "mortgage": {"ипотека", "ипотеки", "ипотеку", "семейная", "недвижимость", "застройщик"},
    "deposit": {"вклад", "вклада", "вкладу", "депозит", "проценты", "капитализация", "пополнять"},
    "borrower": {"заемщик", "заемщика", "долговая", "доход", "гражданство", "отказ"},
    "faq": {"заявка", "статус", "уведомление", "приложение", "закрыть"},
}


@dataclass(frozen=True)
class RetrievedContext:
    chunk_id: str
    text: str
    metadata: dict[str, str]
    score: float
    retrieval_mode: str

    @property
    def citation(self) -> str:
        return f"{self.metadata.get('source', self.metadata.get('doc_id', self.chunk_id))} / {self.metadata.get('section', '')}"


class BM25Index:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = list(chunks)
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freq[token] += 1
        self.idf = {
            term: math.log(1 + (len(self.documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        query_terms = tokenize(query)
        results: list[SearchResult] = []
        for idx, chunk in enumerate(self.documents):
            if filters and any(chunk.metadata.get(key) != value for key, value in filters.items()):
                continue
            score = 0.0
            length = self.doc_lengths[idx] or 1
            freqs = self.term_freqs[idx]
            for term in query_terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / max(self.avgdl, 1))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denominator
            if score > 0:
                results.append(
                    SearchResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=score,
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[:k]


class BankRetriever:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_model: EmbeddingModel,
        chunks: list[Chunk],
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)

    def infer_filters(self, query: str) -> dict[str, str]:
        tokens = set(tokenize(query))
        scores = {
            product_type: len(tokens & keywords)
            for product_type, keywords in PRODUCT_KEYWORDS.items()
        }
        product_type, score = max(scores.items(), key=lambda item: item[1])
        if score > 0:
            return {"product_type": product_type}
        return {}

    def expand_queries(self, query: str) -> list[str]:
        expansions = [query]
        tokens = set(tokenize(query))
        if tokens & {"ставка", "процент", "проценты"}:
            expansions.append(f"{query} процентная ставка тариф доходность")
        if tokens & {"документы", "паспорт", "снилс"}:
            expansions.append(f"{query} документы паспорт СНИЛС подтверждение дохода")
        if tokens & {"досрочно", "закрыть", "погасить"}:
            expansions.append(f"{query} досрочное погашение закрытие штраф комиссия")
        if tokens & {"возраст", "требования", "стаж"}:
            expansions.append(f"{query} требования заемщик возраст стаж доход")
        return expansions

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = "hybrid",
        filters: dict[str, str] | None = None,
        use_self_query: bool = True,
        use_multi_query: bool = True,
        compress: bool = True,
    ) -> list[RetrievedContext]:
        effective_filters = filters or (self.infer_filters(query) if use_self_query else {})
        queries = self.expand_queries(query) if use_multi_query else [query]
        merged: dict[str, RetrievedContext] = {}
        for expanded_query in queries:
            for result in self._retrieve_one(expanded_query, k=k, mode=mode, filters=effective_filters):
                previous = merged.get(result.chunk_id)
                if previous is None or result.score > previous.score:
                    text = _compress_text(query, result.text) if compress else result.text
                    merged[result.chunk_id] = RetrievedContext(
                        chunk_id=result.chunk_id,
                        text=text,
                        metadata=result.metadata,
                        score=result.score,
                        retrieval_mode=mode,
                    )
        reranked = sorted(
            merged.values(),
            key=lambda item: (item.score + _lexical_overlap(query, item.text)),
            reverse=True,
        )
        return reranked[:k]

    def _retrieve_one(
        self,
        query: str,
        k: int,
        mode: SearchMode,
        filters: dict[str, str] | None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_model.embed_query(query)
        if mode == "similarity":
            return self.vector_store.similarity_search(query_embedding, k=k, filters=filters)
        if mode == "mmr":
            return self.vector_store.mmr_search(query_embedding, k=k, filters=filters)
        if mode == "hybrid":
            vector_results = self.vector_store.similarity_search(query_embedding, k=k * 2, filters=filters)
            bm25_results = self.bm25.search(query, k=k * 2, filters=filters)
            return _merge_hybrid(vector_results, bm25_results, k=k)
        raise ValueError(f"Unknown retrieval mode: {mode}")


def _merge_hybrid(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int,
    vector_weight: float = 0.55,
) -> list[SearchResult]:
    max_vector = max((item.score for item in vector_results), default=1.0) or 1.0
    max_bm25 = max((item.score for item in bm25_results), default=1.0) or 1.0
    merged: dict[str, SearchResult] = {}
    scores: dict[str, float] = defaultdict(float)

    for item in vector_results:
        merged[item.chunk_id] = item
        scores[item.chunk_id] += vector_weight * (item.score / max_vector)
    for item in bm25_results:
        merged[item.chunk_id] = item
        scores[item.chunk_id] += (1 - vector_weight) * (item.score / max_bm25)

    ranked = sorted(merged.values(), key=lambda item: scores[item.chunk_id], reverse=True)
    return [
        SearchResult(
            chunk_id=item.chunk_id,
            text=item.text,
            metadata=item.metadata,
            score=scores[item.chunk_id],
        )
        for item in ranked[:k]
    ]


def _lexical_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def _compress_text(query: str, text: str, max_chars: int = 520) -> str:
    if len(text) <= max_chars:
        return text
    query_tokens = set(tokenize(query))
    sentences = [part.strip() for part in text.replace("\n", " ").split(". ") if part.strip()]
    scored = sorted(
        sentences,
        key=lambda sentence: len(query_tokens & set(tokenize(sentence))),
        reverse=True,
    )
    selected: list[str] = []
    total = 0
    for sentence in scored:
        candidate = sentence if sentence.endswith(".") else f"{sentence}."
        if total + len(candidate) > max_chars and selected:
            continue
        selected.append(candidate)
        total += len(candidate)
        if total >= max_chars:
            break
    return " ".join(selected)[:max_chars].strip()
