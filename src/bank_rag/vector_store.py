from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from bank_rag.chunking import Chunk
from bank_rag.embeddings import cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, str]
    score: float


@dataclass(frozen=True)
class VectorRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float]


def _matches_filter(metadata: dict[str, str], filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


class LocalVectorStore:
    """Persistent vector store used for offline demos and deterministic tests."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.records: list[VectorRecord] = []

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
        path: Path | str,
    ) -> "LocalVectorStore":
        store = cls(path)
        store.records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        store.persist()
        return store

    @classmethod
    def load(cls, path: Path | str) -> "LocalVectorStore":
        store = cls(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store.records = [VectorRecord(**record) for record in data["records"]]
        return store

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(record) for record in self.records]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        scored = [
            SearchResult(
                chunk_id=record.chunk_id,
                text=record.text,
                metadata=record.metadata,
                score=cosine_similarity(query_embedding, record.embedding),
            )
            for record in self.records
            if _matches_filter(record.metadata, filters)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def mmr_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        fetch_k: int = 12,
        lambda_mult: float = 0.65,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        candidates = sorted(
            [
                record
                for record in self.records
                if _matches_filter(record.metadata, filters)
            ],
            key=lambda record: cosine_similarity(query_embedding, record.embedding),
            reverse=True,
        )[:fetch_k]
        selected: list[VectorRecord] = []
        selected_results: list[SearchResult] = []

        while candidates and len(selected) < k:
            scored_candidates: list[tuple[float, VectorRecord]] = []
            for record in candidates:
                query_score = cosine_similarity(query_embedding, record.embedding)
                diversity_penalty = max(
                    (cosine_similarity(record.embedding, chosen.embedding) for chosen in selected),
                    default=0.0,
                )
                mmr_score = lambda_mult * query_score - (1 - lambda_mult) * diversity_penalty
                scored_candidates.append((mmr_score, record))

            _, best = max(scored_candidates, key=lambda item: item[0])
            candidates.remove(best)
            selected.append(best)
            selected_results.append(
                SearchResult(
                    chunk_id=best.chunk_id,
                    text=best.text,
                    metadata=best.metadata,
                    score=cosine_similarity(query_embedding, best.embedding),
                )
            )

        return selected_results


class ChromaVectorStore:
    """ChromaDB-backed vector store for the production-style project path."""

    def __init__(self, persist_directory: Path | str, collection_name: str = "bank_rag"):
        import chromadb

        self.persist_directory = Path(persist_directory)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(collection_name)

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
        persist_directory: Path | str,
        collection_name: str = "bank_rag",
    ) -> "ChromaVectorStore":
        store = cls(persist_directory, collection_name=collection_name)
        chunk_list = list(chunks)
        ids = [chunk.chunk_id for chunk in chunk_list]
        if ids:
            store.collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunk_list],
                metadatas=[chunk.metadata for chunk in chunk_list],
                embeddings=embeddings,
            )
        return store

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        where = filters or None
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        return [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata={str(key): str(value) for key, value in metadata.items()},
                score=1.0 / (1.0 + distance),
            )
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]
