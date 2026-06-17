from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bank_rag.chunking import Chunk, ChunkStrategy, chunk_documents
from bank_rag.documents import PROJECT_ROOT, load_documents
from bank_rag.embeddings import EmbeddingModel, create_embedding_model
from bank_rag.retrieval import BankRetriever
from bank_rag.vector_store import ChromaVectorStore, LocalVectorStore

StoreBackend = Literal["json", "chroma"]


DEFAULT_INDEX_PATH = PROJECT_ROOT / "artifacts" / "bank_rag_vectors.json"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "artifacts" / "chroma"


@dataclass
class BankRAGIndex:
    chunks: list[Chunk]
    embedding_model: EmbeddingModel
    vector_store: LocalVectorStore
    retriever: BankRetriever
    storage_path: Path


def build_local_index(
    strategy: ChunkStrategy = "recursive",
    embedding_backend: str = "hash",
    storage_path: Path | str | None = None,
    chunk_size: int = 420,
    chunk_overlap: int = 60,
) -> BankRAGIndex:
    documents = load_documents()
    chunks = chunk_documents(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding_model = create_embedding_model(backend=embedding_backend)
    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])
    index_path = Path(storage_path) if storage_path else DEFAULT_INDEX_PATH
    vector_store = LocalVectorStore.from_chunks(chunks, embeddings, path=index_path)
    retriever = BankRetriever(vector_store=vector_store, embedding_model=embedding_model, chunks=chunks)
    return BankRAGIndex(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        storage_path=index_path,
    )


def build_chroma_index(
    strategy: ChunkStrategy = "recursive",
    embedding_backend: str = "sentence-transformers",
    persist_directory: Path | str = DEFAULT_CHROMA_PATH,
    chunk_size: int = 420,
    chunk_overlap: int = 60,
) -> ChromaVectorStore:
    documents = load_documents()
    chunks = chunk_documents(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding_model = create_embedding_model(backend=embedding_backend)
    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])
    return ChromaVectorStore.from_chunks(chunks, embeddings, persist_directory=persist_directory)
