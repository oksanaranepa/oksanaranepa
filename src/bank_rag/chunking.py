from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from bank_rag.documents import BankDocument, clean_text

ChunkStrategy = Literal["size", "sentence", "recursive"]


try:  # LangChain is the primary implementation required by the assignment.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - exercised only in minimal offline environments.
    RecursiveCharacterTextSplitter = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def _fallback_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _split_text(
    text: str,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == "size":
        if RecursiveCharacterTextSplitter is None:
            return _fallback_split(text, chunk_size, chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            separators=[" ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    if strategy == "sentence":
        if RecursiveCharacterTextSplitter is None:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            return _fallback_split(" ".join(sentences), chunk_size, chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            separators=[". ", "? ", "! ", "\n", "; ", ", ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    if strategy == "recursive":
        if RecursiveCharacterTextSplitter is None:
            return _fallback_split(text, chunk_size, chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_documents(
    documents: Iterable[BankDocument],
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 420,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for section in doc.sections:
            section_text = clean_text(f"{doc.title}\n{section.heading}\n{section.text}")
            split_texts = _split_text(section_text, strategy, chunk_size, chunk_overlap)
            for idx, text in enumerate(split_texts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{section.heading}:{strategy}:{idx}",
                        text=clean_text(text),
                        metadata={
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "product_type": doc.product_type,
                            "updated_at": doc.updated_at,
                            "source": doc.source,
                            "section": section.heading,
                            "chunk_strategy": strategy,
                            "chunk_index": str(idx),
                        },
                    )
                )
    return chunks
