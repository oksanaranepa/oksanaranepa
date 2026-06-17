from __future__ import annotations

import argparse
import json
from pathlib import Path

from bank_rag.chunking import ChunkStrategy
from bank_rag.documents import write_clean_documents
from bank_rag.embeddings import EMBEDDING_RATIONALE
from bank_rag.evaluation import (
    compare_chunking_strategies,
    evaluate_rag_quality,
    evaluate_retrieval,
    measure_performance,
)
from bank_rag.indexing import DEFAULT_INDEX_PATH, build_chroma_index, build_local_index
from bank_rag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking RAG consultant MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Clean and write normalized documents")
    prepare.add_argument("--output", default="artifacts/clean_bank_documents.json")

    build = subparsers.add_parser("build-index", help="Build embeddings and persist a vector index")
    build.add_argument("--strategy", choices=["size", "sentence", "recursive"], default="recursive")
    build.add_argument("--store", choices=["json", "chroma"], default="json")
    build.add_argument("--embedding-backend", choices=["hash", "sentence-transformers"], default="hash")

    ask = subparsers.add_parser("ask", help="Ask the RAG banking consultant")
    ask.add_argument("question")
    ask.add_argument("--product-type", choices=["loan", "mortgage", "deposit", "borrower", "faq"])
    ask.add_argument("--mode", choices=["similarity", "mmr", "hybrid"], default="hybrid")

    evaluate = subparsers.add_parser("evaluate-retrieval", help="Calculate Hit Rate@k and MRR")
    evaluate.add_argument("--strategy", choices=["size", "sentence", "recursive"], default="recursive")
    evaluate.add_argument("--k", type=int, default=5)

    subparsers.add_parser("compare-chunking", help="Compare size, sentence, and recursive chunking")
    subparsers.add_parser("evaluate-rag", help="Calculate RAGAS or offline proxy metrics")
    subparsers.add_parser("performance", help="Measure baseline and cached response latency")

    args = parser.parse_args()

    if args.command == "prepare":
        output = write_clean_documents(Path(args.output))
        print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "build-index":
        if args.store == "chroma":
            build_chroma_index(strategy=args.strategy, embedding_backend=args.embedding_backend)
            payload = {
                "store": "chroma",
                "embedding_rationale": EMBEDDING_RATIONALE,
            }
        else:
            index = build_local_index(strategy=args.strategy, embedding_backend=args.embedding_backend)
            payload = {
                "store": "json",
                "path": str(index.storage_path),
                "chunks": len(index.chunks),
                "embedding_rationale": EMBEDDING_RATIONALE,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        pipeline = RAGPipeline.from_defaults()
        response = pipeline.answer(
            args.question,
            product_type=args.product_type,
            mode=args.mode,
        )
        print(response.answer)
        print(json.dumps({"latency_ms": response.latency_ms, "metadata": response.metadata}, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-retrieval":
        result = evaluate_retrieval(strategy=args.strategy, k=args.k, storage_path=DEFAULT_INDEX_PATH)
        metrics = result["metrics"]
        print(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "hit_rate_at_k": metrics.hit_rate_at_k,
                    "mrr": metrics.mrr,
                    "questions": metrics.questions,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "compare-chunking":
        print(json.dumps(compare_chunking_strategies(), ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-rag":
        print(json.dumps(evaluate_rag_quality(), ensure_ascii=False, indent=2))
        return

    if args.command == "performance":
        print(json.dumps(measure_performance(), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
