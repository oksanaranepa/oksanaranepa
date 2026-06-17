from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bank_rag.chunking import ChunkStrategy
from bank_rag.documents import load_eval_questions
from bank_rag.embeddings import tokenize
from bank_rag.indexing import build_local_index
from bank_rag.pipeline import RAGPipeline


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_rate_at_k: float
    mrr: float
    questions: int


def evaluate_retrieval(
    strategy: ChunkStrategy = "recursive",
    k: int = 5,
    storage_path: Path | str | None = None,
) -> dict[str, Any]:
    index = build_local_index(strategy=strategy, storage_path=storage_path)
    questions = load_eval_questions()
    hits = 0
    reciprocal_ranks: list[float] = []
    per_question = []

    for item in questions:
        results = index.retriever.retrieve(
            item["question"],
            k=k,
            mode="hybrid",
            filters={"product_type": item["product_type"]},
        )
        rank = 0
        for idx, result in enumerate(results, start=1):
            if (
                result.metadata.get("doc_id") == item["expected_doc_id"]
                and result.metadata.get("section") == item["expected_section"]
            ):
                rank = idx
                break
        hit = rank > 0
        hits += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        per_question.append(
            {
                "question": item["question"],
                "expected": f"{item['expected_doc_id']} / {item['expected_section']}",
                "hit": hit,
                "rank": rank,
                "top_chunks": [
                    f"{result.metadata.get('doc_id')} / {result.metadata.get('section')}"
                    for result in results
                ],
            }
        )

    metrics = RetrievalMetrics(
        hit_rate_at_k=hits / len(questions),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        questions=len(questions),
    )
    return {"metrics": metrics, "per_question": per_question}


def compare_chunking_strategies(k: int = 5) -> dict[str, Any]:
    comparison = {}
    for strategy in ("size", "sentence", "recursive"):
        result = evaluate_retrieval(strategy=strategy, k=k)
        metrics: RetrievalMetrics = result["metrics"]
        comparison[strategy] = {
            "hit_rate_at_k": round(metrics.hit_rate_at_k, 4),
            "mrr": round(metrics.mrr, 4),
            "questions": metrics.questions,
        }
    return comparison


def evaluate_rag_quality(pipeline: RAGPipeline | None = None) -> dict[str, Any]:
    """Evaluate generated answers with Ragas when available, otherwise use deterministic proxies.

    The fallback keeps CI and local demos independent of paid LLM credentials. It reports the
    same core metric names: faithfulness, answer relevancy, and context relevancy.
    """
    active_pipeline = pipeline or RAGPipeline.from_defaults()
    questions = load_eval_questions()[:8]
    responses = [active_pipeline.answer(item["question"]) for item in questions]

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": [item["question"] for item in questions],
                "answer": [response.answer for response in responses],
                "contexts": [[context.text for context in response.contexts] for response in responses],
                "ground_truth": [item["expected_section"] for item in questions],
            }
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return {"framework": "ragas", "metrics": dict(result)}
    except Exception as exc:
        proxy_rows = []
        for item, response in zip(questions, responses):
            answer_tokens = set(tokenize(response.answer))
            context_tokens = set(tokenize(" ".join(context.text for context in response.contexts)))
            question_tokens = set(tokenize(item["question"]))
            expected_tokens = set(tokenize(item["expected_section"]))
            faithfulness_score = _safe_overlap(answer_tokens, context_tokens)
            answer_relevancy_score = _safe_overlap(question_tokens, answer_tokens)
            context_relevancy_score = max(
                _safe_overlap(question_tokens | expected_tokens, set(tokenize(context.text)))
                for context in response.contexts
            )
            proxy_rows.append(
                {
                    "faithfulness": faithfulness_score,
                    "answer_relevancy": answer_relevancy_score,
                    "context_relevancy": context_relevancy_score,
                }
            )

        return {
            "framework": "offline_proxy",
            "ragas_error": str(exc),
            "metrics": {
                "faithfulness": statistics.mean(row["faithfulness"] for row in proxy_rows),
                "answer_relevancy": statistics.mean(row["answer_relevancy"] for row in proxy_rows),
                "context_relevancy": statistics.mean(row["context_relevancy"] for row in proxy_rows),
            },
        }


def measure_performance(question: str = "Какая ставка по семейной ипотеке?") -> dict[str, Any]:
    pipeline = RAGPipeline.from_defaults(cache_enabled=False)
    cached_pipeline = RAGPipeline.from_defaults(cache_enabled=True)

    cold_times = [_time_answer(pipeline, question) for _ in range(3)]
    cached_times = [_time_answer(cached_pipeline, question) for _ in range(3)]
    return {
        "question": question,
        "baseline_avg_ms": round(statistics.mean(cold_times) * 1000, 2),
        "cached_avg_ms": round(statistics.mean(cached_times) * 1000, 2),
        "improvement": "answer cache and reused in-memory index",
    }


def _time_answer(pipeline: RAGPipeline, question: str) -> float:
    started = time.perf_counter()
    pipeline.answer(question)
    return time.perf_counter() - started


def _safe_overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)
