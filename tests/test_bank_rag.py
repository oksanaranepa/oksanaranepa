from bank_rag.chunking import chunk_documents
from bank_rag.documents import clean_text, load_documents
from bank_rag.evaluation import evaluate_retrieval
from bank_rag.indexing import build_local_index
from bank_rag.llm import create_llm
from bank_rag.pipeline import RAGPipeline


def test_clean_text_normalizes_spaces_and_control_chars():
    assert clean_text("  Кредит\t\tУдобный\x00 \r\n\n\n ставка  ") == "Кредит Удобный\n\nставка"


def test_load_documents_has_canonical_schema():
    documents = load_documents()
    assert len(documents) == 5
    assert {document.product_type for document in documents} >= {"loan", "mortgage", "deposit"}
    assert all(document.sections for document in documents)


def test_chunking_strategies_create_metadata():
    documents = load_documents()
    for strategy in ("size", "sentence", "recursive"):
        chunks = chunk_documents(documents, strategy=strategy, chunk_size=220, chunk_overlap=30)
        assert chunks
        assert all(chunk.metadata["chunk_strategy"] == strategy for chunk in chunks)
        assert all("doc_id" in chunk.metadata and "section" in chunk.metadata for chunk in chunks)


def test_hybrid_retrieval_respects_product_filter(tmp_path):
    index = build_local_index(storage_path=tmp_path / "vectors.json")
    results = index.retriever.retrieve(
        "Можно ли пополнять вклад Доходный плюс?",
        filters={"product_type": "deposit"},
        mode="hybrid",
    )
    assert results
    assert results[0].metadata["doc_id"] == "deposit_tariffs"
    assert all(result.metadata["product_type"] == "deposit" for result in results)


def test_retrieval_metrics_are_high_on_synthetic_eval_set(tmp_path):
    result = evaluate_retrieval(storage_path=tmp_path / "vectors.json")
    metrics = result["metrics"]
    assert metrics.questions == 20
    assert metrics.hit_rate_at_k >= 0.85
    assert metrics.mrr >= 0.75


def test_rag_answer_contains_valid_source_citation(tmp_path):
    index = build_local_index(storage_path=tmp_path / "vectors.json")
    pipeline = RAGPipeline(index=index, llm=create_llm())
    response = pipeline.answer("Какая ставка начинается по ипотеке Семейная?", product_type="mortgage")
    assert "Источники:" in response.answer
    assert response.contexts
    assert response.metadata["citation_validation"] is True
