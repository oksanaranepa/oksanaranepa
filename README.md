# Банковский консультант: RAG-система для ответов клиентам

MVP Retrieval-Augmented Generation системы для автоматизации ответов клиентам банка
по кредитам, ипотеке, депозитам, требованиям к заемщикам и FAQ. Проект построен так,
чтобы запускаться офлайн в учебной среде, но поддерживает production-бэкенды:
`intfloat/multilingual-e5-large` для эмбеддингов, ChromaDB для векторной БД и
OpenAI/Ollama для LLM.

## Что реализовано

- 5 синтетических документов в едином JSON-формате: условия кредитования, ипотека,
  тарифы депозитов, требования к заемщикам, FAQ.
- Очистка текста от лишних пробелов и управляющих символов.
- 3 стратегии чанкинга на LangChain text splitters:
  - `size` — по фиксированному размеру;
  - `sentence` — sentence-aware разбиение;
  - `recursive` — рекурсивное разбиение по абзацам, строкам, предложениям и словам.
- Эмбеддинги:
  - production: `intfloat/multilingual-e5-large`;
  - offline/CI: deterministic hashing embeddings без загрузки модели.
- Хранилища:
  - локальное JSON-хранилище для воспроизводимых тестов;
  - ChromaDB для полноценной векторной базы.
- Ретрив:
  - similarity search;
  - MMR;
  - hybrid search = vector + BM25;
  - фильтрация по `product_type`;
  - self-query фильтрация;
  - multi-query расширение;
  - контекстное сжатие;
  - простое reranking-ранжирование по лексическому overlap.
- 20 вопросов для оценки ретрива с эталонными документами и разделами.
- Метрики `Hit Rate@k` и `MRR`.
- RAG-цепочка с историей диалога, обработкой отсутствия контекста, цитированием
  источников и проверкой цитат.
- RAGAS-интеграция при наличии зависимостей и LLM-ключей; офлайн fallback считает
  proxy-метрики `faithfulness`, `answer_relevancy`, `context_relevancy`.
- Измерение производительности и оптимизация через кеш ответов и повторное
  использование in-memory индекса.

## Установка

Минимальный офлайн-запуск тестов:

```bash
PYTHONPATH=src python -m pytest
```

Полная установка CLI:

```bash
python -m pip install -e ".[dev,eval]"
```

> Для production-эмбеддингов потребуется загрузка модели SentenceTransformers.
> Для OpenAI backend задайте `OPENAI_API_KEY`.

## Быстрый старт

Подготовить очищенные документы:

```bash
bank-rag prepare
```

Собрать локальный индекс:

```bash
bank-rag build-index --strategy recursive --store json --embedding-backend hash
```

Собрать ChromaDB-индекс с multilingual E5:

```bash
bank-rag build-index --strategy recursive --store chroma --embedding-backend sentence-transformers
```

Задать вопрос:

```bash
bank-rag ask "Какая ставка начинается по ипотеке Семейная?" --product-type mortgage
```

Сравнить стратегии чанкинга:

```bash
bank-rag compare-chunking
```

Оценить ретрив:

```bash
bank-rag evaluate-retrieval --strategy recursive --k 5
```

Оценить RAG-качество:

```bash
bank-rag evaluate-rag
```

Измерить производительность:

```bash
bank-rag performance
```

## Обоснование модели эмбеддингов

Для production-сценария выбрана `intfloat/multilingual-e5-large`, потому что модель:

- обучена для retrieval-задач;
- поддерживает русский и английский языки;
- использует формат `query:` / `passage:`, что хорошо подходит RAG;
- устойчива к терминологии банковских продуктов.

В тестах по умолчанию используется hashing backend, чтобы проект работал без
интернета, GPU и скачивания весов.

## Структура

```text
data/
  bank_documents.json   # база знаний MVP
  eval_questions.json   # 20 вопросов для оценки ретрива
src/bank_rag/
  chunking.py           # стратегии чанкинга LangChain
  documents.py          # загрузка и очистка данных
  embeddings.py         # E5 и offline embeddings
  vector_store.py       # JSON и ChromaDB хранилища
  retrieval.py          # similarity, MMR, hybrid, BM25, compression
  llm.py                # prompt, OpenAI/Ollama/offline LLM
  pipeline.py           # RAG-цепочка
  evaluation.py         # Hit Rate, MRR, RAGAS/proxy, performance
  cli.py                # CLI
tests/
  test_bank_rag.py
```

## Анализ и возможные улучшения

Текущий MVP рассчитан на синтетическую базу знаний. Для промышленного внедрения
следующие улучшения наиболее важны:

1. Автоматическая загрузка актуальных банковских регламентов из CMS/Confluence.
2. Версионирование документов и индекса, чтобы ответы ссылались на конкретную
   редакцию продукта.
3. Cross-encoder reranker для точного ранжирования top-k результатов.
4. Guardrails для запрета финансовых обещаний вне источников.
5. Online evaluation на реальных обращениях с разметкой операторов.
6. Кеширование эмбеддингов и batch-indexing при обновлении базы знаний.
