# oksanaranepa

## Банковский RAG-консультант для Google Colab

В репозитории добавлен ноутбук `banking_rag_colab.ipynb` с демонстрационной RAG-системой для автоматизации ответов клиентам банка.

Что входит:

- синтетическая база из 5 банковских документов;
- чанкинг документов разными стратегиями;
- эмбеддинги Hugging Face и векторная база Chroma;
- similarity/MMR/hybrid retrieval с BM25 и Reciprocal Rank Fusion;
- RAG-цепочка LangChain + OpenAI;
- reranking через CrossEncoder;
- базовая оценка retrieval и опциональная оценка Ragas;
- простые замеры времени ответа и LRU-кеш.

### Запуск в Colab

1. Откройте `banking_rag_colab.ipynb` в Google Colab.
2. Выполните первую ячейку установки зависимостей.
3. Введите `OPENAI_API_KEY`, когда ноутбук запросит ключ. Если оставить поле пустым, retrieval-часть будет работать, а LLM/Ragas-шаги будут пропущены.
4. Запустите ячейки по порядку.

После инициализации RAG-цепочки можно задавать вопросы через:

```python
answer_question("Какой срок кредита?")
```

По умолчанию используется модель эмбеддингов `intfloat/multilingual-e5-large`. В Colab ее можно заменить до запуска ячейки с эмбеддингами:

```python
import os
os.environ["EMBEDDING_MODEL_NAME"] = "intfloat/multilingual-e5-base"
```
