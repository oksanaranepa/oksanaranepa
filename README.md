# Advanced Guardrails Chatbot

Прототип чат-бота с защитным слоем безопасности и этики вокруг локальной LLM.

Пайплайн:

1. Пользовательский промпт проверяется до генерации:
   - токсичность через Detoxify, если библиотека установлена;
   - fallback-эвристики для локальной разработки без ML-зависимостей;
   - PII через регулярные выражения: email, телефон, паспорт РФ, СНИЛС, банковские карты.
2. Безопасный промпт передается локальной модели:
   - `transformers` pipeline для Hugging Face моделей;
   - детерминированная fallback-модель для тестов и демонстрации.
3. Ответ проверяется после генерации:
   - отсутствие утечки PII;
   - токсичность;
   - фактологическое соответствие локальной базе знаний через TF-IDF similarity.

В каталоге `configs/nemo` есть пример конфигурации NeMo Guardrails, которую можно адаптировать для production-оркестрации.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
guardrails-chatbot "Можно ли отправить паспорт 1234 567890?"
```

По умолчанию используется безопасная stub-модель, чтобы пример запускался без скачивания LLM.

## Запуск с локальной Hugging Face моделью

```bash
pip install -e ".[local-ml]"
guardrails-chatbot \
  --model-id distilgpt2 \
  --knowledge-base data/knowledge_base.txt \
  "Расскажи, какие правила безопасности есть у ассистента"
```

Для более крупных моделей может потребоваться GPU и предварительно скачанные веса.

## Структура

- `src/guardrails_chatbot/pii.py` — PII-детекторы и маскирование.
- `src/guardrails_chatbot/toxicity.py` — Detoxify adapter и эвристический fallback.
- `src/guardrails_chatbot/model.py` — локальная генерация через `transformers` или stub.
- `src/guardrails_chatbot/fact_validator.py` — простая RAG-style проверка ответа по локальной базе знаний.
- `src/guardrails_chatbot/guardrails.py` — orchestration layer.
- `configs/nemo` — пример NeMo Guardrails конфигурации.

## Тесты

```bash
pip install -e ".[dev]"
pytest
```
