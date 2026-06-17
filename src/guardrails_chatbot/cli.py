"""Command-line entry point for the guardrails chatbot."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from guardrails_chatbot.fact_validator import InMemoryVectorFactValidator
from guardrails_chatbot.guardrails import GuardrailEngine
from guardrails_chatbot.model import build_chat_model
from guardrails_chatbot.pii import PIIDetector
from guardrails_chatbot.toxicity import build_toxicity_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local chatbot with advanced guardrails.")
    parser.add_argument("prompt", help="User prompt to process.")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional local Hugging Face model id/path. Uses stub model when omitted.",
    )
    parser.add_argument(
        "--knowledge-base",
        default=None,
        help="Path to a text file used for vector-style factuality validation.",
    )
    parser.add_argument(
        "--toxicity-threshold",
        type=float,
        default=0.7,
        help="Toxicity threshold for Detoxify or heuristic fallback.",
    )
    parser.add_argument(
        "--no-detoxify",
        action="store_true",
        help="Use heuristic toxicity detector even when Detoxify is installed.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full guardrail decision as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = GuardrailEngine(
        model=build_chat_model(args.model_id),
        pii_detector=PIIDetector(),
        toxicity_detector=build_toxicity_detector(
            prefer_detoxify=not args.no_detoxify,
            threshold=args.toxicity_threshold,
        ),
        fact_validator=InMemoryVectorFactValidator.from_file(args.knowledge_base),
    )

    decision = engine.chat(args.prompt)
    if args.json:
        print(json.dumps(_to_jsonable(decision), ensure_ascii=False, indent=2))
    else:
        print(decision.response)
    return 0 if decision.allowed else 2


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
