from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "bank_documents.json"
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_questions.json"


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class BankSection:
    heading: str
    text: str


@dataclass(frozen=True)
class BankDocument:
    doc_id: str
    title: str
    product_type: str
    updated_at: str
    source: str
    sections: tuple[BankSection, ...]

    def as_text(self) -> str:
        parts = [f"{self.title}", f"Тип продукта: {self.product_type}", f"Источник: {self.source}"]
        for section in self.sections:
            parts.append(f"{section.heading}\n{section.text}")
        return "\n\n".join(parts)


def clean_text(value: str) -> str:
    """Normalize user-facing bank text while preserving sentence boundaries."""
    text = _CONTROL_CHARS.sub("", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _clean_record(raw: dict[str, Any]) -> BankDocument:
    sections = tuple(
        BankSection(
            heading=clean_text(section["heading"]),
            text=clean_text(section["text"]),
        )
        for section in raw["sections"]
    )
    return BankDocument(
        doc_id=clean_text(raw["doc_id"]),
        title=clean_text(raw["title"]),
        product_type=clean_text(raw["product_type"]),
        updated_at=clean_text(raw["updated_at"]),
        source=clean_text(raw["source"]),
        sections=sections,
    )


def load_documents(path: Path | str = DEFAULT_DOCUMENTS_PATH) -> list[BankDocument]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_clean_record(record) for record in records]


def load_eval_questions(path: Path | str = DEFAULT_EVAL_PATH) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{key: clean_text(str(value)) for key, value in record.items()} for record in records]


def write_clean_documents(
    output_path: Path | str,
    input_path: Path | str = DEFAULT_DOCUMENTS_PATH,
) -> Path:
    """Persist normalized documents in the same canonical schema."""
    documents = load_documents(input_path)
    output = []
    for doc in documents:
        output.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "product_type": doc.product_type,
                "updated_at": doc.updated_at,
                "source": doc.source,
                "sections": [
                    {"heading": section.heading, "text": section.text}
                    for section in doc.sections
                ],
            }
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
