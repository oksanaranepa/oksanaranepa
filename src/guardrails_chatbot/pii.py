"""PII detection and masking helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PIIMatch:
    """A detected personally identifiable information span."""

    kind: str
    value: str
    start: int
    end: int


class PIIDetector:
    """Regex-based PII detector tuned for common Russian and global identifiers."""

    _PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "email",
            re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        ),
        (
            "phone",
            re.compile(
                r"(?<!\d)(?:\+7|8)?[\s\-.(]*\d{3}[\s\-.)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)"
            ),
        ),
        (
            "russian_passport",
            re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)"),
        ),
        (
            "snils",
            re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}\s?\d{2}(?!\d)"),
        ),
        (
            "credit_card",
            re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
        ),
    )

    def find(self, text: str) -> list[PIIMatch]:
        """Return all PII spans detected in text."""

        matches: list[PIIMatch] = []
        for kind, pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                if kind == "credit_card" and not self._looks_like_card(value):
                    continue
                matches.append(
                    PIIMatch(
                        kind=kind,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        return sorted(matches, key=lambda item: (item.start, item.end))

    def has_pii(self, text: str) -> bool:
        """Return true when text contains any PII."""

        return bool(self.find(text))

    def mask(self, text: str, matches: Iterable[PIIMatch] | None = None) -> str:
        """Replace PII spans with stable placeholders."""

        spans = list(matches if matches is not None else self.find(text))
        if not spans:
            return text

        masked: list[str] = []
        cursor = 0
        for match in spans:
            if match.start < cursor:
                continue
            masked.append(text[cursor : match.start])
            masked.append(f"[{match.kind.upper()}]")
            cursor = match.end
        masked.append(text[cursor:])
        return "".join(masked)

    @staticmethod
    def _looks_like_card(value: str) -> bool:
        digits = [int(char) for char in value if char.isdigit()]
        if not 13 <= len(digits) <= 19:
            return False

        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0
