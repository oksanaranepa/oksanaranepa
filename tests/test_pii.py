from guardrails_chatbot.pii import PIIDetector


def test_detects_and_masks_common_pii() -> None:
    detector = PIIDetector()

    text = "Почта test@example.com, паспорт 1234 567890, карта 4111 1111 1111 1111."

    matches = detector.find(text)
    kinds = {match.kind for match in matches}

    assert {"email", "russian_passport", "credit_card"} <= kinds
    assert detector.mask(text, matches) == (
        "Почта [EMAIL], паспорт [RUSSIAN_PASSPORT], карта [CREDIT_CARD]."
    )


def test_ignores_non_luhn_long_numbers_as_credit_cards() -> None:
    detector = PIIDetector()

    matches = detector.find("Номер заказа 1234 5678 9012 3456.")

    assert all(match.kind != "credit_card" for match in matches)
