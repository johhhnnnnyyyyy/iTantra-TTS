import itantra_tts


def test_package_exposes_the_normalization_contract() -> None:
    assert hasattr(itantra_tts, "NormalizationResult")


def test_english_normalization_expands_safe_spoken_forms_and_segments_sentences() -> None:
    result = itantra_tts.SpeechNormalizer().normalize(
        "Meet me at 5:30. Bring ₹200 and 10%!", language="en"
    )

    assert result.original_text == "Meet me at 5:30. Bring ₹200 and 10%!"
    assert result.normalized_text == (
        "Meet me at five thirty. Bring two hundred rupees and ten percent!"
    )
    assert result.sentences == (
        "Meet me at five thirty.",
        "Bring two hundred rupees and ten percent!",
    )
    assert result.warnings == ()


def test_hindi_normalization_expands_safe_spoken_forms_and_splits_at_danda() -> None:
    result = itantra_tts.SpeechNormalizer().normalize(
        "बैठक 5:30 है। ₹200 और 10% लाना।", language="hi"
    )

    assert result.normalized_text == "बैठक पांच बजकर तीस मिनट है। दो सौ रुपये और दस प्रतिशत लाना।"
    assert result.sentences == (
        "बैठक पांच बजकर तीस मिनट है।",
        "दो सौ रुपये और दस प्रतिशत लाना।",
    )
    assert result.warnings == ()


def test_unsafe_decimal_currency_and_ambiguous_date_are_preserved_with_warnings() -> None:
    result = itantra_tts.SpeechNormalizer().normalize("Price ₹200.50 on 05/06/2026.", language="en")

    assert result.normalized_text == "Price ₹200.50 on 05/06/2026."
    assert {warning.code for warning in result.warnings} == {
        "AMBIGUOUS_DATE_LEFT_UNCHANGED",
        "UNSAFE_DECIMAL_CURRENCY_LEFT_UNCHANGED",
    }
