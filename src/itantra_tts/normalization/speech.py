"""Deterministic, conservative English/Hindi text normalization for TTS."""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass


RULE_VERSION = "w2-v1"
_SUPPORTED_LANGUAGES = frozenset({"en", "hi"})
_WHITESPACE = re.compile(r"\s+")
_AMBIGUOUS_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_DECIMAL_CURRENCY = re.compile(r"₹\d+\.\d+")
_DECIMAL_NUMBER = re.compile(r"\b\d+\.\d+\b")
_CURRENCY = re.compile(r"₹(\d+)\b")
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PERCENTAGE = re.compile(r"\b(\d+)%")
_NUMBER = re.compile(r"\b\d+\b")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।])\s+")
_PUNCTUATION_SPACE = re.compile(r"\s+([,.;:!?।])")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")


@dataclass(frozen=True, slots=True)
class NormalizationWarning:
    """A non-destructive rule outcome safe for UI and caller handling."""

    code: str


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Original text and the safe speech-ready representation."""

    original_text: str
    normalized_text: str
    sentences: tuple[str, ...]
    rule_version: str
    warnings: tuple[NormalizationWarning, ...]


class SpeechNormalizer:
    """Normalize only transformations whose spoken form is defined by W2 rules."""

    def normalize(self, text: str, language: str) -> NormalizationResult:
        """Return segmentation and warnings without mutating or discarding the original text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        if language not in _SUPPORTED_LANGUAGES:
            raise ValueError("W2 supports only English and Hindi normalization")

        warnings: list[NormalizationWarning] = []
        normalized = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()
        normalized, protected = _protect_unsafe_forms(normalized, warnings)
        if language == "en":
            normalized = _normalize_english(normalized, warnings)
        else:
            normalized = _normalize_hindi(normalized, warnings)
        normalized = _restore_protected_forms(normalized, protected)
        normalized = _PUNCTUATION_SPACE.sub(r"\1", normalized)
        sentence_parts = _SENTENCE_BOUNDARY.split(normalized)
        sentences = tuple(sentence.strip() for sentence in sentence_parts if sentence.strip())
        return NormalizationResult(
            original_text=text,
            normalized_text=normalized,
            sentences=sentences,
            rule_version=RULE_VERSION,
            warnings=tuple(warnings),
        )


def _protect_unsafe_forms(
    text: str, warnings: list[NormalizationWarning]
) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def protect(match: re.Match[str], code: str) -> str:
        token = f"\ue000{chr(ord('A') + len(protected))}\ue001"
        protected[token] = match.group(0)
        warnings.append(NormalizationWarning(code))
        return token

    text = _AMBIGUOUS_DATE.sub(lambda match: protect(match, "AMBIGUOUS_DATE_LEFT_UNCHANGED"), text)
    text = _DECIMAL_CURRENCY.sub(
        lambda match: protect(match, "UNSAFE_DECIMAL_CURRENCY_LEFT_UNCHANGED"), text
    )
    text = _DECIMAL_NUMBER.sub(
        lambda match: protect(match, "UNSAFE_DECIMAL_NUMBER_LEFT_UNCHANGED"), text
    )
    return text, protected


def _restore_protected_forms(text: str, protected: dict[str, str]) -> str:
    for token, original in protected.items():
        text = text.replace(token, original)
    return text


def _normalize_english(text: str, warnings: list[NormalizationWarning]) -> str:
    for abbreviation, spoken in {"Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus"}.items():
        text = text.replace(abbreviation, spoken)
    text = _CURRENCY.sub(lambda match: f"{_english_number(match.group(1))} rupees", text)
    text = _TIME.sub(
        lambda match: _english_time(int(match.group(1)), int(match.group(2))),
        text,
    )
    text = _PERCENTAGE.sub(lambda match: f"{_english_number(match.group(1))} percent", text)
    text = _NUMBER.sub(lambda match: _safe_number(match, _english_number, warnings), text)
    return _ACRONYM.sub(lambda match: " ".join(match.group(0)), text)


def _normalize_hindi(text: str, warnings: list[NormalizationWarning]) -> str:
    text = _CURRENCY.sub(lambda match: f"{_hindi_number(match.group(1))} रुपये", text)
    text = _TIME.sub(
        lambda match: _hindi_time(int(match.group(1)), int(match.group(2))),
        text,
    )
    text = _PERCENTAGE.sub(lambda match: f"{_hindi_number(match.group(1))} प्रतिशत", text)
    return _NUMBER.sub(lambda match: _safe_number(match, _hindi_number, warnings), text)


def _safe_number(
    match: re.Match[str],
    number_to_words: Callable[[str], str],
    warnings: list[NormalizationWarning],
) -> str:
    value = match.group(0)
    if (len(value) > 1 and value.startswith("0")) or int(value) > 9999:
        warnings.append(NormalizationWarning("UNSAFE_NUMBER_LEFT_UNCHANGED"))
        return value
    return number_to_words(value)


def _english_time(hour: int, minute: int) -> str:
    if minute == 0:
        return f"{_english_number(str(hour))} o'clock"
    return f"{_english_number(str(hour))} {_english_number(str(minute))}"


def _hindi_time(hour: int, minute: int) -> str:
    if minute == 0:
        return f"{_hindi_number(str(hour))} बजे"
    return f"{_hindi_number(str(hour))} बजकर {_hindi_number(str(minute))} मिनट"


def _english_number(value: str) -> str:
    number = int(value)
    under_twenty = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
        "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if number < 20:
        return under_twenty[number]
    if number < 100:
        return tens[number // 10] + ("" if number % 10 == 0 else f" {under_twenty[number % 10]}")
    if number < 1000:
        return under_twenty[number // 100] + " hundred" + (
            "" if number % 100 == 0 else f" {_english_number(str(number % 100))}"
        )
    return under_twenty[number // 1000] + " thousand" + (
        "" if number % 1000 == 0 else f" {_english_number(str(number % 1000))}"
    )


_HINDI_UNDER_HUNDRED = (
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस",
    "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर",
    "अठहत्तर", "उन्नासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे",
    "अट्ठानवे", "निन्यानवे",
)


def _hindi_number(value: str) -> str:
    number = int(value)
    if number < 100:
        return _HINDI_UNDER_HUNDRED[number]
    if number < 1000:
        return _HINDI_UNDER_HUNDRED[number // 100] + " सौ" + (
            "" if number % 100 == 0 else f" {_hindi_number(str(number % 100))}"
        )
    return _HINDI_UNDER_HUNDRED[number // 1000] + " हजार" + (
        "" if number % 1000 == 0 else f" {_hindi_number(str(number % 1000))}"
    )
