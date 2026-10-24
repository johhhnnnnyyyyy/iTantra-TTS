"""Conservative, unreviewed cardinal expansion for Indic research."""
import re

_COMPLEX = re.compile(
    r"\d[.:,/+\-]\d|\d\s+\d|[₹$€%]|[+\-]\s*\d|\b0\d+\b|\d{3,}|[^\W\d_]\d|\d[^\W\d_]"
)
_NUMBER = re.compile(r"\b\d{1,2}\b")
_AM_PM = re.compile(r"\b\d{1,2}\s*[ap]\.?m\.?(?!\w)", re.IGNORECASE)
# Experimental vocabulary informed by the MIT-licensed Bodhan AI indic_normalizer.
_VOCAB = {
    "bn": ("দশমিক", "টাকা"), "gu": ("દશાંશ", "રૂપિયા"),
    "kn": ("ದಶಮಾಂಶ", "ರೂಪಾಯಿ"), "ml": ("ദശാംശം", "രൂപ"),
    "mr": ("दशांश", "रुपये"), "or": ("ଦଶମିକ", "ଟଙ୍କା"),
    "ta": ("புள்ளி", "ரூபாய்"), "te": ("దశాంశ", "రూపాయలు"),
}
_DECIMAL = re.compile(r"(?<![\w./+₹\-])(0|[1-9][0-9]?)\.([0-9]{1,3})(?![\w°%\-]|\.\d)")
_RUPEES = re.compile(r"(?<![\w+\-])₹\s*([1-9][0-9]{0,3})(?![\w,:/%\-]|\.\d)")


def expand_cardinals(text: str, language: str) -> tuple[str, str]:
    """Expand plain 0–99 numbers only; leave ambiguous numeric messages intact."""
    notice = "Experimental cardinal rules; native-speaker review pending."
    if not any(c.isdecimal() for c in text):
        return text, notice
    try:
        from indic_numtowords import num2words
    except ImportError:
        return text, notice + " Warning: UNNORMALIZED_NUMERIC_FORM; number library unavailable."
    try:
        protected: dict[str, str] = {}

        def protect(value: str) -> str:
            index = 0xE000 + len(protected)
            while chr(index) in text or chr(index) in protected:
                index += 1
            token = chr(index)
            protected[token] = value
            return token

        def expand_rupees(match: re.Match[str]) -> str:
            amount = int(match.group(1))
            if amount < 2:
                return match.group(0)
            return protect(f"{num2words(amount, lang=language)} {_VOCAB[language][1]}")

        def expand_decimal(match: re.Match[str]) -> str:
            whole, fraction = match.groups()
            digits = " ".join(num2words(int(digit), lang=language) for digit in fraction)
            return protect(f"{num2words(int(whole), lang=language)} {_VOCAB[language][0]} {digits}")

        masked = text
        if language in _VOCAB:
            masked = _RUPEES.sub(expand_rupees, masked)
            masked = _DECIMAL.sub(expand_decimal, masked)
        if _COMPLEX.search(masked) or _AM_PM.search(text):
            return text, notice + (
                " Warning: UNNORMALIZED_NUMERIC_FORM; spoken output may be incorrect."
            )
        normalized = _NUMBER.sub(lambda match: num2words(int(match.group()), lang=language), masked)
        for token, value in protected.items():
            normalized = normalized.replace(token, value)
        return normalized, notice
    except (KeyError, ValueError, TypeError, OverflowError):
        return text, notice + " Warning: UNNORMALIZED_NUMERIC_FORM; conversion failed."
